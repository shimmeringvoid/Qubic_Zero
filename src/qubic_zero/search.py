"""Tree search guided by a network, as in AlphaZero.

The search grows a tree of positions below the one to be decided.  Each
simulation walks down the tree, at every position taking the move with the
best sum of two terms: how well the move has done so far (its mean value
``Q``), and a bonus that is large for moves the network's policy likes and
that have been tried little::

    Q(s, a) + c_puct * P(s, a) * sqrt(1 + visits of s) / (1 + N(s, a))

At the first position that is not in the tree yet, the network is asked once
for a policy and a value; the position is added and the value is passed back
up, changing sign at every step because the players alternate.  A finished
game needs no network: its value is exact.  The move finally played is chosen
from the visit counts at the root.

With a ``discount`` below 1 a value loses a little for every move it is away,
so that a quick win is worth more than a slow one and a late loss is better
than an early one.

Two implementations are kept on purpose:

* :func:`reference_search` handles one position with ordinary Python objects.
  It is slow and easy to check by reading.
* :class:`BatchedSearch` runs the same search for thousands of positions at
  once, with the trees stored as arrays, so that one call to the network
  serves every game.  This is what self-play uses, on a GPU or a CPU.

Given the same evaluations they visit exactly the same moves the same number
of times, which ``tests/test_search.py`` checks.
"""

from __future__ import annotations

import math

import numpy as np

from .games import LineGame

try:
    import torch
except ImportError:  # the reference search does not need PyTorch
    torch = None

__all__ = ["reference_search", "BatchedSearch", "masked_policy"]


def masked_policy(scores: np.ndarray, board: np.ndarray) -> np.ndarray:
    """Probabilities over the empty cells from a network's policy scores (a softmax that skips taken cells)."""
    scores = np.where(board == 0, scores, -np.inf).astype(np.float64)
    scores = np.exp(scores - scores.max())
    return scores / scores.sum()


class _Node:
    __slots__ = ("board", "mover", "prior", "visits", "total", "children", "over", "value")

    def __init__(self, board, mover, n_cells):
        self.board, self.mover = board, mover
        self.prior = np.zeros(n_cells)
        self.visits = np.zeros(n_cells)
        self.total = np.zeros(n_cells)
        self.children: dict[int, _Node] = {}
        self.over, self.value = False, 0.0


def reference_search(
    game: LineGame,
    board: np.ndarray,
    evaluate,
    n_sims: int,
    c_puct: float = 1.5,
    discount: float = 1.0,
    root_noise: np.ndarray | None = None,
    noise_weight: float = 0.25,
):
    """Search one position.  Returns ``(visits, mean_values)`` for the moves at the root.

    ``evaluate(position) -> (probabilities over the cells, value)`` stands for the
    network: it is shown a position from the mover's point of view.
    ``root_noise`` (probabilities over the cells) is mixed into the root's
    policy with weight ``noise_weight``; self-play uses this to keep trying moves
    that the network has written off.
    """
    board = np.asarray(board, dtype=np.int8)
    root = _Node(board, int(game.to_move(board)), game.n_cells)
    root.prior, _ = evaluate(board * root.mover)
    root.prior = np.asarray(root.prior, dtype=np.float64)
    if root_noise is not None:
        root.prior = (1.0 - noise_weight) * root.prior + noise_weight * np.asarray(root_noise, dtype=np.float64)

    for _ in range(n_sims):
        node, path = root, []
        while True:
            if node.over:
                value = node.value
                break
            mean = np.where(node.visits > 0, node.total / np.maximum(node.visits, 1.0), 0.0)
            bonus = c_puct * node.prior * math.sqrt(1.0 + node.visits.sum()) / (1.0 + node.visits)
            score = np.where(node.board == 0, mean + bonus, -np.inf)
            move = int(score.argmax())
            path.append((node, move))
            child = node.children.get(move)
            if child is None:
                after = node.board.copy()
                after[move] = node.mover
                child = node.children[move] = _Node(after, -node.mover, game.n_cells)
                if game.winner(after) != 0:
                    child.over, child.value = True, -1.0  # the player to move there has lost
                elif (after != 0).all():
                    child.over, child.value = True, 0.0
                else:
                    prior, value = evaluate(after * child.mover)
                    child.prior, child.value = np.asarray(prior, dtype=np.float64), float(value)
                value = child.value
                break
            node = child
        for node, move in reversed(path):
            value = -discount * value
            node.visits[move] += 1.0
            node.total[move] += value

    mean = np.where(root.visits > 0, root.total / np.maximum(root.visits, 1.0), 0.0)
    return root.visits, mean


class BatchedSearch:
    """The same search for many positions at once (see the module's description).

    Parameters
    ----------
    game : the game.
    evaluate : ``evaluate(positions) -> (policy scores, values)`` for a batch of
        positions seen from the mover's side, as tensors; usually a network.
    n_sims : simulations per position.
    c_puct, discount : as in :func:`reference_search`.
    device, dtype : where and in which precision the trees are kept.

    ``search.run(boards)`` returns ``(visits, mean_values, root_value)``: for
    every position the visit count and mean value of each move, and the value
    of the position as the search sees it (the visit-weighted mean).
    """

    def __init__(
        self, game: LineGame, evaluate, n_sims: int, c_puct: float = 1.5, discount: float = 1.0, device="cpu", dtype=None
    ) -> None:
        if torch is None:
            raise ImportError('BatchedSearch needs PyTorch: pip install -e ".[train]"')
        self.game, self.evaluate = game, evaluate
        self.n_sims, self.c_puct, self.discount = n_sims, c_puct, discount
        self.device = torch.device(device)
        self.dtype = dtype or torch.float32
        self.incidence = torch.from_numpy(game.incidence).to(self.device, self.dtype)

    def _priors(self, scores, boards):
        scores = scores.to(self.dtype).masked_fill(boards != 0, -torch.inf)
        return torch.softmax(scores, dim=1)

    @torch.no_grad() if torch is not None else (lambda f: f)
    def run(self, boards, root_noise=None, noise_weight: float = 0.25):
        game, dev, dtype = self.game, self.device, self.dtype
        boards = torch.as_tensor(boards, device=dev).to(torch.int8)
        n_games, n_cells = boards.shape
        n_nodes = self.n_sims + 1
        rows = torch.arange(n_games, device=dev)

        node_board = torch.zeros((n_games, n_nodes, n_cells), dtype=torch.int8, device=dev)
        node_mover = torch.zeros((n_games, n_nodes), dtype=torch.int8, device=dev)
        node_over = torch.zeros((n_games, n_nodes), dtype=torch.bool, device=dev)
        node_value = torch.zeros((n_games, n_nodes), dtype=dtype, device=dev)
        prior = torch.zeros((n_games, n_nodes, n_cells), dtype=dtype, device=dev)
        visits = torch.zeros((n_games, n_nodes, n_cells), dtype=dtype, device=dev)
        total = torch.zeros((n_games, n_nodes, n_cells), dtype=dtype, device=dev)
        child = torch.full((n_games, n_nodes, n_cells), -1, dtype=torch.int32, device=dev)
        used = torch.ones(n_games, dtype=torch.long, device=dev)  # nodes in each tree so far

        mover = (1 - 2 * boards.sum(dim=1, dtype=torch.int32)).to(torch.int8)
        node_board[:, 0], node_mover[:, 0] = boards, mover
        scores, _ = self.evaluate(boards * mover[:, None])
        root_prior = self._priors(scores, boards)
        if root_noise is not None:
            noise = torch.as_tensor(root_noise, device=dev, dtype=dtype)
            root_prior = (1.0 - noise_weight) * root_prior + noise_weight * noise
        prior[:, 0] = root_prior

        depth_limit = min(self.n_sims, n_cells) + 1
        path_node = torch.zeros((n_games, depth_limit), dtype=torch.long, device=dev)
        path_move = torch.zeros((n_games, depth_limit), dtype=torch.long, device=dev)

        for _ in range(self.n_sims):
            current = torch.zeros(n_games, dtype=torch.long, device=dev)
            length = torch.zeros(n_games, dtype=torch.long, device=dev)
            walking = torch.ones(n_games, dtype=torch.bool, device=dev)
            leaf_value = torch.zeros(n_games, dtype=dtype, device=dev)
            grow = torch.zeros(n_games, dtype=torch.bool, device=dev)
            grow_from = torch.zeros(n_games, dtype=torch.long, device=dev)
            grow_move = torch.zeros(n_games, dtype=torch.long, device=dev)
            depth = 0
            # 1. walk down every tree until a move leads out of it, or the game is over
            while bool(walking.any()):
                seen = visits[rows, current]
                mean = torch.where(seen > 0, total[rows, current] / seen.clamp(min=1.0), torch.zeros_like(seen))
                bonus = self.c_puct * prior[rows, current] * torch.sqrt(1.0 + seen.sum(dim=1, keepdim=True)) / (1.0 + seen)
                score = (mean + bonus).masked_fill(node_board[rows, current] != 0, -torch.inf)
                move = score.argmax(dim=1)
                path_node[:, depth] = torch.where(walking, current, path_node[:, depth])
                path_move[:, depth] = torch.where(walking, move, path_move[:, depth])
                length += walking
                below = child[rows, current, move].long()
                new = walking & (below < 0)
                grow |= new
                grow_from = torch.where(new, current, grow_from)
                grow_move = torch.where(new, move, grow_move)
                step = walking & (below >= 0)
                current = torch.where(step, below, current)
                ended = step & node_over[rows, current]
                leaf_value = torch.where(ended, node_value[rows, current], leaf_value)
                walking = step & ~ended
                depth += 1

            # 2. add one position to each tree that the walk left, and ask the network about it
            g = grow.nonzero(as_tuple=True)[0]
            if g.numel():
                parent, move, new_node = grow_from[g], grow_move[g], used[g]
                who = node_mover[g, parent]
                after = node_board[g, parent].clone()
                after[torch.arange(g.numel(), device=dev), move] = who
                sums = after.to(dtype) @ self.incidence.T
                won = (sums * who[:, None].to(dtype)).max(dim=1).values == game.k
                full = (after != 0).all(dim=1)
                over = won | full
                value = torch.where(won, -torch.ones((), dtype=dtype, device=dev), torch.zeros((), dtype=dtype, device=dev))
                value = value.expand(g.numel()).clone()
                open_ = (~over).nonzero(as_tuple=True)[0]
                if open_.numel():
                    positions = after[open_] * (-who[open_])[:, None]
                    scores, net_value = self.evaluate(positions)
                    prior[g[open_], new_node[open_]] = self._priors(scores, after[open_])
                    value[open_] = net_value.to(dtype)
                node_board[g, new_node], node_mover[g, new_node] = after, -who
                node_over[g, new_node], node_value[g, new_node] = over, value
                child[g, parent, move] = new_node.to(torch.int32)
                used[g] += 1
                leaf_value[g] = value

            # 3. pass the values back up, changing sign (and discounting) at every step
            value = leaf_value
            for level in range(int(length.max()) - 1, -1, -1):
                on_path = length > level
                value = torch.where(on_path, -self.discount * value, value)
                b = on_path.nonzero(as_tuple=True)[0]
                nodes, moves = path_node[b, level], path_move[b, level]
                visits[b, nodes, moves] += 1.0
                total[b, nodes, moves] += value[b]

        root_visits, root_total = visits[:, 0], total[:, 0]
        mean = torch.where(root_visits > 0, root_total / root_visits.clamp(min=1.0), torch.zeros_like(root_visits))
        root_value = root_total.sum(dim=1) / root_visits.sum(dim=1).clamp(min=1.0)
        return root_visits, mean, root_value
