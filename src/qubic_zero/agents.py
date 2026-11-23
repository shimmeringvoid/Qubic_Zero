"""Players made from a network.

Each is a player in the sense of :mod:`qubic_zero.players`: called with a
batch of positions, it answers with a move for each.  They differ in how much
stands between the network and the move, which is what the evaluations
compare:

* :class:`PolicyPlayer`: the network alone.  The move its policy scores highest.
* :class:`SearchPlayer`: the network and the tree search.  The most visited
  move, without the noise that self-play adds.
* :class:`ThreatPlayer`: any player with the exact threat search in front.  A
  forced win is taken whenever that search finds one.
"""

from __future__ import annotations

import numpy as np
import torch

from .exact import threat_search
from .games import LineGame
from .search import BatchedSearch

__all__ = ["PolicyPlayer", "SearchPlayer", "ThreatPlayer", "one_at_a_time"]


def _device_of(net) -> torch.device:
    return next(net.parameters()).device


class PolicyPlayer:
    """The network alone: the empty cell that its policy scores highest."""

    def __init__(self, net) -> None:
        self.net = net

    @torch.no_grad()
    def scores_and_values(self, game: LineGame, boards: np.ndarray):
        """Policy scores (taken cells set to -inf) and values, as arrays."""
        self.net.eval()
        boards = np.asarray(boards, dtype=np.int8).reshape(-1, game.n_cells)
        positions = torch.from_numpy(boards * game.to_move(boards)[:, None]).to(_device_of(self.net))
        scores, values = self.net(positions)
        scores = scores.masked_fill(positions != 0, -torch.inf)
        return scores.cpu().numpy(), values.cpu().numpy()

    def __call__(self, game: LineGame, boards: np.ndarray, rng=None) -> np.ndarray:
        return self.scores_and_values(game, boards)[0].argmax(axis=1)


class SearchPlayer:
    """The network with ``n_sims`` simulations of tree search: the most visited move."""

    def __init__(self, net, n_sims: int, c_puct: float = 1.5, discount: float = 1.0, batch: int = 4096) -> None:
        self.net, self.n_sims, self.c_puct, self.discount, self.batch = net, n_sims, c_puct, discount, batch

    @torch.no_grad()
    def visits(self, game: LineGame, boards: np.ndarray):
        """Visit counts and the search's value of each position."""
        self.net.eval()
        search = BatchedSearch(game, self.net, self.n_sims, self.c_puct, self.discount, device=_device_of(self.net))
        boards = np.asarray(boards, dtype=np.int8).reshape(-1, game.n_cells)
        counts, values = [], []
        for start in range(0, len(boards), self.batch):
            v, _, root = search.run(torch.from_numpy(boards[start : start + self.batch]))
            counts.append(v.cpu().numpy())
            values.append(root.cpu().numpy())
        return np.concatenate(counts), np.concatenate(values)

    def __call__(self, game: LineGame, boards: np.ndarray, rng=None) -> np.ndarray:
        return self.visits(game, boards)[0].argmax(axis=1)


class ThreatPlayer:
    """``player``, except that a win by force is taken whenever the exact threat search finds one."""

    def __init__(self, player, max_nodes: int = 100_000) -> None:
        self.player, self.max_nodes = player, max_nodes

    def __call__(self, game: LineGame, boards: np.ndarray, rng=None) -> np.ndarray:
        boards = np.asarray(boards, dtype=np.int8).reshape(-1, game.n_cells)
        moves = np.full(len(boards), -1, dtype=np.int64)
        for i, board in enumerate(boards):
            found = threat_search(game, board, self.max_nodes)
            if found.win:
                moves[i] = found.move
        rest = moves < 0
        if rest.any():
            moves[rest] = self.player(game, boards[rest], rng)
        return moves


def one_at_a_time(game: LineGame, player, rng=None):
    """``player`` as a function of a single position: ``f(board) -> move``."""
    return lambda board: int(player(game, np.asarray(board)[None], rng)[0])
