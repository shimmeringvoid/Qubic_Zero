"""Scoring players against exact answers and against the reference players.

* On tic-tac-toe every position has a known value, so a player can be scored
  on all of them (:func:`table_score`, :func:`value_error`).
* On the larger games exact answers exist for the tactical positions that the
  threat search settles; :class:`TacticsSuite` collects such positions and
  scores a player on them.
* :func:`match` plays a player against an opponent from both sides.

:func:`standard_evaluators` bundles these into the functions that a
:class:`~qubic_zero.train.Trainer` calls during a run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .exact import Table, _bits, _scan, solve, solve_table, threat_moves, to_bits
from .games import LineGame
from .players import play_games, random_player, rule_player

__all__ = ["table_score", "value_error", "match", "solved_sample", "TacticsSuite", "standard_evaluators"]


def table_score(table: Table, player, rng=None) -> dict:
    """How a player does on every position of a fully solved game.

    ``kept`` is the share of positions in which its move keeps the value of the
    position (a win stays a win, a draw stays a draw); ``mistakes`` counts the
    others.  ``fastest`` is the share in which the move is also among the
    quickest wins or the longest defences.
    """
    idx = np.flatnonzero(table.decision)
    moves = player(table.game, table.boards[idx], rng)
    kept = table.best[idx, moves]
    out = {"kept": float(kept.mean()), "mistakes": int((~kept).sum()), "fastest": float(table.fastest[idx, moves].mean())}
    for name, outcome in (("won", 1), ("drawn", 0), ("lost", -1)):
        where = table.outcome[idx] == outcome
        out[f"mistakes_in_{name}"] = int((~kept[where]).sum())
    return out


def value_error(table: Table, values: np.ndarray) -> dict:
    """Error of value estimates (one per position with a move to make) against the exact values of ``table``."""
    idx = np.flatnonzero(table.decision)
    exact = table.value[idx]
    smallest = np.abs(exact[exact != 0]).min() if (exact != 0).any() else 1.0
    said = np.sign(values) * (np.abs(values) > smallest / 2)  # nearer to a draw than to the mildest win or loss: a draw
    return {
        "value_error": float(np.abs(values - exact).mean()),
        "value_outcome_right": float((said == table.outcome[idx]).mean()),
    }


def match(game: LineGame, player, opponent, n_games: int, rng: np.random.Generator) -> dict:
    """``n_games`` games with ``player`` moving first and ``n_games`` with it moving second."""
    first = play_games(game, player, opponent, n_games, rng)
    second = play_games(game, opponent, player, n_games, rng)
    return {
        "first_won": first.first_wins / n_games,
        "first_drawn": first.draws / n_games,
        "first_lost": first.second_wins / n_games,
        "second_won": second.second_wins / n_games,
        "second_drawn": second.draws / n_games,
        "second_lost": second.first_wins / n_games,
    }


def _mixed_player(game, boards, rng, careful: float = 0.7):
    """Mostly the rule-based player, sometimes a random move: games that visit all sorts of positions."""
    rules = rng.random(len(boards)) < careful
    return np.where(rules, rule_player(game, boards, rng), random_player(game, boards, rng))


def solved_sample(game: LineGame, n: int = 1000, seed: int = 0, max_nodes: int = 2_000_000) -> Table:
    """A sample of positions of a game, each solved exactly, for games too large to solve in full.

    The positions come from games between the rule-based and the random player
    and are distinct up to symmetry.  Each move of each position is searched to
    the end of the game (:func:`~qubic_zero.exact.solve`), so the result can be
    used like the table of a fully solved game: ``table_score(sample, player)``.
    Depths are not worked out, so ``fastest`` is the same as ``best`` here.
    Practical on 3 x 3 x 3; out of reach on Qubic.
    """
    rng = np.random.default_rng(seed)
    chosen: dict[bytes, np.ndarray] = {}
    while len(chosen) < n:
        boards = game.empty(256)
        for _ in range(game.n_cells):
            boards = boards[~game.finished(boards)]
            if len(boards) == 0:
                break
            keep = rng.random(len(boards)) < 0.15
            for image in game.canonical(boards[keep])[0] if keep.any() else []:
                if len(chosen) < n:
                    chosen.setdefault(image.tobytes(), image)
            boards = game.play(boards, _mixed_player(game, boards, rng))
    boards = np.stack(list(chosen.values()))
    shared: dict = {}
    best = np.zeros(boards.shape, dtype=bool)
    outcome = np.zeros(len(boards), dtype=np.int8)
    for i, board in enumerate(boards):
        values = {}
        for move in np.flatnonzero(board == 0):
            after = game.play(board, move)
            values[int(move)] = 1 if game.winner(after) != 0 else -solve(game, after, max_nodes=max_nodes, table=shared)
        outcome[i] = max(values.values())
        best[i, [m for m, v in values.items() if v == outcome[i]]] = True
    none = np.full(boards.shape, -1, dtype=np.int64)
    index = {b.tobytes(): i for i, b in enumerate(boards)}
    return Table(game, 1.0, boards, outcome.astype(float), outcome, np.zeros(len(boards), dtype=bool), best, best, none, index)


@dataclass
class TacticsSuite:
    """Positions with an exact answer, found by the threat search.

    Three kinds, each an array of positions with a table of the moves that
    count as right:

    * ``win``: the player to move can complete a line.  Right: any cell that does.
    * ``block``: the opponent threatens to complete a line, and the player to move
      cannot win first.  Right: the one cell that stops it; every other move loses
      at once.
    * ``force``: no line can be completed yet, but a win can be forced with
      threats.  Right: any first move of such a win that the threat search can
      prove.  Other moves may win too, more slowly, so the share of right moves
      here is a lower bound on the share of winning moves.
    """

    game: LineGame
    boards: dict
    right: dict

    @classmethod
    def build(cls, game: LineGame, per_kind: int = 300, seed: int = 0, max_tries: int = 200) -> TacticsSuite:
        """Collect positions from games between the rule-based and the random player (so that all kinds occur)."""
        rng = np.random.default_rng(seed)
        bits = _bits(game)
        found = {kind: {} for kind in ("win", "block", "force")}

        for _ in range(max_tries):
            if all(len(v) >= per_kind for v in found.values()):
                break
            boards = game.empty(256)
            for _ply in range(game.n_cells):
                boards = boards[~game.finished(boards)]
                if len(boards) == 0:
                    break
                images, _ = game.canonical(boards[:48])
                for board in images:
                    key = board.tobytes()
                    att, dfn = to_bits(game, board)
                    win, must, _ = _scan(bits, att, dfn)
                    if win:
                        kind, moves = "win", threat_moves(game, board)
                    elif must:
                        kind, moves = "block", ([must.bit_length() - 1] if must & (must - 1) == 0 else [])
                    else:
                        kind, moves = "force", threat_moves(game, board, max_nodes=20_000)
                    if moves and len(found[kind]) < per_kind and key not in found[kind]:
                        right = np.zeros(game.n_cells, dtype=bool)
                        right[moves] = True
                        found[kind][key] = (board, right)
                boards = game.play(boards, _mixed_player(game, boards, rng))
        boards = {kind: np.stack([b for b, _ in v.values()]) for kind, v in found.items() if v}
        right = {kind: np.stack([r for _, r in v.values()]) for kind, v in found.items() if v}
        return cls(game, boards, right)

    def score(self, player, rng=None) -> dict:
        """Share of positions of each kind in which the player's move is a right one."""
        out = {}
        for kind, boards in self.boards.items():
            moves = player(self.game, boards, rng)
            out[kind] = float(self.right[kind][np.arange(len(boards)), moves].mean())
        return out

    def __repr__(self) -> str:
        return "TacticsSuite(" + ", ".join(f"{len(b)} {kind}" for kind, b in self.boards.items()) + f" on {self.game.name})"


def standard_evaluators(game: LineGame, config, n_games: int = 200, per_kind: int = 200, seed: int = 12345) -> list:
    """The evaluations written to a run's log.

    Every one is done twice: for the network alone (keys starting with ``net_``),
    and for the network with the tree search it is trained with (``search_``).

    * Tic-tac-toe: every position against the exact table.
    * 3 x 3 x 3: a sample of 1,500 exactly solved positions, the tactics suite,
      and matches against the rule-based player from both sides.
    * Qubic: the tactics suite and the matches.
    """
    from .agents import PolicyPlayer, SearchPlayer

    def both(trainer, boards):
        """Moves and values of the two players on the same positions."""
        scores, values = PolicyPlayer(trainer.net).scores_and_values(game, boards)
        searcher = SearchPlayer(trainer.net, config.n_sims, config.c_puct, config.discount)
        visits, searched = searcher.visits(game, boards)
        return {"net": (scores.argmax(axis=1), values), "search": (visits.argmax(axis=1), searched)}

    def fixed(moves):
        return lambda game, boards, rng=None: moves

    def against_exact(table, with_values):
        idx = np.flatnonzero(table.decision)

        def evaluator(trainer):
            out = {}
            for name, (moves, values) in both(trainer, table.boards[idx]).items():
                out.update({f"{name}_{k}": v for k, v in table_score(table, fixed(moves)).items()})
                if with_values:
                    out.update({f"{name}_{k}": v for k, v in value_error(table, values).items()})
            return out

        return evaluator

    if game.name == "3x3":
        return [against_exact(solve_table(game, discount=config.discount), with_values=True)]

    suite = TacticsSuite.build(game, per_kind=per_kind, seed=seed)

    def tactics(trainer):
        out = {}
        for kind, boards in suite.boards.items():
            for name, (moves, _) in both(trainer, boards).items():
                out[f"{name}_{kind}"] = float(suite.right[kind][np.arange(len(boards)), moves].mean())
        return out

    def against_rules(trainer):
        rng = np.random.default_rng(seed)
        players = {
            "net": PolicyPlayer(trainer.net),
            "search": SearchPlayer(trainer.net, config.n_sims, config.c_puct, config.discount),
        }
        out = {}
        for name, player in players.items():
            out.update({f"{name}_{k}": v for k, v in match(game, player, rule_player, n_games, rng).items()})
        return out

    evaluators = [tactics, against_rules]
    if game.name == "3x3x3":
        evaluators.insert(0, against_exact(solved_sample(game, n=1500, seed=seed), with_values=False))
    return evaluators
