"""Reference players and a way to let any two players meet.

A player is a function ``player(game, boards, rng) -> moves`` that is handed a
whole batch of positions, all with the same side to move, and answers with one
cell for each.  Two players of this kind are provided here because every
learner needs opponents whose strength does not depend on learning:

* :func:`random_player` picks any empty cell;
* :func:`rule_player` wins if it can, blocks if it must, and otherwise picks
  any empty cell.

The learned players in :mod:`qubic_zero.agents` have the same signature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .games import LineGame

__all__ = ["random_player", "rule_player", "completing_cells", "play_games", "MatchResult"]


def random_player(game: LineGame, boards: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A uniformly random empty cell for every board."""
    return np.where(boards == 0, rng.random(boards.shape), -1.0).argmax(axis=1)


def completing_cells(game: LineGame, boards: np.ndarray, player: np.ndarray) -> np.ndarray:
    """Empty cells with which ``player`` (+1 or -1 per board) would complete a line; boolean (B, n_cells)."""
    player = np.asarray(player).reshape(-1, 1)
    own = (boards == player).astype(np.float32) @ game.incidence.T
    other = (boards == -player).astype(np.float32) @ game.incidence.T
    ready = ((own == game.k - 1) & (other == 0)).astype(np.float32)  # lines that lack one stone
    return ((ready @ game.incidence) > 0) & (boards == 0)


def rule_player(game: LineGame, boards: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Win if you can, block if you must, otherwise any empty cell."""
    me = game.to_move(boards)
    wins = completing_cells(game, boards, me)
    blocks = completing_cells(game, boards, -me)
    score = np.where(boards == 0, rng.random(boards.shape), -1.0) + 2.0 * blocks + 4.0 * wins
    return score.argmax(axis=1)


@dataclass
class MatchResult:
    """Outcomes of a set of games between two players.

    ``outcome`` is +1 where the player who moved first won, -1 where the second
    player won and 0 for a draw; ``plies`` is the number of moves each game took.
    """

    outcome: np.ndarray
    plies: np.ndarray

    @property
    def first_wins(self) -> int:
        return int((self.outcome == 1).sum())

    @property
    def second_wins(self) -> int:
        return int((self.outcome == -1).sum())

    @property
    def draws(self) -> int:
        return int((self.outcome == 0).sum())

    def __repr__(self) -> str:
        return (
            f"MatchResult({len(self.outcome)} games: first player won {self.first_wins}, "
            f"second {self.second_wins}, drawn {self.draws}; {self.plies.mean():.1f} moves on average)"
        )


def play_games(
    game: LineGame, first, second, n_games: int, rng: np.random.Generator, start: np.ndarray | None = None
) -> MatchResult:
    """Play ``n_games`` games side by side, ``first`` moving first.

    ``start`` optionally gives the positions to start from (all with the same
    side to move); by default every game starts from the empty board.
    """
    boards = game.empty(n_games) if start is None else np.array(start, dtype=np.int8)
    n_games = boards.shape[0]
    players = (first, second) if game.to_move(boards[:1])[0] == 1 else (second, first)
    outcome = game.winner(boards)
    plies = np.zeros(n_games, dtype=np.int64)
    active = ~game.finished(boards)
    turn = 0
    while active.any():
        idx = np.flatnonzero(active)
        moves = players[turn % 2](game, boards[idx], rng)
        boards[idx] = game.play(boards[idx], moves)
        plies[idx] += 1
        outcome[idx] = game.winner(boards[idx])
        active[idx] = ~game.finished(boards[idx])
        turn += 1
    return MatchResult(outcome=outcome, plies=plies)
