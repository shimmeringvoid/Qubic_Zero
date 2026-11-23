"""Play against a trained network, with NumPy alone.

:class:`NumpyPlayer` is the trained player without PyTorch and without a GPU:
the line network as plain arrays, the one-position tree search of
:mod:`qubic_zero.search`, and (unless switched off) the exact threat search
in front of both, so that a win by force is taken whenever one is found.

From a terminal::

    python -m qubic_zero.play results/3x3x3-discount-0.95/seed-0/weights/latest.npz

plays a game against the player stored in that file; add ``--second`` to let
it move first.
"""

from __future__ import annotations

import argparse

import numpy as np

from .exact import threat_search
from .games import LineGame
from .net import NumpyLineNet
from .search import masked_policy, reference_search

__all__ = ["NumpyPlayer", "main"]


class NumpyPlayer:
    """A trained line network that plays with tree search, using NumPy only.

    ``n_sims`` simulations are run for every move and the most visited move is
    played.  With ``threats=True`` the exact threat search is asked first, and
    its move is played when it finds a win by force.  ``player.said`` holds what
    was last decided and why.
    """

    def __init__(self, weights, n_sims: int = 200, c_puct: float = 1.5, discount: float = 0.98, threats: bool = True) -> None:
        self.net = NumpyLineNet(weights)
        self.game: LineGame = self.net.game
        self.n_sims, self.c_puct, self.discount, self.threats = n_sims, c_puct, discount, threats
        self.said = ""

    def _evaluate(self, position):
        scores, value = self.net(position)
        return masked_policy(scores[0], position), float(value[0])

    def move(self, board: np.ndarray) -> int:
        """The move for one position."""
        board = np.asarray(board, dtype=np.int8)
        if self.threats:
            forced = threat_search(self.game, board)
            if forced.win:
                self.said = "a win by force"
                return int(forced.move)
        visits, values = reference_search(self.game, board, self._evaluate, self.n_sims, self.c_puct, self.discount)
        move = int(visits.argmax())
        self.said = f"value {values[move]:+.2f} after {int(visits[move])} of {self.n_sims} visits"
        return move

    def __call__(self, game: LineGame, boards: np.ndarray, rng=None) -> np.ndarray:
        """As a player in the sense of :mod:`qubic_zero.players`: a move for each of a batch of positions."""
        return np.array([self.move(board) for board in np.asarray(boards).reshape(-1, game.n_cells)])


def _show(game: LineGame, board: np.ndarray) -> str:
    if len(game.shape) == 3:
        heading = "   ".join(f"layer {layer + 1}".ljust(2 * game.shape[2] - 1) for layer in range(game.shape[0]))
        return heading + "\n" + game.render(board)
    return game.render(board)


def _ask(game: LineGame, board: np.ndarray) -> int:
    names = "layer row column" if len(game.shape) == 3 else "row column"
    while True:
        answer = input(f"your move ({names}, counted from 1): ").split()
        try:
            cell = int(np.ravel_multi_index([int(a) - 1 for a in answer], game.shape))
        except ValueError:
            print(f"  that is not a cell: give {len(game.shape)} numbers, each from 1 to {game.shape[0]}")
            continue
        if board[cell] != 0:
            print("  that cell is taken")
            continue
        return cell


def main(argv=None) -> None:
    """``python -m qubic_zero.play WEIGHTS``: a game in the terminal against the network stored in ``WEIGHTS``."""
    parser = argparse.ArgumentParser(description="Play against a trained network in the terminal (NumPy only).")
    parser.add_argument("weights", help="a .npz file written by qubic_zero.net.save_weights")
    parser.add_argument("--second", action="store_true", help="let the network move first")
    parser.add_argument("--sims", type=int, default=200, help="simulations of the tree search per move")
    parser.add_argument("--no-threats", action="store_true", help="play without the exact threat search")
    args = parser.parse_args(argv)
    player = NumpyPlayer(args.weights, n_sims=args.sims, threats=not args.no_threats)
    game = player.game
    board = game.empty()
    human = -1 if args.second else 1
    print(f"{game.name}: you are {'X' if human == 1 else 'O'}, and X moves first\n")
    while not game.finished(board):
        if game.to_move(board) == human:
            print(_show(game, board))
            move = _ask(game, board)
        else:
            move = player.move(board)
            where = " ".join(str(int(c) + 1) for c in np.unravel_index(move, game.shape))
            print(f"the network plays {where}  ({player.said})\n")
        board = game.play(board, move)
    print(_show(game, board))
    winner = int(game.winner(board))
    print("a draw" if winner == 0 else ("you win" if winner == human else "the network wins"))


if __name__ == "__main__":
    main()
