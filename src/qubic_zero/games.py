"""Games of cells and lines.

Tic-tac-toe, its 3 x 3 x 3 cousin and Qubic (4 x 4 x 4) are one game played on
different boards: two players take turns claiming a cell, and the first to own
every cell of a line wins.  Such a game is fully described by a table saying
which cells lie on which line, and everything in this package (the rules, the
symmetries, the exact searches, the network and the tree search) is written
against that table.  The small games are therefore not toy copies of the big
one: they run the same code.

A position is an array of ``n_cells`` small integers: +1 where the first
player ("X") has a stone, -1 for the second player ("O"), 0 for an empty cell.
Whose turn it is follows from the stones.  Arrays may carry any number of
leading batch dimensions.
"""

from __future__ import annotations

import itertools
from functools import cached_property, lru_cache

import numpy as np

__all__ = ["LineGame", "tictactoe", "cube3", "qubic", "game_by_name", "GAME_NAMES"]


class LineGame:
    """A board of cells and the lines on it.

    Parameters
    ----------
    name : short name, such as ``"4x4x4"``.
    shape : shape of the board, used for drawing and for convolutions.
    lines : integer array (n_lines, k); each row lists the cells of one line.
    """

    def __init__(self, name: str, shape: tuple[int, ...], lines) -> None:
        self.name = name
        self.shape = tuple(int(s) for s in shape)
        self.lines = np.asarray(lines, dtype=np.int64)
        self.n_cells = int(np.prod(self.shape))
        self.n_lines, self.k = self.lines.shape
        incidence = np.zeros((self.n_lines, self.n_cells), dtype=np.float32)
        incidence[np.arange(self.n_lines)[:, None], self.lines] = 1.0
        self.incidence = incidence  # incidence[l, c] = 1 when cell c lies on line l
        self.degree = incidence.sum(axis=0).astype(np.int64)  # lines through each cell

    def __repr__(self) -> str:
        return f"LineGame({self.name!r}: {self.n_cells} cells, {self.n_lines} lines of {self.k})"

    # --- construction ---------------------------------------------------------
    @classmethod
    def grid(cls, n: int, dim: int) -> LineGame:
        """The board with ``n`` cells along each of ``dim`` axes, and every straight line of ``n`` cells."""
        cells = list(itertools.product(range(n), repeat=dim))
        index = {cell: i for i, cell in enumerate(cells)}
        lines = set()
        for start in cells:
            for step in itertools.product((-1, 0, 1), repeat=dim):
                if not any(step):
                    continue
                points = [tuple(s + j * d for s, d in zip(start, step)) for j in range(n)]
                if all(0 <= x < n for point in points for x in point):
                    lines.add(tuple(sorted(index[point] for point in points)))
        return cls("x".join([str(n)] * dim), (n,) * dim, sorted(lines))

    # --- rules ----------------------------------------------------------------
    def empty(self, n: int | None = None) -> np.ndarray:
        """An empty board, or ``n`` of them."""
        return np.zeros(self.n_cells if n is None else (n, self.n_cells), dtype=np.int8)

    def to_move(self, boards: np.ndarray) -> np.ndarray:
        """+1 where the first player is to move, -1 where the second is."""
        return (1 - 2 * boards.sum(axis=-1)).astype(np.int8)

    def line_sums(self, boards: np.ndarray) -> np.ndarray:
        """Stones on every line, first player's minus second player's; shape (..., n_lines)."""
        return boards.astype(np.float32) @ self.incidence.T

    def winner(self, boards: np.ndarray) -> np.ndarray:
        """+1 or -1 where that player owns a whole line, 0 elsewhere."""
        sums = self.line_sums(boards)
        return (sums.max(axis=-1) == self.k).astype(np.int8) - (sums.min(axis=-1) == -self.k).astype(np.int8)

    def finished(self, boards: np.ndarray) -> np.ndarray:
        """True where the game is over: somebody has a line, or the board is full."""
        return (self.winner(boards) != 0) | (boards != 0).all(axis=-1)

    def play(self, boards: np.ndarray, moves) -> np.ndarray:
        """The positions after the player to move claims cell ``moves`` (one move per board)."""
        boards = np.array(boards, dtype=np.int8, copy=True)
        flat = boards.reshape(-1, self.n_cells)
        moves = np.asarray(moves, dtype=np.int64).reshape(-1)
        rows = np.arange(flat.shape[0])
        if (flat[rows, moves] != 0).any():
            raise ValueError("a move was made on a cell that is already taken")
        flat[rows, moves] = self.to_move(flat)
        return boards

    # --- symmetries -----------------------------------------------------------
    @cached_property
    def symmetries(self) -> np.ndarray:
        """Every relabelling of the cells that maps each line onto a line; shape (n_symmetries, n_cells).

        Found by exhaustive search from the table of lines alone, so nothing about
        the geometry of the board is assumed.  ``boards[..., symmetries[i]]`` is
        the i-th symmetric image of ``boards``, and the same indexing transforms
        anything else that has one entry per cell, such as a policy.  A cell
        ``c`` of the image is cell ``symmetries[i][c]`` of the original.
        """
        return _line_preserving_permutations(self.n_cells, [tuple(line) for line in self.lines])

    def canonical(self, boards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One representative of each position's symmetry class, and the symmetry that produces it.

        Returns ``(images, which)`` with ``images[b] == boards[b][symmetries[which[b]]]``.
        Two positions get the same image exactly when a symmetry maps one onto the other.
        """
        boards = np.asarray(boards, dtype=np.int8)
        single = boards.ndim == 1
        flat = boards.reshape(-1, self.n_cells)
        images = flat[:, self.symmetries]  # (B, G, n_cells)
        digits = (images + 1).astype(np.uint64)
        shifts = (2 * (np.arange(self.n_cells) % 32)).astype(np.uint64)
        low = (digits[..., :32] << shifts[:32]).sum(axis=-1, dtype=np.uint64)
        high = (digits[..., 32:] << shifts[32:]).sum(axis=-1, dtype=np.uint64)
        which = np.lexsort((low, high), axis=-1)[:, 0]
        chosen = images[np.arange(flat.shape[0]), which]
        if single:
            return chosen[0], int(which[0])
        return chosen.reshape(boards.shape), which.reshape(boards.shape[:-1])

    # --- for people -----------------------------------------------------------
    def render(self, board: np.ndarray) -> str:
        """The board as text: X, O and dots, one square layer after another for a cube."""
        marks = np.array(["O", ".", "X"])[np.asarray(board, dtype=np.int64).reshape(self.shape) + 1]
        if marks.ndim == 2:
            marks = marks[None]
        rows = [
            "   ".join(" ".join(layer[r]) for layer in marks)  # the layers side by side
            for r in range(marks.shape[1])
        ]
        return "\n".join(rows)


def _line_preserving_permutations(n_cells: int, lines: list[tuple[int, ...]]) -> np.ndarray:
    """All permutations of the cells under which the image of every line is a line (backtracking)."""
    line_set = {frozenset(line) for line in lines}
    through = [[] for _ in range(n_cells)]
    for line in lines:
        for c in line:
            through[c].append(line)
    degree = [len(t) for t in through]
    collinear = [[False] * n_cells for _ in range(n_cells)]
    for line in lines:
        for a in line:
            for b in line:
                collinear[a][b] = a != b

    # Visit the cells in an order that completes lines early, so that wrong images are rejected early.
    order, seen, frontier = [], set(), [0]
    while len(order) < n_cells:
        if not frontier:
            frontier = [min(set(range(n_cells)) - seen)]
        c = frontier.pop(0)
        if c in seen:
            continue
        seen.add(c)
        order.append(c)
        frontier.extend(x for line in through[c] for x in line if x not in seen)
    position = {c: i for i, c in enumerate(order)}
    closes = [[] for _ in range(n_cells)]  # the lines whose last cell (in this order) is c
    for line in lines:
        closes[max(line, key=position.get)].append(line)

    image = [-1] * n_cells
    used = [False] * n_cells
    found = []

    def extend(depth: int) -> None:
        if depth == n_cells:
            found.append(tuple(image))
            return
        c = order[depth]
        for target in range(n_cells):
            if used[target] or degree[target] != degree[c]:
                continue
            if any(collinear[c][a] != collinear[target][image[a]] for a in order[:depth]):
                continue
            image[c] = target
            if all(frozenset(image[x] for x in line) in line_set for line in closes[c]):
                used[target] = True
                extend(depth + 1)
                used[target] = False
            image[c] = -1

    extend(0)
    return np.array(sorted(found), dtype=np.int64)


@lru_cache(maxsize=None)
def tictactoe() -> LineGame:
    """Ordinary 3 x 3 tic-tac-toe: 9 cells, 8 lines.  A draw with best play."""
    return LineGame.grid(3, 2)


@lru_cache(maxsize=None)
def cube3() -> LineGame:
    """3 x 3 x 3 tic-tac-toe: 27 cells, 49 lines.  An easy win for the first player."""
    return LineGame.grid(3, 3)


@lru_cache(maxsize=None)
def qubic() -> LineGame:
    """Qubic, 4 x 4 x 4 tic-tac-toe: 64 cells, 76 lines.  A win for the first player (Patashnik, 1980)."""
    return LineGame.grid(4, 3)


GAME_NAMES = {"3x3": tictactoe, "3x3x3": cube3, "4x4x4": qubic}


def game_by_name(name: str) -> LineGame:
    """The game called ``name``: one of ``"3x3"``, ``"3x3x3"``, ``"4x4x4"``."""
    try:
        return GAME_NAMES[name]()
    except KeyError:
        raise ValueError(f"unknown game {name!r}; choose from {sorted(GAME_NAMES)}") from None
