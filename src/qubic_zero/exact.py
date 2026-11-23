"""Exact answers: what a learner is scored against.

Three tools, in order of how much of a game they can cover:

* :func:`solve_table` works out the value of *every* reachable position by
  exhaustive search.  Only tiny games allow that (3 x 3 has 5,478 positions).
* :func:`solve` finds the value of one position by searching its game tree to
  the end.  It is practical on 3 x 3 x 3 and hopeless on Qubic.
* :func:`threat_search` looks only for a win by force: a chain of moves that
  each threaten to complete a line, so that every reply is dictated, ending in
  two threats at once.  When it reports a win, the win is certain; when it
  reports none, there may still be a win that starts with a quiet move.  This is
  the search that carries over to Qubic, and it is the one both published
  solutions of the game are built on (Patashnik, 1980; Allis, 1994).

:func:`check_wins_every_game` then turns "this player wins every game it
starts" into a finite check: it tries every reply at every turn.

The searches work on bitboards: a set of cells is a Python integer with one
bit per cell, and a position is the pair (stones of the player to move, stones
of the other player).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .games import LineGame

__all__ = [
    "Table",
    "solve_table",
    "solve",
    "threat_search",
    "threat_moves",
    "ThreatResult",
    "check_wins_every_game",
    "WalkReport",
    "SearchBudgetExceeded",
    "to_bits",
]


class SearchBudgetExceeded(RuntimeError):
    """A search was stopped because it reached its limit on the number of positions."""


# --- bitboards -----------------------------------------------------------------
class _Bits:
    def __init__(self, game: LineGame) -> None:
        self.k = game.k
        self.n_cells = game.n_cells
        self.full = (1 << game.n_cells) - 1
        self.line_masks = [sum(1 << int(c) for c in line) for line in game.lines]
        # empty cells are tried in this order: those on the most lines first
        self.by_degree = [1 << int(c) for c in np.argsort(-game.degree, kind="stable")]


def _bits(game: LineGame) -> _Bits:
    cached = getattr(game, "_bits_cache", None)
    if cached is None:
        cached = game._bits_cache = _Bits(game)
    return cached


def to_bits(game: LineGame, board: np.ndarray) -> tuple[int, int]:
    """A position as (stones of the player to move, stones of the other player), one bit per cell."""
    board = np.asarray(board)
    me = int(game.to_move(board))
    mine = sum(1 << int(c) for c in np.flatnonzero(board == me))
    theirs = sum(1 << int(c) for c in np.flatnonzero(board == -me))
    return mine, theirs


def _cells(mask: int) -> list[int]:
    cells = []
    while mask:
        low = mask & -mask
        cells.append(low.bit_length() - 1)
        mask ^= low
    return cells


def _scan(bits: _Bits, att: int, dfn: int):
    """What matters tactically in a position where ``att`` is to move.

    Returns ``(win, must, pairs)``: a cell (as a bit) that completes a line for
    the player to move, or 0; the cells where the other player threatens to
    complete a line; and, for every empty cell that would create a threat of the
    player to move, the cells that this threat (or these threats) would aim at.
    """
    k = bits.k
    must = 0
    pairs: dict[int, int] = {}
    for mask in bits.line_masks:
        a = att & mask
        d = dfn & mask
        if d == 0:
            count = a.bit_count()
            if count == k - 1:
                return mask ^ a, 0, pairs
            if count == k - 2:
                rest = mask ^ a
                low = rest & -rest
                high = rest ^ low
                pairs[low] = pairs.get(low, 0) | high
                pairs[high] = pairs.get(high, 0) | low
        elif a == 0 and d.bit_count() == k - 1:
            must |= mask ^ d
    return 0, must, pairs


# --- forced wins by threats ------------------------------------------------------
@dataclass
class ThreatResult:
    """Outcome of :func:`threat_search`.

    ``move`` is a first move of a forced win for the player to move, or ``None``
    if the search found none.  ``complete`` is False when the search ran into its
    limit, in which case "none found" means less still.  ``nodes`` counts the
    positions looked at.
    """

    move: int | None
    nodes: int
    complete: bool = True

    @property
    def win(self) -> bool:
        return self.move is not None


def _threat_dfs(bits: _Bits, att: int, dfn: int, memo: dict, budget: list) -> int:
    """A first move (as a bit) of a win by continuous threats for ``att``, or 0."""
    key = (att, dfn)
    known = memo.get(key)
    if known is not None:
        return known
    budget[0] -= 1
    if budget[0] < 0:
        raise SearchBudgetExceeded
    win, must, pairs = _scan(bits, att, dfn)
    if win:
        memo[key] = win
        return win
    result = 0
    if must:
        # The other player threatens: the only move is to block, and the chain of
        # threats continues only if the block itself makes a threat.
        candidates = [must] if (must & (must - 1)) == 0 and must in pairs else []
    else:
        # two threats at once first: they win on the spot
        candidates = sorted(pairs, key=lambda c: (pairs[c] & (pairs[c] - 1)) == 0)
    for cell in candidates:
        aims = pairs[cell]
        if aims & (aims - 1):  # two or more cells threatened: the reply can cover only one
            result = cell
            break
        # one threat: the reply is forced onto that cell, and then it is ``att``'s move again
        if _threat_dfs(bits, att | cell, dfn | aims, memo, budget):
            result = cell
            break
    memo[key] = result
    return result


def threat_search(game: LineGame, board: np.ndarray, max_nodes: int = 200_000) -> ThreatResult:
    """Look for a win by force for the player to move: threats all the way.

    The player to move may complete a line at once; otherwise every move of the
    winning line must threaten to complete one, which leaves the opponent a
    single reply (the block), until a move makes two threats at once.  Blocks
    that themselves threaten are followed too: they force the attacker in turn.
    A reported win is a certain win.  ``None`` only means that no win of this
    particular kind was found.
    """
    bits = _bits(game)
    att, dfn = to_bits(game, board)
    budget = [max_nodes]
    try:
        move = _threat_dfs(bits, att, dfn, {}, budget)
    except SearchBudgetExceeded:
        return ThreatResult(move=None, nodes=max_nodes, complete=False)
    return ThreatResult(move=(move.bit_length() - 1) if move else None, nodes=max_nodes - budget[0])


def threat_moves(game: LineGame, board: np.ndarray, max_nodes: int = 200_000) -> list[int]:
    """Every first move from which :func:`threat_search` can force a win for the player to move."""
    bits = _bits(game)
    att, dfn = to_bits(game, board)
    win, must, pairs = _scan(bits, att, dfn)
    k = bits.k
    if win:  # report every cell that completes a line, not only the first one found
        return sorted(
            (mask ^ (att & mask)).bit_length() - 1
            for mask in bits.line_masks
            if (dfn & mask) == 0 and (att & mask).bit_count() == k - 1
        )
    if must:
        candidates = [must] if (must & (must - 1)) == 0 and must in pairs else []
    else:
        candidates = list(pairs)
    memo: dict = {}
    budget = [max_nodes]
    winners = set()
    for cell in candidates:
        aims = pairs[cell]
        try:
            if aims & (aims - 1) or _threat_dfs(bits, att | cell, dfn | aims, memo, budget):
                winners.add(cell.bit_length() - 1)
        except SearchBudgetExceeded:
            break
    return sorted(winners)


# --- the value of one position ---------------------------------------------------
def _solve(bits: _Bits, att: int, dfn: int, table: dict, budget: list, use_threats: bool) -> int:
    key = (att, dfn)
    known = table.get(key)
    if known is not None:
        return known
    budget[0] -= 1
    if budget[0] < 0:
        raise SearchBudgetExceeded
    win, must, pairs = _scan(bits, att, dfn)
    if win:
        table[key] = 1
        return 1
    empty = bits.full & ~(att | dfn)
    if not empty:
        table[key] = 0
        return 0
    if must:
        if must & (must - 1):  # two lines to block and one move to do it with
            table[key] = -1
            return -1
        moves = [must]
    else:
        if use_threats:
            try:
                if _threat_dfs(bits, att, dfn, {}, [2_000]):
                    table[key] = 1
                    return 1
            except SearchBudgetExceeded:
                pass
        threats = sorted(pairs, key=lambda c: (pairs[c] & (pairs[c] - 1)) == 0)
        quiet = [c for c in bits.by_degree if c & empty and c not in pairs]
        moves = threats + quiet
    best = -1
    for cell in moves:
        value = -_solve(bits, dfn, att | cell, table, budget, use_threats)
        if value > best:
            best = value
            if best == 1:
                break
    table[key] = best
    return best


def solve(
    game: LineGame, board: np.ndarray, max_nodes: int = 2_000_000, use_threats: bool = True, table: dict | None = None
) -> int | None:
    """The value of a position for the player to move, by searching to the end of the game.

    Returns +1 (a win with best play), 0 (a draw) or -1 (a loss); ``None`` if the
    search would need more than ``max_nodes`` positions.  ``table`` may be a
    dictionary that is shared between calls, so that positions are not searched
    twice.  With ``use_threats=False`` the search does not lean on
    :func:`threat_search`, which makes it an independent check of that function.
    """
    bits = _bits(game)
    att, dfn = to_bits(game, board)
    if game.winner(np.asarray(board)) != 0:
        return -1  # the player who just moved has a line
    try:
        return _solve(bits, att, dfn, {} if table is None else table, [max_nodes], use_threats)
    except SearchBudgetExceeded:
        return None


# --- every position of a tiny game -------------------------------------------------
@dataclass
class Table:
    """Every reachable position of a game with its exact value (see :func:`solve_table`).

    All arrays are indexed by position.  ``value`` is from the point of view of
    the player to move and lies in [-1, 1]; ``outcome`` is its sign.  ``best``
    marks the moves that keep the outcome (a win stays a win, a draw a draw);
    ``fastest`` marks the moves that also reach it as quickly, or put it off as
    long, as possible, which is what a discount below 1 rewards.
    """

    game: LineGame
    discount: float
    boards: np.ndarray
    value: np.ndarray
    outcome: np.ndarray
    terminal: np.ndarray
    best: np.ndarray
    fastest: np.ndarray
    child: np.ndarray
    index: dict = field(repr=False, default_factory=dict)

    @property
    def decision(self) -> np.ndarray:
        """The positions in which there is a move to make."""
        return ~self.terminal

    def lookup(self, boards: np.ndarray) -> np.ndarray:
        """Row numbers of the given positions."""
        boards = np.asarray(boards, dtype=np.int8).reshape(-1, self.game.n_cells)
        return np.array([self.index[b.tobytes()] for b in boards], dtype=np.int64)


def solve_table(game: LineGame, discount: float = 1.0, max_positions: int = 2_000_000) -> Table:
    """Exact values of all positions that can arise from the empty board.

    A finished game is worth -1 to the player to move if the other player has
    just completed a line, and 0 if the board is full.  Otherwise a position is
    worth the best of ``-discount * value(next position)`` over its moves.  With
    ``discount=1`` values are +1, 0 or -1; with a discount below 1 a win in few
    moves is worth more than a win in many, and a late loss is better than an
    early one.
    """
    n = game.n_cells
    index: dict[bytes, int] = {game.empty().tobytes(): 0}
    boards = [game.empty()]
    frontier = [0]
    children: dict[int, list[tuple[int, int]]] = {}
    while frontier:
        batch = np.stack([boards[i] for i in frontier])
        over = game.finished(batch)
        mover = game.to_move(batch)
        following = []
        for row, i in enumerate(frontier):
            if over[row]:
                continue
            links = []
            for move in np.flatnonzero(batch[row] == 0):
                child = batch[row].copy()
                child[move] = mover[row]
                key = child.tobytes()
                j = index.get(key)
                if j is None:
                    j = index[key] = len(boards)
                    boards.append(child)
                    following.append(j)
                    if len(boards) > max_positions:
                        raise SearchBudgetExceeded(f"{game.name} has more than {max_positions:,} positions")
                links.append((int(move), j))
            children[i] = links
        frontier = following

    boards_array = np.stack(boards)
    total = len(boards)
    terminal = game.finished(boards_array)
    value = np.zeros(total)
    value[terminal] = -np.abs(game.winner(boards_array[terminal]))
    child = np.full((total, n), -1, dtype=np.int64)
    move_value = np.full((total, n), -np.inf)
    # children always have more stones than their parents, and were numbered later
    for i in range(total - 1, -1, -1):
        if terminal[i]:
            continue
        for move, j in children[i]:
            child[i, move] = j
            move_value[i, move] = -discount * value[j]
        value[i] = move_value[i].max()
    value = np.round(value, 12) + 0.0  # no negative zeros
    outcome = np.sign(value).astype(np.int8)
    legal = child >= 0
    child_outcome = np.where(legal, outcome[np.maximum(child, 0)], 0)
    best = legal & (-child_outcome == outcome[:, None])
    fastest = legal & np.isclose(move_value, value[:, None])
    return Table(game, discount, boards_array, value, outcome, terminal, best, fastest, child, index)


# --- "wins every game it starts", checked ---------------------------------------------
@dataclass
class WalkReport:
    """What :func:`check_wins_every_game` found.

    ``decisions`` counts the distinct positions in which the player had to
    choose a move.  ``wins`` counts the lines of play that ended with its line
    on the board, ``by_threats`` the positions closed by a forced win that the
    threat search vouched for.  ``failures`` lists positions in which the game
    ended without a win for the player; ``open`` lists positions that were not
    examined because the limit was reached.  The claim is proven when both
    lists are empty.
    """

    decisions: int = 0
    wins: int = 0
    by_threats: int = 0
    deepest: int = 0
    failures: list = field(default_factory=list)
    open: list = field(default_factory=list)

    @property
    def proven(self) -> bool:
        return not self.failures and not self.open


def check_wins_every_game(
    game: LineGame,
    player,
    start: np.ndarray | None = None,
    threats: str = "follow",
    max_decisions: int = 200_000,
    max_failures: int = 20,
) -> WalkReport:
    """Does ``player`` win from ``start`` whatever the opponent replies?

    ``player(board) -> move`` must be deterministic.  It is always asked about
    the canonical image of a position (:meth:`LineGame.canonical`) and its answer
    is mapped back, so that positions which differ by a symmetry need to be
    examined only once; the player that is checked is therefore "``player``
    applied to the canonical image", and that is the player to use afterwards.

    ``threats`` says how the exact threat search takes part:

    * ``"off"``: the player's own move is followed in every position;
    * ``"follow"``: where the threat search finds a forced win, its move is played
      instead of the player's, and the win is then verified move by move like
      everything else;
    * ``"trust"``: as above, but a position with a forced win is closed at once,
      on the word of the threat search.  Much less work, and no longer a check of
      the threat search itself.
    """
    if threats not in ("off", "follow", "trust"):
        raise ValueError("threats must be 'off', 'follow' or 'trust'")
    report = WalkReport()
    seen: set[bytes] = set()
    board = game.empty() if start is None else np.array(start, dtype=np.int8)
    me = int(game.to_move(board))

    def visit(board: np.ndarray, depth: int) -> None:
        image, which = game.canonical(board)
        key = image.tobytes()
        if key in seen:
            return
        seen.add(key)
        if len(report.failures) >= max_failures:
            return
        if report.decisions >= max_decisions:
            report.open.append(image)
            return
        report.decisions += 1
        report.deepest = max(report.deepest, depth)
        move = None
        if threats != "off":
            forced = threat_search(game, image)
            if forced.win:
                if threats == "trust":
                    report.by_threats += 1
                    return
                move = forced.move
        if move is None:
            move = int(player(image))
        after = image.copy()
        if after[move] != 0:
            report.failures.append((image, "the player chose a cell that was taken"))
            return
        after[move] = me
        if game.winner(after) == me:
            report.wins += 1
            return
        replies = np.flatnonzero(after == 0)
        if replies.size == 0:
            report.failures.append((after, "the board filled up without a winner"))
            return
        for reply in replies:
            nxt = after.copy()
            nxt[reply] = -me
            if game.winner(nxt) == -me:
                report.failures.append((nxt, "the opponent completed a line"))
            elif (nxt != 0).all():
                report.failures.append((nxt, "the board filled up without a winner"))
            else:
                visit(nxt, depth + 1)

    visit(board, 0)
    return report
