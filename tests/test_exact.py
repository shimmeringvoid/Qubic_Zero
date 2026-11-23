import numpy as np
import pytest

from qubic_zero import (
    check_wins_every_game,
    cube3,
    play_games,
    qubic,
    random_player,
    rule_player,
    solve,
    solve_table,
    threat_moves,
    threat_search,
    tictactoe,
)


@pytest.fixture(scope="module")
def table():
    return solve_table(tictactoe())


def test_tictactoe_has_5478_positions_and_is_a_draw(table):
    game = table.game
    assert len(table.boards) == 5478 and int(table.decision.sum()) == 4520
    winners = game.winner(table.boards[table.terminal])
    assert ((winners == 1).sum(), (winners == -1).sum(), (winners == 0).sum()) == (626, 316, 16)
    assert table.value[0] == 0 and table.best[0].all()  # every opening move keeps the draw
    assert set(np.unique(table.value).tolist()) == {-1.0, 0.0, 1.0}
    assert (table.best == table.fastest).all()


def test_table_values_obey_the_minimax_rule(table):
    for i in np.flatnonzero(table.decision)[::37]:
        kids = table.child[i][table.child[i] >= 0]
        assert table.value[i] == max(-table.value[kids])
        assert (table.best[i] == ((table.child[i] >= 0) & (-table.value[np.maximum(table.child[i], 0)] == table.value[i]))).all()
    assert (table.lookup(table.boards[[5, 99, 1234]]) == [5, 99, 1234]).all()


def test_a_discount_keeps_the_outcomes_and_prefers_quick_wins(table):
    discounted = solve_table(tictactoe(), discount=0.9)
    assert (discounted.outcome == table.outcome).all()
    assert (discounted.best == table.best).all()
    assert (discounted.fastest <= discounted.best).all() and (discounted.fastest.sum() < discounted.best.sum())
    game = table.game
    board = np.array([1, 1, 0, -1, -1, 0, 0, 0, 0], dtype=np.int8)  # X wins at once on cell 2
    i = discounted.lookup(board)[0]
    assert discounted.value[i] == pytest.approx(0.9) and discounted.fastest[i].nonzero()[0].tolist() == [2]
    assert game.winner(game.play(board, 2)) == 1


def test_threat_search_never_claims_a_win_that_is_not_there(table):
    game = table.game
    found = 0
    for i in np.flatnonzero(table.decision):
        result = threat_search(game, table.boards[i])
        if result.win:
            found += 1
            assert table.outcome[i] == 1 and table.best[i, result.move]
        for move in threat_moves(game, table.boards[i]):
            assert table.best[i, move] and table.outcome[i] == 1
    assert found > 2500  # most won positions of tic-tac-toe are won by force


def test_solve_agrees_with_the_table(table):
    game = table.game
    for i in np.flatnonzero(table.decision)[::11]:
        assert solve(game, table.boards[i]) == table.outcome[i]
        assert solve(game, table.boards[i], use_threats=False) == table.outcome[i]
    won = table.boards[np.flatnonzero(table.terminal & (table.value == -1))[0]]
    assert solve(game, won) == -1


def test_three_cubed_is_a_win_for_the_first_player():
    game = cube3()
    assert solve(game, game.empty()) == 1
    assert solve(game, game.empty(), use_threats=False) == 1
    centre = game.play(game.empty(), 13)
    assert solve(game, centre) == -1  # whoever must answer the centre move has lost


def test_threat_search_is_sound_on_three_cubed():
    game = cube3()
    rng = np.random.default_rng(0)
    boards = game.empty(300)
    shared: dict = {}
    claimed = 0
    for _ in range(6):
        boards = boards[~game.finished(boards)]
        boards = game.play(boards, random_player(game, boards, rng))
        for board in boards[~game.finished(boards)][:40]:
            result = threat_search(game, board)
            if result.win:
                claimed += 1
                assert solve(game, board, use_threats=False, table=shared) == 1
                after = game.play(board, result.move)
                assert game.winner(after) != 0 or solve(game, after, use_threats=False, table=shared) == -1
    assert claimed > 50


def _board(game, xs, os):
    board = game.empty()
    board[list(xs)] = 1
    board[list(os)] = -1
    return board


def test_threat_search_on_qubic_positions_built_by_hand():
    game = qubic()
    cell = lambda x, y, z: 16 * x + 4 * y + z  # noqa: E731

    # three in a row: the fourth cell wins at once
    board = _board(game, [cell(0, 0, 0), cell(0, 0, 1), cell(0, 0, 2)], [cell(3, 3, 3), cell(3, 3, 2), cell(2, 0, 0)])
    assert threat_search(game, board).move == cell(0, 0, 3)
    assert threat_moves(game, board) == [cell(0, 0, 3)]

    # two lines through one empty cell, two stones on each: playing there threatens twice
    board = _board(game, [cell(0, 0, 1), cell(0, 0, 2), cell(1, 0, 0), cell(2, 0, 0)], [cell(3, 3, 3), cell(3, 3, 2), cell(2, 2, 1), cell(1, 3, 1)])
    result = threat_search(game, board)
    assert result.win and cell(0, 0, 0) in threat_moves(game, board)

    # the opponent has three in a row: with nothing forcing of one's own, there is no forced win
    board = _board(game, [cell(0, 0, 0), cell(1, 1, 1), cell(2, 3, 0)], [cell(3, 0, 0), cell(3, 0, 1), cell(3, 0, 2)])
    assert not threat_search(game, board).win and threat_moves(game, board) == []

    # nothing on the board: no forced win of this kind
    assert not threat_search(game, game.empty()).win

    # a tiny limit is reported, not hidden
    busy = play_until(game, 14)
    assert threat_search(game, busy, max_nodes=0).complete is False


def play_until(game, plies):
    rng = np.random.default_rng(7)
    board = game.empty(1)
    for _ in range(plies):
        board = game.play(board, random_player(game, board, rng))
    return board[0]


def test_forced_wins_found_on_qubic_hold_up_in_play():
    """From a position with a forced win, following the search beats a rule-based defender every time."""
    game = qubic()
    rng = np.random.default_rng(1)
    boards = game.empty(400)
    for _ in range(12):
        boards = game.play(boards, rule_player(game, boards, rng))
    boards = boards[~game.finished(boards)]
    starts = [b for b in boards if threat_search(game, b).win][:25]
    assert len(starts) >= 10

    def attacker(game, positions, rng):
        return np.array([threat_search(game, p).move for p in positions])

    for start in starts:
        result = play_games(game, attacker, rule_player, 1, rng, start=start[None])
        assert result.outcome[0] == game.to_move(start)


def test_a_perfect_player_passes_the_exhaustive_check_on_three_cubed():
    game = cube3()
    shared: dict = {}

    def perfect(board):
        for move in np.flatnonzero(board == 0):
            after = game.play(board, move)
            if game.winner(after) != 0 or solve(game, after, table=shared) == -1:
                return move
        raise AssertionError("no winning move in a position that should be won")

    reports = {mode: check_wins_every_game(game, perfect, threats=mode) for mode in ("off", "follow", "trust")}
    assert all(report.proven and report.decisions > 0 for report in reports.values())
    assert reports["trust"].by_threats > 0 and reports["off"].wins > 0
    assert reports["trust"].decisions <= reports["follow"].decisions


def test_the_exhaustive_check_catches_a_player_that_can_be_beaten():
    game = cube3()
    report = check_wins_every_game(game, lambda board: int(np.flatnonzero(board == 0)[0]), threats="off")
    assert not report.proven and report.failures
    position, reason = report.failures[0]
    assert "opponent" in reason or "filled" in reason
    # tic-tac-toe is a draw, so no first player passes
    draw = check_wins_every_game(tictactoe(), lambda board: int(np.flatnonzero(board == 0)[0]))
    assert not draw.proven
    limited = check_wins_every_game(game, lambda board: int(np.flatnonzero(board == 0)[0]), threats="off", max_decisions=3, max_failures=10**6)
    assert limited.open
