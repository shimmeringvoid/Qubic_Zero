import numpy as np

from qubic_zero import completing_cells, cube3, play_games, qubic, random_player, rule_player, tictactoe


def test_random_player_only_picks_empty_cells():
    game = qubic()
    rng = np.random.default_rng(0)
    boards = game.empty(200)
    for _ in range(30):
        moves = random_player(game, boards, rng)
        assert (boards[np.arange(200), moves] == 0).all()
        boards = game.play(boards, moves)
    assert len(set(random_player(game, game.empty(200), rng).tolist())) > 30


def test_rule_player_wins_when_it_can_and_blocks_when_it_must():
    game = tictactoe()
    rng = np.random.default_rng(0)
    can_win = np.array([1, 1, 0, -1, -1, 0, 0, 0, 0], dtype=np.int8)  # X to move: 2 wins, 5 would only block
    must_block = np.array([1, 1, 0, -1, 0, 0, 0, 0, 0], dtype=np.int8)  # O to move
    for _ in range(20):
        assert rule_player(game, can_win[None], rng)[0] == 2
        assert rule_player(game, must_block[None], rng)[0] == 2
    assert completing_cells(game, can_win[None], np.array([1])).nonzero()[1].tolist() == [2]
    assert completing_cells(game, can_win[None], np.array([-1])).nonzero()[1].tolist() == [5]


def test_rule_player_beats_random_player_from_either_side():
    rng = np.random.default_rng(3)
    for game in (cube3(), qubic()):
        as_first = play_games(game, rule_player, random_player, 300, rng)
        as_second = play_games(game, random_player, rule_player, 300, rng)
        assert as_first.first_wins > 280 and as_second.second_wins > 280


def test_games_have_sensible_lengths_and_outcomes():
    game = qubic()
    result = play_games(game, random_player, random_player, 300, np.random.default_rng(4))
    assert result.plies.min() >= 2 * game.k - 1 and result.plies.max() <= game.n_cells
    assert result.first_wins + result.second_wins + result.draws == 300
    # a first-player win takes an odd number of moves, a second-player win an even number
    assert (result.plies[result.outcome == 1] % 2 == 1).all()
    assert (result.plies[result.outcome == -1] % 2 == 0).all()


def test_games_can_start_from_given_positions():
    game = tictactoe()
    start = np.array([[1, 1, 0, -1, -1, 0, 0, 0, 0]] * 10, dtype=np.int8)
    result = play_games(game, rule_player, random_player, 10, np.random.default_rng(5), start=start)
    assert (result.outcome == 1).all() and (result.plies == 1).all()
