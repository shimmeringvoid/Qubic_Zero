import itertools

import numpy as np
import pytest

from qubic_zero import LineGame, cube3, game_by_name, play_games, qubic, random_player, tictactoe


@pytest.mark.parametrize(
    "name, cells, lines, k, symmetries",
    [("3x3", 9, 8, 3, 8), ("3x3x3", 27, 49, 3, 48), ("4x4x4", 64, 76, 4, 192)],
)
def test_boards_have_the_known_numbers_of_lines_and_symmetries(name, cells, lines, k, symmetries):
    game = game_by_name(name)
    assert (game.n_cells, game.n_lines, game.k) == (cells, lines, k)
    assert game.symmetries.shape == (symmetries, cells)
    assert int(game.incidence.sum()) == lines * k


def test_qubic_has_16_cells_on_seven_lines_and_48_on_four():
    game = qubic()
    assert sorted(np.bincount(game.degree)[[4, 7]].tolist()) == [16, 48]
    # the sixteen rich cells are the eight corners and the eight cells of the inner cube
    rich = np.argwhere((game.degree == 7).reshape(4, 4, 4))
    assert all(set(cell) <= {0, 3} or set(cell) <= {1, 2} for cell in rich.tolist())


@pytest.mark.parametrize("make", [tictactoe, cube3, qubic])
def test_symmetries_map_lines_to_lines_and_form_a_group(make):
    game = make()
    perms = game.symmetries
    line_set = {frozenset(line) for line in game.lines.tolist()}
    for perm in perms:
        assert {frozenset(perm[line].tolist()) for line in game.lines} == line_set
    as_set = {tuple(p) for p in perms.tolist()}
    assert tuple(range(game.n_cells)) in as_set
    rng = np.random.default_rng(0)
    for i, j in rng.integers(0, len(perms), size=(200, 2)):
        assert tuple(perms[i][perms[j]].tolist()) in as_set


def test_the_48_rotations_and_reflections_of_the_cube_are_among_qubics_192():
    game = qubic()
    cells = np.array(list(itertools.product(range(4), repeat=3)))
    as_set = {tuple(p) for p in game.symmetries.tolist()}
    geometric = set()
    for axes in itertools.permutations(range(3)):
        for flips in itertools.product((False, True), repeat=3):
            moved = cells[:, list(axes)].copy()
            for a, flip in enumerate(flips):
                if flip:
                    moved[:, a] = 3 - moved[:, a]
            geometric.add(tuple(np.ravel_multi_index(moved.T, (4, 4, 4)).tolist()))
    assert len(geometric) == 48 and geometric <= as_set


def test_swapping_the_two_inner_layers_is_a_symmetry_that_is_not_geometric():
    game = qubic()
    relabel = np.array([0, 2, 1, 3])
    cells = np.array(list(itertools.product(range(4), repeat=3)))
    perm = tuple(np.ravel_multi_index(relabel[cells].T, (4, 4, 4)).tolist())
    assert perm in {tuple(p) for p in game.symmetries.tolist()}
    # ... and it separates cells that were neighbours
    assert abs(relabel[0] - relabel[1]) == 2


def test_rules_on_a_small_board():
    game = tictactoe()
    board = game.empty()
    assert game.to_move(board) == 1 and game.winner(board) == 0 and not game.finished(board)
    for move in (0, 3, 1, 4):
        board = game.play(board, move)
    assert board.tolist() == [1, 1, 0, -1, -1, 0, 0, 0, 0] and game.to_move(board) == 1
    won = game.play(board, 2)
    assert game.winner(won) == 1 and game.finished(won)
    with pytest.raises(ValueError):
        game.play(board, 0)
    lost = game.play(game.play(board, 8), 5)
    assert game.winner(lost) == -1
    assert "X X ." in game.render(board)


def test_batches_of_boards():
    game = qubic()
    boards = game.empty(5)
    boards = game.play(boards, [0, 1, 2, 3, 4])
    assert boards.sum() == 5 and (game.to_move(boards) == -1).all()
    line = game.lines[10]
    full = game.empty()
    full[line] = -1
    assert game.winner(np.stack([full, game.empty()])).tolist() == [-1, 0]


@pytest.mark.parametrize("make", [tictactoe, cube3, qubic])
def test_canonical_image_is_the_same_for_symmetric_positions(make):
    game = make()
    rng = np.random.default_rng(1)
    boards = game.empty(40)
    for _ in range(5):
        boards = game.play(boards, random_player(game, boards, rng))
    images, which = game.canonical(boards)
    assert (images == boards[np.arange(40)[:, None], game.symmetries[which]]).all()
    shuffled = boards[np.arange(40)[:, None], game.symmetries[rng.integers(0, len(game.symmetries), 40)]]
    assert (game.canonical(shuffled)[0] == images).all()
    one_image, one_which = game.canonical(boards[3])
    assert (one_image == images[3]).all() and one_which == which[3]
    # positions that are not symmetric to each other get different images
    distinct = {b.tobytes() for b in images}
    assert len(distinct) > 1


def test_three_cubed_never_ends_in_a_draw():
    game = cube3()
    result = play_games(game, random_player, random_player, 500, np.random.default_rng(2))
    assert result.draws == 0


def test_any_table_of_lines_makes_a_game():
    game = LineGame("pair", (1, 3), [[0, 1], [1, 2]])
    assert game.n_cells == 3 and game.k == 2 and len(game.symmetries) == 2
    assert game.winner(np.array([1, 1, 0])) == 1
