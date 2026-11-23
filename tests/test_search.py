"""The two implementations of the tree search must be the same search."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from qubic_zero import cube3, qubic, random_player, solve_table, tictactoe  # noqa: E402
from qubic_zero.search import BatchedSearch, masked_policy, reference_search  # noqa: E402


def stand_in_network(game, seed=0):
    """A fixed, arbitrary function of the position in place of a network, the same for one position or many."""
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=(game.n_cells, game.n_cells)), rng.normal(size=game.n_cells)

    def one(position):
        position = np.asarray(position, dtype=np.float64)
        return np.sin(3.0 * (a @ position)), float(np.tanh(b @ position))

    def many(positions):
        results = [one(p) for p in positions.cpu().numpy()]
        return torch.tensor(np.array([r[0] for r in results])), torch.tensor(np.array([r[1] for r in results]))

    def for_reference(position):
        scores, value = one(position)
        scores = torch.tensor(scores).masked_fill(torch.tensor(position != 0), -torch.inf)
        return torch.softmax(scores[None], dim=1)[0].numpy(), value

    return many, for_reference


def midgame(game, n, plies, seed=0):
    rng = np.random.default_rng(seed)
    boards = game.empty(4 * n)
    for _ in range(plies):
        boards = game.play(boards, random_player(game, boards, rng))
    return boards[~game.finished(boards)][:n]


@pytest.mark.parametrize("make, plies, discount", [(tictactoe, 2, 1.0), (tictactoe, 4, 0.9), (cube3, 5, 0.95), (qubic, 12, 0.98)])
def test_batched_search_visits_exactly_what_the_reference_search_visits(make, plies, discount):
    game = make()
    many, one = stand_in_network(game)
    boards = np.concatenate([game.empty(1), midgame(game, 7, plies)])
    rng = np.random.default_rng(1)
    noise = rng.gamma(0.3, size=boards.shape) * (boards == 0)
    noise /= noise.sum(axis=1, keepdims=True)
    search = BatchedSearch(game, many, n_sims=70, c_puct=1.5, discount=discount, dtype=torch.float64)
    for root_noise in (None, noise):
        visits, values, root_value = search.run(boards, root_noise=root_noise)
        for i, board in enumerate(boards):
            ref_visits, ref_values = reference_search(
                game, board, one, 70, c_puct=1.5, discount=discount, root_noise=None if root_noise is None else root_noise[i]
            )
            assert (visits[i].numpy() == ref_visits).all()
            assert np.allclose(values[i].numpy(), ref_values, atol=1e-12)
        assert (visits.sum(dim=1) == 70).all()
        assert (visits.numpy()[boards != 0] == 0).all()
        assert np.allclose(root_value.numpy(), (visits * values).sum(dim=1).numpy() / 70)


def test_search_finds_a_win_and_a_block_without_any_network_knowledge():
    game = tictactoe()

    def blank(positions):  # a network that knows nothing: every move equally likely, every position even
        return torch.zeros(positions.shape, dtype=torch.float64), torch.zeros(len(positions), dtype=torch.float64)

    win = np.array([1, 1, 0, -1, -1, 0, 0, 0, 0], dtype=np.int8)  # X to move, 2 wins
    block = np.array([1, 1, 0, -1, 0, 0, 0, 0, 0], dtype=np.int8)  # O to move, must take 2
    visits, values, _ = BatchedSearch(game, blank, n_sims=400, dtype=torch.float64).run(np.stack([win, block]))
    assert visits.argmax(dim=1).tolist() == [2, 2]
    assert values[0, 2] == 1.0  # a finished game is scored exactly


def test_with_enough_simulations_the_search_plays_tictactoe_almost_perfectly_without_a_network():
    game = tictactoe()
    table = solve_table(game)

    def blank(positions):
        return torch.zeros(positions.shape), torch.zeros(len(positions))

    idx = np.flatnonzero(table.decision)[::8]
    visits, _, root_value = BatchedSearch(game, blank, n_sims=2000, c_puct=2.0).run(table.boards[idx])
    moves = visits.argmax(dim=1).numpy()
    assert table.best[idx, moves].mean() > 0.99
    assert np.abs(root_value.numpy() - table.value[idx]).mean() < 0.2


def test_masked_policy_skips_taken_cells():
    p = masked_policy(np.array([0.0, 5.0, 0.0]), np.array([0, 1, 0]))
    assert p.tolist() == [0.5, 0.0, 0.5]
