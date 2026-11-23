import numpy as np
import pytest

torch = pytest.importorskip("torch")

from qubic_zero import cube3, qubic, tictactoe  # noqa: E402
from qubic_zero.net import NumpyLineNet, count_weights, load_weights, make_net, save_weights  # noqa: E402


def some_positions(game, n=6, seed=0):
    return np.random.default_rng(seed).integers(-1, 2, size=(n, game.n_cells)).astype(np.int8)


@pytest.mark.parametrize("make", [tictactoe, cube3, qubic])
def test_line_network_treats_every_symmetric_position_alike(make):
    """Relabel the cells by any symmetry of the game, and the output is relabelled the same way."""
    game = make()
    net = make_net(game, "line", seed=1, width=16, depth=3).double().eval()
    boards = torch.from_numpy(some_positions(game))
    with torch.no_grad():
        scores, values = net(boards)
        assert scores.shape == (6, game.n_cells) and values.shape == (6,)
        assert (values.abs() <= 1).all()
        for perm in torch.from_numpy(game.symmetries):
            moved_scores, moved_values = net(boards[:, perm])
            assert torch.allclose(moved_scores, scores[:, perm], atol=1e-9)
            assert torch.allclose(moved_values, values, atol=1e-9)


def test_convolutional_network_does_not_know_the_relabelling_symmetries():
    game = qubic()
    net = make_net(game, "conv", seed=1, channels=8, blocks=1).eval()
    boards = torch.from_numpy(some_positions(game))
    relabel = np.array([0, 2, 1, 3])  # swap the two inner layers along every axis
    cells = np.stack(np.unravel_index(np.arange(64), (4, 4, 4)), axis=1)
    perm = torch.from_numpy(np.ravel_multi_index(relabel[cells].T, (4, 4, 4)))
    assert tuple(perm.tolist()) in {tuple(p) for p in game.symmetries.tolist()}
    with torch.no_grad():
        scores, values = net(boards)
        moved_scores, moved_values = net(boards[:, perm])
    assert scores.shape == (6, 64) and values.shape == (6,)
    assert not torch.allclose(moved_values, values, atol=1e-4)
    square = make_net(tictactoe(), "conv", seed=0, channels=8, blocks=1)
    assert square(torch.from_numpy(some_positions(tictactoe())))[0].shape == (6, 9)


def test_the_seed_alone_decides_the_initial_weights():
    game = tictactoe()
    a = make_net(game, "line", seed=3, width=16, depth=2)
    torch.rand(1000)  # whatever happens in between must not matter
    b = make_net(game, "line", seed=3, width=16, depth=2)
    c = make_net(game, "line", seed=4, width=16, depth=2)
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    assert not all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), c.state_dict().values()))
    assert count_weights(a) == sum(p.numel() for p in a.parameters()) > 1000
    with pytest.raises(ValueError):
        make_net(game, "transformer")


@pytest.mark.parametrize("kind, sizes", [("line", {"width": 16, "depth": 2}), ("conv", {"channels": 8, "blocks": 1})])
def test_weights_survive_being_saved_and_loaded(tmp_path, kind, sizes):
    game = qubic()
    net = make_net(game, kind, seed=5, **sizes).eval()
    path = save_weights(net, tmp_path / "weights.npz")
    again = load_weights(path)
    boards = torch.from_numpy(some_positions(game))
    with torch.no_grad():
        for x, y in zip(net(boards), again(boards)):
            assert torch.allclose(x, y, atol=1e-6)
    assert again.config() == net.config()


def test_a_line_network_runs_without_pytorch(tmp_path):
    for game in (tictactoe(), qubic()):
        net = make_net(game, "line", seed=6, width=24, depth=3).eval()
        plain = NumpyLineNet(save_weights(net, tmp_path / f"{game.name}.npz"))
        boards = some_positions(game, n=9)
        with torch.no_grad():
            scores, values = net(torch.from_numpy(boards))
        plain_scores, plain_values = plain(boards)
        assert np.allclose(plain_scores, scores.numpy(), atol=1e-4)
        assert np.allclose(plain_values, values.numpy(), atol=1e-5)
        assert plain(boards[0])[0].shape == (1, game.n_cells)
    conv = save_weights(make_net(qubic(), "conv", channels=8, blocks=1), tmp_path / "conv.npz")
    with pytest.raises(ValueError):
        NumpyLineNet(conv)
