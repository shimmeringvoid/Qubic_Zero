import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from qubic_zero import cube3, random_player, tictactoe  # noqa: E402
from qubic_zero.agents import SearchPlayer  # noqa: E402
from qubic_zero.net import make_net, save_weights  # noqa: E402
from qubic_zero.play import NumpyPlayer, main  # noqa: E402


@pytest.fixture(scope="module")
def stored(tmp_path_factory):
    folder = tmp_path_factory.mktemp("weights")
    return {game.name: save_weights(make_net(game, "line", seed=2, width=16, depth=2), folder / f"{game.name}.npz") for game in (tictactoe(), cube3())}


def test_numpy_player_agrees_with_the_pytorch_player(stored):
    game = cube3()
    rng = np.random.default_rng(0)
    boards = game.empty(12)
    for _ in range(4):
        boards = game.play(boards, random_player(game, boards, rng))
    boards = boards[~game.finished(boards)]
    plain = NumpyPlayer(stored["3x3x3"], n_sims=40, discount=0.95, threats=False)
    from qubic_zero.net import load_weights

    reference = SearchPlayer(load_weights(stored["3x3x3"]), 40, discount=0.95)
    moves = plain(game, boards)
    assert (boards[np.arange(len(boards)), moves] == 0).all()
    assert (moves == reference(game, boards)).mean() >= 0.8  # the same search; rounding may break a tie differently
    assert "visits" in plain.said


def test_numpy_player_takes_a_forced_win_when_the_threat_search_finds_one(stored):
    game = tictactoe()
    player = NumpyPlayer(stored["3x3"], n_sims=2)
    assert player.move(np.array([1, 1, 0, -1, -1, 0, 0, 0, 0])) == 2 and player.said == "a win by force"
    assert player.move(game.empty()) in range(9)


def test_playing_needs_no_pytorch(stored):
    code = (
        "import sys; sys.modules['torch'] = None\n"  # makes any 'import torch' fail
        "import numpy as np\n"
        "from qubic_zero.play import NumpyPlayer\n"
        f"player = NumpyPlayer({str(stored['3x3x3'])!r}, n_sims=20)\n"
        "print(player.move(np.zeros(27, dtype=np.int8)))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert int(result.stdout) in range(27)


def test_a_game_in_the_terminal(stored, monkeypatch, capsys):
    answers = iter(["9 9", "1 1", "1 1", "1 2", "1 3", "2 1", "2 2", "2 3", "3 1", "3 2", "3 3"])

    def answer(prompt):
        return next(answers)

    monkeypatch.setattr("builtins.input", answer)
    main([str(stored["3x3"]), "--sims", "10"])
    out = capsys.readouterr().out
    assert "that is not a cell" in out and "the network plays" in out
    assert out.strip().splitlines()[-1] in ("a draw", "you win", "the network wins")
