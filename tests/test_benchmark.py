import pytest

torch = pytest.importorskip("torch")

from qubic_zero.benchmark import hours_for, main, measure  # noqa: E402
from qubic_zero.train import Config  # noqa: E402


def test_speed_is_measured_and_turned_into_hours():
    config = Config(game="3x3", width=8, depth=1, n_games=16, n_sims=4, batch_size=16, positions=10_000)
    speed = measure(config, seconds=1.0, device="cpu")
    assert speed["device"].startswith("CPU")
    assert speed["positions_per_second"] > 0 and speed["steps_per_second"] > 0
    # every searched position costs about one call of the network per simulation
    assert 2 < speed["evaluations_per_second"] / speed["positions_per_second"] <= config.n_sims + 1
    assert hours_for(config, speed, 20_000) == pytest.approx(2 * hours_for(config, speed))
    slower = {**speed, "positions_per_second": speed["positions_per_second"] / 2}
    assert hours_for(config, slower) > hours_for(config, speed)


def test_speed_test_from_the_command_line(capsys):
    main(["--preset", "3x3", "--seconds", "1", "--n-games", "16", "--n-sims", "4"])
    out = capsys.readouterr().out
    assert "positions searched per second" in out and "20,000,000 positions: about" in out
