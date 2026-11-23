import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from qubic_zero import completing_cells, tictactoe  # noqa: E402
from qubic_zero.evaluate import standard_evaluators  # noqa: E402
from qubic_zero.train import Config, Replay, Trainer, main  # noqa: E402

SMALL = Config(
    game="3x3", width=16, depth=2, n_games=64, n_sims=12, discount=0.9, noise_alpha=1.0, sampled_moves=3,
    memory=5_000, batch_size=64, reuse=2.0, lr=3e-3, warmup=300, positions=64 * 60, seed=0,
    report_every=10, evaluate_every=30, snapshot_every=30, checkpoint_minutes=1e9,
)  # fmt: skip


def test_replay_keeps_the_most_recent_positions():
    replay = Replay(5, 2)
    replay.add(np.ones((3, 2), dtype=np.int8), np.full((3, 2), 0.5, dtype=np.float32), np.arange(3, dtype=np.float32))
    assert replay.filled == 3
    replay.add(-np.ones((4, 2), dtype=np.int8), np.full((4, 2), 0.5, dtype=np.float32), np.arange(10, 14, dtype=np.float32))
    assert replay.filled == 5 and sorted(replay.value.tolist()) == [2, 10, 11, 12, 13]
    boards, policy, value = replay.sample(50, np.random.default_rng(0))
    assert boards.shape == (50, 2) and set(value.tolist()) <= {2, 10, 11, 12, 13}
    other = Replay(5, 2)
    other.restore(replay.state())
    assert (other.value[:5] == replay.value).all() and other.cursor == replay.cursor and other.filled == 5


def test_what_self_play_stores_is_what_happened():
    game = tictactoe()
    trainer = Trainer(SMALL, quiet=True)
    for _ in range(25):
        trainer.round()
    replay = trainer.replay
    n = replay.filled
    boards, policy, value = replay.boards[:n], replay.policy[:n], replay.value[:n]
    assert n > 200 and trainer.games > 40 and trainer.stored == n
    # positions are seen by the player to move, who has as many stones as the other player or one fewer
    assert set(boards.sum(axis=1).tolist()) <= {0, -1}
    assert not game.finished(boards).any()
    # the policy target is a distribution over the empty cells
    assert np.allclose(policy.sum(axis=1), 1.0, atol=1e-5) and (policy[boards != 0] == 0).all()
    # the value target is 0 for a draw, or +-discount^(moves to the end) for a decided game
    powers = {round(sign * 0.9**t, 5) for t in range(1, 10) for sign in (1, -1)} | {0.0}
    assert {round(float(v), 5) for v in value} <= powers
    # "wins on the next move" must be true on the board, and so must "loses in two"
    wins_now = np.isclose(value, 0.9)
    assert wins_now.any()
    assert completing_cells(game, boards[wins_now], np.ones(wins_now.sum())).any(axis=1).all()
    loses_next = np.isclose(value, -0.81)
    assert completing_cells(game, boards[loses_next], -np.ones(loses_next.sum())).any(axis=1).all()


def test_a_short_run_learns_tictactoe_and_writes_its_log(tmp_path):
    config = SMALL.replace(positions=64 * 240, n_sims=16, evaluate_every=120)
    trainer = Trainer(config, tmp_path / "run", quiet=True)
    trainer.evaluators = standard_evaluators(trainer.game, config)
    history = trainer.run()
    scored = [h for h in history if "net_kept" in h]
    assert scored[0]["rounds"] == 0 and scored[-1]["rounds"] == 240
    assert scored[-1]["net_kept"] > scored[0]["net_kept"] + 0.15  # the network alone has learned a good deal
    assert scored[-1]["search_mistakes"] < scored[0]["search_mistakes"] / 3  # and so has the network with its search
    assert scored[-1]["net_value_error"] < scored[0]["net_value_error"]
    lines = [json.loads(line) for line in (tmp_path / "run" / "log.jsonl").read_text().splitlines()]
    assert lines == history and all("positions" in line for line in lines)
    assert {"config.json", "checkpoint.pt", "log.jsonl", "weights"} <= {p.name for p in (tmp_path / "run").iterdir()}
    assert (tmp_path / "run" / "weights" / "latest.npz").exists() and (tmp_path / "run" / "weights" / "round_0000030.npz").exists()
    assert history[-1]["lr"] == pytest.approx(config.lr * config.lr_final)


def test_a_run_that_is_stopped_and_continued_ends_like_one_that_was_not(tmp_path):
    straight = Trainer(SMALL, tmp_path / "straight", quiet=True)
    straight.run()
    first = Trainer(SMALL, tmp_path / "stopped", quiet=True)
    first.run(rounds=23)
    assert first.rounds == 23
    second = Trainer(SMALL, tmp_path / "stopped", quiet=True)  # finds the checkpoint and goes on from it
    assert second.rounds == 23 and second.replay.filled == first.replay.filled
    second.run()
    assert second.rounds == straight.rounds == 60 and second.games == straight.games and second.steps == straight.steps
    for a, b in zip(straight.net.state_dict().values(), second.net.state_dict().values()):
        assert torch.equal(a, b)
    assert [h["rounds"] for h in second.history] == [h["rounds"] for h in straight.history]
    with pytest.raises(ValueError):
        Trainer(SMALL.replace(width=32), tmp_path / "stopped", quiet=True)


def test_a_run_can_be_started_from_the_command_line(tmp_path, capsys):
    args = ["--run-dir", str(tmp_path / "cli"), "--game", "3x3", "--width", "8", "--depth", "1", "--n-games", "16", "--n-sims", "4",
            "--positions", "320", "--memory", "500", "--warmup", "50", "--batch-size", "16", "--evaluate-every", "20"]  # fmt: skip
    main(args)
    out = capsys.readouterr().out
    assert "stopped at round 20" in out and "320 positions" in out and "scored  network alone:" in out
    assert json.loads((tmp_path / "cli" / "config.json").read_text())["n_sims"] == 4
    main(["--run-dir", str(tmp_path / "cli"), "--positions", "480"])  # the same run, given a longer budget
    out = capsys.readouterr().out
    assert "continuing" in out and "stopped at round 30" in out
