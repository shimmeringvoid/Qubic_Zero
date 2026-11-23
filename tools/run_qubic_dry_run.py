"""Short self-play runs on Qubic, on a CPU, with both networks: does the pipeline start to learn the real game?

    python tools/run_qubic_dry_run.py         # four runs, two at a time; about 25 minutes on two cores

These are dry runs: a few thousand games each, far too few to play Qubic well.
They show that every part works on 64 cells and what the first stretch of
learning looks like, and they are what the compute estimate starts from.  The
logs and weights go to results/4x4x4-dry-line and results/4x4x4-dry-conv.
"""

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
SEEDS = range(2)
SETTINGS = {
    "4x4x4-dry-line": {"net": "line", "width": 48, "depth": 4},
    "4x4x4-dry-conv": {"net": "conv", "channels": 18, "blocks": 3},
}
COMMON = dict(
    n_games=128, n_sims=24, discount=0.98, noise_alpha=0.3, sampled_moves=8, memory=100_000, batch_size=256,
    warmup=3_000, positions=128 * 800, report_every=25, evaluate_every=100, snapshot_every=10**9,
)  # fmt: skip


def run(job):
    name, seed = job
    import torch

    from qubic_zero.evaluate import standard_evaluators
    from qubic_zero.net import count_weights
    from qubic_zero.train import Config, Trainer

    torch.set_num_threads(1)
    config = Config(game="4x4x4", seed=seed, **COMMON, **SETTINGS[name])
    folder = RESULTS / name / f"seed-{seed}"
    log = folder / "log.jsonl"
    if log.exists() and not (folder / "checkpoint.pt").exists():
        if json.loads(log.read_text().splitlines()[-1])["positions"] >= config.positions:
            return f"{name} seed {seed}: already complete"
        log.unlink()
    trainer = Trainer(config, folder, device="cpu", quiet=True)
    trainer.evaluators = standard_evaluators(trainer.game, config, n_games=100, per_kind=200)
    started = time.perf_counter()
    trainer.run()
    (folder / "checkpoint.pt").unlink()
    return f"{name} seed {seed}: {count_weights(trainer.net):,} weights, {trainer.games:,} games in {time.perf_counter() - started:.0f} s"


if __name__ == "__main__":
    wanted = sys.argv[1] if len(sys.argv) > 1 else ""
    jobs = [(name, seed) for seed in SEEDS for name in SETTINGS if name.startswith(wanted)]
    with Pool(2) as pool:
        for line in pool.imap_unordered(run, jobs):
            print(line, flush=True)
