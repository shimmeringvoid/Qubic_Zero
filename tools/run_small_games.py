"""Train the players of the small games whose logs and weights are stored in results/.

    python tools/run_small_games.py            # everything, two runs at a time (about 40 minutes on two cores)
    python tools/run_small_games.py 3x3        # only the runs whose name starts with "3x3"

Each run is one seed of one setting.  A run that is already complete is left
alone, so the script can be stopped and started again.  Every process uses one
thread, which keeps the numbers the same however many runs share the machine.
"""

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
SEEDS = range(5)
#: name of the setting -> (preset, changes to it)
SETTINGS = {
    "3x3-discount-0.9": ("3x3", {}),
    "3x3-discount-1": ("3x3", {"discount": 1.0}),
    "3x3x3-discount-0.95": ("3x3x3", {}),
    "3x3x3-discount-1": ("3x3x3", {"discount": 1.0}),
}


def run(job):
    name, seed = job
    import torch

    from qubic_zero.evaluate import standard_evaluators
    from qubic_zero.train import PRESETS, Trainer

    torch.set_num_threads(1)
    preset, changes = SETTINGS[name]
    config = PRESETS[preset].replace(seed=seed, **changes)
    folder = RESULTS / name / f"seed-{seed}"
    log = folder / "log.jsonl"
    if log.exists() and not (folder / "checkpoint.pt").exists():
        if json.loads(log.read_text().splitlines()[-1])["positions"] >= config.positions:
            return f"{name} seed {seed}: already complete"
        log.unlink()  # stopped without a checkpoint: nothing to continue from, so start again
    trainer = Trainer(config, folder, device="cpu", quiet=True)
    trainer.evaluators = standard_evaluators(trainer.game, config)
    started = time.perf_counter()
    trainer.run()
    (folder / "checkpoint.pt").unlink()  # only needed to continue a run; the weights are in weights/latest.npz
    return f"{name} seed {seed}: {trainer.games:,} games in {time.perf_counter() - started:.0f} s"


if __name__ == "__main__":
    wanted = sys.argv[1] if len(sys.argv) > 1 else ""
    jobs = [(name, seed) for name in SETTINGS if name.startswith(wanted) for seed in SEEDS]
    with Pool(2) as pool:
        for line in pool.imap_unordered(run, jobs):
            print(line, flush=True)
