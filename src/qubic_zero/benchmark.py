"""How fast is self-play on this machine, and how long would a run take?

Training time is spent in two places: the tree searches of self-play (one call
to the network per simulation, for all games at once) and the gradient steps.
:func:`measure` times both for a given configuration, with a fresh network,
and :func:`hours_for` turns the two rates into the length of a run.

The estimate is rough on purpose.  Games get longer or shorter as the network
learns, and a trained network grows deeper trees than an untrained one.  It is
good for telling one hour from ten, not for more.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from .train import PRESETS, Config, Trainer, policy_value_loss

__all__ = ["measure", "hours_for", "main"]


def _wait(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _name(device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "cpu":
        return f"CPU, {torch.get_num_threads()} threads"
    return str(device)


def measure(config: Config, seconds: float = 60.0, device=None) -> dict:
    """Time self-play and learning for ``config``; about ``seconds`` in all.

    Returns the device, positions searched per second in self-play, calls of the
    network per second (one per position and simulation), and gradient steps per
    second at the configured batch size.
    """
    trainer = Trainer(config, device=device, quiet=True)
    dev = trainer.device
    rows = {"n": 0}
    net = trainer.net

    def counting(positions):
        rows["n"] += len(positions)
        return net(positions)

    trainer.search.evaluate = counting
    trainer.selfplay.step()  # the first call pays for start-up costs
    _wait(dev)
    rows["n"] = 0
    rounds, start = 0, time.perf_counter()
    while time.perf_counter() - start < 0.7 * seconds:
        trainer.selfplay.step()
        rounds += 1
    _wait(dev)
    play = time.perf_counter() - start

    rng = np.random.default_rng(0)
    cells = trainer.game.n_cells
    boards = torch.from_numpy(rng.integers(-1, 2, size=(config.batch_size, cells)).astype(np.int8)).to(dev)
    policy = torch.softmax(torch.randn(config.batch_size, cells, device=dev), dim=1)
    value = torch.zeros(config.batch_size, device=dev)
    net.train()
    steps, start = 0, time.perf_counter()
    while time.perf_counter() - start < 0.3 * seconds or steps < 3:
        policy_loss, value_loss = policy_value_loss(net, boards, policy, value)
        trainer.optimiser.zero_grad(set_to_none=True)
        (policy_loss + value_loss).backward()
        trainer.optimiser.step()
        steps += 1
    _wait(dev)
    learn = time.perf_counter() - start
    return {
        "device": _name(dev),
        "positions_per_second": rounds * config.n_games / play,
        "evaluations_per_second": rows["n"] / play,
        "steps_per_second": steps / learn,
    }


def hours_for(config: Config, speed: dict, positions: int | None = None) -> float:
    """Hours that a run of ``positions`` searched positions (default: the configured length) would take at ``speed``."""
    positions = config.positions if positions is None else positions
    steps = config.reuse * positions / config.batch_size
    return (positions / speed["positions_per_second"] + steps / speed["steps_per_second"]) / 3600.0


def main(argv=None) -> None:
    """``python -m qubic_zero.benchmark``: time a configuration on this machine and print how long runs would take."""
    parser = argparse.ArgumentParser(description="Time self-play and learning on this machine.")
    parser.add_argument("--preset", default="4x4x4", choices=sorted(PRESETS))
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--device", default=None, help="cuda, cpu or mps (default: cuda if there is one, otherwise cpu)")
    for name in ("n_games", "n_sims", "width", "depth", "batch_size"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=int, default=None)
    args = vars(parser.parse_args(argv))
    preset, seconds, device = args.pop("preset"), args.pop("seconds"), args.pop("device")
    config = PRESETS[preset].replace(**{name: value for name, value in args.items() if value is not None})
    speed = measure(config, seconds, device)
    print(
        f"{speed['device']}: {speed['positions_per_second']:,.0f} positions searched per second, "
        f"{speed['evaluations_per_second']:,.0f} positions through the network per second, "
        f"{speed['steps_per_second']:.1f} gradient steps per second"
    )
    for positions in (1_000_000, 5_000_000, 20_000_000):
        print(f"  {positions:>12,} positions: about {hours_for(config, speed, positions):.1f} hours")


if __name__ == "__main__":
    main()
