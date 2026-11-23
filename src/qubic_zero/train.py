"""Learning by self-play.

The loop is AlphaZero's.  Many games are played side by side, every move
chosen by a tree search that the current network guides.  When a game ends,
each of its positions is stored with two things the network should have said
there: how the search spread its visits over the moves (the policy target)
and how the game went on to end for the player to move (the value target).
The network is trained on random samples of the stored positions while the
games go on, and the improved network at once guides the next searches.

Nothing but the rules enters: no opening book, no games of other players, no
hand-made features, and no exact search.

A run is a folder.  It holds the settings, a log with one line of JSON per
report, snapshots of the weights, and a checkpoint from which the run can go
on exactly where it stopped.  From a terminal::

    python -m qubic_zero.train --game 4x4x4 --run-dir runs/qubic --hours 11

starts a run, or continues it if the folder already holds one.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import signal
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch

from .games import LineGame, game_by_name
from .net import count_weights, make_net, save_weights
from .search import BatchedSearch

__all__ = ["Config", "SelfPlay", "Replay", "Trainer", "policy_value_loss", "main"]


@dataclass
class Config:
    """Every setting of a run.  The defaults are sized for Qubic on a GPU."""

    game: str = "4x4x4"
    # the network
    net: str = "line"  # "line" or "conv"
    width: int = 64  # line network: size of the vector kept per cell and per line
    depth: int = 4  # line network: number of exchanges between cells and lines
    channels: int = 32  # convolutional network
    blocks: int = 4
    # self-play
    n_games: int = 1024  # games played side by side
    n_sims: int = 100  # simulations of the tree search per move
    c_puct: float = 1.5
    discount: float = 0.98  # a result is worth this much less for every move it is away
    noise_alpha: float = 0.3  # shape of the Dirichlet noise mixed into the root's policy
    noise_weight: float = 0.25
    sampled_moves: int = 8  # this many moves of a game are drawn from the visit counts; then the top move is played
    # learning
    memory: int = 500_000  # positions remembered
    batch_size: int = 512
    reuse: float = 4.0  # how often a position is used for a gradient step, on average
    lr: float = 2e-3
    lr_final: float = 0.1  # the learning rate falls linearly to this fraction of ``lr``
    weight_decay: float = 1e-4
    warmup: int = 5_000  # positions to collect before learning starts
    positions: int = 5_000_000  # length of the run: positions searched in self-play
    # bookkeeping
    seed: int = 0
    report_every: int = 20  # rounds (one move in every game) between lines in the log
    evaluate_every: int = 200  # rounds between evaluations
    snapshot_every: int = 1_000  # rounds between numbered copies of the weights
    checkpoint_minutes: float = 10.0

    def replace(self, **changes) -> Config:
        return Config(**{**asdict(self), **changes})


def policy_value_loss(net, boards, policy, value):
    """AlphaZero's loss: cross-entropy to the search's visit shares plus squared error to the game's result."""
    scores, predicted = net(boards)
    scores = scores.masked_fill(boards != 0, -1e9)
    policy_loss = -(policy * torch.log_softmax(scores, dim=1)).sum(dim=1).mean()
    value_loss = torch.mean((predicted - value) ** 2)
    return policy_loss, value_loss


class Replay:
    """The last ``size`` positions of self-play, with their targets."""

    def __init__(self, size: int, n_cells: int) -> None:
        self.boards = np.zeros((size, n_cells), dtype=np.int8)
        self.policy = np.zeros((size, n_cells), dtype=np.float32)
        self.value = np.zeros(size, dtype=np.float32)
        self.size, self.filled, self.cursor = size, 0, 0

    def add(self, boards, policy, value) -> None:
        n = len(value)
        if n > self.size:
            boards, policy, value, n = boards[-self.size :], policy[-self.size :], value[-self.size :], self.size
        idx = (self.cursor + np.arange(n)) % self.size
        self.boards[idx], self.policy[idx], self.value[idx] = boards, policy, value
        self.cursor = (self.cursor + n) % self.size
        self.filled = min(self.filled + n, self.size)

    def sample(self, n: int, rng: np.random.Generator):
        idx = rng.integers(0, self.filled, size=n)
        return self.boards[idx], self.policy[idx], self.value[idx]

    def state(self) -> dict:
        keep = slice(0, self.filled)
        return {
            "boards": self.boards[keep],
            "policy": self.policy[keep],
            "value": self.value[keep],
            "cursor": np.array(self.cursor),
        }

    def restore(self, state: dict) -> None:
        n = len(state["value"])
        self.boards[:n], self.policy[:n], self.value[:n] = state["boards"], state["policy"], state["value"]
        self.filled, self.cursor = n, int(state["cursor"])


class SelfPlay:
    """``n_games`` games in progress, all moved on together by :meth:`step`."""

    def __init__(self, game: LineGame, search: BatchedSearch, config: Config, rng: np.random.Generator) -> None:
        self.game, self.search, self.config, self.rng = game, search, config, rng
        dev, n, cells = search.device, config.n_games, game.n_cells
        self.boards = torch.zeros((n, cells), dtype=torch.int8, device=dev)
        self.plies = torch.zeros(n, dtype=torch.long, device=dev)
        self.seen = torch.zeros((n, cells, cells), dtype=torch.int8, device=dev)  # [game, move number]: the position
        self.shares = torch.zeros((n, cells, cells), dtype=torch.float32, device=dev)  # and the search's visit shares
        self.rows = torch.arange(n, device=dev)
        self.incidence = torch.from_numpy(game.incidence).to(dev)

    def step(self) -> dict:
        """One move in every game.  Returns the positions of the games that ended, with their targets."""
        cfg, game, dev = self.config, self.game, self.search.device
        boards, rows = self.boards, self.rows
        mover = (1 - 2 * boards.sum(dim=1, dtype=torch.int32)).to(torch.int8)

        noise = None
        if cfg.noise_weight > 0:
            empty = (boards == 0).cpu().numpy()
            noise = self.rng.gamma(cfg.noise_alpha, size=empty.shape) * empty
            noise = torch.from_numpy(noise / noise.sum(axis=1, keepdims=True)).to(dev)
        visits, _, root_value = self.search.run(boards, root_noise=noise, noise_weight=cfg.noise_weight)
        shares = (visits / visits.sum(dim=1, keepdim=True)).float()

        self.seen[rows, self.plies] = boards * mover[:, None]
        self.shares[rows, self.plies] = shares

        # the first moves of a game are drawn from the visit shares, the later ones are the most visited
        cumulative = shares.double().cumsum(dim=1).cpu().numpy()
        draw = self.rng.random((len(rows), 1)) * cumulative[:, -1:]
        sampled = torch.from_numpy((cumulative > draw).argmax(axis=1)).to(dev)
        moves = torch.where(self.plies < cfg.sampled_moves, sampled, visits.argmax(dim=1))
        boards[rows, moves] = mover
        self.plies += 1

        sums = boards.float() @ self.incidence.T
        won = (sums * mover[:, None].float()).max(dim=1).values == game.k
        done = won | (boards != 0).all(dim=1)
        out = {"games": 0, "root_value": float(root_value.mean())}
        if bool(done.any()):
            d = done.nonzero(as_tuple=True)[0]
            length = self.plies[d]  # moves played in each finished game
            ply = torch.arange(game.n_cells, device=dev)[None, :]
            recorded = ply < length[:, None]
            # the player who made the last move won (or nobody did); positions an even number of moves
            # before the last one were that player's, and a result t moves away is discounted t times
            away = length[:, None] - ply
            sign = 1.0 - 2.0 * ((away - 1) % 2).float()
            target = won[d, None].float() * sign * cfg.discount ** away.clamp(min=0).float()
            out.update(
                boards=self.seen[d][recorded].cpu().numpy(),
                policy=self.shares[d][recorded].cpu().numpy(),
                value=target[recorded].cpu().numpy().astype(np.float32),
                games=int(d.numel()),
                first_wins=int((won[d] & (mover[d] == 1)).sum()),
                second_wins=int((won[d] & (mover[d] == -1)).sum()),
                moves=int(length.sum()),
            )
            boards[d] = 0
            self.plies[d] = 0
        return out

    def state(self) -> dict:
        return {name: getattr(self, name).cpu().numpy() for name in ("boards", "plies", "seen", "shares")}

    def restore(self, state: dict) -> None:
        for name, value in state.items():
            getattr(self, name).copy_(torch.from_numpy(value).to(self.search.device))


class Trainer:
    """A run: network, games in progress, memory, optimiser, log and checkpoints.

    ``evaluators`` are functions ``f(trainer) -> dict`` that are called every
    ``config.evaluate_every`` rounds; what they return is written to the log.
    """

    def __init__(self, config: Config, run_dir=None, device=None, evaluators=(), quiet: bool = False) -> None:
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.game = game_by_name(config.game)
        sizes = (
            {"width": config.width, "depth": config.depth}
            if config.net == "line"
            else {"channels": config.channels, "blocks": config.blocks}
        )
        self.net = make_net(self.game, config.net, seed=config.seed, **sizes).to(self.device)
        self.optimiser = torch.optim.AdamW(self.net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.rng = np.random.default_rng(config.seed)
        self.search = BatchedSearch(
            self.game, self._evaluate, config.n_sims, config.c_puct, config.discount, device=self.device
        )
        self.selfplay = SelfPlay(self.game, self.search, config, self.rng)
        self.replay = Replay(config.memory, self.game.n_cells)
        self.evaluators = list(evaluators)
        self.quiet = quiet
        self.rounds = self.steps = self.games = self.stored = 0
        self.seconds = 0.0
        self.debt = 0.0  # gradient steps owed to positions already stored
        self.window = _Window()
        self.history: list[dict] = []
        self.run_dir = None if run_dir is None else Path(run_dir)
        if self.run_dir is not None:
            (self.run_dir / "weights").mkdir(parents=True, exist_ok=True)
            if (self.run_dir / "checkpoint.pt").exists():
                self.load()
            (self.run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))

    # --- the network in the search -------------------------------------------------
    def _evaluate(self, positions):
        self.net.eval()
        return self.net(positions)

    @property
    def positions(self) -> int:
        """Positions searched in self-play so far."""
        return self.rounds * self.config.n_games

    def learning_rate(self) -> float:
        cfg = self.config
        progress = min(1.0, self.positions / cfg.positions)
        return cfg.lr * (1.0 + (cfg.lr_final - 1.0) * progress)

    # --- one round --------------------------------------------------------------------
    def round(self) -> None:
        """One move in every game, then the gradient steps that the new positions pay for."""
        cfg = self.config
        out = self.selfplay.step()
        self.rounds += 1
        self.window.add("root_value", out["root_value"])
        if out["games"]:
            self.replay.add(out["boards"], out["policy"], out["value"])
            self.games += out["games"]
            self.stored += len(out["value"])
            for key in ("games", "first_wins", "second_wins", "moves"):
                self.window.count(key, out[key])
            if self.replay.filled >= cfg.warmup:
                self.debt += cfg.reuse * len(out["value"]) / cfg.batch_size
        steps = int(self.debt)
        self.debt -= steps
        if steps:
            self.learn(steps)

    def learn(self, steps: int) -> None:
        cfg, dev = self.config, self.device
        self.net.train()
        for group in self.optimiser.param_groups:
            group["lr"] = self.learning_rate()
        symmetries = self.game.symmetries if cfg.net != "line" else None
        for _ in range(steps):
            boards, policy, value = self.replay.sample(cfg.batch_size, self.rng)
            if symmetries is not None:  # a line network treats symmetric positions alike by construction
                perm = symmetries[self.rng.integers(0, len(symmetries), size=len(value))]
                boards, policy = np.take_along_axis(boards, perm, axis=1), np.take_along_axis(policy, perm, axis=1)
            boards = torch.from_numpy(boards).to(dev)
            policy_loss, value_loss = policy_value_loss(
                self.net, boards, torch.from_numpy(policy).to(dev), torch.from_numpy(value).to(dev)
            )
            self.optimiser.zero_grad(set_to_none=True)
            (policy_loss + value_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
            self.optimiser.step()
            self.steps += 1
            self.window.add("policy_loss", policy_loss.item())
            self.window.add("value_loss", value_loss.item())

    # --- running ------------------------------------------------------------------------
    def run(self, hours: float | None = None, rounds: int | None = None) -> list[dict]:
        """Go on until the run's budget of positions is used up, or for ``hours``, or for ``rounds`` more rounds.

        The run is checkpointed every ``config.checkpoint_minutes`` and when this
        function returns, also if it returns because of Ctrl-C or a termination signal.
        """
        cfg = self.config
        started = last_save = time.perf_counter()
        stop_round = None if rounds is None else self.rounds + rounds
        stop = {"now": False}
        previous = _catch_signals(stop)
        try:
            if self.rounds == 0 and self.evaluators:
                self.report(evaluate=True)
            while self.positions < cfg.positions and not stop["now"]:
                if stop_round is not None and self.rounds >= stop_round:
                    break
                if hours is not None and time.perf_counter() - started > 3600 * hours:
                    break
                tick = time.perf_counter()
                self.round()
                self.seconds += time.perf_counter() - tick
                evaluate = bool(self.evaluators) and self.rounds % cfg.evaluate_every == 0
                if evaluate or self.rounds % cfg.report_every == 0:
                    self.report(evaluate=evaluate)
                if self.run_dir is not None:
                    if self.rounds % cfg.snapshot_every == 0:
                        save_weights(self.net, self.run_dir / "weights" / f"round_{self.rounds:07d}.npz")
                    if time.perf_counter() - last_save > 60 * cfg.checkpoint_minutes:
                        self.save()
                        last_save = time.perf_counter()
        finally:
            _restore_signals(previous)
            if self.positions >= cfg.positions and (not self.history or self.history[-1]["rounds"] != self.rounds):
                self.report(evaluate=bool(self.evaluators))
            if self.run_dir is not None:
                self.save()
        return self.history

    def report(self, evaluate: bool = False) -> dict:
        record = {
            "rounds": self.rounds,
            "positions": self.positions,
            "games": self.games,
            "stored": self.stored,
            "steps": self.steps,
            "seconds": round(self.seconds, 2),
            "lr": self.learning_rate(),
            **self.window.flush(),
        }
        if evaluate:
            self.net.eval()
            for evaluator in self.evaluators:
                record.update(evaluator(self))
        self.history.append(record)
        if self.run_dir is not None:
            with open(self.run_dir / "log.jsonl", "a") as log:
                log.write(json.dumps(record) + "\n")
        if not self.quiet:
            shown = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in record.items() if not isinstance(v, (list, dict))}
            print(" ".join(f"{k}={v}" for k, v in shown.items()), flush=True)
        return record

    # --- stopping and going on ------------------------------------------------------------
    def save(self) -> Path:
        """Write the weights and everything needed to continue this run, safely (never half a file)."""
        state = {
            "config": asdict(self.config),
            "net": self.net.state_dict(),
            "optimiser": self.optimiser.state_dict(),
            "counters": {k: getattr(self, k) for k in ("rounds", "steps", "games", "stored", "seconds", "debt")},
            "rng": self.rng.bit_generator.state,
            "replay": _pack(self.replay.state()),
            "selfplay": _pack(self.selfplay.state()),
            "window": self.window.state(),
            "history": self.history,
        }
        path = self.run_dir / "checkpoint.pt"
        temporary = path.with_suffix(".tmp")
        torch.save(state, temporary)
        os.replace(temporary, path)
        weights = self.run_dir / "weights" / "latest.npz"
        save_weights(self.net, weights.with_suffix(".tmp.npz"))
        os.replace(weights.with_suffix(".tmp.npz"), weights)
        return path

    def load(self) -> None:
        state = torch.load(self.run_dir / "checkpoint.pt", map_location=self.device, weights_only=False)
        stored = {f.name: state["config"][f.name] for f in fields(Config) if f.name in state["config"]}
        fixed = ("game", "net", "width", "depth", "channels", "blocks", "n_games", "memory", "seed")
        changed = [name for name in fixed if stored.get(name) != getattr(self.config, name)]
        if changed:
            raise ValueError(f"{self.run_dir} holds a run with different {', '.join(changed)}; use another folder")
        self.net.load_state_dict(state["net"])
        self.optimiser.load_state_dict(state["optimiser"])
        for name, value in state["counters"].items():
            setattr(self, name, value)
        self.rng.bit_generator.state = state["rng"]
        self.replay.restore(_unpack(state["replay"]))
        self.selfplay.restore(_unpack(state["selfplay"]))
        self.window.restore(state["window"])
        self.history = state["history"]
        if not self.quiet:
            print(f"continuing {self.run_dir} from round {self.rounds:,} ({self.positions:,} positions, {self.games:,} games)")


def _pack(arrays: dict) -> bytes:
    """Arrays as compressed bytes: positions and visit shares are mostly zeros and shrink many times over."""
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def _unpack(blob: bytes) -> dict:
    with np.load(io.BytesIO(blob)) as stored:
        return {name: stored[name] for name in stored.files}


class _Window:
    """Running sums between two reports."""

    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.totals: dict[str, int] = {}

    def add(self, key: str, value: float) -> None:
        self.sums[key] = self.sums.get(key, 0.0) + value
        self.counts[key] = self.counts.get(key, 0) + 1

    def count(self, key: str, value: int) -> None:
        self.totals[key] = self.totals.get(key, 0) + value

    def flush(self) -> dict:
        out = {key: self.sums[key] / self.counts[key] for key in self.sums}
        games = self.totals.get("games", 0)
        if games:
            out["first_player_share"] = self.totals["first_wins"] / games
            out["draw_share"] = 1.0 - (self.totals["first_wins"] + self.totals["second_wins"]) / games
            out["moves_per_game"] = self.totals["moves"] / games
        self.sums, self.counts, self.totals = {}, {}, {}
        return out

    def state(self) -> dict:
        return {"sums": self.sums, "counts": self.counts, "totals": self.totals}

    def restore(self, state: dict) -> None:
        self.sums, self.counts, self.totals = state["sums"], state["counts"], state["totals"]


def _catch_signals(stop: dict):
    def handler(signum, frame):
        stop["now"] = True

    previous = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous[sig] = signal.signal(sig, handler)
        except ValueError:  # not in the main thread (a notebook, for instance): Ctrl-C then arrives as an exception
            pass
    return previous


def _restore_signals(previous) -> None:
    for sig, handler in previous.items():
        signal.signal(sig, handler)


def main(argv=None) -> None:
    """``python -m qubic_zero.train``: start a run, or continue the one in ``--run-dir``."""
    from .evaluate import standard_evaluators

    parser = argparse.ArgumentParser(description="Train a player by self-play (see the module's description).")
    parser.add_argument("--run-dir", required=True, help="folder of the run; continued if it already holds one")
    parser.add_argument("--hours", type=float, default=None, help="stop (and checkpoint) after this many hours")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda if there is one)")
    for f in fields(Config):
        parser.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=None, help=f"default {f.default}")
    args = vars(parser.parse_args(argv))
    run_dir, hours, device = Path(args.pop("run_dir")), args.pop("hours"), args.pop("device")
    given = {name: value for name, value in args.items() if value is not None}
    stored = run_dir / "config.json"
    config = Config(**json.loads(stored.read_text())) if stored.exists() else Config()
    config = config.replace(**given)
    trainer = Trainer(config, run_dir, device=device)
    trainer.evaluators = standard_evaluators(trainer.game, config)
    print(f"{config.game}: {count_weights(trainer.net):,} weights on {trainer.device}; {config.positions:,} positions planned")
    trainer.run(hours=hours)
    print(f"stopped at round {trainer.rounds:,}: {trainer.positions:,} positions, {trainer.games:,} games, {trainer.seconds / 3600:.2f} h")


if __name__ == "__main__":
    main()
