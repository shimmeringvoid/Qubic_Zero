"""Which network picks up Qubic's tactics faster: the line network or a convolutional one?

    python tools/compare_networks.py          # about 10 minutes on two CPU cores; writes results/network-comparison.json

Both are trained on the same Qubic positions with exact labels from the threat
search, and scored on positions they have not seen (nor any symmetric image of
them).  The value head learns to say whether the player to move can force a
win; the policy head learns to point at a move that is proven right: a first
move of a forced win or, where the opponent threatens, the block.  This is not
how the players are trained (that is self-play); it is a controlled test of
how readily each network represents what self-play will have to teach it.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

import qubic_zero as qz
from qubic_zero.evaluate import _mixed_player
from qubic_zero.exact import _bits, _scan, to_bits
from qubic_zero.net import count_weights, make_net

OUT = Path(__file__).resolve().parent.parent / "results" / "network-comparison.json"
N_TRAIN, N_TEST = 24_000, 6_000
STEPS, BATCH, LR = 800, 256, 2e-3
SCORE_AT = [0, 25, 50, 100, 200, 400, 800]
SEEDS = range(3)
NETWORKS = {
    "line": ("line", {"width": 64, "depth": 4}, False),
    "conv, with all symmetric images": ("conv", {"channels": 24, "blocks": 3}, True),
    "conv": ("conv", {"channels": 24, "blocks": 3}, False),
}


def labelled_positions(game, n, seed):
    """Distinct positions (up to symmetry) with what the threat search knows about them."""
    rng = np.random.default_rng(seed)
    bits = _bits(game)
    found = {}
    while len(found) < n:
        boards = game.empty(256)
        for _ in range(game.n_cells):
            boards = boards[~game.finished(boards)]
            if len(boards) == 0:
                break
            keep = rng.random(len(boards)) < 0.25
            if keep.any():
                for image in game.canonical(boards[keep])[0]:
                    found.setdefault(image.tobytes(), image)
            boards = game.play(boards, _mixed_player(game, boards, rng))
    boards = np.stack(list(found.values()))
    boards = boards[rng.permutation(len(boards))[:n]]  # early and late positions alike in every part of the list
    right = np.zeros(boards.shape, dtype=bool)
    length = np.zeros(len(boards), dtype=np.int64)  # of the shortest forced win; 0 if there is none
    for i, board in enumerate(boards):
        moves = qz.threat_moves(game, board, max_nodes=20_000)
        if moves:
            right[i, moves] = True
            length[i] = qz.forced_win_length(game, board, limit=30) or 30
        else:
            _, must, _ = _scan(bits, *to_bits(game, board))
            if must and must & (must - 1) == 0:
                right[i, must.bit_length() - 1] = True  # the one move that does not lose at once
    return boards, right, length


def score(net, game, boards, right, length):
    net.eval()
    with torch.no_grad():
        positions = torch.from_numpy(boards * game.to_move(boards)[:, None])
        scores, values = net(positions)
        moves = scores.masked_fill(positions != 0, -torch.inf).argmax(dim=1).numpy()
        says_forced = values.numpy() > 0
    hit = right[np.arange(len(boards)), moves]
    forced = length > 0
    block = ~forced & right.any(axis=1)
    out = {
        "value_right": float((says_forced == forced).mean()),
        "value_right_forced": float(says_forced[forced].mean()),
        "value_right_not_forced": float((~says_forced[~forced]).mean()),
        "move_right_block": float(hit[block].mean()),
    }
    for name, where in (("1", length == 1), ("2", length == 2), ("3-4", (length >= 3) & (length <= 4)),
                        ("5-7", (length >= 5) & (length <= 7)), ("8+", length >= 8), ("2+", length >= 2)):  # fmt: skip
        out[f"move_right_force_{name}"] = float(hit[where].mean())
        out[f"value_right_force_{name}"] = float(says_forced[where].mean())
    return out


def train(name, seed, game, data, test):
    kind, sizes, augment = NETWORKS[name]
    net = make_net(game, kind, seed=seed, **sizes)
    optimiser = torch.optim.Adam(net.parameters(), lr=LR)
    rng = np.random.default_rng(seed)
    boards, right, length = data
    target_value = np.where(length > 0, 1.0, -1.0).astype(np.float32)
    has_move = right.any(axis=1)
    target_policy = (right / np.maximum(right.sum(axis=1, keepdims=True), 1)).astype(np.float32)
    curve, seconds = [], 0.0
    for step in range(STEPS + 1):
        if step in SCORE_AT:
            curve.append({"step": step, "seconds": round(seconds, 1), **score(net, game, *test)})
        if step == STEPS:
            break
        start = time.perf_counter()
        net.train()
        for group in optimiser.param_groups:
            group["lr"] = LR * (1.0 - 0.9 * step / STEPS)
        idx = rng.integers(0, len(boards), size=BATCH)
        b, p = boards[idx] * game.to_move(boards[idx])[:, None], target_policy[idx]
        if augment:
            perm = game.symmetries[rng.integers(0, len(game.symmetries), size=BATCH)]
            b, p = np.take_along_axis(b, perm, axis=1), np.take_along_axis(p, perm, axis=1)
        b = torch.from_numpy(b)
        scores, values = net(b)
        log_p = torch.log_softmax(scores.masked_fill(b != 0, -1e9), dim=1)
        weight = torch.from_numpy(has_move[idx].astype(np.float32))
        policy_loss = (-(torch.from_numpy(p) * log_p).sum(dim=1) * weight).sum() / weight.sum().clamp(min=1)
        value_loss = torch.mean((values - torch.from_numpy(target_value[idx])) ** 2)
        optimiser.zero_grad()
        (policy_loss + value_loss).backward()
        optimiser.step()
        seconds += time.perf_counter() - start
    return {"network": name, "seed": seed, "weights": count_weights(net), "curve": curve, "seconds": round(seconds, 1)}


def main():
    torch.set_num_threads(2)
    game = qz.qubic()
    started = time.perf_counter()
    boards, right, length = labelled_positions(game, N_TRAIN + N_TEST, seed=2026)
    data = (boards[:N_TRAIN], right[:N_TRAIN], length[:N_TRAIN])
    test = (boards[N_TRAIN:], right[N_TRAIN:], length[N_TRAIN:])
    forced, block = length > 0, (length == 0) & right.any(axis=1)
    meta = {
        "train": N_TRAIN, "test": N_TEST, "steps": STEPS, "batch": BATCH,
        "share_forced": float(forced.mean()), "share_block": float(block.mean()),
        "test_counts": {"forced": int(forced[N_TRAIN:].sum()), "block": int(block[N_TRAIN:].sum()),
                        **{f"length_{k}": int((length[N_TRAIN:] == k).sum()) for k in range(1, 8)},
                        "length_8+": int((length[N_TRAIN:] >= 8).sum())},
        "random_move_right_force_2+": float((right[N_TRAIN:][length[N_TRAIN:] >= 2].sum(axis=1)
                                             / (boards[N_TRAIN:][length[N_TRAIN:] >= 2] == 0).sum(axis=1)).mean()),
        "labelling_seconds": round(time.perf_counter() - started, 1),
    }  # fmt: skip
    print(json.dumps(meta), flush=True)
    runs = []
    for seed in SEEDS:
        for name in NETWORKS:
            runs.append(train(name, seed, game, data, test))
            last = runs[-1]["curve"][-1]
            print(f"{name!r} seed {seed}: value {last['value_right']:.3f}, forced moves {last['move_right_force_2+']:.3f}, "
                  f"blocks {last['move_right_block']:.3f} in {runs[-1]['seconds']:.0f} s", flush=True)  # fmt: skip
            OUT.write_text(json.dumps({"meta": meta, "runs": runs}, indent=1))


if __name__ == "__main__":
    main()
