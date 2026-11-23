"""The networks: one policy head and one value head on a shared body.

A network is shown a position from the point of view of the player to move
(+1 own stones, -1 the opponent's, 0 empty) and returns a score for every
cell (the policy, before the softmax) and one number in [-1, 1] (the value).

* :class:`LineNet` reads the board the way the rules define it.  Cells send
  what they hold to the lines they lie on, lines send what they see back to
  their cells, and that exchange is repeated a few times.  Nothing in it knows
  where a cell sits in space, only which lines it shares with which cells.  A
  relabelling of the cells that maps lines to lines therefore relabels the
  output in the same way, exactly and without training: all 192 symmetries of
  Qubic are built in.
* :class:`ConvNet` is the usual residual tower of convolutions, here as the
  control.  It treats neighbouring cells as related, which is how AlphaZero
  reads a Go board.

PyTorch is needed to train (``pip install -e ".[train]"``).  A trained
:class:`LineNet` can be saved as plain arrays and run by
:class:`NumpyLineNet`, which needs NumPy only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .games import LineGame, game_by_name

try:
    import torch
    from torch import nn
except ImportError:  # playing does not need PyTorch
    torch = None
    nn = None

__all__ = ["LineNet", "ConvNet", "make_net", "count_weights", "save_weights", "load_weights", "NumpyLineNet"]


def _need_torch() -> None:
    if torch is None:
        raise ImportError('this needs PyTorch: pip install -e ".[train]"')


if torch is not None:

    class _LineBlock(nn.Module):
        """One exchange: cells to lines, lines back to cells."""

        def __init__(self, width: int) -> None:
            super().__init__()
            self.to_line = nn.Linear(width, width)
            self.line = nn.Linear(width, width)
            self.cell = nn.Sequential(nn.Linear(3 * width, width), nn.ReLU(), nn.Linear(width, width))
            self.norm = nn.LayerNorm(width)

        def forward(self, h, incidence, per_line: float, per_cell: float):
            said = torch.relu(self.to_line(h))  # what each cell tells its lines
            lines = torch.relu(self.line((incidence @ said) / per_line))  # each line sums up its cells
            heard = (incidence.T @ lines) / per_cell  # each cell sums up its lines
            everyone = h.mean(dim=1, keepdim=True).expand_as(h)  # and hears the board as a whole
            h = self.norm(h + self.cell(torch.cat([h, heard, everyone], dim=-1)))
            return h, lines

    class LineNet(nn.Module):
        """Message passing between the cells and the lines of a game (see the module's description).

        ``width`` is the size of the vector kept for every cell and every line,
        ``depth`` the number of exchanges.  After ``depth`` exchanges a cell knows
        about cells up to ``depth`` lines away.
        """

        kind = "line"

        def __init__(self, game: LineGame, width: int = 64, depth: int = 4) -> None:
            super().__init__()
            self.game_name, self.width, self.depth = game.name, width, depth
            self.register_buffer("incidence", torch.from_numpy(game.incidence).clone())
            self.per_line = float(game.k)
            self.per_cell = float(game.degree.mean())
            self.embed = nn.Embedding(3, width)  # the opponent's stone, empty, own stone
            self.blocks = nn.ModuleList(_LineBlock(width) for _ in range(depth))
            self.policy = nn.Linear(width, 1)
            self.value = nn.Sequential(nn.Linear(2 * width, width), nn.ReLU(), nn.Linear(width, 1))

        def forward(self, boards):
            """``boards``: (batch, n_cells) from the mover's point of view.  Returns (policy scores, values)."""
            h = self.embed(boards.long() + 1)
            lines = None
            for block in self.blocks:
                h, lines = block(h, self.incidence, self.per_line, self.per_cell)
            scores = self.policy(h).squeeze(-1)
            summary = torch.cat([h.mean(dim=1), lines.mean(dim=1)], dim=-1)
            return scores, torch.tanh(self.value(summary)).squeeze(-1)

        def config(self) -> dict:
            return {"kind": "line", "game": self.game_name, "width": self.width, "depth": self.depth}

    class _ConvBlock(nn.Module):
        def __init__(self, conv, channels: int) -> None:
            super().__init__()
            self.conv1, self.conv2 = conv(channels, channels, 3, padding=1), conv(channels, channels, 3, padding=1)
            self.norm1, self.norm2 = nn.GroupNorm(1, channels), nn.GroupNorm(1, channels)

        def forward(self, x):
            y = torch.relu(self.norm1(self.conv1(x)))
            return torch.relu(x + self.norm2(self.conv2(y)))

    class ConvNet(nn.Module):
        """A residual tower of 3 x 3 (x 3) convolutions with AlphaZero's two heads.

        Two input images (own stones, the opponent's), ``blocks`` residual blocks
        of ``channels`` channels, a policy head and a value head that each end in
        dense layers.  Works on square and on cubic boards.
        """

        kind = "conv"

        def __init__(self, game: LineGame, channels: int = 32, blocks: int = 4) -> None:
            super().__init__()
            if len(game.shape) not in (2, 3):
                raise ValueError("ConvNet needs a square or a cubic board")
            self.game_name, self.channels, self.n_blocks, self.shape = game.name, channels, blocks, game.shape
            conv = nn.Conv2d if len(game.shape) == 2 else nn.Conv3d
            self.first = conv(2, channels, 3, padding=1)
            self.tower = nn.Sequential(*[_ConvBlock(conv, channels) for _ in range(blocks)])
            self.policy_conv, self.value_conv = conv(channels, 2, 1), conv(channels, 1, 1)
            self.policy = nn.Linear(2 * game.n_cells, game.n_cells)
            self.value = nn.Sequential(nn.Linear(game.n_cells, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, boards):
            x = torch.stack([boards == 1, boards == -1], dim=1).to(self.first.weight.dtype).reshape(-1, 2, *self.shape)
            x = self.tower(torch.relu(self.first(x)))
            scores = self.policy(torch.relu(self.policy_conv(x)).flatten(1))
            return scores, torch.tanh(self.value(torch.relu(self.value_conv(x)).flatten(1))).squeeze(-1)

        def config(self) -> dict:
            return {"kind": "conv", "game": self.game_name, "channels": self.channels, "blocks": self.n_blocks}

else:  # pragma: no cover
    LineNet = ConvNet = None


def make_net(game: LineGame, kind: str = "line", seed: int = 0, **sizes):
    """A new network with initial weights drawn from ``seed``.

    The generator of PyTorch is seeded *here*, immediately before the weights are
    created, so that two networks made with the same arguments are identical
    whatever ran before.  ``sizes`` are passed to :class:`LineNet` (``width``,
    ``depth``) or :class:`ConvNet` (``channels``, ``blocks``).
    """
    _need_torch()
    torch.manual_seed(seed)
    if kind == "line":
        return LineNet(game, **sizes)
    if kind == "conv":
        return ConvNet(game, **sizes)
    raise ValueError(f"unknown kind of network {kind!r}")


def count_weights(net) -> int:
    return sum(p.numel() for p in net.parameters())


def save_weights(net, path) -> Path:
    """Write a network as plain arrays (``.npz``) with a description of what it is.

    Arrays are stored as 32-bit floats.  The file can be loaded again by
    :func:`load_weights` (as a PyTorch network) and, for a :class:`LineNet`, by
    :class:`NumpyLineNet`.
    """
    path = Path(path)
    arrays = {name: value.detach().cpu().numpy().astype(np.float32) for name, value in net.state_dict().items()}
    np.savez_compressed(path, __config__=np.array(json.dumps(net.config())), **arrays)
    return path


def load_weights(path):
    """The PyTorch network stored by :func:`save_weights`."""
    _need_torch()
    with np.load(Path(path)) as stored:
        config = json.loads(str(stored["__config__"]))
        arrays = {name: stored[name] for name in stored.files if name != "__config__"}
    game = game_by_name(config.pop("game"))
    net = make_net(game, config.pop("kind"), **config)
    net.load_state_dict({name: torch.from_numpy(value) for name, value in arrays.items()})
    return net.eval()


class NumpyLineNet:
    """A trained :class:`LineNet`, run with NumPy alone.

    ``NumpyLineNet(path)(boards)`` returns the same policy scores and values as
    the PyTorch network (to rounding).  This is what plays when PyTorch is not
    installed.
    """

    def __init__(self, path) -> None:
        with np.load(Path(path)) as stored:
            self.config = json.loads(str(stored["__config__"]))
            self.w = {name: stored[name].astype(np.float32) for name in stored.files if name != "__config__"}
        if self.config["kind"] != "line":
            raise ValueError("only a line network can be run without PyTorch")
        self.game = game_by_name(self.config["game"])
        self.depth = self.config["depth"]
        self.incidence = self.game.incidence
        self.per_line = np.float32(self.game.k)
        self.per_cell = np.float32(self.game.degree.mean())

    def _linear(self, x, name):
        return x @ self.w[f"{name}.weight"].T + self.w[f"{name}.bias"]

    def __call__(self, boards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        boards = np.asarray(boards).reshape(-1, self.game.n_cells)
        h = self.w["embed.weight"][boards.astype(np.int64) + 1]
        lines = None
        for i in range(self.depth):
            block = f"blocks.{i}"
            said = np.maximum(self._linear(h, f"{block}.to_line"), 0)
            lines = np.maximum(self._linear((self.incidence @ said) / self.per_line, f"{block}.line"), 0)
            heard = (self.incidence.T @ lines) / self.per_cell
            everyone = np.broadcast_to(h.mean(axis=1, keepdims=True), h.shape)
            update = np.maximum(self._linear(np.concatenate([h, heard, everyone], axis=-1), f"{block}.cell.0"), 0)
            h = h + self._linear(update, f"{block}.cell.2")
            mean = h.mean(axis=-1, keepdims=True)
            spread = np.sqrt(((h - mean) ** 2).mean(axis=-1, keepdims=True) + 1e-5)
            h = (h - mean) / spread * self.w[f"{block}.norm.weight"] + self.w[f"{block}.norm.bias"]
        scores = self._linear(h, "policy")[..., 0]
        summary = np.concatenate([h.mean(axis=1), lines.mean(axis=1)], axis=-1)
        value = self._linear(np.maximum(self._linear(summary, "value.0"), 0), "value.2")[..., 0]
        return scores, np.tanh(value)
