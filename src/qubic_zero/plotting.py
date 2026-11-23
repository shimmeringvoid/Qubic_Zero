"""Figure style, drawings of boards, and reading the logs of training runs.

Used by the notebooks; needs matplotlib and pandas (``pip install -e ".[notebooks]"``).
One palette is used everywhere, so that a colour means the same thing in
every figure: blue for the network alone, orange for the network with its
tree search, aqua for a third series when there is one.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

from .games import LineGame

__all__ = [
    "SERIES",
    "INK",
    "INK_SECONDARY",
    "MUTED",
    "GREY",
    "SURFACE",
    "use_style",
    "repo_root",
    "savefig",
    "load_log",
    "load_runs",
    "seed_lines",
    "draw_board",
]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
GREY = "#b4b2a9"  # de-emphasised marks
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def use_style() -> None:
    """Apply the shared matplotlib style.  Call once at the top of a notebook."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "font.size": 10,
            "text.color": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelsize": 10,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": AXIS,
            "ytick.color": AXIS,
            "xtick.labelcolor": MUTED,
            "ytick.labelcolor": MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.prop_cycle": cycler(color=SERIES),
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "lines.markersize": 6.5,
            "lines.markeredgecolor": SURFACE,
            "lines.markeredgewidth": 1.4,
            "scatter.edgecolors": SURFACE,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "figure.constrained_layout.use": True,
        }
    )


# --- files -----------------------------------------------------------------------
def repo_root() -> Path:
    """Root of the repository (the folder holding ``pyproject.toml``)."""
    here = Path(__file__).resolve()
    for start in (here, Path.cwd().resolve()):
        for parent in (start, *start.parents):
            if (parent / "pyproject.toml").exists():
                return parent
    return Path.cwd().resolve()


def savefig(fig, name: str) -> Path:
    """Save a figure as ``figures/<name>.png`` in the repository and return the path."""
    folder = repo_root() / "figures"
    folder.mkdir(exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path)
    return path


def load_log(run_dir):
    """The log of one training run as a table: one row per report, one column per quantity."""
    import pandas as pd

    lines = (Path(run_dir) / "log.jsonl").read_text().splitlines()
    return pd.DataFrame([json.loads(line) for line in lines])


def load_runs(setting: str) -> dict:
    """The logs of all seeds of one setting stored under ``results/``: ``{seed: table}``."""
    folder = repo_root() / "results" / setting
    runs = {int(path.name.split("-")[1]): load_log(path) for path in sorted(folder.glob("seed-*"))}
    if not runs:
        raise FileNotFoundError(f"no runs under {folder}; see tools/run_small_games.py")
    return runs


# --- drawing -----------------------------------------------------------------------
def seed_lines(ax, runs: dict, column: str, color: str, label: str, x: str = "positions", scale: float = 1.0):
    """One thin line per seed and a bold line for their median, for the reports that hold ``column``."""
    tables = [run.dropna(subset=[column]) for run in runs.values() if column in run]
    for table in tables:
        ax.plot(table[x], table[column] * scale, color=color, lw=0.9, alpha=0.4, marker="none")
    xs = tables[0][x].to_numpy()
    median = np.median(np.stack([table[column].to_numpy() for table in tables]), axis=0) * scale
    (line,) = ax.plot(xs, median, color=color, lw=2.0, marker="o", ms=4.5, label=label)
    return line


def draw_board(
    game: LineGame,
    board=None,
    ax=None,
    highlight=None,
    highlight_color: str = SERIES[2],
    labels=None,
    title: str | None = None,
    marks: bool = True,
):
    """Draw a position: the layers of a cube side by side, X as a blue cross and O as an orange ring.

    ``highlight`` is a list of cells to shade (a line, a move); ``labels`` maps
    cells to short strings written into them.
    """
    board = np.zeros(game.n_cells, dtype=np.int8) if board is None else np.asarray(board)
    shape = game.shape if len(game.shape) == 3 else (1, *game.shape)
    layers, rows, cols = shape
    if ax is None:
        _, ax = plt.subplots(figsize=(1.9 * layers * cols / 4 + 0.3, 1.9 * rows / 4 + 0.4))
    gap = 0.6
    highlight = set() if highlight is None else {int(c) for c in highlight}
    labels = labels or {}
    for cell in range(game.n_cells):
        layer, row, col = np.unravel_index(cell, shape)
        x0, y0 = layer * (cols + gap) + col, rows - 1 - row
        face = highlight_color if cell in highlight else SURFACE
        ax.add_patch(plt.Rectangle((x0, y0), 1, 1, facecolor=face, alpha=0.3 if cell in highlight else 1.0, edgecolor="none"))
        ax.add_patch(plt.Rectangle((x0, y0), 1, 1, facecolor="none", edgecolor=AXIS, lw=0.8))
        cx, cy = x0 + 0.5, y0 + 0.5
        if marks and board[cell] == 1:
            ax.plot([cx - 0.24, cx + 0.24], [cy - 0.24, cy + 0.24], color=SERIES[0], lw=2, solid_capstyle="round")
            ax.plot([cx - 0.24, cx + 0.24], [cy + 0.24, cy - 0.24], color=SERIES[0], lw=2, solid_capstyle="round")
        elif marks and board[cell] == -1:
            ax.add_patch(plt.Circle((cx, cy), 0.26, facecolor="none", edgecolor=SERIES[1], lw=2))
        if cell in labels:
            ax.text(cx, cy, labels[cell], ha="center", va="center", fontsize=8, color=INK_SECONDARY)
    ax.set_xlim(-0.1, layers * (cols + gap) - gap + 0.1)
    ax.set_ylim(-0.1, rows + 0.1)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title)
    return ax
