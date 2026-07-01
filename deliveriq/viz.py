"""One shared, restrained chart style for every static figure (matplotlib)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # fixed order, never cycled
NEUTRAL = "#8a8983"


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
        "axes.titlesize": 12, "axes.titleweight": "semibold",
        "axes.labelsize": 10, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 2, "font.size": 10,
    })


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
