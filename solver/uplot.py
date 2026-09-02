"""
uplot.py  -  Shared plotting style and helpers for the UTSS case study.

Enforces the project rule: NEVER use the colour black, anywhere
(text, axes, ticks, lines, edges).  All "dark" elements use a deep
navy ink colour instead of black, and a clean colourful palette is used
for data series.  All figures use generous margins so text never
overlaps the curves/contours.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

# ----------------------------------------------------------------------
# Colour system  (NO BLACK ANYWHERE)
# ----------------------------------------------------------------------
INK      = "#1d2f45"   # deep navy used instead of black for text/axes
INK_SOFT = "#3c5066"   # softer navy for grid / secondary text
PALETTE  = [
    "#1b6ca8",  # ocean blue
    "#d1495b",  # rose red
    "#2a9d8f",  # teal green
    "#e9a000",  # amber
    "#7b4ea3",  # violet
    "#e07a3f",  # burnt orange
    "#3a86ff",  # bright blue
    "#0f9d58",  # green
    "#c44da1",  # magenta
    "#5c6f4a",  # olive
]

# A perceptual blue->teal->amber->red map (no black, no pure white ends)
FIELD_CMAP = LinearSegmentedColormap.from_list(
    "utss_field",
    ["#2c3e8c", "#1b6ca8", "#2a9d8f", "#9ccb3b", "#e9a000", "#d1495b"],
)
CF_CMAP = LinearSegmentedColormap.from_list(
    "utss_cf", ["#264b96", "#2a9d8f", "#e9a000", "#d1495b"]
)
GAMMA_CMAP = LinearSegmentedColormap.from_list(
    "utss_gamma", ["#1b6ca8", "#5bc0be", "#e9a000", "#d1495b"]
)


def apply_style():
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 170,
        "figure.facecolor": "white",
        "axes.facecolor": "#fbfcfe",
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "normal",
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.linewidth": 1.1,
        "axes.grid": True,
        "grid.color": "#cdd7e2",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.8,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "legend.edgecolor": INK_SOFT,
        "legend.framealpha": 0.92,
        "lines.linewidth": 2.0,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        "figure.autolayout": False,
    })


def new_fig(w=8.4, h=5.4):
    fig, ax = plt.subplots(figsize=(w, h))
    return fig, ax


def finish(fig, path, caption=None):
    """Tidy layout (so text never overlaps data) and save."""
    fig.tight_layout(pad=1.4)
    if caption:
        fig.subplots_adjust(bottom=0.16)
        fig.text(0.5, 0.015, caption, ha="center", va="bottom",
                 fontsize=8.5, color=INK_SOFT, style="italic")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path

