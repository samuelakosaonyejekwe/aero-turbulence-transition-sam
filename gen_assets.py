"""
gen_assets.py
Build assets/banner.png and assets/social-preview.png from the generated CSVs.

These two were hand-made, and they were the only artefacts in the project with
no generating source.  They drifted, as everything without one does: the banner
and the social card both claimed "~37 % drag reduction" against the 50.4 % the
solution returns, and "<= 4 % validation error" against a five-plate mean of
8.3 % and a worst case of 16.5 %.  They are the FIRST thing a reader sees - the
banner heads the README and the card is what the GitHub Pages link unfurls with
- so they were the most visible wrong numbers in the repository.

Every figure on them is now read from the same CSVs the report is checked
against, and the wing render is the one 05_postprocessing/three_d writes, so
the images cannot say anything the solution does not.

Author: Akosa Samuel Onyejekwe, 2026.
"""
import os

import numpy as np
import pandas as pd
import utss_paths  # noqa: F401  - anchors the repo root and solver/ on
                   # sys.path, so this script works from any directory
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ASSETS = "assets"
os.makedirs(ASSETS, exist_ok=True)

# The banner's own palette.  Darker than uplot's, because these sit on a
# GitHub page rather than in the report, but the same rule holds: no black.
BG_DARK = "#0b1f3a"
BG_MID = "#12305a"
ACCENT = "#5bc0f8"
TEXT = "#eef4fb"
SOFT = "#a8bed6"
RULE = "#2f6ea8"

WING = "05_postprocessing/three_d/td_Cf.png"


def headline():
    """The three claims, from the CSVs that settle them."""
    nvt = pd.read_csv("04_solution/nlf_vs_turbulent.csv")
    vsum = pd.read_csv("06_validation/validation_summary.csv")
    # the bracket error where the plate has one (T3C4's onset is not resolved
    # to a station), the point error otherwise
    err = vsum.Re_theta_t_err_bracket_pct.fillna(vsum.Re_theta_t_err_pct).abs()
    return dict(
        drag=float(nvt.viscous_drag_reduction_pct.iloc[0]),
        laminar=float(nvt.mean_laminar_pct.iloc[0]),
        onset_mean=float(err.mean()),
        n_plates=int(len(err)),
    )


def _panel(fig, w_in, h_in):
    """Dark gradient ground, drawn rather than shipped as an image."""
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "utss_bg", [BG_DARK, BG_MID, BG_DARK])
    ax.imshow(grad, extent=(0, 1, 0, 1), aspect="auto", cmap=cmap,
              vmin=0, vmax=1, zorder=0)
    return ax


def _wing_crop():
    """The wing surface cut out of the 3-D render, without its furniture.

    Pasting the whole figure in shrinks the wing to a fifth of the panel and
    carries its title, colorbar and axis labels with it, unreadable at that
    size.  The surface is found instead by its SATURATION - it is the only
    strongly coloured thing on a light-grey 3-D box - and the colorbar, which
    is also saturated, is identified as the narrow columns that are coloured
    over most of the figure height and excluded.
    """
    im = plt.imread(WING)
    rgb = im[..., :3]
    mx = rgb.max(-1); mn = rgb.min(-1)
    mask = (mx - mn > 0.25) & (mx > 0.3)
    if not mask.any():
        return im
    h, w = mask.shape
    bar = mask.sum(0) > 0.25*h            # the colorbar strip
    right = int(np.nonzero(bar)[0].min()) - int(0.02*w) if bar.any() else w
    m2 = mask[:, :right]
    if not m2.any():
        return im
    ys, xs = np.nonzero(m2)
    # Asymmetric on purpose.  The tick labels sit BELOW the surface, so equal
    # padding either cut them through the middle of a glyph or dragged the
    # whole axis frame in; a small bottom margin excludes them cleanly and the
    # wing stays the subject.
    px = int(0.035*w)
    y0 = max(0, ys.min() - int(0.055*h)); y1 = min(h, ys.max() + int(0.012*h))
    x0 = max(0, xs.min() - px); x1 = min(right, xs.max() + px)
    return im[y0:y1, x0:x1]


def _wing(fig, rect):
    """The 3-D wing render 05_postprocessing writes, cropped to the surface."""
    if not os.path.exists(WING):
        return
    ax = fig.add_axes(rect)
    ax.imshow(_wing_crop())
    ax.axis("off")
    ax.set_zorder(2)


def _stat(ax, x, y, big, small, size_big, size_small):
    ax.text(x, y, big, color=ACCENT, fontsize=size_big, fontweight="bold",
            ha="left", va="center", family="DejaVu Sans")
    ax.text(x + 0.075, y, small, color=SOFT, fontsize=size_small,
            ha="left", va="center", family="DejaVu Sans")


def banner(h):
    fig = plt.figure(figsize=(16.0, 4.0), dpi=100)
    ax = _panel(fig, 16.0, 4.0)
    _wing(fig, [0.700, 0.075, 0.290, 0.85])
    ax.add_patch(Rectangle((0.70, 0), 0.30, 1, transform=ax.transAxes,
                           color=BG_DARK, alpha=0.0, zorder=1))
    ax.text(0.043, 0.845, "AERODYNAMICS   ·   CFD CASE STUDY", color=ACCENT,
            fontsize=13, fontweight="bold", va="center")
    ax.text(0.040, 0.665, "AETHER-NLF 25", color=TEXT, fontsize=46,
            fontweight="bold", va="center")
    ax.text(0.043, 0.505,
            "Laminar → Turbulent Boundary-Layer Transition over a 3-D NLF Wing",
            color=TEXT, fontsize=17, va="center")
    ax.plot([0.043, 0.30], [0.425, 0.425], color=RULE, lw=2.4)
    _stat(ax, 0.043, 0.315, "%.0f%%" % h["drag"], "viscous drag reduction", 21, 13)
    _stat(ax, 0.275, 0.315, "%.0f%%" % h["laminar"], "laminar chord", 21, 13)
    _stat(ax, 0.455, 0.315, "4-in-1", "transition kernel", 21, 13)
    ax.text(0.043, 0.145, "Akosa Samuel Onyejekwe", color=TEXT, fontsize=14,
            fontweight="bold", va="center")
    ax.text(0.043, 0.065, "UTSS — Universal Transition & Skin-friction Solver",
            color=SOFT, fontsize=11.5, va="center")
    fig.savefig(f"{ASSETS}/banner.png", dpi=100, facecolor=BG_DARK)
    plt.close(fig)


def social(h):
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
    ax = _panel(fig, 12.8, 6.4)
    _wing(fig, [0.520, 0.115, 0.470, 0.77])
    ax.text(0.050, 0.875, "AERODYNAMICS   ·   CFD CASE STUDY", color=ACCENT,
            fontsize=14, fontweight="bold", va="center")
    ax.text(0.047, 0.755, "AETHER-NLF 25", color=TEXT, fontsize=48,
            fontweight="bold", va="center")
    ax.text(0.050, 0.630,
            "Laminar → Turbulent Boundary-Layer\nTransition over a 3-D NLF Wing",
            color=TEXT, fontsize=19, va="center", linespacing=1.35)
    ax.plot([0.050, 0.335], [0.525, 0.525], color=RULE, lw=2.6)
    rows = [("%.0f%%" % h["drag"], "viscous drag reduction (NLF vs turbulent)"),
            ("%.1f%%" % h["onset_mean"],
             "mean transition-onset error, %d flat plates" % h["n_plates"]),
            ("4-in-1", "natural · bypass · separation · cross-flow")]
    for i, (big, small) in enumerate(rows):
        y = 0.435 - 0.083*i
        ax.text(0.050, y, big, color=ACCENT, fontsize=25, fontweight="bold",
                ha="left", va="center")
        ax.text(0.175, y, small, color=SOFT, fontsize=14, ha="left", va="center")
    ax.text(0.050, 0.115, "Akosa Samuel Onyejekwe", color=TEXT, fontsize=19,
            fontweight="bold", va="center")
    ax.text(0.050, 0.048, "UTSS  —  Universal Transition & Skin-friction Solver",
            color=SOFT, fontsize=13, va="center")
    fig.savefig(f"{ASSETS}/social-preview.png", dpi=100, facecolor=BG_DARK)
    plt.close(fig)


if __name__ == "__main__":
    h = headline()
    banner(h)
    social(h)
    print("assets rebuilt from the CSVs: %.1f %% drag reduction, %.1f %% mean "
          "laminar chord, %.1f %% mean onset error over %d plates"
          % (h["drag"], h["laminar"], h["onset_mean"], h["n_plates"]))
    print("wrote", ", ".join(sorted(os.listdir(ASSETS))))
