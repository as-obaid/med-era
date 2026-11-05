"""Shared figure style and output for the results figures (CPU, matplotlib; imported only by the figure builders).

Every figure is written as PNG (300 dpi) and SVG to results/figures/, with the author and title in the file
metadata and no timestamp, so a rebuild from the same results gives the same SVG.

Colours. Eras use the Okabe-Ito blue / orange / green (the same as the W&B report); the three pass the categorical
palette checks (lightness band, chroma floor, CVD and normal-vision separation for all pairs), and orange is below
3:1 on white, so lines are also labelled directly. The diverging map runs blue (negative: lower perplexity, better)
through a neutral grey to red (positive: worse), with the red arm matched to the blue arm's lightness in OKLab so
equal magnitudes look equally strong; blue/red stays separable under protanopia and deuteranopia.
"""
from pathlib import Path

AUTHOR = "Muhammad Obaidullah"
DPI = 300

ERA_COLORS = {"w1": "#0072B2", "w2": "#E69F00", "w3": "#009E73"}
ERA_YEARS = {"w1": "2011–15", "w2": "2016–20", "w3": "2021–26"}
ARM_LABELS = {"qwen": "Qwen", "gemma_al": "Gemma-AL", "gemma_ex": "Gemma-EX"}
ARM_SUBTITLES = {"qwen": "Qwen3.5-27B, dense", "gemma_al": "Gemma-4-26B-A4B, all-linear",
                 "gemma_ex": "Gemma-4-26B-A4B, all-linear + experts"}

# blue arm: reference ramp steps 550/400/250/100; grey midpoint; red arm at the same OKLab lightness steps
DIVERGING = ["#1c5cab", "#3987e5", "#86b6ef", "#cde2fb", "#f0efec", "#fdd5d1", "#f0968f", "#e34948", "#9e3432"]

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.linewidth": 0.8,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "text.color": INK,
    "legend.frameon": False,
    "svg.hashsalt": "medera",
    "path.simplify": False,
}


def diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("medera_diverging", DIVERGING, N=256)


def save(fig, out_dir: Path, stem: str, title: str) -> list[Path]:
    """Write <stem>.png and <stem>.svg; returns both paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png, svg = out_dir / f"{stem}.png", out_dir / f"{stem}.svg"
    fig.savefig(png, dpi=DPI, facecolor=SURFACE, metadata={"Title": title, "Author": AUTHOR})
    fig.savefig(svg, facecolor=SURFACE, metadata={"Title": title, "Creator": AUTHOR, "Date": None})
    return [png, svg]
