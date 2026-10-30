"""House style helpers following the `scientific-figure-making` skill.

API mirrors https://github.com/ChenLiu-1996/figures4papers
(`scientific-figure-making/references/api.md`).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "highlight": "#FFD700",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "ink": "#272727",
    "ink_soft": "#4D4D4D",
    "ink_light": "#767676",
}

DEFAULT_COLORS = [
    PALETTE["blue_main"],
    PALETTE["green_3"],
    PALETTE["red_strong"],
    PALETTE["teal"],
    PALETTE["violet"],
    PALETTE["neutral"],
]


@dataclass(frozen=True)
class FigureStyle:
    font_size: int = 16
    axes_linewidth: float = 2.5
    use_tex: bool = False
    font_family: tuple[str, ...] = ("Arial", "Helvetica", "DejaVu Sans", "sans-serif")


def apply_publication_style(style: FigureStyle | None = None) -> None:
    style = style or FigureStyle()
    plt.rcParams.update(
        {
            "font.family": list(style.font_family),
            "font.size": style.font_size,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": style.axes_linewidth,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def create_subplots(nrows: int = 1, ncols: int = 1, figsize=None, **kwargs):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return fig, np.atleast_1d(np.asarray(axes)).ravel()


_SUPPORTED = {"pdf", "svg", "eps", "png", "jpg", "jpeg", "tif", "tiff"}


def finalize_figure(fig, out_path, formats=None, dpi: int = 300, close: bool = True, pad: float = 0.05, **kwargs):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if formats is None:
        formats = [out_path.suffix.lstrip(".")] if out_path.suffix else ["pdf", "svg"]
    saved = []
    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        if fmt not in _SUPPORTED:
            raise ValueError(f"unsupported format: {fmt}")
        target = out_path.with_suffix(f".{fmt}")
        fig.savefig(target, dpi=dpi, bbox_inches="tight", pad_inches=pad, **kwargs)
        saved.append(target)
    if close:
        plt.close(fig)
    return saved


def rounded_box(ax, x, y, w, h, facecolor="white", edgecolor=PALETTE["ink"], lw=1.6, radius=0.012, zorder=2, **kw):
    from matplotlib.patches import FancyBboxPatch

    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=zorder,
        **kw,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, xy_from, xy_to, color=PALETTE["ink_soft"], lw=1.8, zorder=3, shrink=2.0):
    from matplotlib.patches import FancyArrowPatch

    a = FancyArrowPatch(
        xy_from,
        xy_to,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=lw,
        color=color,
        zorder=zorder,
        shrinkA=shrink,
        shrinkB=shrink,
    )
    ax.add_patch(a)
    return a


def blank_canvas(figsize):
    """Full-bleed axes in 0-1 coordinates for concept diagrams."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    return fig, ax


def place_image(ax, path, x, y, w, h, border=PALETTE["ink_soft"], lw=1.0, zorder=4):
    """Draw an image into 0-1 canvas coordinates, preserving the given box."""
    from matplotlib.patches import Rectangle
    from PIL import Image

    img = np.asarray(Image.open(path).convert("RGB"))
    ax.imshow(img, extent=(x, x + w, y, y + h), aspect="auto", zorder=zorder, interpolation="bilinear")
    if border is not None:
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=border, linewidth=lw, zorder=zorder + 1))
