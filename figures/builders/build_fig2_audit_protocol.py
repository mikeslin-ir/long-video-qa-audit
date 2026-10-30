"""Figure 2: matched-contrast audit protocol."""
from __future__ import annotations

from pathlib import Path

from matplotlib.patches import FancyArrowPatch, Rectangle

from f4p_style import (
    PALETTE,
    FigureStyle,
    apply_publication_style,
    blank_canvas,
    finalize_figure,
    place_image,
)

HERE = Path(__file__).resolve().parent
FRAMES = HERE / "fig2_materials" / "frames"

INK, SOFT, LINE = PALETTE["ink"], PALETTE["ink_soft"], PALETTE["neutral"]
BLUE, ORANGE = PALETTE["blue_main"], "#B65F00"
PURPLE, GREY = PALETTE["violet"], "#8C8C8C"


def panel_title(ax, x: float, letter: str, title: str) -> None:
    ax.text(x, 0.945, f"({letter})", fontsize=7.5, color=INK,
            fontweight="bold", va="center")
    ax.text(x + 0.035, 0.945, title, fontsize=7.5, color=INK,
            fontweight="bold", va="center")


def rect(ax, x, y, w, h, edge=LINE, fill="white", lw=0.8, dashed=False) -> None:
    ax.add_patch(Rectangle((x, y), w, h, linewidth=lw, edgecolor=edge,
                           facecolor=fill, linestyle="--" if dashed else "-"))


def arrow(ax, start, end, color=SOFT, dashed=False) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=7,
                                 linewidth=0.9, color=color,
                                 linestyle="--" if dashed else "-",
                                 shrinkA=1, shrinkB=1))


def build() -> None:
    apply_publication_style(FigureStyle(font_size=7, axes_linewidth=0.8))
    fig, ax = blank_canvas((7.0, 1.72))

    for x in (0.315, 0.665):
        ax.plot([x, x], [0.06, 0.91], color=LINE, linewidth=0.7)

    panel_title(ax, 0.015, "a", "Hold the substrate fixed")
    paths = sorted(FRAMES.glob("select_frame_*.png"))
    shown = [paths[index] for index in (0, 2, 4, 7)]
    for index, path in enumerate(shown):
        place_image(ax, path, 0.02 + index * 0.068, 0.72, 0.060, 0.105,
                    border="white", lw=0.4)
    ax.text(0.15, 0.685, r"shared candidate pool $\mathcal{C}_1$", fontsize=5.1,
            color=SOFT, ha="center")

    ax.text(0.02, 0.595, r"Question $q$", fontsize=5.7, color=INK,
            fontweight="bold", va="center")
    ax.text(0.16, 0.595, r"Frame budget $K=8$", fontsize=5.7, color=INK,
            fontweight="bold", va="center")
    ax.plot([0.02, 0.295], [0.555, 0.555], color=LINE, linewidth=0.6)

    fixed = [
        "backbone + weights", "decoding", "prompt + order",
        "grader + rubric", "item set",
    ]
    for index, item in enumerate(fixed):
        y = 0.475 - index * 0.078
        rect(ax, 0.022, y - 0.012, 0.012, 0.024, edge=GREY, fill=GREY, lw=0)
        ax.text(0.045, y, item, fontsize=5.3, color=SOFT, va="center")
    ax.text(0.02, 0.07, r"$\Rightarrow$ difference attributable to one flip",
            fontsize=5.2, color=INK, fontweight="bold", va="center")

    panel_title(ax, 0.33, "b", "Flip one routing axis")
    ax.text(0.465, 0.84, "ARM A", fontsize=5.0, color=GREY,
            fontweight="bold", ha="center")
    ax.text(0.595, 0.84, "ARM B", fontsize=5.0, color=GREY,
            fontweight="bold", ha="center")

    rows = [
        (BLUE, "SELECT", "uniform frames", "propose · filter · align · select", False),
        (ORANGE, "INJECT", "frames + question", "+ ASR / OCR", False),
        (PURPLE, "EXPERT", "generalist VLM", "AV expert", True),
    ]
    for index, (color, name, arm_a, arm_b, dashed) in enumerate(rows):
        y = 0.655 - index * 0.235
        ax.text(0.335, y + 0.055, name, fontsize=5.8, color=color,
                fontweight="bold", va="center")
        rect(ax, 0.405, y, 0.12, 0.11, edge=LINE, fill="white")
        ax.text(0.465, y + 0.055, arm_a, fontsize=5.0, color=SOFT,
                ha="center", va="center")
        arrow(ax, (0.530, y + 0.055), (0.552, y + 0.055), color=color,
              dashed=dashed)
        rect(ax, 0.557, y, 0.095, 0.11, edge=color, fill="white", lw=1.0,
             dashed=dashed)
        if index == 0:
            ax.text(0.6045, y + 0.068, "cascade", fontsize=5.1, color=color,
                    fontweight="bold", ha="center")
            ax.text(0.6045, y + 0.035, "P · F · A · S", fontsize=4.5,
                    color=SOFT, ha="center")
        else:
            ax.text(0.6045, y + 0.055, arm_b, fontsize=4.8, color=color,
                    fontweight="bold", ha="center", va="center")

    panel_title(ax, 0.68, "c", "Measure paired outcomes")
    ax.text(0.685, 0.815, "same items under both arms", fontsize=5.6,
            color=SOFT, va="center")
    arm_a = [1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1]
    arm_b = [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1]
    cell_w, cell_h, gap = 0.018, 0.072, 0.004
    for row, (label, values, color) in enumerate((("A", arm_a, GREY), ("B", arm_b, BLUE))):
        y = 0.650 - row * 0.098
        ax.text(0.685, y + cell_h / 2, label, fontsize=5.6, color=color,
                fontweight="bold", va="center")
        for index, correct in enumerate(values):
            x = 0.705 + index * (cell_w + gap)
            rect(ax, x, y, cell_w, cell_h, edge=color if correct else LINE,
                 fill=color if correct else "white", lw=0.6)
            if arm_a[index] != arm_b[index]:
                rect(ax, x - 0.002, y - 0.006, cell_w + 0.004, cell_h + 0.012,
                     edge=ORANGE, fill="none", lw=0.8)
    ax.text(0.705, 0.470, "discordant pairs carry the test", fontsize=5.4,
            color=ORANGE, fontweight="bold", va="center")

    tx, ty, cw, ch = 0.695, 0.105, 0.048, 0.105
    ax.text(tx, 0.360, "exact McNemar", fontsize=5.7, color=INK,
            fontweight="bold", va="center")
    for index, (label, hot) in enumerate((("a", False), ("b", True), ("c", True), ("d", False))):
        row, col = divmod(index, 2)
        x, y = tx + col * cw, ty + (1 - row) * ch
        rect(ax, x, y, cw, ch, edge=LINE, fill="#FFF6EC" if hot else "white")
        ax.text(x + cw / 2, y + ch / 2, label, fontsize=5.5,
                color=ORANGE if hot else GREY, fontweight="bold" if hot else "normal",
                ha="center", va="center")
    ax.text(0.805, 0.265, r"$\Delta=(b-c)/n$", fontsize=5.8, color=INK, va="center")
    ax.text(0.805, 0.195, "bootstrap 95% CI", fontsize=5.5, color=SOFT, va="center")
    ax.text(0.805, 0.125, r"exact $p$; TOST $\pm5$ pt", fontsize=5.5,
            color=SOFT, va="center")

    finalize_figure(fig, HERE / "fig2_pipeline", formats=["pdf", "png"],
                    dpi=400, pad=0.012)


if __name__ == "__main__":
    build()
