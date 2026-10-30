"""Figure 3: subset-scale effects do not survive full-bucket evaluation.

Paired dot/slope plot of the same controlled contrasts measured at
n=200 and n=900 on an otherwise identical pipeline. Numbers are the
values reported in the paper's F2 analysis; 95% CIs are shown where
available and the equivalence margin is drawn where a TOST verdict is
claimed.

    python paper/figures/build_fig3_scalegap.py
    -> paper/figures/fig3_scalegap.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from f4p_style import PALETTE, FigureStyle, apply_publication_style, finalize_figure

HERE = Path(__file__).resolve().parent

SUB = PALETTE["red_strong"]     # subset-scale estimate
FULL = PALETTE["blue_main"]     # full-bucket estimate
INK, SOFT, LIGHT = PALETTE["ink"], PALETTE["ink_soft"], PALETTE["ink_light"]

# label, delta@200, delta@900, CI@900 (or None), equivalence margin (or None)
CONTRASTS = [
    ("Planner vs.\nalways-inject ASR", -3.0, +0.6, (-1.2, +2.2), 3.0),
    ("Planner vs.\nuniform",           +3.5, +2.2, (-0.7, +5.1), None),
    ("Cascade $+$ ASR vs.\nuniform",   +6.5, +1.7, (-1.6, +4.8), None),
    ("Cascade vs.\nuniform",           +3.5, +1.2, (-1.7, +4.1), 5.0),
]


def build() -> None:
    apply_publication_style(FigureStyle(font_size=7, axes_linewidth=1.0))
    fig, ax = plt.subplots(figsize=(3.35, 1.92))
    fig.subplots_adjust(left=0.31, right=0.98, top=0.77, bottom=0.20)

    ax.axvline(0.0, color=INK, lw=1.0, zorder=2)

    for y, (label, d200, d900, ci, margin) in enumerate(CONTRASTS):
        if margin is not None:
            ax.add_patch(plt.Rectangle((-margin, y - 0.34), 2 * margin, 0.68,
                                       facecolor=PALETTE["neutral"], alpha=0.28,
                                       linewidth=0, zorder=1))
        ax.annotate("", xy=(d900, y), xytext=(d200, y),
                    arrowprops=dict(arrowstyle="-|>", color=LIGHT, linewidth=1.1,
                                    shrinkA=3.0, shrinkB=3.0), zorder=3)
        if ci is not None:
            ax.plot([ci[0], ci[1]], [y, y], color=FULL, lw=1.6, alpha=0.45,
                    solid_capstyle="butt", zorder=3)
        ax.scatter([d200], [y], s=30, facecolor="white", edgecolor=SUB,
                   linewidth=1.4, zorder=5)
        ax.scatter([d900], [y], s=34, facecolor=FULL, edgecolor=FULL,
                   linewidth=1.0, zorder=5)
        ax.text(d200, y + 0.30, f"{d200:+.1f}", fontsize=5.8, color=SUB,
                ha="center", va="bottom", fontweight="bold")
        ax.text(d900, y - 0.30, f"{d900:+.1f}", fontsize=5.8, color=FULL,
                ha="center", va="top", fontweight="bold")

    ax.set_yticks(range(len(CONTRASTS)))
    ax.set_yticklabels([c[0] for c in CONTRASTS], fontsize=6.2, linespacing=1.3)
    ax.set_ylim(-1.15, len(CONTRASTS) - 0.28)
    ax.set_xlim(-6.2, 9.4)
    ax.set_xlabel("paired accuracy difference (pt)", fontsize=6.8, labelpad=2)
    ax.tick_params(axis="x", labelsize=6.2, width=1.0, length=3)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)

    ax.scatter([], [], s=30, facecolor="white", edgecolor=SUB, linewidth=1.4,
               label="subset estimate ($n{=}200$)")
    ax.scatter([], [], s=34, facecolor=FULL, edgecolor=FULL, linewidth=1.0,
               label="full bucket ($n{=}900$) with 95% CI")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.31, 0.99),
               fontsize=5.6, handletextpad=0.4, labelspacing=0.25,
               borderaxespad=0, ncols=1)

    finalize_figure(fig, HERE.parent / "fig3_scalegap", formats=["pdf", "png"], dpi=400, pad=0.02)


if __name__ == "__main__":
    build()
