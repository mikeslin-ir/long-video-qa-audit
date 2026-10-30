#!/usr/bin/env python3
"""Generate the table-derived result figures from the numbers reported in the paper.

P0  fig_f2_heatmap.pdf     -- exploratory selection x injection grid
P3  fig_routing_ladder.pdf -- expert-routing head-room (base/routed/AV/oracle)

The numbers are the ones printed in the paper's tables (tab:f2grid, tab:avexpert) and
by scripts/routed_expert_analysis.py. Figure 4 (fig_f4_expert) is built from the
per-item predictions by figures/builders/build_fig4_expert.py, and Figure 3 by
figures/builders/build_fig3_scalegap.py. No model calls.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = Path(__file__).resolve().parents[1] / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def p0_heatmap() -> None:
    # tab:f2grid (Video-MME long, n=200, K=8). rows = selector, cols = injection
    cols = ["no-inject", "+ASR", "+OCR", "+ASR+OCR"]
    rows = ["Uniform-K", "Cascade"]
    acc = np.array([[50.0, 51.5, 51.0, 50.5],
                    [53.5, 56.5, 55.5, 55.0]])
    fig, ax = plt.subplots(figsize=(4.6, 2.1))
    im = ax.imshow(acc, cmap="YlGnBu", aspect="auto", vmin=49, vmax=57)
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    for i in range(acc.shape[0]):
        for j in range(acc.shape[1]):
            best = (i == 1 and j == 1)
            ax.text(j, i, f"{acc[i,j]:.1f}", ha="center", va="center",
                    color="white" if acc[i, j] > 54 else "black",
                    fontweight="bold" if best else "normal")
    # Highlight the largest observed cell without attributing an interaction.
    ax.add_patch(plt.Rectangle((0.5, 0.5), 1, 1, fill=False,
                               edgecolor="crimson", lw=2.2))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Accuracy (%)", fontsize=8)
    ax.set_title("Selection $\\times$ injection at subset scale")
    ax.text(1.5, 2.05,
            r"red cell: cascade$+$ASR is $+6.5$ pt vs. uniform/no-inject",
            ha="center", va="center", color="crimson", fontsize=8)
    fig.savefig(FIG / "fig_f2_heatmap.pdf")
    plt.close(fig)
    print("wrote fig_f2_heatmap.pdf")


def p3_ladder() -> None:
    # matched K=8 full split (MUSIC-AVQA n=9129, LLM-judge micro)
    labels = ["Audio-blind\n(base)", "Routed\n(text cue)",
              "Always-AV", "Oracle\n(per-item)"]
    vals = [66.1, 78.6, 81.2, 87.9]
    colors = ["#9ecae1", "#4292c6", "#08519c", "#6baed6"]
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    bars = ax.bar(labels, vals, color=colors, width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.5, f"{v:.1f}",
                ha="center", fontsize=8.5, fontweight="bold")
    # head-room bracket between always-AV and oracle
    ax.annotate("", xy=(3, 87.9), xytext=(3, 81.2),
                arrowprops=dict(arrowstyle="<->", color="crimson", lw=1.3))
    ax.text(3.05, 84.6, r"$+6.7$ pt" + "\nhead-room", color="crimson",
            fontsize=7.8, va="center", ha="left")
    ax.set_ylim(60, 92)
    ax.set_ylabel("Micro accuracy (%)")
    ax.set_title("Expert-routing head-room (MUSIC-AVQA)")
    fig.savefig(FIG / "fig_routing_ladder.pdf")
    plt.close(fig)
    print("wrote fig_routing_ladder.pdf")


if __name__ == "__main__":
    p0_heatmap()
    p3_ladder()
