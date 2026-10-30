"""Figure 4: what answering-model choice buys on MUSIC-AVQA, and how much of it is audio.

A dumbbell chart per question type on the type-stratified subset of the split (200
questions per type, K=8). Open circle: audio-blind Qwen2-VL-7B. Filled square: the
audio-visual expert VideoLLaMA2.1-7B-AV. Open diamond: the same expert with its
soundtrack replaced by digital silence. All three arms answer the same questions and
are graded by one grader (``--grader``, default the in-loop matcher), because the LLM
judge accepts paraphrases the matcher rejects and mixing the two would bias the
comparison against the silenced arm. The numbers come from
``results/audio_ablation/arms.csv.gz``, so the figure cannot drift from
``scripts/audio_ablation_analysis.py``.

    python figures/builders/build_fig4_expert.py [--out-dir figures] [--grader judge|inloop]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "scripts")]

import audio_ablation_analysis as aa  # noqa: E402
from f4p_style import PALETTE, FigureStyle, apply_publication_style, finalize_figure  # noqa: E402

ORDER = [
    "Visual-Counting", "Visual-Location", "Audio-Counting", "Audio-Comparative",
    "Audio-Visual-Existential", "Audio-Visual-Counting", "Audio-Visual-Comparative",
    "Audio-Visual-Location", "Audio-Visual-Temporal",
]

INK, SOFT = PALETTE["ink"], PALETTE["ink_soft"]
NEUTRAL = "#A9A9A9"
EXPERT_COLOR = PALETTE["blue_main"]
HIGHLIGHT = PALETTE["red_strong"]


def collect(grader: str) -> dict:
    """Per-type accuracies (%) of the three arms, and the micro-average post-stratified to the full split."""
    table = aa.read_arms()
    if table is None:
        raise SystemExit("results/audio_ablation/arms.csv.gz is missing; run "
                         "scripts/audio_ablation_analysis.py --outputs <raw outputs> --write-arms")
    qids, arms, types = aa.arm_arrays(table, grader)
    if len(qids) < len(table):
        raise SystemExit(f"{len(table) - len(qids)} questions have no {grader} verdict for some arm")
    types = np.array(types)
    out = {"types": ORDER, "n": [int((types == t).sum()) for t in ORDER]}
    for arm in aa.ARM_KEYS:
        out[arm] = [100 * float(arms[arm][types == t].mean()) for t in ORDER]
    with contextlib.redirect_stdout(io.StringIO()):
        pooled = aa.ablation(table, grader)["micro_poststratified"]["accuracy"]
    out["micro_blind"], out["micro_av"], out["micro_silent"] = pooled["blind"], pooled["av"], pooled["silent"]
    return out


def build(data: dict, out_dir: Path) -> None:
    labels = [t.replace("Audio-Visual-", "AV-") for t in data["types"]]
    blind, expert, silent = (np.array(data[k], dtype=float) for k in ("blind", "av", "silent"))
    hilite = len(labels) - 1

    apply_publication_style(FigureStyle(font_size=7, axes_linewidth=0.8))
    fig, ax = plt.subplots(figsize=(3.35, 2.62))
    fig.subplots_adjust(left=0.27, right=0.97, top=0.84, bottom=0.15)

    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=4.8, markerfacecolor="white",
               markeredgecolor=NEUTRAL, markeredgewidth=1.15),
        Line2D([], [], marker="s", linestyle="none", markersize=4.8, markerfacecolor=EXPERT_COLOR,
               markeredgecolor="white", markeredgewidth=0.45),
        Line2D([], [], marker="D", linestyle="none", markersize=3.6, markerfacecolor="white",
               markeredgecolor=EXPERT_COLOR, markeredgewidth=1.0),
    ]
    names = ["Audio-blind", "AV expert", "AV, silenced"]
    fig.legend(handles, names, loc="upper left", bbox_to_anchor=(0.27, 0.985), ncol=len(names),
               handletextpad=0.3, columnspacing=0.7, borderaxespad=0, fontsize=6.2)
    fig.text(0.97, 0.925, rf"Micro: {data['micro_blind']:.1f} $\rightarrow$ {data['micro_av']:.1f}  "
             rf"($+{data['micro_av'] - data['micro_blind']:.1f}$)",
             ha="right", va="center", fontsize=6.0, color=SOFT)

    y = np.arange(len(labels))
    ax.axhspan(3.55, 8.48, color=EXPERT_COLOR, alpha=0.055, zorder=0)
    ax.axhspan(7.52, 8.48, color=HIGHLIGHT, alpha=0.09, zorder=0)
    ax.axhline(3.5, color=PALETTE["neutral"], linewidth=0.8, zorder=1)

    for i, (base, routed, quiet) in enumerate(zip(blind, expert, silent)):
        color = HIGHLIGHT if i == hilite else EXPERT_COLOR
        ax.plot([base, routed], [i, i], color=color, linewidth=2.4 if i == hilite else 1.5,
                solid_capstyle="round", alpha=0.95, zorder=2)
        ax.scatter(base, i, s=27, color="white", edgecolor=NEUTRAL, linewidth=1.15, zorder=3)
        ax.scatter(routed, i, s=31, marker="s", color=color, edgecolor="white", linewidth=0.45, zorder=4)
        ax.scatter(quiet, i + 0.24, s=15, marker="D", facecolor="white", edgecolor=color,
                   linewidth=0.9, zorder=5)
        ax.text(max(routed, quiet) + 1.5, i, f"+{routed - base:.1f}", color=color, fontsize=6.0, va="center",
                ha="left", fontweight="bold" if i == hilite else "normal")

    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    for tick, i in zip(ax.get_yticklabels(), range(len(labels))):
        if i == hilite:
            tick.set_color(HIGHLIGHT)
            tick.set_fontweight("bold")

    ax.set_xticks([40, 60, 80, 100])
    ax.set_xlim(30, 100)
    ax.set_xlabel("Accuracy (%)", color=SOFT, labelpad=3)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.65, linestyle=":", zorder=0)
    ax.tick_params(axis="both", length=0, labelsize=6.5)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(PALETTE["neutral"])
    finalize_figure(fig, out_dir / "fig_f4_expert", formats=["pdf", "png"], dpi=400, pad=0.015)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=HERE.parent, help="where fig_f4_expert.{pdf,png} are written")
    parser.add_argument("--grader", choices=["judge", "inloop"], default="judge")
    args = parser.parse_args()
    data = collect(args.grader)
    for name, n, b, a, s in zip(data["types"], data["n"], data["blind"], data["av"], data["silent"]):
        print(f"{name:<26}n={n:>4}  blind {b:5.1f}  silent {s:5.1f}  AV {a:5.1f}")
    print(f"micro (post-stratified): blind {data['micro_blind']:.1f}  silent {data['micro_silent']:.1f}  "
          f"AV {data['micro_av']:.1f}")
    build(data, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
