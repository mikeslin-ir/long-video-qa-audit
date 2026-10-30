#!/usr/bin/env python3
"""Qualitative grid (Fig. 3): what tool the planner selects, and the benefit.

Six real cases, two per row, spanning the three routing axes across three
datasets. Each panel shows the agent's top-scored keyframe, the tool it
selected, and the resulting answer flip (baseline wrong -> tool-routed right).
All answers come from the actual judged / scored runs; frames are the
agent's own selected keyframes. No model calls.

Axes:
  SELECT  -- cascade keyframe selection      (Video-MME long)
  INJECT  -- OCR text / ASR narration        (EgoTextVQA, Video-MME)
  ROUTE   -- dispatch to an audio expert      (MUSIC-AVQA)
"""
from __future__ import annotations
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from PIL import Image

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 8.0, "font.family": "DejaVu Sans",
})

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = ROOT / "paper" / "figures"

# tool families -> (short tag, colour)
SELECT = ("SELECT", "#3182bd")
INJECT = ("INJECT", "#e6550d")
ROUTE = ("ROUTE", "#238b45")

# Each case: qdir, forced frame (or None=auto top-score), dataset, axis,
# tool label, question, (baseline label, baseline answer),
# (tool label, tool answer), gold.  Display answers are kept short.
# fail=True marks a failure case: the routed tool answer is *still* wrong;
# `note` then explains the failure mode.
CASES = [
    dict(qdir="trackB_long_cascade_k8_f0p5/00037_613-1", frame=None,
         dataset="Video-MME (long)", axis=SELECT, tool="cascade keyframes",
         q="What is the main idea of the video?",
         base=("uniform-K", "art & science talent"),
         good=("cascade", "the Mona Lisa"),
         gold="Mona Lisa",
         evidence=("SigLIP", "keyframe align 0.94 vs 0.45 median")),
    dict(qdir="trackB_long_cascade_k8_f0p5/00051_617-3", frame=None,
         dataset="Video-MME (long)", axis=SELECT, tool="cascade keyframes",
         q="Which can NOT be found in the Ellora Caves?",
         base=("uniform-K", "an elephant statue"),
         good=("cascade", "piles of stone"),
         gold="piles of stone",
         evidence=("SigLIP", "keyframe align 0.95 vs 0.84 median")),
    dict(qdir="wave9fix_qwen2vl_egotextvqa_cascade_injectOCR/"
               "00118_egotextvqa_outdoor_370_6.0", frame=None,
         dataset="EgoTextVQA", axis=INJECT, tool="OCR text",
         q="Where do I go to take the Major Deegan Expwy?",
         base=("no OCR", "right"),
         good=("+OCR", "exit 3"),
         gold="exit 3",
         evidence=("OCR", "\u201cmaj deegan \u00b7 exit \u00b7 3 \u00b7 harlem riv dr\u201d")),
    dict(qdir="trackB_long_cascade_k8_f0p5_injectASR/00007_603-1", frame=None,
         dataset="Video-MME (long)", axis=INJECT, tool="ASR narration",
         q="Which disease causes the most deaths?",
         base=("no ASR", "Malaria"),
         good=("+ASR", "Tuberculosis"),
         gold="Tuberculosis",
         evidence=("ASR", "\u201c\u2026killed more than malaria\u2026 we call it tuberculosis\u201d")),
    dict(qdir="wave9av_videollama2av_music_avqa_uniform/00841_00008161_4400",
         frame="frame_00000000.jpg",
         dataset="MUSIC-AVQA", axis=ROUTE, tool="audio expert",
         q="What is the first instrument that comes in?",
         base=("audio-blind VLM", "Clarinet"),
         good=("AV expert", "Bassoon"),
         gold="bassoon",
         evidence=("audio", "expert hears bassoon enter first")),
    dict(qdir="wave9av_videollama2av_music_avqa_uniform/00846_00005616_4441",
         frame="frame_00001028.jpg",
         dataset="MUSIC-AVQA", axis=ROUTE, tool="audio expert",
         q="Where is the first sounding instrument?",
         base=("audio-blind VLM", "on the right"),
         good=("AV expert", "left"),
         gold="left",
         evidence=("audio", "first onset localized to the left")),
    # ---- failure cases (routed tool still wrong) ----
    dict(qdir="wave9fix_qwen2vl_egotextvqa_cascade_injectOCR/"
               "00004_egotextvqa_outdoor_4260_12.0", frame=None,
         dataset="EgoTextVQA", axis=INJECT, tool="OCR text", fail=True,
         q="Which places on this street offer quick snacks?",
         base=("no OCR", "convenience store"),
         good=("+OCR", "Vision Express"),
         gold="Greggs",
         evidence=("OCR", "\u201cgreggs \u00b7 snack& \u00b7 sainsburys\u201d (garbled)"),
         note="Answer was in the OCR but too garbled; the VLM ignored it."),
    dict(qdir="wave9av_videollama2av_music_avqa_uniform/00004_00000100_35",
         frame=None,
         dataset="MUSIC-AVQA", axis=ROUTE, tool="audio expert", fail=True,
         q="How many sounding violins are in the video?",
         base=("audio-blind VLM", "6"),
         good=("AV expert", "Two"),
         gold="three",
         evidence=("audio", "resolves only 2 violin voices in the mix"),
         note="Expert under-counts overlapping same-timbre sources."),
]


def _is_dark(path: Path, thresh: float = 40.0) -> bool:
    im = Image.open(path).convert("L").resize((48, 48))
    return float(np.asarray(im).mean()) < thresh


def top_frame(case) -> Path:
    d = OUT / case["qdir"]
    if case["frame"]:
        return d / "selected_frames" / case["frame"]
    rj = None
    for name in ("cascade_result.json", "result.json"):
        if (d / name).exists():
            rj = json.load(open(d / name)); break
    sf = sorted(rj["selected_frames"], key=lambda x: x.get("fused_score", 0.0),
                reverse=True) if rj else []
    for s in sf:                                   # skip near-black fade frames
        p = d / "selected_frames" / Path(s["image_path"]).name
        if p.exists() and not _is_dark(p):
            return p
    return d / "selected_frames" / Path(sf[0]["image_path"]).name


def _mean_lum(path: Path) -> float:
    im = Image.open(path).convert("L").resize((48, 48))
    return float(np.asarray(im).mean())


def strip_frames(case, n=5):
    """`n` time-ordered frames for the strip, always including the agent's
    selected keyframe (highlighted). Pure-black fade frames are dropped, but
    dim frames are kept so every example shows the same number of photos;
    brightness only governs which frame is the highlighted keyframe."""
    d = OUT / case["qdir"] / "selected_frames"
    allf = sorted(d.glob("frame_*.jpg"))
    top = top_frame(case)
    pool = [f for f in allf if f == top or _mean_lum(f) >= 12.0] or allf
    if len(pool) <= n:
        chosen = list(pool)
    else:
        idx = [round(i * (len(pool) - 1) / (n - 1)) for i in range(n)]
        chosen = [pool[i] for i in sorted(set(idx))]
        if top not in chosen:                      # force-include the keyframe
            chosen[min(range(len(chosen)),
                       key=lambda k: abs(pool.index(chosen[k]) - pool.index(top)))] = top
            chosen = sorted(set(chosen), key=lambda f: pool.index(f))
    return [(f, f == top) for f in chosen]


def draw_panel(fig, sub, case):
    fail = case.get("fail", False)
    inner = sub.subgridspec(2, 1, height_ratios=[1.0, 1.32], hspace=0.12)
    tag, col = case["axis"]

    # ---- frame strip (more photos; selected keyframe highlighted) ----
    # The highlighted frame means different things per axis: SELECT actually
    # *selects* a keyframe; OCR reads text off that frame; ASR/audio evidence
    # comes from the time-axis soundtrack, so the frame is merely a sample.
    ev_kind = case["evidence"][0] if case.get("evidence") else None
    kf_label = {"OCR": "OCR'd frame",
                "ASR": "sampled frame",
                "audio": "sampled frame"}.get(ev_kind, "agent's keyframe")
    frames = strip_frames(case)
    strip = inner[0].subgridspec(1, len(frames), wspace=0.04)
    for j, (fp, is_top) in enumerate(frames):
        axi = fig.add_subplot(strip[0, j])
        axi.imshow(Image.open(fp).convert("RGB"), interpolation="lanczos")
        axi.set_xticks([]); axi.set_yticks([])
        ec = col if is_top else "#bdbdbd"
        lw = 2.1 if is_top else 0.7
        for s in axi.spines.values():
            s.set_edgecolor(ec); s.set_linewidth(lw)
        if is_top:
            axi.set_xlabel(kf_label, fontsize=5.6, color=col,
                           labelpad=1.5)

    # ---- text + answer boxes ----
    ax = fig.add_subplot(inner[1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    # tag pill + tool name
    ax.add_patch(FancyBboxPatch((0.0, 0.865), 0.21, 0.135,
                 boxstyle="round,pad=0.008", linewidth=0,
                 facecolor=col, transform=ax.transAxes, clip_on=False))
    ax.text(0.105, 0.932, tag, color="white", fontsize=7.0, fontweight="bold",
            ha="center", va="center", transform=ax.transAxes)
    ax.text(0.235, 0.932, case["tool"], color=col, fontsize=7.4,
            fontweight="bold", va="center", transform=ax.transAxes)
    # success / failure badge on the right
    if fail:
        ax.add_patch(FancyBboxPatch((0.80, 0.865), 0.20, 0.135,
                     boxstyle="round,pad=0.008", linewidth=0,
                     facecolor="#d94801", transform=ax.transAxes, clip_on=False))
        ax.text(0.90, 0.932, "FAILURE", color="white", fontsize=6.6,
                fontweight="bold", ha="center", va="center",
                transform=ax.transAxes)
    ax.text(1.0, 0.812, f"gold: {case['gold']}", color="#888", fontsize=6.4,
            style="italic", ha="right", va="center", transform=ax.transAxes)
    # question (always reserves 2 lines so boxes align across the grid)
    qy = 0.718
    for ln in textwrap.wrap("Q: " + case["q"], width=46)[:2]:
        ax.text(0.0, qy, ln, fontsize=7.2, va="center", transform=ax.transAxes)
        qy -= 0.082
    # injected evidence the agent fed the VLM (real OCR / audio / align read-out)
    ev = case.get("evidence")
    if ev:
        ev_kind, ev_text = ev
        ax.add_patch(FancyBboxPatch((0.0, 0.470), 0.125, 0.066,
                     boxstyle="round,pad=0.006", linewidth=0,
                     facecolor=col, transform=ax.transAxes, clip_on=False))
        ax.text(0.0625, 0.503, ev_kind, color="white", fontsize=5.6,
                fontweight="bold", ha="center", va="center",
                transform=ax.transAxes)
        ax.text(0.145, 0.503, ev_text, color="#333", fontsize=6.2,
                style="italic", va="center", transform=ax.transAxes)
    # baseline (wrong) / tool-routed boxes — fixed y for grid alignment
    bl, ba = case["base"]
    gl, ga = case["good"]
    base_y, good_y, box_h = 0.268, 0.130, 0.112
    ax.add_patch(FancyBboxPatch((0.0, base_y), 0.99, box_h,
                 boxstyle="round,pad=0.010", linewidth=1.0,
                 edgecolor="#cb181d", facecolor="#fde0df",
                 transform=ax.transAxes, clip_on=False))
    ax.text(0.022, base_y + 0.056, f"\u2717  {bl}:  {ba}", color="#7a0a0a",
            fontsize=7.2, va="center", transform=ax.transAxes)
    # tool-routed box: green (success) or amber-red still-wrong (failure)
    if fail:
        gec, gfc, gtc, mark, gfw = "#d94801", "#fee6ce", "#7a3000", "\u2717", "normal"
    else:
        gec, gfc, gtc, mark, gfw = "#238b45", "#d9f0e1", "#08512b", "\u2713", "bold"
    ax.add_patch(FancyBboxPatch((0.0, good_y), 0.99, box_h,
                 boxstyle="round,pad=0.010", linewidth=1.0,
                 edgecolor=gec, facecolor=gfc,
                 transform=ax.transAxes, clip_on=False))
    ax.text(0.022, good_y + 0.056, f"{mark}  {gl}:  {ga}", color=gtc,
            fontsize=7.2, fontweight=gfw, va="center", transform=ax.transAxes)
    # failure note (wrapped to stay within the panel)
    if case.get("note"):
        note = textwrap.wrap(case["note"], width=60)[:1]
        ax.text(0.0, 0.046, note[0], color="#d94801", fontsize=6.2,
                style="italic", va="center", transform=ax.transAxes)


def _render(cases, out, hspace=0.26, rowh=2.18, columns=2):
    rows = (len(cases) + columns - 1) // columns
    fig = plt.figure(figsize=(3.55 * columns, rowh * rows))
    outer = fig.add_gridspec(rows, columns, hspace=hspace, wspace=0.12,
                             left=0.012, right=0.988, top=0.99, bottom=0.012)
    for i, case in enumerate(cases):
        draw_panel(fig, outer[i // columns, i % columns], case)
    fig.savefig(out, dpi=400)
    fig.savefig(out.with_suffix(".svg"), dpi=400)
    plt.close(fig)
    print("wrote", out.name, f"({len(cases)} cases, {rows} rows)")


def main():
    # Submission figures use only datasets represented in the main paper.
    allowed = [CASES[i] for i in (0, 1, 3, 4, 5, 7)]
    _render(allowed, FIG / "fig_p2_qualitative.pdf")
    _render(allowed[:3], FIG / "fig_p2_qualitative_main.pdf",
        hspace=0.08, rowh=1.42, columns=1)
    _render(allowed[3:], FIG / "fig_p2_qualitative_supp.pdf",
        hspace=0.08, rowh=1.42, columns=1)


if __name__ == "__main__":
    main()
