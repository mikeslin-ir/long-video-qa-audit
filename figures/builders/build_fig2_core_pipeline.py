"""Figure 2: VideoRouteAudit core evidence-routing pipeline."""
from __future__ import annotations

from pathlib import Path

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from f4p_style import PALETTE, FigureStyle, apply_publication_style, blank_canvas, finalize_figure, place_image

HERE = Path(__file__).resolve().parent
FRAMES = HERE / "fig2_materials" / "frames"

INK, SOFT, LIGHT = PALETTE["ink"], PALETTE["ink_soft"], PALETTE["ink_light"]
LINE, PAPER = PALETTE["neutral"], "#F5F5F4"
SELECT, INJECT, EXPERT, PLAN = PALETTE["blue_main"], "#C2660A", PALETTE["violet"], PALETTE["teal"]


def box(ax, x, y, w, h, title, detail="", color=LINE, fill="white", title_color=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0,rounding_size=0.014",
                                linewidth=1.05, edgecolor=color, facecolor=fill, zorder=2))
    ax.text(x + w / 2, y + h * (0.70 if detail else 0.5), title, ha="center", va="center",
            fontsize=6.2, color=title_color or color, fontweight="bold", zorder=4)
    if detail:
        ax.text(x + w / 2, y + h * 0.30, detail, ha="center", va="center", fontsize=4.75,
                color=SOFT, linespacing=1.05, zorder=4)


def arrow(ax, start, end, color=LIGHT, dashed=False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8,
                                 linewidth=1.0, color=color,
                                 linestyle="--" if dashed else "-", zorder=3,
                                 shrinkA=1.5, shrinkB=1.5))


def waveform(ax, x, y, w, h):
    bars = [0.18, 0.38, 0.73, 0.35, 0.90, 0.52, 0.24, 0.61, 0.37, 0.76, 0.47, 0.20]
    for i, amp in enumerate(bars):
        xpos = x + (i + 0.5) * w / len(bars)
        ax.plot([xpos, xpos], [y + h * (1 - amp) / 2, y + h * (1 + amp) / 2],
                color=INJECT, linewidth=0.8, solid_capstyle="round", zorder=4)


def text_icon(ax, x, y, w, h):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.004",
                                linewidth=0.8, edgecolor=INJECT, facecolor="white", zorder=3))
    for frac, end in [(0.72, 0.80), (0.50, 0.65), (0.28, 0.72)]:
        ax.plot([x + w * .17, x + w * end], [y + h * frac, y + h * frac], color=INJECT,
                linewidth=.65, zorder=4)


def build() -> None:
    apply_publication_style(FigureStyle(font_size=7, axes_linewidth=1.0))
    fig, ax = blank_canvas((7.0, 2.54))
    ax.text(.012, .967, "VideoRouteAudit routes evidence before one matched VLM call", fontsize=8.0,
            color=INK, fontweight="bold", va="top")
    ax.text(.012, .928, "Four visual stages choose frames; an independent text branch optionally augments the prompt.",
            fontsize=5.3, color=SOFT, va="top")

    # Real full-span decoded frames act as the candidate-pool input.
    paths = sorted(FRAMES.glob("select_frame_*.png"))
    shown = [paths[i] for i in [0, 2, 4, 7]]
    x0, fw, fh, gap = .014, .070, .100, .006
    for i, path in enumerate(shown):
        place_image(ax, path, x0 + i * (fw + gap), .655, fw, fh, border="white", lw=.55)
    ax.text(.170, .774, "video $V$: shared full-span candidate pool", fontsize=5.0,
            color=SOFT, ha="center", va="bottom")
    box(ax, .045, .510, .235, .070, "question $q$ + budget $K$", color=LINE, fill=PAPER, title_color=INK)

    # Planner is explicitly a controller, not an evidence stage.
    box(ax, .345, .805, .305, .082, "planner $\\pi$",
        "rule / LLM + procedural memory", color=PLAN,
        fill="#F2FAFA", title_color=PLAN)
    ax.text(.497, .792, "JSON plan", color=PLAN, fontsize=4.5, style="italic", ha="center")

    stages = [("PROPOSE", "$\\mathcal{C}_1$", "STACFP + stride"),
              ("FILTER", "$\\mathcal{C}_2$", "remove duplicates"),
              ("ALIGN", "$u_q(n)$", "question-conditioned score"),
              ("SELECT", "$\\mathcal{F}$", "submodular greedy, $|\\mathcal{F}|\\leq K$")]
    sx, sw, sh = .335, .088, .125
    for i, (title, symbol, detail) in enumerate(stages):
        x = sx + i * .103
        box(ax, x, .620, sw, sh, title, symbol, color=SELECT,
            fill="#F4F8FC", title_color=SELECT)
        ax.text(x + sw / 2, .602, detail, ha="center", va="top", fontsize=3.65,
            color=SOFT, linespacing=1.0)
        arrow(ax, (.280, .690), (x, .690), SELECT) if i == 0 else arrow(ax, (x - .013, .690), (x, .690), SELECT)
        arrow(ax, (x + sw / 2, .805), (x + sw / 2, .745), PLAN, dashed=True)

    # ASR/OCR are mutually optional sources that meet the visual path at injection.
    waveform(ax, .360, .455, .050, .040)
    ax.text(.420, .475, "ASR", color=INJECT, fontsize=5.3, fontweight="bold", va="center")
    text_icon(ax, .458, .445, .024, .055)
    ax.text(.491, .475, "OCR", color=INJECT, fontsize=5.3, fontweight="bold", va="center")
    box(ax, .548, .432, .126, .085, "INJECT", "none / ASR / OCR", color=INJECT,
        fill="#FFF8F1", title_color=INJECT)
    arrow(ax, (.518, .475), (.548, .475), INJECT)

    # Merge and reason once; expert branch deliberately separated as a model swap.
    box(ax, .765, .612, .090, .145, "PROMPT", "$\\mathcal{F}$ + $q\\oplus\\mathcal{E}$", color=INK,
        fill=PAPER, title_color=INK)
    arrow(ax, (.732, .690), (.765, .690), SELECT)
    arrow(ax, (.674, .475), (.810, .612), INJECT)
    box(ax, .875, .612, .090, .145, "VLM", "fixed backbone\nwithin contrast", color=INK,
        fill="white", title_color=INK)
    arrow(ax, (.855, .690), (.875, .690), INK)
    ax.text(.982, .690, "$\\hat a$", fontsize=8.4, color=INK, ha="center", va="center", fontweight="bold")
    arrow(ax, (.964, .690), (.976, .690), INK)
    box(ax, .875, .432, .090, .085, "AV expert", "F3b model swap", color=EXPERT,
        fill="#FBF6FA", title_color=EXPERT)
    arrow(ax, (.920, .612), (.920, .517), EXPERT, dashed=True)

    ax.plot([.012, .988], [.365, .365], color=LINE, linewidth=.85)
    ax.text(.012, .333, "Matched contrasts make an effect attributable", color=INK,
            fontsize=6.6, fontweight="bold", va="center")
    ax.text(.012, .297, "Within a contrast, the same questions, candidate pool, $K$, prompt scaffold, decoder and LLM judge are held fixed.",
            color=SOFT, fontsize=5.15, va="center")
    controls = [(SELECT, "SELECT", "selector rule"), (INJECT, "INJECT", "ASR / OCR source"),
                (EXPERT, "EXPERT", "answering model")]
    for i, (color, title, detail) in enumerate(controls):
        x = .195 + i * .235
        box(ax, x, .090, .205, .130, title, detail, color=color,
            fill="#F5F8FC" if i == 0 else "#FFF8F1" if i == 1 else "#FBF6FA", title_color=color)
        ax.text(x + .1025, .062, "one decision changes", fontsize=4.65, color=color,
                ha="center", va="center", style="italic")

    finalize_figure(fig, HERE / "fig2_pipeline", formats=["pdf", "png"], dpi=400, pad=.012)


if __name__ == "__main__":
    build()