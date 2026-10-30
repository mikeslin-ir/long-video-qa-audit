#!/usr/bin/env python3
"""Aggregate the cross-backbone (LLaVA-NeXT-Video-7B) K-sweep.

Honest, pre-registered analysis. The pre-registered hypothesis (H1) is:

    The cross-family selection gain Delta = acc(cascade) - acc(uniform) is
    largest under a *tight frame budget* (small K) on *frame-sensitive*
    domains, and shrinks / vanishes as K grows (a weak backbone that already
    sees enough frames cannot benefit from better selection).

We report the *entire* sweep -- every (dataset, K) cell, including null and
negative ones -- and let the data speak. No cell is hidden. Delta_(cascade-uniform)
is computed on per-item paired correctness matched by qid, with an exact
two-sided McNemar p-value and a 10k percentile bootstrap 95% CI (via
scripts/stats_analysis.py).

Outputs:
  * stdout table (one row per dataset x K)
  * outputs/xbackbone_ksweep_table.tex   (LaTeX booktabs body)
  * outputs/xbackbone_ksweep_delta.pdf + .svg  (Delta-vs-K, dpi=600, vector)

Usage:
    python scripts/aggregate_xbackbone_ksweep.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import stats_analysis as st  # noqa: E402  (must import as a normal module)

OUT = st.OUT

# ---- run-directory layout ------------------------------------------------
# For each dataset, map K -> (uniform_dir, cascade_dir). K=8 lives in the
# xbackbone_lnv_* dirs; K in {2,4,16,32} in ksweep_lnv_* dirs. MLVU tasks are
# their own per-task sweeps (only K in {2,8} were launched).

def _ks(prefix: str, ks: list[int], k8_uniform: str | None, k8_cascade: str | None):
    d: dict[int, tuple[str, str]] = {}
    for k in ks:
        d[k] = (f"{prefix}_k{k}_uniform", f"{prefix}_k{k}_cascade")
    if k8_uniform and k8_cascade:
        d[8] = (k8_uniform, k8_cascade)
    return dict(sorted(d.items()))


DATASETS: dict[str, dict[int, tuple[str, str]]] = {
    "XD-Violence": _ks(
        "ksweep_lnv_xd_violence", [2, 4, 16, 32],
        "xbackbone_lnv_xd_violence_uniform", "xbackbone_lnv_xd_violence_cascade",
    ),
    "UCF-Crime": _ks(
        "ksweep_lnv_ucf_crime", [2, 4, 16, 32],
        "xbackbone_lnv_ucf_crime_uniform", "xbackbone_lnv_ucf_crime_cascade",
    ),
    "Video-MME (long, control)": _ks(
        "ksweep_lnv_video_mme", [4, 16],
        "xbackbone_lnv_uniform", "xbackbone_lnv_cascade",
    ),
    # MLVU frame-sensitive tasks (second cross-family witness). Only K in {2,8}.
    "MLVU findNeedle": {
        2: ("mlvu_ksweep_findNeedle_k2_uniform", "mlvu_ksweep_findNeedle_k2_cascade"),
        8: ("mlvu_ksweep_findNeedle_k8_uniform", "mlvu_ksweep_findNeedle_k8_cascade"),
    },
    "MLVU count": {
        2: ("mlvu_ksweep_count_k2_uniform", "mlvu_ksweep_count_k2_cascade"),
        8: ("mlvu_ksweep_count_k8_uniform", "mlvu_ksweep_count_k8_cascade"),
    },
    "MLVU order": {
        2: ("mlvu_ksweep_order_k2_uniform", "mlvu_ksweep_order_k2_cascade"),
        8: ("mlvu_ksweep_order_k8_uniform", "mlvu_ksweep_order_k8_cascade"),
    },
    "MLVU anomaly_reco": {
        2: ("mlvu_ksweep_anomaly_reco_k2_uniform", "mlvu_ksweep_anomaly_reco_k2_cascade"),
        8: ("mlvu_ksweep_anomaly_reco_k8_uniform", "mlvu_ksweep_anomaly_reco_k8_cascade"),
    },
}


def _exists(d: str) -> bool:
    return os.path.isfile(os.path.join(OUT, d, "rows.jsonl"))


def analyze() -> list[dict]:
    rows: list[dict] = []
    for ds, kmap in DATASETS.items():
        for k, (u_dir, c_dir) in kmap.items():
            if not (_exists(u_dir) and _exists(c_dir)):
                rows.append({"ds": ds, "k": k, "missing": True})
                continue
            a, b = st.pair(u_dir, c_dir)  # a=uniform, b=cascade
            n = len(a)
            acc_u, acc_c = a.mean() * 100, b.mean() * 100
            b01, b10, p = st.mcnemar_exact_p(a, b)
            delta, lo, hi = st.bootstrap_ci(a, b)
            rows.append({
                "ds": ds, "k": k, "missing": False, "n": n,
                "acc_u": acc_u, "acc_c": acc_c, "delta": delta,
                "lo": lo, "hi": hi, "p": p, "disc": b01 + b10,
            })
    return rows


def print_table(rows: list[dict]) -> None:
    hdr = f"{'dataset':<26} {'K':>3} {'n':>4} {'unif':>6} {'casc':>6} {'Δ':>7} {'95% CI':>16} {'p':>7} {'sig':>3}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r["missing"]:
            print(f"{r['ds']:<26} {r['k']:>3}  --- (pending) ---")
            continue
        sig = "***" if r["p"] < 0.01 else ("**" if r["p"] < 0.05 else ("*" if r["p"] < 0.1 else ""))
        ci = f"[{r['lo']:+.1f},{r['hi']:+.1f}]"
        print(f"{r['ds']:<26} {r['k']:>3} {r['n']:>4} {r['acc_u']:>5.1f}% {r['acc_c']:>5.1f}% "
              f"{r['delta']:>+6.1f} {ci:>16} {r['p']:>7.3f} {sig:>3}")


def write_latex(rows: list[dict]) -> None:
    path = os.path.join(OUT, "xbackbone_ksweep_table.tex")
    lines = []
    cur = None
    for r in rows:
        if r["missing"]:
            continue
        if r["ds"] != cur:
            cur = r["ds"]
            safe = cur.replace("_", r"\_")
            lines.append("\\midrule")
            lines.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{safe}}}}} \\\\")
        sig = "$^{\\ast\\ast\\ast}$" if r["p"] < 0.01 else ("$^{\\ast\\ast}$" if r["p"] < 0.05 else ("$^{\\ast}$" if r["p"] < 0.1 else ""))
        lines.append(
            f"\\quad $K{{=}}{r['k']}$ & {r['n']} & {r['acc_u']:.1f} & {r['acc_c']:.1f} & "
            f"${r['delta']:+.1f}${sig} & $[{r['lo']:+.1f},{r['hi']:+.1f}]$ & {r['p']:.3f} \\\\"
        )
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n[wrote] {path}")


def make_figure(rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # group by dataset
    by_ds: dict[str, list[dict]] = {}
    for r in rows:
        if r["missing"]:
            continue
        by_ds.setdefault(r["ds"], []).append(r)
    if not by_ds:
        print("[figure] no completed cells yet; skipping")
        return

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    for i, (ds, rs) in enumerate(by_ds.items()):
        rs = sorted(rs, key=lambda r: r["k"])
        ks = [r["k"] for r in rs]
        ds_delta = [r["delta"] for r in rs]
        yerr = [[r["delta"] - r["lo"] for r in rs], [r["hi"] - r["delta"] for r in rs]]
        ctrl = "control" in ds.lower()
        ax.errorbar(
            ks, ds_delta, yerr=yerr, marker=markers[i % len(markers)],
            capsize=3, lw=1.6 if not ctrl else 1.2,
            ls="--" if ctrl else "-", label=ds,
            color="0.5" if ctrl else None, zorder=2,
        )
        # star significant points
        for r in rs:
            if r["p"] < 0.05:
                ax.annotate("$\\ast$", (r["k"], r["delta"]), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=12)
    ax.axhline(0, color="k", lw=0.8, zorder=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16, 32])
    ax.set_xticklabels([2, 4, 8, 16, 32])
    ax.set_xlabel("frame budget $K$ (log scale)")
    ax.set_ylabel(r"$\Delta$ accuracy (cascade $-$ uniform), pp")
    ax.set_title("Cross-family selection gain vs.\\ frame budget\n(LLaVA-NeXT-Video-7B; $\\ast$: paired McNemar $p{<}0.05$)", fontsize=9)
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "svg"):
        p = os.path.join(OUT, f"xbackbone_ksweep_delta.{ext}")
        fig.savefig(p, dpi=600, bbox_inches="tight")
        print(f"[wrote] {p}")
    plt.close(fig)


if __name__ == "__main__":
    rows = analyze()
    print_table(rows)
    write_latex(rows)
    make_figure(rows)
