#!/usr/bin/env python3
"""Pooled paired-significance analysis across event-sparse datasets.

Reviewer ask (priority 2): the headline F2 super-additivity is +6.5pt at
p=0.066 on Video-MME long (n=200) alone. Pool the matched event-sparse
items across Video-MME long / XD-Violence / UCF-Crime and re-run the
paired tests to see whether the effect crosses p<0.05.

This script reads cached per-item predictions (rows.jsonl) only; it runs
no models and fabricates nothing. Each contrast pairs by `qid` *within*
each dataset (so the McNemar pairing is exact and same-item), then pools
the per-item discordant pairs across datasets for the combined test.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

# Matched n=200, K=8 runs per event-sparse dataset.
RUNS = {
    "video_mme": {
        "uniform":      "trackB_long_uniform_k8",
        "uniform+ASR":  "trackB_long_uniform_k8_injectASR",
        "cascade":      "trackB_long_cascade_k8_f0p5",
        "cascade+ASR":  "trackB_long_cascade_k8_f0p5_injectASR",
        "planner":      "trackB_long_cascade_k8_llmPlanner_planInject_proceduralV3_injectOnly",
    },
    "xd_violence": {
        "uniform":      "trackB_xd_violence_uniform_k8",
        "cascade":      "trackB_xd_violence_cascade_k8_f0p5",
        "cascade+ASR":  "trackB_xd_violence_cascade_k8_f0p5_injectASR",
        "planner":      "trackB_xd_violence_cascade_k8_llmPlanner_planInject_proceduralV1",
    },
    "ucf_crime": {
        "uniform":      "trackB_ucf_crime_uniform_k8",
        "cascade":      "trackB_ucf_crime_cascade_k8_f0p5",
        "cascade+ASR":  "trackB_ucf_crime_cascade_k8_f0p5_injectASR",
        "planner":      "trackB_ucf_crime_cascade_k8_llmPlanner_planInject_proceduralV1",
    },
}


def load_correct(run: str) -> dict:
    """qid -> bool correct, from a run's rows.jsonl."""
    p = OUT / run / "rows.jsonl"
    out = {}
    with open(p) as fh:
        for line in fh:
            r = json.loads(line)
            out[r["qid"]] = bool(r["correct"])
    return out


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on discordant counts (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # two-sided exact binomial test, p=0.5
    return float(stats.binomtest(k, n, 0.5, alternative="two-sided").pvalue)


def stratified_mcnemar(strata):
    """Stratified McNemar / Cochran-Mantel-Haenszel for matched pairs.

    strata: list of (b, c) discordant counts per dataset. Combines the
    within-stratum discordances so the test is immune to Simpson's paradox
    from pooling strata with different base rates. Statistic
    M = (sum(c-b))^2 / sum(b+c)  ~  chi^2_1.
    """
    num = sum(c - b for b, c in strata)
    den = sum(b + c for b, c in strata)
    if den == 0:
        return float("nan"), 1.0
    chi2 = num * num / den
    p = float(stats.chi2.sf(chi2, df=1))
    return chi2, p


def bootstrap_ci(pairs, reps=10000, seed=0):
    """95% bootstrap CI of the paired accuracy delta (B - A) in points."""
    rng = np.random.default_rng(seed)
    a = np.array([x[0] for x in pairs], dtype=float)
    b = np.array([x[1] for x in pairs], dtype=float)
    n = len(pairs)
    idx = rng.integers(0, n, size=(reps, n))
    deltas = (b[idx].mean(1) - a[idx].mean(1)) * 100.0
    return np.percentile(deltas, 2.5), np.percentile(deltas, 97.5)


def contrast(name_a, name_b, datasets):
    """Pair method A vs B by qid within each dataset, then pool."""
    per_ds = {}
    pooled_pairs = []
    for ds in datasets:
        runs = RUNS[ds]
        if name_a not in runs or name_b not in runs:
            continue
        ca = load_correct(runs[name_a])
        cb = load_correct(runs[name_b])
        shared = sorted(set(ca) & set(cb))
        pairs = [(ca[q], cb[q]) for q in shared]
        per_ds[ds] = pairs
        pooled_pairs.extend(pairs)
    return per_ds, pooled_pairs


def summarize(label, pairs):
    n = len(pairs)
    if n == 0:
        return None
    acc_a = sum(p[0] for p in pairs) / n * 100
    acc_b = sum(p[1] for p in pairs) / n * 100
    delta = acc_b - acc_a
    # discordant: b=A-correct/B-wrong, c=A-wrong/B-correct
    b = sum(1 for x in pairs if x[0] and not x[1])
    c = sum(1 for x in pairs if not x[0] and x[1])
    p = exact_mcnemar(b, c)
    lo, hi = bootstrap_ci(pairs)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    print(f"  {label:<26} n={n:<4} {acc_a:5.1f} -> {acc_b:5.1f}  "
          f"d={delta:+5.1f}pt  CI[{lo:+.1f},{hi:+.1f}]  "
          f"discord(b={b},c={c})  p={p:.4f} {sig}")
    return dict(n=n, acc_a=acc_a, acc_b=acc_b, delta=delta, ci=(lo, hi), p=p)


def run_contrast(title, a, b, datasets=("video_mme", "xd_violence", "ucf_crime")):
    print(f"\n=== {title}:  {b}  vs  {a} ===")
    per_ds, pooled = contrast(a, b, datasets)
    strata = []
    for ds, pairs in per_ds.items():
        summarize(ds, pairs)
        bb = sum(1 for x in pairs if x[0] and not x[1])
        cc = sum(1 for x in pairs if not x[0] and x[1])
        strata.append((bb, cc))
    print("  " + "-" * 70)
    res = summarize("POOLED (naive McNemar)", pooled)
    chi2, p = stratified_mcnemar(strata)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    print(f"  {'STRATIFIED McNemar (CMH)':<26} "
          f"chi2(1)={chi2:.2f}  p={p:.4f} {sig}")
    return res


if __name__ == "__main__":
    print("Pooled paired-significance analysis (cached predictions only)")
    print("=" * 78)

    # F2: injection effect — cascade+ASR vs uniform (the +6.5pt headline)
    run_contrast("F2a  cascade+ASR vs uniform", "uniform", "cascade+ASR")
    # F2: injection on cascade — cascade+ASR vs cascade (the inject delta)
    run_contrast("F2b  inject delta (cascade+ASR vs cascade)", "cascade", "cascade+ASR")
    # F1: selection — cascade vs uniform
    run_contrast("F1   cascade vs uniform", "uniform", "cascade")
    # F3: planner vs cascade+ASR (always-inject)
    run_contrast("F3   planner vs cascade+ASR", "cascade+ASR", "planner")
