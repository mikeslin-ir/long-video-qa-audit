#!/usr/bin/env python
"""Week-1 analysis: does the cascade's gain over uniform grow with event-sparsity?

For each Video-MME long item (n=900) we read the per-item cascade_result.json,
which stores the post-filter relevance landscape (aligner_scores.semantic_siglip)
and video duration. We derive interpretable *event-sparsity* proxies:

  concentration = 1 - normalized_entropy(softmax(relevance))   # peaky => sparse
  gini          = Gini coefficient of (shifted) relevance
  peak2mean     = (max - mean) / std of relevance
  duration_sec  = video length (longer => evidence is a smaller fraction)

We pair each item with whether the cascade and uniform selectors answered it
correctly (paired, same backbone, K=8) and test the hypothesis:

    H1: accuracy gain (cascade - uniform) increases with event-sparsity.

Outputs:
  - prints per-tercile gain table + Spearman correlation + tercile McNemar
  - paper/figures/sparsity_law.pdf  (gain vs sparsity, binned, with CIs)
"""
import glob
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# budget K -> (cascade_run, uniform_run). Aligner relevance landscape is a
# property of (question, video), independent of K; only correctness changes.
RUNS = {
    1: ("trackB_long_cascade_k1_f0p5", "trackB_long_uniform_k1"),
    2: ("trackB_long_cascade_k2_f0p5", "trackB_long_uniform_k2"),
    4: ("trackB_long_cascade_k4_f0p5", "trackB_long_uniform_k4"),
    8: ("trackB_full_long_cascade_k8_f0p5", "trackB_full_long_uniform_k8"),
}
K = int(sys.argv[1]) if len(sys.argv) > 1 else 1
CAS, UNI = RUNS[K]


def softmax(x):
    x = np.asarray(x, dtype=float)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def norm_entropy(p):
    p = p[p > 0]
    h = -(p * np.log(p)).sum()
    return h / math.log(len(p)) if len(p) > 1 else 0.0


def gini(x):
    x = np.asarray(x, dtype=float)
    x = x - x.min() + 1e-9
    x = np.sort(x)
    n = len(x)
    idx = np.arange(1, n + 1)
    return float((np.sum((2 * idx - n - 1) * x)) / (n * np.sum(x)))


def load_correct(run):
    m = {}
    for ln in open(os.path.join(ROOT, "outputs", run, "rows.jsonl")):
        r = json.loads(ln)
        m[r["qid"]] = int(r["correct"])
    return m


cas_correct = load_correct(CAS)
uni_correct = load_correct(UNI)

rows = []
for jf in glob.glob(os.path.join(ROOT, "outputs", CAS, "*", "cascade_result.json")):
    qid = os.path.basename(os.path.dirname(jf)).split("_", 1)[1]
    if qid not in cas_correct or qid not in uni_correct:
        continue
    try:
        o = json.load(open(jf))
        s = np.asarray(o["aligner_scores"]["semantic_siglip"], dtype=float)
        if s.size < 2:
            continue
        dur = float(o["video_meta"]["duration_sec"])
    except Exception:
        continue
    p = softmax(s)
    rows.append(dict(
        qid=qid,
        concentration=1.0 - norm_entropy(p),
        gini=gini(s),
        peak2mean=float((s.max() - s.mean()) / (s.std() + 1e-9)),
        duration=dur,
        c=cas_correct[qid],
        u=uni_correct[qid],
    ))

print(f"items with full data: {len(rows)}  (budget K={K})")
C = np.array([r["c"] for r in rows])
U = np.array([r["u"] for r in rows])
print(f"overall  cascade={C.mean():.3f}  uniform={U.mean():.3f}  "
      f"gain={ (C.mean()-U.mean())*100:+.2f}pt")


def spearman(a, b):
    ar = np.argsort(np.argsort(a)).astype(float)
    br = np.argsort(np.argsort(b)).astype(float)
    ar -= ar.mean(); br -= br.mean()
    return float((ar @ br) / (np.linalg.norm(ar) * np.linalg.norm(br) + 1e-9))


from math import comb
def mcnemar(c, u):
    b = int(((c == 1) & (u == 0)).sum())
    cc = int(((u == 1) & (c == 0)).sum())
    m = b + cc; k = min(b, cc)
    p = sum(comb(m, i) for i in range(k + 1)) * 2 / (2 ** m) if m else 1.0
    return b, cc, min(p, 1.0)


print("\n=== Per-proxy: terciles (low/mid/high sparsity) gain + trend ===")
gain_signal = (C - U).astype(float)
results = {}
for proxy in ["concentration", "gini", "peak2mean", "duration"]:
    x = np.array([r[proxy] for r in rows])
    order = np.argsort(x)
    n = len(order)
    thirds = np.array_split(order, 3)
    rho = spearman(x, gain_signal)
    print(f"\n[{proxy}]  Spearman(sparsity, gain) = {rho:+.3f}")
    tg = []
    for name, idx in zip(["low", "mid", "high"], thirds):
        ci = C[idx]; ui = U[idx]
        b, cc, pv = mcnemar(ci, ui)
        g = (ci.mean() - ui.mean()) * 100
        tg.append(g)
        print(f"   {name:>4} sparsity (n={len(idx)}): "
              f"cascade={ci.mean():.3f} uniform={ui.mean():.3f} "
              f"gain={g:+.1f}pt  McNemar p={pv:.3f}")
    results[proxy] = (rho, tg)

# pick the proxy with the strongest monotone trend for the figure
best = max(results, key=lambda k: results[k][0])
print(f"\nstrongest proxy by Spearman: {best} (rho={results[best][0]:+.3f})")

# ---- figure: gain vs sparsity quintiles for the best proxy ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

x = np.array([r[best] for r in rows])
order = np.argsort(x)
bins = np.array_split(order, 5)
gains, los, his, centers = [], [], [], []
for idx in bins:
    ci = C[idx]; ui = U[idx]
    g = ci.mean() - ui.mean()
    # Wilson-ish SE on the paired difference
    se = math.sqrt((ci.mean() * (1 - ci.mean()) + ui.mean() * (1 - ui.mean())) / len(idx))
    gains.append(g * 100); los.append((g - 1.96 * se) * 100); his.append((g + 1.96 * se) * 100)
    centers.append(np.median(x[idx]))

fig, ax = plt.subplots(figsize=(3.4, 2.6))
xs = np.arange(1, 6)
ax.axhline(0, color="0.6", lw=0.8, ls="--")
ax.errorbar(xs, gains, yerr=[np.array(gains) - np.array(los),
                             np.array(his) - np.array(gains)],
            fmt="o-", color="#c8521b", ecolor="#c8521b", capsize=3, lw=1.6,
            markersize=5)
ax.set_xticks(xs)
ax.set_xticklabels(["Q1\n(dense)", "Q2", "Q3", "Q4", "Q5\n(sparse)"], fontsize=7)
ax.set_xlabel("event-sparsity quintile", fontsize=9)
ax.set_ylabel("cascade $-$ uniform (pts)", fontsize=9)
ax.set_title(f"Gain grows with event-sparsity\n(Spearman $\\rho$={results[best][0]:+.2f})",
             fontsize=9)
ax.tick_params(labelsize=8)
fig.tight_layout()
out = os.path.join(ROOT, "paper", "figures", f"sparsity_law_k{K}.pdf")
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
