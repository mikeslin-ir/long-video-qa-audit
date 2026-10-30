#!/usr/bin/env python
"""Generate the K-sweep accuracy-vs-budget figure for the audit paper.

Reads per-item correctness from the Video-MME long K-sweep runs, computes
accuracy with Wilson 95% CIs, runs a paired exact McNemar test (cascade vs
uniform) at each K, and renders a publication-quality PDF.

Output: paper/figures/ksweep.pdf
"""
import json
import os
from math import comb, sqrt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KS = [1, 2, 4, 6, 8]


def load(run):
    path = os.path.join(ROOT, "outputs", run, "rows.jsonl")
    m = {}
    with open(path) as f:
        for ln in f:
            r = json.loads(ln)
            m[r["qid"]] = bool(r["correct"])
    return m


def wilson(p, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * p)


uni_acc, uni_lo, uni_hi = [], [], []
cas_acc, cas_lo, cas_hi = [], [], []
pvals = []

for K in KS:
    u = load(f"trackB_long_uniform_k{K}")
    c = load(f"trackB_long_cascade_k{K}_f0p5")
    qs = sorted(set(u) & set(c))
    n = len(qs)
    ua = sum(u[q] for q in qs) / n
    ca = sum(c[q] for q in qs) / n
    ulo, uhi = wilson(ua, n)
    clo, chi = wilson(ca, n)
    b = sum(1 for q in qs if c[q] and not u[q])
    cc = sum(1 for q in qs if u[q] and not c[q])
    p = mcnemar_exact(b, cc)
    uni_acc.append(ua * 100); uni_lo.append(ulo * 100); uni_hi.append(uhi * 100)
    cas_acc.append(ca * 100); cas_lo.append(clo * 100); cas_hi.append(chi * 100)
    pvals.append(p)
    print(f"K={K} n={n} uniform={ua:.3f} cascade={ca:.3f} p={p:.3f}")

x = list(range(len(KS)))
fig, ax = plt.subplots(figsize=(3.4, 2.6))

cas_err = [[cas_acc[i] - cas_lo[i] for i in x], [cas_hi[i] - cas_acc[i] for i in x]]
uni_err = [[uni_acc[i] - uni_lo[i] for i in x], [uni_hi[i] - uni_acc[i] for i in x]]

ax.errorbar(x, cas_acc, yerr=cas_err, marker="o", ms=4, lw=1.6, capsize=2.5,
            color="#c8521b", label="Audited cascade", zorder=3)
ax.errorbar(x, uni_acc, yerr=uni_err, marker="s", ms=3.5, lw=1.4, capsize=2.5,
            color="#3b6fb0", label="Uniform-$K$", zorder=2)

# significance star where p < 0.05 (put above the higher of the two points)
for i, p in enumerate(pvals):
    if p < 0.05:
        ytop = max(cas_hi[i], uni_hi[i])
        ax.annotate("$*$", (x[i], ytop + 1.2), ha="center", va="bottom",
                    fontsize=11, color="black")

ax.set_xticks(x)
ax.set_xticklabels([str(k) for k in KS])
ax.set_xlabel("Frame budget $K$")
ax.set_ylabel("Accuracy (\\%)")
ax.set_title("Video-MME long ($n{=}200$)", fontsize=9)
ax.grid(True, axis="y", ls=":", lw=0.5, alpha=0.6)
ax.legend(fontsize=7.5, loc="lower right", frameon=False)
ax.set_ylim(34, 58)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

fig.tight_layout(pad=0.3)
out = os.path.join(ROOT, "paper", "figures", "ksweep.pdf")
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
