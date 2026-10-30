"""Video-clustered re-analysis of the headline paired contrasts.

Video-MME long has three questions per video, so item-level resampling can
understate uncertainty. This re-runs the F1 intervals, the F2 subset
resampling and the F3(a) pooled test with the video as the resampling unit.
Post-processes per-item outcomes only (``audit_predictions``); runs no inference.

The first three contrasts of section A and sections B and C use the paper's
seed and random stream, so they replay exactly. Contrasts appended after them
(the remaining rows of the full-bucket table) draw from the same stream
afterwards and are reported for the supplementary table.

    python scripts/cluster_robustness.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import audit_predictions as ap  # noqa: E402

B = 10000
B_SUBSET = 20000
SEED = 20270101

VMME_CONTRASTS = [
    ("cascade vs uniform", "trackB_long_n900_k8_uniform", "trackB_long_n900_k8_cascade_f0p5"),
    ("cascade+ASR vs uniform", "trackB_long_n900_k8_uniform", "trackB_long_n900_k8_cascade_f0p5_injectASR"),
    ("planner vs always-ASR", "trackB_long_n900_k8_cascade_f0p5_injectASR",
     "trackB_long_n900_k8_llmPlanner_proceduralV3_injectOnly"),
]

# Remaining rows of the full-bucket table, drawn from their own random stream
# after the sections above so that those replay the paper's numbers exactly.
EXTRA_CONTRASTS = [
    ("AKS port vs uniform", "trackB_long_n900_k8_uniform", "trackB_long_n900_k8_aks"),
    ("KeyScore port vs uniform", "trackB_long_n900_k8_uniform", "trackB_long_n900_k8_keyscore"),
    ("VideoTree port vs uniform", "trackB_long_n900_k8_uniform", "trackB_long_n900_k8_videotree"),
    ("cascade+OCR vs uniform", "trackB_long_n900_k8_uniform", "trackB_long_n900_k8_cascade_f0p5_injectOCR"),
    ("planner vs uniform", "trackB_long_n900_k8_uniform", "trackB_long_n900_k8_llmPlanner_proceduralV3_injectOnly"),
    # evidence block placed before the question (post hoc placement variant)
    ("cascade+ASR, text before the question, vs uniform", "trackB_long_n900_k8_uniform",
     "trackB_long_n900_k8_cascade_f0p5_injectASR_before"),
    ("ASR before the question vs cascade without text", "trackB_long_n900_k8_cascade_f0p5",
     "trackB_long_n900_k8_cascade_f0p5_injectASR_before"),
    ("ASR before vs after the question (cascade)", "trackB_long_n900_k8_cascade_f0p5_injectASR",
     "trackB_long_n900_k8_cascade_f0p5_injectASR_before"),
]

POOLED = [
    ("Video-MME long", "trackB_long_n900_k8_uniform", "trackB_long_n900_k8_cascade_f0p5_injectASR", 200),
    ("XD-Violence", "trackB_xd_violence_uniform_k8", "trackB_xd_violence_cascade_k8_f0p5_injectASR", None),
    ("UCF-Crime", "trackB_ucf_crime_uniform_k8", "trackB_ucf_crime_cascade_k8_f0p5_injectASR", None),
]


def paired(base_run: str, treat_run: str, limit: int | None = None):
    base, treat = ap.load_correct(base_run), ap.load_correct(treat_run)
    qids = [q for q in base if q in treat]
    try:
        qids.sort(key=lambda q: (int(q.split("-")[0]), int(q.split("-")[1])))
    except (ValueError, IndexError):
        qids.sort()
    if limit:
        qids = qids[:limit]
    a = np.array([base[q] for q in qids])
    b = np.array([treat[q] for q in qids])
    vids = np.array([q.split("-")[0] for q in qids])
    return a, b, vids


def discordant(a, b):
    return int(np.sum((a == 0) & (b == 1))), int(np.sum((a == 1) & (b == 0)))


def mcnemar_p(a, b) -> float:
    n01, n10 = discordant(a, b)
    if n01 + n10 == 0:
        return 1.0
    return float(binomtest(n01, n01 + n10, 0.5).pvalue)


def cluster_index(vids):
    groups = defaultdict(list)
    for i, v in enumerate(vids):
        groups[v].append(i)
    return [np.array(ix) for ix in groups.values()]


def boot_ci(a, b, rng, clusters=None, levels=(2.5, 97.5, 5.0, 95.0)):
    n = len(a)
    deltas = np.empty(B)
    if clusters is None:
        for i in range(B):
            ix = rng.integers(0, n, n)
            deltas[i] = 100.0 * (b[ix].mean() - a[ix].mean())
    else:
        k = len(clusters)
        for i in range(B):
            pick = rng.integers(0, k, k)
            ix = np.concatenate([clusters[j] for j in pick])
            deltas[i] = 100.0 * (b[ix].mean() - a[ix].mean())
    return np.percentile(deltas, levels), deltas.std()


def tost(lo90, hi90, margin) -> str:
    return "PASS" if (lo90 > -margin and hi90 < margin) else "fail"


def report_contrast(name, base_run, treat_run, rng) -> float:
    a, b, vids = paired(base_run, treat_run)
    clusters = cluster_index(vids)
    d = 100.0 * (b.mean() - a.mean())
    (i25, i975, i5, i95), isd = boot_ci(a, b, rng)
    (c25, c975, c5, c95), csd = boot_ci(a, b, rng, clusters)
    nb, nc = discordant(a, b)
    print(f"\n--- {name}  (n={len(a)}, videos={len(clusters)}) ---")
    print(f"  delta {d:+.2f} pt   McNemar p={mcnemar_p(a, b):.4f}   b={nb} c={nc} b+c={nb + nc}")
    print(f"  item    95% CI [{i25:+.2f},{i975:+.2f}]  90% CI [{i5:+.2f},{i95:+.2f}]  sd={isd:.2f}")
    print(f"  cluster 95% CI [{c25:+.2f},{c975:+.2f}]  90% CI [{c5:+.2f},{c95:+.2f}]  sd={csd:.2f}")
    print(f"  standard-error ratio (clustered / item) = {csd / isd:.3f}  "
          f"(variance inflation {(csd / isd) ** 2:.3f})")
    for m in (3.0, 5.0):
        print(f"  TOST +-{m:.0f} pt : item {tost(i5, i95, m)} | cluster {tost(c5, c95, m)}")
    return csd / isd


def main() -> None:
    rng = np.random.default_rng(SEED)

    print("=" * 72)
    print("A. Video-MME long, n=900: item-level vs video-clustered intervals")
    print("=" * 72)
    ratios = [report_contrast(*c, rng) for c in VMME_CONTRASTS]
    print(f"\n  standard-error ratio over these contrasts: {min(ratios):.3f}-{max(ratios):.3f}")

    print()
    print("=" * 72)
    print("B. F2 subset resampling: item-level vs video-clustered draws")
    print("=" * 72)
    for name, base_run, treat_run in VMME_CONTRASTS[:2]:
        a, b, vids = paired(base_run, treat_run)
        clusters = cluster_index(vids)
        n, k = len(a), len(clusters)
        full = 100.0 * (b.mean() - a.mean())
        n_vid = round(200 / (n / k))

        d_item = np.empty(B_SUBSET)
        p_item = np.empty(B_SUBSET)
        for i in range(B_SUBSET):
            ix = rng.choice(n, size=200, replace=False)
            d_item[i] = 100.0 * (b[ix].mean() - a[ix].mean())
            p_item[i] = mcnemar_p(a[ix], b[ix])

        d_clu = np.empty(B_SUBSET)
        p_clu = np.empty(B_SUBSET)
        for i in range(B_SUBSET):
            pick = rng.choice(k, size=n_vid, replace=False)
            ix = np.concatenate([clusters[j] for j in pick])
            d_clu[i] = 100.0 * (b[ix].mean() - a[ix].mean())
            p_clu[i] = mcnemar_p(a[ix], b[ix])

        print(f"\n--- {name} (full {full:+.2f} pt; {n_vid} videos ~ {n / k * n_vid:.0f} items) ---")
        for tag, d, p in (("item-level ", d_item, p_item), ("video-clust", d_clu, p_clu)):
            lo, hi = np.percentile(d, [2.5, 97.5])
            print(f"  {tag}: 95% range [{lo:+.2f},{hi:+.2f}] width {hi - lo:.2f}  "
                  f"sign-flip {100.0 * np.mean(np.sign(d) != np.sign(full)):.1f}%  "
                  f"p<0.05 in {100.0 * np.mean(p < 0.05):.1f}%")

    print()
    print("=" * 72)
    print("C. F3(a) pooled stratified McNemar (cascade+ASR vs uniform-no-inject)")
    print("=" * 72)
    tot_b = tot_c = 0
    var = 0.0
    for name, base_run, treat_run, lim in POOLED:
        a, b, _ = paired(base_run, treat_run, lim)
        nb, nc = discordant(a, b)
        d = 100.0 * (b.mean() - a.mean())
        tot_b += nb
        tot_c += nc
        var += (nb + nc) / 4.0
        print(f"  {name:16s} n={len(a):4d}  delta {d:+5.2f}  b={nb:3d} c={nc:3d} "
              f"b+c={nb + nc:3d}  per-domain p={mcnemar_p(a, b):.4f}")
    mean = (tot_b + tot_c) / 2.0
    z = (abs(tot_b - mean) - 0.5) / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(z))
    print(f"  POOLED: b={tot_b} c={tot_c}  z={z:.3f}  stratified McNemar p={p:.4g}")

    print()
    print("=" * 72)
    print("D. Remaining full-bucket rows (own random stream)")
    print("=" * 72)
    rng_extra = np.random.default_rng(SEED + 1)
    extra = [report_contrast(*c, rng_extra) for c in EXTRA_CONTRASTS]
    print(f"\n  standard-error ratio over these contrasts: {min(extra):.3f}-{max(extra):.3f}")


if __name__ == "__main__":
    main()
