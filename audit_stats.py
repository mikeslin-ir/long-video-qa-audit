"""Paired-comparison helpers for the review-response analyses.

Everything operates on per-item 0/1 correctness vectors that are already paired
(same items, same order). ``gain`` is the number of items the variant answers
correctly and the baseline does not; ``loss`` is the reverse.

The original battery in ``scripts/stats_analysis.py`` keeps its own helpers and
its own random stream so that the intervals printed in the paper replay
bit-for-bit; the helpers here are used by the analyses added afterwards.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats


def discordant(base, variant) -> tuple[int, int]:
    """(gain, loss): variant right / baseline wrong, and the reverse."""
    a = np.asarray(base, dtype=int)
    b = np.asarray(variant, dtype=int)
    return int(np.sum((a == 0) & (b == 1))), int(np.sum((a == 1) & (b == 0)))


def mcnemar_exact_p(gain: int, loss: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts."""
    n = gain + loss
    if n == 0:
        return 1.0
    return float(stats.binomtest(min(gain, loss), n, 0.5, alternative="two-sided").pvalue)


def mcnemar_z(gain: int, loss: int) -> tuple[float, float]:
    """Normal-approximation McNemar statistic Z = (gain - loss) / sqrt(gain + loss)."""
    n = gain + loss
    if n == 0:
        return 0.0, 1.0
    z = (gain - loss) / math.sqrt(n)
    return z, float(2 * stats.norm.sf(abs(z)))


def paired_bootstrap_ci(diff, reps: int = 10_000, seed: int = 0, level: float = 0.95):
    """Percentile CI, in points, of the mean of per-item paired differences."""
    d = np.asarray(diff, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(reps, len(d)))
    boots = d[idx].mean(axis=1) * 100.0
    lo, hi = np.percentile(boots, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(d.mean() * 100.0), float(lo), float(hi)


def stratified_bootstrap_ci(strata, reps: int = 10_000, seed: int = 0, level: float = 0.95):
    """Item-weighted mean of paired differences, in points, with items resampled
    *within* each stratum (domain), so the stratum sizes stay fixed."""
    arrays = [np.asarray(d, dtype=float) for d in strata]
    n = sum(len(d) for d in arrays)
    rng = np.random.default_rng(seed)
    totals = np.zeros(reps)
    for d in arrays:
        idx = rng.integers(0, len(d), size=(reps, len(d)))
        totals += d[idx].sum(axis=1)
    boots = totals / n * 100.0
    lo, hi = np.percentile(boots, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(sum(d.sum() for d in arrays) / n * 100.0), float(lo), float(hi)


@dataclass
class DomainWeighted:
    delta: float
    lo: float
    hi: float
    p_permutation: float
    p_normal: float
    weights: dict[str, float]


def domain_weighted_delta(
    diffs: dict[str, np.ndarray],
    weights: dict[str, float] | None = None,
    reps: int = 20_000,
    seed: int = 0,
) -> DomainWeighted:
    """Weighted mean of per-domain paired differences, in points.

    Each domain's mean difference is weighted by ``weights`` (equal by default),
    regardless of how many items the domain contributes. The interval resamples
    items within each domain; the permutation p-value flips the sign of each
    item's paired difference (the randomisation test for paired data), and
    ``p_normal`` is the Gaussian approximation from the bootstrap standard error.
    """
    names = sorted(diffs)
    w = weights or {k: 1.0 / len(names) for k in names}
    total = sum(w[k] for k in names)
    w = {k: w[k] / total for k in names}
    arrays = {k: np.asarray(diffs[k], dtype=float) for k in names}
    rng = np.random.default_rng(seed)

    def weighted(parts: dict[str, np.ndarray]) -> float:
        return sum(w[k] * parts[k].mean() for k in names) * 100.0

    observed = weighted(arrays)
    boots = np.empty(reps)
    flips = np.empty(reps)
    for i in range(reps):
        boots[i] = weighted({k: arrays[k][rng.integers(0, len(arrays[k]), len(arrays[k]))] for k in names})
        flips[i] = weighted({k: arrays[k] * rng.choice((-1.0, 1.0), size=len(arrays[k])) for k in names})
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_perm = (1 + int(np.sum(np.abs(flips) >= abs(observed) - 1e-12))) / (reps + 1)
    se = float(boots.std(ddof=1))
    p_norm = float(2 * stats.norm.sf(abs(observed) / se)) if se > 0 else 1.0
    return DomainWeighted(float(observed), float(lo), float(hi), float(p_perm), p_norm, w)
