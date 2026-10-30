"""Repeated-subsampling stability of paired deltas on the Video-MME long bucket.

Post-processes per-item outcomes only (``audit_predictions``); runs no inference.
Answers: how much of the subset-vs-full-bucket gap is sampling variability
rather than n? The random stream and seed are those of the paper's analysis, so
the printed percentiles and sign-flip rates replay exactly.

    python scripts/subset_stability.py [--out results/subset_stability_cache.npz]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import audit_predictions as ap  # noqa: E402

CONTRASTS = [
    ("cascade vs uniform", "trackB_long_n900_k8_uniform", "trackB_long_n900_k8_cascade_f0p5"),
    ("cascade+ASR vs uniform", "trackB_long_n900_k8_uniform", "trackB_long_n900_k8_cascade_f0p5_injectASR"),
]
SUBSET_N = 200
B = 20000
SEED = 20270101


def discordant(a: np.ndarray, b: np.ndarray) -> tuple[int, int]:
    return (int(np.sum((a == 0) & (b == 1))), int(np.sum((a == 1) & (b == 0))))


def mcnemar_p(a: np.ndarray, b: np.ndarray) -> float:
    n01, n10 = discordant(a, b)
    if n01 + n10 == 0:
        return 1.0
    return float(binomtest(n01, n01 + n10, 0.5).pvalue)


def mde_paired(a: np.ndarray, b: np.ndarray, power: float = 0.80, alpha: float = 0.05) -> float:
    """McNemar normal-approximation MDE in accuracy points at the observed n."""
    n01, n10 = discordant(a, b)
    disc = max((n01 + n10) / len(a), 1e-6)
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z * np.sqrt(disc / len(a)) * 100.0)


def paired_arrays(base_run: str, treat_run: str):
    base, treat = ap.load_correct(base_run), ap.load_correct(treat_run)
    qids = [q for q in base if q in treat]
    qids.sort(key=lambda q: (int(q.split("-")[0]), int(q.split("-")[1])))
    return np.array([base[q] for q in qids]), np.array([treat[q] for q in qids])


def compute() -> tuple[dict[str, dict], dict[str, np.ndarray]]:
    """Per-contrast summary numbers and the raw draws (paper seed and random stream)."""
    rng = np.random.default_rng(SEED)
    summary: dict[str, dict] = {}
    dump: dict[str, np.ndarray] = {}
    for name, base_run, treat_run in CONTRASTS:
        a, b = paired_arrays(base_run, treat_run)
        n = len(a)
        full_delta = 100.0 * (b.mean() - a.mean())
        pre = slice(0, SUBSET_N)
        prefix_delta = 100.0 * (b[pre].mean() - a[pre].mean())

        deltas = np.empty(B)
        pvals = np.empty(B)
        for i in range(B):
            idx = rng.choice(n, size=SUBSET_N, replace=False)
            deltas[i] = 100.0 * (b[idx].mean() - a[idx].mean())
            pvals[i] = mcnemar_p(a[idx], b[idx])

        lo, hi = np.percentile(deltas, [2.5, 97.5])
        fb, fc = discordant(a, b)
        pb, pc = discordant(a[pre], b[pre])
        summary[name] = {
            "n": n, "full_delta": full_delta, "full_p": mcnemar_p(a, b),
            "prefix_delta": prefix_delta, "prefix_p": mcnemar_p(a[pre], b[pre]),
            "median": float(np.median(deltas)), "lo95": float(lo), "hi95": float(hi), "sd": float(deltas.std()),
            "sign_flip_pct": 100.0 * float(np.mean(np.sign(deltas) != np.sign(full_delta))),
            "p_ge_3_5_pct": 100.0 * float(np.mean(deltas >= 3.5)),
            "p_ge_5_0_pct": 100.0 * float(np.mean(deltas >= 5.0)),
            "p_below_05_pct": 100.0 * float(np.mean(pvals < 0.05)),
            "prefix_percentile": 100.0 * float(np.mean(deltas <= prefix_delta)),
            "full_discordant": [fb, fc], "full_mde80": mde_paired(a, b),
            "prefix_discordant": [pb, pc], "prefix_mde80": mde_paired(a[pre], b[pre]),
        }
        key = name.split(" vs ")[0].replace("+", "_").replace(" ", "_")
        dump[f"{key}__deltas"] = deltas
        dump[f"{key}__pvals"] = pvals
        dump[f"{key}__scalars"] = np.array([full_delta, prefix_delta, summary[name]["sign_flip_pct"]])
    return summary, dump


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "subset_stability_cache.npz")
    args = parser.parse_args()

    summary, dump = compute()
    for name, r in summary.items():
        print(f"\n=== {name}  (n={r['n']}) ===")
        print(f"full-bucket delta          : {r['full_delta']:+.2f} pt   p={r['full_p']:.4f}")
        print(f"prefix-{SUBSET_N} delta (paper) : {r['prefix_delta']:+.2f} pt   p={r['prefix_p']:.4f}")
        print(f"random {SUBSET_N}-subsets      : median {r['median']:+.2f}, "
              f"95% range [{r['lo95']:+.2f}, {r['hi95']:+.2f}], sd {r['sd']:.2f}")
        print(f"  sign-flip rate vs full   : {r['sign_flip_pct']:.1f}%")
        print(f"  P(delta >= +3.5 pt)      : {r['p_ge_3_5_pct']:.1f}%")
        print(f"  P(delta >= +5.0 pt)      : {r['p_ge_5_0_pct']:.1f}%")
        print(f"  nominal p<0.05 rate      : {r['p_below_05_pct']:.1f}%")
        print(f"  percentile of paper prefix: {r['prefix_percentile']:.1f}")
        fb, fc = r["full_discordant"]
        pb, pc = r["prefix_discordant"]
        print(f"  discordant full  b={fb} c={fc} b+c={fb + fc} "
              f"({100.0 * (fb + fc) / r['n']:.1f}%)  MDE80={r['full_mde80']:.1f} pt")
        print(f"  discordant prefix b={pb} c={pc} b+c={pb + pc} "
              f"({100.0 * (pb + pc) / SUBSET_N:.1f}%)  MDE80={r['prefix_mde80']:.1f} pt")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **dump)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
