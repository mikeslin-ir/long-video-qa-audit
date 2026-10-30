#!/usr/bin/env python3
"""How much of the pooled injection effect survives beyond the development prefix?

The audit's confirmatory claim (F3a) pools Video-MME long, XD-Violence and
UCF-Crime and tests the cascade-plus-ASR arm against uniform sampling. The
published pool used the first 200 Video-MME long items, which is also the
prefix on which the cascade was developed. This script re-runs the same paired
test on every other cut of the data we have, using only per-item outcomes:

  published   Video-MME long items 1-200  + XD-Violence + UCF-Crime      (n=600)
  all data    Video-MME long items 1-900  + XD-Violence + UCF-Crime      (n=1300)
  held out    Video-MME long items 201-900 + XD-Violence + UCF-Crime     (n=1100)
  VMME 201-900 alone                                                      (n=700)
  equal-domain weights: each domain contributes 1/3 regardless of its n

Pooled tests pair items within a domain and sum the discordant counts, which is
the stratified McNemar test, and pooled intervals resample items within each
domain; the item-weighted estimate therefore weights Video-MME by its item
count, the equal-domain estimate does not.

Usage:
    python scripts/pooled_sensitivity.py [--json results/pooled_sensitivity.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import audit_predictions as ap  # noqa: E402
import audit_stats as st  # noqa: E402

ARMS = {
    "uniform": {
        "vmme200": "trackB_long_uniform_k8",
        "vmme900": "trackB_long_n900_k8_uniform",
        "xd": "trackB_xd_violence_uniform_k8",
        "ucf": "trackB_ucf_crime_uniform_k8",
    },
    "cascade": {
        "vmme200": "trackB_long_cascade_k8_f0p5",
        "vmme900": "trackB_long_n900_k8_cascade_f0p5",
        "xd": "trackB_xd_violence_cascade_k8_f0p5",
        "ucf": "trackB_ucf_crime_cascade_k8_f0p5",
    },
    "cascade+ASR": {
        "vmme200": "trackB_long_cascade_k8_f0p5_injectASR",
        "vmme900": "trackB_long_n900_k8_cascade_f0p5_injectASR",
        "xd": "trackB_xd_violence_cascade_k8_f0p5_injectASR",
        "ucf": "trackB_ucf_crime_cascade_k8_f0p5_injectASR",
    },
    # The anomaly "+ASR" arms above injected nothing (no transcripts existed); these re-run them
    # with Whisper transcripts built first. Video-MME already had transcripts.
    "cascade+ASR (transcribed)": {
        "vmme200": "trackB_long_cascade_k8_f0p5_injectASR",
        "vmme900": "trackB_long_n900_k8_cascade_f0p5_injectASR",
        "xd": "trackB_xd_violence_cascade_k8_f0p5_injectASR_transcribed",
        "ucf": "trackB_ucf_crime_cascade_k8_f0p5_injectASR_transcribed",
    },
}

CONTRASTS = [
    ("F3a  cascade+ASR vs uniform", "uniform", "cascade+ASR"),
    ("F1   cascade vs uniform (selection only)", "uniform", "cascade"),
    ("F2   ASR within cascade", "cascade", "cascade+ASR"),
    ("F3t  cascade+ASR (transcribed) vs uniform", "uniform", "cascade+ASR (transcribed)"),
    ("F2t  ASR (transcribed) within cascade", "cascade", "cascade+ASR (transcribed)"),
]

DOMAIN_NAMES = {"vmme": "Video-MME long", "xd": "XD-Violence", "ucf": "UCF-Crime"}


def paired(base: str, variant: str, qids: list[str] | None = None):
    a, b = ap.load_correct(base), ap.load_correct(variant)
    keys = sorted(set(a) & set(b)) if qids is None else [q for q in qids if q in a and q in b]
    return keys, np.array([a[k] for k in keys]), np.array([b[k] for k in keys])


def summarize(label: str, base, variant, seed: int, strata=None) -> dict:
    """Paired summary; ``strata`` = [(base, variant), ...] makes the CI a stratified
    bootstrap (items resampled within each domain) instead of a flat one."""
    gain, loss = st.discordant(base, variant)
    z, p_z = st.mcnemar_z(gain, loss)
    if strata:
        delta, lo, hi = st.stratified_bootstrap_ci([np.asarray(v) - np.asarray(b) for b, v in strata], seed=seed)
    else:
        delta, lo, hi = st.paired_bootstrap_ci(np.asarray(variant) - np.asarray(base), seed=seed)
    return {
        "label": label, "n": int(len(base)),
        "base_acc": float(np.mean(base) * 100), "variant_acc": float(np.mean(variant) * 100),
        "delta": delta, "ci95": [lo, hi], "gain": gain, "loss": loss,
        "z": z, "p_normal": p_z, "p_exact": st.mcnemar_exact_p(gain, loss),
    }


def analyse(base_arm: str, variant_arm: str) -> dict:
    parts: dict[str, tuple] = {}
    prefix_ids = None
    for key in ("vmme200", "vmme900", "xd", "ucf"):
        keys, a, b = paired(ARMS[base_arm][key], ARMS[variant_arm][key])
        parts[key] = (keys, a, b)
    prefix_ids = set(parts["vmme200"][0])
    full_keys, full_a, full_b = parts["vmme900"]
    mask = np.array([k in prefix_ids for k in full_keys])
    if mask.sum() != len(prefix_ids):
        raise SystemExit("the n=200 prefix is not contained in the n=900 run")

    held_a, held_b = full_a[~mask], full_b[~mask]
    seg = {
        "vmme_prefix_200": (parts["vmme200"][1], parts["vmme200"][2]),
        "vmme_201_900": (held_a, held_b),
        "vmme_900": (full_a, full_b),
        "xd_200": parts["xd"][1:], "ucf_200": parts["ucf"][1:],
    }
    rows = {name: summarize(name, a, b, seed=11 + i) for i, (name, (a, b)) in enumerate(seg.items())}

    def pool(names):
        a = np.concatenate([seg[n][0] for n in names])
        b = np.concatenate([seg[n][1] for n in names])
        return a, b

    scopes = {
        "published (VMME 1-200 + XD + UCF)": ["vmme_prefix_200", "xd_200", "ucf_200"],
        "all data (VMME 1-900 + XD + UCF)": ["vmme_900", "xd_200", "ucf_200"],
        "held out (VMME 201-900 + XD + UCF)": ["vmme_201_900", "xd_200", "ucf_200"],
    }
    pooled = {label: summarize(label, *pool(names), seed=101 + i, strata=[seg[n] for n in names])
              for i, (label, names) in enumerate(scopes.items())}

    def diffs(vmme_key):
        return {"vmme": seg[vmme_key][1] - seg[vmme_key][0],
                "xd": seg["xd_200"][1] - seg["xd_200"][0],
                "ucf": seg["ucf_200"][1] - seg["ucf_200"][0]}

    equal = {}
    for label, vmme_key in (("equal-domain, VMME 1-900", "vmme_900"),
                            ("equal-domain, VMME 201-900", "vmme_201_900")):
        r = st.domain_weighted_delta(diffs(vmme_key), seed=7)
        equal[label] = {"delta": r.delta, "ci95": [r.lo, r.hi],
                        "p_permutation": r.p_permutation, "p_normal": r.p_normal}
    return {"segments": rows, "pooled": pooled, "equal_domain": equal}


def fmt(r: dict) -> str:
    return (f"n={r['n']:<5} {r['base_acc']:5.1f} -> {r['variant_acc']:5.1f}  "
            f"{r['delta']:+5.2f} pt  [{r['ci95'][0]:+.1f},{r['ci95'][1]:+.1f}]  "
            f"gain/loss {r['gain']}/{r['loss']}  Z={r['z']:+.2f}  p={r['p_normal']:.4f}  exact p={r['p_exact']:.4f}")


def available(base_arm: str, variant_arm: str) -> bool:
    """True when every run of both arms exists (live or packaged)."""
    for arm in (base_arm, variant_arm):
        for run in ARMS[arm].values():
            try:
                ap.load_correct(run)
            except FileNotFoundError:
                return False
    return True


def contrasts() -> list[tuple[str, str, str]]:
    return [c for c in CONTRASTS if available(c[1], c[2])]


def main() -> None:
    ap_ = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap_.add_argument("--json", type=Path, default=None, help="also write all numbers to this file")
    args = ap_.parse_args()

    out = {}
    for title, base, variant in contrasts():
        res = analyse(base, variant)
        out[title] = res
        print(f"\n=== {title} ===")
        print("  per segment")
        for name, r in res["segments"].items():
            print(f"    {name:<26}{fmt(r)}")
        print("  pooled (stratified McNemar: discordant counts summed over domains)")
        for name, r in res["pooled"].items():
            print(f"    {name:<38}{fmt(r)}")
        print("  equal domain weights (1/3 each; 20k within-domain bootstrap, sign-flip test)")
        for name, r in res["equal_domain"].items():
            print(f"    {name:<38}{r['delta']:+5.2f} pt  [{r['ci95'][0]:+.1f},{r['ci95'][1]:+.1f}]  "
                  f"p(sign-flip)={r['p_permutation']:.4f}  p(normal)={r['p_normal']:.4f}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
