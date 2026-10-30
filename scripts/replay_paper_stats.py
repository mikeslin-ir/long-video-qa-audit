#!/usr/bin/env python3
"""Recompute the paper's statistics from the packaged per-item outcomes.

Everything the paper quotes is a function of per-item correctness, and every
reported run's per-item outcomes ship in ``results/predictions/``. This script
recomputes those statistics offline (no GPU, no videos, no raw run outputs) and
compares each with ``results/paper_reference_values.json``:

    python scripts/replay_paper_stats.py --check            # fast set, used in CI
    python scripts/replay_paper_stats.py --check --slow     # adds the subset-resampling analysis
    python scripts/replay_paper_stats.py --write-reference  # authors: regenerate the reference file

Counts and p-values must match exactly (p to 4 decimals); accuracies to 0.05 pt;
bootstrap intervals to 0.15 pt (they are replayed with the paper's seeds and random
stream, so they normally match to the printed digit).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

REFERENCE = ROOT / "results" / "paper_reference_values.json"


def tolerance(name: str) -> float:
    leaf = name.rsplit("/", 1)[-1]
    if leaf in {"n", "b01", "b10", "gain", "loss", "agree", "false_accept", "false_reject"}:
        return 0.0
    if leaf in {"p", "p_exact", "p_normal", "p_permutation", "p_sign_flip", "full_p", "prefix_p"}:
        return 6e-4
    if leaf in {"z", "kappa", "precision", "recall"}:
        return 0.006
    if leaf in {"lo95", "hi95", "lo90", "hi90", "lo", "hi", "ci_lo", "ci_hi", "lo95_boot", "hi95_boot"}:
        return 0.15
    return 0.051


def battery_values() -> dict[str, float]:
    import stats_analysis as sa
    out: dict[str, float] = {}
    for comparison in sa.BATTERY:  # sequential: the bootstrap stream depends on the order
        try:
            r = sa.run_one(comparison)
        except FileNotFoundError:
            continue
        for field in ("n", "acc_a", "acc_b", "delta", "lo95", "hi95", "p", "b01", "b10", "lo90", "hi90"):
            out[f"battery/{comparison.label}/{field}"] = float(r[field])
    return out


def pooled_values() -> dict[str, float]:
    import pooled_sensitivity as ps
    out: dict[str, float] = {}
    for title, base, variant in ps.contrasts():
        res = ps.analyse(base, variant)
        tag = title.split()[0]
        for scope, r in res["pooled"].items():
            for field in ("n", "delta", "gain", "loss", "z", "p_normal", "p_exact"):
                out[f"pooled/{tag}/{scope}/{field}"] = float(r[field])
            out[f"pooled/{tag}/{scope}/ci_lo"], out[f"pooled/{tag}/{scope}/ci_hi"] = map(float, r["ci95"])
        for scope, r in res["segments"].items():
            for field in ("n", "delta", "gain", "loss", "p_exact"):
                out[f"segment/{tag}/{scope}/{field}"] = float(r[field])
        for scope, r in res["equal_domain"].items():
            out[f"equal_domain/{tag}/{scope}/delta"] = float(r["delta"])
            out[f"equal_domain/{tag}/{scope}/p_permutation"] = float(r["p_permutation"])
    return out


def judge_values() -> dict[str, float]:
    import judge_validation_report as jv
    rows = jv.load()
    out: dict[str, float] = {}
    by_set = {"all": rows}
    for r in rows:
        by_set.setdefault(r["dataset"], []).append(r)
    for name, subset in by_set.items():
        s = jv.agreement_stats(subset)
        for field in ("n", "agree", "false_accept", "false_reject", "kappa"):
            out[f"judge/{name}/{field}"] = float(s[field])
        out[f"judge/{name}/agreement_pct"] = 100 * float(s["agreement"])
    return out


def music_values() -> dict[str, float]:
    import audio_ablation_analysis as aa
    with contextlib.redirect_stdout(io.StringIO()):
        table = aa.full_split_table()
    out: dict[str, float] = {}
    for name, r in {**table["types"], "micro": table["micro"]}.items():
        for field in ("n", "base", "variant", "delta", "gain", "loss", "p"):
            out[f"music_avqa/{name}/{field}"] = float(r[field])
    arms = aa.read_arms()
    if arms is not None:
        for grader in ("judge", "inloop"):
            with contextlib.redirect_stdout(io.StringIO()):
                ablation = aa.ablation(arms, grader)
            if ablation is None:
                continue
            pooled = ablation["micro_poststratified"]
            for arm, acc in pooled["accuracy"].items():
                out[f"audio_ablation/{grader}/{arm}"] = float(acc)
            for k in ("total", "audio", "non_audio"):
                out[f"audio_ablation/{grader}/{k}/delta"] = float(pooled[k]["delta"])
                out[f"audio_ablation/{grader}/{k}/ci_lo"], out[f"audio_ablation/{grader}/{k}/ci_hi"] = map(
                    float, pooled[k]["ci95"])
            share = pooled["audio_share_pct"]
            out[f"audio_ablation/{grader}/share_pct"] = float(share["point"])
            out[f"audio_ablation/{grader}/share_ci_lo"], out[f"audio_ablation/{grader}/share_ci_hi"] = map(
                float, share["ci95"])
            with contextlib.redirect_stdout(io.StringIO()):
                mm = aa.mismatched_control(arms, grader)
            if mm is not None:
                out[f"audio_mismatched/{grader}/n"] = float(mm["n"])
                for arm, acc in mm["accuracy"].items():
                    out[f"audio_mismatched/{grader}/acc/{arm}"] = float(acc)
                for name, c in mm["contrasts"].items():
                    for field in ("delta", "gain", "loss", "p"):
                        out[f"audio_mismatched/{grader}/{name}/{field}"] = float(c[field])
                    out[f"audio_mismatched/{grader}/{name}/ci_lo"], out[f"audio_mismatched/{grader}/{name}/ci_hi"] = map(
                        float, c["ci95"])
    return out


def dispatch_values() -> dict[str, float]:
    import routed_expert_analysis as rea
    import audit_predictions as ap
    r = rea.compute(ap.load_correct(rea.BASE), ap.load_correct(rea.AV), rea.read_dispatch())
    out = {f"dispatch/{k}": float(r[k]) for k in (
        "n", "routed_to_av", "routed_to_av_pct", "base", "av", "routed", "oracle", "routed_minus_base",
        "p", "headroom_captured_pct", "precision", "recall", "expert_helped", "expert_hurt")}
    out["dispatch/ci_lo"], out["dispatch/ci_hi"] = map(float, r["ci95"])
    return out


def subset_values() -> dict[str, float]:
    import subset_stability as ss
    summary, _ = ss.compute()
    out: dict[str, float] = {}
    for name, r in summary.items():
        for field, value in r.items():
            if isinstance(value, (int, float)):
                out[f"subset/{name}/{field}"] = float(value)
    return out


def compute(slow: bool) -> dict[str, float]:
    values: dict[str, float] = {}
    for label, fn in (("battery", battery_values), ("pooled", pooled_values),
                      ("judge", judge_values), ("music_avqa", music_values),
                      ("dispatch", dispatch_values)):
        print(f"computing {label} ...", flush=True)
        values.update(fn())
    if slow:
        print("computing subset resampling ...", flush=True)
        values.update(subset_values())
    return values


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compare with results/paper_reference_values.json")
    ap.add_argument("--slow", action="store_true", help="also replay the 20,000-subset resampling analysis")
    ap.add_argument("--write-reference", action="store_true", help="authors: (re)generate the reference file")
    args = ap.parse_args()

    values = compute(args.slow)
    if args.write_reference:
        existing = json.loads(REFERENCE.read_text()) if REFERENCE.exists() else {}
        existing.update({k: round(v, 6) for k, v in values.items()})
        REFERENCE.parent.mkdir(parents=True, exist_ok=True)
        REFERENCE.write_text(json.dumps(dict(sorted(existing.items())), indent=1) + "\n")
        print(f"wrote {len(existing)} reference values -> {REFERENCE}")
        return 0
    if not args.check:
        for key, value in sorted(values.items()):
            print(f"{key} = {value:.6g}")
        return 0

    reference = json.loads(REFERENCE.read_text())
    bad = checked = 0
    for key, want in reference.items():
        if key not in values:
            if args.slow or not key.startswith("subset/"):
                print(f"MISSING  {key}")
                bad += 1
            continue
        checked += 1
        if abs(values[key] - want) > tolerance(key) + 1e-9:
            print(f"MISMATCH {key}: replay {values[key]:.6g}, reference {want:.6g}")
            bad += 1
    print(f"checked {checked} values against {REFERENCE.name}: {bad} problems")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
