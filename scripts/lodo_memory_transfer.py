#!/usr/bin/env python3
"""Leave-one-domain-out (LODO) procedural-memory transfer (cached replay).

Addresses the paper's Limitation (ii): the procedural memory is mined on a
fixed set of domains and "new domains require re-mining." We test whether an
inject policy mined on a SUBSET of domains transfers to a held-out domain,
WITHOUT re-running any VLM: the per-question correctness for every inject mode
(none / asr / ocr) is already cached per domain, and the same 200 qids appear
under all three modes (paired), so a policy that picks an inject mode per
question can be evaluated by a pure table lookup.

Two transfer granularities:
  (A) domain-level   -- one inject mode for the whole held-out domain, chosen as
                        the pooled-best mode over the two TRAINING domains.
  (B) type-level     -- an inject mode per question, chosen as the pooled-best
                        mode for that question's (domain-agnostic) semantic type
                        over the two TRAINING domains; unseen types fall back to
                        the global pooled-best prior.

Baselines (fixed, no learning):  no-inject, always-asr, always-ocr.
Upper bounds (peek at held-out):  oracle-domain, oracle-type.

All comparisons within a domain are paired (identical qids), so we report exact
two-sided McNemar and a 10k percentile bootstrap CI of the paired delta, reusing
the methodology of scripts/stats_analysis.py.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
sys.path.insert(0, ROOT)
RNG = np.random.default_rng(20260613)
N_BOOT = 10_000

MODES = ("none", "asr", "ocr")

# (domain -> {mode -> cached run dir})
RUNS = {
    "Video-MME": {
        "none": "trackB_long_cascade_k8_f0p5",
        "asr": "trackB_long_cascade_k8_f0p5_injectASR",
        "ocr": "trackB_long_cascade_k8_f0p5_injectOCR",
    },
    "XD-Violence": {
        "none": "trackB_xd_violence_cascade_k8_f0p5",
        "asr": "trackB_xd_violence_cascade_k8_f0p5_injectASR",
        "ocr": "trackB_xd_violence_cascade_k8_f0p5_injectOCR",
    },
    "UCF-Crime": {
        "none": "trackB_ucf_crime_cascade_k8_f0p5",
        "asr": "trackB_ucf_crime_cascade_k8_f0p5_injectASR",
        "ocr": "trackB_ucf_crime_cascade_k8_f0p5_injectOCR",
    },
}

DOMAIN_LOADER = {
    "Video-MME": ("load_video_mme", {}),
    "XD-Violence": ("load_xd_violence", {}),
    "UCF-Crime": ("load_ucf_crime", {}),
}


# --------------------------------------------------------------------------- IO
def load_correct(dirname: str) -> dict[str, int]:
    path = os.path.join(OUT, dirname, "rows.jsonl")
    out: dict[str, int] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            qid = str(r.get("qid", r.get("id", r.get("question_id"))))
            out[qid] = int(bool(r["correct"]))
    return out


def load_questions(domain: str) -> dict[str, str]:
    from router import qa_datasets as Q
    fn_name, kw = DOMAIN_LOADER[domain]
    items = getattr(Q, fn_name)(**kw)
    return {it.qid: it.question for it in items}


# ---------------------------------------------------- domain-agnostic type tag
# Transparent, auditable keyword rules. Each question maps to exactly one type.
# Order matters: the first matching rule wins.
def tag_question(q: str) -> str:
    s = q.lower()

    def has(*kw: str) -> bool:
        return any(k in s for k in kw)

    # 1) on-screen / scene text  -> OCR is the relevant modality
    if has("written", "text on", "sign", "caption", "subtitle", "logo",
           "label", "screen", "letters", "number on", "word on", "title"):
        return "onscreen_text"
    # 2) spoken / narrated answer -> ASR is the relevant modality
    if has("say", "said", "says", "speak", "spoken", "mention", "according to",
           "narrat", "call it", "named", "explain", "discuss", "talk",
           "voiceover", "voice-over", "tells", "told", "describe"):
        return "speech"
    # 3) anomaly / event detection -> the binary event-QA bucket
    if has("violent", "violence", "fight", "shoot", "crime", "criminal",
           "dangerous", "unusual", "anomal", "abnormal", "weapon", "abuse"):
        return "event"
    # 4) counting / numeric visual
    if has("how many", "count", "number of"):
        return "count"
    # 5) default: a visual-attribute question (color, where, who, which, ...)
    return "visual"


# --------------------------------------------------------------------- stats
def mcnemar_exact_p(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    b01 = int(np.sum((a == 0) & (b == 1)))
    b10 = int(np.sum((a == 1) & (b == 0)))
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    k = min(b01, b10)
    p = float(stats.binomtest(k, n, 0.5, alternative="two-sided").pvalue)
    return b01, b10, p


def bootstrap_ci(a: np.ndarray, b: np.ndarray, level: float = 0.95
                 ) -> tuple[float, float, float]:
    d = (b - a).astype(float)
    n = len(d)
    idx = RNG.integers(0, n, size=(N_BOOT, n))
    boots = d[idx].mean(axis=1) * 100.0
    lo = float(np.percentile(boots, (1 - level) / 2 * 100))
    hi = float(np.percentile(boots, (1 + level) / 2 * 100))
    return d.mean() * 100.0, lo, hi


# --------------------------------------------------------------------- core
def build_tables():
    """Return correct[domain][mode] = {qid:0/1}, qids[domain], qtype[domain]."""
    correct: dict[str, dict[str, dict[str, int]]] = {}
    qids: dict[str, list[str]] = {}
    qtype: dict[str, dict[str, str]] = {}
    for dom, modes in RUNS.items():
        correct[dom] = {m: load_correct(d) for m, d in modes.items()}
        # qids present under all three modes (should be identical = 200)
        common = set.intersection(*[set(correct[dom][m]) for m in MODES])
        qids[dom] = sorted(common)
        questions = load_questions(dom)
        qtype[dom] = {q: tag_question(questions.get(q, "")) for q in qids[dom]}
    return correct, qids, qtype


def acc_vec(correct, dom, mode, qids) -> np.ndarray:
    return np.fromiter((correct[dom][mode][q] for q in qids), dtype=int,
                       count=len(qids))


def pooled_best_mode(correct, qids, qtype, train_doms, qtype_filter=None
                     ) -> str:
    """Mode with highest pooled accuracy over the training domains
    (optionally restricted to questions of a given semantic type)."""
    best_m, best_acc = "none", -1.0
    for m in MODES:
        tot = cor = 0
        for d in train_doms:
            for q in qids[d]:
                if qtype_filter is not None and qtype[d][q] != qtype_filter:
                    continue
                tot += 1
                cor += correct[d][m][q]
        if tot == 0:
            continue
        a = cor / tot
        if a > best_acc:
            best_acc, best_m = a, m
    return best_m if best_acc >= 0 else "none"


def routed_correct_vector(correct, dom, qids, mode_of_q) -> np.ndarray:
    """Per-question correctness when each qid uses its assigned mode."""
    return np.fromiter((correct[dom][mode_of_q[q]][q] for q in qids),
                       dtype=int, count=len(qids))


def main():
    correct, qids, qtype = build_tables()
    domains = list(RUNS.keys())

    report = {"folds": {}, "config": {"modes": MODES, "n_boot": N_BOOT}}

    print("=" * 78)
    print("LEAVE-ONE-DOMAIN-OUT PROCEDURAL-MEMORY TRANSFER (cached replay)")
    print("=" * 78)

    # per-domain per-mode accuracy table (context)
    print("\nPer-domain accuracy by fixed inject mode (n, %):")
    print(f"{'domain':<13} {'n':>4} {'none':>8} {'asr':>8} {'ocr':>8}"
          f" {'oracle':>8}")
    for d in domains:
        accs = {m: acc_vec(correct, d, m, qids[d]).mean() * 100 for m in MODES}
        orac = max(accs.values())
        print(f"{d:<13} {len(qids[d]):>4} "
              f"{accs['none']:>8.1f} {accs['asr']:>8.1f} {accs['ocr']:>8.1f}"
              f" {orac:>8.1f}")

    rows = []
    for held in domains:
        train = [d for d in domains if d != held]
        qd = qids[held]
        none_v = acc_vec(correct, held, "none", qd)

        # ---- baselines (fixed) ----
        base = {m: acc_vec(correct, held, m, qd) for m in MODES}

        # ---- (A) domain-level transfer ----
        m_dom = pooled_best_mode(correct, qids, qtype, train)
        transfer_dom = acc_vec(correct, held, m_dom, qd)

        # ---- (B) type-level transfer ----
        # mine per-type best mode on training; unseen types -> global prior
        global_prior = pooled_best_mode(correct, qids, qtype, train)
        train_types = {qtype[d][q] for d in train for q in qids[d]}
        type_mode = {t: pooled_best_mode(correct, qids, qtype, train,
                                         qtype_filter=t) for t in train_types}
        mode_of_q = {}
        seen = 0
        for q in qd:
            t = qtype[held][q]
            if t in type_mode:
                mode_of_q[q] = type_mode[t]
                seen += 1
            else:
                mode_of_q[q] = global_prior
        transfer_type = routed_correct_vector(correct, held, qd, mode_of_q)
        coverage = seen / len(qd)

        # ---- oracles (peek) ----
        oracle_dom_mode = max(MODES, key=lambda m: base[m].mean())
        oracle_dom = base[oracle_dom_mode]
        # per-type oracle on held-out
        held_types = {qtype[held][q] for q in qd}
        held_type_mode = {t: pooled_best_mode(correct, qids, qtype, [held],
                                              qtype_filter=t)
                          for t in held_types}
        oracle_mode_of_q = {q: held_type_mode[qtype[held][q]] for q in qd}
        oracle_type = routed_correct_vector(correct, held, qd, oracle_mode_of_q)

        def stat(vec):
            d_, lo, hi = bootstrap_ci(none_v, vec)
            _, _, p = mcnemar_exact_p(none_v, vec)
            return dict(acc=float(vec.mean() * 100), d=d_, lo=lo, hi=hi, p=p)

        fold = {
            "held_out": held,
            "train": train,
            "n": len(qd),
            "transfer_domain_mode": m_dom,
            "oracle_domain_mode": oracle_dom_mode,
            "type_coverage": coverage,
            "type_mode_map": type_mode,
            "metrics": {
                "no_inject": float(none_v.mean() * 100),
                "always_asr": stat(base["asr"]),
                "always_ocr": stat(base["ocr"]),
                "transfer_domain": stat(transfer_dom),
                "transfer_type": stat(transfer_type),
                "oracle_domain": stat(oracle_dom),
                "oracle_type": stat(oracle_type),
            },
        }
        report["folds"][held] = fold

        print("\n" + "-" * 78)
        print(f"HELD-OUT: {held}   (train on {', '.join(train)};  n={len(qd)})")
        print(f"  domain-level transferred mode = '{m_dom}'"
              f"   |  type coverage = {coverage*100:.0f}%")
        print(f"  {'method':<20}{'acc%':>7}{'Δ vs none':>12}{'95% CI':>16}"
              f"{'McNemar p':>11}")
        print(f"  {'no-inject (base)':<20}{none_v.mean()*100:>7.1f}"
              f"{'--':>12}{'--':>16}{'--':>11}")
        for key in ("always_asr", "always_ocr", "transfer_domain",
                    "transfer_type", "oracle_domain", "oracle_type"):
            m = fold["metrics"][key]
            print(f"  {key:<20}{m['acc']:>7.1f}{m['d']:>+12.1f}"
                  f"{f'[{m[chr(108)+chr(111)]:+.1f},{m[chr(104)+chr(105)]:+.1f}]':>16}"
                  f"{m['p']:>11.3f}")

        # inject-gain recovered (domain-level)
        og = fold["metrics"]["oracle_domain"]["acc"] - none_v.mean() * 100
        tg = fold["metrics"]["transfer_domain"]["acc"] - none_v.mean() * 100
        rec = (tg / og * 100) if og > 1e-9 else float("nan")
        fold["oracle_gain"] = og
        fold["transfer_gain"] = tg
        fold["pct_oracle_recovered"] = rec
        rows.append((held, none_v.mean()*100,
                     fold["metrics"]["transfer_domain"]["acc"],
                     fold["metrics"]["transfer_type"]["acc"],
                     fold["metrics"]["oracle_domain"]["acc"], rec))

    # ---- aggregate across folds ----
    print("\n" + "=" * 78)
    print("AGGREGATE (mean across 3 LODO folds)")
    print("=" * 78)
    arr = np.array([[r[1], r[2], r[3], r[4]] for r in rows])
    mean = arr.mean(axis=0)
    print(f"  mean no-inject        : {mean[0]:.1f}%")
    print(f"  mean transfer-domain  : {mean[1]:.1f}%")
    print(f"  mean transfer-type    : {mean[2]:.1f}%")
    print(f"  mean oracle-domain    : {mean[3]:.1f}%  (upper bound)")
    gains = arr[:, 3] - arr[:, 0]
    tgain = arr[:, 1] - arr[:, 0]
    rec = float(np.nansum(tgain) / np.nansum(gains) * 100) if np.nansum(gains) else float('nan')
    print(f"  oracle inject-gain recovered by domain-transfer: {rec:.0f}%")
    report["aggregate"] = {
        "mean_no_inject": float(mean[0]),
        "mean_transfer_domain": float(mean[1]),
        "mean_transfer_type": float(mean[2]),
        "mean_oracle_domain": float(mean[3]),
        "pct_oracle_gain_recovered": rec,
    }

    out_dir = os.path.join(OUT, "lodo_memory_transfer")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "lodo_results.json")
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
