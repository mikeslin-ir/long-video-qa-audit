#!/usr/bin/env python3
"""Statistical hardening for the audit paper (zero new experiments).

For each named pairwise comparison between two run directories it computes,
on per-item paired correctness (matched by ``qid``):

* paired n, accuracy of A and B, and the paired accuracy delta (B - A);
* exact two-sided McNemar p-value (binomial on discordant pairs);
* bootstrap 95% CI of the paired delta (percentile, 10k resamples);
* TOST equivalence verdict at margin +/- m (90% CI inclusion rule, alpha=0.05);
* post-hoc power for the observed effect and the minimum detectable effect
  (MDE) at the paired n for 80% power, alpha=0.05.

Usage:
    python scripts/stats_analysis.py            # run the predefined battery
    python scripts/stats_analysis.py --latex    # also emit a LaTeX table body
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import audit_predictions  # noqa: E402

OUT = os.path.join(ROOT, "outputs")
RNG = np.random.default_rng(20260613)
N_BOOT = 10_000


def load_correct(dirname: str) -> dict[str, int]:
    """Return {qid: 0/1 correct} for a run.

    Reads ``outputs/<run>/`` when it exists and the packaged
    ``results/predictions/`` otherwise, so the battery below replays offline.
    """
    return audit_predictions.load_correct(dirname)


def pair(a_dir: str, b_dir: str, first_n: int = 0) -> tuple[np.ndarray, np.ndarray]:
    a, b = load_correct(a_dir), load_correct(b_dir)
    if first_n > 0:
        keys = [k for k in a if k in b][:first_n]
    else:
        keys = sorted(set(a) & set(b))
    return (
        np.fromiter((a[k] for k in keys), dtype=int, count=len(keys)),
        np.fromiter((b[k] for k in keys), dtype=int, count=len(keys)),
    )


def mcnemar_exact_p(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Exact two-sided McNemar. Returns (b01, b10, p)."""
    b01 = int(np.sum((a == 0) & (b == 1)))  # A wrong, B right
    b10 = int(np.sum((a == 1) & (b == 0)))  # A right, B wrong
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    k = min(b01, b10)
    # two-sided exact binomial against p=0.5
    p = float(stats.binomtest(k, n, 0.5, alternative="two-sided").pvalue)
    return b01, b10, p


def bootstrap_ci(a: np.ndarray, b: np.ndarray, level: float = 0.95) -> tuple[float, float, float]:
    """Percentile bootstrap CI of paired delta (mean(b) - mean(a)), in points."""
    d = (b - a).astype(float)
    n = len(d)
    idx = RNG.integers(0, n, size=(N_BOOT, n))
    boots = d[idx].mean(axis=1) * 100.0
    lo = float(np.percentile(boots, (1 - level) / 2 * 100))
    hi = float(np.percentile(boots, (1 + level) / 2 * 100))
    return d.mean() * 100.0, lo, hi


def tost_equivalent(a: np.ndarray, b: np.ndarray, margin_pp: float) -> tuple[bool, float, float]:
    """Equivalence at alpha=0.05 via the 90% CI inclusion rule.

    Equivalent iff the 90% bootstrap CI of the paired delta is contained in
    (-margin, +margin). Returns (is_equivalent, ci90_lo, ci90_hi).
    """
    _, lo, hi = bootstrap_ci(a, b, level=0.90)
    return (lo > -margin_pp and hi < margin_pp), lo, hi


def mde_paired(a: np.ndarray, b: np.ndarray, power: float = 0.80, alpha: float = 0.05) -> float:
    """Minimum detectable paired delta (pp) at given power using the
    discordant-rate (McNemar) normal approximation at the observed n."""
    n = len(a)
    disc = float(np.mean((a == 0) & (b == 1)) + np.mean((a == 1) & (b == 0)))
    disc = max(disc, 1e-6)
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    # delta in proportion of discordant differences; convert to accuracy pp.
    # var of paired diff ~ disc/n ; MDE in mean paired diff:
    return float((z_a + z_b) * np.sqrt(disc / n) * 100.0)


def post_hoc_power(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> float:
    n = len(a)
    d = (b - a).astype(float)
    eff = abs(d.mean())
    sd = d.std(ddof=1)
    if sd == 0:
        return 1.0 if eff > 0 else 0.0
    se = sd / np.sqrt(n)
    z_a = stats.norm.ppf(1 - alpha / 2)
    return float(1 - stats.norm.cdf(z_a - eff / se) + stats.norm.cdf(-z_a - eff / se))


@dataclass
class Cmp:
    label: str
    a_dir: str          # baseline / reference
    b_dir: str          # variant of interest
    margin: float       # equivalence margin in pp (for TOST)
    kind: str           # "equiv" (we want null) or "super" (we want effect)
    first_n: int = 0    # >0: only the first N items of the baseline run, in its native order


BATTERY = [
    # ---- F1: selection-only multi-cue is NULL (we want equivalence) ----
    Cmp("F1 forced-multicue vs uniform", "trackB_long_uniform_k8",
        "trackB_long_cascade_k8_relaxed_f0p5", margin=5.0, kind="equiv"),
    Cmp("F1 forced-multicue vs anchored-cascade", "trackB_long_cascade_k8_f0p5",
        "trackB_long_cascade_k8_relaxed_f0p5", margin=5.0, kind="equiv"),
    Cmp("F1 motion vs uniform", "trackB_long_uniform_k8",
        "trackB_long_motion_k8", margin=5.0, kind="equiv"),
    Cmp("F1 clip vs uniform", "trackB_long_uniform_k8",
        "trackB_long_clip_k8", margin=5.0, kind="equiv"),
    Cmp("F1 anchored-cascade vs uniform", "trackB_long_uniform_k8",
        "trackB_long_cascade_k8_f0p5", margin=5.0, kind="super"),
    # ---- F2: injection is super-additive (we want effect) ----
    Cmp("F2 uniform: +ASR vs no-inject", "trackB_long_uniform_k8",
        "trackB_long_uniform_k8_injectASR", margin=5.0, kind="super"),
    Cmp("F2 cascade: +ASR vs no-inject", "trackB_long_cascade_k8_f0p5",
        "trackB_long_cascade_k8_f0p5_injectASR", margin=5.0, kind="super"),
    Cmp("F2 cascade: +OCR vs no-inject", "trackB_long_cascade_k8_f0p5",
        "trackB_long_cascade_k8_f0p5_injectOCR", margin=5.0, kind="super"),
    Cmp("F2 headline cascade+ASR vs uniform-no-inject", "trackB_long_uniform_k8",
        "trackB_long_cascade_k8_f0p5_injectASR", margin=5.0, kind="super"),
    Cmp("F2 ASR+OCR vs ASR (prompt-crowding)", "trackB_long_cascade_k8_f0p5_injectASR",
        "trackB_long_cascade_k8_f0p5_injectASR_OCR", margin=5.0, kind="equiv"),
    # ---- Full-scale n=900 robustness check (paper wave7 runs) ----
    Cmp("n900 selection-only: cascade vs uniform", "trackB_long_n900_k8_uniform",
        "trackB_long_n900_k8_cascade_f0p5", margin=3.0, kind="equiv"),
    Cmp("n900 cascade+ASR vs uniform", "trackB_long_n900_k8_uniform",
        "trackB_long_n900_k8_cascade_f0p5_injectASR", margin=3.0, kind="super"),
    Cmp("n900 planner vs uniform", "trackB_long_n900_k8_uniform",
        "trackB_long_n900_k8_llmPlanner_proceduralV3_injectOnly", margin=3.0, kind="super"),
    Cmp("n900 planner vs always-inject ASR", "trackB_long_n900_k8_cascade_f0p5_injectASR",
        "trackB_long_n900_k8_llmPlanner_proceduralV3_injectOnly", margin=3.0, kind="equiv"),
    # ---- Added after review. Appended (never inserted) so that the bootstrap
    # ---- stream above, and therefore every earlier interval, is unchanged. ----
    # Remaining rows of the full-bucket selector table.
    Cmp("n900 AKS port vs uniform", "trackB_long_n900_k8_uniform",
        "trackB_long_n900_k8_aks", margin=5.0, kind="equiv"),
    Cmp("n900 KeyScore port vs uniform", "trackB_long_n900_k8_uniform",
        "trackB_long_n900_k8_keyscore", margin=5.0, kind="equiv"),
    Cmp("n900 VideoTree port vs uniform", "trackB_long_n900_k8_uniform",
        "trackB_long_n900_k8_videotree", margin=5.0, kind="equiv"),
    Cmp("n900 cascade+OCR vs uniform", "trackB_long_n900_k8_uniform",
        "trackB_long_n900_k8_cascade_f0p5_injectOCR", margin=5.0, kind="equiv"),
    Cmp("n900 ASR within cascade", "trackB_long_n900_k8_cascade_f0p5",
        "trackB_long_n900_k8_cascade_f0p5_injectASR", margin=3.0, kind="super"),
    # Second backbone (Qwen2-VL-7B) at full scale.
    Cmp("Qwen2-VL n900 cascade vs uniform", "trackB_long_n900_k8_qwen2vl_uniform",
        "trackB_long_n900_k8_qwen2vl_cascade_f0p5", margin=5.0, kind="equiv"),
    Cmp("Qwen2-VL n900 cascade+ASR vs uniform", "trackB_long_n900_k8_qwen2vl_uniform",
        "trackB_long_n900_k8_qwen2vl_cascade_f0p5_injectASR", margin=5.0, kind="super"),
    Cmp("Qwen2-VL n900 ASR within cascade", "trackB_long_n900_k8_qwen2vl_cascade_f0p5",
        "trackB_long_n900_k8_qwen2vl_cascade_f0p5_injectASR", margin=3.0, kind="super"),
    # Second benchmark (EgoSchema, n=500) from the early pipeline.
    Cmp("EgoSchema cascade vs uniform (early pipeline)", "ksweep_uniform_k8_egoschema",
        "ksweep_cascade_k8_egoschema", margin=5.0, kind="equiv"),
    Cmp("EgoSchema random vs uniform (early pipeline)", "ksweep_uniform_k8_egoschema",
        "ksweep_random_k8_egoschema", margin=5.0, kind="equiv"),
    # ---- Review-response runs (scripts/paper_runs/run_review_experiments.sh). Appended after
    # ---- everything above; a comparison whose runs are absent is skipped without drawing. ----
    # Anomaly domains with transcripts that exist (the original +ASR arms there injected nothing).
    Cmp("XD cascade+ASR (transcribed) vs uniform", "trackB_xd_violence_uniform_k8",
        "trackB_xd_violence_cascade_k8_f0p5_injectASR_transcribed", margin=5.0, kind="super"),
    Cmp("XD ASR (transcribed) within cascade", "trackB_xd_violence_cascade_k8_f0p5",
        "trackB_xd_violence_cascade_k8_f0p5_injectASR_transcribed", margin=3.0, kind="super"),
    Cmp("UCF cascade+ASR (transcribed) vs uniform", "trackB_ucf_crime_uniform_k8",
        "trackB_ucf_crime_cascade_k8_f0p5_injectASR_transcribed", margin=5.0, kind="super"),
    Cmp("UCF ASR (transcribed) within cascade", "trackB_ucf_crime_cascade_k8_f0p5",
        "trackB_ucf_crime_cascade_k8_f0p5_injectASR_transcribed", margin=3.0, kind="super"),
    # EgoSchema, n=500, K=8, all selectors through the same harness.
    Cmp("EgoSchema cascade vs uniform", "egoschema_n500_k8_uniform", "egoschema_n500_k8_cascade",
        margin=5.0, kind="equiv"),
    Cmp("EgoSchema AKS port vs uniform", "egoschema_n500_k8_uniform", "egoschema_n500_k8_aks",
        margin=5.0, kind="equiv"),
    Cmp("EgoSchema KeyScore port vs uniform", "egoschema_n500_k8_uniform", "egoschema_n500_k8_keyscore",
        margin=5.0, kind="equiv"),
    Cmp("EgoSchema VideoTree port vs uniform", "egoschema_n500_k8_uniform", "egoschema_n500_k8_videotree",
        margin=5.0, kind="equiv"),
    # Budget K=1 at full scale, and the segment-midpoint uniform baseline.
    Cmp("n900 K=1 cascade vs uniform", "trackB_long_n900_k1_uniform", "trackB_long_n900_k1_cascade_f0p5",
        margin=5.0, kind="super"),
    Cmp("n900 K=1 uniform-center vs uniform", "trackB_long_n900_k1_uniform", "trackB_long_n900_k1_uniform_center",
        margin=5.0, kind="super"),
    Cmp("n900 K=1 cascade vs uniform-center", "trackB_long_n900_k1_uniform_center", "trackB_long_n900_k1_cascade_f0p5",
        margin=5.0, kind="super"),
    Cmp("n900 K=8 uniform-center vs uniform", "trackB_long_n900_k8_uniform", "trackB_long_n900_k8_uniform_center",
        margin=5.0, kind="equiv"),
    Cmp("n900 cascade vs uniform-center", "trackB_long_n900_k8_uniform_center", "trackB_long_n900_k8_cascade_f0p5",
        margin=5.0, kind="equiv"),
    Cmp("n900 AKS port vs uniform-center", "trackB_long_n900_k8_uniform_center", "trackB_long_n900_k8_aks",
        margin=5.0, kind="equiv"),
    # Qwen2-VL with the +ASR arm re-run on the cascade arm's candidate pool.
    Cmp("Qwen2-VL n900 cascade+ASR (same pool) vs uniform", "trackB_long_n900_k8_qwen2vl_uniform",
        "trackB_long_n900_k8_qwen2vl_cascade_f0p5_injectASR_audit", margin=5.0, kind="super"),
    Cmp("Qwen2-VL n900 ASR within cascade (same pool)", "trackB_long_n900_k8_qwen2vl_cascade_f0p5",
        "trackB_long_n900_k8_qwen2vl_cascade_f0p5_injectASR_audit", margin=3.0, kind="super"),
    # The released default planner profile (STACFP + redundancy) at full scale.
    Cmp("n900 STACFP cascade vs uniform", "trackB_long_n900_k8_uniform",
        "trackB_long_n900_k8_stacfp_cascade_f0p5", margin=5.0, kind="equiv"),
    Cmp("n900 STACFP cascade+ASR vs uniform", "trackB_long_n900_k8_uniform",
        "trackB_long_n900_k8_stacfp_cascade_f0p5_injectASR", margin=5.0, kind="super"),
    Cmp("n900 ASR within STACFP cascade", "trackB_long_n900_k8_stacfp_cascade_f0p5",
        "trackB_long_n900_k8_stacfp_cascade_f0p5_injectASR", margin=3.0, kind="super"),
    Cmp("n900 STACFP vs audit cascade", "trackB_long_n900_k8_cascade_f0p5",
        "trackB_long_n900_k8_stacfp_cascade_f0p5", margin=5.0, kind="equiv"),
    # Where the injected block sits in the prompt (before the question instead of after the options).
    Cmp("XD ASR before-question vs cascade", "trackB_xd_violence_cascade_k8_f0p5",
        "trackB_xd_violence_cascade_k8_f0p5_injectASR_transcribed_before", margin=3.0, kind="super"),
    Cmp("XD ASR before vs after the question", "trackB_xd_violence_cascade_k8_f0p5_injectASR_transcribed",
        "trackB_xd_violence_cascade_k8_f0p5_injectASR_transcribed_before", margin=3.0, kind="super"),
    Cmp("UCF ASR before-question vs cascade", "trackB_ucf_crime_cascade_k8_f0p5",
        "trackB_ucf_crime_cascade_k8_f0p5_injectASR_transcribed_before", margin=3.0, kind="super"),
    Cmp("UCF ASR before vs after the question", "trackB_ucf_crime_cascade_k8_f0p5_injectASR_transcribed",
        "trackB_ucf_crime_cascade_k8_f0p5_injectASR_transcribed_before", margin=3.0, kind="super"),
    Cmp("n900 ASR before-question within cascade", "trackB_long_n900_k8_cascade_f0p5",
        "trackB_long_n900_k8_cascade_f0p5_injectASR_before", margin=3.0, kind="super"),
    Cmp("n900 cascade+ASR before-question vs uniform", "trackB_long_n900_k8_uniform",
        "trackB_long_n900_k8_cascade_f0p5_injectASR_before", margin=5.0, kind="super"),
    Cmp("n900 ASR before vs after the question (cascade)", "trackB_long_n900_k8_cascade_f0p5_injectASR",
        "trackB_long_n900_k8_cascade_f0p5_injectASR_before", margin=3.0, kind="super"),
    Cmp("n900 uniform+ASR before-question vs uniform", "trackB_long_n900_k8_uniform",
        "trackB_long_n900_k8_uniform_injectASR_before", margin=3.0, kind="super"),
    # appended last: the shared RNG stream makes the order of the battery matter
    Cmp("Qwen2-VL n900 uniform+ASR (after) vs uniform", "trackB_long_n900_k8_qwen2vl_uniform",
        "trackB_long_n900_k8_qwen2vl_uniform_injectASR", margin=3.0, kind="super"),
    Cmp("Qwen2-VL n900 uniform+ASR (before) vs uniform", "trackB_long_n900_k8_qwen2vl_uniform",
        "trackB_long_n900_k8_qwen2vl_uniform_injectASR_before", margin=3.0, kind="super"),
    Cmp("Qwen2-VL n900 uniform+ASR before vs after", "trackB_long_n900_k8_qwen2vl_uniform_injectASR",
        "trackB_long_n900_k8_qwen2vl_uniform_injectASR_before", margin=3.0, kind="super"),
    # F1 baseline check: segment-midpoint uniform instead of endpoint-inclusive uniform at K=8
    Cmp("n900 midpoint uniform vs uniform", "trackB_long_n900_k8_uniform",
        "trackB_long_n900_k8_uniform_center", margin=3.0, kind="equiv"),
    Cmp("n900 cascade vs midpoint uniform", "trackB_long_n900_k8_uniform_center",
        "trackB_long_n900_k8_cascade_f0p5", margin=5.0, kind="equiv"),
    Cmp("n900 AKS port vs midpoint uniform", "trackB_long_n900_k8_uniform_center",
        "trackB_long_n900_k8_aks", margin=5.0, kind="equiv"),
    # Evidence-placement control: the same uniform frames and the same ASR text in four prompt positions
    Cmp("placement: ASR after instruction vs no text", "trackB_long_n900_k8_uniform",
        "trackB_long_n900_k8_uniform_injectASR", margin=3.0, kind="super"),
    Cmp("placement: ASR between options and instruction vs after",
        "trackB_long_n900_k8_uniform_injectASR", "trackB_long_n900_k8_uniform_injectASR_instruction",
        margin=3.0, kind="super"),
    Cmp("placement: ASR between question and options vs after", "trackB_long_n900_k8_uniform_injectASR",
        "trackB_long_n900_k8_uniform_injectASR_question", margin=3.0, kind="super"),
    Cmp("placement: ASR before the question vs after", "trackB_long_n900_k8_uniform_injectASR",
        "trackB_long_n900_k8_uniform_injectASR_before", margin=3.0, kind="super"),
    Cmp("placement: ASR between options and instruction vs no text", "trackB_long_n900_k8_uniform",
        "trackB_long_n900_k8_uniform_injectASR_instruction", margin=3.0, kind="super"),
    Cmp("placement: ASR between question and options vs no text", "trackB_long_n900_k8_uniform",
        "trackB_long_n900_k8_uniform_injectASR_question", margin=3.0, kind="super"),
    Cmp("placement: ASR before vs between question and options", "trackB_long_n900_k8_uniform_injectASR_question",
        "trackB_long_n900_k8_uniform_injectASR_before", margin=3.0, kind="super"),
]


def run_one(c: Cmp) -> dict:
    a, b = pair(c.a_dir, c.b_dir, c.first_n)
    n = len(a)
    acc_a, acc_b = a.mean() * 100, b.mean() * 100
    b01, b10, p = mcnemar_exact_p(a, b)
    delta, lo95, hi95 = bootstrap_ci(a, b, 0.95)
    is_eq, lo90, hi90 = tost_equivalent(a, b, c.margin)
    mde = mde_paired(a, b)
    power = post_hoc_power(a, b)
    return dict(label=c.label, n=n, acc_a=acc_a, acc_b=acc_b, delta=delta,
                lo95=lo95, hi95=hi95, p=p, b01=b01, b10=b10,
                margin=c.margin, is_eq=is_eq, lo90=lo90, hi90=hi90,
                mde=mde, power=power, kind=c.kind)


def fmt_console(r: dict) -> str:
    if r["kind"] == "equiv":
        verdict = (f"EQUIVALENT (90% CI [{r['lo90']:+.1f},{r['hi90']:+.1f}] "
                   f"within +/-{r['margin']:.0f}pp)") if r["is_eq"] else \
                  (f"NOT-equivalent (90% CI [{r['lo90']:+.1f},{r['hi90']:+.1f}] "
                   f"exceeds +/-{r['margin']:.0f}pp)")
    else:
        sig = "SIGNIFICANT" if r["p"] < 0.05 else "n.s."
        verdict = f"effect {sig}"
    return (
        f"{r['label']}\n"
        f"    n={r['n']}  acc {r['acc_a']:.1f} -> {r['acc_b']:.1f}  "
        f"delta={r['delta']:+.1f}pp  95%CI[{r['lo95']:+.1f},{r['hi95']:+.1f}]\n"
        f"    McNemar exact p={r['p']:.3f} (b01={r['b01']}, b10={r['b10']})  "
        f"power={r['power']:.2f}  MDE@80%={r['mde']:.1f}pp\n"
        f"    => {verdict}"
    )


def fmt_latex_row(r: dict) -> str:
    eq = ("\\checkmark" if r["is_eq"] else "--") if r["kind"] == "equiv" else "n/a"
    return (f"{r['label']} & {r['n']} & {r['acc_a']:.1f} & {r['acc_b']:.1f} & "
            f"{r['delta']:+.1f} & [{r['lo95']:+.1f},{r['hi95']:+.1f}] & "
            f"{r['p']:.3f} & {r['power']:.2f} & {r['mde']:.1f} & {eq} \\\\")


# --- New-datasets headline summary (Wave 9) ------------------------------
# EgoTextVQA question-type categories reported in the source paper's Table 3.
# The dataset's task_type field also carries *scenario* categories (kitchen,
# shopping, ...) which are a different axis and are NOT in that table, so the
# paper-comparable macro average is taken over the question-type subset only.
_EGOTEXT_PAPER_CATS = ["location", "direction", "description",
                       "intention reasoning", "others"]

_NEW_DATASETS = [
    ("MUSIC-AVQA", "wave9fix_qwen2vl_musicavqa_cascade_injectASR"),
    ("EgoTextVQA", "wave9fix_qwen2vl_egotextvqa_cascade_injectOCR"),
    ("TutorialVQA", "wave9fix_qwen2vl_tutorialvqa_cascade_injectASR"),
]


def _rescore_rows(dirname: str) -> list[dict]:
    """Load rows.jsonl and recompute correctness with the *current* fixed
    scorer in run_qa_benchmark.py, so span-scored datasets (TutorialVQA) and
    number-word canon (MUSIC-AVQA) reflect the latest metric without a rerun."""
    import sys
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from router.qa_datasets import normalize_answer, QAItem
    bench = os.path.join(ROOT, "scripts", "run_qa_benchmark.py")
    src = open(bench).read()
    ns = {"normalize_answer": normalize_answer, "QAItem": QAItem}
    exec(src[src.index("def _token_f1"):src.index("def main()")], ns)  # noqa: S102
    _score = ns["_score"]
    rows = []
    path = os.path.join(OUT, dirname, "rows.jsonl")
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            it = QAItem(qid=str(r.get("qid", "")), video_path="", question="q",
                        options=r.get("options") or [], answer=r["gold"],
                        task_type=str(r.get("task_type", "")))
            r["correct"] = int(bool(_score(it, r["pred"],
                                           r["pred"] if it.is_mcq else "")))
            rows.append(r)
    return rows


def summarize_new_datasets(emit_latex: bool = False) -> None:
    print("=" * 78)
    print("New-datasets headline (Wave 9, Qwen2-VL-7B, audited cascade + evidence)")
    print("=" * 78)
    latex_rows = []
    for name, dirname in _NEW_DATASETS:
        try:
            rows = _rescore_rows(dirname)
        except FileNotFoundError:
            print(f"[skip] {name}: missing {dirname}/rows.jsonl")
            continue
        n = len(rows)
        micro = sum(r["correct"] for r in rows) / n * 100.0
        line = f"{name:12} n={n:4}  micro={micro:5.1f}"
        macro_paper = None
        if name == "EgoTextVQA":
            from collections import defaultdict
            tot, cor = defaultdict(int), defaultdict(int)
            for r in rows:
                tt = r.get("task_type") or "NA"
                tot[tt] += 1
                cor[tt] += r["correct"]
            macro_all = sum(cor[k] / tot[k] for k in tot) / len(tot) * 100.0
            paper = [k for k in _EGOTEXT_PAPER_CATS if k in tot]
            macro_paper = (sum(cor[k] / tot[k] for k in paper) / len(paper)
                           * 100.0) if paper else float("nan")
            line += (f"  macro(all {len(tot)})={macro_all:5.1f}"
                     f"  macro(paper {len(paper)} q-types)={macro_paper:5.1f}")
        print(line)
        latex_rows.append((name, n, micro, macro_paper))

    if emit_latex:
        print("\n% --- New-datasets LaTeX body (paste into experiments.tex) ---")
        print("% cols: Dataset & n & Micro-Acc & Macro-Acc(paper q-types)")
        for name, n, micro, macro_paper in latex_rows:
            mp = f"{macro_paper:.1f}" if macro_paper is not None else "--"
            print(f"{name} & {n} & {micro:.1f} & {mp} \\\\")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true", help="emit LaTeX table body")
    ap.add_argument("--new-datasets", action="store_true",
                    help="only print the Wave-9 new-datasets headline summary")
    args = ap.parse_args()

    if args.new_datasets:
        summarize_new_datasets(emit_latex=args.latex)
        return

    results = []
    print("=" * 78)
    print("VideoRouteAudit statistical hardening  (McNemar exact / bootstrap CI / TOST / power)")
    print("=" * 78)
    for c in BATTERY:
        try:
            r = run_one(c)
        except FileNotFoundError as e:
            print(f"[skip] {c.label}: missing {e.filename}")
            continue
        results.append(r)
        print(fmt_console(r))
        print("-" * 78)

    if args.latex:
        print("\n% --- LaTeX table body (paste into experiments.tex) ---")
        print("% cols: Comparison & n & AccA & AccB & Delta & 95% CI & p & power & MDE & equiv")
        for r in results:
            print(fmt_latex_row(r))

    print()
    summarize_new_datasets(emit_latex=args.latex)


if __name__ == "__main__":
    main()
