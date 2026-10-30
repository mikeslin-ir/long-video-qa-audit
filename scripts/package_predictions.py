#!/usr/bin/env python3
"""Package the per-item outcomes of the paper's runs into ``results/predictions/``.

Authors only: it reads the raw run directories (``outputs/<run>/``), which are
not released. The result is a compact, deterministic table that every statistic
in the paper can be recomputed from (see ``scripts/replay_paper_stats.py``):

    results/predictions/predictions.csv.gz   run, qid, correct, pred
    results/predictions/manifest.csv         run, group, dataset, backbone, ...

Only ids, the 0/1 outcome and the model's own short answer are stored; no
videos, frames, transcripts, questions or gold answers.

    python scripts/package_predictions.py \
        --outputs <raw outputs dir> --outputs outputs
    python scripts/package_predictions.py --verify --outputs ...
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import audit_predictions as ap  # noqa: E402

Q25, Q2, LNV, AV = ("Qwen2.5-VL-7B", "Qwen2-VL-7B", "LLaVA-NeXT-Video-7B",
                    "VideoLLaMA2.1-7B-AV")

# (run, group, dataset, backbone, description)
RUNS: list[tuple[str, str, str, str, str]] = [
    # --- Video-MME long, full bucket, K=8 (main table) -------------------------
    ("trackB_long_n900_k8_uniform", "vmme900", "video_mme_long", Q25, "uniform"),
    ("trackB_long_n900_k8_aks", "vmme900", "video_mme_long", Q25, "AKS port"),
    ("trackB_long_n900_k8_keyscore", "vmme900", "video_mme_long", Q25, "KeyScore port"),
    ("trackB_long_n900_k8_videotree", "vmme900", "video_mme_long", Q25, "VideoTree port"),
    ("trackB_long_n900_k8_cascade_f0p5", "vmme900", "video_mme_long", Q25, "cascade, 50% uniform anchors"),
    ("trackB_long_n900_k8_cascade_f0p5_injectASR", "vmme900", "video_mme_long", Q25, "cascade + ASR"),
    ("trackB_long_n900_k8_cascade_f0p5_injectOCR", "vmme900", "video_mme_long", Q25, "cascade + OCR"),
    ("trackB_long_n900_k8_llmPlanner_proceduralV3_injectOnly", "vmme900", "video_mme_long", Q25, "procedural-memory planner (v3, inject-only)"),
    # --- Second backbone at full scale ---------------------------------------
    ("trackB_long_n900_k8_qwen2vl_uniform", "vmme900_qwen2vl", "video_mme_long", Q2, "uniform"),
    ("trackB_long_n900_k8_qwen2vl_uniform_injectASR", "vmme900_qwen2vl", "video_mme_long", Q2, "uniform + ASR, block after the question"),
    ("trackB_long_n900_k8_qwen2vl_uniform_injectASR_before", "vmme900_qwen2vl", "video_mme_long", Q2, "uniform + ASR, block before the question"),
    ("trackB_long_n900_k8_qwen2vl_cascade_f0p5", "vmme900_qwen2vl", "video_mme_long", Q2, "cascade, 50% uniform anchors"),
    ("trackB_long_n900_k8_qwen2vl_cascade_f0p5_injectASR", "vmme900_qwen2vl", "video_mme_long", Q2, "cascade + ASR (original run: larger candidate pool than the cascade arm)"),
    ("trackB_long_n900_k8_qwen2vl_cascade_f0p5_injectASR_audit", "vmme900_qwen2vl", "video_mme_long", Q2, "cascade + ASR, same candidate pool as the cascade arm"),
    # --- Video-MME long, n=200 prefix, K=8 -----------------------------------
    ("trackB_long_uniform_k8", "vmme200", "video_mme_long", Q25, "uniform"),
    ("trackB_long_random_k8", "vmme200", "video_mme_long", Q25, "random"),
    ("trackB_long_motion_k8", "vmme200", "video_mme_long", Q25, "motion-difference only"),
    ("trackB_long_clip_k8", "vmme200", "video_mme_long", Q25, "SigLIP only"),
    ("trackB_long_cascade_k8_f0p5", "vmme200", "video_mme_long", Q25, "cascade, 50% uniform anchors"),
    ("trackB_long_cascade_k8_relaxed_f0p5", "vmme200", "video_mme_long", Q25, "forced multi-cue cascade"),
    ("trackB_long_uniform_k8_injectASR", "vmme200", "video_mme_long", Q25, "uniform + ASR"),
    ("trackB_long_cascade_k8_f0p5_injectASR", "vmme200", "video_mme_long", Q25, "cascade + ASR"),
    ("trackB_long_cascade_k8_f0p5_injectOCR", "vmme200", "video_mme_long", Q25, "cascade + OCR"),
    ("trackB_long_cascade_k8_f0p5_injectASR_OCR", "vmme200", "video_mme_long", Q25, "cascade + ASR + OCR"),
    ("trackB_long_cascade_k8_llmPlanner_planInject_proceduralV3_injectOnly", "vmme200", "video_mme_long", Q25, "planner (v3, inject-only)"),
    ("trackB_long_qwen2vl_k8_llmPlanner_proceduralV3_injectOnly", "vmme200", "video_mme_long", Q2, "planner (v3, inject-only)"),
    # --- budget sweep, n=200 -------------------------------------------------
    *[(f"trackB_long_uniform_k{k}", "ksweep200", "video_mme_long", Q25, f"uniform K={k}") for k in (1, 2, 4, 6)],
    *[(f"trackB_long_cascade_k{k}_f0p5", "ksweep200", "video_mme_long", Q25, f"cascade K={k}") for k in (1, 2, 4, 6)],
    # --- event-sparse anomaly domains, n=200, K=8 ----------------------------
    *[(f"trackB_{d}_{s}", "anomaly200", d, Q25, desc)
      for d in ("xd_violence", "ucf_crime")
      for s, desc in (("uniform_k8", "uniform"),
                      ("cascade_k8_f0p5", "cascade, 50% uniform anchors"),
                      ("cascade_k8_f0p5_injectASR", "cascade + ASR (no transcripts existed: nothing injected)"),
                      ("cascade_k8_f0p5_injectASR_transcribed", "cascade + ASR, transcripts built first"),
                      ("cascade_k8_f0p5_injectASR_transcribed_before", "cascade + ASR, transcripts, evidence placed before the question"),
                      ("cascade_k8_f0p5_injectOCR", "cascade + OCR"),
                      ("cascade_k8_llmPlanner_planInject_proceduralV1", "planner (v1)"))],
    # --- Video-MME long, full bucket, current default planner profile --------
    ("trackB_long_n900_k8_stacfp_cascade_f0p5", "vmme900_stacfp", "video_mme_long", Q25, "cascade, STACFP profile"),
    ("trackB_long_n900_k8_stacfp_cascade_f0p5_injectASR", "vmme900_stacfp", "video_mme_long", Q25, "cascade + ASR, STACFP profile"),
    # --- MUSIC-AVQA, full split, LLM-judge verdicts --------------------------
    ("wave9base_qwen2vl_music_avqa_FULL_noinject", "music_avqa", "music_avqa", Q2, "audio-blind VLM"),
    ("wave9av_videollama2av_music_avqa_FULL", "music_avqa", "music_avqa", AV, "audio-visual expert, soundtrack on"),
    ("audio_ablation_silent_stratified200", "music_avqa_silent", "music_avqa", AV, "audio-visual expert, soundtrack silenced (200 per type)"),
    ("audio_ablation_silent_rest", "music_avqa_silent", "music_avqa", AV, "audio-visual expert, soundtrack silenced (remaining questions)"),
    # --- EgoSchema, n=500, K=8 -----------------------------------------------
    *[(f"ksweep_{s}_k{k}_egoschema", "egoschema_early", "egoschema", Q25, f"{s} K={k} (early pipeline)")
      for s in ("uniform", "cascade", "random") for k in (4, 8)],
    ("anchored_cascade_k8_f0p5_egoschema", "egoschema_early", "egoschema", Q25, "cascade, 50% uniform anchors (early pipeline)"),
    *[(f"egoschema_n500_k8_{s}", "egoschema", "egoschema", Q25, s)
      for s in ("uniform", "cascade", "aks", "keyscore", "videotree")],
    # --- Video-MME long, full bucket, K=1 ------------------------------------
    ("trackB_long_n900_k1_uniform", "vmme900_k1", "video_mme_long", Q25, "uniform K=1"),
    ("trackB_long_n900_k1_uniform_center", "vmme900_k1", "video_mme_long", Q25, "uniform K=1, middle frame"),
    ("trackB_long_n900_k1_cascade_f0p5", "vmme900_k1", "video_mme_long", Q25, "cascade K=1"),
    ("trackB_long_n900_k8_uniform_center", "vmme900", "video_mme_long", Q25, "uniform, segment midpoints"),
    ("trackB_long_n900_k8_cascade_f0p5_injectASR_before", "vmme900_before", "video_mme_long", Q25, "cascade + ASR, evidence placed before the question"),
    ("trackB_long_n900_k8_uniform_injectASR_before", "vmme900_before", "video_mme_long", Q25, "uniform + ASR, evidence placed before the question"),
    ("trackB_long_n900_k8_uniform_injectASR", "vmme900_before", "video_mme_long", Q25, "uniform + ASR, evidence after the answer instruction (audited layout)"),
    ("trackB_long_n900_k8_uniform_injectASR_question", "vmme900_before", "video_mme_long", Q25, "uniform + ASR, evidence between the question and the options"),
    ("trackB_long_n900_k8_uniform_injectASR_instruction", "vmme900_before", "video_mme_long", Q25, "uniform + ASR, evidence between the options and the answer instruction"),
    # --- LLaVA-NeXT-Video cross-backbone check, n=200 ------------------------
    *[(f"xbackbone_lnv{ds}_{s}", "xbackbone_lnv", dn, LNV, s)
      for ds, dn in (("", "video_mme_long"), ("_ucf_crime", "ucf_crime"), ("_xd_violence", "xd_violence"))
      for s in ("uniform", "cascade", "cascadeASR")],
]


def finished(root: Path, run: str) -> bool:
    """A run is packageable once its runner wrote ``qa_summary.json``. Sharded runs
    (``scripts/paper_runs/run_sharded.sh``) leave ``shard*.log`` files and write a
    ``qa_summary.json`` per shard, so they also need the ``DONE`` marker of the driver."""
    run_dir = root / run
    if not (run_dir / "qa_summary.json").exists():
        return False
    return not list(run_dir.glob("shard*.log")) or (run_dir / "DONE").exists()


def find_live(run: str, roots: list[Path]):
    for root in roots:
        outcomes = ap.live_correct(run, root)
        if outcomes is not None:
            return outcomes, root
    return None, None


def read_pred(run: str, root: Path) -> dict[str, tuple[str, str, str]]:
    """qid -> (the model's short answer, task type, injected-evidence length)."""
    path = root / run / "rows.jsonl"
    if not path.exists():
        return {}
    return {str(r["qid"]): (str(r.get("pred", ""))[:80], str(r.get("task_type", "")),
                            str(int(r.get("evidence_text_chars") or 0)))
            for r in ap._jsonl(path)}


def read_decisions(run: str, root: Path) -> list[tuple[str, str, str, str]]:
    """(run, qid, planner, inject_text) for runs whose plans came from an LLM planner."""
    path = root / run / "rows.jsonl"
    out = []
    for r in ap._jsonl(path) if path.exists() else ():
        plan = r.get("planner_output") or {}
        if str(plan.get("planner", "")).startswith("llm"):
            out.append((run, str(r["qid"]), str(plan.get("planner", "")),
                        "+".join(plan.get("inject_text") or [])))
    return out


def modal_proposals(run_dir: Path, sample: int = 120) -> int | None:
    """Most common number of candidate frames the cascade actually proposed for a
    long video, from the per-item ``cascade_result.json`` files (None if absent).
    Short videos (at most 64 proposals, uniform stride only) are not counted."""
    import json
    from collections import Counter
    files = sorted(run_dir.glob("*/cascade_result.json"))
    if not files:
        return None
    step = max(1, len(files) // sample)
    counts = Counter()
    for path in files[::step]:
        try:
            counts[json.loads(path.read_text()).get("n_proposals")] += 1
        except (OSError, ValueError):
            continue
    counts.pop(None, None)
    counts = Counter({n: c for n, c in counts.items() if n > 64})
    return counts.most_common(1)[0][0] if counts else None


def provenance(run: str, root: Path) -> dict[str, str]:
    """What the run actually executed, read back from its own per-item records.

    ``cascade_profile`` is ``audit`` (shot-boundary + motion-difference), ``stacfp``
    (STACFP + redundancy filter), ``llm`` (LLM planner picks its own tools),
    ``unrecorded`` (an early run that did not log its plan) or empty (no cascade).
    ``n_injected`` counts the items whose prompt really received evidence text.
    """
    path = root / run / "rows.jsonl"
    if not path.exists():
        return {"cascade_profile": "", "inject_mode": "", "n_injected": ""}
    rows = list(ap._jsonl(path))
    if not any(str(r.get("selection_method", "")).startswith("cascade") for r in rows):
        profile = ""
    else:
        plans = [r.get("planner_output") or {} for r in rows]
        planners = {str(p.get("planner", "")) for p in plans if p}
        if not any(plans):
            profile = "unrecorded"
        elif "llm" in planners:
            profile = "llm"
        else:
            joined = {name for p in plans for name in (*p.get("proposers", ()), *p.get("filters", ()))}
            if "shot_boundary" in joined or "motion_diff" in joined:
                profile = "audit"
            elif "stacfp" in joined or "redundancy" in joined:
                profile = "stacfp"
            else:
                profile = "uniform-only"
            if "llm+ruleBase" in planners:
                profile += "+llm-inject"
    modes = {r.get("inject_evidence_text") for r in rows} - {None, "none"}
    position = ""
    if modes:
        try:
            import json
            position = str(json.loads((root / run / "qa_summary.json").read_text())
                           .get("args", {}).get("evidence_position", "after"))
        except (OSError, ValueError):
            position = "after"
    modal, pool = "", ""
    if profile and profile != "unrecorded":
        modal_n, planned = modal_proposals(root / run), max((len(r.get("planner_output", {}).get("proposers", ()))
                                                              for r in rows), default=0)
        k_prop = max((int(r.get("planner_output", {}).get("K_proposal") or 200) for r in rows), default=200)
        if modal_n is not None:
            modal = str(modal_n)
            if planned > 1:
                expected = (k_prop // planned) * planned
                pool = "as planned" if modal_n >= 0.95 * expected else "smaller than planned (a proposer was inactive)"
    elif profile == "unrecorded":
        modal_n = modal_proposals(root / run)
        modal = "" if modal_n is None else str(modal_n)
    return {"cascade_profile": profile, "modal_n_proposals": modal, "candidate_pool": pool,
            "inject_mode": "+".join(sorted(modes)), "evidence_position": position,
            "n_injected": str(sum(1 for r in last_row_per_item(rows) if r.get("evidence_text_chars")))}


def last_row_per_item(rows):
    """One row per qid, the last one written: a resumed or sharded run can contain an item twice."""
    return list({str(r["qid"]): r for r in rows}.values())


def build(roots: list[Path]):
    rows, manifest, missing, decisions = [], [], [], []
    for run, group, dataset, backbone, desc in RUNS:
        outcomes, root = find_live(run, roots)
        if outcomes is None:
            missing.append(run)
            continue
        judged = (root / run / "judged_rows.jsonl").exists()
        if not finished(root, run):
            missing.append(f"{run} (still running)")
            continue
        if group.startswith("music_avqa") and not judged:
            missing.append(f"{run} (not yet judged)")
            continue
        preds = read_pred(run, root)
        for qid, ok in outcomes.items():
            rows.append((run, qid, ok, *preds.get(qid, ("", "", ""))))
        decisions.extend(read_decisions(run, root))
        manifest.append({"run": run, "group": group, "dataset": dataset, "backbone": backbone,
                         "description": desc, "n": len(outcomes),
                         "accuracy": f"{100 * sum(outcomes.values()) / len(outcomes):.1f}",
                         "scoring": "llm_judge" if judged else "exact_match",
                         **provenance(run, root)})
    return rows, manifest, missing, decisions


def write(rows, manifest, decisions) -> None:
    ap.PRED_DIR.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["run", "qid", "correct", "pred", "task_type", "evidence_chars"])
    w.writerows(rows)
    with open(ap.PACKAGED, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        gz.write(buf.getvalue().encode())
    dec = io.StringIO()
    w = csv.writer(dec, lineterminator="\n")
    w.writerow(["run", "qid", "planner", "inject_text"])
    w.writerows(decisions)
    with open(ap.PRED_DIR / "planner_decisions.csv.gz", "wb") as raw, \
            gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        gz.write(dec.getvalue().encode())
    with open(ap.MANIFEST, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(manifest)


def verify(roots: list[Path]) -> int:
    ap._packaged.cache_clear()
    packaged, bad, checked = ap._packaged(), 0, 0
    for run, group, *_ in RUNS:
        live, root = find_live(run, roots)
        if live is None or not finished(root, run):
            continue
        if group.startswith("music_avqa") and not (root / run / "judged_rows.jsonl").exists():
            continue
        checked += 1
        if packaged.get(run) != live:
            bad += 1
            print(f"MISMATCH {run}")
    print(f"verified {checked} runs against the working tree: {bad} mismatches")
    return 1 if bad else 0


def main() -> int:
    ex = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ex.add_argument("--outputs", action="append", default=[], help="outputs directory (repeatable)")
    ex.add_argument("--verify", action="store_true", help="check the package against the working tree")
    args = ex.parse_args()
    roots = [Path(p).resolve() for p in (args.outputs or [str(ap.outputs_dir())])]
    if args.verify:
        return verify(roots)
    rows, manifest, missing, decisions = build(roots)
    write(rows, manifest, decisions)
    print(f"packaged {len(manifest)} runs, {len(rows)} per-item outcomes -> {ap.PACKAGED}")
    if missing:
        print(f"not found (skipped): {len(missing)}")
        for m in missing:
            print("  -", m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
