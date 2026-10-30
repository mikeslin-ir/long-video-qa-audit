#!/usr/bin/env python3
"""End-to-end QA benchmark for Agentic Multimodal Routing.

Pipeline per item:
  1. Route + select frames with SkillRouter (optionally submodular, memory).
  2. Send selected frames (and routed expert) to a real multimodal expert.
  3. Score the answer against gold -> accuracy.
  4. (optional) Run the counterfactual oracle on a subset -> evidence necessity.
  5. (optional) Write a memory record (q, plan, reward) for procedural memory.

Usage:
  python scripts/run_qa_benchmark.py --dataset egoschema --limit 20 \
      --expert vlm --real --use_submodular --output_dir outputs/qa_egoschema

  # add memory writing + oracle diagnostics on first 50:
  python scripts/run_qa_benchmark.py --dataset video_mme --limit 200 \
      --real --use_submodular --memory --oracle_limit 50
"""
from __future__ import annotations

import argparse
import cv2
import json
import signal
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from router.qa_datasets import load_dataset, normalize_answer, QAItem
from skill_router import SkillRouter
from baselines import make as make_baseline


class _ItemTimeout(Exception):
    pass


def _install_timeout(seconds: int):
    if seconds <= 0:
        return
    def _handler(signum, frame):
        raise _ItemTimeout(f"item exceeded {seconds}s")
    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)


def _clear_timeout():
    try:
        signal.alarm(0)
    except Exception:
        pass


def _read_rgb(path):
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

# per-dataset routing defaults
DEFAULTS = {
    "egoschema": {"budget": 8, "tau_sec": 2.0, "config": "configs/egoschema_skill_router.yaml"},
    "video_mme": {"budget": 8, "tau_sec": 2.0, "config": "configs/video_mme_skill_router.yaml"},
    "msvd_qa":   {"budget": 6, "tau_sec": 0.8, "config": "configs/msvd_qa_skill_router.yaml"},
    "music_avqa": {"budget": 6, "tau_sec": 1.0, "config": "configs/music_avqa_skill_router.yaml"},
    "egotextvqa": {"budget": 8, "tau_sec": 1.0, "config": "configs/egoschema_skill_router.yaml"},
    "tutorialvqa": {"budget": 8, "tau_sec": 1.0, "config": "configs/video_mme_skill_router.yaml"},
    "ucf_crime": {"budget": 8, "tau_sec": 2.0, "config": "configs/egoschema_skill_router.yaml"},
    "xd_violence": {"budget": 8, "tau_sec": 2.0, "config": "configs/egoschema_skill_router.yaml"},
    "mlvu": {"budget": 8, "tau_sec": 2.0, "config": "configs/video_mme_skill_router.yaml"},
}


def _token_f1(pred: str, gold: str) -> float:
    """Token-level F1 between two normalized strings (open-ended QA)."""
    from collections import Counter
    pt, gt = pred.split(), gold.split()
    if not pt or not gt:
        return 0.0
    common = Counter(pt) & Counter(gt)
    n = sum(common.values())
    if n == 0:
        return 0.0
    p, r = n / len(pt), n / len(gt)
    return 2 * p * r / (p + r)


_NUM_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
}


def _canon_numbers(s: str) -> str:
    """Map number words to digits token-wise so counting answers compare
    regardless of surface form ('three' == '3', 'four' == '4')."""
    return " ".join(_NUM_WORDS.get(t, t) for t in s.split())


def _lcs_len(a: list[str], b: list[str]) -> int:
    """Token-level longest-common-subsequence length (ROUGE-L core)."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


# Trivial/stopword answers that must NOT pass span scoring by coincidentally
# appearing inside a long transcript gold (e.g. "yes" occurs in many spans).
_SPAN_STOPWORDS = {
    "yes", "no", "the", "a", "an", "to", "of", "is", "it", "this", "that",
    "and", "or", "i", "you", "s", "t",
}
# Min fraction of the prediction's tokens that must lie within the gold span.
_SPAN_RECALL_TAU = 0.6


def _span_recall(pred: str, gold: str) -> float:
    """How much of the (short) prediction is covered by the (long) gold span.

    TutorialVQA gold answers are 100+ word transcript paragraphs while the VLM
    emits a concise phrase (e.g. "content aware fill"). Token-F1 against such a
    long gold is mathematically tiny, so we instead measure pred-side LCS
    recall: the fraction of the prediction that appears, in order, within the
    span. Pure stopword/yes-no predictions are rejected so they cannot match a
    long gold by chance.
    """
    pt = pred.split()
    content = [t for t in pt if t not in _SPAN_STOPWORDS]
    if len(content) < 1:
        return 0.0
    gt = gold.split()
    return _lcs_len(pt, gt) / max(len(pt), 1)


def _score(item: QAItem, pred_answer: str, pred_letter: str) -> bool:
    if item.is_mcq:
        return bool(pred_letter) and pred_letter == item.answer
    pred = normalize_answer(pred_answer)
    gold = item.answer  # already normalized at load time
    if not gold:
        return False
    # Canonicalize number words <-> digits (MUSIC-AVQA counting answers are
    # words like "three" while the VLM emits "3").
    pred_c, gold_c = _canon_numbers(pred), _canon_numbers(gold)
    if pred == gold or pred_c == gold_c:
        return True
    # Span-answer datasets (e.g. TutorialVQA) store the gold as a long
    # transcript paragraph; the VLM answers concisely. Token-F1 against a
    # 100+ word gold can never reach 0.5, so for long golds we instead accept
    # when the prediction is largely contained within the gold span.
    gold_tokens = gold_c.split()
    if len(gold_tokens) >= 12:
        return _span_recall(pred_c, gold_c) >= _SPAN_RECALL_TAU
    # Open-ended matching: VLM output is free-form, so strict equality is far
    # too harsh ("one instrument" vs "one", "Monster is more expensive" vs
    # "monster is more expensive"). Accept (a) whole-phrase containment for
    # short gold answers and (b) token-F1 >= 0.5 for longer/generative answers.
    if len(gold_tokens) <= 5:
        if f" {gold_c} " in f" {pred_c} " or (pred_c and f" {pred_c} " in f" {gold_c} "):
            return True
    return _token_f1(pred_c, gold_c) >= 0.5




def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DEFAULTS))
    ap.add_argument("--selector", default="ours",
                    choices=["ours", "uniform", "uniform_center", "random", "motion", "clip",
                             "aks", "keyscore", "videotree", "sevila", "videoagent"],
                    help="frame selector: ours=SkillRouter; uniform (endpoint-"
                         "inclusive)/uniform_center (segment midpoints)/random "
                         "(trivial); motion/clip (content-aware single-skill "
                         "baselines); aks (Tang et al. CVPR'25, matched-feature "
                         "native port); sevila/videoagent (external SoTA wrappers, "
                         "require the respective repo on PYTHONPATH)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--budget", type=int, default=0,
                    help="override frames per question (K). 0 uses dataset default")
    ap.add_argument("--tau_sec", type=float, default=0.0,
                    help="override temporal spacing. 0 uses dataset default")
    ap.add_argument("--expert", default=None,
                    help="force an expert (vlm/alm/avlm/caplm). default=routed")
    ap.add_argument("--real", action="store_true",
                    help="use real Qwen backbones (else CPU stubs)")
    ap.add_argument("--vlm_model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--vlm_backbone", default="qwen2_vl",
                    choices=["qwen2_vl", "llava_video", "llava_next_video",
                             "videollama2_av"],
                    help="VLM backbone class: qwen2_vl (default, Qwen2.5-VL-7B), "
                         "llava_video (LLaVA-Video-7B-Qwen2 from lmms-lab; "
                         "requires the `llava` package), llava_next_video "
                         "(LLaVA-NeXT-Video-7B HF-native, Llama+CLIP, cross-family), "
                         "or videollama2_av "
                         "(VideoLLaMA2.1-7B-AV audio-visual; requires the "
                         "`videollama2` package in an isolated env).")
    ap.add_argument("--use_submodular", action="store_true")
    ap.add_argument("--memory", action="store_true",
                    help="write a memory record per item")
    ap.add_argument("--oracle_limit", type=int, default=0,
                    help="run counterfactual oracle on first N items")
    ap.add_argument("--item_timeout_sec", type=int, default=180,
                    help="hard timeout per item (sel+expert); 0 disables")
    ap.add_argument("--resume", action="store_true", default=True,
                    help="skip items already in rows.jsonl (default on)")
    ap.add_argument("--no_resume", dest="resume", action="store_false",
                    help="recompute all items even if cached")
    ap.add_argument("--gpu_cleanup_every", type=int, default=10,
                    help="call torch.cuda.empty_cache every N items")
    ap.add_argument("--stratify_per_bucket", type=int, default=0,
                    help="if >0, take first N items per task_type bucket "
                         "(e.g. Video-MME short/medium/long). Overrides --limit")
    ap.add_argument("--cascade", action="store_true",
                    help="use the 3-level Agentic Cascade router (L1 proposers "
                         "-> L2 filters -> L3 aligners -> submodular). "
                         "When set, --use_submodular and skill_router config are ignored.")
    ap.add_argument("--cascade_planner", default="rule_based",
                    choices=["rule_based", "relaxed", "llm"],
                    help="planner type for --cascade (default rule_based). "
                         "'relaxed' = rule_based with force_multicue (generic "
                         "questions still get a multi-tool plan).")
    ap.add_argument("--cascade_uniform_floor", type=float, default=0.0,
                    help="fraction of the final budget reserved for evenly "
                         "spaced uniform anchors (Track A; 0.0 = pure cascade)")
    ap.add_argument("--cascade_profile", default="stacfp", choices=["stacfp", "audit"],
                    help="which L1/L2 tools the rule-based plan enables. 'stacfp' "
                         "(default) = STACFP + EARS + uniform stride with the "
                         "redundancy filter. 'audit' = EARS + uniform stride with the "
                         "motion-difference filter, i.e. what the paper's Video-MME / "
                         "XD-Violence / UCF-Crime cascade runs actually executed "
                         "(their shot-boundary proposer was silently inactive); the "
                         "paper-run drivers pass it.")
    ap.add_argument("--cascade_proposer", default="auto",
                    choices=["auto", "keyframe_cache"],
                    help="L1 proposer policy. 'auto' = the profile's proposers "
                         "(see --cascade_profile). 'keyframe_cache' = use "
                         "the offline shot-keyframe cache + cached SigLIP "
                         "aligner (run scripts/extract_shot_keyframe_features.py "
                         "first). Only affects --cascade_planner rule_based/relaxed.")
    ap.add_argument("--inject_evidence_text", default="none",
                    help="text-evidence injection mode: 'none' = off; "
                         "'asr' = always inject Whisper transcript; "
                         "'ocr' = always inject EasyOCR on selected frames; "
                         "'asr+ocr' (or any '+'-joined subset) = both; "
                         "'plan' = let the cascade planner decide per question "
                         "via plan.inject_text.")
    ap.add_argument("--evidence_window_sec", type=float, default=8.0,
                    help="+/- window around each selected-frame timestamp "
                         "used to extract ASR words for injection")
    ap.add_argument("--evidence_position", default="after", choices=["after", "before", "question", "instruction"],
                    help="where the injected ASR/OCR block goes in the prompt: 'after' = "
                         "appended after the options and answer instruction (what the audited "
                         "runs used), 'before' = ahead of the question, 'question' = between "
                         "the question and the options, 'instruction' = between the options "
                         "and the answer instruction. Qwen VLM backbones only.")
    ap.add_argument("--skip_option_logprobs", action="store_true",
                    help="skip the extra forward pass that scores the option letters (Qwen "
                         "backbones). Predictions are unchanged; per-item time drops by about "
                         "a third, and expert_cost_sec is then not comparable to earlier runs.")
    ap.add_argument("--procedural_memory_file", default="",
                    help="path to a markdown/text file of priors appended to "
                         "the LLM planner's system prompt as a 'PROCEDURAL "
                         "MEMORY' block (no-op unless --cascade_planner llm).")
    ap.add_argument("--planner_aligners", default="",
                    help="comma-separated aligner names that hard-override "
                         "the LLM planner's aligner choice (e.g. "
                         "'semantic_siglip'). Used to isolate the inject_text "
                         "decision from drift in the cue mix.")
    ap.add_argument("--planner_inject_only", action="store_true",
                    help="when used with --cascade_planner llm: consult the "
                         "LLM ONLY for inject_text; take every other plan "
                         "field from the rule-based fallback. Cleanest "
                         "isolation of the per-question inject decision.")
    ap.add_argument("--only_task_type", default="",
                    help="keep only items whose task_type matches this value "
                         "(e.g. 'long' for the Video-MME long bucket)")
    ap.add_argument("--only_qids", default="",
                    help="path to a text file with one qid per line ('#' lines "
                         "are comments); keep only those items, in dataset order")
    ap.add_argument("--av_audio_mode", default="on", choices=["on", "silent", "mismatched"],
                    help="videollama2_av only: 'on' feeds the soundtrack "
                         "(default); 'silent' replaces it by digital silence; "
                         "'mismatched' gives each video the soundtrack of another "
                         "video (same-model audio ablations)")
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    d = DEFAULTS[args.dataset]
    cfg_path = args.config or d["config"]
    budget = int(args.budget) if int(args.budget) > 0 else int(d["budget"])
    tau_sec = float(args.tau_sec) if float(args.tau_sec) > 0 else float(d["tau_sec"])
    out_root = Path(args.output_dir or f"outputs/qa_{args.dataset}")
    out_root.mkdir(parents=True, exist_ok=True)

    if int(args.stratify_per_bucket) > 0:
        # Load full split, then take first N per task_type bucket (stable order).
        all_items = load_dataset(args.dataset, limit=0)
        per_bucket: dict[str, list] = {}
        for it in all_items:
            per_bucket.setdefault(it.task_type or "", []).append(it)
        items = []
        n = int(args.stratify_per_bucket)
        for bucket in sorted(per_bucket.keys()):
            items.extend(per_bucket[bucket][:n])
        counts = {b: len(v[:n]) for b, v in per_bucket.items()}
        print(f"[qa] {args.dataset}: stratified per-bucket={n} -> "
              f"{len(items)} items {counts}", flush=True)
    else:
        items = load_dataset(args.dataset, limit=0 if args.only_task_type else args.limit)
        if args.only_task_type:
            want = args.only_task_type.strip().lower()
            items = [it for it in items if (it.task_type or "").strip().lower() == want]
            if int(args.limit) > 0:
                items = items[:int(args.limit)]
            print(f"[qa] {args.dataset}: filtered task_type=={want!r} -> "
                  f"{len(items)} items", flush=True)
        else:
            print(f"[qa] {args.dataset}: {len(items)} items", flush=True)
    if args.only_qids:
        want = {ln.strip() for ln in Path(args.only_qids).read_text().splitlines()
                if ln.strip() and not ln.lstrip().startswith("#")}
        found = {it.qid for it in items}
        missing = sorted(want - found)
        items = [it for it in items if it.qid in want]
        print(f"[qa] --only_qids {args.only_qids}: kept {len(items)}/{len(want)} "
              f"requested" + (f" ({len(missing)} not found, e.g. {missing[:3]})"
                              if missing else ""), flush=True)
    if not items:
        raise SystemExit(f"no items loaded for {args.dataset}")

    # build router (config may not exist -> default config)
    cfg = {}
    if Path(cfg_path).exists():
        import yaml
        cfg = yaml.safe_load(Path(cfg_path).read_text()) or {}
    if args.memory:
        cfg.setdefault("memory", {})["enabled"] = True
        cfg["memory"]["path"] = str(out_root / "memory.jsonl")
    router = SkillRouter(cfg)
    baseline_selector = None
    if args.selector != "ours":
        baseline_selector = make_baseline(args.selector)

    # plug in real backbones if requested
    oracle = None
    if args.real:
        from router.backbones import (Qwen2VLExpert, LLaVAVideoExpert,
                                       LlavaNextVideoExpert, CaptionLLMRealExpert,
                                       set_evidence_position)
        if args.evidence_position != "after":
            if args.vlm_backbone != "qwen2_vl":
                raise SystemExit("--evidence_position other than after is implemented for the "
                                 "qwen2_vl backbone only")
            set_evidence_position(args.evidence_position)
        # MCQ answers are a single letter (16 tokens is plenty); generative
        # open-ended datasets (e.g. TutorialVQA transcript spans) need a longer
        # decode budget so the answer isn't truncated before it can match.
        _GENERATIVE_DS = {"tutorialvqa"}
        vlm_max_new = 64 if args.dataset in _GENERATIVE_DS else 16
        if args.vlm_backbone == "videollama2_av":
            # Audio-visual backbone (VideoLLaMA2-AV): jointly attends to video +
            # soundtrack, required for MUSIC-AVQA. Swaps the whole "vlm"/"avlm"
            # slot so the MUSIC-AVQA column is its own backbone (never mixed with
            # audio-blind Qwen2-VL in the same ablation).
            from router.backbones import VideoLLaMA2AVExpert
            av_id = (args.vlm_model
                     if "videollama" in args.vlm_model.lower()
                     else "DAMO-NLP-SG/VideoLLaMA2.1-7B-AV")
            av_expert = VideoLLaMA2AVExpert(model_id=av_id, device="cuda",
                                            max_new_tokens=max(vlm_max_new, 64),
                                            audio_mode=args.av_audio_mode)
            if args.av_audio_mode == "mismatched":
                # donors come from the whole dataset, so the map does not depend on --only_qids
                av_expert.set_audio_donors(
                    [it.video_path for it in load_dataset(args.dataset, limit=0)])
            router.experts["vlm"] = av_expert
            router.experts["avlm"] = av_expert
        elif args.vlm_backbone == "llava_video":
            # If user left --vlm_model at the Qwen default, swap to LLaVA-Video.
            llava_id = (args.vlm_model
                        if "llava" in args.vlm_model.lower()
                        else "lmms-lab/LLaVA-Video-7B-Qwen2")
            router.experts["vlm"] = LLaVAVideoExpert(model_id=llava_id,
                                                     device="cuda", max_new_tokens=vlm_max_new)
        elif args.vlm_backbone == "llava_next_video":
            # HF-native LLaVA-NeXT-Video-7B (Llama LLM + CLIP vision): a true
            # cross-architecture-family backbone for the robustness check.
            lnv_id = (args.vlm_model
                      if "llava-next-video" in args.vlm_model.lower()
                      or "llava-hf" in args.vlm_model.lower()
                      else "llava-hf/LLaVA-NeXT-Video-7B-hf")
            router.experts["vlm"] = LlavaNextVideoExpert(model_id=lnv_id,
                                                         device="cuda", max_new_tokens=vlm_max_new)
        else:
            router.experts["vlm"] = Qwen2VLExpert(model_id=args.vlm_model,
                                                  device="cuda", max_new_tokens=vlm_max_new)
            if args.skip_option_logprobs:
                router.experts["vlm"].compute_option_logprobs = False
        router.experts["caplm"] = CaptionLLMRealExpert(device="cuda")
        # warm load the vlm (so first item isn't penalized in timing)
        router.experts["vlm"].load()
        if args.oracle_limit > 0:
            from evidence_skills.oracle_vlm import CounterfactualOracle
            oracle = CounterfactualOracle(router.experts)

    # Cascade router (3-level agentic) -- mutually exclusive with baselines
    cascade_router = None
    if args.cascade:
        from cascade import (CascadeRouter, RuleBasedCascadePlanner,
                             LLMCascadePlanner, default_registry)
        reg = default_registry()
        if args.cascade_planner == "llm" and args.real:
            caplm = router.experts.get("caplm")
            def _gen(prompt: str) -> str:
                from router.experts import ExpertEvidence
                # Planner JSON needs ~300 tokens; default caplm.max_new_tokens=32
                # is enough for MCQ answers but truncates planner JSON. Bump it
                # for this call only and restore afterward.
                old = getattr(caplm, "max_new_tokens", 32)
                try:
                    caplm.max_new_tokens = 384
                    out = caplm.infer(ExpertEvidence(question=prompt, metadata={}))
                finally:
                    caplm.max_new_tokens = old
                return out.answer or out.extras.get("raw", "")
            procedural_memory = ""
            if args.procedural_memory_file:
                from pathlib import Path as _P
                procedural_memory = _P(args.procedural_memory_file).read_text()
                print(f"[qa] loaded procedural memory from {args.procedural_memory_file} "
                      f"({len(procedural_memory)} chars)", flush=True)
            planner = LLMCascadePlanner(
                generate_fn=_gen,
                K_final_override=int(budget),
                tau_sec_override=float(tau_sec),
                uniform_anchor_frac_override=float(args.cascade_uniform_floor),
                aligners_override=(tuple(a.strip() for a in args.planner_aligners.split(",") if a.strip())
                                    or None),
                procedural_memory=procedural_memory,
                inject_only_mode=bool(args.planner_inject_only),
                fallback=RuleBasedCascadePlanner(profile=args.cascade_profile),
            )
        else:
            planner = RuleBasedCascadePlanner(K_final=int(budget),
                                              tau_sec=float(tau_sec),
                                              uniform_anchor_frac=float(args.cascade_uniform_floor),
                                              force_multicue=(args.cascade_planner == "relaxed"),
                                              prefer_cache=(args.cascade_proposer == "keyframe_cache"),
                                              profile=args.cascade_profile)
        cascade_router = CascadeRouter(registry=reg, planner=planner)
        print(f"[qa] cascade enabled (planner={args.cascade_planner}, "
              f"profile={args.cascade_profile}); "
              f"tools={reg.available()}", flush=True)

    rows: list[dict] = []
    fails: list[dict] = []
    t0 = time.time()
    n_correct = 0
    inject_errors = 0
    rows_path = out_root / "rows.jsonl"
    fails_path = out_root / "failures.jsonl"
    if args.resume and rows_path.exists() and rows_path.stat().st_size > 0:
        for ln in rows_path.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
                rows.append(r)
                if r.get("correct"):
                    n_correct += 1
            except Exception:
                pass
        print(f"[qa] resume: loaded {len(rows)} prior rows, "
              f"n_correct={n_correct}", flush=True)
    done_qids = {r.get("qid") for r in rows}
    rows_fp = rows_path.open("a", buffering=1)
    fails_fp = fails_path.open("a", buffering=1)

    for i, it in enumerate(items, 1):
        run_dir = out_root / f"{i:05d}_{it.qid}"
        if args.resume and it.qid in done_qids:
            continue

        _install_timeout(int(args.item_timeout_sec))
        try:
            if cascade_router is not None:
                cres = cascade_router.run(
                    video_path=it.video_path,
                    question=it.question,
                    dataset=args.dataset,
                    output_dir=str(run_dir),
                    K_final_override=budget,
                )
                # Adapt CascadeResult -> dict shape the loop expects
                res = {
                    "selected_frames": [
                        {"frame_index": fi, "timestamp_sec": ts,
                         "image_path": p, "fused_score": s}
                        for fi, ts, p, s in zip(
                            cres.selected_frame_indices,
                            cres.selected_timestamps_sec,
                            cres.selected_image_paths,
                            cres.fused_scores,
                        )
                    ],
                    "selection_method": f"cascade_{args.cascade_planner}",
                    "expert_planned": cres.plan.expected_expert,
                    "runtime_sec": cres.runtime_sec.get("total", 0.0),
                    "selected_skills": cres.plan.aligners,
                    "weights": cres.plan.weights,
                    "planner_output": cres.plan.to_json(),
                }
            else:
                res = router.run(
                    video_path=it.video_path,
                    question=it.question,
                    dataset=args.dataset,
                    budget=budget, tau_sec=tau_sec,
                    output_dir=str(run_dir),
                    use_submodular=args.use_submodular,
                    invoke_expert=False,   # we invoke manually to pass options
                ) if args.selector == "ours" else baseline_selector.select(
                    video_path=it.video_path,
                    question=it.question,
                    dataset=args.dataset,
                    budget=budget,
                    tau_sec=tau_sec,
                    output_dir=str(run_dir),
                )

            # Use the router-saved selected frame images directly; avoids
            # re-decoding the entire candidate set per item.
            sel = res.get("selected_frames", [])
            sel_imgs = []
            for s in sel:
                ip = s.get("image_path")
                if not ip:
                    continue
                img = _read_rgb(ip)
                if img is not None:
                    sel_imgs.append(img)

            expert_name = args.expert or res.get("expert_planned") or "vlm"
            expert = router.experts.get(expert_name, router.experts["vlm"])
            from router.experts import ExpertEvidence

            asr_text = ""
            # Decide which cue text streams to inject:
            #   'none'                  -> nothing
            #   'asr' / 'ocr' / 'asr+ocr' -> always inject those modalities
            #   'plan'                  -> use plan.inject_text from planner
            inject_mods: set[str] = set()
            mode = args.inject_evidence_text
            if mode == "plan":
                plan_obj = (res.get("planner_output") or {}) if isinstance(res, dict) else {}
                inject_mods = {m for m in (plan_obj.get("inject_text") or [])
                               if m in {"asr", "ocr"}}
            elif mode and mode != "none":
                inject_mods = {m.strip().lower() for m in mode.split("+")
                               if m.strip().lower() in {"asr", "ocr"}}
            if inject_mods and sel:
                try:
                    from cascade.evidence_text import build_combined_evidence_text
                    ts_list = [float(s.get("timestamp_sec", 0.0)) for s in sel]
                    asr_text = build_combined_evidence_text(
                        it.video_path, sel_imgs, ts_list,
                        modalities=tuple(sorted(inject_mods)),
                        asr_window_sec=float(args.evidence_window_sec))
                except Exception as _e:
                    asr_text = ""
                    inject_errors += 1
                    if inject_errors <= 3:
                        print(f"[qa] WARNING: evidence-text injection failed for "
                              f"{getattr(it, 'qid', '?')}: {_e!r}", flush=True)

            # Audio-visual experts (e.g. VideoLLaMA2-AV on MUSIC-AVQA) consume
            # the raw video (which carries the soundtrack) rather than only the
            # sampled frames, so pass the path through. Optionally pre-extract a
            # mono waveform when the item is flagged audio-dependent.
            needs_audio = bool((it.meta or {}).get("needs_audio")) or \
                args.vlm_backbone == "videollama2_av"
            audio_wave = None
            if needs_audio:
                try:
                    from video_audio import extract_audio_from_video
                    audio_wave = extract_audio_from_video(it.video_path)
                except Exception:
                    audio_wave = None
            ev = ExpertEvidence(question=it.question, frames=sel_imgs or None,
                                asr_text=asr_text or None,
                                audio=audio_wave,
                                metadata={"dataset": args.dataset,
                                          "options": it.options,
                                          "video_path": it.video_path,
                                          "needs_audio": needs_audio})
            out = expert.infer(ev)
            pred_letter = out.answer if (it.is_mcq and len(out.answer) == 1) else \
                (out.extras.get("raw", "") and "")
            pred_letter = out.answer if it.is_mcq else ""
            correct = _score(it, out.answer, pred_letter)
            n_correct += int(correct)

            row = {
                "qid": it.qid, "dataset": args.dataset,
                "expert": expert_name, "pred": out.answer,
                "gold": it.answer, "correct": bool(correct),
                "is_mcq": it.is_mcq, "task_type": it.task_type,
                "selection_method": res.get("selection_method"),
                "expert_planned": res.get("expert_planned"),
                "select_runtime_sec": res.get("runtime_sec"),
                "expert_cost_sec": out.cost_sec,
                "num_selected": len(sel_imgs),
                "selector": args.selector,
                "budget": budget,
                "tau_sec": tau_sec,
                "av_audio_mode": (args.av_audio_mode
                                  if args.vlm_backbone == "videollama2_av" else None),
                "inject_evidence_text": args.inject_evidence_text,
                "audio_donor": (out.extras or {}).get("audio_donor"),
                "evidence_text_chars": int(out.extras.get("evidence_text_chars", 0))
                                       if out.extras else 0,
                "planner_output": res.get("planner_output"),
            }

            # optional counterfactual oracle on first N MCQ items
            if oracle is not None and i <= args.oracle_limit and it.is_mcq and sel_imgs:
                cf = oracle.evaluate(
                    chosen_name=expert_name, question=it.question,
                    frames=sel_imgs, options=it.options,
                    gold_letter=it.answer,
                    metadata={"dataset": args.dataset})
                row["oracle"] = cf.to_json()
                (run_dir / "oracle.json").write_text(json.dumps(cf.to_json(), indent=2))

            # optional memory write (reward = answer logprob proxy or correctness)
            if args.memory and router.memory is not None:
                reward = 1.0 if correct else 0.0
                plan = res.get("planner_output", {})
                router.memory.add(
                    qid=it.qid, dataset=args.dataset, question=it.question,
                    selected_skills=res.get("selected_skills", []),
                    weights=res.get("weights", {}), reward=reward,
                    expert=expert_name, expert_beta=res.get("expert_beta"))

            rows.append(row)
            rows_fp.write(json.dumps(row) + "\n")
        except _ItemTimeout as e:
            fail = {"qid": it.qid, "error": f"timeout: {e}"}
            fails.append(fail)
            fails_fp.write(json.dumps(fail) + "\n")
            print(f"  [skip] {it.qid}: {fail['error']}", flush=True)
        except Exception as e:
            fail = {"qid": it.qid, "error": repr(e)}
            fails.append(fail)
            fails_fp.write(json.dumps(fail) + "\n")
        finally:
            _clear_timeout()

        if args.gpu_cleanup_every > 0 and (i % args.gpu_cleanup_every == 0):
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        if i % 10 == 0 or i == len(items):
            acc = n_correct / max(1, len(rows))
            print(f"  [{i:>5}/{len(items)}] acc={acc:.3f} "
                  f"elapsed={time.time()-t0:.0f}s fails={len(fails)}", flush=True)

    acc = n_correct / max(1, len(rows))
    injection = None
    if args.inject_evidence_text != "none":
        n_empty = sum(1 for r in rows if not r.get("evidence_text_chars"))
        injection = {"mode": args.inject_evidence_text, "n_items": len(rows),
                     "n_empty": n_empty, "n_errors": inject_errors}
        # With a fixed modality an empty text means the evidence was missing (for
        # ASR: no cached transcript for the clip), not that the planner declined.
        if args.inject_evidence_text != "plan" and n_empty > 0.5 * max(1, len(rows)):
            print(f"[qa] WARNING: --inject_evidence_text {args.inject_evidence_text} "
                  f"injected nothing for {n_empty}/{len(rows)} items, so this arm is "
                  f"(nearly) identical to the no-injection arm. For ASR, build the "
                  f"transcripts first with scripts/prewarm_asr_cache.py.", flush=True)
    summary = {
        "dataset": args.dataset, "expert": args.expert or "routed",
        "selector": args.selector,
        "real": args.real, "use_submodular": args.use_submodular,
        "memory": args.memory, "n": len(rows), "n_correct": n_correct,
        "accuracy": acc, "failed": len(fails),
        "elapsed_sec": time.time() - t0,
        "budget": budget,
        "tau_sec": tau_sec,
        "vlm_backbone": args.vlm_backbone,
        "vlm_model": getattr(router.experts.get("vlm"), "model_id", args.vlm_model),
        "av_audio_mode": (args.av_audio_mode
                          if args.vlm_backbone == "videollama2_av" else None),
        "injection": injection,
        # Full command-line configuration, so a run can be reproduced from its
        # own summary (older summaries did not record the flags).
        "args": vars(args),
    }
    (out_root / "qa_summary.json").write_text(json.dumps(summary, indent=2))
    try:
        rows_fp.close()
        fails_fp.close()
    except Exception:
        pass
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
