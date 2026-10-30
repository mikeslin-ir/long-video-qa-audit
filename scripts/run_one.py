#!/usr/bin/env python3
"""Run the full Agentic Multimodal Routing pipeline on ONE (video, question).

This is the single-item "runbook" entry point. Given any video file and a
free-form question it executes the whole system end to end:

    plan (which skills)  ->  L1 propose  ->  L2 filter  ->  L3 align/fuse
    ->  submodular select K frames  ->  route to an expert  ->  answer.

It reuses the exact same code paths as the benchmark
(`cascade.CascadeRouter` + `router.backbones` experts), so the result matches
what a full run would produce for this item.

Examples
--------
# Visual-only question, rule-based planner, Qwen2.5-VL-7B expert (default):
python scripts/run_one.py \
    --video /path/to/clip.mp4 \
    --question "What is the man in the red shirt doing?" \
    --budget 8

# Narrative long video: inject the Whisper transcript into the VLM prompt:
python scripts/run_one.py --video lecture.mp4 \
    --question "What does the narrator say about the volcano?" \
    --inject asr --budget 8

# Audio-visual question -> route to the audio-native expert (VideoLLaMA2-AV):
python scripts/run_one.py --video concert.mp4 \
    --question "Is the violin on the left louder than the cello?" \
    --expert avlm --budget 8

# Let a small LLM planner decide the plan (needs the caption LLM loaded):
python scripts/run_one.py --video clip.mp4 --question "..." \
    --planner llm

Outputs
-------
A directory (``--output_dir``, default ``outputs/run_one/<timestamp>``) with:
  * ``selected_frames/frame_*.jpg`` -- the K chosen frames,
  * ``cascade_result.json``         -- the full plan + per-stage scores + timing,
  * ``answer.json``                 -- {question, answer, expert, plan, frames}.
The answer and a compact summary are also printed to stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_rgb(path: str):
    import cv2
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def build_planner(kind: str, budget: int, tau_sec: float,
                  procedural_memory_file: str, caplm_expert):
    """Return a cascade planner. ``llm`` needs a caption-LLM generate fn."""
    from cascade import RuleBasedCascadePlanner, LLMCascadePlanner

    if kind == "llm":
        if caplm_expert is None:
            raise SystemExit(
                "[run_one] --planner llm needs the caption LLM; rerun without "
                "--no_caplm so the planner has a generate function.")
        from router.experts import ExpertEvidence

        def _gen(prompt: str) -> str:
            old = getattr(caplm_expert, "max_new_tokens", 32)
            try:
                caplm_expert.max_new_tokens = 384
                out = caplm_expert.infer(ExpertEvidence(question=prompt, metadata={}))
            finally:
                caplm_expert.max_new_tokens = old
            return out.answer or out.extras.get("raw", "")

        procedural_memory = ""
        if procedural_memory_file:
            procedural_memory = Path(procedural_memory_file).read_text()
        return LLMCascadePlanner(
            generate_fn=_gen,
            K_final_override=int(budget),
            tau_sec_override=float(tau_sec),
            procedural_memory=procedural_memory,
        )
    # rule-based (default) / relaxed multi-cue
    return RuleBasedCascadePlanner(
        K_final=int(budget),
        tau_sec=float(tau_sec),
        force_multicue=(kind == "relaxed"),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="path to the video file")
    ap.add_argument("--question", required=True, help="free-form question")
    ap.add_argument("--dataset", default="custom",
                    help="dataset tag (affects keyword routing defaults); "
                         "use 'music_avqa'/'video_mme' to enable AV routing rules")
    ap.add_argument("--budget", type=int, default=8, help="K final frames")
    ap.add_argument("--tau_sec", type=float, default=2.0,
                    help="temporal diversity (submodular NMS) window")
    ap.add_argument("--planner", default="rule",
                    choices=["rule", "relaxed", "llm"],
                    help="rule=keyword planner (default), relaxed=force multi-cue, "
                         "llm=small LLM planner")
    ap.add_argument("--procedural_memory_file", default="",
                    help="markdown priors appended to the LLM planner prompt "
                         "(only used with --planner llm)")
    ap.add_argument("--expert", default=None,
                    choices=["vlm", "avlm"],
                    help="force an expert; default = the plan's routed expert")
    ap.add_argument("--vlm_model", default="Qwen/Qwen2.5-VL-7B-Instruct",
                    help="HF id for the visual expert")
    ap.add_argument("--av_model", default="DAMO-NLP-SG/VideoLLaMA2.1-7B-AV",
                    help="HF id for the audio-visual expert")
    ap.add_argument("--inject", default="plan",
                    help="cue text to inject into the VLM prompt: "
                         "none|asr|ocr|asr+ocr|plan (plan=use the planner's "
                         "inject_text decision)")
    ap.add_argument("--evidence_window_sec", type=float, default=6.0,
                    help="+/- window around each selected frame for ASR/OCR")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--output_dir", default="",
                    help="default: outputs/run_one/<timestamp>")
    args = ap.parse_args()

    if not Path(args.video).exists():
        raise SystemExit(f"[run_one] video not found: {args.video}")

    out_dir = Path(args.output_dir) if args.output_dir else \
        ROOT / "outputs" / "run_one" / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. build the expert pool (load only what we need) -----------------
    from router.backbones import Qwen2VLExpert
    needs_av = (args.expert == "avlm")
    experts: dict[str, object] = {}

    print(f"[run_one] loading visual expert {args.vlm_model} ...", flush=True)
    vlm = Qwen2VLExpert(model_id=args.vlm_model, device="cuda",
                        max_new_tokens=int(args.max_new_tokens))
    vlm.load()
    experts["vlm"] = vlm

    if needs_av:
        from router.backbones import VideoLLaMA2AVExpert
        print(f"[run_one] loading AV expert {args.av_model} ...", flush=True)
        av = VideoLLaMA2AVExpert(model_id=args.av_model, device="cuda",
                                 max_new_tokens=int(args.max_new_tokens))
        av.load()
        experts["avlm"] = av
    else:
        # Default: route 'avlm' to the visual expert so plans asking for it
        # still run (single-GPU: we never co-resident two 7B models).
        experts["avlm"] = vlm

    # ---- 2. plan + run the cascade -----------------------------------------
    from cascade import CascadeRouter, default_registry
    reg = default_registry()
    planner = build_planner(args.planner, args.budget, args.tau_sec,
                            args.procedural_memory_file,
                            caplm_expert=None)
    router = CascadeRouter(registry=reg, planner=planner)
    print(f"[run_one] tools available: {reg.available()}", flush=True)

    cres = router.run(
        video_path=args.video,
        question=args.question,
        dataset=args.dataset,
        output_dir=str(out_dir),
        K_final_override=int(args.budget),
    )
    plan = cres.plan
    print(f"[run_one] PLAN: proposers={plan.proposers} filters={plan.filters} "
          f"aligners={plan.aligners} weights={plan.weights} "
          f"expert={plan.expected_expert}", flush=True)
    print(f"[run_one] selected {len(cres.selected_image_paths)} frames at "
          f"{[round(t,1) for t in cres.selected_timestamps_sec]} s", flush=True)

    # ---- 3. optional cue-text injection ------------------------------------
    sel = cres.selected_image_paths
    sel_imgs = [im for im in (_read_rgb(p) for p in sel) if im is not None]
    ts_list = [float(t) for t in cres.selected_timestamps_sec]

    inject_mods: set[str] = set()
    if args.inject == "plan":
        inject_mods = {m for m in (plan.inject_text or []) if m in {"asr", "ocr"}}
    elif args.inject and args.inject != "none":
        inject_mods = {m.strip().lower() for m in args.inject.split("+")
                       if m.strip().lower() in {"asr", "ocr"}}

    asr_text = ""
    if inject_mods and sel_imgs:
        try:
            from cascade.evidence_text import build_combined_evidence_text
            asr_text = build_combined_evidence_text(
                args.video, sel_imgs, ts_list,
                modalities=tuple(sorted(inject_mods)),
                asr_window_sec=float(args.evidence_window_sec))
            print(f"[run_one] injected {sorted(inject_mods)} "
                  f"({len(asr_text)} chars)", flush=True)
        except Exception as e:
            print(f"[run_one] inject failed ({e}); continuing without", flush=True)

    # ---- 4. route to the expert + answer -----------------------------------
    expert_name = args.expert or plan.expected_expert or "vlm"
    expert = experts.get(expert_name, experts["vlm"])

    needs_audio = (expert_name == "avlm")
    audio_wave = None
    if needs_audio:
        try:
            from video_audio import extract_audio_from_video
            audio_wave = extract_audio_from_video(args.video)
        except Exception:
            audio_wave = None

    from router.experts import ExpertEvidence
    ev = ExpertEvidence(
        question=args.question,
        frames=sel_imgs or None,
        asr_text=asr_text or None,
        audio=audio_wave,
        metadata={"dataset": args.dataset, "video_path": args.video,
                  "needs_audio": needs_audio},
    )
    print(f"[run_one] routing to expert='{expert_name}' ...", flush=True)
    t0 = time.time()
    out = expert.infer(ev)
    infer_sec = time.time() - t0

    # ---- 5. persist + print -------------------------------------------------
    answer = {
        "question": args.question,
        "video": args.video,
        "answer": out.answer,
        "expert": expert_name,
        "inject": sorted(inject_mods),
        "plan": plan.to_json(),
        "selected_frames": [
            {"frame_index": int(i), "timestamp_sec": float(t), "image_path": p}
            for i, t, p in zip(cres.selected_frame_indices,
                               cres.selected_timestamps_sec,
                               cres.selected_image_paths)
        ],
        "select_runtime_sec": cres.runtime_sec.get("total", 0.0),
        "expert_runtime_sec": infer_sec,
    }
    (out_dir / "answer.json").write_text(json.dumps(answer, indent=2))

    print("\n" + "=" * 60)
    print(f"Q: {args.question}")
    print(f"A: {out.answer}")
    print(f"   expert={expert_name}  inject={sorted(inject_mods) or 'none'}  "
          f"K={len(sel_imgs)}  select={answer['select_runtime_sec']:.1f}s  "
          f"infer={infer_sec:.1f}s")
    print(f"   frames + plan + answer saved to: {out_dir}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
