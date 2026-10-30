#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skill_router import SkillRouter


METHOD_TO_SKILLS = {
    "motion_only": ["motion"],
    "audio_only": ["audio"],
    "asr_only": ["asr_whisper"],
    "ocr_only": ["ocr_text"],
    "semantic_only": ["semantic_siglip"],
    "motion_audio": ["motion", "audio"],
    "asr_audio": ["asr_whisper", "audio"],
    "ocr_semantic": ["ocr_text", "semantic_siglip"],
    "motion_semantic": ["motion", "semantic_siglip"],
    "audio_semantic": ["audio", "semantic_siglip"],
    "asr_semantic": ["asr_whisper", "semantic_siglip"],
    "all_skills": ["motion", "audio", "semantic_siglip", "object_siglip", "asr_whisper", "ocr_text"],
    "skill_router": None,
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Run skill ablations for one question/video.")
    ap.add_argument("--video_path", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--dataset", default="xd_violence")
    ap.add_argument("--budget", type=int, default=6)
    ap.add_argument("--tau_sec", type=float, default=1.0)
    ap.add_argument("--config", default="configs/skill_router.yaml")
    ap.add_argument(
        "--methods",
        default="motion_only,audio_only,asr_only,ocr_only,semantic_only,motion_audio,asr_audio,ocr_semantic,motion_semantic,audio_semantic,asr_semantic,all_skills,skill_router",
    )
    ap.add_argument("--output_dir", default="outputs/skill_router_ablation")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    router = SkillRouter.from_yaml(args.config)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    table = []
    for method in methods:
        if method not in METHOD_TO_SKILLS:
            print(f"[skip] unknown method: {method}")
            continue
        out_dir = out_root / method
        whitelist = METHOD_TO_SKILLS[method]
        res = router.run(
            video_path=args.video_path,
            question=args.question,
            dataset=args.dataset,
            budget=args.budget,
            tau_sec=args.tau_sec,
            output_dir=str(out_dir),
            planner_mode="rule_based",
            enabled_skills=whitelist,
        )
        table.append(
            {
                "method": method,
                "selected_skills": res["selected_skills"],
                "runtime_sec": res["runtime_sec"],
                "result_json": str(out_dir / "result.json"),
            }
        )

    (out_root / "ablation_summary.json").write_text(json.dumps(table, indent=2))
    print(json.dumps(table, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
