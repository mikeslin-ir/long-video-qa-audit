#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skill_router import SkillRouter


def main() -> int:
    ap = argparse.ArgumentParser(description="Run question-conditioned SkillRouter on one video.")
    ap.add_argument("--video_path", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--dataset", default="xd_violence")
    ap.add_argument("--budget", type=int, default=6)
    ap.add_argument("--planner", choices=["rule_based", "llm"], default="rule_based")
    ap.add_argument("--skills", default="", help="Optional comma-separated skill whitelist")
    ap.add_argument("--tau_sec", type=float, default=1.0)
    ap.add_argument("--config", default="configs/skill_router.yaml")
    ap.add_argument("--output_dir", default="outputs/skill_router_demo")
    args = ap.parse_args()

    router = SkillRouter.from_yaml(args.config)
    skills = [s.strip() for s in args.skills.split(",") if s.strip()]
    res = router.run(
        video_path=args.video_path,
        question=args.question,
        dataset=args.dataset,
        budget=args.budget,
        tau_sec=args.tau_sec,
        output_dir=args.output_dir,
        planner_mode=args.planner,
        enabled_skills=skills or None,
    )
    print(json.dumps(
        {
            "video_id": res["video_id"],
            "selected_skills": res["selected_skills"],
            "budget": res["budget"],
            "runtime_sec": res["runtime_sec"],
            "planner_reason": res["planner_reason"],
            "result_json": str(Path(args.output_dir) / "result.json"),
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
