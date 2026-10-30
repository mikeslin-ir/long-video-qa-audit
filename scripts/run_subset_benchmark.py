#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skill_router import SkillRouter


def _collect_videos(dataset_root: Path, exts: tuple[str, ...]) -> list[Path]:
    videos: list[Path] = []
    for ext in exts:
        videos.extend(dataset_root.rglob(f"*{ext}"))
    return sorted({p.resolve() for p in videos if p.is_file()})


def _default_question(dataset: str) -> str:
    ds = (dataset or "").lower()
    if ds == "xd_violence":
        return "Where are the most relevant moments of violence, fighting, or accidents?"
    if ds == "ucf_crime":
        return "Where are the most relevant moments of crime or suspicious interaction?"
    return "Which frames are most relevant to answer this question?"


def _skill_histogram(rows: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        for s in r.get("selected_skills", []):
            out[s] = out.get(s, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def main() -> int:
    ap = argparse.ArgumentParser(description="Run SkillRouter on a sampled subset of videos and summarize results.")
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--dataset", default="xd_violence")
    ap.add_argument("--question", default="")
    ap.add_argument("--config", default="configs/skill_router.yaml")
    ap.add_argument("--planner", choices=["rule_based", "llm"], default="rule_based")
    ap.add_argument("--budget", type=int, default=6)
    ap.add_argument("--tau_sec", type=float, default=1.0)
    ap.add_argument("--max_videos", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exts", default=".mp4,.avi,.mov,.mkv")
    ap.add_argument("--output_dir", default="outputs/benchmark_subset")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    exts = tuple(e.strip() for e in args.exts.split(",") if e.strip())
    videos = _collect_videos(dataset_root, exts)
    if not videos:
        raise RuntimeError(f"No videos found in {dataset_root} with extensions {exts}")

    rnd = random.Random(args.seed)
    rnd.shuffle(videos)
    videos = videos[: max(1, int(args.max_videos))]

    question = args.question.strip() or _default_question(args.dataset)
    router = SkillRouter.from_yaml(args.config)

    rows: list[dict] = []
    failures: list[dict] = []
    for i, vp in enumerate(videos, start=1):
        run_dir = out_root / f"{i:03d}_{vp.stem}"
        try:
            res = router.run(
                video_path=str(vp),
                question=question,
                dataset=args.dataset,
                budget=args.budget,
                tau_sec=args.tau_sec,
                output_dir=str(run_dir),
                planner_mode=args.planner,
                enabled_skills=None,
            )
            rows.append(
                {
                    "video_path": str(vp),
                    "video_id": res.get("video_id", vp.stem),
                    "runtime_sec": float(res.get("runtime_sec", 0.0)),
                    "num_candidates": int(res.get("num_candidates", 0)),
                    "selected_skills": list(res.get("selected_skills", [])),
                    "planner_reason": res.get("planner_reason", ""),
                }
            )
            print(f"[{i}/{len(videos)}] OK  {vp.name}")
        except Exception as e:
            failures.append({"video_path": str(vp), "error": str(e)})
            print(f"[{i}/{len(videos)}] ERR {vp.name}: {e}")

    n_ok = len(rows)
    runtime = [r["runtime_sec"] for r in rows]
    candidates = [r["num_candidates"] for r in rows]

    aggregate = {
        "dataset": args.dataset,
        "dataset_root": str(dataset_root),
        "question": question,
        "config": args.config,
        "planner": args.planner,
        "budget": int(args.budget),
        "tau_sec": float(args.tau_sec),
        "requested_videos": int(args.max_videos),
        "processed_videos": n_ok,
        "failed_videos": len(failures),
        "mean_runtime_sec": (sum(runtime) / n_ok) if n_ok else 0.0,
        "mean_num_candidates": (sum(candidates) / n_ok) if n_ok else 0.0,
        "skill_histogram": _skill_histogram(rows),
    }

    (out_root / "benchmark_rows.json").write_text(json.dumps(rows, indent=2))
    (out_root / "benchmark_failures.json").write_text(json.dumps(failures, indent=2))
    (out_root / "benchmark_aggregate.json").write_text(json.dumps(aggregate, indent=2))

    print(json.dumps(aggregate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
