#!/usr/bin/env python3
"""Run a baseline selector across a full dataset split.

Example:
    python scripts/run_baseline_benchmark.py --method uniform --dataset xd_violence
    python scripts/run_baseline_benchmark.py --method clip    --dataset msrvtt_qa
    python scripts/run_baseline_benchmark.py --method sevila  --dataset egoschema
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

from baselines import make as make_baseline
from scripts.run_full_benchmark import (
    DATASET_REGISTRY,
    collect_videos,
    load_question_map,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--dataset", required=True, choices=list(DATASET_REGISTRY))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output_dir", default=None)
    args = ap.parse_args()

    info = DATASET_REGISTRY[args.dataset]
    root = Path(info["root"])
    if not root.exists():
        raise SystemExit(f"dataset root not found: {root}")
    out_root = Path(args.output_dir or f"outputs/baseline_{args.method}_{args.dataset}")
    out_root.mkdir(parents=True, exist_ok=True)

    videos = collect_videos(root)
    if args.limit > 0:
        videos = videos[: args.limit]
    qmap = load_question_map(args.dataset, root)
    default_q = info["question"] or "Which frames are most relevant?"

    selector = make_baseline(args.method)
    rows: list[dict] = []
    failures: list[dict] = []
    t0 = time.time()
    for i, vp in enumerate(videos, 1):
        q = qmap.get(vp.stem, default_q) or default_q
        run_dir = out_root / f"{i:05d}_{vp.stem}"
        try:
            res = selector.select(
                video_path=str(vp), question=q, dataset=args.dataset,
                budget=info["budget"], tau_sec=info["tau_sec"],
                output_dir=str(run_dir),
            )
            rows.append({
                "video": vp.name, "question": q,
                "selected_frames": res.get("selected_frames", []),
                "selection_method": res.get("selection_method", args.method),
                "runtime_sec": res.get("runtime_sec"),
            })
        except Exception as e:
            failures.append({"video": vp.name, "error": repr(e)})
        if i % 25 == 0 or i == len(videos):
            print(f"  [{i:>5}/{len(videos)}]  elapsed={time.time()-t0:.1f}s  "
                  f"failures={len(failures)}", flush=True)

    summary = {
        "method": args.method,
        "dataset": args.dataset,
        "processed_videos": len(rows),
        "failed_videos": len(failures),
        "budget": info["budget"],
        "tau_sec": info["tau_sec"],
    }
    (out_root / "benchmark_aggregate.json").write_text(json.dumps(summary, indent=2))
    (out_root / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    if failures:
        (out_root / "failures.jsonl").write_text("\n".join(json.dumps(r) for r in failures))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
