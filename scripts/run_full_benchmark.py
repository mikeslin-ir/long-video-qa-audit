#!/usr/bin/env python3
"""Run the SkillRouter on the FULL split of a dataset (no max_videos cap)
and write a benchmark_aggregate.json.

This is a thin wrapper around scripts/run_subset_benchmark.py — it just
removes the max_videos cap and adds per-dataset defaults so the same
command-line works across XD-Violence, UCF-Crime, MSRVTT-QA, MSVD-QA,
EgoSchema, and Video-MME.

Usage:
    python scripts/run_full_benchmark.py --dataset xd_violence
    python scripts/run_full_benchmark.py --dataset msrvtt_qa --planner llm
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skill_router import SkillRouter
from data_paths import get_data_root

# ---------- per-dataset defaults --------------------------------------------

DATA = get_data_root()

DATASET_REGISTRY: dict[str, dict] = {
    "xd_violence": {
        "root": str(DATA / "XD_violence" / "test"),
        "config": "configs/xd_violence_skill_router.yaml",
        "question": (
            "Where are the most relevant moments of violence, fighting, "
            "or accidents?"
        ),
        "budget": 6, "tau_sec": 1.0,
    },
    "ucf_crime": {
        "root": str(DATA / "UCF-Crime" / "Anomaly-Videos-Part-1"),
        "config": "configs/ucf_crime_skill_router.yaml",
        "question": (
            "Where are the most relevant moments of crime or suspicious "
            "interaction?"
        ),
        "budget": 6, "tau_sec": 1.0,
    },
    "msrvtt_qa": {
        "root": str(DATA / "MSRVTT" / "TestVideo"),
        "config": "configs/msrvtt_qa_skill_router.yaml",
        "question": "",   # fed per-clip from the QA json
        "budget": 6, "tau_sec": 0.8,
    },
    "msvd_qa": {
        "root": str(DATA / "MSVD_videos"),
        "config": "configs/msvd_qa_skill_router.yaml",
        "question": "",
        "budget": 6, "tau_sec": 0.8,
    },
    "egoschema": {
        "root": str(DATA / "EgoSchema"),
        "config": "configs/egoschema_skill_router.yaml",
        "question": "",
        "budget": 8, "tau_sec": 2.0,
    },
    "video_mme": {
        "root": str(DATA / "videomme"),
        "config": "configs/video_mme_skill_router.yaml",
        "question": "",
        "budget": 8, "tau_sec": 2.0,
    },
}

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def collect_videos(root: Path) -> list[Path]:
    out: list[Path] = []
    for ext in VIDEO_EXTS:
        out.extend(root.rglob(f"*{ext}"))
    return sorted({p.resolve() for p in out if p.is_file()})


def load_question_map(dataset: str, root: Path) -> dict[str, str]:
    """For QA datasets, load (video_id -> question). Returns {} if no
    QA json is found; the per-dataset default question is then used."""
    candidates = {
        "msrvtt_qa": [root / "msrvtt_qa.test.json", root.parent / "MSRVTT-QA" / "test_q.json"],
        "msvd_qa":   [root.parent / "MSVD-QA" / "test_qa.json"],
        "egoschema": [root / "questions.json"],
        "video_mme": [root / "videomme.json", root / "test.json"],
    }
    for p in candidates.get(dataset, []):
        if p.exists():
            try:
                obj = json.loads(p.read_text())
                # heuristic: list of dicts with video/question fields
                if isinstance(obj, list):
                    return {
                        str(r.get("video") or r.get("video_id") or r.get("id")):
                            str(r.get("question") or r.get("query") or "")
                        for r in obj if r.get("question") or r.get("query")
                    }
            except Exception:
                continue
    return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASET_REGISTRY))
    ap.add_argument("--planner", choices=["rule_based", "llm"], default="rule_based")
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--limit", type=int, default=0,
                    help="Optional cap. 0 (default) = process the entire split.")
    ap.add_argument("--use_submodular", action="store_true",
                    help="Use the submodular greedy selector instead of plain "
                         "temporal NMS (Sec. submodular).")
    args = ap.parse_args()

    info = DATASET_REGISTRY[args.dataset]
    root = Path(info["root"])
    if not root.exists():
        raise SystemExit(f"dataset root not found: {root}")

    out_root = Path(args.output_dir or f"outputs/full_{args.dataset}")
    out_root.mkdir(parents=True, exist_ok=True)

    videos = collect_videos(root)
    if args.limit > 0:
        videos = videos[: args.limit]

    qmap = load_question_map(args.dataset, root)
    default_q = info["question"] or "Which frames are most relevant?"

    router = SkillRouter.from_yaml(info["config"])
    rows: list[dict] = []
    failures: list[dict] = []
    t0 = time.time()

    for i, vp in enumerate(videos, start=1):
        q = qmap.get(vp.stem, default_q) or default_q
        run_dir = out_root / f"{i:05d}_{vp.stem}"
        try:
            res = router.run(
                video_path=str(vp),
                question=q,
                dataset=args.dataset,
                budget=info["budget"],
                tau_sec=info["tau_sec"],
                planner=args.planner,
                output_dir=str(run_dir),
                use_submodular=args.use_submodular,  # ignored if not yet wired
            )
            rows.append({
                "video": vp.name,
                "question": q,
                "selected_skills": res.get("selected_skills", []),
                "runtime_sec": res.get("runtime_sec"),
                "num_candidates": res.get("num_candidates"),
                "timestamps": res.get("timestamps"),
            })
        except Exception as e:
            failures.append({"video": vp.name, "error": repr(e)})

        if i % 25 == 0 or i == len(videos):
            print(f"  [{i:>5}/{len(videos)}]  elapsed={time.time()-t0:.1f}s  "
                  f"failures={len(failures)}", flush=True)

    summary = {
        "dataset": args.dataset,
        "planner": args.planner,
        "use_submodular": args.use_submodular,
        "processed_videos": len(rows),
        "failed_videos": len(failures),
        "mean_runtime_sec": (sum(r["runtime_sec"] for r in rows) / max(1, len(rows))),
        "skill_histogram": _hist(r["selected_skills"] for r in rows),
        "budget": info["budget"],
        "tau_sec": info["tau_sec"],
    }
    (out_root / "benchmark_aggregate.json").write_text(json.dumps(summary, indent=2))
    (out_root / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    if failures:
        (out_root / "failures.jsonl").write_text("\n".join(json.dumps(r) for r in failures))
    print(json.dumps(summary, indent=2))
    return 0


def _hist(seqs: Iterable[list[str]]) -> dict[str, int]:
    h: dict[str, int] = {}
    for s in seqs:
        for k in s:
            h[k] = h.get(k, 0) + 1
    return dict(sorted(h.items(), key=lambda kv: (-kv[1], kv[0])))


if __name__ == "__main__":
    sys.exit(main())
