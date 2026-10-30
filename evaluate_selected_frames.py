#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_result_jsons(root: Path) -> list[dict]:
    out = []
    for p in root.rglob("result.json"):
        try:
            out.append(json.loads(p.read_text()))
        except Exception:
            continue
    return out


def _interval_metrics(selected_ts: list[float], gt_intervals: list[tuple[float, float]]) -> dict:
    if not selected_ts:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    hit = 0
    for t in selected_ts:
        inside = any(a <= t <= b for a, b in gt_intervals)
        hit += int(inside)
    precision = hit / len(selected_ts)
    # proxy recall over intervals: at least one selected frame in interval
    rec_hit = 0
    for a, b in gt_intervals:
        rec_hit += int(any(a <= t <= b for t in selected_ts))
    recall = rec_hit / max(len(gt_intervals), 1)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate selected-frame outputs from SkillRouter.")
    ap.add_argument("--results_root", required=True, help="Folder containing result.json files")
    ap.add_argument(
        "--gt_intervals_json",
        default="",
        help="Optional JSON mapping video_id -> [[start_sec,end_sec], ...] for interval metrics",
    )
    args = ap.parse_args()

    rows = _load_result_jsons(Path(args.results_root))
    gt = {}
    if args.gt_intervals_json:
        gt = json.loads(Path(args.gt_intervals_json).read_text())

    summary = []
    for r in rows:
        vid = r.get("video_id", "")
        ts = [float(x["timestamp_sec"]) for x in r.get("selected_frames", [])]
        m = {"precision": None, "recall": None, "f1": None}
        if vid in gt:
            intervals = [(float(a), float(b)) for a, b in gt[vid]]
            m = _interval_metrics(ts, intervals)
        summary.append(
            {
                "video_id": vid,
                "dataset": r.get("dataset"),
                "method": r.get("method"),
                "num_selected": len(ts),
                "num_skills": len(r.get("selected_skills", [])),
                "runtime_sec": r.get("runtime_sec"),
                **m,
            }
        )

    out = Path(args.results_root) / "evaluation_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
