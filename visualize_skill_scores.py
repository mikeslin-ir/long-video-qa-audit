#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description="Visualize skill/fused scores and selected frames.")
    ap.add_argument("--result_json", required=True)
    ap.add_argument("--output_png", default="")
    ap.add_argument("--gt_intervals", default="", help="Optional JSON string, e.g. [[1.0,2.0],[4.0,5.2]]")
    args = ap.parse_args()

    r = json.loads(Path(args.result_json).read_text())
    fused = np.load(r["fused_scores_path"])
    timestamps = np.load(r["timestamps_path"])
    scores = np.load(r["skill_scores_path"])

    fig, ax = plt.subplots(figsize=(12, 4))
    for k in scores.files:
        ax.plot(timestamps, scores[k], label=k, alpha=0.6)
    ax.plot(timestamps, fused, label="fused", linewidth=2.5, color="black")

    for fr in r.get("selected_frames", []):
        ax.axvline(float(fr["timestamp_sec"]), color="red", linestyle="--", alpha=0.5)

    if args.gt_intervals:
        intervals = json.loads(args.gt_intervals)
        for a, b in intervals:
            ax.axvspan(float(a), float(b), color="green", alpha=0.15)

    ax.set_title(f"Skill scores: {r.get('video_id')} | {r.get('dataset')}")
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Score")
    ax.legend(loc="upper right", ncol=3)
    ax.grid(alpha=0.2)

    out = Path(args.output_png) if args.output_png else Path(args.result_json).with_suffix(".png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
