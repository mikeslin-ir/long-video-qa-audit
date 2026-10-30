"""Prewarm Whisper ASR cache for an entire dataset bucket.

Iterates over all unique videos in a dataset (optionally filtered by
task_type) and runs ASRWhisperSkill._transcribe_with_timestamps on each.
Results are written to `results/cascade_cache/asr/<video_hash>.json`.

Subsequent cascade runs (and evidence-text injection) hit the cache and
skip the Whisper forward pass.

Usage:
    python scripts/prewarm_asr_cache.py --dataset video_mme --task_type long
    python scripts/prewarm_asr_cache.py --dataset video_mme  # all 900x3
    python scripts/prewarm_asr_cache.py --dataset xd_violence --limit 200

Injection of the transcript (``--inject_evidence_text asr``) reads this cache and
silently injects nothing for a video that is not in it; run this script first for
every dataset you inject ASR into. The cache directory is
``$LONG_VIDEO_QA_CASCADE_CACHE`` (default ``results/cascade_cache``).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make repo root importable when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_paths import get_cascade_cache_dir
from router.qa_datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--task_type", default="",
                    help="filter items by task_type (e.g. 'long')")
    ap.add_argument("--cache_dir", default=get_cascade_cache_dir())
    ap.add_argument("--model_id", default="openai/whisper-tiny.en")
    ap.add_argument("--limit", type=int, default=0,
                    help="load only the first N items, exactly as run_qa_benchmark.py "
                         "--limit does, so the transcribed clips are the evaluated ones")
    ap.add_argument("--limit_videos", type=int, default=0,
                    help="only prewarm first N unique videos (debug)")
    ap.add_argument("--shard", default="",
                    help="I/N: transcribe only every N-th video starting at I, so N "
                         "processes (for example CPU-only: CUDA_VISIBLE_DEVICES=) can share one "
                         "dataset")
    ap.add_argument("--max_new", type=int, default=0,
                    help="stop (exit code 0) after transcribing this many new videos; "
                         "re-running continues where it stopped. Bounds the process's memory "
                         "growth on long videos")
    args = ap.parse_args()

    from evidence_skills.asr_skill import ASRWhisperSkill
    skill = ASRWhisperSkill(cache_dir=args.cache_dir, model_id=args.model_id)

    items = load_dataset(args.dataset, limit=args.limit)
    if args.task_type:
        want = args.task_type.strip().lower()
        items = [it for it in items if (it.task_type or "").strip().lower() == want]
        print(f"[prewarm-asr] filtered to task_type={want!r}: {len(items)} items",
              flush=True)

    # Unique videos (preserve first-seen order so we can checkpoint)
    seen = {}
    for it in items:
        vp = getattr(it, "video_path", None)
        if vp and vp not in seen:
            seen[vp] = True
    videos = list(seen.keys())
    if args.limit_videos > 0:
        videos = videos[: args.limit_videos]
    if args.shard:
        index, count = (int(x) for x in args.shard.split("/"))
        videos = videos[index::count]

    n = len(videos)
    print(f"[prewarm-asr] dataset={args.dataset} unique videos: {n}",
          flush=True)

    # Count cache hits ahead of time
    cache_root = Path(args.cache_dir) / "asr"
    cache_root.mkdir(parents=True, exist_ok=True)
    already = 0
    todo = []
    from evidence_skills.base import video_cache_key
    for vp in videos:
        key = video_cache_key(vp, f"asr_{skill.model_id}_{skill.sr}")
        if (cache_root / f"{key}.json").exists():
            already += 1
        else:
            todo.append(vp)
    print(f"[prewarm-asr] already cached: {already}/{n}; to transcribe: {len(todo)}",
          flush=True)

    if not todo:
        print("[prewarm-asr] nothing to do.", flush=True)
        return

    t0 = time.time()
    if args.max_new > 0 and len(todo) > args.max_new:
        print(f"[prewarm-asr] --max_new {args.max_new}: {len(todo) - args.max_new} videos "
              "left for the next run", flush=True)
        todo, finished = todo[: args.max_new], False
    else:
        finished = True
    for i, vp in enumerate(todo, 1):
        ti = time.time()
        try:
            words = skill._transcribe_with_timestamps(vp)
            dt = time.time() - ti
            eta = (time.time() - t0) / i * (len(todo) - i)
            print(f"[prewarm-asr] [{i:>4}/{len(todo)}] +{dt:5.1f}s "
                  f"words={len(words):>4} eta={eta/60:.1f}min {Path(vp).name}",
                  flush=True)
        except Exception as e:
            print(f"[prewarm-asr] [{i:>4}/{len(todo)}] FAIL {Path(vp).name}: {e}",
                  flush=True)
    if finished:
        print(f"[prewarm-asr] done in {(time.time()-t0)/60:.1f}min", flush=True)
    else:
        print(f"[prewarm-asr] batch of {len(todo)} finished in {(time.time()-t0)/60:.1f}min; "
              "run again to continue", flush=True)


if __name__ == "__main__":
    main()
