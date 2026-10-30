#!/usr/bin/env python3
"""Deterministic stratified split of MUSIC-AVQA for the same-model audio ablation.

Writes two qid lists:

  qids_stratified<N>.txt  N questions per question type (all of them for a type
                          with fewer than N): the quick first phase.
  qids_rest.txt           every other question: the slow second phase.

Together they cover the full split exactly once, so the two phases can be
pooled. Both files are consumed by ``run_qa_benchmark.py --only_qids``.

    python scripts/make_audio_ablation_subset.py --per-type 200
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def stratified_split(qid_types, per_type: int, seed: int):
    """Split ``(qid, question_type)`` pairs into a stratified subset and the rest.

    Returns ``(subset, rest)``; both keep the input (dataset) order.
    """
    by_type: dict[str, list[str]] = defaultdict(list)
    order: list[str] = []
    for qid, qtype in qid_types:
        by_type[qtype].append(qid)
        order.append(qid)
    if len(set(order)) != len(order):
        raise ValueError("qids must be unique")
    rng = random.Random(seed)
    chosen: set[str] = set()
    for qtype in sorted(by_type):
        qids = by_type[qtype]
        chosen.update(rng.sample(qids, min(per_type, len(qids))))
    subset = [q for q in order if q in chosen]
    rest = [q for q in order if q not in chosen]
    return subset, rest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="music_avqa")
    ap.add_argument("--per-type", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "audio_ablation"))
    args = ap.parse_args()

    from router.qa_datasets import load_dataset

    items = load_dataset(args.dataset, limit=0)
    subset, rest = stratified_split(
        [(it.qid, it.task_type) for it in items], args.per_type, args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    header = (f"# {args.dataset} qids for the audio ablation; "
              f"per-type={args.per_type} seed={args.seed}")
    sub_path = out / f"qids_stratified{args.per_type}.txt"
    rest_path = out / "qids_rest.txt"
    sub_path.write_text("\n".join([header, *subset]) + "\n")
    rest_path.write_text("\n".join([header.replace("qids", "remaining qids"), *rest]) + "\n")

    counts: dict[str, int] = defaultdict(int)
    wanted = set(subset)
    for it in items:
        if it.qid in wanted:
            counts[it.task_type] += 1
    print(f"subset: {len(subset)} qids -> {sub_path}")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    print(f"rest:   {len(rest)} qids -> {rest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
