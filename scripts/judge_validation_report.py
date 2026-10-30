#!/usr/bin/env python3
"""Agreement between the LLM judge and human labels, broken down by dataset.

The judge (Qwen2.5-VL-32B-Instruct text-only rescoring, ``scripts/llm_judge.py``)
grades open-ended answers. It is used for MUSIC-AVQA, which carries the F3(b)
expert contrast; Video-MME and EgoSchema are multiple choice and are scored by
exact match on the parsed option letter, without a judge.

The validation set is 150 stratified decisions with a human label each. This
script reports agreement per dataset because the pooled number hides that the
judge's only disagreements are on EgoTextVQA, which no headline claim uses.

    python scripts/judge_validation_report.py            # from the shipped CSV
    python scripts/judge_validation_report.py --build <dir with the human-annotation sheets>

``--build`` is for the authors: it joins the annotation sheet with the judge key
and keeps only ids and the two binary verdicts (no questions, gold answers or
predictions).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "judge_validation" / "judge_vs_human.csv"
FIELDS = ["idx", "dataset", "qid", "task_type", "judge_yes", "human_yes"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def cohen_kappa(judge: list[int], human: list[int]) -> float:
    n = len(judge)
    if n == 0:
        return float("nan")
    po = sum(j == h for j, h in zip(judge, human)) / n
    pj, ph = sum(judge) / n, sum(human) / n
    pe = pj * ph + (1 - pj) * (1 - ph)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def agreement_stats(rows: list[dict]) -> dict:
    """Agreement, error types and Wilson interval for one group of decisions."""
    judge = [int(r["judge_yes"]) for r in rows]
    human = [int(r["human_yes"]) for r in rows]
    agree = sum(j == h for j, h in zip(judge, human))
    lo, hi = wilson(agree, len(rows))
    return {
        "n": len(rows), "agree": agree, "agreement": agree / len(rows) if rows else float("nan"),
        "wilson95": (lo, hi),
        "false_accept": sum(j == 1 and h == 0 for j, h in zip(judge, human)),
        "false_reject": sum(j == 0 and h == 1 for j, h in zip(judge, human)),
        "kappa": cohen_kappa(judge, human),
    }


def load(path: Path = CSV_PATH) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def build(src: Path, dest: Path = CSV_PATH) -> int:
    human = {int(r["idx"]): r["human_yes"] for r in csv.DictReader(open(src / "annotation_sheet.csv", newline=""))}
    out = []
    for line in open(src / "judge_key.jsonl"):
        if not line.strip():
            continue
        r = json.loads(line)
        if human.get(int(r["idx"]), "") == "":
            continue
        out.append({"idx": r["idx"], "dataset": r["dataset"], "qid": r["qid"], "task_type": r.get("task_type", ""),
                    "judge_yes": int(r["judge_yes"]), "human_yes": int(float(human[int(r["idx"])]))})
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(sorted(out, key=lambda r: int(r["idx"])))
    print(f"wrote {len(out)} decisions -> {dest}")
    return 0


def report(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["dataset"]].append(r)
    result = {"all": agreement_stats(rows), **{k: agreement_stats(v) for k, v in sorted(groups.items())}}
    print(f"{'subset':<14}{'n':>5}{'agree':>7}{'rate':>8}   {'95% Wilson':<16}{'FA':>4}{'FR':>4}{'kappa':>8}")
    for name, s in result.items():
        lo, hi = s["wilson95"]
        print(f"{name:<14}{s['n']:>5}{s['agree']:>7}{100 * s['agreement']:>7.1f}%   "
              f"[{100 * lo:5.1f},{100 * hi:5.1f}]  {s['false_accept']:>4}{s['false_reject']:>4}{s['kappa']:>8.2f}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", type=Path, default=None, metavar="DIR",
                    help="authors only: build the shipped CSV from DIR/annotation_sheet.csv and DIR/judge_key.jsonl")
    args = ap.parse_args()
    if args.build:
        return build(args.build)
    report(load())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
