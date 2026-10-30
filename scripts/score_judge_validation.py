#!/usr/bin/env python3
"""Score the human-vs-judge agreement after annotation_sheet.csv is filled in.

Computes, overall and per dataset:
  * raw agreement %
  * Cohen's kappa
  * judge false-accept rate (judge=yes, human=no)
  * judge false-reject rate (judge=no, human=yes)
  * a bootstrap 95% CI for kappa

Run AFTER a human fills the `human_yes` column (1/0) in
results/judge_validation/annotation_sheet.csv.
"""
from __future__ import annotations
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "results" / "judge_validation"
SHEET = OUTDIR / "annotation_sheet.csv"
KEY = OUTDIR / "judge_key.jsonl"


def cohen_kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1 = sum(a) / n
    pb1 = sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def main() -> None:
    key = {json.loads(l)["idx"]: json.loads(l) for l in KEY.open() if l.strip()}
    human: dict[int, int] = {}
    with SHEET.open() as f:
        for row in csv.DictReader(f):
            v = (row.get("human_yes") or "").strip()
            if v in ("0", "1"):
                human[int(row["idx"])] = int(v)

    idxs = sorted(i for i in human if i in key)
    if not idxs:
        print("[err] no annotated rows yet. Fill `human_yes` (1/0) in", SHEET)
        return

    def report(label: str, sub: list[int]) -> None:
        j = [key[i]["judge_yes"] for i in sub]
        h = [human[i] for i in sub]
        n = len(sub)
        agree = sum(1 for x, y in zip(j, h) if x == y) / n * 100
        kappa = cohen_kappa(j, h)
        fa = sum(1 for i in sub if key[i]["judge_yes"] == 1 and human[i] == 0)
        fr = sum(1 for i in sub if key[i]["judge_yes"] == 0 and human[i] == 1)
        rng = random.Random(0)
        boots = []
        for _ in range(10000):
            s = [sub[rng.randrange(n)] for _ in range(n)]
            boots.append(cohen_kappa([key[i]["judge_yes"] for i in s],
                                     [human[i] for i in s]))
        boots = [x for x in boots if x == x]
        boots.sort()
        lo = boots[int(0.025 * len(boots))]
        hi = boots[int(0.975 * len(boots))]
        print(f"{label:14s} n={n:3d}  agree={agree:5.1f}%  kappa={kappa:.3f} "
              f"[{lo:.3f},{hi:.3f}]  false-accept={fa}  false-reject={fr}")

    print("Judge (Qwen2.5-VL-32B) vs human agreement:")
    report("OVERALL", idxs)
    for ds in sorted({key[i]["dataset"] for i in idxs}):
        report(ds, [i for i in idxs if key[i]["dataset"] == ds])


if __name__ == "__main__":
    main()
