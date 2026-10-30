#!/usr/bin/env python3
"""Build a BLIND stratified human-annotation sheet to validate the open LLM judge
(Qwen2.5-VL-32B) as a substitute for gpt-4o-mini.

Design notes
------------
* Only the two judge-scored datasets are sampled (music_avqa, egotextvqa).
  TutorialVQA is scored by span/token-F1, not the judge, so it is excluded.
* Stratified by (dataset, judge_yes) so BOTH judge error directions
  (false-accept and false-reject) are estimable -> valid Cohen's kappa.
* The annotation sheet is BLIND: it does NOT show the judge's yes/no, to avoid
  anchoring the human. The judge labels are written to a separate KEY file.
* Annotator fills the `human_yes` column with 1 (pred is correct) or 0 (wrong).

Outputs
-------
  results/judge_validation/annotation_sheet.csv   # give this to the human
  results/judge_validation/judge_key.jsonl        # judge labels (kept hidden)
"""
from __future__ import annotations
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "results" / "judge_validation"
OUTDIR.mkdir(parents=True, exist_ok=True)

# One representative judged run per judge-scored dataset (the headline configs).
SOURCES = {
    "music_avqa": ROOT / "outputs/wave9av_videollama2av_music_avqa_uniform/judged_rows.jsonl",
    "egotextvqa": ROOT / "outputs/wave9fix_qwen2vl_egotextvqa_cascade_injectOCR/judged_rows.jsonl",
}

# How many to sample per (dataset, judge_yes) stratum.
PER_STRATUM = {
    ("music_avqa", True): 45,
    ("music_avqa", False): 45,
    ("egotextvqa", True): 30,
    ("egotextvqa", False): 30,
}

SEED = 20260615


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open() if l.strip()]


def main() -> None:
    rng = random.Random(SEED)
    picked: list[dict] = []
    for ds, path in SOURCES.items():
        if not path.exists():
            print(f"[warn] missing {path}")
            continue
        rows = load(path)
        for yes in (True, False):
            pool = [r for r in rows if bool(r.get("judge_yes")) == yes]
            k = PER_STRATUM.get((ds, yes), 0)
            rng.shuffle(pool)
            sel = pool[:k]
            for r in sel:
                picked.append({
                    "dataset": ds,
                    "qid": r["qid"],
                    "question": r.get("question", ""),
                    "gold": r.get("gold", ""),
                    "pred": r.get("pred", ""),
                    "task_type": r.get("task_type", ""),
                    "judge_yes": int(bool(r.get("judge_yes"))),
                    "judge_score": r.get("judge_score", ""),
                })
            print(f"[sample] {ds} judge_yes={yes}: pool={len(pool)} took={len(sel)}")

    # Shuffle so the blind sheet does not reveal strata ordering.
    rng.shuffle(picked)
    for i, r in enumerate(picked):
        r["idx"] = i

    # 1) BLIND annotation sheet (no judge label, no judge score).
    sheet = OUTDIR / "annotation_sheet.csv"
    with sheet.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "dataset", "question", "gold_answer", "model_prediction", "human_yes"])
        for r in picked:
            w.writerow([r["idx"], r["dataset"], r["question"], r["gold"], r["pred"], ""])

    # 2) Hidden KEY with the judge labels, joined later on idx.
    key = OUTDIR / "judge_key.jsonl"
    with key.open("w") as f:
        for r in picked:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n[done] {len(picked)} items")
    print(f"  blind sheet -> {sheet}")
    print(f"  hidden key  -> {key}")
    print("\nAnnotate: open the CSV, judge whether each `model_prediction` is a")
    print("correct answer to `question` given `gold_answer`; write 1 (correct)")
    print("or 0 (wrong) in `human_yes`. Then run scripts/score_judge_validation.py")


if __name__ == "__main__":
    main()
