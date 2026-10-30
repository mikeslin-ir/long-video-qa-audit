#!/usr/bin/env python3
"""Aggregate QA run summaries into CSV and Markdown tables.

Writes:
  outputs/qa_experiment_table.csv
  outputs/qa_experiment_table.md
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def load_rows(run_dir: Path) -> tuple[int, float]:
    rows_path = run_dir / "rows.jsonl"
    if not rows_path.exists():
        return 0, 0.0
    n = 0
    total_cost = 0.0
    for ln in rows_path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
        except Exception:
            continue
        n += 1
        total_cost += float(r.get("select_runtime_sec") or 0.0) + float(r.get("expert_cost_sec") or 0.0)
    avg_cost = (total_cost / n) if n > 0 else 0.0
    return n, avg_cost


def collect() -> list[dict]:
    rows = []
    if not OUT.exists():
        return rows

    for d in sorted(OUT.iterdir()):
        if not d.is_dir():
            continue
        s = d / "qa_summary.json"
        if not s.exists():
            continue
        try:
            js = json.loads(s.read_text())
        except Exception:
            continue

        n_rows, avg_cost = load_rows(d)
        rec = {
            "run": d.name,
            "dataset": js.get("dataset", ""),
            "selector": js.get("selector", ""),
            "expert": js.get("expert", ""),
            "n": int(js.get("n") or 0),
            "accuracy": float(js.get("accuracy") or 0.0),
            "failed": int(js.get("failed") or 0),
            "elapsed_sec": float(js.get("elapsed_sec") or 0.0),
            "avg_item_sec": avg_cost,
            "rows_present": n_rows,
        }
        rows.append(rec)

    rows.sort(key=lambda x: (x["dataset"], x["run"]))
    return rows


def write_csv(rows: list[dict]) -> Path:
    out = OUT / "qa_experiment_table.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "dataset",
                "selector",
                "expert",
                "n",
                "accuracy",
                "failed",
                "elapsed_sec",
                "avg_item_sec",
                "rows_present",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return out


def write_md(rows: list[dict]) -> Path:
    out = OUT / "qa_experiment_table.md"
    lines = []
    lines.append("# QA Experiment Summary")
    lines.append("")
    lines.append("| run | dataset | selector | expert | n | acc | failed | elapsed_h | avg_item_s |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['run']} | {r['dataset']} | {r['selector']} | {r['expert']} | "
            f"{r['n']} | {100.0*r['accuracy']:.2f}% | {r['failed']} | "
            f"{r['elapsed_sec']/3600.0:.2f} | {r['avg_item_sec']:.2f} |"
        )
    lines.append("")
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> int:
    rows = collect()
    if not rows:
        print("no qa_summary.json files found under outputs/")
        return 0
    csv_path = write_csv(rows)
    md_path = write_md(rows)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
