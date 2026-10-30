#!/usr/bin/env python3
"""Per-stage runtime of a run, from its per-item timing records.

The paper's cost numbers are means over the items of a run directory:

* ``select_runtime_sec`` and ``expert_cost_sec`` in ``rows.jsonl`` (frame selection and
  the answering model's call),
* the per-stage timers ``runtime_sec`` in each item's ``cascade_result.json`` (cascade
  runs only),
* ``elapsed_sec / n`` in ``qa_summary.json`` (wall clock, including anything the
  per-item timers do not see, such as OCR extraction).

    python scripts/cost_summary.py trackB_long_n900_k8_cascade_f0p5_injectASR
    python scripts/cost_summary.py RUN [RUN ...] --json results/cost_summary.json

Run directories are looked up under ``outputs/`` (``LONG_VIDEO_QA_OUTPUTS``). Raw run
outputs are not redistributed, so this needs the authors' working tree or your own reruns.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import audit_predictions as ap  # noqa: E402

STAGES = ("plan", "L1_propose", "decode", "L2_filter", "L3_align", "select", "total")


def _stat(values: list[float], scale: float = 1.0) -> dict | None:
    if not values:
        return None
    return {"n": len(values), "mean": scale * statistics.fmean(values), "median": scale * statistics.median(values)}


def summarize(run_dir: Path) -> dict:
    summary = json.loads((run_dir / "qa_summary.json").read_text())
    rows = [json.loads(line) for line in (run_dir / "rows.jsonl").read_text().splitlines() if line.strip()]
    stage_ms: dict[str, list[float]] = {name: [] for name in STAGES}
    for path in sorted(run_dir.glob("*/cascade_result.json")):
        timers = json.loads(path.read_text()).get("runtime_sec", {})
        for name in STAGES:
            if name in timers:
                stage_ms[name].append(1000.0 * float(timers[name]))
    select = [float(r["select_runtime_sec"]) for r in rows if "select_runtime_sec" in r]
    answer = [float(r["expert_cost_sec"]) for r in rows if "expert_cost_sec" in r]
    n = int(summary["n"])
    return {
        "run": run_dir.name,
        "n": n,
        "wall_sec_per_item": float(summary["elapsed_sec"]) / n,
        "select_sec": _stat(select),
        "answer_sec": _stat(answer),
        "select_plus_answer_sec": (statistics.fmean(select) + statistics.fmean(answer)) if select and answer else None,
        "cascade_stage_ms": {name: _stat(v) for name, v in stage_ms.items() if v},
    }


def render(result: dict) -> str:
    lines = [f"{result['run']}  (n={result['n']})"]
    for name, stat in result["cascade_stage_ms"].items():
        lines.append(f"  {name:<12}{stat['mean']:>10.1f} ms mean{stat['median']:>10.1f} ms median")
    if result["select_sec"]:
        lines.append(f"  selection   {result['select_sec']['mean']:>10.2f} s  (median {result['select_sec']['median']:.2f})")
    if result["answer_sec"]:
        lines.append(f"  answer      {result['answer_sec']['mean']:>10.2f} s  (median {result['answer_sec']['median']:.2f})")
    if result["select_plus_answer_sec"] is not None:
        lines.append(f"  select+answer {result['select_plus_answer_sec']:>8.2f} s/item")
    lines.append(f"  wall clock  {result['wall_sec_per_item']:>10.2f} s/item  (elapsed_sec / n)")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="run directory names under the outputs directory")
    parser.add_argument("--outputs", type=Path, default=None, help="outputs directory (default: LONG_VIDEO_QA_OUTPUTS or ./outputs)")
    parser.add_argument("--json", type=Path, default=None, help="also write the summaries as JSON")
    args = parser.parse_args()
    base = args.outputs or ap.outputs_dir()
    results = []
    for run in args.runs:
        run_dir = base / run
        if not (run_dir / "qa_summary.json").exists():
            print(f"missing run directory: {run_dir}", file=sys.stderr)
            return 1
        results.append(summarize(run_dir))
        print(render(results[-1]))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
