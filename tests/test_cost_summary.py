from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import cost_summary as cs  # noqa: E402


def _make_run(root: Path) -> Path:
    run = root / "demo_run"
    run.mkdir()
    rows = [
        {"qid": "a", "select_runtime_sec": 3.0, "expert_cost_sec": 9.0},
        {"qid": "b", "select_runtime_sec": 5.0, "expert_cost_sec": 11.0},
    ]
    (run / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (run / "qa_summary.json").write_text(json.dumps({"n": 2, "elapsed_sec": 28.0}))
    for name, decode in (("00001_a", 2.0), ("00002_b", 4.0)):
        item = run / name
        item.mkdir()
        (item / "cascade_result.json").write_text(json.dumps({"runtime_sec": {"decode": decode, "select": 0.1, "total": decode + 0.5}}))
    return run


def test_summary_reports_stage_means_and_wall_clock(tmp_path):
    result = cs.summarize(_make_run(tmp_path))
    assert result["n"] == 2
    assert result["wall_sec_per_item"] == pytest.approx(14.0)
    assert result["select_plus_answer_sec"] == pytest.approx(14.0)
    assert result["cascade_stage_ms"]["decode"]["mean"] == pytest.approx(3000.0)
    assert result["cascade_stage_ms"]["decode"]["median"] == pytest.approx(3000.0)
    assert "L1_propose" not in result["cascade_stage_ms"]
    assert "wall clock" in cs.render(result)


def test_uniform_runs_without_cascade_timers_still_summarize(tmp_path):
    run = _make_run(tmp_path)
    for item in run.glob("*/cascade_result.json"):
        item.unlink()
    result = cs.summarize(run)
    assert result["cascade_stage_ms"] == {}
    assert result["select_sec"]["mean"] == pytest.approx(4.0)
