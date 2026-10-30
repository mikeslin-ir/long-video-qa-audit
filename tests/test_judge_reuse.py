from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import llm_judge as lj  # noqa: E402


class FakeJudge:
    model_id = "fake"

    def __init__(self, yes_when=("yes",)):
        self.calls = []
        self.yes_when = yes_when

    def judge(self, question, gold, pred):
        self.calls.append((question, gold, pred))
        return pred in self.yes_when, 4 if pred in self.yes_when else 0, ""


def _setup(tmp_path, monkeypatch):
    items = [SimpleNamespace(qid=q, question=f"question {q}?", answer="yes", task_type="Audio")
             for q in ("q1", "q2", "q3", "q4")]
    monkeypatch.setattr(lj, "load_dataset", lambda name: items)
    judged = tmp_path / "judged_arm"
    judged.mkdir()
    (judged / "judged_rows.jsonl").write_text("\n".join(json.dumps(
        {"qid": q, "question": f"question {q}?", "gold": "yes", "pred": p, "task_type": "Audio",
         "judge_yes": y, "judge_score": 4 if y else 0})
        for q, p, y in (("q1", "yes", True), ("q2", "no", False), ("q3", "Yes.", True))))
    run = tmp_path / "silent_arm"
    run.mkdir()
    rows = [{"qid": "q1", "pred": "yes"},      # same answer as the judged arm -> reused
            {"qid": "q2", "pred": "yes"},      # different answer -> judged by the model
            {"qid": "q3", "pred": "Yes."},     # same -> reused
            {"qid": "q4", "pred": "no"}]       # not in the judged arm -> judged by the model
    (run / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return run, judged


def test_reuses_verdicts_of_identical_inputs_and_judges_the_rest(tmp_path, monkeypatch):
    run, judged = _setup(tmp_path, monkeypatch)
    judge = FakeJudge()
    summary = lj.rescore(run, "music_avqa", judge, reuse_dirs=[judged])
    assert [c[2] for c in judge.calls] == ["yes", "no"]
    assert summary["n_reused_verdicts"] == 2 and summary["n_judged_by_model"] == 2
    out = [json.loads(line) for line in (run / "judged_rows.jsonl").read_text().splitlines()]
    assert [r["judge_yes"] for r in out] == [True, True, True, False]
    assert [r.get("verdict_from") for r in out] == ["judged_arm", None, "judged_arm", None]
    assert not (run / "judged_rows.partial.jsonl").exists()


def test_verdict_does_not_transfer_when_the_question_or_gold_differs(tmp_path, monkeypatch):
    run, judged = _setup(tmp_path, monkeypatch)
    rows = [json.loads(line) for line in (judged / "judged_rows.jsonl").read_text().splitlines()]
    rows[0]["gold"] = "no"
    (judged / "judged_rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    judge = FakeJudge()
    lj.rescore(run, "music_avqa", judge, reuse_dirs=[judged])
    assert len(judge.calls) == 3


def test_interrupted_run_resumes_from_partial_verdicts(tmp_path, monkeypatch):
    run, judged = _setup(tmp_path, monkeypatch)
    (run / "judged_rows.partial.jsonl").write_text(json.dumps(
        {"qid": "q2", "question": "question q2?", "gold": "yes", "pred": "yes", "task_type": "Audio",
         "judge_yes": True, "judge_score": 4}) + "\n")
    judge = FakeJudge()
    lj.rescore(run, "music_avqa", judge, reuse_dirs=[judged])
    assert [c[2] for c in judge.calls] == ["no"]          # q2 came from the partial file
    assert not (run / "judged_rows.partial.jsonl").exists()


def test_audit_counts_how_many_reused_verdicts_the_model_reproduces(tmp_path, monkeypatch):
    run, judged = _setup(tmp_path, monkeypatch)
    judge = FakeJudge(yes_when=("yes",))                  # disagrees with the stored verdict for "Yes."
    summary = lj.rescore(run, "music_avqa", judge, reuse_dirs=[judged], audit_reuse=2)
    assert summary["reuse_audit"] == {"checked": 2, "agree": 1}


def test_without_reuse_every_row_goes_to_the_model(tmp_path, monkeypatch):
    run, _ = _setup(tmp_path, monkeypatch)
    judge = FakeJudge()
    summary = lj.rescore(run, "music_avqa", judge)
    assert len(judge.calls) == 4 and "n_reused_verdicts" not in summary
