from __future__ import annotations

import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

import audit_predictions as ap
import audit_stats as st


def test_discordant_counts_orientation():
    base = [1, 1, 0, 0, 1, 0]
    variant = [1, 0, 1, 1, 1, 0]
    assert st.discordant(base, variant) == (2, 1)  # (variant right / base wrong, reverse)


def test_exact_mcnemar_matches_published_pooled_counts():
    # Pooled F3(a) counts reported in the paper: 54 gained vs 26 lost.
    assert st.mcnemar_exact_p(54, 26) == pytest.approx(0.0023, abs=5e-5)
    assert st.mcnemar_exact_p(0, 0) == 1.0
    assert st.mcnemar_exact_p(7, 7) == pytest.approx(1.0)


def test_normal_mcnemar_matches_published_statistic():
    z, p = st.mcnemar_z(54, 26)
    assert z == pytest.approx(3.13, abs=0.005)
    assert p == pytest.approx(0.0017, abs=5e-5)
    assert st.mcnemar_z(0, 0) == (0.0, 1.0)


def test_bootstrap_interval_is_seeded_and_brackets_the_mean():
    diff = np.array([1] * 30 + [-1] * 10 + [0] * 60)
    mean, lo, hi = st.paired_bootstrap_ci(diff, reps=2000, seed=3)
    assert mean == pytest.approx(20.0)
    assert lo < mean < hi
    assert st.paired_bootstrap_ci(diff, reps=2000, seed=3) == (mean, lo, hi)


def test_domain_weighting_ignores_domain_size():
    big = np.zeros(900)
    small = np.ones(100)
    res = st.domain_weighted_delta({"big": big, "small": small}, reps=500, seed=1)
    assert res.delta == pytest.approx(50.0)  # mean of 0 and 100, not 10
    assert res.weights == {"big": 0.5, "small": 0.5}
    pooled = np.concatenate([big, small]).mean() * 100
    assert pooled == pytest.approx(10.0)


def test_domain_weighted_null_and_effect():
    rng = np.random.default_rng(0)
    null = {k: rng.choice([-1, 0, 1], size=300, p=[0.1, 0.8, 0.1]).astype(float) for k in "ab"}
    effect = {k: rng.choice([-1, 0, 1], size=300, p=[0.05, 0.7, 0.25]).astype(float) for k in "ab"}
    assert st.domain_weighted_delta(null, reps=1000, seed=2).p_permutation > 0.05
    strong = st.domain_weighted_delta(effect, reps=1000, seed=2)
    assert strong.p_permutation < 0.01 and strong.lo > 0


def _write_run(root, name, rows, judged=None):
    run = root / name
    run.mkdir(parents=True)
    (run / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    if judged is not None:
        (run / "judged_rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in judged))


def test_live_correct_prefers_judge_verdict(tmp_path):
    _write_run(tmp_path, "open", [{"qid": "a", "correct": False}, {"qid": "b", "correct": False}],
               judged=[{"qid": "a", "judge_yes": True}, {"qid": "b", "judge_yes": False}])
    _write_run(tmp_path, "mcq", [{"qid": "a", "correct": True}, {"qid": "b", "correct": False}])
    assert ap.live_correct("open", tmp_path) == {"a": 1, "b": 0}
    assert ap.live_correct("mcq", tmp_path) == {"a": 1, "b": 0}
    assert ap.live_correct("absent", tmp_path) is None


def test_load_correct_falls_back_to_packaged_table(tmp_path, monkeypatch):
    package = tmp_path / "predictions.csv.gz"
    with gzip.open(package, "wt", newline="") as fh:
        fh.write("run,qid,correct,pred\nr1,q1,1,A\nr1,q2,0,B\n")
    monkeypatch.setattr(ap, "PACKAGED", package)
    monkeypatch.setenv("LONG_VIDEO_QA_OUTPUTS", str(tmp_path / "nowhere"))
    ap._packaged.cache_clear()
    try:
        assert ap.load_correct("r1") == {"q1": 1, "q2": 0}
        assert ap.available_runs() == ["r1"]
        with pytest.raises(FileNotFoundError):
            ap.load_correct("missing")
    finally:
        ap._packaged.cache_clear()


def test_packaged_predictions_match_the_headline_numbers():
    """The shipped table reproduces the full-bucket accuracies printed in the paper."""
    if not ap.PACKAGED.exists():
        pytest.skip("packaged predictions not present")
    expected = {
        "trackB_long_n900_k8_uniform": 46.2,
        "trackB_long_n900_k8_cascade_f0p5": 47.4,
        "trackB_long_n900_k8_cascade_f0p5_injectASR": 47.9,
    }
    for run, acc in expected.items():
        out = ap.load_correct(run)
        assert len(out) == 900
        assert math.isclose(100 * sum(out.values()) / len(out), acc, abs_tol=0.05)


def test_manifest_agrees_with_the_packaged_outcomes():
    """Every manifest row (n, accuracy, injected count) is recomputable from the shipped table."""
    if not ap.PACKAGED.exists() or not ap.MANIFEST.exists():
        pytest.skip("packaged predictions not present")
    import csv

    ap._packaged.cache_clear()
    packaged = ap._packaged()
    injected: dict[str, int] = {}
    with gzip.open(ap.PACKAGED, "rt", newline="") as fh:
        for row in csv.DictReader(fh):
            if int(row.get("evidence_chars") or 0) > 0:
                injected[row["run"]] = injected.get(row["run"], 0) + 1
    with open(ap.MANIFEST, newline="") as fh:
        manifest = list(csv.DictReader(fh))
    assert manifest, "empty manifest"
    for row in manifest:
        outcomes = packaged[row["run"]]
        assert len(outcomes) == int(row["n"]), row["run"]
        assert math.isclose(100 * sum(outcomes.values()) / len(outcomes), float(row["accuracy"]), abs_tol=0.051), row["run"]
        assert injected.get(row["run"], 0) == int(row["n_injected"] or 0), row["run"]
    assert set(packaged) == {r["run"] for r in manifest}


def test_first_n_keeps_the_native_order_of_the_baseline_run(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import stats_analysis as sa

    package = tmp_path / "predictions.csv.gz"
    qids = ["q9", "q1", "q5", "q3"]  # native order is deliberately not sorted
    with gzip.open(package, "wt", newline="") as fh:
        fh.write("run,qid,correct,pred\n")
        for q, a, b in zip(qids, (1, 0, 1, 0), (1, 1, 0, 0)):
            fh.write(f"base,{q},{a},A\nvariant,{q},{b},A\n")
    monkeypatch.setattr(ap, "PACKAGED", package)
    monkeypatch.setenv("LONG_VIDEO_QA_OUTPUTS", str(tmp_path / "nowhere"))
    ap._packaged.cache_clear()
    try:
        a, b = sa.pair("base", "variant", first_n=2)
        assert list(a) == [1, 0] and list(b) == [1, 1]  # q9, q1 -- not the sorted q1, q3
        a_all, _ = sa.pair("base", "variant")
        assert len(a_all) == 4
    finally:
        ap._packaged.cache_clear()


def test_dispatch_analysis_on_a_toy_split():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import routed_expert_analysis as rea

    base = {"a": 1, "b": 0, "c": 0, "d": 1}
    av = {"a": 1, "b": 1, "c": 0, "d": 0}
    pick = {"a": True, "b": True, "c": False, "d": False}
    r = rea.compute(base, av, pick, reps=200)
    assert r["n"] == 4 and r["routed_to_av"] == 2
    assert (r["base"], r["av"], r["routed"], r["oracle"]) == (50.0, 50.0, 75.0, 75.0)
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (1, 1, 0, 2)
    assert r["headroom_captured_pct"] == pytest.approx(100.0)
    assert rea.needs_audio("Which instrument sounds first?") and not rea.needs_audio("What color is the shirt?")


def test_shipped_dispatch_decisions_cover_the_full_split():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import routed_expert_analysis as rea

    if not rea.DISPATCH.exists() or not ap.PACKAGED.exists():
        pytest.skip("packaged predictions not present")
    decisions = rea.read_dispatch()
    assert len(decisions) == 9129
    assert sum(decisions.values()) == 7746


def test_sharded_runs_need_their_done_marker_before_they_are_packaged(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import package_predictions as pp

    run = tmp_path / "demo"
    run.mkdir()
    assert not pp.finished(tmp_path, "demo")          # no summary yet
    (run / "qa_summary.json").write_text("{}")
    assert pp.finished(tmp_path, "demo")              # an ordinary run
    (run / "shard0.log").write_text("")
    assert not pp.finished(tmp_path, "demo")          # a shard finished, the run may not have
    (run / "DONE").write_text("")
    assert pp.finished(tmp_path, "demo")
