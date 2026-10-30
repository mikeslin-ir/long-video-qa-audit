from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import audio_ablation_analysis as aa  # noqa: E402

pytestmark = pytest.mark.skipif(not aa.ARMS_FILE.exists(), reason="results/audio_ablation/arms.csv.gz not shipped")


def _quiet(fn, *args):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args)


@pytest.mark.parametrize("grader", ["judge", "inloop"])
def test_arms_table_pairs_all_three_arms_on_the_same_questions(grader):
    table = aa.read_arms()
    qids, arms, types = aa.arm_arrays(table, grader)
    assert len(qids) == len(table) == len(types) == 1799
    assert set(arms) == {"blind", "av", "silent"}
    assert all(len(v) == len(qids) for v in arms.values())
    assert set(np.unique(np.concatenate(list(arms.values())))) <= {0, 1}


@pytest.mark.parametrize("grader", ["judge", "inloop"])
def test_silencing_the_soundtrack_removes_only_a_small_share_of_the_swap(grader):
    out = _quiet(aa.ablation, aa.read_arms(), grader)["micro_poststratified"]
    assert out["accuracy"]["blind"] < out["accuracy"]["silent"] < out["accuracy"]["av"]
    swap, audio, rest = (out[k]["delta"] for k in ("total", "audio", "non_audio"))
    assert swap == pytest.approx(audio + rest)
    assert 0 < audio < 0.2 * swap
    lo, hi = out["audio_share_pct"]["ci95"]
    assert lo < out["audio_share_pct"]["point"] < hi < 25


def test_the_judge_graded_swap_matches_the_full_split_micro_average():
    # the stratified sample, post-stratified to the full split's type mix, reproduces the full-split swap
    swap = _quiet(aa.ablation, aa.read_arms(), "judge")["micro_poststratified"]["total"]["delta"]
    full = _quiet(aa.full_split_table)["micro"]["delta"]
    assert abs(swap - full) < 0.5


def test_ablation_refuses_a_grader_that_misses_verdicts():
    table = {q: dict(r) for q, r in aa.read_arms().items()}
    first = next(iter(table))
    table[first]["silent_judge"] = ""
    assert _quiet(aa.ablation, table, "judge") is None


def test_every_contrast_uses_one_grader():
    table = aa.read_arms()
    for grader in ("judge", "inloop"):
        qids, arms, _ = aa.arm_arrays(table, grader)
        assert list(arms["silent"]) == [int(table[q][f"silent_{grader}"]) for q in qids]


@pytest.mark.parametrize("grader", ["judge", "inloop"])
def test_mismatched_soundtrack_sits_between_blind_and_silenced(grader):
    table = aa.read_arms()
    if not any(str(r.get("mismatched_" + grader, "")) != "" for r in table.values()):
        pytest.skip("mismatched-audio arm not shipped")
    out = _quiet(aa.mismatched_control, table, grader)
    acc = out["accuracy"]
    # the wrong soundtrack misleads the expert (below silence) yet it keeps most of the swap over the blind model
    assert acc["blind"] < acc["mismatched"] < acc["silent"] < acc["av"]
    c = out["contrasts"]
    assert c["silent->mismatched"]["ci95"][1] < 0
    assert c["blind->mismatched"]["ci95"][0] > 0
