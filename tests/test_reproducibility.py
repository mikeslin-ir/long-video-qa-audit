"""Reproducibility / determinism tests for the VideoRouteAudit selection stack.

These tests pin down the paper's determinism claim: given fixed evidence
signals, every stage of the CPU selection pipeline (normalization, fusion,
temporal-NMS, submodular facility-location, and the rule-based planner) is
bit-for-bit reproducible across repeated calls, independent of dict insertion
order and free of any hidden RNG state.

They are intentionally GPU-free (no VLM / no torch) so they run fast in CI and
can execute alongside GPU experiments without contention.
"""
from __future__ import annotations

import numpy as np

from fusion import fuse_skill_scores, minmax_normalize, zscore_sigmoid_normalize
from router.planner import RuleBasedSkillPlanner
from submodular import submodular_select
from temporal_nms import select_frames_with_temporal_nms


def _make_signals(n: int = 64, seed: int = 20260613):
    """Deterministic synthetic evidence signals (RNG used ONLY to build inputs)."""
    rng = np.random.default_rng(seed)
    motion = rng.random(n).astype(np.float32)
    audio = rng.random(n).astype(np.float32)
    semantic = rng.random(n).astype(np.float32)
    timestamps = np.sort(rng.uniform(0.0, 120.0, size=n)).astype(np.float64)
    return motion, audio, semantic, timestamps


def test_normalization_is_deterministic():
    x = _make_signals()[0]
    assert np.array_equal(minmax_normalize(x), minmax_normalize(x))
    assert np.array_equal(zscore_sigmoid_normalize(x), zscore_sigmoid_normalize(x))


def test_fusion_is_deterministic():
    motion, audio, semantic, _ = _make_signals()
    skills = {"motion": motion, "audio": audio, "semantic_siglip": semantic}
    weights = {"motion": 0.5, "audio": 0.3, "semantic_siglip": 0.2}
    fused_a, _ = fuse_skill_scores(skills, weights, normalization="minmax")
    fused_b, _ = fuse_skill_scores(skills, weights, normalization="minmax")
    assert np.array_equal(fused_a, fused_b)


def test_fusion_independent_of_dict_order():
    """Fusion must not depend on the insertion order of the skill dicts."""
    motion, audio, semantic, _ = _make_signals()
    skills1 = {"motion": motion, "audio": audio, "semantic_siglip": semantic}
    weights1 = {"motion": 0.5, "audio": 0.3, "semantic_siglip": 0.2}
    skills2 = {"semantic_siglip": semantic, "audio": audio, "motion": motion}
    weights2 = {"semantic_siglip": 0.2, "motion": 0.5, "audio": 0.3}
    fused1, _ = fuse_skill_scores(skills1, weights1, normalization="minmax")
    fused2, _ = fuse_skill_scores(skills2, weights2, normalization="minmax")
    assert np.allclose(fused1, fused2)


def test_temporal_nms_is_deterministic():
    motion, audio, semantic, ts = _make_signals()
    fused, _ = fuse_skill_scores(
        {"motion": motion, "audio": audio, "semantic_siglip": semantic},
        {"motion": 0.5, "audio": 0.3, "semantic_siglip": 0.2},
        normalization="minmax",
    )
    for K in (1, 2, 4, 8):
        sel_a = select_frames_with_temporal_nms(fused, ts, K=K, tau_sec=2.0)
        sel_b = select_frames_with_temporal_nms(fused, ts, K=K, tau_sec=2.0)
        assert sel_a == sel_b


def test_submodular_selection_is_deterministic():
    """Lazy-greedy facility-location must return an identical, sorted set."""
    motion, audio, semantic, ts = _make_signals()
    fused, _ = fuse_skill_scores(
        {"motion": motion, "audio": audio, "semantic_siglip": semantic},
        {"motion": 0.5, "audio": 0.3, "semantic_siglip": 0.2},
        normalization="minmax",
    )
    for K in (1, 2, 4, 8):
        sel_a = submodular_select(fused, ts, K=K, tau_sec=2.0)
        sel_b = submodular_select(fused, ts, K=K, tau_sec=2.0)
        assert sel_a == sel_b
        assert len(sel_a) == K
        # returned indices are sorted by timestamp
        assert list(ts[sel_a]) == sorted(ts[sel_a])


def test_planner_is_deterministic():
    planner = RuleBasedSkillPlanner()
    q, ds = "Is there fighting and an explosion?", "xd_violence"
    out_a = planner.plan(q, ds)
    out_b = planner.plan(q, ds)
    assert out_a["selected_skills"] == out_b["selected_skills"]


def test_full_cpu_selection_stack_reproducible():
    """Two independent passes of the whole CPU stack yield identical frames."""
    def _run():
        motion, audio, semantic, ts = _make_signals()
        fused, _ = fuse_skill_scores(
            {"motion": motion, "audio": audio, "semantic_siglip": semantic},
            {"motion": 0.5, "audio": 0.3, "semantic_siglip": 0.2},
            normalization="minmax",
        )
        return submodular_select(fused, ts, K=8, tau_sec=2.0)

    assert _run() == _run()
