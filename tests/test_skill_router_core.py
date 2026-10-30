from __future__ import annotations

import numpy as np

from fusion import fuse_skill_scores, minmax_normalize, zscore_sigmoid_normalize
from router.planner import RuleBasedSkillPlanner
from temporal_nms import select_frames_with_temporal_nms


def test_minmax_constant_returns_zeros():
    x = np.array([3.0, 3.0, 3.0])
    y = minmax_normalize(x)
    assert np.allclose(y, np.zeros_like(x))


def test_zscore_constant_returns_zeros():
    x = np.array([2.0, 2.0, 2.0])
    y = zscore_sigmoid_normalize(x)
    assert np.allclose(y, np.zeros_like(x))


def test_fusion_weighted_sum_shape():
    a = np.array([0.0, 1.0, 0.5], dtype=np.float32)
    b = np.array([1.0, 0.0, 0.5], dtype=np.float32)
    fused, norm = fuse_skill_scores(
        {"motion": a, "audio": b},
        {"motion": 0.7, "audio": 0.3},
        normalization="minmax",
    )
    assert fused.shape == (3,)
    assert set(norm.keys()) == {"motion", "audio"}


def test_planner_violence_selects_three_core_skills():
    planner = RuleBasedSkillPlanner()
    out = planner.plan("Is there fighting and explosion?", "xd_violence")
    s = set(out["selected_skills"])
    assert "motion" in s
    assert "audio" in s
    assert "semantic_siglip" in s


def test_temporal_nms_respects_tau():
    scores = np.array([0.1, 0.9, 0.8, 0.2, 0.95])
    ts = np.array([0.0, 0.5, 0.9, 2.0, 2.2])
    sel = select_frames_with_temporal_nms(scores, ts, K=2, tau_sec=0.6)
    assert len(sel) == 2
    assert ts[sel[1]] - ts[sel[0]] >= 0.6 or np.isclose(ts[sel[1]] - ts[sel[0]], 0.6)
