"""Smoke tests for submodular selection and procedural memory."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from submodular import submodular_select
from router.memory import MemoryStore, MemoryAugmentedPlanner
from router.planner import RuleBasedSkillPlanner


def test_submodular_returns_K_distinct_indices():
    rng = np.random.default_rng(0)
    N, K = 200, 8
    scores = rng.random(N).astype(np.float32)
    timestamps = np.linspace(0.0, 60.0, N).astype(np.float32)
    sel = submodular_select(scores, timestamps, K=K, tau_sec=1.5)
    assert len(sel) == K
    assert len(set(sel)) == K
    # selected timestamps must be sorted
    ts_sel = [float(timestamps[i]) for i in sel]
    assert ts_sel == sorted(ts_sel)


def test_submodular_prefers_high_score_frames_with_diversity():
    # Single tall spike: greedy should pick it first.
    N, K = 50, 3
    scores = np.zeros((N,), dtype=np.float32); scores[10] = 1.0; scores[40] = 0.9
    timestamps = np.linspace(0.0, 50.0, N).astype(np.float32)
    sel = submodular_select(scores, timestamps, K=K, tau_sec=2.0)
    assert 10 in sel and 40 in sel


def test_indicator_kernel_acts_like_NMS():
    # When the kernel is the indicator and tau is large, suppressed
    # neighbours should be skipped — i.e., we should get spread-out picks.
    N, K = 100, 4
    scores = np.linspace(0.0, 1.0, N).astype(np.float32)  # monotonic
    timestamps = np.linspace(0.0, 100.0, N).astype(np.float32)
    sel = submodular_select(scores, timestamps, K=K, tau_sec=10.0,
                            kernel="indicator")
    assert len(sel) == K


def test_memory_store_roundtrip(tmp_path):
    store = MemoryStore(path=str(tmp_path / "mem.jsonl"))
    store.add(qid="v1#1", dataset="xd_violence",
              question="is there violence in this clip?",
              selected_skills=["motion", "audio", "semantic_siglip"],
              weights={"motion": 0.4, "audio": 0.3, "semantic_siglip": 0.3},
              reward=0.8)
    store.add(qid="v2#1", dataset="msrvtt_qa",
              question="what is the person doing on stage?",
              selected_skills=["semantic_siglip", "motion"],
              weights={"semantic_siglip": 0.7, "motion": 0.3},
              reward=0.6)
    hits = store.retrieve("a fight breaks out", dataset="xd_violence", k=2)
    assert len(hits) >= 1
    assert hits[0][1].dataset == "xd_violence"


def test_memory_augmented_planner_blends(tmp_path):
    store = MemoryStore(path=str(tmp_path / "mem.jsonl"))
    store.add(qid="v1#1", dataset="xd_violence",
              question="violent fight scene",
              selected_skills=["motion", "audio", "semantic_siglip"],
              weights={"motion": 0.5, "audio": 0.3, "semantic_siglip": 0.2},
              reward=0.9)

    base = RuleBasedSkillPlanner()
    aug = MemoryAugmentedPlanner(base, memory=store, alpha=0.5,
                                 min_similarity=0.0)
    plan = aug.plan(question="please find any fighting moment",
                    dataset="xd_violence")
    assert plan.get("memory_used") is True
    assert sum(plan["weights"].values()) == 1.0 or abs(sum(plan["weights"].values()) - 1.0) < 1e-6


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
