"""Agentic 3-level cascade for video frame selection.

L1 (proposers): cheap event/shot signals  -> N_proposal time anchors
L2 (filters):   low-level features (motion / color) -> K_keyframe candidates
L3 (aligners):  heavy question-conditioned features (SigLIP / OCR / ASR) -> scores
L4 (submodular): pick K frames balancing relevance + temporal coverage

The Cascade is described by a `CascadePlan` (see plan.py), which is produced
by an `AgenticPlanner` (see planner.py) based on the question, dataset, and
video metadata. `CascadeRouter` (see router.py) executes the plan.
"""
from __future__ import annotations

from .plan import CascadePlan
from .protocols import FrameProposer, KeyframeFilter, EvidenceAligner
from .registry import CascadeRegistry, default_registry
from .planner import RuleBasedCascadePlanner, LLMCascadePlanner
from .router import CascadeRouter, CascadeResult

__all__ = [
    "CascadePlan",
    "FrameProposer",
    "KeyframeFilter",
    "EvidenceAligner",
    "CascadeRegistry",
    "default_registry",
    "RuleBasedCascadePlanner",
    "LLMCascadePlanner",
    "CascadeRouter",
    "CascadeResult",
]
