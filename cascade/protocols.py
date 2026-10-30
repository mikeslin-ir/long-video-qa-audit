"""Tool protocols for the cascade.

Three orthogonal roles, each declared as a runtime-checkable Protocol so any
class with the right methods is automatically usable as a tool.

Cost annotations (`cost_per_video`, `cost_per_frame`) are *informational*
heuristics used by the agentic planner when balancing latency vs accuracy --
they are not enforced at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class ProposalSet:
    """Output of an L1 proposer."""
    frame_indices: np.ndarray   # int64, sorted ascending
    timestamps_sec: np.ndarray  # float64, same length as frame_indices
    scores: np.ndarray | None = None  # optional confidence/saliency per anchor
    source: str = ""            # name of the proposer that produced this

    def __len__(self) -> int:
        return int(len(self.frame_indices))


@runtime_checkable
class FrameProposer(Protocol):
    """L1: cheap pass that proposes a small set of *time anchors* over the video.

    Implementations should be O(video_length) but with a small constant
    (sub-second decoding, no neural inference). Examples: uniform stride,
    shot-boundary detection, motion/audio event peaks.
    """
    name: str
    cost_per_video: float  # seconds (informational)

    def propose(
        self,
        video_path: str,
        question: str | None,
        meta: dict,
        K: int,
    ) -> ProposalSet:
        ...


@runtime_checkable
class KeyframeFilter(Protocol):
    """L2: refine a ProposalSet down to K_keyframe candidates using low-level
    per-frame features (frame diff, color histogram, light motion).

    Operates on already-decoded frames (passed by the router as
    list[np.ndarray] for the proposed indices) so cost scales with
    len(proposals), not full video length.
    """
    name: str
    cost_per_frame: float

    def filter(
        self,
        proposal: ProposalSet,
        frames_rgb: list[np.ndarray],
        question: str | None,
        meta: dict,
        K: int,
    ) -> ProposalSet:
        ...


@runtime_checkable
class EvidenceAligner(Protocol):
    """L3: heavy question-conditioned scoring (SigLIP / OCR / ASR / object).

    Receives the surviving keyframes from L2 and returns a per-frame relevance
    score (higher = more aligned with the question).
    """
    name: str
    cost_per_frame: float
    modality: str  # "vision" | "text" | "audio"

    def align(
        self,
        keyframes: ProposalSet,
        frames_rgb: list[np.ndarray],
        video_path: str,
        question: str | None,
        meta: dict,
    ) -> np.ndarray:  # shape (len(keyframes),)
        ...
