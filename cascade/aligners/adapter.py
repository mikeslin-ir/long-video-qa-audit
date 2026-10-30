"""Adapter that wraps any existing `evidence_skills.EvidenceSkill` as an
`EvidenceAligner`. This lets us reuse SigLIP / OCR / ASR / Object skills
without changes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..protocols import ProposalSet
from evidence_skills.base import EvidenceSkill, FrameInfo


@dataclass
class EvidenceSkillAligner:
    name: str
    modality: str           # "vision" | "text" | "audio"
    cost_per_frame: float
    skill: EvidenceSkill

    def align(
        self,
        keyframes: ProposalSet,
        frames_rgb: list[np.ndarray],
        video_path: str,
        question: str | None,
        meta: dict,
    ) -> np.ndarray:
        infos = [
            FrameInfo(
                frame_index=int(keyframes.frame_indices[i]),
                timestamp_sec=float(keyframes.timestamps_sec[i]),
                image=frames_rgb[i] if i < len(frames_rgb) else None,
            )
            for i in range(len(keyframes))
        ]
        s = self.skill.compute(
            video_path=video_path,
            frames=infos,
            question=question,
            metadata=dict(meta),
        )
        s = np.asarray(s, dtype=np.float64)
        if len(s) != len(infos):
            # interpolate if backend returns a different resolution
            if len(s) == 0:
                return np.zeros(len(infos), dtype=np.float64)
            x_old = np.linspace(0.0, 1.0, len(s))
            x_new = np.linspace(0.0, 1.0, len(infos))
            s = np.interp(x_new, x_old, s)
        return s
