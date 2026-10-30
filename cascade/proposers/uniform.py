"""Uniform stride proposer: always available, baseline-quality.

Picks K_proposal evenly spaced frame indices across the whole video.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..protocols import ProposalSet


@dataclass
class UniformStrideProposer:
    name: str = "uniform_stride"
    cost_per_video: float = 0.5  # sec, just opens the video

    def propose(self, video_path: str, question: str | None, meta: dict, K: int) -> ProposalSet:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        cap.release()
        if total <= 0:
            return ProposalSet(np.array([], dtype=np.int64), np.array([], dtype=np.float64),
                               source=self.name)

        K = max(1, min(int(K), total))
        # Even spacing including endpoints
        idx = np.linspace(0, total - 1, K).round().astype(np.int64)
        idx = np.unique(idx)
        ts = idx.astype(np.float64) / max(fps, 1e-6)
        return ProposalSet(idx, ts, source=self.name)
