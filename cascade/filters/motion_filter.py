"""MotionDiffFilter: keep frames with the largest grayscale change from their
immediate predecessor within the proposal set. Pure NumPy/OpenCV, no models.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..protocols import ProposalSet


@dataclass
class MotionDiffFilter:
    name: str = "motion_diff"
    cost_per_frame: float = 0.002
    downscale: int = 4   # spatial downsample for diff (4x faster)

    def filter(
        self,
        proposal: ProposalSet,
        frames_rgb: list[np.ndarray],
        question: str | None,
        meta: dict,
        K: int,
    ) -> ProposalSet:
        if len(frames_rgb) == 0:
            return proposal

        # Grayscale + downscaled
        grays = []
        for f in frames_rgb:
            g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
            if self.downscale > 1:
                g = cv2.resize(g, (g.shape[1] // self.downscale, g.shape[0] // self.downscale))
            grays.append(g.astype(np.float32))

        # |diff to previous|; first frame gets diff to next as proxy
        n = len(grays)
        if n == 1:
            scores = np.array([1.0], dtype=np.float64)
        else:
            diffs = np.zeros(n, dtype=np.float64)
            for i in range(1, n):
                diffs[i] = float(np.mean(np.abs(grays[i] - grays[i - 1])))
            diffs[0] = diffs[1]  # mirror
            scores = diffs

        K = max(1, min(int(K), n))
        top = np.argsort(-scores)[:K]
        keep = np.sort(top)
        return ProposalSet(
            frame_indices=proposal.frame_indices[keep],
            timestamps_sec=proposal.timestamps_sec[keep],
            scores=scores[keep],
            source=f"{proposal.source}+{self.name}",
        )
