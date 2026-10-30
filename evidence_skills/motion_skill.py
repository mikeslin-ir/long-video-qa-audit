from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .base import EvidenceSkill, FrameInfo, video_cache_key


class MotionSkill(EvidenceSkill):
    name = "motion"

    def __init__(self, cache_dir: str = "results/skill_router_cache"):
        self.cache_dir = Path(cache_dir)

    def compute(
        self,
        video_path: str,
        frames: list[FrameInfo],
        question: str | None,
        metadata: dict[str, Any],
    ) -> np.ndarray:
        if not frames:
            return np.zeros((0,), dtype=np.float32)
        key = video_cache_key(video_path, f"motion_{len(frames)}")
        cache_path = self.cache_dir / "motion" / f"{key}.npy"
        if cache_path.exists():
            return np.load(cache_path).astype(np.float32)

        out = np.zeros((len(frames),), dtype=np.float32)
        prev_gray = None
        for i, f in enumerate(frames):
            if f.image is None:
                continue
            gray = cv2.cvtColor(f.image, cv2.COLOR_RGB2GRAY)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray,
                    gray,
                    None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0,
                )
                mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                out[i] = float(np.mean(mag))
            prev_gray = gray

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, out)
        return out
