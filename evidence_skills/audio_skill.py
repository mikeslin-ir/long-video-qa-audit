from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from video_audio import extract_audio_from_video

from .base import EvidenceSkill, FrameInfo, video_cache_key


class AudioEnergySkill(EvidenceSkill):
    name = "audio"
    requires_audio = True

    def __init__(
        self,
        cache_dir: str = "results/skill_router_cache",
        sr: int = 22050,
        window_sec: float = 0.5,
    ):
        self.cache_dir = Path(cache_dir)
        self.sr = int(sr)
        self.window_sec = float(window_sec)

    def compute(
        self,
        video_path: str,
        frames: list[FrameInfo],
        question: str | None,
        metadata: dict[str, Any],
    ) -> np.ndarray:
        if not frames:
            return np.zeros((0,), dtype=np.float32)
        key = video_cache_key(video_path, f"audio_{len(frames)}_{self.sr}_{self.window_sec}")
        cache_path = self.cache_dir / "audio" / f"{key}.npy"
        if cache_path.exists():
            return np.load(cache_path).astype(np.float32)

        y = _extract_audio_with_fallback(video_path, sr=self.sr)

        if y is None or len(y) == 0:
            out = np.zeros((len(frames),), dtype=np.float32)
        else:
            half = max(1, int(self.window_sec * self.sr / 2.0))
            out = np.zeros((len(frames),), dtype=np.float32)
            for i, f in enumerate(frames):
                c = int(f.timestamp_sec * self.sr)
                s = max(0, c - half)
                e = min(len(y), c + half)
                if e <= s:
                    out[i] = 0.0
                else:
                    out[i] = float(np.mean(np.abs(y[s:e])))

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, out)
        return out


def _extract_audio_with_fallback(video_path: str, sr: int) -> np.ndarray:
    return extract_audio_from_video(video_path, sr=sr)
