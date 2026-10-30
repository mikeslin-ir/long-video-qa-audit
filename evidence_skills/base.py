from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class FrameInfo:
    frame_index: int
    timestamp_sec: float
    image: np.ndarray | None = None
    image_path: str | None = None
    embedding_path: str | None = None


class EvidenceSkill(ABC):
    name: str = "base"
    requires_audio: bool = False
    requires_question: bool = False

    @abstractmethod
    def compute(
        self,
        video_path: str,
        frames: list[FrameInfo],
        question: str | None,
        metadata: dict[str, Any],
    ) -> np.ndarray:
        """Return per-frame score array with shape [len(frames)]."""


def video_cache_key(video_path: str, extra: str = "") -> str:
    p = str(Path(video_path).resolve())
    return sha1(f"{p}|{extra}".encode()).hexdigest()[:20]


def sample_candidate_frames(
    video_path: str,
    stride: int = 5,
    max_frames: int | None = None,
) -> tuple[list[FrameInfo], dict[str, Any]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if total_frames <= 0:
        cap.release()
        return [], {
            "fps": fps,
            "total_frames": 0,
            "duration_sec": 0.0,
            "frame_indices": [],
        }

    indices = list(range(0, total_frames, max(1, stride)))
    if indices[-1] != total_frames - 1:
        indices.append(total_frames - 1)
    if max_frames is not None:
        indices = indices[:max_frames]

    frames: list[FrameInfo] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(
            FrameInfo(
                frame_index=int(idx),
                timestamp_sec=float(idx) / max(fps, 1e-6),
                image=frame_rgb,
            )
        )
    cap.release()

    metadata = {
        "fps": fps,
        "total_frames": total_frames,
        "duration_sec": float(total_frames) / max(fps, 1e-6),
        "frame_indices": [f.frame_index for f in frames],
    }
    return frames, metadata
