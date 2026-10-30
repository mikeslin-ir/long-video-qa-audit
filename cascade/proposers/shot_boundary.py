"""Shot-boundary proposer (PySceneDetect backend).

Detects scene boundaries and emits one anchor per shot (the mid-frame of each
shot). For very long videos with many shots, returns the top-K shots by length
to stay within the K_proposal budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..protocols import ProposalSet


@dataclass
class ShotBoundaryProposer:
    name: str = "shot_boundary"
    cost_per_video: float = 2.0
    cache_dir: str = "results/cascade_cache"
    threshold: float = 30.0   # ContentDetector threshold (higher = fewer shots)

    def __post_init__(self):
        # Lazy: only import PySceneDetect on first use, raise gracefully if missing
        try:
            from scenedetect import SceneManager, open_video  # noqa: F401
            from scenedetect.detectors import ContentDetector   # noqa: F401
            self._available = True
        except Exception as e:
            self._available = False
            self._import_error = e
        Path(self.cache_dir, "shot_boundary").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ public
    def propose(self, video_path: str, question: str | None, meta: dict, K: int) -> ProposalSet:
        if not self._available:
            raise RuntimeError(f"PySceneDetect unavailable: {self._import_error}")

        cache_path = self._cache_path(video_path)
        if cache_path.exists():
            data = np.load(cache_path)
            shot_starts, shot_ends = data["starts"], data["ends"]
            fps = float(data["fps"])
        else:
            shot_starts, shot_ends, fps = self._detect_shots(video_path)
            np.savez(cache_path, starts=shot_starts, ends=shot_ends, fps=np.float64(fps))

        if len(shot_starts) == 0:
            return ProposalSet(np.array([], dtype=np.int64), np.array([], dtype=np.float64),
                               source=self.name)

        # Anchor = mid-frame of each shot; score = shot duration in seconds
        mids = ((shot_starts + shot_ends) // 2).astype(np.int64)
        durs = (shot_ends - shot_starts).astype(np.float64) / max(fps, 1e-6)

        # Cap to top-K shots by duration (longest first), but keep temporal order
        if len(mids) > K:
            top = np.argsort(-durs)[:K]
            keep = np.sort(top)
            mids = mids[keep]
            durs = durs[keep]

        ts = mids.astype(np.float64) / max(fps, 1e-6)
        return ProposalSet(mids, ts, scores=durs, source=self.name)

    # ----------------------------------------------------------------- helpers
    def _cache_path(self, video_path: str) -> Path:
        from hashlib import sha1
        key = sha1(f"{Path(video_path).resolve()}|{self.threshold}".encode()).hexdigest()[:20]
        return Path(self.cache_dir, "shot_boundary", f"{key}.npz")

    def _detect_shots(self, video_path: str) -> tuple[np.ndarray, np.ndarray, float]:
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import ContentDetector

        video = open_video(str(video_path))
        fps = float(video.frame_rate or 30.0)

        sm = SceneManager()
        sm.add_detector(ContentDetector(threshold=self.threshold))
        sm.detect_scenes(video=video)
        scene_list = sm.get_scene_list()

        if not scene_list:
            # Single-shot video
            cap = cv2.VideoCapture(str(video_path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return (np.array([0], dtype=np.int64),
                    np.array([max(0, total - 1)], dtype=np.int64), fps)

        starts = np.array([int(s.get_frames()) for s, _ in scene_list], dtype=np.int64)
        ends = np.array([int(e.get_frames()) - 1 for _, e in scene_list], dtype=np.int64)
        return starts, ends, fps
