"""EARS event proposer: combine motion magnitude + audio energy to produce
event-saliency peaks. Wraps the EARS code (github.com/mikeslin-ir/EARS) without
modifying it. Point ``EARS_PATH`` at a checkout of it; by default a sibling
``../EARS`` directory next to this repository is used.
"""
from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..protocols import ProposalSet

# Make `import motion_features`, `import audio_features` etc. work by appending
# the EARS folder to sys.path lazily at import time.
_EARS_PATH = (Path(os.environ["EARS_PATH"]).expanduser() if os.environ.get("EARS_PATH", "").strip()
              else Path(__file__).resolve().parents[3] / "EARS")
if _EARS_PATH.exists() and str(_EARS_PATH) not in sys.path:
    sys.path.append(str(_EARS_PATH))


@dataclass
class EARSEventProposer:
    name: str = "ears_event"
    cost_per_video: float = 3.0
    cache_dir: str = "results/cascade_cache"
    motion_stride: int = 15     # frames between optical-flow samples
    motion_max_frames: int = 4096
    audio_segment_sec: float = 1.0
    alpha: float = 1.0          # audio weight
    beta: float = 1.0           # motion weight

    def __post_init__(self):
        Path(self.cache_dir, "ears_event").mkdir(parents=True, exist_ok=True)
        try:
            from motion_features import motion_magnitude_downsampled  # noqa: F401
            from audio_features import audio_energy_from_video        # noqa: F401
            from event_attention import compute_event_attention_from_signals  # noqa: F401
            self._available = True
        except Exception as e:
            self._available = False
            self._import_error = e
            warnings.warn(f"[cascade] ears_event is registered but EARS is not importable ({e!r}); "
                          f"plans that list it will run without it. Set EARS_PATH to a checkout of "
                          f"github.com/mikeslin-ir/EARS (looked in {_EARS_PATH}).", RuntimeWarning)

    def propose(self, video_path: str, question: str | None, meta: dict, K: int) -> ProposalSet:
        if not self._available:
            raise RuntimeError(f"EARS unavailable: {self._import_error}")

        from motion_features import motion_magnitude_downsampled
        from audio_features import audio_energy_from_video
        from event_attention import compute_event_attention_from_signals
        import cv2

        cache_path = self._cache_path(video_path)
        if cache_path.exists():
            data = np.load(cache_path)
            A = data["A"]
            sampled_idx = data["idx"]
            fps = float(data["fps"])
        else:
            cap = cv2.VideoCapture(str(video_path))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            # Motion: per-(stride)-frame magnitude
            M = motion_magnitude_downsampled(
                video_path, stride=self.motion_stride, max_frames=self.motion_max_frames,
            )
            # Audio energy per 1s segment; if no audio, returns zeros vector
            try:
                E = audio_energy_from_video(video_path, segment_sec=self.audio_segment_sec)
            except Exception:
                E = np.zeros(max(1, int(total / max(fps, 1e-6) / self.audio_segment_sec)))

            A = compute_event_attention_from_signals(E, M, alpha=self.alpha, beta=self.beta)

            # Reconstruct the frame indices the motion downsampling actually used
            n = min(total, self.motion_max_frames) if total > 0 else len(A)
            sampled_idx = np.array(list(range(0, n, self.motion_stride)), dtype=np.int64)
            if sampled_idx.size == 0 or sampled_idx[-1] != n - 1:
                sampled_idx = np.append(sampled_idx, max(0, n - 1))
            # Match lengths
            L = min(len(A), len(sampled_idx))
            A = A[:L]
            sampled_idx = sampled_idx[:L]

            np.savez(cache_path, A=A.astype(np.float32),
                     idx=sampled_idx.astype(np.int64), fps=np.float64(fps))

        if len(A) == 0:
            return ProposalSet(np.array([], dtype=np.int64), np.array([], dtype=np.float64),
                               source=self.name)

        # Pick top-K event-attention peaks (keep temporal order in output)
        K = max(1, min(int(K), len(A)))
        top = np.argsort(-A)[:K]
        keep = np.sort(top)
        chosen_idx = sampled_idx[keep]
        ts = chosen_idx.astype(np.float64) / max(fps, 1e-6)
        return ProposalSet(chosen_idx, ts, scores=A[keep].astype(np.float64), source=self.name)

    def _cache_path(self, video_path: str) -> Path:
        from hashlib import sha1
        key = sha1(
            f"{Path(video_path).resolve()}|{self.motion_stride}|{self.audio_segment_sec}|{self.alpha}|{self.beta}".encode()
        ).hexdigest()[:20]
        return Path(self.cache_dir, "ears_event", f"{key}.npz")
