"""Keyframe-cache proposer (L1).

Reads shot-aware keyframes pre-computed offline by
``scripts/extract_shot_keyframe_features.py`` instead of decoding the whole
video at query time. Each cached video stores one keyframe per detected shot
(PySceneDetect mid-frame) plus question-independent SigLIP image embeddings.

This proposer only emits the *time anchors* (frame indices + timestamps); the
stored embeddings are consumed downstream by the cache-aware aligner. If a
video has no cache entry, ``propose`` raises FileNotFoundError so the router
can fall back to another proposer (e.g. uniform_stride).
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path

import numpy as np

from ..protocols import ProposalSet


def keyframe_cache_key(video_path: str) -> str:
    """Stable per-video key, identical to the offline extractor."""
    p = str(Path(video_path).resolve())
    return sha1(p.encode()).hexdigest()[:20]


@dataclass
class KeyframeCacheProposer:
    name: str = "keyframe_cache"
    cost_per_video: float = 0.05  # sec, just reads an .npz (no decode/inference)
    cache_dir: str = "results/keyframe_cache"

    def _npz_path(self, video_path: str) -> Path:
        return Path(self.cache_dir) / f"{keyframe_cache_key(video_path)}.npz"

    def has_cache(self, video_path: str) -> bool:
        return self._npz_path(video_path).exists()

    def propose(self, video_path: str, question: str | None, meta: dict, K: int) -> ProposalSet:
        npz_path = self._npz_path(video_path)
        if not npz_path.exists():
            raise FileNotFoundError(
                f"no keyframe cache for {video_path} at {npz_path}; "
                "run scripts/extract_shot_keyframe_features.py first"
            )

        data = np.load(npz_path, allow_pickle=True)
        idx = np.asarray(data["keyframe_indices"], dtype=np.int64)
        ts = np.asarray(data["timestamps"], dtype=np.float64)
        if idx.size == 0:
            return ProposalSet(np.array([], dtype=np.int64),
                               np.array([], dtype=np.float64), source=self.name)

        # Shot duration as saliency score (longer shots = more screen time).
        starts = np.asarray(data["starts"], dtype=np.int64)
        ends = np.asarray(data["ends"], dtype=np.int64)
        fps = float(data["fps"]) if "fps" in data else 30.0
        if starts.size == idx.size:
            durs = (ends - starts).astype(np.float64) / max(fps, 1e-6)
        else:
            durs = np.ones((idx.size,), dtype=np.float64)

        # Cap to top-K shots by duration, but keep temporal order.
        if idx.size > K > 0:
            top = np.argsort(-durs)[:K]
            keep = np.sort(top)
            idx, ts, durs = idx[keep], ts[keep], durs[keep]

        return ProposalSet(idx, ts, scores=durs, source=self.name)
