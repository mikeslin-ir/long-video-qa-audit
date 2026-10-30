"""Cache-aware SigLIP aligner (L3).

Reuses the question-independent SigLIP *image* embeddings pre-computed offline
by ``scripts/extract_shot_keyframe_features.py``. At query time it only runs a
single SigLIP *text* forward over the evidence prompts and a dot product --
no per-frame image encoding, no video decode.

Falls back to encoding the passed-in RGB frames when a keyframe is not present
in the cache (e.g. a uniform-anchor frame mixed in by the router), so the
aligner is always safe to use.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..protocols import ProposalSet
from ..proposers.keyframe_cache import keyframe_cache_key
from evidence_skills.semantic_siglip_skill import SemanticSigLIPSkill
from evidence_skills.base import FrameInfo


@dataclass
class CachedSigLIPAligner:
    name: str = "semantic_siglip_cached"
    modality: str = "vision"
    cost_per_frame: float = 0.0  # image emb is precomputed; text encode is per-video
    cache_dir: str = "results/keyframe_cache"
    model_id: str = "google/siglip-base-patch16-224"
    _skill: SemanticSigLIPSkill = field(default=None, repr=False)  # type: ignore[assignment]

    def __post_init__(self):
        # Reuse the SigLIP skill purely for its text encoder + image fallback.
        self._skill = SemanticSigLIPSkill(model_id=self.model_id)

    def _load_cache(self, video_path: str) -> dict | None:
        npz_path = Path(self.cache_dir) / f"{keyframe_cache_key(video_path)}.npz"
        if not npz_path.exists():
            return None
        data = np.load(npz_path, allow_pickle=True)
        idx = np.asarray(data["keyframe_indices"], dtype=np.int64)
        emb = np.asarray(data["siglip_emb"], dtype=np.float32)
        if idx.size == 0 or emb.shape[0] != idx.size:
            return None
        return {int(fi): emb[r] for r, fi in enumerate(idx.tolist())}

    def _resolve_prompts(self, question: str | None, meta: dict) -> list[str]:
        prompts = meta.get("evidence_prompts") or []
        if prompts:
            return [str(p) for p in prompts]
        return [question or "the key visual event in this video"]

    def align(
        self,
        keyframes: ProposalSet,
        frames_rgb: list[np.ndarray],
        video_path: str,
        question: str | None,
        meta: dict,
    ) -> np.ndarray:
        n = len(keyframes)
        if n == 0:
            return np.zeros((0,), dtype=np.float64)
        if not self._skill._load():
            return np.zeros((n,), dtype=np.float64)

        cache = self._load_cache(video_path)

        # Gather an image embedding per keyframe: from cache when available,
        # else encode the passed-in RGB frame on the fly.
        emb_dim = None
        rows: list[np.ndarray | None] = []
        miss_positions: list[int] = []
        for i in range(n):
            fi = int(keyframes.frame_indices[i])
            vec = cache.get(fi) if cache else None
            if vec is not None:
                rows.append(vec.astype(np.float32))
                emb_dim = vec.shape[0]
            else:
                rows.append(None)
                miss_positions.append(i)

        # Encode any cache-miss frames in one batch.
        if miss_positions:
            miss_frames = [
                FrameInfo(
                    frame_index=int(keyframes.frame_indices[i]),
                    timestamp_sec=float(keyframes.timestamps_sec[i]),
                    image=frames_rgb[i] if i < len(frames_rgb) else None,
                )
                for i in miss_positions
            ]
            enc = self._skill._encode_images(miss_frames)
            for pos, vec in zip(miss_positions, enc):
                rows[pos] = vec.astype(np.float32)
                emb_dim = vec.shape[0]

        if emb_dim is None:
            return np.zeros((n,), dtype=np.float64)
        image_emb = np.stack(
            [r if r is not None else np.zeros((emb_dim,), dtype=np.float32) for r in rows],
            axis=0,
        )

        # One text forward (cheap), then dot product.
        prompts = self._resolve_prompts(question, meta)
        text_emb = self._skill._encode_text(prompts)
        sims = image_emb @ text_emb.T
        return sims.max(axis=1).astype(np.float64)
