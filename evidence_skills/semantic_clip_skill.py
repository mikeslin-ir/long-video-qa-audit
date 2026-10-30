from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import numpy as np
from PIL import Image

from .base import EvidenceSkill, FrameInfo, video_cache_key


class SemanticCLIPSkill(EvidenceSkill):
    name = "semantic_clip"
    requires_question = True

    def __init__(
        self,
        cache_dir: str = "results/skill_router_cache",
        model_id: str = "openai/clip-vit-base-patch32",
        batch_size: int = 32,
    ):
        self.cache_dir = Path(cache_dir)
        self.model_id = model_id
        self.batch_size = int(batch_size)
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._warned = False

    def _load(self) -> bool:
        if self._model is not None:
            return True
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._processor = CLIPProcessor.from_pretrained(self.model_id)
            self._model = CLIPModel.from_pretrained(self.model_id).to(self._device).eval()
            return True
        except Exception as e:
            if not self._warned:
                warnings.warn(f"[SemanticCLIPSkill] failed to load {self.model_id}: {e}")
                self._warned = True
            return False

    @staticmethod
    def _default_prompts(dataset: str | None) -> list[str]:
        d = (dataset or "").lower()
        if d in {"xd_violence", "ucf_crime"}:
            return [
                "a surveillance video showing an abnormal or criminal event",
                "a person fighting",
                "a person stealing",
                "a weapon",
                "a car accident",
                "fire or explosion",
            ]
        return []

    def _resolve_prompts(self, question: str | None, metadata: dict[str, Any]) -> list[str]:
        prompts = metadata.get("evidence_prompts") or []
        if prompts:
            return [str(p) for p in prompts]
        fallback = self._default_prompts(metadata.get("dataset"))
        if fallback:
            return fallback
        return [question or "the key visual event in this video"]

    def embed(
        self,
        video_path: str,
        frames: list[FrameInfo],
        question: str | None,
        metadata: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Cached (image_emb [N,D], text_emb [P,D]); None if the model is unavailable."""
        if not frames or not self._load():
            return None

        prompts = self._resolve_prompts(question, metadata)
        frame_idx = [f.frame_index for f in frames]
        vkey = video_cache_key(video_path, f"clipimg_{self.model_id}_{len(frames)}_{frame_idx[:3]}_{frame_idx[-3:]}")
        pkey = video_cache_key(video_path, f"cliptxt_{self.model_id}_{'|'.join(prompts)}")
        img_cache = self.cache_dir / "semantic_clip" / f"{vkey}_img.npy"
        txt_cache = self.cache_dir / "semantic_clip" / f"{pkey}_txt.npy"
        img_cache.parent.mkdir(parents=True, exist_ok=True)

        if img_cache.exists():
            image_emb = np.load(img_cache)
        else:
            image_emb = self._encode_images(frames)
            np.save(img_cache, image_emb)

        if txt_cache.exists():
            text_emb = np.load(txt_cache)
        else:
            text_emb = self._encode_text(prompts)
            np.save(txt_cache, text_emb)

        return image_emb, text_emb

    def compute(
        self,
        video_path: str,
        frames: list[FrameInfo],
        question: str | None,
        metadata: dict[str, Any],
    ) -> np.ndarray:
        if not frames:
            return np.zeros((0,), dtype=np.float32)
        pair = self.embed(video_path, frames, question, metadata)
        if pair is None:
            return np.zeros((len(frames),), dtype=np.float32)
        image_emb, text_emb = pair
        sims = image_emb @ text_emb.T
        return sims.max(axis=1).astype(np.float32)

    def _encode_images(self, frames: list[FrameInfo]) -> np.ndarray:
        import torch

        all_emb = []
        for i in range(0, len(frames), self.batch_size):
            batch = frames[i:i + self.batch_size]
            pil = [Image.fromarray(f.image) for f in batch if f.image is not None]
            if not pil:
                continue
            inputs = self._processor(images=pil, return_tensors="pt", padding=True).to(self._device)
            with torch.no_grad():
                feats = self._model.get_image_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            all_emb.append(feats.cpu().numpy().astype(np.float32))
        if not all_emb:
            return np.zeros((len(frames), 512), dtype=np.float32)
        out = np.concatenate(all_emb, axis=0)
        if out.shape[0] < len(frames):
            pad = np.zeros((len(frames) - out.shape[0], out.shape[1]), dtype=np.float32)
            out = np.concatenate([out, pad], axis=0)
        return out[: len(frames)]

    def _encode_text(self, prompts: list[str]) -> np.ndarray:
        import torch

        inputs = self._processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(self._device)
        with torch.no_grad():
            feats = self._model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return feats.cpu().numpy().astype(np.float32)
