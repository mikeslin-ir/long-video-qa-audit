from __future__ import annotations

from typing import Any

import numpy as np

from .base import FrameInfo, video_cache_key
from .semantic_siglip_skill import SemanticSigLIPSkill


def _contains_any(text: str, words: tuple[str, ...]) -> bool:
    t = (text or "").lower()
    return any(w in t for w in words)


class ObjectSigLIPSkill(SemanticSigLIPSkill):
    """Open-vocabulary object evidence scoring using SigLIP text-image similarity."""

    name = "object_siglip"
    requires_question = True

    def compute(
        self,
        video_path: str,
        frames: list[FrameInfo],
        question: str | None,
        metadata: dict[str, Any],
    ) -> np.ndarray:
        if not frames:
            return np.zeros((0,), dtype=np.float32)
        if not self._load():
            return np.zeros((len(frames),), dtype=np.float32)

        prompts = self._resolve_object_prompts(question=question, dataset=metadata.get("dataset"))

        frame_idx = [f.frame_index for f in frames]
        vkey = video_cache_key(
            video_path,
            f"objsiglip_img_{self.model_id}_{len(frames)}_{frame_idx[:3]}_{frame_idx[-3:]}",
        )
        pkey = video_cache_key(video_path, f"objsiglip_txt_{self.model_id}_{'|'.join(prompts)}")
        img_cache = self.cache_dir / "object_siglip" / f"{vkey}_img.npy"
        txt_cache = self.cache_dir / "object_siglip" / f"{pkey}_txt.npy"
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

        sims = image_emb @ text_emb.T
        return sims.max(axis=1).astype(np.float32)

    @staticmethod
    def _resolve_object_prompts(question: str | None, dataset: str | None) -> list[str]:
        q = (question or "").lower()
        d = (dataset or "").lower()

        violence_words = (
            "violence",
            "violent",
            "fight",
            "fighting",
            "assault",
            "shoot",
            "explosion",
            "accident",
            "weapon",
        )
        steal_words = (
            "steal",
            "stealing",
            "shoplift",
            "robbery",
            "arrest",
            "handcuff",
        )

        if _contains_any(q, violence_words) or d == "xd_violence":
            return [
                "a person",
                "a weapon",
                "a gun",
                "a knife",
                "fire",
                "an explosion",
                "a damaged car",
                "blood",
            ]
        if _contains_any(q, steal_words) or d == "ucf_crime":
            return [
                "a person",
                "a handbag",
                "a backpack",
                "a cash register",
                "a store shelf",
                "a vehicle",
                "a police officer",
            ]
        return [
            "a person",
            "a vehicle",
            "an indoor scene",
            "an outdoor scene",
        ]


class ObjectSkillStub(ObjectSigLIPSkill):
    """Backward-compatible alias; now backed by real SigLIP object scoring."""

    name = "object_stub"
