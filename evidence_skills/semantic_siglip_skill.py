from __future__ import annotations

import warnings

from .semantic_clip_skill import SemanticCLIPSkill


class SemanticSigLIPSkill(SemanticCLIPSkill):
    name = "semantic_siglip"

    def __init__(
        self,
        cache_dir: str = "results/skill_router_cache",
        model_id: str = "google/siglip-base-patch16-224",
        batch_size: int = 32,
    ):
        super().__init__(cache_dir=cache_dir, model_id=model_id, batch_size=batch_size)

    def _load(self) -> bool:
        if self._model is not None:
            return True
        try:
            import torch
            from transformers import AutoModel, AutoProcessor

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = AutoModel.from_pretrained(self.model_id).to(self._device).eval()
            return True
        except Exception as e:
            if not self._warned:
                warnings.warn(
                    f"[SemanticSigLIPSkill] model unavailable ({self.model_id}). Skill skipped: {e}"
                )
                self._warned = True
            return False

    def _encode_images(self, frames):
        import torch
        from PIL import Image
        import numpy as np

        all_emb = []
        for i in range(0, len(frames), self.batch_size):
            batch = frames[i:i + self.batch_size]
            pil = [Image.fromarray(f.image) for f in batch if f.image is not None]
            if not pil:
                continue
            inputs = self._processor(images=pil, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self._device)
            with torch.no_grad():
                feats = self._model.get_image_features(pixel_values=pixel_values)
                if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
                    feats = feats.pooler_output
                elif hasattr(feats, "last_hidden_state"):
                    feats = feats.last_hidden_state[:, 0]
                feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            all_emb.append(feats.cpu().numpy().astype(np.float32))
        if not all_emb:
            return np.zeros((len(frames), 768), dtype=np.float32)
        out = np.concatenate(all_emb, axis=0)
        if out.shape[0] < len(frames):
            pad = np.zeros((len(frames) - out.shape[0], out.shape[1]), dtype=np.float32)
            out = np.concatenate([out, pad], axis=0)
        return out[: len(frames)]

    def _encode_text(self, prompts):
        import torch

        inputs = self._processor(
            text=prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        input_ids = inputs["input_ids"].to(self._device)
        with torch.no_grad():
            feats = self._model.get_text_features(input_ids=input_ids)
            if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
                feats = feats.pooler_output
            elif hasattr(feats, "last_hidden_state"):
                feats = feats.last_hidden_state[:, 0]
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return feats.cpu().numpy().astype("float32")
