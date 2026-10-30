"""CLAP-based audio-language skill.

Uses pre-extracted LAION-CLAP music+speech audio embeddings
(``laion/larger_clap_music_and_speech``) found at
``agentic_routing/features/<dataset>/clap/<video_id>.npz`` (key ``embed``,
shape ``[512]``) and re-encodes the question text on-the-fly with the same
CLAP text tower. The frame score is the question-audio cosine similarity
tiled across all candidate frames -- the precomputed embed is clip-level,
so it carries no temporal information, but the constant per-video score
is still a valid planner signal that pushes the cascade toward
audio-aware experts when audio is relevant to the question (e.g.,
``music_avqa``).

Lazy-loads the text encoder; if neither the npz file nor the
``transformers`` CLAP class are available, returns a zero score (i.e. the
planner falls back to other skills without crashing).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from data_paths import get_data_root

from .base import EvidenceSkill, FrameInfo, video_cache_key


_DEFAULT_FEATURE_ROOTS = [
    str(get_data_root() / "agentic_routing" / "features"),
]


def _resolve_clap_feature(
    video_path: str,
    feature_roots: list[str],
    dataset_hint: str | None = None,
) -> Path | None:
    """Map a video path to the matching pre-extracted CLAP .npz, if any.

    The extractor stores files as ``<root>/<dataset>/clap/<video_stem>.npz``
    where ``video_stem`` is the bare video id (e.g. ``00000002``). We
    search under each root for the first matching stem in any subfolder
    named ``clap``. ``dataset_hint`` (e.g. ``"music_avqa"``) prunes the
    search to that subtree when known.
    """
    stem = Path(video_path).stem
    for root_str in feature_roots:
        root = Path(root_str)
        if not root.exists():
            continue
        if dataset_hint:
            cand = root / dataset_hint / "clap" / f"{stem}.npz"
            if cand.exists():
                return cand
        for clap_dir in root.glob("*/clap"):
            cand = clap_dir / f"{stem}.npz"
            if cand.exists():
                return cand
    return None


_QUESTION_TEMPLATE = "the sound of {}"
_KEYWORD_RE = re.compile(r"[a-zA-Z][a-zA-Z']+")


def _question_to_clap_prompt(question: str) -> str:
    """Lift a natural-language question into a CLAP-friendly audio prompt.

    CLAP was trained on short audio-caption pairs ("the sound of ..."),
    not on questions. We strip leading wh-words and keep content tokens.
    Empty result falls back to the raw question.
    """
    q = (question or "").lower().strip().rstrip("?")
    q = re.sub(r"^(what|when|where|which|who|how|why|is|are|does|do|can)\s+", "", q)
    toks = _KEYWORD_RE.findall(q)
    stop = {"the", "a", "an", "of", "in", "on", "at", "to", "for", "and",
            "or", "is", "it", "this", "that", "with", "by", "as", "be",
            "from", "into", "than"}
    keep = [t for t in toks if t not in stop]
    body = " ".join(keep) if keep else q
    return _QUESTION_TEMPLATE.format(body) if body else "music"


class CLAPSkill(EvidenceSkill):
    """Question-conditioned audio relevance from pre-extracted CLAP embeds."""

    name = "clap_alm"
    requires_audio = True
    requires_question = True

    def __init__(
        self,
        cache_dir: str = "results/skill_router_cache",
        model_id: str = "laion/larger_clap_music_and_speech",
        feature_roots: list[str] | None = None,
        device: str = "cpu",
    ):
        self.cache_dir = Path(cache_dir)
        self.model_id = model_id
        self.feature_roots = list(feature_roots) if feature_roots else list(_DEFAULT_FEATURE_ROOTS)
        self.device = device
        self._text_model = None
        self._text_processor = None
        self._load_failed = False

    # -- lazy model load -------------------------------------------------
    def _ensure_text_encoder(self) -> bool:
        if self._text_model is not None:
            return True
        if self._load_failed:
            return False
        try:
            import torch  # noqa: PLC0415
            from transformers import ClapModel, ClapProcessor  # noqa: PLC0415
        except Exception:
            self._load_failed = True
            return False
        try:
            self._text_model = ClapModel.from_pretrained(self.model_id).to(self.device).eval()
            self._text_processor = ClapProcessor.from_pretrained(self.model_id)
            self._torch = torch
        except Exception:
            self._load_failed = True
            return False
        return True

    # -- main skill API --------------------------------------------------
    def compute(
        self,
        video_path: str,
        frames: list[FrameInfo],
        question: str | None,
        metadata: dict[str, Any],
    ) -> np.ndarray:
        n = len(frames)
        if n == 0:
            return np.zeros((0,), dtype=np.float32)
        q = (question or "").strip()
        if not q:
            return np.zeros((n,), dtype=np.float32)

        dataset_hint = None
        if isinstance(metadata, dict):
            dataset_hint = metadata.get("dataset")

        score_cache = self.cache_dir / "clap_alm"
        score_cache.mkdir(parents=True, exist_ok=True)
        key = video_cache_key(video_path, f"clap_{self.model_id}|{q.lower()}")
        cpath = score_cache / f"{key}.json"
        if cpath.exists():
            try:
                cached = json.loads(cpath.read_text())
                scalar = float(cached.get("score", 0.0))
                return np.full((n,), scalar, dtype=np.float32)
            except Exception:
                pass

        feat_path = _resolve_clap_feature(video_path, self.feature_roots, dataset_hint)
        if feat_path is None:
            cpath.write_text(json.dumps({"score": 0.0, "reason": "no_feature"}))
            return np.zeros((n,), dtype=np.float32)
        try:
            embed = np.load(feat_path)["embed"].astype(np.float32)
        except Exception:
            cpath.write_text(json.dumps({"score": 0.0, "reason": "load_failed"}))
            return np.zeros((n,), dtype=np.float32)
        if embed.ndim > 1:
            embed = embed.mean(axis=0)
        a_norm = float(np.linalg.norm(embed)) or 1.0
        embed = embed / a_norm

        if not self._ensure_text_encoder():
            cpath.write_text(json.dumps({"score": 0.0, "reason": "no_text_encoder"}))
            return np.zeros((n,), dtype=np.float32)

        prompt = _question_to_clap_prompt(q)
        torch = self._torch
        try:
            with torch.no_grad():
                inputs = self._text_processor(text=[prompt], return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                out = self._text_model.get_text_features(**inputs)
                # transformers >=5: get_text_features returns
                # BaseModelOutputWithPooling; pooler_output is the
                # 512-d joint-space text embedding (matches the audio
                # ``embed`` produced by extract_audio_features.py).
                if hasattr(out, "pooler_output") and out.pooler_output is not None:
                    t_tensor = out.pooler_output
                elif hasattr(out, "last_hidden_state"):
                    t_tensor = out.last_hidden_state.mean(dim=1)
                else:
                    t_tensor = out
                t_embed = t_tensor.cpu().numpy().astype(np.float32)[0]
        except Exception as e:
            cpath.write_text(json.dumps({"score": 0.0, "reason": f"text_encode_failed: {type(e).__name__}: {e}"}))
            return np.zeros((n,), dtype=np.float32)
        t_norm = float(np.linalg.norm(t_embed)) or 1.0
        t_embed = t_embed / t_norm
        sim = float(np.dot(embed, t_embed))
        # CLAP cosine in [-1, 1]; rescale to [0, 1] so it composes with
        # other skills that produce non-negative weights.
        scalar = max(0.0, (sim + 1.0) / 2.0)
        cpath.write_text(json.dumps({"score": scalar, "prompt": prompt}))
        return np.full((n,), scalar, dtype=np.float32)
