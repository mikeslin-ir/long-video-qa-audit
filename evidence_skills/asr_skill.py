from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from video_audio import extract_audio_from_video

from .base import EvidenceSkill, FrameInfo, video_cache_key


class ASRWhisperSkill(EvidenceSkill):
    name = "asr_whisper"
    requires_audio = True
    requires_question = True

    def __init__(
        self,
        cache_dir: str = "results/skill_router_cache",
        model_id: str = "openai/whisper-tiny.en",
        sr: int = 16000,
        window_sec: float = 1.0,
    ):
        self.cache_dir = Path(cache_dir)
        self.model_id = model_id
        self.sr = int(sr)
        self.window_sec = float(window_sec)
        self._pipe = None
        self._warned = False

    def _load(self) -> bool:
        if self._pipe is not None:
            return True
        try:
            import torch
            from transformers import pipeline

            device = 0 if torch.cuda.is_available() else -1
            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model_id,
                device=device,
            )
            return True
        except Exception as e:
            if not self._warned:
                warnings.warn(f"[ASRWhisperSkill] failed to load {self.model_id}: {e}")
                self._warned = True
            return False

    @staticmethod
    def _question_keywords(question: str | None) -> set[str]:
        text = (question or "").lower()
        tokens = re.findall(r"[a-zA-Z0-9']+", text)
        stop = {
            "what",
            "when",
            "where",
            "which",
            "who",
            "whom",
            "whose",
            "why",
            "how",
            "is",
            "are",
            "was",
            "were",
            "do",
            "does",
            "did",
            "the",
            "a",
            "an",
            "in",
            "on",
            "at",
            "to",
            "of",
            "for",
            "with",
            "this",
            "that",
            "video",
            "person",
            "people",
        }
        return {t for t in tokens if len(t) >= 3 and t not in stop}

    def _transcribe_with_timestamps(self, video_path: str) -> list[dict[str, Any]]:
        if not self._load():
            return []

        key = video_cache_key(video_path, f"asr_{self.model_id}_{self.sr}")
        cache_path = self.cache_dir / "asr" / f"{key}.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text())
            except Exception:
                pass

        y = extract_audio_from_video(video_path, sr=self.sr)
        if y is None or len(y) == 0:
            return []

        try:
            out = self._pipe(
                {"raw": y, "sampling_rate": self.sr},
                return_timestamps="word",
            )
            chunks = out.get("chunks", []) if isinstance(out, dict) else []
            words = []
            for ch in chunks:
                ts = ch.get("timestamp", (None, None))
                s, e = ts if isinstance(ts, (tuple, list)) and len(ts) == 2 else (None, None)
                if s is None or e is None:
                    continue
                txt = str(ch.get("text", "")).strip()
                if not txt:
                    continue
                words.append({"start": float(s), "end": float(e), "text": txt.lower()})
            cache_path.write_text(json.dumps(words))
            return words
        except Exception:
            return []

    def compute(
        self,
        video_path: str,
        frames: list[FrameInfo],
        question: str | None,
        metadata: dict[str, Any],
    ) -> np.ndarray:
        if not frames:
            return np.zeros((0,), dtype=np.float32)

        words = self._transcribe_with_timestamps(video_path)
        if not words:
            return np.zeros((len(frames),), dtype=np.float32)

        keywords = self._question_keywords(question)
        half = max(0.25, self.window_sec / 2.0)
        scores = np.zeros((len(frames),), dtype=np.float32)

        for i, f in enumerate(frames):
            t = float(f.timestamp_sec)
            ws = t - half
            we = t + half
            active = [w for w in words if (w["start"] <= we and w["end"] >= ws)]
            if not active:
                continue

            if keywords:
                hit = 0
                for w in active:
                    toks = re.findall(r"[a-zA-Z0-9']+", w["text"])
                    if any(tok in keywords for tok in toks):
                        hit += 1
                scores[i] = float(hit) / float(len(active)) if active else 0.0
            else:
                # If no useful keywords in question, use speech presence as weak evidence.
                scores[i] = min(1.0, 0.2 + 0.05 * len(active))

        return scores.astype(np.float32)


class ASRSkillStub(ASRWhisperSkill):
    """Backward-compatible alias; now backed by real Whisper ASR scoring."""

    name = "asr_stub"
