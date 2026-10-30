from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .base import EvidenceSkill, FrameInfo, video_cache_key


class OCRTextSkill(EvidenceSkill):
    name = "ocr_text"
    requires_question = True

    def __init__(
        self,
        cache_dir: str = "results/skill_router_cache",
        backend: str = "auto",
        conf_threshold: float = 0.35,
    ):
        self.cache_dir = Path(cache_dir)
        self.backend = backend
        self.conf_threshold = float(conf_threshold)
        self._warned = False
        self._reader = None

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
            "whose",
            "why",
            "how",
            "is",
            "are",
            "was",
            "were",
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
            "scene",
            "shown",
        }
        return {t for t in tokens if len(t) >= 3 and t not in stop}

    def _easyocr_reader(self):
        if self._reader is not None:
            return self._reader
        import easyocr

        self._reader = easyocr.Reader(["en"], gpu=False)
        return self._reader

    def _ocr_with_easyocr(self, frame: np.ndarray) -> list[tuple[str, float]]:
        reader = self._easyocr_reader()
        out = reader.readtext(frame)
        found: list[tuple[str, float]] = []
        for item in out:
            if len(item) < 3:
                continue
            txt = str(item[1]).strip().lower()
            conf = float(item[2])
            if txt:
                found.append((txt, conf))
        return found

    def _ocr_with_tesseract(self, frame: np.ndarray) -> list[tuple[str, float]]:
        import cv2
        import pytesseract

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
        n = len(data.get("text", []))
        found: list[tuple[str, float]] = []
        for i in range(n):
            txt = str(data["text"][i]).strip().lower()
            conf_raw = data["conf"][i]
            try:
                conf = float(conf_raw) / 100.0
            except Exception:
                conf = 0.0
            if txt:
                found.append((txt, conf))
        return found

    def _choose_backend(self) -> str | None:
        if self.backend in {"tesseract", "easyocr"}:
            return self.backend
        if self.backend != "auto":
            return None

        try:
            import pytesseract  # noqa: F401

            return "tesseract"
        except Exception:
            pass
        try:
            import easyocr  # noqa: F401

            return "easyocr"
        except Exception:
            return None

    def _extract_frame_tokens(self, frame: np.ndarray, backend: str) -> list[tuple[str, float]]:
        try:
            if backend == "tesseract":
                return self._ocr_with_tesseract(frame)
            if backend == "easyocr":
                return self._ocr_with_easyocr(frame)
            return []
        except Exception:
            return []

    def _cache_path(self, video_path: str, frame_indices: list[int], backend: str) -> Path:
        key = video_cache_key(video_path, f"ocr_{backend}_{frame_indices[:4]}_{frame_indices[-4:]}")
        p = self.cache_dir / "ocr" / f"{key}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def compute(
        self,
        video_path: str,
        frames: list[FrameInfo],
        question: str | None,
        metadata: dict[str, Any],
    ) -> np.ndarray:
        if not frames:
            return np.zeros((0,), dtype=np.float32)

        backend = self._choose_backend()
        if backend is None:
            if not self._warned:
                warnings.warn(
                    "[OCRTextSkill] no OCR backend found (install pytesseract or easyocr). Returning zeros."
                )
                self._warned = True
            return np.zeros((len(frames),), dtype=np.float32)

        frame_indices = [f.frame_index for f in frames]
        cache_path = self._cache_path(video_path, frame_indices, backend)
        cached: dict[str, list[list[Any]]] = {}
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text())
            except Exception:
                cached = {}

        keywords = self._question_keywords(question)
        scores = np.zeros((len(frames),), dtype=np.float32)
        changed = False

        for i, f in enumerate(frames):
            frame_key = str(f.frame_index)
            if frame_key in cached:
                entries = cached[frame_key]
            else:
                if f.image is None:
                    entries = []
                else:
                    entries = self._extract_frame_tokens(f.image, backend)
                cached[frame_key] = [[t, float(c)] for (t, c) in entries]
                changed = True

            # Only trust medium-confidence OCR tokens for question matching.
            tokens = [
                (str(t), float(c))
                for t, c in cached.get(frame_key, [])
                if float(c) >= self.conf_threshold and str(t).strip()
            ]

            if not tokens:
                scores[i] = 0.0
                continue

            if keywords:
                hits = 0
                total = 0
                for txt, conf in tokens:
                    tt = re.findall(r"[a-zA-Z0-9']+", txt)
                    if not tt:
                        continue
                    total += 1
                    if any(tok in keywords for tok in tt):
                        hits += 1
                scores[i] = float(hits) / float(max(total, 1))
            else:
                # Fallback signal: text presence when question has no specific keywords.
                scores[i] = min(1.0, 0.15 + 0.08 * len(tokens))

        if changed:
            try:
                cache_path.write_text(json.dumps(cached))
            except Exception:
                pass

        return scores.astype(np.float32)


class OCRSkillStub(OCRTextSkill):
    """Backward-compatible alias; now backed by real OCR text scoring."""

    name = "ocr_stub"
