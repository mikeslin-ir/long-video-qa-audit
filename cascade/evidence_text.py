"""Build textual evidence (ASR transcript) aligned with selected frames.

This module assembles a short text passage that summarizes the spoken
narration around the K frames a selector chose to forward to the VLM.
The text is injected into the VLM prompt so the reasoner can ground its
answer in both pixels *and* dense narration, instead of relying on
pixels alone.

ASR transcript cache is produced by :class:`ASRWhisperSkill` during the
selection stage and persisted as JSON word/chunk records under
``<cache_dir>/asr/<video_hash>.json``::

    [{"start": float, "end": float, "text": str}, ...]

We re-read that cache directly here (no model load required), align it
with the selected frame timestamps, and format a compact bullet list of
"near-frame" speech windows plus a short global summary.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

from data_paths import get_cascade_cache_dir
from evidence_skills.base import video_cache_key

DEFAULT_CACHE_DIR = get_cascade_cache_dir()
DEFAULT_ASR_MODEL = "openai/whisper-tiny.en"
DEFAULT_ASR_SR = 16000


def _format_ts(t: float) -> str:
    """mm:ss formatting for a non-negative second offset."""
    t = max(0.0, float(t))
    m = int(t // 60)
    s = int(round(t - 60 * m))
    if s == 60:
        m += 1
        s = 0
    return f"{m:02d}:{s:02d}"


def load_asr_words(
    video_path: str,
    cache_dir: str = DEFAULT_CACHE_DIR,
    model_id: str = DEFAULT_ASR_MODEL,
    sr: int = DEFAULT_ASR_SR,
) -> list[dict]:
    """Return cached Whisper word/chunk records for a video, or [] if absent."""
    key = video_cache_key(video_path, f"asr_{model_id}_{sr}")
    path = Path(cache_dir) / "asr" / f"{key}.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except Exception:
        return []
    out = []
    for w in data:
        try:
            out.append({"start": float(w["start"]),
                        "end": float(w["end"]),
                        "text": str(w.get("text", "")).strip()})
        except Exception:
            continue
    return out


def _words_in_window(words: Sequence[dict], t0: float, t1: float) -> list[dict]:
    return [w for w in words if w["end"] >= t0 and w["start"] <= t1]


def _join_text(words: Iterable[dict], max_chars: int = 400) -> str:
    out = []
    total = 0
    for w in words:
        txt = w["text"].strip()
        if not txt:
            continue
        if total + len(txt) + 1 > max_chars and out:
            out.append("...")
            break
        out.append(txt)
        total += len(txt) + 1
    # Collapse extra spaces
    joined = " ".join(out)
    return " ".join(joined.split())


def build_asr_evidence_text(
    video_path: str,
    frame_timestamps_sec: Sequence[float],
    cache_dir: str = DEFAULT_CACHE_DIR,
    window_sec: float = 8.0,
    per_frame_chars: int = 240,
    global_chars: int = 400,
) -> str:
    """Construct a compact ASR evidence passage aligned with selected frames.

    Returns the empty string when no ASR is available for the video. The
    returned string is intended to be appended to the MCQ prompt as
    grounding context, *not* as new facts the model must trust blindly.
    """
    words = load_asr_words(video_path, cache_dir=cache_dir)
    if not words:
        return ""

    half = float(window_sec) / 2.0
    lines: list[str] = []
    for t in frame_timestamps_sec:
        active = _words_in_window(words, t - half, t + half)
        if not active:
            continue
        passage = _join_text(active, max_chars=per_frame_chars)
        if passage:
            lines.append(f"[{_format_ts(t)}] {passage}")

    summary = _join_text(words, max_chars=global_chars)

    out_parts: list[str] = []
    if lines:
        out_parts.append("Transcript near the selected frames:")
        out_parts.extend(f"- {ln}" for ln in lines)
    if summary:
        out_parts.append(f"Overall transcript (truncated): {summary}")

    return "\n".join(out_parts)


# ----------------------------------------------------------------------------
# OCR evidence text (per-frame, on-demand on the K_final selected frames only)
# ----------------------------------------------------------------------------

# Module-level singleton so we don't re-init the EasyOCR reader per item.
_OCR_READER = None
_OCR_BACKEND = None  # "easyocr" | "tesseract" | None


def _get_ocr_reader():
    """Lazily build a single OCR reader. Returns (reader, backend_name) or
    (None, None) when no backend is installed."""
    global _OCR_READER, _OCR_BACKEND
    if _OCR_READER is not None or _OCR_BACKEND == "tesseract":
        return _OCR_READER, _OCR_BACKEND
    # Try easyocr first (pure pip; GPU optional)
    try:
        import easyocr  # noqa
        _OCR_READER = easyocr.Reader(["en"], gpu=False, verbose=False)
        _OCR_BACKEND = "easyocr"
        return _OCR_READER, _OCR_BACKEND
    except Exception:
        pass
    try:
        import pytesseract  # noqa
        _OCR_BACKEND = "tesseract"
        return None, _OCR_BACKEND
    except Exception:
        return None, None


def _ocr_one_image(img_rgb, reader, backend, conf_threshold: float = 0.35) -> str:
    """Run OCR on a single RGB ndarray; return cleaned text or ''."""
    if img_rgb is None:
        return ""
    try:
        if backend == "easyocr":
            out = reader.readtext(img_rgb)
            toks = []
            for item in out:
                if len(item) < 3:
                    continue
                txt = str(item[1]).strip()
                try:
                    conf = float(item[2])
                except Exception:
                    conf = 0.0
                if txt and conf >= conf_threshold and len(txt) >= 2:
                    toks.append(txt)
            return " ".join(toks)
        if backend == "tesseract":
            import cv2, pytesseract
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            gray = cv2.equalizeHist(gray)
            data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
            toks = []
            for i in range(len(data.get("text", []))):
                t = str(data["text"][i]).strip()
                try:
                    c = float(data["conf"][i]) / 100.0
                except Exception:
                    c = 0.0
                if t and c >= conf_threshold and len(t) >= 2:
                    toks.append(t)
            return " ".join(toks)
    except Exception:
        return ""
    return ""


def _dedupe_keep_order(items):
    seen = set()
    out = []
    for x in items:
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


def build_ocr_evidence_text(
    selected_images: Sequence,
    selected_timestamps_sec: Sequence[float],
    per_frame_chars: int = 160,
    max_lines: int | None = None,
    conf_threshold: float = 0.35,
) -> str:
    """OCR each selected frame and return a compact 'On-screen text' block.

    ``selected_images`` is a sequence of RGB numpy arrays (already decoded by
    the harness). When no OCR backend is installed, or no readable text is
    found, returns the empty string.
    """
    if not selected_images:
        return ""
    reader, backend = _get_ocr_reader()
    if backend is None:
        return ""

    lines: list[str] = []
    for img, t in zip(selected_images, selected_timestamps_sec):
        text = _ocr_one_image(img, reader, backend, conf_threshold=conf_threshold)
        if not text:
            continue
        toks = _dedupe_keep_order(text.split())
        text = " ".join(toks)
        if len(text) > per_frame_chars:
            text = text[:per_frame_chars].rsplit(" ", 1)[0] + "..."
        lines.append(f"[{_format_ts(t)}] {text}")
        if max_lines is not None and len(lines) >= max_lines:
            break

    if not lines:
        return ""
    out = ["On-screen text detected in the selected frames:"]
    out += [f"- {ln}" for ln in lines]
    return "\n".join(out)


def build_combined_evidence_text(
    video_path: str,
    selected_images: Sequence,
    selected_timestamps_sec: Sequence[float],
    modalities: Sequence[str] = ("asr", "ocr"),
    cache_dir: str = DEFAULT_CACHE_DIR,
    asr_window_sec: float = 8.0,
) -> str:
    """Concatenate evidence blocks for multiple cue text streams.

    `modalities` is a subset of {"asr", "ocr"}. Empty modalities or empty
    individual blocks are silently skipped.
    """
    blocks: list[str] = []
    if "asr" in modalities:
        b = build_asr_evidence_text(
            video_path, selected_timestamps_sec,
            cache_dir=cache_dir, window_sec=asr_window_sec)
        if b:
            blocks.append(b)
    if "ocr" in modalities:
        b = build_ocr_evidence_text(
            selected_images, selected_timestamps_sec)
        if b:
            blocks.append(b)
    return "\n\n".join(blocks)
