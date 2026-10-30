"""Dataset-root configuration shared by the public entry points."""
from __future__ import annotations

import os
from pathlib import Path


def get_cascade_cache_dir() -> str:
    """Cache for proposer outputs, ASR transcripts and OCR (``LONG_VIDEO_QA_CASCADE_CACHE``)."""
    value = os.environ.get("LONG_VIDEO_QA_CASCADE_CACHE", "")
    return value.strip() or "results/cascade_cache"


def get_data_root() -> Path:
    value = os.environ.get("LONG_VIDEO_QA_DATA_ROOT")
    if value is None:
        return Path.home() / "project" / "datasets"
    if not value.strip():
        raise ValueError("LONG_VIDEO_QA_DATA_ROOT must not be empty")
    return Path(value).expanduser().resolve()
