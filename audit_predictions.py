"""Per-item outcome access for the audit.

Every statistic in the paper is a function of per-item correctness. The raw run
outputs (tens of GB of frames, caches and logs) are not released; their per-item
outcomes are packaged in ``results/predictions/`` and this module reads either
source transparently:

1. ``outputs/<run>/`` if it exists (the authors' working tree), else
2. the packaged ``results/predictions/predictions.csv.gz``.

"Correct" is defined in exactly one place, :func:`live_correct`: the LLM-judge
verdict (``judged_rows.jsonl``) for judged open-ended runs such as MUSIC-AVQA,
and the in-loop exact-match score (``rows.jsonl``) otherwise.

Set ``LONG_VIDEO_QA_OUTPUTS`` to point at a different ``outputs`` directory.
"""
from __future__ import annotations

import csv
import gzip
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parent
PRED_DIR = ROOT / "results" / "predictions"
PACKAGED = PRED_DIR / "predictions.csv.gz"
MANIFEST = PRED_DIR / "manifest.csv"


def outputs_dir() -> Path:
    return Path(os.environ.get("LONG_VIDEO_QA_OUTPUTS", ROOT / "outputs"))


def _jsonl(path: Path) -> Iterator[dict]:
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def live_correct(run: str, outputs: Path | None = None) -> dict[str, int] | None:
    """qid -> 0/1 from a working-tree run directory, or None if it is absent."""
    base = (outputs or outputs_dir()) / run
    judged, rows = base / "judged_rows.jsonl", base / "rows.jsonl"
    if judged.exists():
        return {str(r["qid"]): int(bool(r["judge_yes"])) for r in _jsonl(judged)}
    if rows.exists():
        return {str(r["qid"]): int(bool(r["correct"])) for r in _jsonl(rows)}
    return None


@lru_cache(maxsize=1)
def _packaged() -> dict[str, dict[str, int]]:
    table: dict[str, dict[str, int]] = {}
    if not PACKAGED.exists():
        return table
    with gzip.open(PACKAGED, "rt", newline="") as fh:
        for row in csv.DictReader(fh):
            table.setdefault(row["run"], {})[row["qid"]] = int(row["correct"])
    return table


@lru_cache(maxsize=1)
def _packaged_types() -> dict[str, dict[str, str]]:
    table: dict[str, dict[str, str]] = {}
    if not PACKAGED.exists():
        return table
    with gzip.open(PACKAGED, "rt", newline="") as fh:
        for row in csv.DictReader(fh):
            table.setdefault(row["run"], {})[row["qid"]] = row.get("task_type", "")
    return table


def available_runs() -> list[str]:
    return sorted(_packaged())


def load_task_types(run: str) -> dict[str, str]:
    """qid -> task type of ``run`` (for example the MUSIC-AVQA question type)."""
    rows = outputs_dir() / run / "rows.jsonl"
    if rows.exists():
        return {str(r["qid"]): str(r.get("task_type", "")) for r in _jsonl(rows)}
    packaged = _packaged_types().get(run)
    if packaged is None:
        raise FileNotFoundError(f"run {run!r} is neither in {outputs_dir()} nor in {PACKAGED}")
    return dict(packaged)


def load_correct(run: str) -> dict[str, int]:
    """qid -> 0/1 correctness of ``run`` from the working tree or the package."""
    live = live_correct(run)
    if live is not None:
        return live
    packaged = _packaged().get(run)
    if packaged is None:
        raise FileNotFoundError(
            f"run {run!r} is neither in {outputs_dir()} nor in {PACKAGED}")
    return dict(packaged)


def manifest() -> list[dict[str, str]]:
    if not MANIFEST.exists():
        return []
    with open(MANIFEST, newline="") as fh:
        return list(csv.DictReader(fh))
