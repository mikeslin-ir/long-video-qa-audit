from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from data_paths import get_data_root


def test_default_data_root_preserves_reference_layout(monkeypatch):
    monkeypatch.delenv("LONG_VIDEO_QA_DATA_ROOT", raising=False)
    assert get_data_root() == Path.home() / "project" / "datasets"


def test_data_root_override(monkeypatch, tmp_path):
    monkeypatch.setenv("LONG_VIDEO_QA_DATA_ROOT", str(tmp_path))
    assert get_data_root() == tmp_path


def test_data_root_expands_home(monkeypatch):
    monkeypatch.setenv("LONG_VIDEO_QA_DATA_ROOT", "~/video-qa-data")
    assert get_data_root() == Path.home() / "video-qa-data"


@pytest.mark.parametrize("value", ["", "  "])
def test_empty_data_root_is_an_error(monkeypatch, value):
    monkeypatch.setenv("LONG_VIDEO_QA_DATA_ROOT", value)
    with pytest.raises(ValueError, match="must not be empty"):
        get_data_root()


def test_entry_points_share_configured_root(tmp_path):
    root = Path(__file__).resolve().parents[1]
    code = """
from pathlib import Path
from data_paths import get_data_root
from router.qa_datasets import DATA, _AR_ROOT
from evidence_skills.clap_skill import _DEFAULT_FEATURE_ROOTS
from scripts.run_full_benchmark import DATASET_REGISTRY

root = get_data_root()
assert DATA == root
assert _AR_ROOT == root / "agentic_routing"
assert _DEFAULT_FEATURE_ROOTS == [str(root / "agentic_routing" / "features")]
assert Path(DATASET_REGISTRY["video_mme"]["root"]) == root / "videomme"
assert all(Path(row["root"]).is_relative_to(root) for row in DATASET_REGISTRY.values())
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env={**os.environ, "LONG_VIDEO_QA_DATA_ROOT": str(tmp_path)},
        check=True,
    )
