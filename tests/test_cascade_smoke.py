"""Model-free cascade integration test on a synthetic video."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade import (
    CascadeRouter,
    RuleBasedCascadePlanner,
)
from cascade.filters.redundancy_filter import RedundancyFilter
from cascade.proposers.uniform import UniformStrideProposer
from cascade.protocols import ProposalSet
from cascade.registry import CascadeRegistry


class _BrightnessAligner:
    name = "test_brightness"
    cost_per_frame = 0.0
    modality = "vision"

    def align(
        self,
        keyframes: ProposalSet,
        frames_rgb: list[np.ndarray],
        video_path: str,
        question: str | None,
        meta: dict,
    ) -> np.ndarray:
        return np.array([frame.mean() for frame in frames_rgb], dtype=np.float64)


def _make_synthetic_video(path: Path, n_frames: int = 60, fps: int = 10,
                          w: int = 64, h: int = 64) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    assert vw.isOpened(), "failed to open synthetic-video writer"
    for t in range(n_frames):
        img = np.full((h, w, 3), t * 4 % 255, dtype=np.uint8)
        # add a moving square to give the motion filter something to chew
        x = (t * 3) % (w - 8)
        img[8:16, x:x + 8] = 255
        vw.write(img)
    vw.release()


def _run_smoke(td: Path) -> None:
    vp = td / "tiny.mp4"
    _make_synthetic_video(vp)
    assert vp.exists(), "failed to write synthetic video"

    reg = CascadeRegistry()
    reg.register_proposer("uniform_stride", UniformStrideProposer())
    reg.register_filter("redundancy", RedundancyFilter())
    reg.register_aligner("test_brightness", _BrightnessAligner())

    planner = RuleBasedCascadePlanner(K_proposal_long=32, K_proposal_short=16,
                                      K_keyframe=8, K_final=4)
    plan = planner.plan(
        question="What color is the moving square?",
        dataset="general",
        meta={"duration_sec": 6.0},
        registry=reg,
    )
    assert plan.proposers, "no proposers chosen"
    plan.aligners = ["test_brightness"]
    plan.weights = {"test_brightness": 1.0}

    router = CascadeRouter(registry=reg, planner=planner)
    out_dir = td / "out"
    res = router.run(
        video_path=str(vp),
        question="What color is the moving square?",
        dataset="general",
        output_dir=str(out_dir),
        plan=plan,
    )
    assert len(res.selected_frame_indices) == plan.K_final
    assert len(res.aligner_scores["test_brightness"]) == res.n_keyframes
    assert (out_dir / "cascade_result.json").exists()
    for path in res.selected_image_paths:
        assert Path(path).exists(), f"missing selected frame {path}"


def test_cascade_runs_without_models(tmp_path):
    _run_smoke(tmp_path)


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        _run_smoke(Path(td))
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
