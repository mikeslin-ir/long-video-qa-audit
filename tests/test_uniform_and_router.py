from __future__ import annotations

import warnings

import cv2
import numpy as np
import pytest

from baselines.selectors import _decode_uniform_frames, make
from cascade.plan import CascadePlan
from cascade.protocols import ProposalSet
from cascade.registry import CascadeRegistry
from cascade.router import CascadeRouter


def _video(path, frames=100):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 32))
    for i in range(frames):
        writer.write(np.full((32, 32, 3), i * 2, dtype=np.uint8))
    writer.release()
    return str(path)


def test_endpoint_inclusive_uniform_includes_first_and_last_frame(tmp_path):
    path = _video(tmp_path / "v.mp4")
    _, _, idx8 = _decode_uniform_frames(path, 8)
    assert idx8[0] == 0 and idx8[-1] >= 98
    _, _, idx1 = _decode_uniform_frames(path, 1)
    assert idx1 == [0]  # the single frame at K=1 is the first frame


def test_centered_uniform_takes_segment_midpoints(tmp_path):
    path = _video(tmp_path / "v.mp4")
    _, _, idx = _decode_uniform_frames(path, 4, centered=True)
    assert idx[0] >= 10 and idx[-1] <= 90 and len(idx) == 4
    assert np.allclose(np.diff(idx), 25, atol=1)
    _, _, idx1 = _decode_uniform_frames(path, 1, centered=True)
    assert 45 <= idx1[0] <= 55  # the middle frame


def test_uniform_center_selector_is_registered():
    assert make("uniform_center").centered and not make("uniform").centered
    assert make("uniform_center").name == "uniform_center"


class _Raises:
    name = "boom"

    def propose(self, *args, **kwargs):
        raise RuntimeError("PySceneDetect unavailable")


class _Fixed:
    name = "fixed"

    def propose(self, video_path, question, meta, K):
        idx = np.arange(0, 100, 10)
        return ProposalSet(idx, idx / 10.0, source=self.name)


def test_router_warns_when_a_planned_proposer_fails():
    reg = CascadeRegistry()
    reg.register_proposer("boom", _Raises())
    reg.register_proposer("fixed", _Fixed())
    router = CascadeRouter(registry=reg, planner=object())
    plan = CascadePlan(proposers=["boom", "fixed"], K_proposal=20)
    with pytest.warns(RuntimeWarning, match="proposer 'boom' failed and was skipped"):
        pool = router._run_proposers(plan, "unused.mp4", "q", {})
    assert len(pool) == 10  # the surviving proposer still supplies its candidates


def test_router_is_silent_when_every_tool_works():
    reg = CascadeRegistry()
    reg.register_proposer("fixed", _Fixed())
    router = CascadeRouter(registry=reg, planner=object())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        router._run_proposers(CascadePlan(proposers=["fixed"], K_proposal=20), "unused.mp4", "q", {})
