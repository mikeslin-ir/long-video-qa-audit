"""CascadeRouter: executes a CascadePlan over a video.

Pipeline:
    L1 proposers -> union of frame anchors (capped at K_proposal)
    L2 filters   -> narrow to K_keyframe
    L3 aligners  -> per-frame relevance scores (weighted sum)
    submodular   -> pick K_final frames balancing relevance + coverage
"""
from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .plan import CascadePlan
from .planner import RuleBasedCascadePlanner
from .protocols import ProposalSet
from .registry import CascadeRegistry, default_registry
from submodular import submodular_select_facility_location


def _warn_skipped(stage: str, name: str, exc: Exception) -> None:
    """A planned tool that raises is skipped; say so (once per distinct error)
    instead of silently running a smaller pipeline than the plan describes."""
    warnings.warn(f"[cascade] {stage} {name!r} failed and was skipped: {exc!r}",
                  RuntimeWarning, stacklevel=3)


# ----------------------------------------------------------------------------
# Result type
# ----------------------------------------------------------------------------
@dataclass
class CascadeResult:
    plan: CascadePlan
    selected_frame_indices: list[int]
    selected_timestamps_sec: list[float]
    selected_image_paths: list[str]
    aligner_scores: dict[str, list[float]]
    fused_scores: list[float]
    n_proposals: int
    n_keyframes: int
    runtime_sec: dict[str, float]   # per-stage breakdown
    video_meta: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "plan": self.plan.to_json(),
            "selected_frames": [
                {"frame_index": int(i), "timestamp_sec": float(t), "image_path": p,
                 "fused_score": float(s)}
                for i, t, p, s in zip(self.selected_frame_indices,
                                      self.selected_timestamps_sec,
                                      self.selected_image_paths,
                                      self.fused_scores)
            ],
            "aligner_scores": {k: list(map(float, v)) for k, v in self.aligner_scores.items()},
            "n_proposals": self.n_proposals,
            "n_keyframes": self.n_keyframes,
            "runtime_sec": self.runtime_sec,
            "video_meta": self.video_meta,
        }


# ----------------------------------------------------------------------------
# Router
# ----------------------------------------------------------------------------
@dataclass
class CascadeRouter:
    registry: Optional[CascadeRegistry] = None
    planner: Any = None

    def __post_init__(self):
        if self.registry is None:
            self.registry = default_registry()
        if self.planner is None:
            self.planner = RuleBasedCascadePlanner()

    # ----------------------------------------------------------------- public
    def run(
        self,
        video_path: str,
        question: str,
        dataset: str,
        output_dir: str | Path,
        plan: Optional[CascadePlan] = None,
        K_final_override: Optional[int] = None,
    ) -> CascadeResult:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        t_start = time.time()
        rt: dict[str, float] = {}

        # Video meta (cheap)
        video_meta = self._video_meta(video_path)

        # 0. Plan
        if plan is None:
            t0 = time.time()
            plan = self.planner.plan(question, dataset, video_meta, registry=self.registry)
            rt["plan"] = time.time() - t0

        if K_final_override is not None:
            plan.K_final = int(K_final_override)

        # 1. L1: proposers (union of anchors, capped at K_proposal)
        t0 = time.time()
        proposal = self._run_proposers(plan, video_path, question, video_meta)
        rt["L1_propose"] = time.time() - t0

        # Decode the proposed frames once (used by L2 and L3)
        t0 = time.time()
        frames_rgb = self._decode_frames(video_path, proposal.frame_indices)
        rt["decode"] = time.time() - t0

        # 2. L2: filters
        t0 = time.time()
        keyframes, frames_rgb = self._run_filters(plan, proposal, frames_rgb,
                                                   question, video_meta)
        rt["L2_filter"] = time.time() - t0

        # 3. L3: aligners (weighted fusion)
        t0 = time.time()
        aligner_scores, fused = self._run_aligners(plan, keyframes, frames_rgb,
                                                   video_path, question, video_meta)
        rt["L3_align"] = time.time() - t0

        # 4. Select K_final, optionally reserving a uniform-anchor floor for
        #    guaranteed temporal coverage (Track A).
        t0 = time.time()
        K_final = int(plan.K_final)
        frac = float(getattr(plan, "uniform_anchor_frac", 0.0) or 0.0)
        frac = min(max(frac, 0.0), 1.0)
        n_anchor = int(round(frac * K_final)) if frac > 0.0 else 0
        n_anchor = min(n_anchor, K_final)
        n_cascade = K_final - n_anchor

        # 4a. Cascade picks (submodular over the fused relevance scores).
        cascade_local = submodular_select_facility_location(
            scores=fused,
            timestamps=keyframes.timestamps_sec,
            K=max(int(n_cascade), 0) if n_cascade > 0 else 0,
            tau_sec=float(plan.tau_sec),
        ) if n_cascade > 0 else []

        # 4b. Uniform anchors over the full video (independent of cascade scores).
        anchor_frames: list[tuple[int, float, np.ndarray]] = []
        if n_anchor > 0:
            total = int(video_meta.get("total_frames", 0) or 0)
            fps = float(video_meta.get("fps", 30.0) or 30.0)
            if total > 0:
                anchor_idx = np.linspace(0, total - 1, n_anchor).round().astype(int)
                anchor_idx = np.unique(anchor_idx)
                anchor_imgs = self._decode_frames(video_path, anchor_idx)
                for ai, img in zip(anchor_idx, anchor_imgs):
                    anchor_frames.append((int(ai), float(ai) / max(fps, 1e-6), img))
        rt["select"] = time.time() - t0

        # 5. Persist the chosen frames as JPEGs (mirrors SkillRouter output layout)
        sel_dir = out_dir / "selected_frames"
        sel_dir.mkdir(parents=True, exist_ok=True)
        sel_idx, sel_ts, sel_paths, sel_scores = [], [], [], []
        seen_fi: set[int] = set()

        def _write(fi: int, ts: float, img: np.ndarray, score: float):
            if fi in seen_fi:
                return
            seen_fi.add(fi)
            jp = sel_dir / f"frame_{fi:08d}.jpg"
            cv2.imwrite(str(jp), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            sel_idx.append(fi); sel_ts.append(ts); sel_paths.append(str(jp))
            sel_scores.append(float(score))

        for i in cascade_local:
            _write(int(keyframes.frame_indices[i]),
                   float(keyframes.timestamps_sec[i]),
                   frames_rgb[i], float(fused[i]))
        for fi, ts, img in anchor_frames:
            _write(fi, ts, img, 0.0)

        # Sort the final set by timestamp for a clean temporal order.
        if sel_idx:
            order = np.argsort(np.asarray(sel_ts, dtype=np.float64))
            sel_idx = [sel_idx[i] for i in order]
            sel_ts = [sel_ts[i] for i in order]
            sel_paths = [sel_paths[i] for i in order]
            sel_scores = [sel_scores[i] for i in order]

        rt["total"] = time.time() - t_start

        result = CascadeResult(
            plan=plan,
            selected_frame_indices=sel_idx,
            selected_timestamps_sec=sel_ts,
            selected_image_paths=sel_paths,
            aligner_scores={k: v.tolist() for k, v in aligner_scores.items()},
            fused_scores=fused.tolist(),
            n_proposals=len(proposal),
            n_keyframes=len(keyframes),
            runtime_sec=rt,
            video_meta=video_meta,
        )
        (out_dir / "cascade_result.json").write_text(json.dumps(result.to_json(), indent=2))
        return result

    # ---------------------------------------------------------------- helpers
    def _video_meta(self, video_path: str) -> dict[str, Any]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"fps": 30.0, "total_frames": 0, "duration_sec": 0.0}
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return {"fps": fps, "total_frames": total,
                "duration_sec": float(total) / max(fps, 1e-6)}

    def _run_proposers(self, plan: CascadePlan, video_path: str, question: str,
                       meta: dict) -> ProposalSet:
        union_idx: list[int] = []
        union_ts: list[float] = []
        per_proposer = max(8, plan.K_proposal // max(1, len(plan.proposers)))
        for name in plan.proposers:
            tool = self.registry.proposers.get(name)
            if tool is None:
                continue
            try:
                ps = tool.propose(video_path, question, meta, per_proposer)
            except Exception as exc:
                _warn_skipped("proposer", name, exc)
                continue
            union_idx.extend(int(x) for x in ps.frame_indices)
            union_ts.extend(float(x) for x in ps.timestamps_sec)

        if not union_idx:
            # Fallback: uniform stride
            from .proposers.uniform import UniformStrideProposer
            ps = UniformStrideProposer().propose(video_path, question, meta, plan.K_proposal)
            return ps

        # Dedup + sort + cap at K_proposal
        idx_arr = np.array(union_idx, dtype=np.int64)
        ts_arr = np.array(union_ts, dtype=np.float64)
        order = np.argsort(idx_arr)
        idx_arr = idx_arr[order]
        ts_arr = ts_arr[order]
        # unique on frame_index (within +/-1 to absorb rounding)
        keep = [0]
        for j in range(1, len(idx_arr)):
            if idx_arr[j] - idx_arr[keep[-1]] >= 1:
                keep.append(j)
        idx_arr = idx_arr[keep]
        ts_arr = ts_arr[keep]
        if len(idx_arr) > plan.K_proposal:
            # Subsample evenly to keep temporal coverage
            sel = np.linspace(0, len(idx_arr) - 1, plan.K_proposal).round().astype(int)
            idx_arr = idx_arr[sel]
            ts_arr = ts_arr[sel]
        return ProposalSet(idx_arr, ts_arr, source="+".join(plan.proposers))

    def _decode_frames(self, video_path: str, indices: np.ndarray) -> list[np.ndarray]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []
        frames = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, bgr = cap.read()
            if not ok or bgr is None:
                # placeholder so list length matches indices
                frames.append(np.zeros((1, 1, 3), dtype=np.uint8))
                continue
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    def _run_filters(self, plan: CascadePlan, proposal: ProposalSet,
                     frames_rgb: list[np.ndarray], question: str, meta: dict
                     ) -> tuple[ProposalSet, list[np.ndarray]]:
        current = proposal
        current_frames = frames_rgb
        for name in plan.filters:
            tool = self.registry.filters.get(name)
            if tool is None:
                continue
            try:
                filtered = tool.filter(current, current_frames, question, meta, plan.K_keyframe)
            except Exception as exc:
                _warn_skipped("filter", name, exc)
                continue
            # Map kept frames back to the new list of decoded images
            kept_indices_set = set(int(x) for x in filtered.frame_indices)
            new_frames = []
            for i, fi in enumerate(current.frame_indices):
                if int(fi) in kept_indices_set:
                    new_frames.append(current_frames[i])
            current = filtered
            current_frames = new_frames

        # If no filter ran (or all failed), just truncate to K_keyframe by uniform sampling
        if len(current) > plan.K_keyframe:
            sel = np.linspace(0, len(current) - 1, plan.K_keyframe).round().astype(int)
            current = ProposalSet(
                frame_indices=current.frame_indices[sel],
                timestamps_sec=current.timestamps_sec[sel],
                scores=current.scores[sel] if current.scores is not None else None,
                source=current.source,
            )
            current_frames = [current_frames[i] for i in sel]
        return current, current_frames

    def _run_aligners(self, plan: CascadePlan, keyframes: ProposalSet,
                      frames_rgb: list[np.ndarray], video_path: str,
                      question: str, meta: dict
                      ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        scores: dict[str, np.ndarray] = {}
        meta_with_prompts = dict(meta)
        meta_with_prompts["evidence_prompts"] = list(plan.evidence_prompts)
        for name in plan.aligners:
            tool = self.registry.aligners.get(name)
            if tool is None:
                continue
            try:
                s = tool.align(keyframes, frames_rgb, video_path, question, meta_with_prompts)
            except Exception as exc:
                _warn_skipped("aligner", name, exc)
                continue
            scores[name] = np.asarray(s, dtype=np.float64)

        if not scores:
            return {}, np.ones(len(keyframes), dtype=np.float64)

        # min-max normalize then weighted sum
        weights = {k: float(plan.weights.get(k, 0.0)) for k in scores.keys()}
        sw = sum(max(v, 0.0) for v in weights.values())
        if sw <= 0:
            weights = {k: 1.0 / len(scores) for k in scores.keys()}
        else:
            weights = {k: max(v, 0.0) / sw for k, v in weights.items()}

        fused = np.zeros(len(keyframes), dtype=np.float64)
        for name, s in scores.items():
            lo, hi = float(np.min(s)), float(np.max(s))
            sn = (s - lo) / max(hi - lo, 1e-9) if hi > lo else np.zeros_like(s)
            fused += weights[name] * sn
        return scores, fused
