"""Baseline frame selectors used for comparison against the SkillRouter.

All selectors share a uniform interface:

    select(video_path, question, dataset, budget, tau_sec, output_dir, **kw)
        -> dict   (with at least 'selected_frames' and 'selection_method')

Implemented natively here:
    - UniformSelector
    - RandomSelector            (for sanity)
    - MotionOnlySelector        (motion energy peaks + temporal NMS)
    - CLIPOnlySelector          (semantic CLIP / SigLIP top-K + temporal NMS)
    - PySceneDetectSelector     (shot-boundary + budget K)

External baselines are exposed as thin wrappers; they expect the
relevant third-party repo to be available on PYTHONPATH.  If not, the
wrapper raises a clear RuntimeError so the benchmark script can record
the missing dependency and continue.

    - SeViLASelector            (Yu et al., NeurIPS 2023)
    - VideoAgentSelector        (Wang et al., ECCV 2024)

For both external baselines we provide a CLI hook:
    python -m baselines.run --method sevila ...
"""
from __future__ import annotations

import heapq
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# ---------- helpers --------------------------------------------------------


def _save_frames(frames_rgb: list[np.ndarray], out_dir: Path,
                 timestamps: list[float], frame_indices: list[int]) -> list[dict]:
    sel_dir = out_dir / "selected_frames"
    sel_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for img, ts, fi in zip(frames_rgb, timestamps, frame_indices):
        img_name = f"frame_{fi:08d}.jpg"
        path = sel_dir / img_name
        if img is not None:
            cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        rows.append({
            "frame_index": int(fi),
            "timestamp_sec": float(ts),
            "image_path": str(path),
        })
    return rows


def _decode_uniform_frames(video_path: str, K: int, centered: bool = False
                           ) -> tuple[list[np.ndarray], list[float], list[int]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if n <= 0:
        cap.release()
        raise RuntimeError(f"no frames decoded from {video_path}")

    if K >= n:
        idxs = list(range(n))
    elif centered:
        # midpoint of each of K equal segments
        idxs = [min(n - 1, int((i + 0.5) * n / K)) for i in range(K)]
    else:
        # endpoint-inclusive spacing: includes the first and last frame, and for
        # K=1 only the first frame
        idxs = [int(round(i * (n - 1) / max(K - 1, 1))) for i in range(K)]
    frames, ts = [], []
    last = -1
    for i in idxs:
        if i != last:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, bgr = cap.read()
        last = i + 1
        if not ok or bgr is None:
            continue
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        ts.append(i / float(fps))
    cap.release()
    return frames, ts, idxs[: len(frames)]


# ---------- native baselines ----------------------------------------------


@dataclass
class UniformSelector:
    name: str = "uniform"
    centered: bool = False

    def select(self, *, video_path: str, question: str, dataset: str,
               budget: int, tau_sec: float, output_dir: str,
               **kw) -> dict[str, Any]:
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        frames, ts, idxs = _decode_uniform_frames(video_path, K=int(budget), centered=self.centered)
        rows = _save_frames(frames, out, ts, idxs)
        res = {
            "video_id": Path(video_path).stem,
            "question": question, "dataset": dataset,
            "selection_method": self.name,
            "selected_frames": rows,
            "budget": int(budget),
            "runtime_sec": time.time() - t0,
        }
        (out / "result.json").write_text(json.dumps(res, indent=2))
        return res


@dataclass
class UniformCenteredSelector(UniformSelector):
    """Uniform sampling at segment midpoints (a stronger K=1 / small-K baseline than
    the endpoint-inclusive ``uniform``, which spends frames on the first and last
    frame of the video)."""
    name: str = "uniform_center"
    centered: bool = True


@dataclass
class RandomSelector:
    name: str = "random"
    seed: int = 0

    def select(self, *, video_path: str, question: str, dataset: str,
               budget: int, tau_sec: float, output_dir: str,
               **kw) -> dict[str, Any]:
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        cap = cv2.VideoCapture(video_path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        rng = random.Random(self.seed)
        idxs = sorted(rng.sample(range(n), k=min(int(budget), n))) if n > 0 else []
        frames, ts = [], []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, bgr = cap.read()
            if not ok or bgr is None:
                continue
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            ts.append(i / float(fps))
        cap.release()
        rows = _save_frames(frames, out, ts, idxs[: len(frames)])
        res = {
            "video_id": Path(video_path).stem,
            "question": question, "dataset": dataset,
            "selection_method": self.name,
            "selected_frames": rows,
            "budget": int(budget),
            "runtime_sec": time.time() - t0,
        }
        (out / "result.json").write_text(json.dumps(res, indent=2))
        return res


def _routed_score_only(skill_name: str):
    """Return a selector that uses ONLY the given skill's score curve."""

    @dataclass
    class _OneSkill:
        name: str = f"only_{skill_name}"

        def select(self, *, video_path: str, question: str, dataset: str,
                   budget: int, tau_sec: float, output_dir: str,
                   **kw) -> dict[str, Any]:
            # Lazy import to avoid pulling skill deps when not needed
            from skill_router import SkillRouter
            cfg = dict(kw.get("config") or {})
            # Bound the candidate pool so single-skill scoring stays tractable on
            # long videos (otherwise stride=5 on a 30-60 min clip yields tens of
            # thousands of frames). 128/stride-10 keeps motion/clip scoring cheap
            # for the baseline comparison; can be overridden via config.
            cfg.setdefault("max_candidates", 128)
            cfg.setdefault("candidate_stride", 10)
            router = SkillRouter(cfg)
            return router.run(
                video_path=video_path, question=question, dataset=dataset,
                budget=budget, tau_sec=tau_sec, output_dir=output_dir,
                enabled_skills=[skill_name],
                use_submodular=kw.get("use_submodular", False),
            )
    return _OneSkill()


def MotionOnlySelector():
    return _routed_score_only("motion")


def CLIPOnlySelector():
    return _routed_score_only("semantic_siglip")


# ---------- AKS (Tang et al., CVPR 2025) -- native re-implementation -------
#
# AKS contributes a *selection rule*: given a per-frame prompt-relevance
# score curve, it recursively bisects the timeline and allocates the frame
# budget to segments whose top scores stand out from their local mean
# (high "peakiness"), giving more frames to relevant-and-concentrated
# regions and falling back to uniform-like coverage elsewhere.  This is a
# faithful port of the public `frame_select.py::meanstd` recursion
# (https://github.com/ncTimTang/AKS).  We feed it the SAME SigLIP relevance
# curve and SAME uniform candidate pool the cascade uses, so the only
# difference vs. our selector is the selection algorithm (AKS adaptive
# bisection vs. submodular facility-location); the relevance features are
# matched.  Defaults t1=0.8, all_depth=5 follow the AKS paper.


def _aks_meanstd(segments: list[dict], n: int, t1: float, t2: float,
                 all_depth: int) -> list[dict]:
    """Recursive bisection. Each segment is {'score': np.ndarray,
    'idx': np.ndarray (global frame positions), 'depth': int}.
    Returns a flat list of terminal segments."""
    keep, split = [], []
    for seg in segments:
        score = seg["score"]
        depth = seg["depth"]
        if len(score) == 0:
            continue
        mean = float(np.mean(score))
        std = float(np.std(score))
        top_n = heapq.nlargest(min(n, len(score)), range(len(score)),
                               score.__getitem__)
        mean_diff = float(np.mean([score[t] for t in top_n])) - mean
        if mean_diff > t1 and std > t2:
            keep.append(seg)                       # peaky enough: stop here
        elif depth < all_depth and len(score) >= 2:
            h = len(score) // 2
            split.append(dict(score=score[:h], idx=seg["idx"][:h],
                              depth=depth + 1))
            split.append(dict(score=score[h:], idx=seg["idx"][h:],
                              depth=depth + 1))
        else:
            keep.append(seg)
    if split:
        keep = keep + _aks_meanstd(split, n, t1, t2, all_depth)
    return keep


def aks_select(scores: np.ndarray, K: int, t1: float = 0.8, t2: float = -100.0,
               all_depth: int = 5) -> list[int]:
    """Return up to K selected indices into `scores` using the AKS rule."""
    m = len(scores)
    if m <= K:
        return list(range(m))
    s = np.asarray(scores, dtype=np.float64)
    rng = s.max() - s.min()
    norm = (s - s.min()) / rng if rng > 0 else np.zeros_like(s)
    terminals = _aks_meanstd(
        [dict(score=norm, idx=np.arange(m), depth=0)], K, t1, t2, all_depth)
    picked: list[int] = []
    for seg in terminals:
        f_num = max(1, int(K / (2 ** seg["depth"])))
        topk = heapq.nlargest(min(f_num, len(seg["score"])),
                              range(len(seg["score"])), seg["score"].__getitem__)
        picked.extend(int(seg["idx"][t]) for t in topk)
    picked = sorted(set(picked))
    if len(picked) > K:                            # trim to budget by top score
        picked = sorted(picked, key=lambda i: -s[i])[:K]
        picked.sort()
    elif len(picked) < K:                          # pad with uniform fill
        fill = [int(round(i * (m - 1) / max(K - 1, 1))) for i in range(K)]
        for f in fill:
            if f not in picked:
                picked.append(f)
            if len(picked) >= K:
                break
        picked = sorted(set(picked))[:K]
    return picked


@dataclass
class AKSSelector:
    """Adaptive Keyframe Sampling (Tang et al., CVPR 2025), native port.

    Matched-feature reproduction: SigLIP prompt-relevance over the cascade's
    uniform candidate pool, then the AKS adaptive-bisection selection rule.
    """
    name: str = "aks"

    def select(self, *, video_path: str, question: str, dataset: str,
               budget: int, tau_sec: float, output_dir: str,
               **kw) -> dict[str, Any]:
        from evidence_skills import SemanticSigLIPSkill, sample_candidate_frames
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        cfg = dict(kw.get("config") or {})
        max_cand = int(cfg.get("max_candidates", 256))
        stride = _full_span_stride(video_path, max_cand,
                                   floor=int(cfg.get("candidate_stride", 5)))
        cache_dir = cfg.get("cache_dir", "results/skill_router_cache")
        frames, meta = sample_candidate_frames(video_path, stride=stride,
                                               max_frames=max_cand)
        if not frames:
            raise RuntimeError(f"no candidate frames for {video_path}")
        skill = SemanticSigLIPSkill(
            cache_dir=cache_dir,
            model_id=cfg.get("semantic_siglip_model",
                             "google/siglip-base-patch16-224"))
        scores = np.asarray(skill.compute(
            video_path=video_path, frames=frames, question=question,
            metadata={"dataset": dataset, **meta}), dtype=np.float32)
        picked = aks_select(scores, K=int(budget),
                            t1=float(cfg.get("aks_t1", 0.8)),
                            all_depth=int(cfg.get("aks_all_depth", 5)))
        frames_rgb = [frames[i].image for i in picked]
        ts = [float(frames[i].timestamp_sec) for i in picked]
        fidx = [int(frames[i].frame_index) for i in picked]
        rows = _save_frames(frames_rgb, out, ts, fidx)
        for r, i in zip(rows, picked):
            r["relevance_score"] = float(scores[i])
        res = {
            "video_id": Path(video_path).stem, "question": question,
            "dataset": dataset, "selection_method": self.name,
            "selected_frames": rows, "budget": int(budget),
            "num_candidates": len(frames),
            "runtime_sec": time.time() - t0,
        }
        (out / "result.json").write_text(json.dumps(res, indent=2))
        return res


# ---------- shared matched-feature front-end ------------------------------
#
# KeyScore and VideoTree below are ported the same way AKS is: they consume
# the SAME SigLIP candidate pool and the SAME embeddings the cascade uses,
# and differ from our selector only in the selection rule.


def _full_span_stride(video_path: str, max_candidates: int, floor: int = 5) -> int:
    """Stride that makes `max_candidates` frames cover the entire video."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    cap.release()
    if total <= 0 or max_candidates <= 0:
        return max(1, floor)
    return max(floor, math.ceil(total / max_candidates))


def _siglip_pool(video_path: str, question: str, dataset: str, cfg: dict):
    """Candidate frames + cached SigLIP (image_emb, text_emb) for this item.

    The stride is derived from video length so the pool spans the WHOLE video:
    `sample_candidate_frames` truncates with `indices[:max_frames]`, so passing a
    small fixed stride would confine every baseline to the opening seconds.
    """
    from evidence_skills import SemanticSigLIPSkill, sample_candidate_frames
    max_cand = int(cfg.get("max_candidates", 256))
    stride = _full_span_stride(video_path, max_cand,
                              floor=int(cfg.get("candidate_stride", 5)))
    frames, meta = sample_candidate_frames(video_path, stride=stride,
                                           max_frames=max_cand)
    if not frames:
        raise RuntimeError(f"no candidate frames for {video_path}")
    skill = SemanticSigLIPSkill(
        cache_dir=cfg.get("cache_dir", "results/skill_router_cache"),
        model_id=cfg.get("semantic_siglip_model", "google/siglip-base-patch16-224"))
    pair = skill.embed(video_path=video_path, frames=frames, question=question,
                       metadata={"dataset": dataset, **meta})
    if pair is None:
        raise RuntimeError("SigLIP unavailable; cannot run matched-feature baseline")
    image_emb, text_emb = pair
    return frames, np.asarray(image_emb, np.float32), np.asarray(text_emb, np.float32)


def _emit(self_name: str, frames, picked: list[int], scores, video_path: str,
          question: str, dataset: str, budget: int, output_dir: str, t0: float):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    rows = _save_frames([frames[i].image for i in picked], out,
                        [float(frames[i].timestamp_sec) for i in picked],
                        [int(frames[i].frame_index) for i in picked])
    if scores is not None:
        for r, i in zip(rows, picked):
            r["relevance_score"] = float(scores[i])
    res = {
        "video_id": Path(video_path).stem, "question": question,
        "dataset": dataset, "selection_method": self_name,
        "selected_frames": rows, "budget": int(budget),
        "num_candidates": len(frames), "runtime_sec": time.time() - t0,
    }
    (out / "result.json").write_text(json.dumps(res, indent=2))
    return res


# ---------- KeyScore (Lin et al., 2025) -- native re-implementation --------
#
# KeyScore contributes a *closed-form frame score* combining three signals
# over a shared embedding space: (1) semantic text-frame similarity,
# (2) temporal representativeness (frame vs. pooled video embedding), and
# (3) a leave-one-out "contextual drop": how much the text-video similarity
# falls when frame i is removed.  Each component is min-max normalised and
# mixed as alpha*semantic + beta*temporal + gamma*drop, then the top-K
# frames are kept in temporal order.  This is a faithful port of the public
# `src/scoring/FrameScorer.py` with the authors' default weights
# (alpha=0.85, beta=0.05, gamma=0.1) under mean pooling, for which the
# leave-one-out term has a closed form.


def keyscore_scores(image_emb: np.ndarray, text_emb: np.ndarray,
                    alpha: float = 0.85, beta: float = 0.05,
                    gamma: float = 0.1) -> np.ndarray:
    def _l2(x, axis=-1):
        n = np.linalg.norm(x, axis=axis, keepdims=True)
        return x / np.clip(n, 1e-8, None)

    def _minmax(x):
        lo, hi = float(x.min()), float(x.max())
        return (x - lo) / (hi - lo) if hi - lo > 1e-8 else np.zeros_like(x)

    f = _l2(np.asarray(image_emb, np.float64))          # [N, D]
    t = _l2(np.asarray(text_emb, np.float64))           # [P, D]
    t = _l2(t.mean(axis=0)) if t.ndim == 2 else _l2(t)  # [D]
    n = f.shape[0]

    v = _l2(f.mean(axis=0))                             # pooled video embedding
    sim_semantic = f @ t
    sim_temporal = f @ v
    if n <= 1:
        sim_drop = np.zeros(n)
    else:
        v_loo = _l2((f.sum(axis=0)[None, :] - f) / (n - 1))
        sim_drop = float(v @ t) - (v_loo @ t)

    return (alpha * _minmax(sim_semantic)
            + beta * _minmax(sim_temporal)
            + gamma * _minmax(sim_drop)).astype(np.float32)


@dataclass
class KeyScoreSelector:
    """KeyScore (Lin et al., 2025), matched-feature native port."""
    name: str = "keyscore"

    def select(self, *, video_path: str, question: str, dataset: str,
               budget: int, tau_sec: float, output_dir: str,
               **kw) -> dict[str, Any]:
        t0 = time.time()
        cfg = dict(kw.get("config") or {})
        frames, image_emb, text_emb = _siglip_pool(video_path, question, dataset, cfg)
        scores = keyscore_scores(image_emb, text_emb,
                                 alpha=float(cfg.get("keyscore_alpha", 0.85)),
                                 beta=float(cfg.get("keyscore_beta", 0.05)),
                                 gamma=float(cfg.get("keyscore_gamma", 0.10)))
        K = min(int(budget), len(frames))
        picked = sorted(np.argsort(-scores)[:K].tolist())
        return _emit(self.name, frames, picked, scores, video_path, question,
                     dataset, budget, output_dir, t0)


# ---------- VideoTree (Wang et al., CVPR 2025) -- selection rule only ------
#
# VideoTree builds an adaptive breadth/depth tree over clustered frame
# embeddings and expands query-relevant branches.  We port only its
# *selection rule*: k-means over the same SigLIP candidate embeddings,
# clusters ranked by peak query relevance, budget allocated breadth-first
# to the most relevant clusters, and the highest-relevance frame taken as
# each cluster's representative.  Its caption/LLM reasoning stage is NOT
# ported -- that would change the reasoner and confound the comparison,
# which is exactly the entanglement this audit removes.  Reported as a
# selection-rule control, not as a reproduction of the full system.


def videotree_select(image_emb: np.ndarray, rel: np.ndarray, K: int,
                     seed: int = 0) -> list[int]:
    from sklearn.cluster import KMeans

    n = len(rel)
    if n <= K:
        return list(range(n))
    n_clusters = min(max(K, 2), n)
    labels = KMeans(n_clusters=n_clusters, n_init=4, random_state=seed
                    ).fit_predict(np.asarray(image_emb, np.float64))
    order = sorted(range(n_clusters),
                   key=lambda c: -(rel[labels == c].max() if np.any(labels == c) else -np.inf))
    picked: list[int] = []
    for c in order:                     # breadth-first over relevance-ranked clusters
        members = np.flatnonzero(labels == c)
        if members.size:
            picked.append(int(members[int(np.argmax(rel[members]))]))
        if len(picked) >= K:
            break
    for i in np.argsort(-rel):          # depth fill if clusters ran out
        if len(picked) >= K:
            break
        if int(i) not in picked:
            picked.append(int(i))
    return sorted(picked[:K])


@dataclass
class VideoTreeSelector:
    """VideoTree-style adaptive clustering, selection rule only."""
    name: str = "videotree"

    def select(self, *, video_path: str, question: str, dataset: str,
               budget: int, tau_sec: float, output_dir: str,
               **kw) -> dict[str, Any]:
        t0 = time.time()
        cfg = dict(kw.get("config") or {})
        frames, image_emb, text_emb = _siglip_pool(video_path, question, dataset, cfg)
        rel = (image_emb @ text_emb.T).max(axis=1).astype(np.float32)
        picked = videotree_select(image_emb, rel, K=min(int(budget), len(frames)))
        return _emit(self.name, frames, picked, rel, video_path, question,
                     dataset, budget, output_dir, t0)


# ---------- external baselines (thin wrappers) ----------------------------



@dataclass
class SeViLASelector:
    """Wrapper around the SeViLA repo (Yu et al., NeurIPS 2023).

    Expected layout:
        $SEVILA_REPO/run_scripts/SeViLA/inference/select.py
    or any module that exposes a `select_frames(video_path, question, K)`
    callable.  Override SEVILA_ENTRY env var to point to it.
    """
    name: str = "sevila"

    def select(self, *, video_path: str, question: str, dataset: str,
               budget: int, tau_sec: float, output_dir: str,
               **kw) -> dict[str, Any]:
        import importlib
        import os
        entry = os.environ.get("SEVILA_ENTRY", "sevila_inference.select")
        try:
            mod = importlib.import_module(entry)
            indices = mod.select_frames(video_path=video_path,
                                        question=question, K=int(budget))
        except Exception as e:
            raise RuntimeError(
                f"SeViLA entry '{entry}' is not available: {e!r}. "
                "Install SeViLA and set SEVILA_ENTRY to the import path of "
                "a module exposing select_frames(video_path, question, K)."
            )
        # decode indices
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames, ts = [], []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, bgr = cap.read()
            if not ok or bgr is None:
                continue
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            ts.append(int(i) / float(fps))
        cap.release()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        rows = _save_frames(frames, out, ts, list(indices)[: len(frames)])
        res = {
            "video_id": Path(video_path).stem, "question": question,
            "dataset": dataset, "selection_method": self.name,
            "selected_frames": rows, "budget": int(budget),
        }
        (out / "result.json").write_text(json.dumps(res, indent=2))
        return res


@dataclass
class VideoAgentSelector:
    """Wrapper around the VideoAgent repo (Wang et al., ECCV 2024).

    Override VIDEOAGENT_ENTRY to a module exposing
    `select_frames(video_path, question, K)`.
    """
    name: str = "videoagent"

    def select(self, *, video_path: str, question: str, dataset: str,
               budget: int, tau_sec: float, output_dir: str,
               **kw) -> dict[str, Any]:
        import importlib, os
        entry = os.environ.get("VIDEOAGENT_ENTRY", "videoagent.select")
        try:
            mod = importlib.import_module(entry)
            indices = mod.select_frames(video_path=video_path,
                                        question=question, K=int(budget))
        except Exception as e:
            raise RuntimeError(
                f"VideoAgent entry '{entry}' is not available: {e!r}. "
                "Install VideoAgent and set VIDEOAGENT_ENTRY accordingly."
            )
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames, ts = [], []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, bgr = cap.read()
            if not ok or bgr is None:
                continue
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            ts.append(int(i) / float(fps))
        cap.release()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        rows = _save_frames(frames, out, ts, list(indices)[: len(frames)])
        res = {
            "video_id": Path(video_path).stem, "question": question,
            "dataset": dataset, "selection_method": self.name,
            "selected_frames": rows, "budget": int(budget),
        }
        (out / "result.json").write_text(json.dumps(res, indent=2))
        return res


# ---------- registry -------------------------------------------------------

REGISTRY = {
    "uniform":   UniformSelector,
    "uniform_center": UniformCenteredSelector,
    "random":    RandomSelector,
    "motion":    MotionOnlySelector,
    "clip":      CLIPOnlySelector,
    "aks":       AKSSelector,
    "keyscore":  KeyScoreSelector,
    "videotree": VideoTreeSelector,
    "sevila":    SeViLASelector,
    "videoagent": VideoAgentSelector,
}


def make(name: str):
    if name not in REGISTRY:
        raise KeyError(f"unknown baseline: {name}; available={list(REGISTRY)}")
    factory = REGISTRY[name]
    return factory() if callable(factory) else factory
