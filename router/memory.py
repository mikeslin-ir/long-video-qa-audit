"""Procedural memory for the skill-router planner.

A simple, dependency-light memory store of past
    (question, dataset, decision, observed_reward)
records.  At inference time we retrieve the top-k nearest historical
records by question embedding and use them to *augment* the base
planner's decision.

Embedding backend (in priority order):
    1. sentence-transformers / E5-small  (if installed)
    2. SigLIP / CLIP text encoder (already a project dep)
    3. simple bag-of-words hashing fallback (deterministic, CPU-only)

The store format is a single JSONL file plus a sibling .npy of
embeddings to avoid recomputing them on load.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np


# ---------- embeddings -----------------------------------------------------

_EMB_DIM = 256


def _hash_bow_embed(text: str, dim: int = _EMB_DIM) -> np.ndarray:
    """Deterministic CPU-only fallback embedding (FeatureHashing-style)."""
    v = np.zeros((dim,), dtype=np.float32)
    for tok in (text or "").lower().split():
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sgn = 1.0 if (h // dim) % 2 == 0 else -1.0
        v[idx] += sgn
    n = float(np.linalg.norm(v))
    if n > 0:
        v /= n
    return v


def _try_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        m = SentenceTransformer("intfloat/e5-small-v2")
        def encode(texts: list[str]) -> np.ndarray:
            return np.asarray(m.encode(texts, normalize_embeddings=True), dtype=np.float32)
        return encode
    except Exception:
        return None


class _Embedder:
    def __init__(self) -> None:
        self._fn = _try_sentence_transformer()

    def encode_one(self, text: str) -> np.ndarray:
        if self._fn is not None:
            v = self._fn([text])[0]
            return np.asarray(v, dtype=np.float32)
        return _hash_bow_embed(text)

    @property
    def dim(self) -> int:
        if self._fn is None:
            return _EMB_DIM
        # encode a tiny string to discover the dim once
        return int(self._fn(["dim probe"]).shape[1])


# ---------- store ----------------------------------------------------------

@dataclass
class MemoryRecord:
    qid: str
    dataset: str
    question: str
    selected_skills: list[str]
    weights: dict[str, float]
    reward: float                 # higher is better; in [0, 1] when possible
    timestamp: float
    expert: str | None = None     # routed expert name (optional, back-compat)
    expert_beta: dict[str, float] | None = None  # categorical over experts

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(line: str) -> "MemoryRecord":
        d = json.loads(line)
        # back-compat: tolerate older records without expert fields
        d.setdefault("expert", None)
        d.setdefault("expert_beta", None)
        return MemoryRecord(**d)


class MemoryStore:
    """JSONL + parallel np.float32 embedding matrix on disk."""

    def __init__(self, path: str | Path, embedder: Optional[_Embedder] = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.emb_path = self.path.with_suffix(".emb.npy")
        self.embedder = embedder or _Embedder()
        self._records: list[MemoryRecord] = []
        self._embeddings: np.ndarray | None = None
        self._load()

    # ----- io --------------------------------------------------------------

    def _load(self) -> None:
        if self.path.exists():
            with self.path.open() as f:
                self._records = [MemoryRecord.from_json(line)
                                 for line in f if line.strip()]
        if self.emb_path.exists() and len(self._records) > 0:
            arr = np.load(self.emb_path)
            if arr.shape[0] == len(self._records):
                self._embeddings = arr.astype(np.float32)
        if self._embeddings is None and self._records:
            self._embeddings = np.stack(
                [self.embedder.encode_one(r.question) for r in self._records]
            ).astype(np.float32)
            np.save(self.emb_path, self._embeddings)

    def _flush(self) -> None:
        with self.path.open("w") as f:
            for r in self._records:
                f.write(r.to_json() + "\n")
        if self._embeddings is not None:
            np.save(self.emb_path, self._embeddings)

    # ----- write -----------------------------------------------------------

    def add(
        self,
        qid: str,
        dataset: str,
        question: str,
        selected_skills: list[str],
        weights: dict[str, float],
        reward: float,
        expert: str | None = None,
        expert_beta: dict[str, float] | None = None,
    ) -> None:
        rec = MemoryRecord(
            qid=qid, dataset=dataset, question=question,
            selected_skills=list(selected_skills),
            weights={k: float(v) for k, v in weights.items()},
            reward=float(reward),
            timestamp=time.time(),
            expert=expert,
            expert_beta=(
                {k: float(v) for k, v in expert_beta.items()}
                if expert_beta else None
            ),
        )
        emb = self.embedder.encode_one(question).reshape(1, -1).astype(np.float32)
        self._records.append(rec)
        if self._embeddings is None:
            self._embeddings = emb
        else:
            self._embeddings = np.concatenate([self._embeddings, emb], axis=0)
        self._flush()

    def __len__(self) -> int:
        return len(self._records)

    # ----- read ------------------------------------------------------------

    def retrieve(
        self,
        question: str,
        dataset: str | None = None,
        k: int = 5,
        min_reward: float = 0.0,
    ) -> list[tuple[float, MemoryRecord]]:
        if not self._records or self._embeddings is None:
            return []
        q = self.embedder.encode_one(question).astype(np.float32)
        q_n = q / (float(np.linalg.norm(q)) + 1e-9)
        E = self._embeddings
        E_n = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        sims = E_n @ q_n  # cosine
        order = np.argsort(-sims)
        out: list[tuple[float, MemoryRecord]] = []
        for idx in order:
            r = self._records[int(idx)]
            if dataset and r.dataset != dataset:
                continue
            if r.reward < min_reward:
                continue
            out.append((float(sims[int(idx)]), r))
            if len(out) >= k:
                break
        return out


# ---------- memory-augmented planner --------------------------------------

class MemoryAugmentedPlanner:
    """Wraps any base planner; blends retrieved decisions into its plan.

    Blending rule
    -------------
    - skill set: union of (a) base.selected_skills and (b) the
      most-frequent skill from the top-k retrieved, weighted by
      cosine-similarity * memory_weight.
    - weights: convex combination
          w = (1 - alpha) * base_w + alpha * mem_w
      where mem_w is the similarity-weighted average of retrieved
      decisions' weights, restricted to the union skill set.

    If the store is empty or no record passes the dataset filter,
    falls back to the base planner's output unchanged.
    """

    def __init__(
        self,
        base_planner: Any,
        memory: MemoryStore,
        alpha: float = 0.4,
        k: int = 5,
        min_similarity: float = 0.3,
        min_reward: float = 0.0,
    ) -> None:
        self.base = base_planner
        self.memory = memory
        self.alpha = float(alpha)
        self.k = int(k)
        self.min_similarity = float(min_similarity)
        self.min_reward = float(min_reward)

    def plan(
        self,
        question: str,
        dataset: str,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        base_plan = self.base.plan(question=question, dataset=dataset, config=config)
        hits = self.memory.retrieve(
            question=question,
            dataset=dataset,
            k=self.k,
            min_reward=self.min_reward,
        )
        hits = [(s, r) for s, r in hits if s >= self.min_similarity]
        if not hits:
            base_plan = dict(base_plan)
            base_plan.setdefault("memory_used", False)
            return base_plan

        # similarity-weighted skill votes
        votes: dict[str, float] = {}
        wsum: dict[str, float] = {}
        wcnt: dict[str, float] = {}
        sim_total = 0.0
        for sim, rec in hits:
            sim_total += sim
            for s in rec.selected_skills:
                votes[s] = votes.get(s, 0.0) + sim
            for s, w in rec.weights.items():
                wsum[s] = wsum.get(s, 0.0) + sim * float(w)
                wcnt[s] = wcnt.get(s, 0.0) + sim
        mem_w = {s: wsum[s] / wcnt[s] for s in wsum if wcnt.get(s, 0.0) > 0}

        base_skills = list(base_plan.get("selected_skills", []))
        base_weights = dict(base_plan.get("weights", {}))

        # merged skill set: base + top-1 most-voted memory skill not already in
        merged = list(base_skills)
        for s, _ in sorted(votes.items(), key=lambda kv: -kv[1]):
            if s in merged:
                continue
            merged.append(s)
            if len(merged) >= max(3, len(base_skills) + 1):
                break

        # blend weights on the union
        out_w: dict[str, float] = {}
        for s in merged:
            b = float(base_weights.get(s, 0.0))
            m = float(mem_w.get(s, 0.0))
            out_w[s] = (1 - self.alpha) * b + self.alpha * m
        # renormalize
        total = sum(max(v, 0.0) for v in out_w.values())
        if total > 0:
            out_w = {s: max(v, 0.0) / total for s, v in out_w.items()}

        # ---- expert-beta blend (memory votes for which expert) -----------
        base_beta = dict(base_plan.get("expert_beta", {}) or {})
        mem_beta_sum: dict[str, float] = {}
        mem_beta_cnt = 0.0
        for sim, rec in hits:
            if rec.expert_beta:
                for e, p in rec.expert_beta.items():
                    mem_beta_sum[e] = mem_beta_sum.get(e, 0.0) + sim * float(p)
                mem_beta_cnt += sim
            elif rec.expert:
                mem_beta_sum[rec.expert] = mem_beta_sum.get(rec.expert, 0.0) + sim
                mem_beta_cnt += sim
        out_beta: dict[str, float] = {}
        if base_beta or mem_beta_sum:
            keys = set(base_beta) | set(mem_beta_sum)
            mem_beta = (
                {e: mem_beta_sum.get(e, 0.0) / mem_beta_cnt for e in keys}
                if mem_beta_cnt > 0 else {}
            )
            for e in keys:
                b = float(base_beta.get(e, 0.0))
                m = float(mem_beta.get(e, 0.0))
                out_beta[e] = (1 - self.alpha) * b + self.alpha * m
            tot = sum(max(v, 0.0) for v in out_beta.values())
            if tot > 0:
                out_beta = {e: max(v, 0.0) / tot for e, v in out_beta.items()}

        out = dict(base_plan)
        out["selected_skills"] = merged
        out["weights"] = out_w
        if out_beta:
            out["expert_beta"] = out_beta
            out["expert"] = max(out_beta, key=out_beta.get)
        out["memory_used"] = True
        out["memory_hits"] = [
            {"similarity": s, "qid": r.qid, "reward": r.reward,
             "selected_skills": r.selected_skills, "weights": r.weights,
             "expert": r.expert}
            for s, r in hits
        ]
        out["memory_blend_alpha"] = self.alpha
        out["reason"] = (out.get("reason", "") or "") + " | memory-augmented"
        return out
