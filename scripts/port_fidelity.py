#!/usr/bin/env python3
"""Do the harness ports implement the published selection rules?

The audit's selector ports reuse the cascade's SigLIP candidate pool and change
only the selection rule. This script checks each rule against the reference it
was ported from, on identical score arrays:

* AKS (Tang et al., CVPR 2025): the harness's ``aks_select`` against a verbatim
  transcription of the official ``meanstd`` bisection and per-segment top-k
  selection (github.com/ncTimTang/AKS, ``frame_select.py``, commit b0b8a58).
  The official routine returns a variable number of frames; the harness port
  additionally forces exactly K (top-score trim, uniform pad) so budgets match.
* KeyScore (Lin et al., 2025): ``keyscore_scores`` (closed-form leave-one-out)
  against the reference ``keyscore.KeyScore`` implementation, plus how much the
  top-K set moves between the historical weights the port uses (0.85/0.05/0.10)
  and the reference API's current defaults (0.70/0.10/0.20).
* VideoTree has no released selection routine to compare against; its port is
  documented as a selection-rule control only.

Score arrays are synthetic by default (smooth relevance curves with sparse
peaks, random unit-norm embeddings). ``--embeddings DIR`` instead reads saved
``scores_*.npy`` / ``img_*.npy`` + ``txt_*.npy`` arrays, for example dumped from
real SigLIP pools.

    python scripts/port_fidelity.py
    python scripts/port_fidelity.py --keyscore-repo ~/code/keyscore --json results/port_fidelity.json
    python scripts/port_fidelity.py --dump-real 100 --embeddings /tmp/pools     # authors, needs the videos
    python scripts/port_fidelity.py --embeddings /tmp/pools --keyscore-repo ~/code/keyscore
"""
from __future__ import annotations

import argparse
import heapq
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def official_aks_select(scores, K: int, t1: float = 0.8, t2: float = -100.0, all_depth: int = 5) -> list[int]:
    """Verbatim logic of the official ``meanstd`` + selection loop for one array.

    Differences from the official script are limited to taking the frame ids as
    positions 0..N-1 and returning the sorted selection; no padding, trimming or
    ``max(1, .)`` is applied, so the output length can differ from K.
    """
    def meanstd(dic_scores, n, fns):
        split_scores, split_fn, no_split_scores, no_split_fn = [], [], [], []
        for dic_score, fn in zip(dic_scores, fns):
            score, depth = dic_score["score"], dic_score["depth"]
            mean, std = np.mean(score), np.std(score)
            top_n = heapq.nlargest(n, range(len(score)), score.__getitem__)
            mean_diff = np.mean([score[t] for t in top_n]) - mean
            if mean_diff > t1 and std > t2:
                no_split_scores.append(dic_score)
                no_split_fn.append(fn)
            elif depth < all_depth:
                h = len(score) // 2
                split_scores += [dict(score=score[:h], depth=depth + 1), dict(score=score[h:], depth=depth + 1)]
                split_fn += [fn[:h], fn[h:]]
            else:
                no_split_scores.append(dic_score)
                no_split_fn.append(fn)
        if split_scores:
            rest_scores, rest_fn = meanstd(split_scores, n, split_fn)
        else:
            rest_scores, rest_fn = [], []
        return no_split_scores + rest_scores, no_split_fn + rest_fn

    s = np.asarray(scores, dtype=np.float64)
    if len(s) < K:
        return list(range(len(s)))
    norm = (s - s.min()) / (s.max() - s.min())
    segs, fns = meanstd([dict(score=norm, depth=0)], K, [np.arange(len(s))])
    out: list[int] = []
    for seg, fn in zip(segs, fns):
        f_num = int(K / 2 ** seg["depth"])
        topk = heapq.nlargest(f_num, range(len(seg["score"])), seg["score"].__getitem__)
        out.extend(int(fn[t]) for t in topk)
    return sorted(out)


def synthetic_scores(rng: np.random.Generator, n: int = 256) -> np.ndarray:
    """A smooth relevance curve with 0-4 sparse peaks, like a SigLIP score track."""
    base = np.convolve(rng.normal(size=n + 20), np.ones(21) / 21, mode="valid")[:n] * 0.3
    for _ in range(int(rng.integers(0, 5))):
        centre, width = rng.integers(0, n), rng.integers(2, 10)
        base += rng.uniform(0.3, 1.5) * np.exp(-0.5 * ((np.arange(n) - centre) / width) ** 2)
    return base + rng.normal(scale=0.02, size=n)


def synthetic_embeddings(rng: np.random.Generator, n: int = 256, d: int = 64):
    def unit(x):
        return x / np.linalg.norm(x, axis=-1, keepdims=True)
    topic = unit(rng.normal(size=d))
    drift = np.cumsum(rng.normal(scale=0.15, size=(n, d)), axis=0)
    frames = unit(rng.normal(size=(n, d)) * 0.5 + drift * 0.2 + np.outer(rng.random(n) ** 3, topic))
    text = unit(rng.normal(size=(1, d)) * 0.3 + topic)
    return frames.astype(np.float32), text.astype(np.float32)


def aks_report(score_arrays, K: int = 8) -> dict:
    from baselines.selectors import aks_select
    n = len(score_arrays)
    exact_k = same_when_k = 0
    jaccards, lengths = [], []
    same_after_budget = 0
    depth_k = max(1, int(np.log2(K)))
    d3_exact = 0
    d3_jaccard, topk_jaccard = [], []
    for s in score_arrays:
        official = official_aks_select(s, K)
        port = aks_select(np.asarray(s, dtype=np.float32), K)
        lengths.append(len(official))
        if len(official) == K:
            exact_k += 1
            same_when_k += int(official == port)
        a, b = set(official), set(port)
        jaccards.append(len(a & b) / max(1, len(a | b)))
        budgeted = sorted(official, key=lambda i: -s[i])[:K] if len(official) > K else official
        same_after_budget += int(len(official) >= K and sorted(budgeted) == port)
        # The official rule at the depth where it can still emit K frames (2^depth <= K).
        shallow = official_aks_select(s, K, all_depth=depth_k)
        d3_exact += int(len(shallow) == K)
        d3_jaccard.append(len(set(shallow) & b) / max(1, len(set(shallow) | b)))
        naive = set(np.argsort(-np.asarray(s))[:K].tolist())
        topk_jaccard.append(len(naive & b) / max(1, len(naive | b)))
    lengths = np.asarray(lengths)
    return {
        "arrays": n, "K": K,
        "official_returns_exactly_K": exact_k / n,
        "official_returns_more_than_K": float(np.mean(lengths > K)),
        "official_returns_fewer_than_K": float(np.mean(lengths < K)),
        "mean_official_length": float(lengths.mean()),
        "port_equals_official_when_official_has_K": same_when_k / exact_k if exact_k else float("nan"),
        "port_equals_official_after_top_score_trim": same_after_budget / max(1, int(np.sum(lengths >= K))),
        "mean_jaccard": float(np.mean(jaccards)),
        f"official_at_depth_{depth_k}_returns_exactly_K": d3_exact / n,
        f"mean_jaccard_port_vs_official_depth_{depth_k}": float(np.mean(d3_jaccard)),
        "mean_jaccard_port_vs_naive_top_K": float(np.mean(topk_jaccard)),
    }


def keyscore_report(pairs, K: int, keyscore_repo: Path | None) -> dict:
    from baselines.selectors import keyscore_scores
    out: dict = {"pairs": len(pairs), "K": K}
    historical, current = (0.85, 0.05, 0.10), (0.70, 0.10, 0.20)
    moved = []
    for img, txt in pairs:
        a = keyscore_scores(img, txt, *historical)
        b = keyscore_scores(img, txt, *current)
        top_a, top_b = set(np.argsort(-a)[:K].tolist()), set(np.argsort(-b)[:K].tolist())
        moved.append(len(top_a & top_b) / K)
    out["top_k_overlap_historical_vs_current_weights"] = float(np.mean(moved))
    if keyscore_repo is None:
        out["reference"] = "not run (pass --keyscore-repo)"
        return out
    try:
        import torch
        sys.path.insert(0, str(keyscore_repo))
        from keyscore.scoring.keyscore import KeyScore
    except Exception as exc:  # reference package or torch unavailable
        out["reference"] = f"not run ({exc!r})"
        return out
    ref = KeyScore(*historical)
    max_diff, same_topk = 0.0, 0
    for img, txt in pairs:
        got = keyscore_scores(img, txt, *historical)
        want = ref.score(torch.tensor(img), torch.tensor(txt[0]))["scores"].numpy()
        max_diff = max(max_diff, float(np.abs(got - want).max()))
        same_topk += int(set(np.argsort(-got)[:K].tolist()) == set(np.argsort(-want)[:K].tolist()))
    out["reference"] = "keyscore.KeyScore (historical weights)"
    out["max_abs_score_difference"] = max_diff
    out["same_top_k_set"] = same_topk / len(pairs)
    return out


def dump_real(n: int, cache_dir: str, out_dir: Path) -> None:
    """Save SigLIP pools (the ports' matched-feature front end) for the first
    ``n`` Video-MME long questions, one per video, as img_/txt_/scores_ arrays."""
    from baselines.selectors import _siglip_pool
    from router.qa_datasets import load_dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    seen, kept = set(), 0
    for it in load_dataset("video_mme", limit=0):
        if (it.task_type or "").strip().lower() != "long" or it.video_path in seen:
            continue
        seen.add(it.video_path)
        _, img, txt = _siglip_pool(it.video_path, it.question, "video_mme", {"cache_dir": cache_dir})
        np.save(out_dir / f"img_{kept:04d}.npy", img)
        np.save(out_dir / f"txt_{kept:04d}.npy", txt)
        np.save(out_dir / f"scores_{kept:04d}.npy", (img @ txt.T).max(axis=1).astype(np.float32))
        kept += 1
        if kept >= n:
            break
    print(f"saved {kept} pools to {out_dir}")


def load_arrays(directory: Path):
    scores = [np.load(p) for p in sorted(directory.glob("scores_*.npy"))]
    pairs = [(np.load(i), np.load(str(i).replace("img_", "txt_"))) for i in sorted(directory.glob("img_*.npy"))]
    return scores, pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arrays", type=int, default=500, help="number of synthetic arrays")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--embeddings", type=Path, default=None, help="directory of saved real arrays")
    ap.add_argument("--keyscore-repo", type=Path, default=None, help="checkout of the KeyScore reference code")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--dump-real", type=int, default=0, metavar="N",
                    help="authors: first save N real SigLIP pools into --embeddings, then exit")
    ap.add_argument("--cache-dir", default="results/skill_router_cache",
                    help="SigLIP embedding cache used by --dump-real")
    args = ap.parse_args()

    if args.dump_real:
        if not args.embeddings:
            ap.error("--dump-real needs --embeddings DIR")
        dump_real(args.dump_real, args.cache_dir, args.embeddings)
        return 0

    if args.embeddings:
        scores, pairs = load_arrays(args.embeddings)
        source = str(args.embeddings)
    else:
        rng = np.random.default_rng(args.seed)
        pairs = [synthetic_embeddings(rng) for _ in range(min(args.arrays, 200))]
        scores = [synthetic_scores(rng) for _ in range(args.arrays)]
        source = f"synthetic (seed {args.seed})"
    result = {"source": source, "aks": aks_report(scores, args.K) if scores else None,
              "keyscore": keyscore_report(pairs, args.K, args.keyscore_repo) if pairs else None}
    print(json.dumps(result, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
