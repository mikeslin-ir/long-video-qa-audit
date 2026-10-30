from __future__ import annotations

import numpy as np


def minmax_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.astype(np.float32)
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def zscore_sigmoid_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.astype(np.float32)
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd <= 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    z = (x - mu) / sd
    y = 1.0 / (1.0 + np.exp(-z))
    return y.astype(np.float32)


def rank_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.astype(np.float32)
    order = np.argsort(np.argsort(x))
    if len(order) == 1:
        return np.array([1.0], dtype=np.float32)
    return (order.astype(np.float64) / (len(order) - 1)).astype(np.float32)


def normalize_scores(x: np.ndarray, method: str = "minmax") -> np.ndarray:
    m = method.lower()
    if m == "minmax":
        return minmax_normalize(x)
    if m in {"zsigmoid", "zscore_sigmoid"}:
        return zscore_sigmoid_normalize(x)
    if m == "rank":
        return rank_normalize(x)
    raise ValueError(f"Unknown normalization method: {method}")


def fuse_skill_scores(
    skill_scores: dict[str, np.ndarray],
    weights: dict[str, float],
    normalization: str = "minmax",
    lambda_uniform: float = 0.0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Normalize each skill score then compute weighted sum fusion."""
    if not skill_scores:
        return np.zeros((0,), dtype=np.float32), {}

    names = list(skill_scores.keys())
    T = len(skill_scores[names[0]])
    normalized: dict[str, np.ndarray] = {}

    for name, s in skill_scores.items():
        s = np.asarray(s)
        if len(s) != T:
            raise ValueError(f"length mismatch for {name}: {len(s)} != {T}")
        normalized[name] = normalize_scores(s, method=normalization)

    fused = np.zeros((T,), dtype=np.float64)
    for name, s in normalized.items():
        w = float(weights.get(name, 0.0))
        fused += w * s

    if lambda_uniform > 0:
        uniform = np.ones((T,), dtype=np.float64)
        fused = (1.0 - lambda_uniform) * fused + lambda_uniform * uniform

    return fused.astype(np.float32), normalized
