"""Submodular budget-K selection (N1).

Implements the facility-location form

    f_q(S) = sum_{n=1..N}  max_{m in S}  u_q(n) * k(t_n, t_m),

with a Gaussian temporal kernel
    k(t, t') = exp( -(t - t')^2 / (2 * tau^2) ).

The standard Nemhauser-Wolsey-Fisher (1978) result gives a (1 - 1/e)
approximation for monotone submodular maximization under a cardinality
constraint, which we obtain via lazy greedy.

We also expose a soft-NMS variant (use_indicator=True), where the kernel
is the indicator k(t, t') = 1[|t - t'| <= tau].  In that special case
the algorithm reduces to classical temporal NMS.
"""
from __future__ import annotations

import heapq

import numpy as np


def _gauss_kernel(timestamps: np.ndarray, tau_sec: float) -> np.ndarray:
    """N x N Gaussian similarity matrix in the time axis."""
    t = timestamps.reshape(-1, 1)
    d2 = (t - t.T) ** 2
    return np.exp(-d2 / (2.0 * (tau_sec ** 2 + 1e-9))).astype(np.float64)


def _indicator_kernel(timestamps: np.ndarray, tau_sec: float) -> np.ndarray:
    t = timestamps.reshape(-1, 1)
    return (np.abs(t - t.T) <= tau_sec).astype(np.float64)


def submodular_select_facility_location(
    scores: np.ndarray,
    timestamps: np.ndarray,
    K: int,
    tau_sec: float,
    *,
    use_indicator: bool = False,
) -> list[int]:
    """Lazy greedy maximization of the facility-location utility.

    Returns sorted timestamps of the K chosen indices.  Provides a
    (1 - 1/e) approximation guarantee versus the unknown optimum.
    """
    u = np.asarray(scores, dtype=np.float64)
    ts = np.asarray(timestamps, dtype=np.float64)
    n = len(u)
    if n == 0 or K <= 0:
        return []
    if len(ts) != n:
        raise ValueError("timestamps length must match scores length")

    K_kernel = _indicator_kernel(ts, tau_sec) if use_indicator else _gauss_kernel(ts, tau_sec)
    # Pre-compute per-frame coverage value to be maximised over m in S
    # i.e.  contribution_j[i] = u[i] * K_kernel[i, j]   for adding j to S
    # We need f(S) = sum_i  max_{j in S} contribution_j[i].

    contrib = u[:, None] * K_kernel        # shape (n, n); column j -> when j is added
    best_so_far = np.zeros((n,), dtype=np.float64)   # current  max over S
    selected: list[int] = []
    selected_set: set[int] = set()

    # Lazy greedy via max-heap of (negative-marginal-gain, last-eval-step, index).
    # `step` tracks the number of elements already accepted into S; an item is
    # up-to-date and safe to accept iff its `last_step` equals the current `step`.
    init_gain = contrib.sum(axis=0)  # f({j}) = sum_i u[i]*k(i,j)
    heap: list[tuple[float, int, int]] = [
        (-float(init_gain[j]), 0, int(j)) for j in range(n)
    ]
    heapq.heapify(heap)

    step = 0
    while len(selected) < K and heap:
        neg_gain, last_step, j = heapq.heappop(heap)
        if j in selected_set:
            continue
        if last_step == step:  # marginal gain evaluated against current S
            selected.append(j)
            selected_set.add(j)
            best_so_far = np.maximum(best_so_far, contrib[:, j])
            step += 1
        else:
            # Recompute marginal gain against current S, then re-insert.
            new_cov = np.maximum(best_so_far, contrib[:, j])
            mg = float((new_cov - best_so_far).sum())
            heapq.heappush(heap, (-mg, step, j))

    if len(selected) < K:
        # fallback (shouldn't happen for n>=K): top-up by score
        rest = [i for i in np.argsort(-u).tolist() if i not in selected_set]
        selected.extend(rest[: K - len(selected)])

    selected = selected[:K]
    selected.sort(key=lambda i: ts[i])
    return [int(i) for i in selected]


def submodular_select(
    scores,
    timestamps,
    K: int,
    tau_sec: float,
    *,
    kernel: str = "gauss",
) -> list[int]:
    """Convenience entry point that picks the kernel."""
    use_indicator = kernel == "indicator"
    return submodular_select_facility_location(
        scores=np.asarray(scores),
        timestamps=np.asarray(timestamps),
        K=K,
        tau_sec=tau_sec,
        use_indicator=use_indicator,
    )
