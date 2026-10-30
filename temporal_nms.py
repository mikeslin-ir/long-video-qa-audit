from __future__ import annotations

import numpy as np


def select_frames_with_temporal_nms(
    scores: np.ndarray,
    timestamps: np.ndarray,
    K: int,
    tau_sec: float,
) -> list[int]:
    """Greedy top-score selection with temporal suppression window."""
    scores = np.asarray(scores, dtype=np.float64)
    ts = np.asarray(timestamps, dtype=np.float64)
    n = len(scores)
    if n == 0 or K <= 0:
        return []
    if len(ts) != n:
        raise ValueError("timestamps length must match scores length")

    alive = np.ones((n,), dtype=bool)
    selected: list[int] = []

    while len(selected) < K and np.any(alive):
        cand = np.where(alive)[0]
        best = cand[np.argmax(scores[cand])]
        selected.append(int(best))
        alive[np.abs(ts - ts[best]) <= tau_sec] = False

    if len(selected) < K:
        remain = [i for i in np.argsort(-scores).tolist() if i not in selected]
        for i in remain:
            selected.append(int(i))
            if len(selected) >= K:
                break

    selected = selected[:K]
    selected.sort(key=lambda i: ts[i])
    return selected
