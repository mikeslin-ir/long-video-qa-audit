"""RedundancyFilter: drop temporally-redundant near-duplicate frames from the
proposal pool using cheap low-level features (grayscale histogram).

Unlike MotionDiffFilter (which *selects* high-motion frames and therefore
discards static-but-informative frames such as a static speaker or an
on-screen sign), this filter only removes frames that are near-duplicates of a
neighbouring kept frame. Static, semantically-distinct frames are preserved.

The number of kept frames is content-adaptive: a redundancy threshold is
derived from the distribution of adjacent-frame similarities, and the result is
clamped into a budget band ``[K_min, K_max]`` so very dynamic videos keep more
frames while static videos keep fewer. No neural model is used, so this runs
cheaply in the online (no-cache) setting.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..protocols import ProposalSet


@dataclass
class RedundancyFilter:
    name: str = "redundancy"
    cost_per_frame: float = 0.001
    downscale: int = 4            # spatial downsample before histogram
    hist_bins: int = 64           # grayscale histogram bins
    # Adaptive threshold: tau = clip(mean(sim) + k_std * std(sim), tau_lo, tau_hi).
    # A frame whose similarity to the last kept frame is >= tau is redundant.
    k_std: float = 0.5
    tau_lo: float = 0.80
    tau_hi: float = 0.985
    K_min: int = 8                # never drop below this many frames
    # K (passed by the router == plan.K_keyframe) is used as the upper budget.

    # ---- features ---------------------------------------------------------
    def _hist(self, frame_rgb: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        if self.downscale > 1:
            g = cv2.resize(
                g, (max(1, g.shape[1] // self.downscale),
                    max(1, g.shape[0] // self.downscale))
            )
        h = cv2.calcHist([g], [0], None, [self.hist_bins], [0, 256]).flatten()
        n = float(np.linalg.norm(h))
        return (h / n).astype(np.float32) if n > 0 else h.astype(np.float32)

    @staticmethod
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))  # both L2-normalized

    # ---- main -------------------------------------------------------------
    def filter(
        self,
        proposal: ProposalSet,
        frames_rgb: list[np.ndarray],
        question: str | None,
        meta: dict,
        K: int,
    ) -> ProposalSet:
        n = len(frames_rgb)
        if n <= 1:
            return proposal

        K_max = max(int(K), self.K_min)
        feats = [self._hist(f) for f in frames_rgb]

        # Adjacent-frame similarities (temporal neighbours).
        adj = np.array(
            [self._cos(feats[i], feats[i - 1]) for i in range(1, n)],
            dtype=np.float64,
        )
        tau = float(np.clip(adj.mean() + self.k_std * adj.std(),
                            self.tau_lo, self.tau_hi))

        # Greedy temporal dedup: keep a frame only if it is novel enough
        # relative to the previously kept frame. Track novelty for budgeting.
        kept = [0]
        novelty = {0: 1.0}
        for i in range(1, n):
            sim = self._cos(feats[i], feats[kept[-1]])
            if sim < tau:
                kept.append(i)
                novelty[i] = 1.0 - sim

        # ---- budget guards ------------------------------------------------
        if len(kept) > K_max:
            # Too many survivors: keep the first frame + the most novel ones.
            ranked = sorted(kept, key=lambda i: novelty.get(i, 0.0), reverse=True)
            keep_set = set(ranked[:K_max])
            keep_set.add(kept[0])
            kept = sorted(keep_set)
        elif len(kept) < self.K_min:
            # Too few survivors: add back the most novel of the dropped frames.
            kept_set = set(kept)
            dropped = [i for i in range(n) if i not in kept_set]
            # Novelty of a dropped frame = distance to its nearest kept neighbour.
            def drop_novelty(i: int) -> float:
                nearest = min(kept, key=lambda k: abs(k - i))
                return 1.0 - self._cos(feats[i], feats[nearest])
            dropped.sort(key=drop_novelty, reverse=True)
            need = min(self.K_min, n) - len(kept)
            kept = sorted(kept_set.union(dropped[:max(0, need)]))

        keep = np.asarray(kept, dtype=np.int64)
        base_scores = (proposal.scores[keep]
                       if proposal.scores is not None else None)
        return ProposalSet(
            frame_indices=proposal.frame_indices[keep],
            timestamps_sec=proposal.timestamps_sec[keep],
            scores=base_scores,
            source=f"{proposal.source}+{self.name}",
        )
