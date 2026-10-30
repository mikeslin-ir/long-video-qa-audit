"""STACFP proposer: Spatial-Temporal Adaptive Clustering Frame Proposals.

Adapts our prior work (CaptionBasedVideoKeyFrameSelection/.../FrameSampler_STACFP)
into the cascade `FrameProposer` protocol.

Idea: describe each sampled frame by a 3D color histogram concatenated with its
normalized timestamp (weighted by `gamma_time`), cluster the frames with KMeans
where K is chosen adaptively by silhouette score, and emit one representative
anchor per cluster (the frame closest to its centroid). This yields a
content-adaptive number of anchors that cover the distinct spatio-temporal
"states" of the video -- denser where the scene changes, sparser where it is
static -- instead of a fixed uniform stride.

Cheap (no neural inference): only OpenCV decoding + histograms + KMeans. Results
are cached to ``cache_dir`` keyed by video path + config so repeated runs are
free.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..protocols import ProposalSet


@dataclass
class STACFPProposer:
    name: str = "stacfp"
    cost_per_video: float = 1.5  # sec (decode + histograms + KMeans sweep)
    cache_dir: str = "results/cascade_cache"
    frame_step: int = 10                 # sample every Nth frame for features
    color_space: str = "HSV"             # "HSV" or "LAB"
    hist_bins: tuple[int, int, int] = (8, 8, 8)
    gamma_time: float = 1.0              # weight on the normalized-time feature
    k_min: int = 5
    k_max: int = 30
    resize_shorter_to: Optional[int] = 256
    random_state: int = 42
    # --- self-adaptive cut-density gating (Option C) ---------------------
    # STACFP already computes a per-frame colour histogram; the density of
    # large adjacent-histogram jumps is a *free* estimate of how heavily the
    # video is edited. We use it to switch between two limiting regimes:
    #   low  cut density -> centroid clustering (long-take / static video)
    #   high cut density -> snap anchors to cuts (behaves like shot detection)
    # This lets a single proposer subsume uniform sampling and shot boundary
    # detection, so the planner never has to know the video structure ahead
    # of time.
    cut_abs_min: float = 0.35       # min adjacent-hist L2 dist to call a "cut"
    cut_k_std: float = 2.0          # outlier factor over the distance dist.
    cut_density_hi: float = 6.0     # cuts/minute above which we snap-to-cuts

    def __post_init__(self):
        try:
            from sklearn.cluster import KMeans  # noqa: F401
            from sklearn.metrics import silhouette_score  # noqa: F401
            from sklearn.preprocessing import normalize  # noqa: F401
            self._available = True
        except Exception as e:  # pragma: no cover - optional dep
            self._available = False
            self._import_error = e
        self.color_space = self.color_space.upper()
        Path(self.cache_dir, "stacfp").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ public
    def propose(self, video_path: str, question: str | None, meta: dict, K: int) -> ProposalSet:
        if not self._available:
            raise RuntimeError(f"sklearn unavailable for STACFP: {self._import_error}")

        cache_path = self._cache_path(video_path)
        if cache_path.exists():
            data = np.load(cache_path)
            idx, ts = data["idx"].astype(np.int64), data["ts"].astype(np.float64)
            cut_density = float(data["cut_density"]) if "cut_density" in data else 0.0
            mode = str(data["mode"]) if "mode" in data else "cluster"
        else:
            idx, ts, cut_density, mode = self._cluster_anchors(video_path, int(K))
            np.savez(cache_path, idx=idx, ts=ts,
                     cut_density=np.float64(cut_density), mode=np.str_(mode))

        # Surface the free measurement so it lands in cascade_result.json and
        # can seed the LLM planner's procedural memory.
        if isinstance(meta, dict):
            meta["stacfp_cut_density"] = float(cut_density)
            meta["stacfp_mode"] = str(mode)

        if idx.size == 0:
            return ProposalSet(np.array([], dtype=np.int64),
                               np.array([], dtype=np.float64), source=self.name)

        # Respect the proposal budget: keep at most K anchors, evenly subsampled
        # in time so coverage is preserved when the silhouette picks > K clusters.
        if idx.size > K:
            keep = np.linspace(0, idx.size - 1, K).round().astype(int)
            keep = np.unique(keep)
            idx, ts = idx[keep], ts[keep]
        return ProposalSet(idx, ts, source=f"{self.name}:{mode}")

    # ----------------------------------------------------------------- helpers
    def _cache_path(self, video_path: str) -> Path:
        from hashlib import sha1
        key = sha1(
            f"{Path(video_path).resolve()}|{self.frame_step}|{self.color_space}"
            f"|{self.hist_bins}|{self.gamma_time}|{self.k_min}|{self.k_max}"
            f"|{self.resize_shorter_to}|{self.cut_abs_min}|{self.cut_k_std}"
            f"|{self.cut_density_hi}".encode()
        ).hexdigest()[:20]
        return Path(self.cache_dir, "stacfp", f"{key}.npz")

    def _cluster_anchors(self, video_path: str, K: int
                         ) -> tuple[np.ndarray, np.ndarray, float, str]:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import normalize

        X, frame_idxs, times_sec, H = self._extract_features(video_path)
        n = int(X.shape[0])
        if n == 0:
            return (np.array([], dtype=np.int64), np.array([], dtype=np.float64),
                    0.0, "cluster")
        if n <= 2:
            order = np.argsort(frame_idxs)
            return (frame_idxs[order].astype(np.int64),
                    times_sec[order].astype(np.float64), 0.0, "cluster")

        # ---- free cut-density estimate from the histograms we already have ---
        cut_density, cut_positions = self._estimate_cut_density(H, times_sec)

        # ---- high-cut regime: snap anchors to detected cuts (shot-like) ------
        if cut_density >= self.cut_density_hi and len(cut_positions) >= 1:
            sel_idx, sel_ts = self._anchors_from_cuts(
                cut_positions, frame_idxs, times_sec, K)
            if sel_idx.size > 0:
                return sel_idx, sel_ts, float(cut_density), "snap_cuts"

        # ---- low-cut regime: spatio-temporal centroid clustering -------------
        # Cap the silhouette sweep upper bound by the proposal budget too.
        k_hi = min(self.k_max, K, n - 1)
        k_lo = max(2, self.k_min)
        k_candidates = list(range(k_lo, max(k_lo, k_hi) + 1))

        best = {"k": None, "score": -np.inf, "labels": None, "centers": None}
        for k in k_candidates:
            km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
            labels = km.fit_predict(X)
            try:
                score = float(silhouette_score(X, labels, metric="euclidean"))
            except Exception:
                continue
            if score > best["score"]:
                best = {"k": k, "score": score, "labels": labels,
                        "centers": km.cluster_centers_}

        if best["k"] is None:
            k = min(max(2, self.k_min), n)
            km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
            best["labels"] = km.fit_predict(X)
            best["centers"] = km.cluster_centers_

        reps = self._representatives_by_centroid(X, best["labels"], best["centers"])
        sel = sorted(int(frame_idxs[i]) for i in reps)
        sel_idx = np.array(sel, dtype=np.int64)
        # Map back to timestamps via the per-frame time array.
        idx_to_ts = {int(f): float(t) for f, t in zip(frame_idxs, times_sec)}
        sel_ts = np.array([idx_to_ts[i] for i in sel], dtype=np.float64)
        return sel_idx, sel_ts, float(cut_density), "cluster"

    def _estimate_cut_density(self, H: np.ndarray, times_sec: np.ndarray
                              ) -> tuple[float, np.ndarray]:
        """Cut density (cuts/minute) from adjacent L2-normalised histogram jumps.

        A "cut" is an adjacent-frame distance that is both an outlier of the
        distance distribution (> mean + k_std*std) and above an absolute floor
        (`cut_abs_min`), so gradual illumination/camera drift in a long take is
        not mistaken for an edit. Returns the density and the local indices
        (into the sampled array) *after* which a cut occurs.
        """
        if H.shape[0] < 3:
            return 0.0, np.array([], dtype=int)
        d = np.linalg.norm(np.diff(H, axis=0), axis=1)  # adjacent distances
        tau = max(self.cut_abs_min,
                  float(d.mean() + self.cut_k_std * d.std()))
        cut_positions = np.where(d > tau)[0]  # cut between i and i+1
        span_sec = float(times_sec[-1] - times_sec[0])
        span_min = max(span_sec / 60.0, 1e-6)
        cut_density = float(len(cut_positions)) / span_min
        return cut_density, cut_positions

    def _anchors_from_cuts(self, cut_positions: np.ndarray, frame_idxs: np.ndarray,
                           times_sec: np.ndarray, K: int
                           ) -> tuple[np.ndarray, np.ndarray]:
        """Mid-frame of each shot segment delimited by the detected cuts.

        Mirrors `ShotBoundaryProposer` (anchor = shot mid-frame) but reuses the
        histograms STACFP already computed -- no second decode / detector pass.
        """
        n = len(frame_idxs)
        # Segment boundaries (local indices): 0, cut+1, ..., n
        bounds = [0] + [int(c) + 1 for c in cut_positions] + [n]
        bounds = sorted(set(b for b in bounds if 0 <= b <= n))
        mids: list[int] = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b <= a:
                continue
            mids.append((a + b - 1) // 2)
        if not mids:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
        sel_idx = np.array([int(frame_idxs[m]) for m in mids], dtype=np.int64)
        sel_ts = np.array([float(times_sec[m]) for m in mids], dtype=np.float64)
        order = np.argsort(sel_idx)
        return sel_idx[order], sel_ts[order]

    def _extract_features(self, video_path: str):
        from sklearn.preprocessing import normalize

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or np.isclose(fps, 0.0):
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        frame_idxs, times_sec, hists = [], [], []
        read_idx = 0
        while True:
            if not cap.grab():
                break
            if read_idx % self.frame_step != 0:
                read_idx += 1
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                read_idx += 1
                continue
            frame = self._maybe_resize(frame)
            hists.append(self._color_histogram_3d(frame))
            frame_idxs.append(read_idx)
            times_sec.append(read_idx / fps)
            read_idx += 1
        cap.release()

        if not hists:
            return (np.empty((0, 0)), np.array([], dtype=int),
                    np.array([], dtype=np.float32), np.empty((0, 0), dtype=np.float32))

        H = normalize(np.vstack(hists).astype(np.float32), norm="l2")
        duration = (total_frames - 1) / fps if total_frames > 1 else (max(times_sec) if times_sec else 1.0)
        t_norm = np.array(times_sec, dtype=np.float32) / max(duration, 1e-8)
        X = np.hstack([H, (self.gamma_time * t_norm)[:, None]])
        return (X, np.array(frame_idxs, dtype=int),
                np.array(times_sec, dtype=np.float32), H)

    def _maybe_resize(self, frame: np.ndarray) -> np.ndarray:
        if self.resize_shorter_to is None:
            return frame
        h, w = frame.shape[:2]
        short = min(h, w)
        if short <= self.resize_shorter_to:
            return frame
        scale = self.resize_shorter_to / float(short)
        return cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))),
                          interpolation=cv2.INTER_AREA)

    def _color_histogram_3d(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self.color_space == "HSV":
            img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            ranges = [0, 180, 0, 256, 0, 256]
        else:  # LAB
            img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
            ranges = [0, 256, 0, 256, 0, 256]
        hist = cv2.calcHist([img], [0, 1, 2], None, list(self.hist_bins), ranges)
        return hist.flatten()

    @staticmethod
    def _representatives_by_centroid(X: np.ndarray, labels: np.ndarray,
                                     centers: np.ndarray) -> list[int]:
        reps = []
        for k in range(centers.shape[0]):
            idxs = np.where(labels == k)[0]
            if len(idxs) == 0:
                continue
            d = np.linalg.norm(X[idxs] - centers[k][None, :], axis=1)
            reps.append(int(idxs[int(np.argmin(d))]))
        return reps
