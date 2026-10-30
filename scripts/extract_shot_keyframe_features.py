#!/usr/bin/env python3
"""Offline shot-aware keyframe + heavy-feature extractor.

Pipeline (all question-independent, run once per video):

    PySceneDetect (ContentDetector)  ->  shot boundaries
        -> one keyframe per shot (mid-frame)
        -> SigLIP image embedding per keyframe  (reused at query time)
        -> OCR tokens per keyframe               (reused at query time)

Why this is cheap: SigLIP image features and OCR text are independent of the
question. At query time the online router only needs a SigLIP *text* encode
(one short forward) + a dot product, instead of decoding/encoding every frame.

Outputs (per video, keyed by a stable hash of the resolved path):
    results/keyframe_cache/<key>.npz
        starts, ends, fps                 : shot boundaries (frame indices)
        keyframe_indices                  : [S] mid-frame index per shot
        timestamps                        : [S] seconds
        siglip_emb                        : [S, D] L2-normalized image embeddings
        siglip_model                      : str
    results/keyframe_cache/<key>.ocr.json
        { "<keyframe_index>": [[text, conf], ...], ... }
    results/keyframe_cache/<key>.meta.json
        { video_path, fps, total_frames, num_shots, siglip_model, ocr_backend }

Idempotent: skips a video if its .npz already exists (unless --overwrite).
"""
from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha1
from pathlib import Path

import cv2
import numpy as np

# Repo root on sys.path so `evidence_skills` imports work when run as a script.
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evidence_skills.base import FrameInfo  # noqa: E402
from evidence_skills.semantic_siglip_skill import SemanticSigLIPSkill  # noqa: E402
from evidence_skills.ocr_skill import OCRTextSkill  # noqa: E402


def _video_key(video_path: str) -> str:
    p = str(Path(video_path).resolve())
    return sha1(p.encode()).hexdigest()[:20]


def detect_shots(video_path: str, threshold: float) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Return (starts, ends, fps, total_frames) using PySceneDetect."""
    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import ContentDetector

    video = open_video(str(video_path))
    fps = float(video.frame_rate or 30.0)

    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold))
    sm.detect_scenes(video=video)
    scene_list = sm.get_scene_list()

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if not scene_list:
        # Single-shot fallback.
        return (np.array([0], dtype=np.int64),
                np.array([max(0, total - 1)], dtype=np.int64), fps, total)

    starts = np.array([int(s.get_frames()) for s, _ in scene_list], dtype=np.int64)
    ends = np.array([int(e.get_frames()) - 1 for _, e in scene_list], dtype=np.int64)
    return starts, ends, fps, total


def read_keyframes(video_path: str, indices: list[int]) -> list[FrameInfo]:
    """Decode only the requested frame indices (random seek)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames: list[FrameInfo] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, bgr = cap.read()
        if not ok or bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(FrameInfo(frame_index=int(idx),
                                timestamp_sec=float(idx) / max(fps, 1e-6),
                                image=rgb))
    cap.release()
    return frames


def process_video(
    video_path: str,
    out_dir: Path,
    siglip: SemanticSigLIPSkill,
    ocr: OCRTextSkill,
    ocr_backend: str | None,
    threshold: float,
    overwrite: bool,
) -> str:
    key = _video_key(video_path)
    npz_path = out_dir / f"{key}.npz"
    if npz_path.exists() and not overwrite:
        return "skip"

    starts, ends, fps, total = detect_shots(video_path, threshold)
    if len(starts) == 0:
        return "empty"

    # One keyframe per shot = mid-frame.
    keyframe_indices = ((starts + ends) // 2).astype(np.int64)
    keyframe_indices = np.clip(keyframe_indices, 0, max(total - 1, 0))

    frames = read_keyframes(video_path, keyframe_indices.tolist())
    if not frames:
        return "decode_fail"

    decoded_idx = np.array([f.frame_index for f in frames], dtype=np.int64)
    timestamps = decoded_idx.astype(np.float64) / max(fps, 1e-6)

    # --- Heavy feature 1: SigLIP image embeddings (question-independent) -----
    siglip_emb = np.zeros((len(frames), 768), dtype=np.float32)
    if siglip._load():
        siglip_emb = siglip._encode_images(frames)

    # --- Heavy feature 2: OCR tokens (question-independent) ------------------
    ocr_tokens: dict[str, list[list]] = {}
    if ocr_backend is not None:
        for f in frames:
            entries = ocr._extract_frame_tokens(f.image, ocr_backend)
            ocr_tokens[str(f.frame_index)] = [[t, float(c)] for (t, c) in entries]

    np.savez(
        npz_path,
        starts=starts,
        ends=ends,
        fps=np.float64(fps),
        keyframe_indices=decoded_idx,
        timestamps=timestamps,
        siglip_emb=siglip_emb,
        siglip_model=np.array(siglip.model_id),
    )
    (out_dir / f"{key}.ocr.json").write_text(json.dumps(ocr_tokens))
    (out_dir / f"{key}.meta.json").write_text(json.dumps({
        "video_path": str(Path(video_path).resolve()),
        "fps": fps,
        "total_frames": int(total),
        "num_shots": int(len(starts)),
        "num_keyframes": int(len(frames)),
        "siglip_model": siglip.model_id,
        "ocr_backend": ocr_backend,
        "scene_threshold": threshold,
    }, indent=2))
    return "ok"


def iter_video_paths(args: argparse.Namespace) -> list[str]:
    paths: list[str] = []
    if args.video:
        paths.append(args.video)
    if args.manifest:
        for line in Path(args.manifest).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                vp = obj.get("video_path") or obj.get("video")
            except json.JSONDecodeError:
                vp = line
            if vp:
                paths.append(vp)
    # De-dup while preserving order.
    seen, uniq = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="single video path")
    src.add_argument("--manifest", help="jsonl/txt with one video per line "
                                         "(jsonl may use key 'video_path')")
    ap.add_argument("--out_dir", default="results/keyframe_cache")
    ap.add_argument("--scene_threshold", type=float, default=30.0,
                    help="PySceneDetect ContentDetector threshold "
                         "(higher = fewer shots)")
    ap.add_argument("--siglip_model", default="google/siglip-base-patch16-224")
    ap.add_argument("--ocr_backend", default="auto",
                    choices=["auto", "easyocr", "tesseract", "none"],
                    help="'none' skips OCR (vision-only datasets)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="process at most N videos (0 = all)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    siglip = SemanticSigLIPSkill(model_id=args.siglip_model)
    ocr = OCRTextSkill()
    if args.ocr_backend == "none":
        ocr_backend = None
    else:
        ocr.backend = args.ocr_backend
        ocr_backend = ocr._choose_backend()
        if ocr_backend is None:
            print("[warn] no OCR backend available; continuing without OCR",
                  file=sys.stderr)

    videos = iter_video_paths(args)
    if args.limit > 0:
        videos = videos[: args.limit]

    counts: dict[str, int] = {}
    for i, vp in enumerate(videos, 1):
        if not Path(vp).exists():
            counts["missing"] = counts.get("missing", 0) + 1
            print(f"[{i}/{len(videos)}] MISSING {vp}", file=sys.stderr)
            continue
        try:
            status = process_video(vp, out_dir, siglip, ocr, ocr_backend,
                                   args.scene_threshold, args.overwrite)
        except Exception as e:  # keep going on per-video failures
            status = f"error:{type(e).__name__}"
            print(f"[{i}/{len(videos)}] ERROR {vp}: {e}", file=sys.stderr)
        counts[status] = counts.get(status, 0) + 1
        if i % 25 == 0 or i == len(videos):
            print(f"[{i}/{len(videos)}] {dict(sorted(counts.items()))}")

    print("DONE", dict(sorted(counts.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
