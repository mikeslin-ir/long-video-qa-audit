from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    import librosa

    LIBROSA_AVAILABLE = True
except Exception:
    LIBROSA_AVAILABLE = False


def load_video_capture(path: str | Path):
    """Open a video file with OpenCV. Caller must release the capture."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    return cap


def get_video_info(cap) -> tuple[int, float, int, int]:
    """Return (num_frames, fps, width, height)."""
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return n, fps, w, h


def load_frame_at_index(path: str | Path, frame_index: int) -> np.ndarray | None:
    """Load a single frame by zero-based index as RGB, or None on decode failure."""
    cap = load_video_capture(path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def extract_audio_from_video(path: str | Path, sr: int = 22050) -> np.ndarray:
    """
    Extract 1D mono waveform from a video file with ffmpeg + librosa.

    Returns a silence fallback if ffmpeg/librosa/audio stream is unavailable.
    """
    import subprocess
    import tempfile
    import warnings

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name

    try:
        r = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(path),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                str(sr),
                "-ac",
                "1",
                "-f",
                "wav",
                wav_path,
            ],
            capture_output=True,
            timeout=60,
        )
        if r.returncode != 0 or not Path(wav_path).exists():
            if not getattr(extract_audio_from_video, "_warned_ffmpeg_fail", False):
                warnings.warn(
                    "Audio extraction failed (ffmpeg error or no audio track). Using silence.",
                    UserWarning,
                    stacklevel=2,
                )
                extract_audio_from_video._warned_ffmpeg_fail = True
            return np.zeros(sr * 10, dtype=np.float32)

        if not LIBROSA_AVAILABLE:
            return np.zeros(sr * 10, dtype=np.float32)

        y, _ = librosa.load(wav_path, sr=sr, mono=True)
        return y.astype(np.float32)
    except FileNotFoundError:
        if not getattr(extract_audio_from_video, "_warned_ffmpeg_missing", False):
            warnings.warn(
                "ffmpeg not found. Audio skill will use silence fallback.",
                UserWarning,
                stacklevel=2,
            )
            extract_audio_from_video._warned_ffmpeg_missing = True
        return np.zeros(sr * 10, dtype=np.float32)
    except (subprocess.TimeoutExpired, Exception):
        if not getattr(extract_audio_from_video, "_warned_audio_fail", False):
            warnings.warn(
                "Audio extraction failed (timeout or error). Using silence fallback.",
                UserWarning,
                stacklevel=2,
            )
            extract_audio_from_video._warned_audio_fail = True
        return np.zeros(sr * 10, dtype=np.float32)
    finally:
        Path(wav_path).unlink(missing_ok=True)
