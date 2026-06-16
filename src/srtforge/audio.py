"""Extract audio from a video file using ffmpeg."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


class FFmpegError(RuntimeError):
    """Raised when ffmpeg is missing or fails."""


def extract_audio(video_path: Path, sample_rate: int = 16000) -> Path:
    """Extract mono PCM WAV at the given sample rate into a temp file.

    Returns the path to the WAV file. Caller is responsible for deletion.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FFmpegError("ffmpeg not found on PATH. Install it (e.g. `nb install ffmpeg`).")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    out_path = Path(tmp.name)

    cmd = [
        ffmpeg,
        "-y",
        "-i", str(video_path),
        "-vn",                       # drop video
        "-ac", "1",                  # mono
        "-ar", str(sample_rate),     # sample rate
        "-c:a", "pcm_s16le",         # 16-bit PCM
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        out_path.unlink(missing_ok=True)
        raise FFmpegError(
            f"ffmpeg failed (exit {proc.returncode}):\n{proc.stderr.strip()[-2000:]}"
        )
    if out_path.stat().st_size == 0:
        out_path.unlink(missing_ok=True)
        raise FFmpegError("ffmpeg produced no audio. Does the file have an audio track?")
    return out_path
