"""Burn subtitles permanently into a video using ffmpeg (libass)."""

from __future__ import annotations

import glob
import re
import shutil
import subprocess
from pathlib import Path

from .audio import FFmpegError

# Where fuller ffmpeg builds (with libass) tend to live.
_FFMPEG_GLOBS = (
    "/opt/nanobrew/prefix/Cellar/ffmpeg-full/*/bin/ffmpeg",
    "/opt/nanobrew/prefix/Cellar/ffmpeg/*/bin/ffmpeg",
    "/opt/homebrew/Cellar/ffmpeg*/*/bin/ffmpeg",
    "/usr/local/Cellar/ffmpeg*/*/bin/ffmpeg",
)


def _has_subtitles_filter(binary: str) -> bool:
    """True if the ffmpeg ``binary`` exposes the libass ``subtitles`` filter."""
    try:
        out = subprocess.run(
            [binary, "-hide_banner", "-filters"],
            capture_output=True, text=True, timeout=30,
        ).stdout
    except Exception:
        return False
    return re.search(r"(?m)^\s*\S+\s+subtitles\s", out) is not None


def _resolve_ffmpeg_with_subtitles() -> str:
    """Find an ffmpeg that supports subtitle burn-in (libass).

    Prefers the one on PATH; otherwise searches common install locations for a
    fuller build (e.g. ``ffmpeg-full``).
    """
    candidates: list[str] = []
    on_path = shutil.which("ffmpeg")
    if on_path:
        candidates.append(on_path)
    for pattern in _FFMPEG_GLOBS:
        candidates.extend(sorted(glob.glob(pattern), reverse=True))

    seen: set[str] = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if _has_subtitles_filter(c):
            return c
    raise FFmpegError(
        "No ffmpeg with the 'subtitles' filter (libass) was found. Your default "
        "ffmpeg lacks libass. Install a full build, e.g. `nb install ffmpeg-full`."
    )


def _escape_filter_value(value: str) -> str:
    """Escape a value embedded in an ffmpeg filtergraph option.

    The filtergraph parser treats these characters as syntax even when the
    whole filtergraph is passed as a single subprocess argument.
    """
    special = "\\':,[];"
    return "".join(f"\\{ch}" if ch in special else ch for ch in value)


def _escape_sub_path(path: Path) -> str:
    """Escape a subtitle path for the ffmpeg ``subtitles`` filter value."""
    return _escape_filter_value(str(path))


def _force_style(font_size: int | None, font_name: str | None) -> str:
    parts: list[str] = []
    if font_name:
        parts.append(f"FontName={font_name}")
    if font_size:
        parts.append(f"FontSize={font_size}")
    return ",".join(parts)


def burn_subtitles(
    video: Path,
    srt: Path,
    output: Path,
    crf: int = 18,
    preset: str = "slow",
    font_size: int | None = None,
    font_name: str | None = None,
) -> Path:
    """Render ``srt`` onto ``video`` and write ``output``.

    Framerate and resolution are preserved; audio is stream-copied. Video is
    re-encoded with libx264 at the given CRF/preset (CRF 18 ~ visually lossless).
    Returns the output path.
    """
    ffmpeg = _resolve_ffmpeg_with_subtitles()
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")
    if not srt.exists():
        raise FileNotFoundError(f"Subtitle file not found: {srt}")

    vf = f"subtitles=filename={_escape_sub_path(srt)}"
    style = _force_style(font_size, font_name)
    if style:
        vf += f":force_style={_escape_filter_value(style)}"

    cmd = [
        ffmpeg,
        "-y",
        "-i", str(video),
        "-vf", vf,
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", preset,
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",          # keep original audio losslessly
        "-movflags", "+faststart",
        str(output),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FFmpegError(
            f"ffmpeg failed to burn subtitles (exit {proc.returncode}):\n"
            f"{proc.stderr.strip()[-2000:]}"
        )
    return output
