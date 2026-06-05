"""Convert Whisper segments into SRT subtitle text."""

from __future__ import annotations

from typing import Any, Iterable


def _format_timestamp(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm."""
    if seconds < 0:
        seconds = 0.0
    millis = round(seconds * 1000)
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def segments_to_srt(segments: Iterable[dict[str, Any]]) -> str:
    """Render Whisper segments to an SRT document string."""
    blocks: list[str] = []
    index = 1
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = _format_timestamp(float(seg.get("start", 0.0)))
        end = _format_timestamp(float(seg.get("end", 0.0)))
        blocks.append(f"{index}\n{start} --> {end}\n{text}\n")
        index += 1
    return "\n".join(blocks)
