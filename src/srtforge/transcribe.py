"""Transcribe audio with local Whisper via MLX (Apple Silicon GPU)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"


def transcribe(
    audio_path: Path,
    model: str = DEFAULT_MODEL,
    language: str | None = None,
    word_timestamps: bool = False,
) -> list[dict[str, Any]]:
    """Transcribe audio and return Whisper segments.

    Each segment is a dict with at least 'start', 'end', 'text'.
    """
    import mlx_whisper  # imported lazily so --help is fast

    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=model,
        language=language,
        word_timestamps=word_timestamps,
        verbose=False,
    )
    return result.get("segments", [])
