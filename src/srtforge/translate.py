"""Context-aware subtitle translation using a local MLX LLM.

Translates the full cue sequence as a continuous conversation (windowed for
long videos) while preserving a strict 1:1 mapping between source and target
cues, so the original timestamps remain valid.

Two inference backends are supported transparently:
  * ``mlx_lm``  for text-only models (e.g. Qwen3.5-9B-OptiQ-4bit)
  * ``mlx_vlm`` for multimodal models (e.g. gemma-4-26b-a4b-it-4bit)
The loader tries ``mlx_lm`` first and falls back to ``mlx_vlm``.
"""

from __future__ import annotations

import re
from typing import Any

DEFAULT_TRANSLATE_MODEL = "mlx-community/gemma-4-26b-a4b-it-4bit"

_TAG_RE = re.compile(r"\[\[(\d+)\]\]\s*(.*)")


class TranslationError(RuntimeError):
    """Raised when translation cannot be completed."""


# --------------------------------------------------------------------------- #
# Backend abstraction
# --------------------------------------------------------------------------- #
class _Backend:
    """Wraps an MLX LLM so we can call .chat(user_text) regardless of library."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.kind: str
        try:
            from mlx_lm import load as lm_load
        except ImportError as lm_import_error:
            lm_load = None
            lm_error: Exception | None = lm_import_error
        else:
            lm_error = None

        if lm_load is not None:
            try:
                self.model, self.tok = lm_load(model_id)
            except Exception as exc:
                lm_error = exc
            else:
                self.kind = "lm"
                return

        try:
            from mlx_vlm import load as vlm_load
            from mlx_vlm.utils import load_config
        except ImportError as exc:
            raise TranslationError(
                "No supported MLX translation backend is installed "
                f"(mlx-lm import/load failed: {lm_error}; "
                f"mlx-vlm import failed: {exc})."
            ) from exc

        try:
            self.model, self.proc = vlm_load(model_id)
            self.config = load_config(model_id)
        except Exception as exc:
            raise TranslationError(
                f"Failed to load translation model with mlx-lm or mlx-vlm "
                f"(mlx-lm: {lm_error}; mlx-vlm: {exc})"
            ) from exc
        self.kind = "vlm"

    def chat(self, user_text: str, max_tokens: int) -> str:
        if self.kind == "lm":
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler

            messages = [{"role": "user", "content": user_text}]
            try:
                # Disable "thinking" for reasoning models (e.g. Qwen3) so the
                # output is the translation only, not a reasoning trace.
                prompt = self.tok.apply_chat_template(
                    messages, add_generation_prompt=True, enable_thinking=False
                )
            except TypeError:
                prompt = self.tok.apply_chat_template(
                    messages, add_generation_prompt=True
                )
            raw = generate(
                self.model,
                self.tok,
                prompt,
                max_tokens=max_tokens,
                sampler=make_sampler(temp=0.0),
                verbose=False,
            )
            return _strip_reasoning(raw)

        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        prompt = apply_chat_template(
            self.proc, self.config, user_text, add_generation_prompt=True, num_images=0
        )
        result = generate(
            self.model, self.proc, prompt, max_tokens=max_tokens, verbose=False
        )
        return _strip_reasoning(getattr(result, "text", str(result)))


def _strip_reasoning(text: str) -> str:
    """Remove <think>...</think> reasoning blocks some models emit."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# --------------------------------------------------------------------------- #
# Prompt construction & parsing
# --------------------------------------------------------------------------- #
def _build_prompt(
    target_language: str,
    window_texts: list[str],
    context_pairs: list[tuple[str, str]],
) -> str:
    lines = [
        f"You are a professional subtitle translator. Translate the subtitle "
        f"lines below into {target_language}.",
        "Read them as one continuous conversation so the translation flows "
        "naturally across lines (correct pronouns, tense, and sentence "
        "continuation across cues).",
        "",
        "Rules:",
        "- Output EXACTLY one line per input line, in the same order.",
        "- Prefix each output line with its [[n]] tag exactly as given.",
        "- Never merge, split, add, reorder, or omit lines.",
        f"- Translate naturally and idiomatically into {target_language}; "
        "do not transliterate.",
        "- Output ONLY the tagged translated lines, nothing else.",
    ]
    if context_pairs:
        lines.append("")
        lines.append(
            "Preceding context (already translated, for reference only -- do "
            "NOT translate or repeat these):"
        )
        for src, tgt in context_pairs:
            lines.append(f"  source: {src}")
            lines.append(f"  target: {tgt}")
    lines.append("")
    lines.append("Lines to translate:")
    for i, text in enumerate(window_texts, start=1):
        lines.append(f"[[{i}]] {text}")
    return "\n".join(lines)


def _parse_numbered(text: str, expected: int) -> dict[int, str] | None:
    """Parse '[[n]] translation' lines. Returns mapping or None if incomplete."""
    out: dict[int, str] = {}
    for line in text.splitlines():
        m = _TAG_RE.search(line.strip())
        if not m:
            continue
        idx = int(m.group(1))
        if 1 <= idx <= expected:
            out[idx] = m.group(2).strip()
    if len(out) == expected and all(i in out for i in range(1, expected + 1)):
        return out
    return None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def translate_segments(
    segments: list[dict[str, Any]],
    target_language: str,
    model: str = DEFAULT_TRANSLATE_MODEL,
    window: int = 40,
    context_cues: int = 6,
    backend: _Backend | None = None,
) -> list[dict[str, Any]]:
    """Return new segments with text translated into ``target_language``.

    Timestamps are preserved. Cue count is preserved (1:1 mapping). Translated
    text is returned as a single line; the caller is responsible for wrapping
    or splitting it to fit subtitle line limits (see ``segment.fit_cues``).
    ``backend`` may be injected for testing; otherwise it is loaded lazily.
    """
    # Index of non-empty cues we actually translate.
    targets = [i for i, s in enumerate(segments) if (s.get("text") or "").strip()]
    if not targets:
        return [dict(s) for s in segments]

    if backend is None:
        backend = _Backend(model)

    out = [dict(s) for s in segments]
    context: list[tuple[str, str]] = []  # (source, target) rolling buffer

    for start in range(0, len(targets), window):
        batch = targets[start : start + window]
        # Flatten any wrapped newlines to a single line for the model.
        window_texts = [
            " ".join((segments[i].get("text") or "").split()) for i in batch
        ]
        max_tokens = 64 * len(window_texts) + 256

        translated = _translate_window(
            backend, target_language, window_texts, context, max_tokens
        )

        for local_idx, seg_idx in enumerate(batch, start=1):
            out[seg_idx]["text"] = translated[local_idx]

        # Refresh rolling context with the tail of this window.
        pairs = list(zip(window_texts, [translated[i] for i in range(1, len(batch) + 1)]))
        context = pairs[-context_cues:]

    return out


def _translate_window(
    backend: _Backend,
    target_language: str,
    window_texts: list[str],
    context_pairs: list[tuple[str, str]],
    max_tokens: int,
) -> dict[int, str]:
    """Translate one window with retry, then per-line fallback for alignment."""
    expected = len(window_texts)
    prompt = _build_prompt(target_language, window_texts, context_pairs)

    parsed = _parse_numbered(backend.chat(prompt, max_tokens), expected)
    if parsed is not None:
        return parsed

    # Retry once with a stricter reminder.
    strict = prompt + (
        f"\n\nIMPORTANT: You must output exactly {expected} lines, each tagged "
        f"[[1]]..[[{expected}]]."
    )
    parsed = _parse_numbered(backend.chat(strict, max_tokens), expected)
    if parsed is not None:
        return parsed

    # Fallback: translate each line independently to guarantee alignment.
    result: dict[int, str] = {}
    for i, text in enumerate(window_texts, start=1):
        one = backend.chat(
            f"Translate the following text into {target_language}. "
            f"Output only the translation, with no quotes or notes:\n{text}",
            max_tokens=64 + 4 * len(text),
        ).strip()
        result[i] = one or text
    return result
