from __future__ import annotations

import difflib
import re
import sys
from pathlib import Path

MIN_GLOBAL_SIMILARITY = 0.30
MIN_BLOCK_SIMILARITY = 0.60
MAX_INITIAL_PROMPT_CHARS = 1_000

_TRAILING_PUNCT_RE = re.compile(r"([^\w]+)$")


def read_transcript(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return ""
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"^#{1,6}\s+", " ", text, flags=re.MULTILINE)
    text = re.sub(r"^>\s?", " ", text, flags=re.MULTILINE)
    text = re.sub(r"^-\s+", " ", text, flags=re.MULTILINE)
    text = re.sub(r"^---+\s*$", " ", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[(\d{1,2}:\d{2}(?::\d{2})?)\]", " ", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _transfer_trailing_punct(whisper_token: str, replacement: str) -> str:
    m = _TRAILING_PUNCT_RE.search(whisper_token)
    if not m:
        return replacement
    punct = m.group(1)
    stripped = _TRAILING_PUNCT_RE.sub("", replacement)
    return stripped + punct


def initial_prompt_from_transcript(text: str, max_chars: int = MAX_INITIAL_PROMPT_CHARS) -> str:
    """Return a short transcript prefix suitable for Whisper's initial prompt."""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars].rsplit(" ", 1)[0].strip()
    return truncated or text[:max_chars].strip()


def correct_segments(segments: list[dict], reference_text: str) -> list[dict]:
    if not reference_text or not reference_text.strip():
        return segments

    whisper_tokens: list[str] = []
    seg_offsets: list[tuple[int, int, str]] = []

    for seg in segments:
        words = seg.get("words") or []
        start_idx = len(whisper_tokens)
        if words:
            for w in words:
                token = (w.get("word") or "").strip()
                if token:
                    whisper_tokens.append(token)
            source = "words"
        else:
            text = (seg.get("text") or "").strip()
            tokens = text.split() if text else []
            whisper_tokens.extend(tokens)
            source = "text"
        seg_offsets.append((start_idx, len(whisper_tokens), source))

    if not whisper_tokens:
        return segments

    ref_tokens = reference_text.split()
    whisper_lower = [t.lower() for t in whisper_tokens]
    ref_lower = [t.lower() for t in ref_tokens]

    sm = difflib.SequenceMatcher(None, whisper_lower, ref_lower, autojunk=False)
    if sm.ratio() < MIN_GLOBAL_SIMILARITY:
        print(
            f"[srtforge] transcript similarity {sm.ratio():.2f} below threshold "
            f"{MIN_GLOBAL_SIMILARITY}; skipping correction",
            file=sys.stderr,
        )
        return segments

    corrected = list(whisper_tokens)

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        elif tag == "replace":
            w_slice = whisper_lower[i1:i2]
            r_slice = ref_lower[j1:j2]
            if len(w_slice) == 1 and len(r_slice) == 1:
                if w_slice[0] != r_slice[0]:
                    corrected[i1] = _transfer_trailing_punct(
                        whisper_tokens[i1], ref_tokens[j1]
                    )
            else:
                block_sm = difflib.SequenceMatcher(
                    None, " ".join(w_slice), " ".join(r_slice)
                )
                if block_sm.ratio() >= MIN_BLOCK_SIMILARITY:
                    for k in range(len(w_slice)):
                        if k < len(r_slice):
                            if w_slice[k] != r_slice[k]:
                                corrected[i1 + k] = _transfer_trailing_punct(
                                    whisper_tokens[i1 + k], ref_tokens[j1 + k]
                                )
        elif tag == "delete":
            pass
        elif tag == "insert":
            pass

    new_segments: list[dict] = []
    for idx, seg in enumerate(segments):
        start_idx, end_idx, source = seg_offsets[idx]
        seg_tokens = corrected[start_idx:end_idx]

        new_seg = dict(seg)
        if source == "words":
            new_words = []
            token_i = 0
            for w in seg["words"]:
                orig = (w.get("word") or "").strip()
                if not orig:
                    new_words.append(dict(w))
                    continue
                new_w = dict(w)
                new_w["word"] = seg_tokens[token_i]
                token_i += 1
                new_words.append(new_w)
            new_seg["words"] = new_words
            new_seg["text"] = " ".join(nw["word"] for nw in new_words if (nw.get("word") or "").strip())
        else:
            new_seg["text"] = " ".join(seg_tokens)
        new_segments.append(new_seg)

    return new_segments
