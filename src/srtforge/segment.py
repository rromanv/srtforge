"""Re-segment Whisper output into readable, sentence-aligned subtitle cues.

Whisper times cues by speech pauses, which produces cues that run long, break
mid-sentence, or mix a sentence ending with the start of the next one. This
module rebuilds cues from word-level timestamps so that:
  * cues align to sentence boundaries (mostly one sentence per cue),
  * text fits within a max line length and line count,
  * timing respects a reading speed and min/max duration (and never overlaps).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# --- Netflix-style defaults ------------------------------------------------- #
MAX_CPL = 42          # max characters per line
MAX_LINES = 2         # max lines per cue
READING_CPS = 17.0    # characters per second a viewer can comfortably read
MIN_DURATION = 0.8    # seconds
MAX_DURATION = 7.0    # seconds
MIN_GAP = 0.084       # ~2 frames @ 24fps, gap between consecutive cues

# Sentence-final punctuation (incl. Spanish/CJK variants).
_SENT_END = tuple(".!?…。！？")
# Preferred clause-break punctuation when splitting a long sentence.
_CLAUSE_PUNCT = tuple(",;:、，；：")
# Tokens that end with '.' but are not sentence ends.
_ABBREV = {
    "mr.", "mrs.", "ms.", "dr.", "sr.", "sra.", "srta.", "st.", "vs.",
    "etc.", "e.g.", "i.e.", "no.", "fig.", "approx.", "prof.",
}
_DECIMAL_RE = re.compile(r"\d[.,]$")


@dataclass
class Opts:
    max_cpl: int = MAX_CPL
    max_lines: int = MAX_LINES
    reading_cps: float = READING_CPS
    min_duration: float = MIN_DURATION
    max_duration: float = MAX_DURATION
    min_gap: float = MIN_GAP


@dataclass
class _Word:
    text: str
    start: float
    end: float


@dataclass
class _Sentence:
    words: list[_Word] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words).strip()

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end


# --------------------------------------------------------------------------- #
# Word flattening & sentence splitting
# --------------------------------------------------------------------------- #
def _flatten_words(segments: list[dict[str, Any]]) -> list[_Word]:
    words: list[_Word] = []
    for seg in segments:
        for w in seg.get("words") or []:
            text = (w.get("word") or "").strip()
            if not text:
                continue
            words.append(_Word(text, float(w["start"]), float(w["end"])))
    return words


def _is_sentence_end(token: str) -> bool:
    if not token.endswith(_SENT_END):
        return False
    low = token.lower()
    if low in _ABBREV:
        return False
    if _DECIMAL_RE.search(token):  # e.g. "3." or "3,"
        return False
    return True


def _split_sentences(words: list[_Word]) -> list[_Sentence]:
    sentences: list[_Sentence] = []
    current = _Sentence()
    for w in words:
        current.words.append(w)
        if _is_sentence_end(w.text):
            sentences.append(current)
            current = _Sentence()
    if current.words:
        sentences.append(current)
    return sentences


# --------------------------------------------------------------------------- #
# Line wrapping
# --------------------------------------------------------------------------- #
def _greedy_lines(text: str, max_cpl: int) -> list[str]:
    """Pack words into lines, each <= max_cpl (a single over-long word may
    exceed it, since it cannot be broken)."""
    lines: list[str] = []
    cur = ""
    for w in text.split():
        cand = w if not cur else cur + " " + w
        if cur and len(cand) > max_cpl:
            lines.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return lines


def _fits(text: str, opts: Opts) -> bool:
    """True if text fits within opts.max_lines lines of opts.max_cpl chars."""
    return len(_greedy_lines(text, opts.max_cpl)) <= opts.max_lines


def wrap_lines(text: str, max_cpl: int = MAX_CPL, max_lines: int = MAX_LINES) -> str:
    """Wrap text into at most ``max_lines`` lines of <= max_cpl chars.

    Guarantees no line exceeds ``max_cpl`` (except an unbreakable long word)
    as long as the text fits in ``max_lines`` lines. For a 2-line result it
    prefers a balanced split. Text that does not fit is packed greedily and the
    overflow is appended to the last line (callers should split first).
    """
    text = " ".join(text.split())
    if len(text) <= max_cpl:
        return text
    if max_lines <= 1:
        return text

    greedy = _greedy_lines(text, max_cpl)
    if len(greedy) > max_lines:
        # Doesn't fit; keep the first lines and dump the remainder on the last.
        head = greedy[: max_lines - 1]
        tail = " ".join(greedy[max_lines - 1 :])
        return "\n".join(head + [tail])

    if max_lines == 2 and len(greedy) == 2:
        # Try to balance the two lines without exceeding max_cpl.
        words = text.split(" ")
        best: tuple[int, list[str]] | None = None
        for i in range(1, len(words)):
            a, b = " ".join(words[:i]), " ".join(words[i:])
            if len(a) <= max_cpl and len(b) <= max_cpl:
                diff = abs(len(a) - len(b))
                if best is None or diff < best[0]:
                    best = (diff, [a, b])
        if best is not None:
            return "\n".join(best[1])
    return "\n".join(greedy)


# --------------------------------------------------------------------------- #
# Cue construction
# --------------------------------------------------------------------------- #
def _split_long_sentence(sent: _Sentence, opts: Opts) -> list[_Sentence]:
    """Split a sentence that exceeds one block into word-aligned chunks,
    preferring clause-punctuation break points near the limit."""
    chunks: list[_Sentence] = []
    cur = _Sentence()
    last_clause_break = -1  # index within cur.words after a clause punct
    for w in sent.words:
        prospective = _Sentence(cur.words + [w])
        if cur.words and not _fits(prospective.text, opts):
            # Prefer to break at the last clause punctuation if we have one.
            if 0 <= last_clause_break < len(cur.words) - 1:
                head = _Sentence(cur.words[: last_clause_break + 1])
                tail_words = cur.words[last_clause_break + 1 :]
                chunks.append(head)
                cur = _Sentence(list(tail_words))
            else:
                chunks.append(cur)
                cur = _Sentence()
            last_clause_break = -1
        cur.words.append(w)
        if w.text.endswith(_CLAUSE_PUNCT):
            last_clause_break = len(cur.words) - 1
    if cur.words:
        chunks.append(cur)
    return chunks


def _build_cues(sentences: list[_Sentence], opts: Opts) -> list[dict[str, Any]]:
    # 1) Ensure every unit fits within the line limits: split over-long ones.
    units: list[_Sentence] = []
    for s in sentences:
        if _fits(s.text, opts):
            units.append(s)
        else:
            units.extend(_split_long_sentence(s, opts))

    # 2) Mostly one sentence per cue; merge a SHORT trailing unit if both fit.
    cues: list[dict[str, Any]] = []
    i = 0
    while i < len(units):
        cur = units[i]
        merged_words = list(cur.words)
        j = i + 1
        while j < len(units):
            nxt = units[j]
            combined = _Sentence(merged_words + nxt.words).text
            combined_dur = nxt.end - merged_words[0].start
            short = len(nxt.text) <= opts.max_cpl  # "short" trailing sentence
            fits = _fits(combined, opts)
            paced = len(combined) <= opts.reading_cps * min(combined_dur, opts.max_duration)
            if short and fits and combined_dur <= opts.max_duration and paced:
                merged_words += nxt.words
                j += 1
            else:
                break
        s = _Sentence(merged_words)
        cues.append({"start": s.start, "end": s.end, "text": s.text})
        i = j
    return cues


# --------------------------------------------------------------------------- #
# Pacing / timing pass
# --------------------------------------------------------------------------- #
def _apply_pacing(cues: list[dict[str, Any]], opts: Opts) -> list[dict[str, Any]]:
    n = len(cues)
    for k, cue in enumerate(cues):
        chars = len(cue["text"].replace("\n", " "))
        start, end = cue["start"], cue["end"]
        if end < start:
            end = start
        # Minimum on-screen time for readability.
        min_needed = max(opts.min_duration, chars / opts.reading_cps)
        if end - start < min_needed:
            wanted = start + min_needed
            # Don't run into the next cue (leave a gap).
            if k + 1 < n:
                limit = cues[k + 1]["start"] - opts.min_gap
                end = min(wanted, max(start, limit))
            else:
                end = wanted
        # Cap maximum duration.
        if end - start > opts.max_duration:
            end = start + opts.max_duration
        cue["end"] = end
    # Final overlap cleanup (in case extension still collided).
    for k in range(n - 1):
        if cues[k]["end"] > cues[k + 1]["start"] - opts.min_gap:
            cues[k]["end"] = max(
                cues[k]["start"], cues[k + 1]["start"] - opts.min_gap
            )
    return cues


# --------------------------------------------------------------------------- #
# Fallback for segments without word timestamps
# --------------------------------------------------------------------------- #
def _fallback_cue(seg: dict[str, Any], opts: Opts) -> list[dict[str, Any]]:
    text = (seg.get("text") or "").strip()
    if not text:
        return []
    start, end = float(seg.get("start", 0.0)), float(seg.get("end", 0.0))
    if _fits(text, opts):
        return [{"start": start, "end": end, "text": wrap_lines(text, opts.max_cpl, opts.max_lines)}]
    # Char-split proportionally across the segment duration.
    words = text.split(" ")
    out: list[dict[str, Any]] = []
    chunk: list[str] = []
    total = len(text)
    span = max(end - start, 0.001)
    consumed = 0
    for w in words:
        cand = (" ".join(chunk + [w])).strip()
        if chunk and not _fits(cand, opts):
            chunk_text = " ".join(chunk)
            c_start = start + span * (consumed / total)
            consumed += len(chunk_text) + 1
            c_end = start + span * (consumed / total)
            out.append({"start": c_start, "end": c_end,
                        "text": wrap_lines(chunk_text, opts.max_cpl, opts.max_lines)})
            chunk = [w]
        else:
            chunk.append(w)
    if chunk:
        chunk_text = " ".join(chunk)
        c_start = start + span * (consumed / total)
        out.append({"start": c_start, "end": end,
                    "text": wrap_lines(chunk_text, opts.max_cpl, opts.max_lines)})
    return out


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def fit_cues(cues: list[dict[str, Any]], opts: Opts | None = None) -> list[dict[str, Any]]:
    """Fit each cue's text within the line limits, splitting over-capacity cues.

    Used for translated text, whose length differs from the source. A cue whose
    text does not fit within the line limits is split into multiple cues at word
    boundaries, with the original time span apportioned by character count.
    """
    opts = opts or Opts()
    out: list[dict[str, Any]] = []
    for c in cues:
        text = " ".join((c.get("text") or "").split())
        if not text:
            continue
        start, end = float(c["start"]), float(c["end"])
        if _fits(text, opts):
            out.append({"start": start, "end": end,
                        "text": wrap_lines(text, opts.max_cpl, opts.max_lines)})
            continue
        # Split into chunks that each fit the line limits, at word boundaries.
        chunks: list[str] = []
        cur = ""
        for w in text.split(" "):
            cand = (cur + " " + w).strip()
            if cur and not _fits(cand, opts):
                chunks.append(cur)
                cur = w
            else:
                cur = cand
        if cur:
            chunks.append(cur)
        span = max(end - start, 0.001)
        total = sum(len(ch) for ch in chunks) or 1
        t = start
        for n, ch in enumerate(chunks):
            e = end if n == len(chunks) - 1 else t + span * (len(ch) / total)
            out.append({"start": t, "end": e,
                        "text": wrap_lines(ch, opts.max_cpl, opts.max_lines)})
            t = e
    return _apply_pacing(out, opts)


def resegment(segments: list[dict[str, Any]], opts: Opts | None = None) -> list[dict[str, Any]]:
    """Rebuild Whisper segments into readable, sentence-aligned cues.

    Requires word-level timestamps (segment['words']). Segments lacking words
    fall back to char-based splitting of their own text.
    """
    opts = opts or Opts()
    has_words = any(seg.get("words") for seg in segments)
    if not has_words:
        out: list[dict[str, Any]] = []
        for seg in segments:
            out.extend(_fallback_cue(seg, opts))
        return _apply_pacing(out, opts)

    words = _flatten_words(segments)
    if not words:
        return []
    sentences = _split_sentences(words)
    cues = _build_cues(sentences, opts)
    cues = _apply_pacing(cues, opts)
    # Wrap text into <=max_lines lines.
    for cue in cues:
        cue["text"] = wrap_lines(cue["text"], opts.max_cpl, opts.max_lines)
    return cues
