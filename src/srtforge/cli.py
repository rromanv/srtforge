"""Command-line interface for srtforge."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from . import __version__
from .audio import FFmpegError, extract_audio
from .merge import burn_subtitles
from .segment import Opts, fit_cues, resegment
from .srt import segments_to_srt
from .transcribe import DEFAULT_MODEL, transcribe
from .transcript import correct_segments, initial_prompt_from_transcript, read_transcript
from .translate import DEFAULT_TRANSLATE_MODEL, TranslationError, translate_segments


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def _crf(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed <= 51:
        raise argparse.ArgumentTypeError("must be between 0 and 51")
    return parsed


def _translated_path(output: Path, language: str) -> Path:
    """Insert a language tag before the suffix: video.srt -> video.<lang>.srt."""
    tag = re.sub(r"[^\w.-]+", "-", language.strip().lower()).strip(".-_")
    tag = tag or "translated"
    return output.with_suffix(f".{tag}{output.suffix}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="srtforge",
        description="Generate SRT subtitles from a video using local Whisper (MLX).",
        epilog="Also: `srtforge merge VIDEO SUBS.srt -o OUT.mp4` to burn subtitles "
               "into a video. Run `srtforge merge -h` for details.",
    )
    p.add_argument("video", type=Path, help="Path to the input video (e.g. MP4).")
    p.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output .srt path (default: alongside the video).",
    )
    p.add_argument(
        "-m", "--model", default=DEFAULT_MODEL,
        help=f"Whisper model / HF repo (default: {DEFAULT_MODEL}).",
    )
    p.add_argument(
        "-l", "--language", default=None,
        help="Language code (e.g. en, es). Default: auto-detect.",
    )
    p.add_argument(
        "--word-timestamps", action="store_true",
        help="Compute word-level timestamps (slower).",
    )
    p.add_argument(
        "--transcript", type=Path, default=None, metavar="PATH",
        help="Path to a .txt or .md file with the correct transcript text. "
             "Corrects aligned word substitutions (timing still from audio).",
    )
    p.add_argument(
        "--no-resegment", dest="resegment", action="store_false",
        help="Disable sentence-aware re-cueing; keep raw Whisper segments.",
    )
    p.add_argument(
        "--max-line-length", type=_positive_int, default=Opts.max_cpl, metavar="N",
        help=f"Max characters per subtitle line (default: {Opts.max_cpl}).",
    )
    p.add_argument(
        "--max-lines", type=_positive_int, default=Opts.max_lines, metavar="N",
        help=f"Max lines per subtitle cue (default: {Opts.max_lines}).",
    )
    p.add_argument(
        "--reading-speed", type=_positive_float, default=Opts.reading_cps, metavar="CPS",
        help=f"Reading speed in characters/second (default: {Opts.reading_cps}).",
    )
    p.add_argument(
        "-t", "--translate", default=None, metavar="LANG",
        help="Also translate subtitles into LANG (e.g. es, Spanish, Japanese) "
             "using a local MLX LLM. Writes a second .<lang>.srt file.",
    )
    p.add_argument(
        "--translate-model", default=DEFAULT_TRANSLATE_MODEL,
        help=f"MLX LLM used for translation (default: {DEFAULT_TRANSLATE_MODEL}).",
    )
    p.add_argument("--version", action="version", version=f"srtforge {__version__}")
    return p


def _gen_main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    output = args.output or args.video.with_suffix(".srt")
    translating = bool(args.translate)
    total = 4 if translating else 3
    opts = Opts(
        max_cpl=args.max_line_length,
        max_lines=args.max_lines,
        reading_cps=args.reading_speed,
    )
    # Sentence-aware re-cueing needs word-level timestamps.
    need_words = args.word_timestamps or args.resegment
    wav_path = None
    try:
        transcript_text = read_transcript(args.transcript) if args.transcript else ""
        _log(f"[1/{total}] Extracting audio from {args.video} ...")
        wav_path = extract_audio(args.video)

        _log(f"[2/{total}] Transcribing with {args.model} (downloads model on first run) ...")
        segments = transcribe(
            wav_path,
            model=args.model,
            language=args.language,
            word_timestamps=need_words,
            initial_prompt=initial_prompt_from_transcript(transcript_text) or None,
        )

        if transcript_text:
            segments = correct_segments(segments, transcript_text)

        if args.resegment:
            segments = resegment(segments, opts)

        _log(f"[3/{total}] Writing {output} ...")
        srt_text = segments_to_srt(segments)
        output.write_text(srt_text, encoding="utf-8")
        if not srt_text:
            _log("Warning: no speech detected; wrote an empty SRT file.")
        else:
            _log(f"  {srt_text.count('-->')} subtitle entries written.")

        if translating:
            if not srt_text:
                _log("Skipping translation: no speech to translate.")
            else:
                t_out = _translated_path(output, args.translate)
                _log(
                    f"[4/{total}] Translating to {args.translate} with "
                    f"{args.translate_model} (downloads model on first run) ..."
                )
                t_segments = translate_segments(
                    segments,
                    target_language=args.translate,
                    model=args.translate_model,
                )
                if args.resegment:
                    # Translated text length differs; fit/split to line limits.
                    t_segments = fit_cues(t_segments, opts)
                t_out.write_text(segments_to_srt(t_segments), encoding="utf-8")
                _log(f"  Translated subtitles written to {t_out}")

        _log("Done.")
        return 0

    except (FFmpegError, FileNotFoundError, TranslationError) as e:
        _log(f"Error: {e}")
        return 1
    except KeyboardInterrupt:
        _log("Interrupted.")
        return 130
    finally:
        if wav_path is not None:
            wav_path.unlink(missing_ok=True)


def build_merge_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="srtforge merge",
        description="Burn a subtitle file permanently into a video (hard subs).",
    )
    p.add_argument("video", type=Path, help="Input video file.")
    p.add_argument("subtitles", type=Path, help="Subtitle file (.srt) to burn in.")
    p.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output video path (default: <video>.subbed<ext>).",
    )
    p.add_argument(
        "--crf", type=_crf, default=18, metavar="N",
        help="x264 quality, lower = better/larger (default: 18, ~visually lossless).",
    )
    p.add_argument(
        "--preset", default="slow",
        help="x264 speed/efficiency preset (default: slow).",
    )
    return p


def _merge_main(argv: list[str]) -> int:
    args = build_merge_parser().parse_args(argv)
    output = args.output or args.video.with_suffix(f".subbed{args.video.suffix}")
    try:
        _log(f"Burning {args.subtitles} into {args.video} (crf={args.crf}, preset={args.preset}) ...")
        burn_subtitles(
            args.video,
            args.subtitles,
            output,
            crf=args.crf,
            preset=args.preset,
        )
        _log(f"Done. Wrote {output}")
        return 0
    except (FFmpegError, FileNotFoundError) as e:
        _log(f"Error: {e}")
        return 1
    except KeyboardInterrupt:
        _log("Interrupted.")
        return 130


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "merge":
        return _merge_main(argv[1:])
    return _gen_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
