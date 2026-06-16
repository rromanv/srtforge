import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from srtforge.cli import _gen_main, _translated_path, build_parser
from srtforge.transcript import MAX_INITIAL_PROMPT_CHARS


class CliTests(unittest.TestCase):
    def test_translated_path_slugs_language_for_filename(self):
        self.assertEqual(
            _translated_path(Path("video.srt"), "Brazilian Portuguese"),
            Path("video.brazilian-portuguese.srt"),
        )
        self.assertEqual(
            _translated_path(Path("video.srt"), "Portuguese/Brazil"),
            Path("video.portuguese-brazil.srt"),
        )

    def test_translated_path_has_empty_fallback(self):
        self.assertEqual(
            _translated_path(Path("video.srt"), "  "),
            Path("video.translated.srt"),
        )

    def test_transcript_flag_parsed_as_path(self):
        args = build_parser().parse_args(["video.mp4", "--transcript", "script.txt"])
        self.assertEqual(args.transcript, Path("script.txt"))

    def test_transcript_flag_defaults_to_none(self):
        args = build_parser().parse_args(["video.mp4"])
        self.assertIsNone(args.transcript)

    def test_transcript_initial_prompt_is_capped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "video.mp4"
            wav = root / "audio.wav"
            transcript = root / "script.txt"
            output = root / "out.srt"
            video.write_text("not real media", encoding="utf-8")
            wav.write_text("not real audio", encoding="utf-8")
            transcript.write_text(" ".join(f"word{i}" for i in range(400)), encoding="utf-8")

            with (
                patch("srtforge.cli.extract_audio", return_value=wav),
                patch(
                    "srtforge.cli.transcribe",
                    return_value=[{"start": 0.0, "end": 1.0, "text": "word0"}],
                ) as transcribe,
            ):
                status = _gen_main(
                    [str(video), "-o", str(output), "--transcript", str(transcript)]
                )

        self.assertEqual(status, 0)
        prompt = transcribe.call_args.kwargs["initial_prompt"]
        self.assertIsNotNone(prompt)
        self.assertLessEqual(len(prompt), MAX_INITIAL_PROMPT_CHARS)


if __name__ == "__main__":
    unittest.main()
