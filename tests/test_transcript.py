import tempfile
import unittest
from pathlib import Path

from srtforge.transcript import (
    MAX_INITIAL_PROMPT_CHARS,
    correct_segments,
    initial_prompt_from_transcript,
    read_transcript,
)


class ReadTranscriptTests(unittest.TestCase):
    def test_read_transcript_txt(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("Hello world this is a test transcript.")
            path = Path(f.name)
        result = read_transcript(path)
        self.assertEqual(result, "Hello world this is a test transcript.")
        path.unlink()

    def test_read_transcript_md_strips_formatting(self):
        md_content = (
            "# Heading One\n\n"
            "This is **bold** and *italic* text.\n\n"
            "A [link](https://example.com) and `code`.\n\n"
            "- bullet item\n"
            "> blockquote\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write(md_content)
            path = Path(f.name)
        result = read_transcript(path)
        self.assertNotIn("#", result)
        self.assertNotIn("**", result)
        self.assertNotIn("*", result)
        self.assertNotIn("[link](https://example.com)", result)
        self.assertIn("link", result)
        self.assertIn("bold", result)
        self.assertIn("italic", result)
        self.assertIn("bullet item", result)
        self.assertIn("blockquote", result)
        # Should be collapsed whitespace (no double spaces or newlines)
        self.assertNotIn("\n", result)
        self.assertNotIn("  ", result)
        path.unlink()

    def test_read_transcript_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            read_transcript(Path("/nonexistent/path/to/file.txt"))

    def test_read_transcript_empty_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("")
            path = Path(f.name)
        result = read_transcript(path)
        self.assertEqual(result, "")
        path.unlink()


class CorrectSegmentsTests(unittest.TestCase):
    def test_replaces_misspelled_word(self):
        segments = [
            {
                "start": 0.0,
                "end": 2.5,
                "text": "Helo world, this is a tset.",
            }
        ]
        reference = "Hello world, this is a test."
        result = correct_segments(segments, reference)
        self.assertIn("Hello", result[0]["text"])
        self.assertIn("test", result[0]["text"])
        self.assertEqual(result[0]["start"], 0.0)
        self.assertEqual(result[0]["end"], 2.5)

    def test_preserves_word_level_timestamps(self):
        segments = [
            {
                "start": 0.0,
                "end": 3.0,
                "text": "Helo wrld today.",
                "words": [
                    {"word": "Helo", "start": 0.0, "end": 0.5},
                    {"word": "wrld", "start": 0.5, "end": 1.0},
                    {"word": "today.", "start": 1.5, "end": 2.5},
                ],
            }
        ]
        reference = "Hello world today."
        result = correct_segments(segments, reference)
        words = result[0]["words"]
        # Timestamps must be strictly preserved
        self.assertEqual(words[0]["start"], 0.0)
        self.assertEqual(words[0]["end"], 0.5)
        self.assertEqual(words[1]["start"], 0.5)
        self.assertEqual(words[1]["end"], 1.0)
        self.assertEqual(words[2]["start"], 1.5)
        self.assertEqual(words[2]["end"], 2.5)
        # Word text should be corrected
        self.assertEqual(words[0]["word"], "Hello")
        self.assertEqual(words[1]["word"], "world")

    def test_mixed_word_and_text_segments_preserve_text_fallback(self):
        segments = [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "Helo world.",
                "words": [
                    {"word": "Helo", "start": 0.0, "end": 0.4},
                    {"word": "world.", "start": 0.4, "end": 0.9},
                ],
            },
            {
                "start": 1.0,
                "end": 2.0,
                "text": "This is tset.",
            },
        ]
        reference = "Hello world. This is test."

        result = correct_segments(segments, reference)

        self.assertEqual(result[0]["text"], "Hello world.")
        self.assertEqual(result[1]["text"], "This is test.")

    def test_returns_unchanged_when_reference_empty(self):
        segments = [
            {"start": 0.0, "end": 1.0, "text": "Some text here."}
        ]
        result = correct_segments(segments, "")
        self.assertEqual(result, segments)

    def test_returns_unchanged_for_unrelated_reference(self):
        segments = [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "The quick brown fox jumps over the lazy dog.",
            }
        ]
        # Completely unrelated reference — very low similarity
        reference = "Quantum mechanics describes the behavior of subatomic particles in probabilistic terms."
        result = correct_segments(segments, reference)
        # Should not crash and should return segments unchanged
        self.assertEqual(result[0]["start"], 0.0)
        self.assertEqual(result[0]["end"], 2.0)
        self.assertEqual(
            result[0]["text"],
            "The quick brown fox jumps over the lazy dog.",
        )


class InitialPromptTests(unittest.TestCase):
    def test_initial_prompt_from_transcript_returns_short_text_unchanged(self):
        self.assertEqual(
            initial_prompt_from_transcript("  Hello   world.  "),
            "Hello world.",
        )

    def test_initial_prompt_from_transcript_caps_long_text_at_word_boundary(self):
        text = " ".join(f"word{i}" for i in range(400))

        prompt = initial_prompt_from_transcript(text)

        self.assertLessEqual(len(prompt), MAX_INITIAL_PROMPT_CHARS)
        self.assertFalse(prompt.endswith(" "))
        self.assertNotIn("word399", prompt)


if __name__ == "__main__":
    unittest.main()
