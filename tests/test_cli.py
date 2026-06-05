import unittest
from pathlib import Path

from srtforge.cli import _translated_path


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


if __name__ == "__main__":
    unittest.main()
