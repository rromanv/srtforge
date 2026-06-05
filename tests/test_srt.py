import unittest

from srtforge.srt import segments_to_srt


class SrtTests(unittest.TestCase):
    def test_segments_to_srt_formats_timestamps(self):
        text = segments_to_srt(
            [{"start": 1.2345, "end": 62.0, "text": "Hello\nworld"}]
        )

        self.assertIn("1\n00:00:01,234 --> 00:01:02,000\nHello\nworld", text)

    def test_empty_text_is_skipped(self):
        self.assertEqual(segments_to_srt([{"text": "  "}]), "")


if __name__ == "__main__":
    unittest.main()
