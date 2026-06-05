import unittest

from srtforge.segment import Opts, resegment, wrap_lines


class SegmentTests(unittest.TestCase):
    def test_wrap_lines_balances_two_lines(self):
        wrapped = wrap_lines("one two three four five six", max_cpl=16, max_lines=2)
        self.assertEqual(wrapped, "one two three\nfour five six")

    def test_resegment_uses_word_timestamps(self):
        segments = [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "Hello world. Goodbye now.",
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.3},
                    {"word": "world.", "start": 0.3, "end": 0.7},
                    {"word": "Goodbye", "start": 1.0, "end": 1.4},
                    {"word": "now.", "start": 1.4, "end": 1.8},
                ],
            }
        ]

        cues = resegment(segments, Opts(max_cpl=20, max_lines=1))

        self.assertEqual([cue["text"] for cue in cues], ["Hello world.", "Goodbye now."])
        self.assertLessEqual(cues[0]["end"], cues[1]["start"])


if __name__ == "__main__":
    unittest.main()
