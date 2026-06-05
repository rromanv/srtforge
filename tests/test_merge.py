import unittest
from pathlib import Path

from srtforge.merge import _escape_sub_path


class MergeTests(unittest.TestCase):
    def test_subtitle_path_is_escaped_for_filtergraph(self):
        self.assertEqual(
            _escape_sub_path(Path("John's clip, [final].srt")),
            r"John\'s clip\, \[final\].srt",
        )


if __name__ == "__main__":
    unittest.main()
