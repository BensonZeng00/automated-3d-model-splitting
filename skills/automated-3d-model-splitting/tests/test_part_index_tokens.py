"""Part selection must retain every ID across documented separators."""

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from split3mf.assembly import parse_part_index_tokens


class PartIndexTokenTests(unittest.TestCase):
    def test_documented_separators_preserve_all_selected_parts(self):
        for separator in (",", " ", "\t", "\n", ", \t"):
            with self.subTest(separator=repr(separator)):
                self.assertEqual(
                    parse_part_index_tokens(separator.join(("P01", "p02", "3", "P01"))),
                    {1, 2, 3},
                )

    def test_empty_selection(self):
        for value in (None, "", " , \t\n"):
            with self.subTest(value=value):
                self.assertEqual(parse_part_index_tokens(value), set())
