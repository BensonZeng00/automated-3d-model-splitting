from pathlib import Path
import sys
import unittest
from collections import Counter
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from split3mf.rim_chord_repair import separate_source_chords


class RimChordRepairTests(unittest.TestCase):
    def test_generated_chord_is_separated_without_changing_source_or_rim(self):
        vertices = [[0., 0, 0], [1., 0, 0], [1., 1, 0], [0., 1, 0],
                    [.5, .2, -1], [.5, .8, -1]]
        source = [[0, 1, 2], [0, 2, 3]]
        faces = source + [[0, 2, 4], [2, 0, 5]]
        original = np.asarray(vertices).copy()
        record = separate_source_chords(vertices, faces, 2, [0, 1, 2, 3], [0, 0, -1], 1)
        self.assertEqual(record['split_source_chords'], 1)
        np.testing.assert_array_equal(vertices[:len(original)], original)
        self.assertEqual(faces[:2], source)
        edges = Counter(tuple(sorted((a, b))) for f in faces
                        for a, b in zip(f, f[1:] + f[:1]))
        self.assertEqual(edges[(0, 2)], 2)
        self.assertEqual(max(edges.values()), 2)
        self.assertLess(vertices[6][2], 0)


if __name__ == '__main__':
    unittest.main()
