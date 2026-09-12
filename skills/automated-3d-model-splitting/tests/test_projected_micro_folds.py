from pathlib import Path
import sys
import unittest
from collections import Counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from split3mf.projected_micro_folds import crossing_pairs, isolate_micro_folds
from split3mf.print_tolerance import PrintTolerance, tolerance_scope


class ProjectedMicroFoldTests(unittest.TestCase):
    def setUp(self):
        self.xy = np.array([[0, 0], [10, 0], [10, 10], [5, 10],
                            [5.1, 9.9], [5, 9.9], [5.1, 10], [0, 10]])
        self.xyz = np.column_stack((self.xy, np.arange(len(self.xy)) * .01))

    def test_restoring_ears_preserves_every_original_boundary_edge(self):
        for reverse in (False, True):
            points = self.xyz[::-1] if reverse else self.xyz
            projected = self.xy[::-1] if reverse else self.xy
            original = points.copy()
            result = isolate_micro_folds(points, projected)
            self.assertIsNotNone(result)
            self.assertEqual(crossing_pairs(projected[list(result.retained_indices)]), [])
            self.assertTrue(result.evidence['accepted'])
            edges = Counter()
            ring = result.retained_indices
            for left, right in zip(ring, ring[1:] + ring[:1]):
                edges[tuple(sorted((left, right)))] += 1
            for ear in result.ear_indices:
                for left, right in zip(ear, ear[1:] + ear[:1]):
                    edges[tuple(sorted((left, right)))] += 1
            expected = {tuple(sorted((index, (index + 1) % len(points))))
                        for index in range(len(points))}
            self.assertEqual({edge for edge, count in edges.items() if count == 1}, expected)
            self.assertLessEqual(max(edges.values()), 2)
            np.testing.assert_array_equal(points, original)

    def test_strict_and_large_physical_folds_are_rejected(self):
        with tolerance_scope(PrintTolerance(micro_area_mm2=0)):
            self.assertIsNone(isolate_micro_folds(self.xyz, self.xy))
        self.assertIsNone(isolate_micro_folds(self.xyz * 100, self.xy * 100))

    def test_simple_boundary_is_unchanged(self):
        self.assertIsNone(isolate_micro_folds(self.xyz[:4], self.xy[:4]))


if __name__ == '__main__':
    unittest.main()
