from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from split3mf import common
common.load_core_dependencies()
from split3mf.recognition import summarize_components


class ConfirmedRegionClassificationTests(unittest.TestCase):
    def test_confirmed_part_and_noise_override_face_count(self):
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
            [2, 0, 0], [3, 0, 0], [2, 1, 0],
            [4, 0, 0], [5, 0, 0], [4, 1, 0],
            [6, 0, 0], [7, 0, 0], [6, 1, 0],
        ], dtype=float)
        faces = np.arange(12, dtype=np.int64).reshape(4, 3)
        components, ignored = summarize_components(
            vertices, faces, ["A", "B", "B", "C"],
            [np.array([0]), np.array([1, 2]), np.array([3])],
            min_faces=2,
            retained_source_min_face_ids={0},
            excluded_source_min_face_ids={1},
        )
        self.assertEqual([component.global_faces.tolist() for component in components], [[0]])
        self.assertEqual(len(ignored), 2)


if __name__ == "__main__":
    unittest.main()
