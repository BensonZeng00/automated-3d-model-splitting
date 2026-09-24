import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from split3mf.contour_simplification import simplify_closed_contour


class ClosedContourSimplificationTests(unittest.TestCase):
    def test_removes_sub_tolerance_jagged_detail_and_preserves_winding(self):
        x = np.linspace(0.0, 20.0, 1001)
        top = np.column_stack((x, 10.0 + 0.03 * np.sin(x * 30.0)))
        right = np.array([[20.0, 0.0]])
        bottom = np.column_stack((x[::-1], np.zeros_like(x)))
        left = np.empty((0, 2))
        contour = np.vstack((top, right, bottom, left))

        retained = simplify_closed_contour(contour, 0.1)
        simplified = contour[retained]

        self.assertLess(len(simplified), len(contour) // 20)
        source_area = np.sum(
            contour[:, 0] * np.roll(contour[:, 1], -1)
            - contour[:, 1] * np.roll(contour[:, 0], -1)
        )
        result_area = np.sum(
            simplified[:, 0] * np.roll(simplified[:, 1], -1)
            - simplified[:, 1] * np.roll(simplified[:, 0], -1)
        )
        self.assertGreater(source_area * result_area, 0.0)

    def test_zero_tolerance_retains_every_vertex(self):
        contour = np.array([[0., 0.], [2., 0.], [2., 2.], [0., 2.]])
        np.testing.assert_array_equal(
            simplify_closed_contour(contour, 0.0), np.arange(4)
        )


if __name__ == "__main__":
    unittest.main()
