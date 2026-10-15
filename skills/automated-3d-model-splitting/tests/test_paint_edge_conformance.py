from __future__ import annotations

import collections
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from split3mf import common

common.load_core_dependencies()

from split3mf.project import expand_vendor_paint_mesh


class PaintEdgeConformanceTests(unittest.TestCase):
    def test_default_neighbor_consumes_painted_shared_edge_midpoint(self) -> None:
        vertices = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2], [2, 1, 3]], dtype=np.int64)

        expanded_vertices, expanded_faces, _colors, diagnostics = (
            expand_vendor_paint_mesh(vertices, faces, ["401", "DEFAULT"])
        )

        midpoint_ids = np.flatnonzero(
            np.all(np.isclose(expanded_vertices, [1.0, 0.0, 0.0]), axis=1)
        )
        self.assertEqual(midpoint_ids.tolist(), [4])
        edge_counts: collections.Counter[tuple[int, int]] = collections.Counter()
        for triangle in expanded_faces:
            for start_id, end_id in zip(triangle, np.roll(triangle, -1)):
                edge_counts[tuple(sorted((int(start_id), int(end_id))))] += 1

        self.assertNotIn((1, 2), edge_counts)
        self.assertEqual(edge_counts[(1, 4)], 2)
        self.assertEqual(edge_counts[(2, 4)], 2)
        self.assertEqual(diagnostics["audited_subdivided_edges"], 1)
        self.assertEqual(diagnostics["audited_subdivided_segments"], 2)
        self.assertEqual(diagnostics["shared_edge_segment_consistency_ratio"], 1.0)
        self.assertEqual(diagnostics["source_face_ownership_ratio"], 1.0)

    def test_repeated_tokens_are_decoded_once(self) -> None:
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)

        _vertices, _faces, _colors, diagnostics = expand_vendor_paint_mesh(
            vertices, faces, ["DEFAULT", "DEFAULT"]
        )

        self.assertEqual(diagnostics["source_faces"], 2)
        self.assertEqual(diagnostics["unique_paint_tokens"], 1)
        self.assertEqual(diagnostics["paint_token_decode_success_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
