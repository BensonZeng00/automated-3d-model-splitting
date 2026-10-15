from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from split3mf.common import Component
from split3mf.source_face_ownership import (
    color_assigned_faces_as_part, complete_component_face_ownership,
)


class SourceFaceOwnershipTests(unittest.TestCase):
    def test_assigned_faces_inherit_host_color_without_repainting_recognized_faces(self):
        source = np.array(["skin", "white", "white", "black"], dtype=object)
        completed_ids = np.array([10, 11, 12, 13])
        recognized_ids = np.array([10, 13])
        output, changed = color_assigned_faces_as_part(
            source, completed_ids, recognized_ids, "skin-color-longer-than-input",
        )
        self.assertEqual(output.tolist(), [
            "skin", "skin-color-longer-than-input",
            "skin-color-longer-than-input", "black",
        ])
        self.assertEqual(changed, 2)
        self.assertEqual(source.tolist(), ["skin", "white", "white", "black"])

    def test_touching_excluded_paint_keeps_original_faces_in_host_part(self):
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 0],
            [3, 0, 0], [4, 0, 0], [3, 1, 0],
        ], dtype=float)
        faces = np.array([
            [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4], [5, 6, 7],
        ], dtype=np.int64)
        component = Component(
            "pink", np.array([0, 1, 3]), 3, 0.75,
            vertices[:5].min(axis=0), vertices[:5].max(axis=0),
            vertices[:5].mean(axis=0),
        )
        completed, report = complete_component_face_ownership(vertices, faces, [component])
        np.testing.assert_array_equal(completed[0].global_faces, [0, 1, 2, 3, 4])
        self.assertEqual(report["assigned_faces"], 2)
        self.assertEqual(report["unresolved_faces"], 0)
        self.assertEqual(
            {patch["assignment_method"] for patch in report["patches"]},
            {"unique_shared_vertices", "nearest_source_part"},
        )
        self.assertEqual(faces[2].tolist(), [2, 3, 4])


if __name__ == "__main__":
    unittest.main()
