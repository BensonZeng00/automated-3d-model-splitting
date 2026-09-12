from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from split3mf import common

common.load_core_dependencies()

from split3mf.mesh import remove_new_redundant_boolean_micro_shells


def near_corner_sliver() -> common.trimesh.Trimesh:
    vertices = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [0.99, 1.0, 1.0],
            [1.0, 0.99, 1.0],
            [1.0, 1.0, 0.9999],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
        dtype=np.int64,
    )
    return common.trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


class BooleanMicroShellCleanupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.body = common.trimesh.creation.box(extents=(2.0, 2.0, 2.0))

    def test_removes_new_covered_zero_thickness_shell(self) -> None:
        candidate = common.trimesh.util.concatenate(
            (self.body, near_corner_sliver())
        )

        cleaned, record = remove_new_redundant_boolean_micro_shells(
            candidate,
            self.body,
        )

        self.assertTrue(record["applied"])
        self.assertEqual(record["removed_component_count"], 1)
        self.assertEqual(record["removed_face_count"], 4)
        self.assertEqual(len(cleaned.faces), len(self.body.faces))
        self.assertTrue(cleaned.is_watertight)
        self.assertTrue(cleaned.is_winding_consistent)

    def test_preserves_source_supported_micro_shell(self) -> None:
        reference = common.trimesh.util.concatenate(
            (self.body, near_corner_sliver())
        )

        cleaned, record = remove_new_redundant_boolean_micro_shells(
            reference,
            reference,
        )

        self.assertFalse(record["applied"])
        self.assertEqual(len(cleaned.faces), len(reference.faces))

    def test_optional_face_provenance_tracks_only_retained_triangles(self) -> None:
        candidate = common.trimesh.util.concatenate((self.body, near_corner_sliver()))
        cleaned, record = remove_new_redundant_boolean_micro_shells(
            candidate, self.body, include_face_indices=True)
        np.testing.assert_array_equal(cleaned.triangles,
            candidate.triangles[np.asarray(record['retained_face_indices'])])
        self.assertNotIn('retained_face_indices',cleaned.metadata['boolean_redundant_micro_shell_cleanup'])

    def test_preserves_new_shell_that_is_not_covered_by_body(self) -> None:
        remote = near_corner_sliver()
        remote.apply_translation((2.0, 0.0, 0.0))
        candidate = common.trimesh.util.concatenate((self.body, remote))

        cleaned, record = remove_new_redundant_boolean_micro_shells(
            candidate,
            self.body,
        )

        self.assertFalse(record["applied"])
        self.assertEqual(len(cleaned.faces), len(candidate.faces))


if __name__ == "__main__":
    unittest.main()
