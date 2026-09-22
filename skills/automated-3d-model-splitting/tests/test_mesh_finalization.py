from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from split3mf import common

common.load_core_dependencies()

from split3mf.common import trimesh
from split3mf.mesh_finalization import (
    SourcePreservingFinalizationPolicy,
    SourcePreservingMeshFinalizer,
    finalize_source_preserving_mesh,
)
from split3mf.validation import validate_mesh_in_memory


def thin_closed_tetrahedron() -> trimesh.Trimesh:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1e-10, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
        dtype=np.int64,
    )
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def ordinary_closed_tetrahedron() -> trimesh.Trimesh:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
        dtype=np.int64,
    )
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


class SourcePreservingMeshFinalizerTests(unittest.TestCase):
    def test_preserves_nonzero_vendor_sliver_instead_of_opening_three_edges(self) -> None:
        source = thin_closed_tetrahedron()
        self.assertLess(np.count_nonzero(source.nondegenerate_faces()), 4)

        finalized = finalize_source_preserving_mesh(
            source,
            protected_source_face_count=4,
        )
        validation = validate_mesh_in_memory(finalized)

        self.assertEqual(len(finalized.faces), 4)
        self.assertEqual(validation["open_edges"], 0)
        self.assertEqual(validation["over_shared_edges"], 0)
        self.assertEqual(
            finalized.metadata["source_preserving_finalization"]["source_faces_missing"],
            0,
        )

    def test_open_source_is_reported_without_repair(self) -> None:
        source = ordinary_closed_tetrahedron()
        open_mesh = trimesh.Trimesh(
            vertices=np.asarray(source.vertices), faces=np.asarray(source.faces)[1:], process=False)
        before_faces = np.asarray(open_mesh.faces).copy()
        result = SourcePreservingMeshFinalizer.finalize(
            open_mesh, SourcePreservingFinalizationPolicy(protected_source_face_count=3))
        np.testing.assert_array_equal(result.mesh.faces, before_faces)
        self.assertEqual(validate_mesh_in_memory(result.mesh)["open_edges"], 3)
        self.assertEqual(result.audit["source_geometry_mutation"], "none")


if __name__ == "__main__":
    unittest.main()
