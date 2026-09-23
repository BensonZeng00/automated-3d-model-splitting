from pathlib import Path
import sys
import unittest

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from split3mf import common
common.load_core_dependencies()
from split3mf.recognition import make_component_from_global_faces, triangle_areas
from split3mf.semantic_partition import apply_semantic_partitions


class SemanticPartitionTests(unittest.TestCase):
    def fixture(self):
        mesh = trimesh.creation.box(extents=[2.0, 2.0, 4.0])
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        component = make_component_from_global_faces(
            vertices, faces, triangle_areas(vertices, faces),
            np.arange(len(faces)), "white",
        )
        return vertices, faces, component

    def test_reviewed_plane_splits_one_material_without_changing_faces(self):
        vertices, faces, component = self.fixture()
        original = faces.copy()
        result, records = apply_semantic_partitions(
            vertices, faces, [component], [{
                "part_index": 1,
                "label": "head/body boundary",
                "origin": [0, 0, 0],
                "normal": [0, 0, 1],
                "confidence_score": 1.0,
                "user_confirmed": True,
            }], 0.65,
        )
        self.assertEqual(len(result), 2)
        self.assertEqual([part.color_code for part in result], ["white", "white"])
        self.assertEqual(sum(part.face_count for part in result), len(faces))
        np.testing.assert_array_equal(faces, original)
        self.assertTrue(records["applied"][0]["source_material_preserved"])

    def test_unreviewed_plane_is_rejected(self):
        vertices, faces, component = self.fixture()
        result, records = apply_semantic_partitions(
            vertices, faces, [component], [{
                "part_index": 1, "origin": [0, 0, 0], "normal": [0, 0, 1],
                "confidence_score": 0.2,
                "user_confirmed": True,
            }], 0.65,
        )
        self.assertEqual(result, [component])
        self.assertEqual(records["rejected"][0]["reason"], "confidence_below_threshold")

    def test_high_confidence_without_explicit_confirmation_is_rejected(self):
        vertices, faces, component = self.fixture()
        result, records = apply_semantic_partitions(
            vertices, faces, [component], [{
                "part_index": 1, "origin": [0, 0, 0], "normal": [0, 0, 1],
                "confidence_score": 1.0,
            }], 0.65,
        )
        self.assertEqual(result, [component])
        self.assertEqual(records["rejected"][0]["reason"], "user_confirmation_required")

    def test_existing_same_material_islands_do_not_block_one_reviewed_cut(self):
        first = trimesh.creation.box(extents=[2.0, 2.0, 4.0])
        second = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        second.apply_translation([4.0, 0.0, -1.0])
        mesh = trimesh.util.concatenate((first, second))
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        component = make_component_from_global_faces(
            vertices, faces, triangle_areas(vertices, faces),
            np.arange(len(faces)), "white",
        )
        result, records = apply_semantic_partitions(
            vertices, faces, [component], [{
                "part_index": 1, "label": "reviewed boundary",
                "origin": [0, 0, 0], "normal": [0, 0, 1],
                "confidence_score": 1.0,
                "user_confirmed": True,
            }], 0.65,
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(records["applied"][0]["source_region_count"], 2)
        self.assertEqual(sum(records["applied"][0]["partition_region_counts"]), 3)


if __name__ == "__main__":
    unittest.main()
