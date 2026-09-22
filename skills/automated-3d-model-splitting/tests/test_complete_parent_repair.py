import sys
import unittest
from pathlib import Path
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from split3mf.common import load_core_dependencies
load_core_dependencies()
from split3mf.complete_parent_repair import repair_complete_parent


class CompleteParentRepairTests(unittest.TestCase):
    def test_identity_seam_closes_without_source_movement(self):
        source = trimesh.creation.box()
        vertices = np.vstack((source.vertices, source.vertices[source.faces[0,0]]))
        faces = source.faces.copy()
        faces[0,0] = len(vertices)-1
        broken = trimesh.Trimesh(vertices, faces, process=False)
        self.assertFalse(broken.is_watertight)
        result, report = repair_complete_parent(broken)
        self.assertTrue(result.is_watertight)
        self.assertTrue(report['valid'])
        self.assertEqual(report['added_or_retriangulated_area_mm2'], 0.)
        np.testing.assert_array_equal(result.triangles, source.triangles)
        np.testing.assert_array_equal(broken.vertices, vertices)

    def test_broad_hole_is_not_converted_to_local_repair(self):
        source = trimesh.creation.box(extents=[10]*3)
        source.update_faces(np.arange(11))
        with self.assertRaises(ValueError):
            repair_complete_parent(source)

    def test_intact_parent_is_not_a_finished_body_claim(self):
        source = trimesh.creation.box()
        result, report = repair_complete_parent(source)
        np.testing.assert_array_equal(result.triangles, source.triangles)
        self.assertIn('assembly_not_validated', report['scope'])


if __name__ == '__main__':
    unittest.main()
