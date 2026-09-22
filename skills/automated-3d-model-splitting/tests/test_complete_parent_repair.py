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
    def test_identity_seam_is_not_repaired_implicitly(self):
        source = trimesh.creation.box()
        vertices = np.vstack((source.vertices, source.vertices[source.faces[0,0]]))
        faces = source.faces.copy(); faces[0,0] = len(vertices)-1
        broken = trimesh.Trimesh(vertices, faces, process=False)
        with self.assertRaises(ValueError):
            repair_complete_parent(broken)
        np.testing.assert_array_equal(broken.vertices, vertices)
        np.testing.assert_array_equal(broken.faces, faces)

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
