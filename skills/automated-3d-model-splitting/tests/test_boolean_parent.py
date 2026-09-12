"""A recessed source patch must not be replaced by a rim-only closure."""
import sys
import unittest
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from split3mf.common import load_core_dependencies
load_core_dependencies()
from split3mf.boolean_parent import build_complete_boolean_parent
from split3mf.domain import PlanarArcRetopologyConfig, PlanarArcRetopologyContext
from split3mf.local_connectors import _manifold64, _trimesh_from_manifold64


class CompleteBooleanParentTests(unittest.TestCase):
    def build(self, mesh):
        return build_complete_boolean_parent(
            mesh.vertices, mesh.faces, part_id='parent', cut_refs=[],
            interface_retopology=PlanarArcRetopologyContext(
                PlanarArcRetopologyConfig(preserve_confirmed_seam=True)),
        )

    def test_recessed_patch_survives_until_exact_subtraction(self):
        # The four-triangle top is depressed at its center. A flat rim closure
        # would introduce a 0.4 mm membrane above the child.
        v = np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0],
                      [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1],[0,0,.6]])
        f = [[0,2,1],[0,3,2],[0,1,5],[0,5,4],[1,2,6],[1,6,5],
             [2,3,7],[2,7,6],[3,0,4],[3,4,7],
             [4,5,8],[5,6,8],[6,7,8],[7,4,8]]
        source = trimesh.Trimesh(v,f,process=False)
        parent, record = self.build(source)
        np.testing.assert_array_equal(parent.vertices,source.vertices)
        np.testing.assert_array_equal(parent.faces,source.faces)
        self.assertEqual(record['cap_faces_added'],0)
        cutter = source.copy()
        cutter.vertices[:4,2] = .3
        body = _trimesh_from_manifold64(_manifold64(parent)-_manifold64(cutter))
        self.assertTrue(body.is_watertight)
        self.assertAlmostEqual(body.bounds[1,2],.3,places=8)
        self.assertAlmostEqual(body.volume,1.2,places=8)

    def test_open_recursive_input_is_rejected(self):
        source=trimesh.creation.box()
        source.update_faces(np.arange(len(source.faces)-1))
        with self.assertRaises(ValueError):
            self.build(source)


if __name__ == '__main__':
    unittest.main()
