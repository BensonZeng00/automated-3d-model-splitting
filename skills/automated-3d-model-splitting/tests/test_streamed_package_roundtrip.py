import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from split3mf.common import load_core_dependencies
load_core_dependencies()
from split3mf.package_io import export_colored_parts_3mf, load_colored_mesh_objects_3mf, validate_colored_parts_3mf


class StreamedPackageTests(unittest.TestCase):
    def test_reload_then_reexport_keeps_geometry_palette_slots_and_paint(self):
        palette = ['#000000FF', '#FFFFFFFF', '#FFFFFFFF']
        mesh = trimesh.creation.box()
        indices = [0,2]*6
        part = dict(part_id='box & name', mesh=mesh, color_hex=palette[2],
                    color_resolution_status='source_metadata', filament_slot_index=2,
                    face_color_hexes=[palette[i] for i in indices],
                    face_filament_slot_indices=indices,
                    face_paint_color_tokens=['4','0C']*6)
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory)/'a.3mf', Path(directory)/'b.3mf'
            export_colored_parts_3mf(a, [part], 'test & roundtrip', source_filament_colors=palette)
            loaded = load_colored_mesh_objects_3mf(a)
            self.assertEqual(loaded[0]['color_hex'], palette[2])
            self.assertEqual(loaded[0]['filament_slot_index'], 2)
            export_colored_parts_3mf(b, loaded, 'reloaded', source_filament_colors=palette)
            again = load_colored_mesh_objects_3mf(b)
            np.testing.assert_array_equal(again[0]['mesh'].triangles, mesh.triangles)
            self.assertEqual(again[0]['face_filament_slot_indices'], indices)
            self.assertEqual(again[0]['face_paint_color_tokens'], ['4','0C']*6)
            self.assertEqual(again[0]['object_color_index'], 2)
            self.assertEqual(again[0]['palette'], palette)
            self.assertTrue(validate_colored_parts_3mf(b, loaded, source_filament_colors=palette)['valid'])
            with zipfile.ZipFile(b) as archive:
                raw = archive.read('3D/3dmodel.model')
            self.assertNotIn(b'_split3mf_payload', raw)
            ET.fromstring(raw)

    def test_streaming_remains_deterministic_for_both_layouts(self):
        parts = [dict(part_id=f'part{i}', mesh=trimesh.creation.box(), color_hex='#FFFFFF') for i in range(2)]
        with tempfile.TemporaryDirectory() as directory:
            for layout in ('assembly', 'separate-items'):
                paths = [Path(directory)/f'{layout}{i}.3mf' for i in range(2)]
                for path in paths:
                    export_colored_parts_3mf(path, parts, 'test', output_layout=layout)
                self.assertEqual(paths[0].read_bytes(), paths[1].read_bytes())
                self.assertEqual(len(load_colored_mesh_objects_3mf(paths[0])), 2)


if __name__ == '__main__':
    unittest.main()
