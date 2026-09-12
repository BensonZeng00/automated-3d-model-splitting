from __future__ import annotations

import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from split3mf import common

common.load_core_dependencies()

from split3mf.common import CORE_NS, trimesh
from split3mf import inward
from split3mf.package_io import export_colored_parts_3mf, validate_colored_parts_3mf
from split3mf.recognition import recognition_colors_from_exterior


def tetrahedron() -> trimesh.Trimesh:
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    faces = np.array(
        [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
        dtype=np.int64,
    )
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


class ExteriorRecognitionRegressionTests(unittest.TestCase):
    def test_occluded_patch_fully_enclosed_by_visible_same_color_is_preserved(self) -> None:
        # Face 0 is an occluded triangle.  Its three edge-neighbors are visible
        # and use the same material, so visibility aliasing must not erase it.
        faces = np.array(
            [
                [0, 1, 2],
                [1, 0, 3],
                [2, 1, 4],
                [0, 2, 5],
            ],
            dtype=np.int64,
        )
        colors = ["DETAIL", "DETAIL", "DETAIL", "DETAIL"]
        visible = np.array([False, True, True, True], dtype=bool)

        recognized, report = recognition_colors_from_exterior(
            colors,
            visible,
            body_color_override="BODY",
            faces=faces,
        )

        self.assertEqual(recognized, colors)
        self.assertEqual(report["protected_enclosed_occluded_faces"], 1)
        self.assertEqual(report["reassigned_occluded_faces"], 0)

    def test_occluded_patch_touching_another_color_is_still_reassigned(self) -> None:
        faces = np.array(
            [
                [0, 1, 2],
                [1, 0, 3],
                [2, 1, 4],
                [0, 2, 5],
            ],
            dtype=np.int64,
        )
        colors = ["DETAIL", "DETAIL", "BODY", "DETAIL"]
        visible = np.array([False, True, True, True], dtype=bool)

        recognized, report = recognition_colors_from_exterior(
            colors,
            visible,
            body_color_override="BODY",
            faces=faces,
        )

        self.assertEqual(recognized[0], "BODY")
        self.assertEqual(report["protected_enclosed_occluded_faces"], 0)
        self.assertEqual(report["reassigned_occluded_faces"], 1)

    def test_occluded_boundary_face_with_two_visible_same_color_edges_is_preserved(self) -> None:
        # Real regression shape: the missed orange triangle has two visible
        # orange edge-neighbors and one occluded body-color edge-neighbor.
        faces = np.array(
            [
                [0, 1, 2],
                [1, 0, 3],
                [2, 1, 4],
                [0, 2, 5],
            ],
            dtype=np.int64,
        )
        colors = ["DETAIL", "DETAIL", "BODY", "DETAIL"]
        visible = np.array([False, True, False, True], dtype=bool)

        recognized, report = recognition_colors_from_exterior(
            colors,
            visible,
            body_color_override="BODY",
            faces=faces,
        )

        self.assertEqual(recognized[0], "DETAIL")
        self.assertEqual(report["protected_enclosed_occluded_faces"], 1)
        self.assertEqual(report["reassigned_occluded_faces"], 0)


class RecursiveCapColorRegressionTests(unittest.TestCase):
    def test_generated_boundary_uses_local_multicolor_surface_not_root_default(self) -> None:
        # The pending wrapper defaults to ORANGE, but the actual source faces
        # touching this parent-contact loop are BEIGE.
        local_faces = np.array(
            [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
            dtype=np.int64,
        )
        source_codes = ["BEIGE", "BEIGE", "BEIGE", "BEIGE"]

        edge_codes = inward.boundary_loop_source_color_codes(
            local_faces,
            source_codes,
            [0, 1, 2, 3],
            fallback_color_code="ORANGE",
        )

        self.assertEqual(edge_codes, ["BEIGE"] * 4)
        self.assertNotIn("ORANGE", edge_codes)

    def test_local_connector_generated_faces_inherit_boundary_material(self) -> None:
        boundary = np.asarray(
            [
                [-2.0, -2.0, 0.0],
                [2.0, -2.0, 0.0],
                [2.0, 2.0, 0.0],
                [-2.0, 2.0, 0.0],
            ],
            dtype=np.float64,
        )
        vertices = np.vstack(
            (
                boundary,
                np.asarray(
                    [
                        [-1.5, -1.5, -1.0],
                        [1.5, -1.5, -1.0],
                        [0.0, 0.0, -3.0],
                    ]
                ),
            )
        )
        generated_faces = [[0, 1, 4], [4, 5, 6]]

        codes = inward.local_connector_face_color_codes_from_boundary(
            vertices,
            generated_faces,
            boundary,
            ["BEIGE"] * 4,
            backing_face_count=1,
            fallback_color_code="ORANGE",
        )

        self.assertEqual(codes, ["BEIGE", "BEIGE"])
        self.assertNotIn("ORANGE", codes)


class BambuRecursivePaintRegressionTests(unittest.TestCase):
    def test_multicolor_pending_package_writes_and_validates_bambu_paint_color(self) -> None:
        mesh = tetrahedron()
        paint_tokens = ["0C", "1C", "8", "1C"]
        parts = [
            {
                "part_id": "P01_PENDING",
                "color_code": "0C",
                "color_name": "orange-root-multicolor-pending",
                "color_hex": "#E86825",
                "filament_slot_index": 2,
                "mesh": mesh,
                "face_color_hexes": [
                    "#E86825FF",
                    "#EDC148FF",
                    "#DDCBBAFF",
                    "#EDC148FF",
                ],
                "face_filament_slot_indices": [2, 3, 1, 3],
                "face_paint_color_tokens": paint_tokens,
                "annotation": {"state_role": "pending_subassembly"},
            }
        ]
        source_palette = ["#2B2B2F", "#DDCBBA", "#E86825", "#EDC148"]
        source_application = "BambuStudio-02.06.01.55"
        source_settings = {
            "filament_colour": source_palette,
            "printer_model": "Bambu Lab A1",
        }

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pending.3mf"
            export_colored_parts_3mf(
                output,
                parts,
                title="pending",
                source_application=source_application,
                source_filament_colors=source_palette,
                source_project_settings=source_settings,
                output_layout="separate-items",
            )
            validation = validate_colored_parts_3mf(
                output,
                parts,
                source_filament_colors=source_palette,
                source_application=source_application,
                source_project_settings=source_settings,
                output_layout="separate-items",
            )
            with zipfile.ZipFile(output) as archive:
                root = ET.fromstring(archive.read("3D/3dmodel.model"))
                model_settings = ET.fromstring(
                    archive.read("Metadata/model_settings.config")
                )

        triangles = root.findall(
            CORE_NS + "resources/" + CORE_NS + "object/" + CORE_NS + "mesh/"
            + CORE_NS + "triangles/" + CORE_NS + "triangle"
        )
        self.assertEqual(
            [triangle.attrib.get("paint_color") for triangle in triangles],
            paint_tokens,
        )
        self.assertTrue(validation["valid"], validation["errors"])
        part_metadata = {
            element.attrib.get("key"): element.attrib.get("value")
            for element in model_settings.find("object/part").findall("metadata")
        }
        self.assertEqual(part_metadata["extruder"], "3")


if __name__ == "__main__":
    unittest.main()
