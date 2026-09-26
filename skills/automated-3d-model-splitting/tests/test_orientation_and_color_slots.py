from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree as ET
import zipfile

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from split3mf import common

common.load_core_dependencies()

from split3mf.common import CORE_NS, MATERIAL_NS, VERSION, trimesh
from split3mf.part_geometry import finalize_recursive_colored_mesh
from split3mf.mesh import orient_mesh_faces_consistently
from split3mf.package_io import export_colored_parts_3mf, validate_colored_parts_3mf
from split3mf.validation import finalize_large_partition_mesh, validate_mesh_in_memory


def tetrahedron() -> trimesh.Trimesh:
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


class OrientationRepairTests(unittest.TestCase):
    def test_nested_negative_shell_is_accepted_as_cavity(self) -> None:
        outer = trimesh.creation.box(extents=(8.0, 8.0, 8.0))
        cavity = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
        cavity.invert()
        mesh = trimesh.util.concatenate((outer, cavity))

        validation = validate_mesh_in_memory(mesh)

        self.assertTrue(validation["watertight"])
        self.assertTrue(validation["winding_consistent"])
        self.assertEqual(validation["inward_closed_components"], 0)
        orientation = validation["closed_component_orientation"]
        self.assertEqual(orientation["negative_signed_component_count"], 1)
        self.assertEqual(sorted(orientation["component_containment_depths"]), [0, 1])
        self.assertTrue(orientation["all_closed_components_outward"])

    def test_outward_orientation_is_checked_per_closed_component(self) -> None:
        dominant = trimesh.creation.box(extents=(8.0, 8.0, 8.0))
        fragment = trimesh.creation.box(extents=(0.4, 0.6, 0.8))
        fragment.apply_translation((0.0, -5.0, -3.0))
        fragment.invert()
        mesh = trimesh.util.concatenate((dominant, fragment))
        vertices_before = np.asarray(mesh.vertices).copy()
        memberships_before = np.sort(np.asarray(mesh.faces), axis=1)

        before = validate_mesh_in_memory(mesh)
        self.assertTrue(before["watertight"])
        self.assertTrue(before["winding_consistent"])
        self.assertEqual(before["inward_closed_components"], 1)

        record = orient_mesh_faces_consistently(mesh)
        after = validate_mesh_in_memory(mesh)

        self.assertEqual(record["inverted_closed_component_count"], 1)
        self.assertEqual(after["inward_closed_components"], 0)
        self.assertTrue(after["all_closed_components_outward"])
        np.testing.assert_array_equal(mesh.vertices, vertices_before)
        np.testing.assert_array_equal(
            np.sort(mesh.faces, axis=1),
            memberships_before,
        )

    def test_recursive_finalizer_protects_thin_source_triangle(self) -> None:
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
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        self.assertLess(np.count_nonzero(mesh.nondegenerate_faces()), 4)

        repaired = finalize_recursive_colored_mesh(
            mesh,
            ["8"] * 4,
            [1] * 4,
            "8",
            protected_source_face_count=4,
        )

        self.assertEqual(len(repaired.faces), 4)
        self.assertEqual(repaired.metadata["protected_source_face_count"], 4)

    def test_recursive_finalizer_preserves_coordinate_coincident_source_faces(self) -> None:
        first = tetrahedron()
        vertices = np.vstack(
            [np.asarray(first.vertices), np.asarray(first.vertices)]
        )
        faces = np.vstack(
            [np.asarray(first.faces), np.asarray(first.faces) + 4]
        )
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            process=False,
        )

        repaired = finalize_recursive_colored_mesh(
            mesh,
            ["8"] * 8,
            [1] * 8,
            "8",
            protected_source_face_count=8,
        )

        self.assertEqual(len(repaired.faces), 8)
        self.assertEqual(len(repaired.vertices), 8)
        self.assertEqual(len(repaired.metadata["face_color_codes"]), 8)

    def test_recursive_finalizer_preserves_source_identity_seam(self) -> None:
        vertices = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,0],[0,1,0]], float)
        faces = np.array([[0,2,1],[0,1,3],[1,2,3],[5,4,3]], int)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        result = finalize_recursive_colored_mesh(
            mesh, ["8"] * 4, [1] * 4, "8", protected_source_face_count=4)
        np.testing.assert_array_equal(result.vertices, vertices)
        np.testing.assert_array_equal(result.faces, faces)
        self.assertEqual(validate_mesh_in_memory(result)["open_edges"], 6)
        self.assertEqual(result.metadata["source_geometry_mutation"], "none")

    def test_recursive_finalizer_preserves_source_hole(self) -> None:
        mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
        mesh.update_faces(np.asarray(mesh.face_normals[:, 2] < 0.9, dtype=bool))
        mesh.remove_unreferenced_vertices()
        faces_before = np.asarray(mesh.faces).copy()
        result = finalize_recursive_colored_mesh(
            mesh, ["8"] * len(mesh.faces), [1] * len(mesh.faces), "1C",
            protected_source_face_count=len(mesh.faces))
        np.testing.assert_array_equal(result.faces, faces_before)
        self.assertEqual(validate_mesh_in_memory(result)["open_edges"], 4)
        self.assertEqual(set(result.metadata["face_color_codes"]), {"8"})

    def test_large_partition_records_winding_without_repair(self) -> None:
        mesh = tetrahedron(); mesh.faces[0] = mesh.faces[0][::-1]
        vertices_before = np.asarray(mesh.vertices).copy(); faces_before = np.asarray(mesh.faces).copy()
        result = finalize_large_partition_mesh(mesh)
        np.testing.assert_array_equal(result.vertices, vertices_before)
        np.testing.assert_array_equal(result.faces, faces_before)
        self.assertFalse(result.metadata["source_topology_diagnostics"]["winding_consistent"])
        self.assertEqual(result.metadata["source_geometry_mutation"], "none")

    def test_large_partition_preserves_open_boundary(self) -> None:
        mesh = tetrahedron(); mesh.update_faces(np.array([True, True, True, False]))
        faces_before = np.asarray(mesh.faces).copy()
        result = finalize_large_partition_mesh(mesh)
        np.testing.assert_array_equal(result.faces, faces_before)
        self.assertEqual(validate_mesh_in_memory(result)["open_edges"], 3)


class SourceColorSlotTests(unittest.TestCase):
    def test_export_preserves_source_palette_order_and_object_slots(self) -> None:
        source_palette = ["#FFFFFF", "#000000", "#C12E1F"]
        part_specs = [
            ("black", "#000000", 1),
            ("red", "#C12E1F", 2),
            ("white", "#FFFFFF", 0),
        ]
        parts = []
        for name, color, slot in part_specs:
            parts.append(
                {
                    "part_id": name,
                    "color_code": name,
                    "color_name": name,
                    "color_hex": color,
                    "filament_slot_index": slot,
                    "color_resolution_status": "source_metadata",
                    "mesh": tetrahedron(),
                    "annotation": {},
                }
            )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "slot_order.3mf"
            summary = export_colored_parts_3mf(
                output,
                parts,
                title="slot order",
                source_filament_colors=source_palette,
            )
            validation = validate_colored_parts_3mf(
                output,
                parts,
                source_filament_colors=source_palette,
            )
            with zipfile.ZipFile(output) as archive:
                root = ET.fromstring(archive.read("3D/3dmodel.model"))

        color_group = root.find(CORE_NS + "resources").find(MATERIAL_NS + "colorgroup")
        colors = [element.attrib["color"] for element in color_group.findall(MATERIAL_NS + "color")]
        objects = [
            element
            for element in root.find(CORE_NS + "resources").findall(CORE_NS + "object")
            if element.find(CORE_NS + "mesh") is not None
        ]
        assembly = [
            element
            for element in root.find(CORE_NS + "resources").findall(CORE_NS + "object")
            if element.find(CORE_NS + "components") is not None
        ]
        self.assertEqual(colors, ["#FFFFFFFF", "#000000FF", "#C12E1FFF"])
        self.assertEqual([int(element.attrib["pindex"]) for element in objects], [1, 2, 0])
        self.assertEqual(len(assembly), 1)
        self.assertEqual(len(root.find(CORE_NS + "build").findall(CORE_NS + "item")), 1)
        self.assertEqual(summary["source_filament_colors"], colors)
        self.assertTrue(summary["source_filament_palette_preserved"])
        self.assertTrue(validation["valid"], validation["errors"])

    def test_duplicate_source_colors_keep_distinct_slot_indices(self) -> None:
        parts = [
            {
                "part_id": "second_white_slot",
                "color_code": "white",
                "color_name": "white",
                "color_hex": "#FFFFFF",
                "filament_slot_index": 1,
                "color_resolution_status": "source_metadata",
                "mesh": tetrahedron(),
                "annotation": {},
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "duplicate_slots.3mf"
            export_colored_parts_3mf(
                output,
                parts,
                title="duplicate slots",
                source_filament_colors=["#FFFFFF", "#FFFFFF", "#000000"],
            )
            with zipfile.ZipFile(output) as archive:
                root = ET.fromstring(archive.read("3D/3dmodel.model"))
        object_element = root.find(CORE_NS + "resources").find(CORE_NS + "object")
        self.assertEqual(object_element.attrib["pindex"], "1")

    def test_bambu_source_writes_complete_project_metadata_and_extruders(self) -> None:
        source_palette = ["#FFFFFF", "#000000", "#C12E1F"]
        part_specs = [
            ("black", "#000000", 1, "source_metadata"),
            ("red", "#C12E1F", 2, "source_metadata"),
            ("white", "#FFFFFF", 0, "source_metadata"),
            ("yellow_override", "#F5C542", None, "explicit_override"),
        ]
        parts = [
            {
                "part_id": name,
                "color_code": name,
                "color_name": name,
                "color_hex": color,
                "filament_slot_index": slot,
                "color_resolution_status": resolution,
                "mesh": tetrahedron(),
                "annotation": {},
            }
            for name, color, slot, resolution in part_specs
        ]
        source_application = "BambuStudio-02.06.01.55"
        source_project_settings = {
            "filament_colour": source_palette,
            "printer_model": "Bambu Lab A1",
            "layer_height": "0.2",
            "_parser_only": {"must": "not be serialized"},
        }

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bambu_project.3mf"
            summary = export_colored_parts_3mf(
                output,
                parts,
                title="Bambu project",
                source_application=source_application,
                source_filament_colors=source_palette,
                source_project_settings=source_project_settings,
            )
            validation = validate_colored_parts_3mf(
                output,
                parts,
                source_filament_colors=source_palette,
                source_application=source_application,
                source_project_settings=source_project_settings,
            )
            with zipfile.ZipFile(output) as archive:
                entries = set(archive.namelist())
                project_config = json.loads(archive.read("Metadata/project_settings.config"))
                model_config = ET.fromstring(archive.read("Metadata/model_settings.config"))

        self.assertTrue(summary["bambu_project_compatible"])
        self.assertTrue(validation["valid"], validation["errors"])
        self.assertTrue(validation["bambu_project_metadata_valid"])
        self.assertIn("Metadata/project_settings.config", entries)
        self.assertIn("Metadata/model_settings.config", entries)
        self.assertIn("Metadata/slice_info.config", entries)
        self.assertEqual(project_config["filament_colour"], source_palette + ["#F5C542"])
        self.assertNotIn("_parser_only", project_config)
        extruders = []
        for object_element in model_config.findall("object"):
            for part_element in object_element.findall("part"):
                metadata = {
                    element.attrib.get("key"): element.attrib.get("value")
                    for element in part_element.findall("metadata")
                    if "key" in element.attrib
                }
                extruders.append(int(metadata["extruder"]))
        self.assertEqual(len(model_config.findall("object")), 1)
        self.assertEqual(extruders, [2, 3, 1, 4])

    def test_separate_items_layout_remains_available(self) -> None:
        parts = [
            {
                "part_id": name,
                "color_code": name,
                "color_name": name,
                "color_hex": color,
                "filament_slot_index": slot,
                "color_resolution_status": "source_metadata",
                "mesh": tetrahedron(),
                "annotation": {},
            }
            for name, color, slot in (("white", "#FFFFFF", 0), ("black", "#000000", 1))
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "separate.3mf"
            export_colored_parts_3mf(
                output,
                parts,
                title="separate",
                source_filament_colors=["#FFFFFF", "#000000"],
                output_layout="separate-items",
            )
            validation = validate_colored_parts_3mf(
                output,
                parts,
                source_filament_colors=["#FFFFFF", "#000000"],
                output_layout="separate-items",
            )
            with zipfile.ZipFile(output) as archive:
                root = ET.fromstring(archive.read("3D/3dmodel.model"))
        self.assertTrue(validation["valid"], validation["errors"])
        self.assertEqual(len(root.find(CORE_NS + "build").findall(CORE_NS + "item")), 2)
        self.assertFalse(
            any(
                element.find(CORE_NS + "components") is not None
                for element in root.find(CORE_NS + "resources").findall(CORE_NS + "object")
            )
        )

    def test_bambu_marker_is_not_emitted_without_project_settings(self) -> None:
        parts = [
            {
                "part_id": "black",
                "color_code": "black",
                "color_name": "black",
                "color_hex": "#000000",
                "filament_slot_index": 0,
                "color_resolution_status": "source_metadata",
                "mesh": tetrahedron(),
                "annotation": {},
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "generic.3mf"
            summary = export_colored_parts_3mf(
                output,
                parts,
                title="generic",
                source_application="BambuStudio-02.06.01.55",
                source_filament_colors=["#000000"],
            )
            with zipfile.ZipFile(output) as archive:
                entries = set(archive.namelist())
                root = ET.fromstring(archive.read("3D/3dmodel.model"))
        metadata = {
            element.attrib.get("name"): element.text
            for element in root.findall(CORE_NS + "metadata")
        }
        self.assertFalse(summary["bambu_project_compatible"])
        self.assertEqual(
            metadata["Application"],
            f"automated-3d-model-splitting {VERSION}",
        )
        self.assertNotIn("BambuStudio:3mfVersion", metadata)
        self.assertNotIn("Metadata/project_settings.config", entries)


if __name__ == "__main__":
    unittest.main()
