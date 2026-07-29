from __future__ import annotations

import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile
import inspect
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from split3mf import common

common.load_core_dependencies()

from split3mf.assembly import (
    advance_strict_recursive_state,
    build_recursive_minimal_layers,
)
from split3mf.common import Component, trimesh
from split3mf.debug_export import (
    cumulative_snapshot_parts,
    execute_strict_recursive_split,
    recursive_artifact_sha256,
    recursive_part_package_payload,
    recursive_input_geometry_context,
    reload_recursive_part_input,
    validate_shared_child_cap_decisions,
)
from split3mf.domain import (
    BoundaryFairingConfig,
    BoundaryFairingContext,
    CapDecision,
)
from split3mf.package_io import (
    CORE_NS,
    export_colored_parts_3mf,
    load_colored_mesh_objects_3mf,
    validate_colored_parts_3mf,
)
from split3mf.inward import finalize_recursive_colored_mesh
from split3mf.recognition import component_owned_face_colors


def tetrahedron_mesh(offset: float = 0.0):
    vertices = np.asarray(
        [
            [offset + 0.0, 0.0, 0.0],
            [offset + 1.0, 0.0, 0.0],
            [offset + 0.0, 1.0, 0.0],
            [offset + 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
        dtype=np.int64,
    )
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def state_part(index: int, role: str, contains: list[int]) -> dict:
    return {
        "part_id": f"P{index:02d}_{role}",
        "mesh": tetrahedron_mesh(float(index) * 2.0),
        "color_code": "TEST",
        "color_name": "test",
        "color_hex": "#FFFFFFFF",
        "filament_slot_index": None,
        "source_part_index": index,
        "contains": contains,
        "state_role": role,
        "origin_step": 0,
        "stats": {},
        "annotation": {
            "source_part_index": index,
            "state_role": role,
            "contains": contains,
        },
    }


class StrictRecursiveExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parents = {
            1: None,
            2: 1,
            3: 1,
            4: 2,
            5: 2,
            6: 4,
            7: 3,
        }
        self.children = {
            1: [2, 3],
            2: [4, 5],
            3: [7],
            4: [6],
            5: [],
            6: [],
            7: [],
        }

    def test_steps_follow_strict_depth_first_preorder(self) -> None:
        steps = build_recursive_minimal_layers(self.parents, self.children)
        self.assertEqual(
            [step["local_body_index"] for step in steps],
            [1, 2, 4, 3],
        )
        self.assertEqual(
            [step["recursion_path"] for step in steps],
            [[1], [1, 2], [1, 2, 4], [1, 3]],
        )
        self.assertTrue(
            all(
                step["execution_order"] == "strict_depth_first_preorder"
                for step in steps
            )
        )

    def test_pipeline_binds_the_strict_executor_with_the_current_signature(self) -> None:
        parameters = inspect.signature(execute_strict_recursive_split).parameters
        self.assertIn("recursive_steps", parameters)
        pipeline_source = (
            SCRIPTS / "split3mf" / "pipeline.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "recursive_steps=recursive_minimal_layers",
            pipeline_source,
        )
        self.assertNotIn(
            "recursive_minimal_layers=recursive_minimal_layers",
            pipeline_source,
        )

    def test_debug_executor_emits_3mf_parts_not_stl(self) -> None:
        source = inspect.getsource(execute_strict_recursive_split)
        self.assertIn("materialize_changed_part", source)
        self.assertIn('f"{entry[\'part_id\']}_mm.3mf"', source)
        self.assertIn("reload_recursive_part_input", source)
        self.assertNotIn('f"{part_id}.stl"', source)

    def test_merged_tiny_faces_use_effective_component_material_in_recursion(self) -> None:
        zero = np.zeros(3, dtype=np.float64)
        components = [
            Component(
                color_code="0C",
                global_faces=np.asarray([0, 1], dtype=np.int64),
                face_count=2,
                area=2.0,
                bbox_min=zero.copy(),
                bbox_max=zero.copy(),
                center=zero.copy(),
            ),
            Component(
                color_code="4C",
                global_faces=np.asarray([2], dtype=np.int64),
                face_count=1,
                area=1.0,
                bbox_min=zero.copy(),
                bbox_max=zero.copy(),
                center=zero.copy(),
            ),
        ]
        self.assertEqual(
            component_owned_face_colors(
                ["0C", "DEFAULT", "4C", "DEFAULT"],
                components,
            ),
            ["0C", "0C", "4C", "DEFAULT"],
        )
        pipeline_source = (
            SCRIPTS / "split3mf" / "pipeline.py"
        ).read_text(encoding="utf-8")
        self.assertIn("colors=recursive_face_colors", pipeline_source)

    def test_recursive_executor_emits_monitoring_checkpoints_without_weakening_input_identity(self) -> None:
        source = inspect.getsource(execute_strict_recursive_split)
        for event in (
            "recursive_executor_start",
            "recursive_step_start",
            "standalone_write_start",
            "standalone_write_done",
            "recursive_input_reload_start",
            "recursive_input_reload_done",
            "local_body_cut_start",
            "local_body_cut_done",
            "child_subassembly_build_start",
            "child_subassembly_build_done",
            "recursive_step_done",
            "recursive_executor_done",
        ):
            with self.subTest(event=event):
                self.assertIn(f'"{event}"', source)
        self.assertIn("require_artifact_identity=True", source)
        self.assertIn("cumulative_3mf_is_input=False", source)
        self.assertIn('"cumulative_3mf_is_recursive_input": False', source)

    def test_previous_output_is_the_next_step_input(self) -> None:
        state0, transition0 = advance_strict_recursive_state(
            {},
            1,
            state_part(1, "root_body", [1]),
            {
                2: state_part(2, "pending_subassembly", [2, 4, 5, 6]),
                3: state_part(3, "pending_subassembly", [3, 7]),
            },
        )
        retained_root_mesh = state0[1]["mesh"]
        retained_branch_mesh = state0[3]["mesh"]
        self.assertEqual(transition0["after_active_indices"], [1, 2, 3])

        state1, transition1 = advance_strict_recursive_state(
            state0,
            2,
            state_part(2, "local_body", [2]),
            {
                4: state_part(4, "pending_subassembly", [4, 6]),
                5: state_part(5, "leaf_insert", [5]),
            },
        )
        self.assertIs(state1[1]["mesh"], retained_root_mesh)
        self.assertIs(state1[3]["mesh"], retained_branch_mesh)
        self.assertEqual(transition1["replaced_index"], 2)
        self.assertEqual(transition1["retained_indices"], [1, 3])
        self.assertEqual(
            transition1["after_active_indices"],
            [1, 2, 3, 4, 5],
        )
        self.assertNotIn(6, state1)
        self.assertNotIn(7, state1)

        state2, transition2 = advance_strict_recursive_state(
            state1,
            4,
            state_part(4, "local_body", [4]),
            {6: state_part(6, "leaf_insert", [6])},
        )
        self.assertEqual(
            transition2["after_active_indices"],
            [1, 2, 3, 4, 5, 6],
        )
        self.assertNotIn(7, state2)

        state3, transition3 = advance_strict_recursive_state(
            state2,
            3,
            state_part(3, "local_body", [3]),
            {7: state_part(7, "leaf_insert", [7])},
        )
        self.assertEqual(
            transition3["after_active_indices"],
            [1, 2, 3, 4, 5, 6, 7],
        )

    def test_cannot_skip_parent_and_expand_future_descendant(self) -> None:
        state0, _transition0 = advance_strict_recursive_state(
            {},
            1,
            state_part(1, "root_body", [1]),
            {
                2: state_part(2, "pending_subassembly", [2, 4, 5, 6]),
                3: state_part(3, "pending_subassembly", [3, 7]),
            },
        )
        with self.assertRaisesRegex(ValueError, "cannot expand P04"):
            advance_strict_recursive_state(
                state0,
                4,
                state_part(4, "local_body", [4]),
                {6: state_part(6, "leaf_insert", [6])},
            )

    def test_cumulative_3mf_contains_only_the_current_state(self) -> None:
        states = []
        transitions = []
        state, transition = advance_strict_recursive_state(
            {},
            1,
            state_part(1, "root_body", [1]),
            {
                2: state_part(2, "pending_subassembly", [2, 4, 5, 6]),
                3: state_part(3, "pending_subassembly", [3, 7]),
            },
        )
        states.append(state)
        transitions.append(transition)
        state, transition = advance_strict_recursive_state(
            state,
            2,
            state_part(2, "local_body", [2]),
            {
                4: state_part(4, "pending_subassembly", [4, 6]),
                5: state_part(5, "leaf_insert", [5]),
            },
        )
        states.append(state)
        transitions.append(transition)
        state, transition = advance_strict_recursive_state(
            state,
            4,
            state_part(4, "local_body", [4]),
            {6: state_part(6, "leaf_insert", [6])},
        )
        states.append(state)
        transitions.append(transition)
        state, transition = advance_strict_recursive_state(
            state,
            3,
            state_part(3, "local_body", [3]),
            {7: state_part(7, "leaf_insert", [7])},
        )
        states.append(state)
        transitions.append(transition)

        expected_names = [
            ["P01_root_body", "P02_pending_subassembly", "P03_pending_subassembly"],
            [
                "P01_root_body",
                "P02_local_body",
                "P03_pending_subassembly",
                "P04_pending_subassembly",
                "P05_leaf_insert",
            ],
            [
                "P01_root_body",
                "P02_local_body",
                "P03_pending_subassembly",
                "P04_local_body",
                "P05_leaf_insert",
                "P06_leaf_insert",
            ],
            [
                "P01_root_body",
                "P02_local_body",
                "P03_local_body",
                "P04_local_body",
                "P05_leaf_insert",
                "P06_leaf_insert",
                "P07_leaf_insert",
            ],
        ]
        with tempfile.TemporaryDirectory() as directory:
            for step_order, (active, transition, expected) in enumerate(
                zip(states, transitions, expected_names)
            ):
                parts = cumulative_snapshot_parts(active, step_order, transition)
                self.assertEqual(
                    [part["part_id"] for part in parts],
                    expected,
                )
                path = Path(directory) / f"step_{step_order:02d}.3mf"
                export_colored_parts_3mf(
                    path,
                    parts,
                    title=f"strict recursive step {step_order}",
                )
                with zipfile.ZipFile(path) as archive:
                    root = ET.fromstring(archive.read("3D/3dmodel.model"))
                resources = root.find(CORE_NS + "resources")
                mesh_names = [
                    element.attrib["name"]
                    for element in resources.findall(CORE_NS + "object")
                    if element.find(CORE_NS + "mesh") is not None
                ]
                self.assertEqual(mesh_names, expected)

    def test_parent_emitted_part_3mf_preserves_face_colors_and_is_reloadable(self) -> None:
        mesh = trimesh.Trimesh(
            vertices=np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float64,
            ),
            faces=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
            process=False,
        )
        entry = {
            "part_id": "S00_P09_black_PENDING_SUBASSEMBLY",
            "mesh": mesh,
            "color_code": "4C",
            "color_name": "black",
            "color_hex": "#000000FF",
            "filament_slot_index": 1,
            "face_color_hexes": ["#000000FF", "#FFFFFFFF"],
            "face_filament_slot_indices": [1, 0],
            "source_part_index": 9,
            "contains": [2, 9],
            "state_role": "pending_subassembly",
            "origin_step": 0,
            "stats": {},
            "annotation": {
                "source_part_index": 9,
                "contains": [2, 9],
                "state_role": "pending_subassembly",
                "origin_step": 0,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "S00_P09_black_PENDING_SUBASSEMBLY_mm.3mf"
            package_part = recursive_part_package_payload(entry)
            source_palette = ["#FFFFFFFF", "#000000FF"]
            export_colored_parts_3mf(
                path,
                [package_part],
                title="P09 parent-emitted recursive input",
                source_filament_colors=source_palette,
                output_layout="separate-items",
            )
            validation = validate_colored_parts_3mf(
                path,
                [package_part],
                source_filament_colors=source_palette,
                output_layout="separate-items",
            )
            topology_only = [
                error
                for error in validation["errors"]
                if "not watertight" not in error and "open edges" not in error
            ]
            self.assertEqual(topology_only, [])
            loaded = load_colored_mesh_objects_3mf(path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(
                loaded[0]["face_color_hexes"],
                ["#000000FF", "#FFFFFFFF"],
            )
            self.assertEqual(loaded[0]["face_filament_slot_indices"], [1, 0])
            reloaded = reload_recursive_part_input(path, entry)
            self.assertEqual(reloaded["annotation"]["contains"], [2, 9])
            entry["source_3mf_path"] = str(path)
            entry["source_3mf_origin_step"] = 0
            entry["source_3mf_sha256"] = recursive_artifact_sha256(path)
            strict_reloaded = reload_recursive_part_input(
                path,
                entry,
                require_artifact_identity=True,
            )
            self.assertEqual(strict_reloaded["annotation"]["contains"], [2, 9])
            with path.open("ab") as stream:
                stream.write(b"\n")
            with self.assertRaisesRegex(ValueError, "SHA-256 changed"):
                reload_recursive_part_input(
                    path,
                    entry,
                    require_artifact_identity=True,
                )
            self.assertEqual(
                Path(path).name,
                "S00_P09_black_PENDING_SUBASSEMBLY_mm.3mf",
            )

    def test_strict_recursive_input_rejects_cumulative_package(self) -> None:
        entry = state_part(2, "pending_subassembly", [2])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "layer_00_CUMULATIVE_mm.3mf"
            path.write_bytes(b"not-a-recursive-input")
            entry["source_3mf_path"] = str(path)
            entry["source_3mf_sha256"] = recursive_artifact_sha256(path)
            with self.assertRaisesRegex(ValueError, "not .*CUMULATIVE"):
                reload_recursive_part_input(
                    path,
                    entry,
                    require_artifact_identity=True,
                )

    def test_reloaded_parent_geometry_is_rebound_as_the_next_step_mesh(self) -> None:
        original_color_info = dict(common.COLOR_INFO)
        common.COLOR_INFO.update(
            {
                "ROOT": {
                    "hex": "#000000",
                    "filament_slot": 0,
                },
                "EYE": {
                    "hex": "#FFFFFF",
                    "filament_slot": 1,
                },
            }
        )
        try:
            root = tetrahedron_mesh(0.0)
            eye = tetrahedron_mesh(4.0)
            source_vertices = np.vstack((root.vertices, eye.vertices))
            source_faces = np.vstack(
                (root.faces, np.asarray(eye.faces) + len(root.vertices))
            )

            def component(code: str, face_ids: np.ndarray):
                points = source_vertices[source_faces[face_ids].reshape(-1)]
                return common.Component(
                    color_code=code,
                    global_faces=face_ids,
                    face_count=int(len(face_ids)),
                    area=1.0,
                    bbox_min=points.min(axis=0),
                    bbox_max=points.max(axis=0),
                    center=points.mean(axis=0),
                )

            source_components = [
                component("ROOT", np.arange(0, 4, dtype=np.int64)),
                component("EYE", np.arange(4, 8, dtype=np.int64)),
            ]
            serialized_vertices = source_vertices + np.asarray([0.0, 0.0, 2.5])
            serialized_mesh = trimesh.Trimesh(
                vertices=serialized_vertices,
                faces=source_faces,
                process=False,
            )
            fairing = BoundaryFairingContext(
                config=BoundaryFairingConfig(
                    mode="constrained",
                    radius_mm=0.5,
                    max_displacement_mm=0.075,
                    feature_angle_degrees=45.0,
                    fidelity_weight=1.0,
                    legacy_iterations=1,
                    legacy_lambda=0.5,
                    legacy_mu=-0.53,
                ),
                source_surface_normals=np.zeros_like(source_vertices),
            )
            context = recursive_input_geometry_context(
                reloaded_target={
                    "mesh": serialized_mesh,
                    "face_color_hexes": ["#000000FF"] * 4
                    + ["#FFFFFFFF"] * 4,
                    "face_filament_slot_indices": [0] * 4 + [1] * 4,
                },
                expected_indices=[1, 2],
                source_vertices=source_vertices,
                source_faces=source_faces,
                source_components=source_components,
                boundary_fairing=fairing,
            )
            np.testing.assert_allclose(
                context["vertices"],
                serialized_vertices,
            )
            self.assertFalse(
                np.allclose(context["vertices"], source_vertices)
            )
            self.assertEqual(
                context["components"][0].global_faces.tolist(),
                [0, 1, 2, 3],
            )
            self.assertEqual(
                context["components"][1].global_faces.tolist(),
                [4, 5, 6, 7],
            )
            np.testing.assert_allclose(
                context["model_center"],
                serialized_vertices.mean(axis=0),
            )
            self.assertEqual(context["reloaded_face_count"], 8)
        finally:
            common.COLOR_INFO.clear()
            common.COLOR_INFO.update(original_color_info)

    def test_recursive_mesh_cleanup_preserves_surviving_face_materials(self) -> None:
        original_color_info = dict(common.COLOR_INFO)
        common.COLOR_INFO.update(
            {
                "0C": {
                    "hex": "#FFFFFF",
                    "rgba": [255, 255, 255, 255],
                    "filament_slot": 2,
                },
                "4C": {
                    "hex": "#000000",
                    "rgba": [0, 0, 0, 255],
                    "filament_slot": 6,
                },
            }
        )
        try:
            mesh = tetrahedron_mesh()
            faces = np.vstack([np.asarray(mesh.faces), np.asarray(mesh.faces[0])])
            mesh = trimesh.Trimesh(
                vertices=np.asarray(mesh.vertices),
                faces=faces,
                process=False,
            )
            cleaned = finalize_recursive_colored_mesh(
                mesh,
                ["0C", "4C", "4C", "4C", "0C"],
                [2, 6, 6, 6, 2],
                "4C",
            )
            self.assertEqual(len(cleaned.faces), 4)
            self.assertEqual(
                cleaned.metadata["face_color_codes"].count("0C"),
                1,
            )
            self.assertEqual(
                cleaned.metadata["face_color_codes"].count("4C"),
                3,
            )
            self.assertEqual(
                sorted(cleaned.metadata["face_filament_slot_indices"]),
                [2, 6, 6, 6],
            )
        finally:
            common.COLOR_INFO.clear()
            common.COLOR_INFO.update(original_color_info)

    def test_recursive_export_blocks_child_cap_decision_drift(self) -> None:
        decision = CapDecision(
            mode="flat",
            source_vertex_ids=(10, 11),
            fit_points=np.zeros((2, 3), dtype=np.float64),
            directions=np.tile(np.asarray([0.0, 0.0, 1.0]), (2, 1)),
            distances=np.asarray([1.0, 1.2]),
            record={"cap_mode": "flat"},
        )
        child_stats = {
            "loop_extensions": [
                {
                    "loop_index": 0,
                    "cap_mode": "local-offset",
                    "extension_min_mm": 1.0,
                    "extension_max_mm": 1.2,
                }
            ]
        }
        with self.assertRaisesRegex(ValueError, "cap mode changed after planning"):
            validate_shared_child_cap_decisions(
                child_index=2,
                decisions={0: decision},
                child_stats=child_stats,
            )


if __name__ == "__main__":
    unittest.main()
