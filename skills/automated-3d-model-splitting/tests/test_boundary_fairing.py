from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from split3mf import common

common.load_core_dependencies()

from split3mf.boundary_fairing import fair_boundary_loop
from split3mf.cli import build_parser
from split3mf.common import Component
from split3mf.domain import BoundaryFairingConfig, BoundaryFairingContext
from split3mf.inward import make_part_mesh, mesh_vertex_inward_normals
from split3mf.mesh import taubin_smooth_loop


def constrained_config(**overrides) -> BoundaryFairingConfig:
    values = {
        "mode": "constrained",
        "radius_mm": 1.20,
        "max_displacement_mm": 0.075,
        "feature_angle_degrees": 60.0,
        "fidelity_weight": 1.0,
        "legacy_iterations": 16,
        "legacy_lambda": 0.5,
        "legacy_mu": -0.53,
    }
    values.update(overrides)
    return BoundaryFairingConfig(**values)


def closed_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1).sum())


def square_edge_error(points: np.ndarray) -> float:
    error = 0.0
    for start in (0, 4, 8, 12):
        end = (start + 4) % len(points)
        indices = [(start + offset) % len(points) for offset in range(5)]
        left = points[start]
        direction = points[end] - left
        direction /= np.linalg.norm(direction)
        offsets = points[indices] - left
        distances = np.linalg.norm(offsets - (offsets @ direction)[:, None] * direction, axis=1)
        error += float(np.dot(distances, distances))
    return error


class BoundaryFairingTests(unittest.TestCase):
    def test_constrained_fairing_locks_corners_and_caps_displacement(self) -> None:
        points = np.array(
            [
                [0.00, 0.00, 0.0],
                [0.25, 0.02, 0.0],
                [0.50, -0.02, 0.0],
                [0.75, 0.02, 0.0],
                [1.00, 0.00, 0.0],
                [0.98, 0.25, 0.0],
                [1.02, 0.50, 0.0],
                [0.98, 0.75, 0.0],
                [1.00, 1.00, 0.0],
                [0.75, 0.98, 0.0],
                [0.50, 1.02, 0.0],
                [0.25, 0.98, 0.0],
                [0.00, 1.00, 0.0],
                [0.02, 0.75, 0.0],
                [-0.02, 0.50, 0.0],
                [0.02, 0.25, 0.0],
            ],
            dtype=np.float64,
        )
        normals = np.tile(np.array([0.0, 0.0, 1.0]), (len(points), 1))
        result, record = fair_boundary_loop(
            points,
            normals,
            np.arange(100, 100 + len(points)),
            constrained_config(),
        )

        displacement = np.linalg.norm(result - points, axis=1)
        self.assertEqual(record["status"], "solved")
        self.assertLessEqual(float(displacement.max()), 0.075 + 1e-9)
        np.testing.assert_allclose(result[[0, 4, 8, 12]], points[[0, 4, 8, 12]], atol=1e-12)
        self.assertLess(square_edge_error(result), square_edge_error(points))
        self.assertTrue(np.any(displacement > 1e-5))

    def test_canonical_source_order_matches_reversed_parent_loop(self) -> None:
        angles = np.linspace(0.0, 2.0 * np.pi, 18, endpoint=False)
        radii = 2.0 + 0.03 * np.sin(5.0 * angles)
        points = np.column_stack((radii * np.cos(angles), radii * np.sin(angles), np.zeros_like(angles)))
        normals = np.tile(np.array([0.0, 0.0, 1.0]), (len(points), 1))
        source_ids = np.arange(500, 500 + len(points))
        config = constrained_config(feature_angle_degrees=180.0)

        forward, _ = fair_boundary_loop(points, normals, source_ids, config)
        reverse_order = np.arange(len(points) - 1, -1, -1)
        reversed_result, _ = fair_boundary_loop(
            points[reverse_order],
            normals[reverse_order],
            source_ids[reverse_order],
            config,
        )
        by_source_id = {int(source_id): point for source_id, point in zip(source_ids[reverse_order], reversed_result)}
        reordered = np.array([by_source_id[int(source_id)] for source_id in source_ids])
        np.testing.assert_allclose(forward, reordered, atol=1e-10)

    def test_smooth_loop_adds_extent_anchors(self) -> None:
        angles = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
        points = np.column_stack((3.0 * np.cos(angles), 2.0 * np.sin(angles), np.zeros_like(angles)))
        normals = np.tile(np.array([0.0, 0.0, 1.0]), (len(points), 1))
        result, record = fair_boundary_loop(
            points,
            normals,
            np.arange(1000, 1000 + len(points)),
            constrained_config(feature_angle_degrees=180.0),
        )
        self.assertGreaterEqual(record["locked_feature_vertices"], 4)
        self.assertLessEqual(float(np.linalg.norm(result - points, axis=1).max()), 0.075 + 1e-9)
        np.testing.assert_allclose(result[[0, 6, 12, 18]], points[[0, 6, 12, 18]], atol=1e-12)

    def test_legacy_mode_preserves_taubin_behavior(self) -> None:
        points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        config = constrained_config(mode="taubin")
        result, record = fair_boundary_loop(
            points,
            np.tile(np.array([0.0, 0.0, 1.0]), (len(points), 1)),
            np.arange(len(points)),
            config,
        )
        expected = taubin_smooth_loop(points, 16, 0.5, -0.53)
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(record["status"], "legacy_compatibility")

    def test_cli_defaults_to_constrained_fairing(self) -> None:
        args = build_parser().parse_args(["--input", "placeholder.3mf"])
        self.assertEqual(args.boundary_fairing_mode, "constrained")
        self.assertEqual(args.boundary_fairing_radius_mm, 1.20)
        self.assertEqual(args.boundary_max_displacement_mm, 0.075)

    def test_part_builder_uses_fairing_context(self) -> None:
        angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
        radii = 1.0 + 0.02 * np.sin(4.0 * angles)
        ring = np.column_stack((radii * np.cos(angles), radii * np.sin(angles), np.zeros_like(angles)))
        vertices = np.vstack((ring, np.array([[0.0, 0.0, 0.0]])))
        center_index = len(ring)
        faces = np.array(
            [[center_index, index, (index + 1) % len(ring)] for index in range(len(ring))],
            dtype=np.int64,
        )
        component = Component(
            color_code="DEFAULT",
            global_faces=np.arange(len(faces), dtype=np.int64),
            face_count=len(faces),
            area=3.0,
            bbox_min=vertices.min(axis=0),
            bbox_max=vertices.max(axis=0),
            center=vertices.mean(axis=0),
        )
        context = BoundaryFairingContext(
            config=constrained_config(feature_angle_degrees=180.0),
            source_surface_normals=mesh_vertex_inward_normals(vertices, faces),
        )
        mesh, stats = make_part_mesh(
            vertices=vertices,
            faces=faces,
            component=component,
            component_index=1,
            assembly_parent_index=None,
            part_id="synthetic_insert",
            max_extension_mm=1.0,
            boundary_fairing=context,
            flat_clearance_mm=0.05,
            fit_clearance_mm=0.30,
            insert_shrink_mm=0.15,
            lead_in_mm=0.20,
            top_edge_clearance_mm=0.05,
            clearance_mode="insert-shrink",
            child_cut_refs=[],
            boundary_neighbor_lookup={},
            component_centers={1: component.center},
            sibling_clearance_mm=0.0,
            socket_overcut_mm=0.0,
            bottom_clearance_mm=0.0,
            inward_override=np.array([0.0, 0.0, -1.0]),
            model_center=np.array([0.0, 0.0, -1.0]),
            cap_mode="adaptive",
            planar_extra_limit_mm=0.30,
        )
        self.assertGreater(len(mesh.faces), len(faces))
        self.assertEqual(stats["boundary_fairing_records"][0]["status"], "solved")
        self.assertLessEqual(
            stats["boundary_fairing_records"][0]["maximum_displacement_mm"],
            0.075 + 1e-9,
        )


if __name__ == "__main__":
    unittest.main()
