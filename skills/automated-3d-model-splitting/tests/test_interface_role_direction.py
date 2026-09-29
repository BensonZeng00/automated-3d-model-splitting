from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from split3mf.common import load_core_dependencies

load_core_dependencies()
from split3mf.interface_assembly import build_pairwise_interface_surfaces
from split3mf.interface_surface import _simple_planar_inner_contour
from split3mf.interface_surface import build_pairwise_interface_surfaces as build_surface_pair
from split3mf.mesh import orthonormal_basis
from split3mf.annulus_projection import AnnulusProjectionAudit
from split3mf.spatial_intersections import first_nonincident_triangle_intersection_3d


def test_projected_triangle_overlap_can_be_clear_in_xyz():
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    points = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0],
        [0.1, 0.1, 1], [0.9, 0.1, 1], [0.1, 0.9, 1],
    ], dtype=float)
    assert first_nonincident_triangle_intersection_3d(faces, points)["valid"]
    points[3:] -= np.array([0, 0, 1])
    assert not first_nonincident_triangle_intersection_3d(faces, points)["valid"]


def test_ellipse_skips_self_intersection_scans(monkeypatch):
    class OpenHostProbe:
        def exits(self, points, directions, distance, *, epsilon):
            return np.full(len(points), np.nan), np.full(len(points), -1)

    boundary = np.array([
        [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0],
    ])
    monkeypatch.setattr(
        "split3mf.interface_surface.first_nonincident_triangle_intersection_3d",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("ellipse self-intersection scan must be skipped")
        ),
    )
    pair = build_surface_pair(
        boundary_points_mm=boundary,
        insertion_direction=np.array([0.0, 0.0, 1.0]),
        socket_inward_direction=np.array([0.0, 0.0, 1.0]),
        scale_ratio=0.5, clearance_mm=0.1,
        mortise_shell_probe=OpenHostProbe(), interface_id="I-test",
    )
    audit = pair.tenon.record["topology_audit"]
    assert audit["three_dimensional_self_intersection"]["performed"] is False
    assert audit["three_dimensional_self_intersection_allowed"] is True
    assert audit["self_intersection_scan_policy"] == "skipped_for_ellipse_candidate_by_user_request"
    assert pair.tenon.record["inner_planar_contour"]["projection_intersection_scan_performed"] is False


def test_inner_contour_always_uses_ellipse():
    crossed = np.array([[0, 0], [2, 2], [0, 2], [2, 0]], dtype=float)
    source = np.column_stack((crossed, np.zeros(len(crossed))))
    fallback, record = _simple_planar_inner_contour(crossed, source)
    assert record["strategy"] == "arclength_parameterized_planar_ellipse"
    assert len(fallback) == len(crossed)
    square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
    _, simple_record = _simple_planar_inner_contour(
        square, np.column_stack((square, np.zeros(len(square))))
    )
    assert simple_record["strategy"] == "arclength_parameterized_planar_ellipse"


def test_ellipse_rejects_insufficient_host_thickness():
    class RadiusLimitedProbe:
        def exits(self, points, directions, distance, *, epsilon):
            blocked = abs(points[0, 0]) < 0.7
            distances = np.full(len(points), 0.3 if blocked else np.nan)
            return distances, np.zeros(len(points), dtype=np.int64)

    boundary = np.array([
        [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0],
    ])
    with pytest.raises(ValueError, match="elliptical inner contour lacks adequate host thickness"):
        build_surface_pair(
            boundary_points_mm=boundary,
            insertion_direction=np.array([0.0, 0.0, 1.0]),
            socket_inward_direction=np.array([0.0, 0.0, 1.0]),
            scale_ratio=0.5, clearance_mm=0.1,
            mortise_shell_probe=RadiusLimitedProbe(), interface_id="I-test",
        )


def test_ellipse_plane_is_normal_to_mating_axis():
    class OpenHostProbe:
        def exits(self, points, directions, distance, *, epsilon):
            return np.full(len(points), np.nan), np.full(len(points), -1)

    axis = np.array([0.6, 0.0, 0.8])
    u, v = orthonormal_basis(axis)
    boundary = np.array([-u-v, u-v, u+v, -u+v])
    pair = build_surface_pair(
        boundary_points_mm=boundary, insertion_direction=axis,
        socket_inward_direction=axis, scale_ratio=0.5,
        clearance_mm=0.1, mortise_shell_probe=OpenHostProbe(),
        interface_id="I-test",
    )
    record = pair.tenon.record
    assert record["inner_planar_contour"]["strategy"] == "arclength_parameterized_planar_ellipse"
    np.testing.assert_allclose(record["ellipse_plane_normal"], axis)
    np.testing.assert_allclose(record["ellipse_extrusion_axis"], axis)
    inner = pair.tenon.vertices[len(boundary):2 * len(boundary)]
    assert np.ptp(inner @ axis) < 1e-8


@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize("reverse_ring", [False, True])
@pytest.mark.parametrize("provided_direction", ["local", "center"])
def test_local_boundary_controls_tip_instead_of_part_center_axis(
    sign, reverse_ring, provided_direction
):
    boundary = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=float)
    if reverse_ring:
        boundary = boundary[::-1]
    axis = np.array([0.6, 0.0, 0.8]) * sign
    local_axis = np.array([0.0, 0.0, 1.0]) * sign
    host = trimesh.creation.box(extents=(4, 4, 2))
    relation = {
        "interface_id": "I001", "parts": ["P01", "P02"],
        "tenon_part": "P01" if sign > 0 else "P02",
        "mortise_part": "P02" if sign > 0 else "P01",
        "mating_axis_toward_mortise": axis.tolist(),
        "insertion_direction": (
            local_axis if provided_direction == "local" else axis
        ).tolist(),
        "socket_inward_direction": (-axis).tolist(),
        "contact": {"shared_boundary_loops": [{
            "tenon_loop_index": 0, "mortise_loop_index": 0,
            "boundary_points_mm": boundary.tolist(),
        }]},
    }
    with patch("split3mf.interface_assembly.build_local_mesh", return_value=(host.vertices, host.faces, {}, [])):
        arrays, summary = build_pairwise_interface_surfaces(
            {"schema": "contact-interface-plan/v1", "interfaces": [relation]},
            vertices=host.vertices, faces=host.faces, components=[None, None],
            scale_ratio=0.5, clearance_mm=0.2,
        )
    record = summary["interfaces"][0]
    count = record["densified_boundary_point_count"]
    assert count == record["simplified_boundary_point_count"]
    assert record["boundary_densification_subdivisions"] == 0
    tenon_vertices = arrays["i001_loop_000_tenon_vertices"]
    tip = arrays["i001_loop_000_tenon_vertices"][count:2*count]
    floor = arrays["i001_loop_000_mortise_vertices"][count:2*count]
    np.testing.assert_allclose(record["construction_axis_toward_mortise"], local_axis)
    np.testing.assert_allclose(
        record["tenon_surface"]["extension_direction"], local_axis, atol=1e-8
    )
    assert np.mean((tip - tenon_vertices[:count]) @ local_axis) > 0
    axial_gap = (floor - tip) @ local_axis
    radial_gap = floor - tip - axial_gap[:, None] * local_axis
    np.testing.assert_allclose(axial_gap, 0.2, atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(radial_gap, axis=1), 0.2, atol=1e-7)
    assert record["clearance_matches_configuration"]
