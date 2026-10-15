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


@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize("reverse_ring", [False, True])
@pytest.mark.parametrize("provided_direction", ["local", "center"])
def test_local_boundary_controls_tip_instead_of_part_center_axis(
    sign, reverse_ring, provided_direction
):
    boundary = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=float)
    if reverse_ring:
        boundary = boundary[::-1]
    # A legacy center-based plan can point out of the confirmed tenon.
    axis = np.array([0.6, 0.0, 0.8]) * -sign
    local_axis = np.array([0.0, 0.0, 1.0]) * sign
    source_points = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=float)
    source_faces = np.array([[0, 2, 1], [0, 3, 2]])
    if sign < 0:
        source_faces = source_faces[:, ::-1]
    host = trimesh.Trimesh(source_points, source_faces, process=False)
    solid = trimesh.creation.box(extents=(4, 4, 2))
    solid.apply_translation([0, 0, sign])
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
            {
                "schema": "contact-interface-plan/v1",
                "relation_table": {"rows": [[
                    "I001", relation["tenon_part"], relation["mortise_part"],
                ]]},
                "interfaces": [relation],
            },
            vertices=solid.vertices, faces=solid.faces, components=[None, None],
            scale_ratio=0.5, clearance_mm=0.2,
        )
    record = summary["interfaces"][0]
    count = record["densified_boundary_point_count"]
    assert count == record["simplified_boundary_point_count"]
    assert record["boundary_densification_subdivisions"] == 0
    tenon_vertices = arrays["i001_loop_000_tenon_vertices"]
    start = record["tenon_surface"]["tip_vertex_start"]
    tip_count = record["tenon_surface"]["tip_vertex_count"]
    tip = arrays["i001_loop_000_tenon_vertices"][start:start+tip_count]
    floor = arrays["i001_loop_000_mortise_vertices"][start:start+tip_count]
    np.testing.assert_allclose(record["construction_axis_toward_mortise"], local_axis)
    np.testing.assert_allclose(
        record["tenon_surface"]["extension_direction"], local_axis, atol=1e-8
    )
    root_start = record["tenon_surface"]["root_vertex_start"]
    root = tenon_vertices[root_start:root_start+tip_count]
    assert np.mean((tip - root) @ local_axis) > 0
    axial_gap = (floor - tip) @ local_axis
    radial_gap = floor - tip - axial_gap[:, None] * local_axis
    np.testing.assert_allclose(axial_gap, 0.2, atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(radial_gap, axis=1), 0.2, atol=1e-7)
    assert record["clearance_matches_configuration"]
