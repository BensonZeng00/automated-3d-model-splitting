from pathlib import Path
import sys
import numpy as np
import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from split3mf.common import load_core_dependencies
load_core_dependencies()
from split3mf.interface_surface import build_pairwise_interface_surfaces, build_mortise_shell_probe


@pytest.mark.parametrize("amplitude", [1., 2., 4.])
@pytest.mark.parametrize("reverse", [False, True])
def test_curved_shoulder_automatically_closes_without_connector(amplitude, reverse):
    angles = np.arange(40)*2*np.pi/40
    outer = np.column_stack((10*np.cos(angles),4*np.sin(angles),
                             amplitude*np.cos(angles)**2))
    if reverse:
        outer = outer[::-1]
    axis = np.array([0.,0.,1.])
    pair = build_pairwise_interface_surfaces(
        boundary_points_mm=outer, insertion_direction=axis,
        socket_inward_direction=axis, scale_ratio=.5, clearance_mm=.2,
        mortise_shell_probe=build_mortise_shell_probe(
            trimesh.creation.box(extents=(100.,100.,100.)).triangles),
        interface_id="curved")
    n = len(outer)
    for surface in (pair.tenon, pair.mortise):
        np.testing.assert_array_equal(surface.vertices[:n], outer)
        assert surface.record["selected_strategy"] == "curved_shoulder_without_connector"
        assert surface.record["closure_face_count"] == 3*n-2
        assert surface.record["connector_wall_face_count"] == 0
        assert len(surface.vertices) == 2*n
        assert surface.record["depth_probe_query_count"] == 0
        base = surface.vertices[n:2*n]
        tip = surface.vertices[n:2*n]
        np.testing.assert_allclose(base @ axis, np.mean(outer @ axis), atol=1e-10)
        np.testing.assert_allclose(tip-base, np.tile(axis*surface.record["extension_depth_mm"],(n,1)),atol=1e-10)
        assert surface.record["topology_audit"]["valid"]
    delta = pair.mortise.vertices[n:2*n]-pair.tenon.vertices[n:2*n]
    np.testing.assert_allclose(delta @ axis,0.,atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(delta[:,:2],axis=1),.2,atol=1e-10)
    np.testing.assert_array_equal(pair.tenon.faces,pair.mortise.faces)


def test_conflicting_directions_still_rejected():
    with pytest.raises(ValueError, match="same validated inward axis"):
        build_pairwise_interface_surfaces(
            boundary_points_mm=np.array([[0.,0.,0.],[3.,0.,0.],[0.,3.,0.]]),
            insertion_direction=np.array([0.,0.,1.]), socket_inward_direction=np.array([0.,0.,-1.]),
            scale_ratio=.5, clearance_mm=.2, mortise_shell_probe=None, interface_id="axis")


@pytest.mark.parametrize("amplitude", [0., .06, .12])
def test_restored_scaled_profile_starts_at_entire_rim(amplitude):
    angles = np.arange(40)*2*np.pi/40
    outer = np.column_stack((10*np.cos(angles), 10*np.sin(angles), amplitude*np.cos(2*angles)))
    pair = build_pairwise_interface_surfaces(
        boundary_points_mm=outer, insertion_direction=np.array([0.,0.,1.]),
        socket_inward_direction=np.array([0.,0.,1.]), scale_ratio=.5,
        clearance_mm=.2, mortise_shell_probe=build_mortise_shell_probe(
            trimesh.creation.box(extents=(100.,100.,100.)).triangles),
        interface_id="restored")
    for surface in (pair.tenon, pair.mortise):
        assert surface.record["selected_strategy"] == "scaled_inner_contour"
        assert surface.record["root_vertex_start"] == 0
        assert surface.record["tip_vertex_start"] == len(outer)
        assert surface.record["closure_face_count"] == 0
        assert len(surface.vertices) == 2*len(outer)
        np.testing.assert_array_equal(surface.vertices[:len(outer)], outer)
        assert np.ptp(surface.vertices[len(outer):, 2]) < 1e-10
    tip = pair.tenon.vertices[len(outer):]
    np.testing.assert_allclose(np.linalg.norm(tip[:, :2], axis=1), 5., atol=1e-10)
    delta = pair.mortise.vertices[len(outer):]-tip
    np.testing.assert_allclose(np.linalg.norm(delta[:, :2], axis=1), .2, atol=1e-10)
    np.testing.assert_allclose(delta[:, 2], .2, atol=1e-10)
