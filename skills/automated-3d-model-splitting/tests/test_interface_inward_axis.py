from pathlib import Path
import sys

import numpy as np
import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from split3mf.interface_direction import inward_interface_axis
from split3mf.local_ray_probe import LocalRayProbe


@pytest.mark.parametrize("reverse", [False, True])
def test_ring_order_and_rotation_do_not_reverse_material_side(reverse):
    ring = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=float)
    triangles = ring[[[0, 1, 2], [0, 2, 3]]]
    rotation = np.array([[0.8, 0, 0.6], [0, 1, 0], [-0.6, 0, 0.8]])
    ring = ring @ rotation.T + [20, -12, 45]
    triangles = triangles @ rotation.T + [20, -12, 45]
    axis, evidence = inward_interface_axis(ring[::-1] if reverse else ring, triangles)
    np.testing.assert_allclose(axis, rotation @ [0, 0, -1], atol=1e-12)
    assert evidence["outward_normal_dot_axis"] < 0


def test_conflicting_source_winding_is_not_certified_by_ray_misses():
    ring = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=float)
    triangles = ring[[[0, 1, 2], [0, 3, 2]]]
    with pytest.raises(ValueError, match="do not establish"):
        inward_interface_axis(ring, triangles)


@pytest.mark.parametrize("surface_patch", [False, True])
def test_legacy_fallback_without_matching_source_boundary(surface_patch, monkeypatch):
    monkeypatch.setattr("split3mf.interface_direction._source_boundary_orientation",
                        lambda *args: (None, {"matched": False}))
    solid = trimesh.creation.box(extents=(4, 4, 4))
    z = 2.0 if surface_patch else 0.0
    ring = np.array([[-1, -1, z], [1, -1, z], [1, 1, z], [-1, 1, z]])
    # Local surface normals alone would choose -Z in both cases.
    axis, audit = inward_interface_axis(
        ring, ring[[[0, 1, 2], [0, 2, 3]]],
        source_probe=LocalRayProbe(solid), toward_mortise=[0, 0, 1], maximum_mm=20,
    )
    np.testing.assert_allclose(axis, [0, 0, -1 if surface_patch else 1])
    assert audit["selection"] == (
        "only_source_interior_side" if surface_patch else "both_interior_toward_confirmed_mortise"
    )


def test_first_hit_distinguishes_entry_from_exit():
    probe = LocalRayProbe(trimesh.creation.box(extents=(4, 4, 4)))
    entry, _ = probe.first_hits([[0, 0, -3]], [[0, 0, 1]], 10)
    exit_distance, _ = probe.exits([[0, 0, -3]], [[0, 0, 1]], 10)
    np.testing.assert_allclose(entry, [1])
    np.testing.assert_allclose(exit_distance, [5])


def test_both_candidates_outside_without_source_boundary_are_rejected(monkeypatch):
    monkeypatch.setattr("split3mf.interface_direction._source_boundary_orientation",
                        lambda *args: (None, {"matched": False}))
    solid = trimesh.creation.box(extents=(4, 4, 4))
    ring = np.array([[-1, -1, 20], [1, -1, 20], [1, 1, 20], [-1, 1, 20]], dtype=float)
    with pytest.raises(ValueError, match="neither source-interior rays"):
        inward_interface_axis(
            ring, ring[[[0, 1, 2], [0, 2, 3]]], source_probe=LocalRayProbe(solid),
            toward_mortise=[0, 0, -1], maximum_mm=50,
        )


@pytest.mark.parametrize("reverse", [False, True])
def test_symmetric_rays_fallback_uses_region_ownership(reverse, monkeypatch):
    monkeypatch.setattr("split3mf.interface_direction._source_boundary_orientation",
                        lambda *args: (None, {"matched": False}))
    ring = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=float)
    if reverse:
        ring = ring[::-1]
    tenon_triangles = []
    mortise_triangles = []
    for start, end in zip(ring, np.roll(ring, -1, axis=0)):
        midpoint = (start + end) * 0.5
        tenon_triangles.append([start, end, midpoint + [0, 0, -3]])
        mortise_triangles.append([start, end, midpoint + [0, 0, 3]])

    class SymmetricInteriorProbe:
        triangles = np.asarray([
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        ], dtype=float)

        def first_hits(self, starts, directions, maximum):
            return np.ones(len(directions)), np.arange(len(directions))

    axis, audit = inward_interface_axis(
        ring,
        np.asarray(tenon_triangles),
        source_probe=SymmetricInteriorProbe(),
        mortise_triangles=np.asarray(mortise_triangles),
        toward_mortise=[0, 0, -1],
        maximum_mm=10,
    )
    np.testing.assert_allclose(axis, [0, 0, 1], atol=1e-12)
    assert audit["source_interior_votes"] == [3, 3]
    assert audit["selection"] == "paired_region_ownership_toward_confirmed_mortise"
    assert audit["region_side_evidence"]["direction_confidence"] >= 0.65


def test_symmetric_rays_and_ambiguous_region_ownership_block_centroid_fallback(monkeypatch):
    monkeypatch.setattr("split3mf.interface_direction._source_boundary_orientation",
                        lambda *args: (None, {"matched": False}))
    ring = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=float)
    triangles = ring[[[0, 1, 2], [0, 2, 3]]]

    class SymmetricInteriorProbe:
        triangles = np.asarray([
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        ], dtype=float)

        def first_hits(self, starts, directions, maximum):
            return np.ones(len(directions)), np.arange(len(directions))

    with pytest.raises(ValueError, match="does not corroborate"):
        inward_interface_axis(
            ring,
            triangles,
            source_probe=SymmetricInteriorProbe(),
            mortise_triangles=triangles,
            toward_mortise=[0, 0, 1],
            maximum_mm=10,
        )


def test_strong_inward_normals_override_outward_surface_proximity(monkeypatch):
    monkeypatch.setattr("split3mf.interface_direction._source_boundary_orientation",
                        lambda *args: (None, {"matched": False}))
    monkeypatch.setattr("split3mf.interface_direction._region_side_direction",
                        lambda *args: (1, {"direction_confidence": 1.0}))
    ring = np.array([[-1., -1., 0.], [1., -1., 0.],
                     [1., 1., 0.], [-1., 1., 0.]])
    triangles = ring[[[0, 1, 2], [0, 2, 3]]]

    class Probe:
        def __init__(self, faces):
            self.triangles = faces

        def first_hits(self, starts, directions, maximum):
            return np.ones(3), np.zeros(3, dtype=int)

    axis, evidence = inward_interface_axis(
        ring, triangles, source_probe=Probe(triangles),
        mortise_triangles=triangles, toward_mortise=[0., 0., 1.], maximum_mm=5.
    )
    np.testing.assert_allclose(axis, [0., 0., -1.])
    assert evidence["region_side_conflict"] is True


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("sign", [-1., 1.])
def test_wrapping_host_cannot_reverse_outward_wound_insert(reverse, sign):
    # A dome above the seam must close farther into -Z, even when surrounding
    # host surfaces are nearer above the seam and would win proximity votes.
    angles = np.linspace(0, 2*np.pi, 32, endpoint=False)
    rim = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(32)))
    triangles = np.array([[rim[i], rim[(i+1) % 32], [0, 0, 2]] for i in range(32)])
    triangles[:, :, 2] *= sign
    if sign < 0:
        triangles = triangles[:, ::-1]
    rotation = trimesh.transformations.rotation_matrix(.73, [1, 2, 3])[:3, :3]
    shift = np.array([31., -22., 19.])
    rim = rim @ rotation.T + shift
    triangles = triangles @ rotation.T + shift
    host_triangles = triangles + (rotation @ [0, 0, sign*.2])

    class UnneededRayProbe:
        def first_hits(self, *args):
            raise AssertionError("source topology must determine the material side")

    axis, audit = inward_interface_axis(
        rim[::-1] if reverse else rim, triangles,
        source_probe=UnneededRayProbe(), mortise_triangles=host_triangles,
        toward_mortise=rotation @ [0, 0, sign], maximum_mm=10,
    )
    np.testing.assert_allclose(axis, rotation @ [0, 0, -sign], atol=1e-12)
    assert audit["selection"] == "opposite_outward_source_boundary"
    assert audit["source_boundary_evidence"]["orientation_consensus"] > .99
