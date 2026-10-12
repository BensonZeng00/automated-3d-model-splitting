from pathlib import Path
import sys

import numpy as np
import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from split3mf.common import load_core_dependencies

load_core_dependencies()

from split3mf.interface_surface import (
    build_mortise_shell_probe,
    build_pairwise_interface_surfaces,
    InterfaceDepthError,
)


def test_mortise_probe_ignores_entry_and_detects_far_wall_exit():
    shell = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    probe = build_mortise_shell_probe(shell.triangles)
    origins = np.array([[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    axes = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    short_distances, _ = probe.exits(origins, axes, maximum_mm=1.5)
    assert np.isnan(short_distances[0])
    assert short_distances[1] == 1.0

    distances, faces = probe.exits(origins, axes, maximum_mm=3.5)
    np.testing.assert_allclose(distances, [3.0, 1.0], atol=1e-8)
    assert np.all(faces >= 0)


def test_mortise_probe_skips_start_surface_contact_but_keeps_far_exit():
    near = np.array([
        [0.007, -1.0, -1.0],
        [0.007, 1.0, -1.0],
        [0.007, 0.0, 1.0],
    ])
    far = near.copy()
    far[:, 0] = 0.25
    probe = build_mortise_shell_probe(np.stack((near, far)))

    distances, faces = probe.exits(
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[1.0, 0.0, 0.0]]),
        maximum_mm=0.4,
        epsilon=0.01,
    )

    np.testing.assert_allclose(distances, [0.25], atol=1e-8)
    assert faces.tolist() == [1]


def test_backoff_queries_once_and_keeps_positive_engagement():
    host = trimesh.creation.box(extents=(4., 4., 2.))
    real = build_mortise_shell_probe(host.triangles)

    class CountingProbe:
        calls = 0
        def exits(self, *args, **kwargs):
            self.calls += 1
            return real.exits(*args, **kwargs)

    probe = CountingProbe()
    outer = np.array([[-1.,-1.,0.], [1.,-1.,0.], [1.,1.,0.], [-1.,1.,0.]])
    pair = build_pairwise_interface_surfaces(
        boundary_points_mm=outer, insertion_direction=np.array([0.,0.,1.]),
        socket_inward_direction=np.array([0.,0.,1.]), scale_ratio=.5,
        clearance_mm=.2, mortise_shell_probe=probe, interface_id="depth")
    assert probe.calls == 1
    assert pair.tenon.record["tenon_extension_depth_mm"] >= .2
    assert pair.tenon.record["connector_wall_face_count"] == 2*len(outer)
    assert pair.mortise.vertices[:, 2].max() + .05 < 1.
