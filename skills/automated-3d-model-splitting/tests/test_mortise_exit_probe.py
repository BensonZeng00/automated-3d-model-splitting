from pathlib import Path
import sys

import numpy as np
import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from split3mf.common import load_core_dependencies

load_core_dependencies()

from split3mf.interface_surface import (
    _mortise_collision_statistics,
    build_mortise_shell_probe,
    build_pairwise_interface_surfaces,
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


def test_mortise_collision_requires_more_than_forty_percent_of_all_probe_rays():
    distances = np.array([0.100001] * 4 + [0.1] * 4)
    stats = _mortise_collision_statistics(distances)

    assert stats["probe_count"] == 8
    assert stats["raw_hit_count"] == 8
    assert stats["blocking_hit_count"] == 4
    assert stats["near_surface_hit_count"] == 4
    assert stats["blocking_ratio"] == 0.5
    assert stats["collision"] is True

    below_threshold = _mortise_collision_statistics(
        np.array([0.100001] * 3 + [0.1] * 4 + [np.nan])
    )
    assert below_threshold["blocking_ratio"] == 0.375
    assert below_threshold["collision"] is False

    exactly_at_threshold = _mortise_collision_statistics(
        np.array([0.100001] * 2 + [0.1] * 3)
    )
    assert exactly_at_threshold["blocking_ratio"] == 0.4
    assert exactly_at_threshold["collision"] is False

    sparse_hits = _mortise_collision_statistics(
        np.concatenate((np.array([0.2, 0.3, 0.4]), np.full(37149, np.nan)))
    )
    assert sparse_hits["raw_hit_count"] == 3
    assert sparse_hits["probe_count"] == 37152
    assert sparse_hits["blocking_ratio"] < 0.01
    assert sparse_hits["collision"] is False


@pytest.mark.parametrize("short,far,expected", [
    (40, 0, False), (41, 0, True),
    (0, 5, False), (0, 6, True),
    (40, 5, False), (39, 6, True), (41, 1, True),
])
def test_disjoint_collision_bands_use_all_rays(short, far, expected):
    values = np.array([1.0] * short + [1.000001] * far + [np.nan] * (100-short-far))
    result = _mortise_collision_statistics(values)
    assert result["distance_bands"]["short"]["hit_count"] == short
    assert result["distance_bands"]["far"]["hit_count"] == far
    assert result["collision"] is expected


@pytest.mark.parametrize("values", [[], [np.nan], [0.0, 0.1, np.nan]])
def test_no_blocking_hits(values):
    result = _mortise_collision_statistics(np.array(values))
    assert result["blocking_hit_count"] == 0
    assert result["collision"] is False


def test_reported_p11_far_hits_trigger_collision():
    result = _mortise_collision_statistics(np.array([3.07435] * 199 + [np.nan] * 1713))
    assert result["distance_bands"]["far"]["ratio"] == 199 / 1912
    assert result["collision"] is True


def test_tenon_insertion_direction_wins_over_conflicting_host_average():
    host = trimesh.creation.box(extents=(4.0, 4.0, 2.0))
    probe = build_mortise_shell_probe(host.triangles)
    boundary = np.array([
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
    ])

    pair = build_pairwise_interface_surfaces(
        boundary_points_mm=boundary,
        insertion_direction=np.array([0.0, 0.0, -1.0]),
        socket_inward_direction=np.array([0.0, 0.0, 1.0]),
        scale_ratio=0.5,
        clearance_mm=0.2,
        mortise_shell_probe=probe,
        interface_id="I001",
    )

    assert pair.tenon.record["extension_direction"][2] < -0.99
    assert pair.tenon.record["stage04_socket_inward_direction"] == [0.0, 0.0, 1.0]
    attempts = pair.tenon.record["extension_depth_attempts"]
    np.testing.assert_allclose(
        [item["candidate_depth_mm"] for item in attempts],
        [5.0, 2.5, 1.25, 0.625],
    )
    assert attempts[0]["collision"] is True
    assert attempts[-1]["collision"] is False
    assert pair.tenon.record["extension_depth_mm"] == 0.625
