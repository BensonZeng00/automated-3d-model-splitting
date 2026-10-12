from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from split3mf.interface_shape import classify_interface_shape


def ring(angles, height):
    return np.column_stack((10*np.cos(angles), 10*np.sin(angles), height))


def test_isolated_edge_noise_does_not_turn_planar_contact_into_curve():
    angles = np.arange(128) * 2*np.pi/128
    z = .01 * np.sin(3*angles)
    z[3] = .23
    result = classify_interface_shape(ring(angles, z))
    assert result["classification"] == "planar"
    assert result["distance_max_mm"] > .2


def test_small_sustained_noise_remains_planar():
    angles = np.arange(128)*2*np.pi/128
    assert classify_interface_shape(ring(angles, .12*np.cos(2*angles)))["classification"] == "planar"


def test_large_sustained_three_dimensional_bow_is_curved():
    angles = np.arange(256) * 2*np.pi/256
    points = np.column_stack((10*np.cos(angles), 10*np.sin(angles), 5*np.cos(2*angles)))
    result = classify_interface_shape(points)
    assert result["classification"] == "curved"
    assert result["relative_sagitta_p90"] >= .20


def test_one_sided_planar_step_is_not_curved():
    angles = np.arange(256) * 2*np.pi/256
    z = np.where(np.cos(angles) > 0, 0.8, 0.0)
    assert classify_interface_shape(ring(angles, z))["classification"] == "planar"


def test_gently_bowed_loop_is_not_curved():
    angles = np.arange(128) * 2*np.pi/128
    points = ring(angles, .3*np.cos(2*angles))
    assert classify_interface_shape(points)["classification"] == "planar"


def test_rotation_and_uneven_tessellation_do_not_change_classification():
    coarse = np.arange(128) * 2*np.pi/128
    dense = np.sort(np.concatenate((coarse, np.linspace(.01,.2,500))))
    rotation, _ = np.linalg.qr(np.array([[1., 2., 3.], [2., 0., 1.], [0., 1., 2.]]))
    for angles in (coarse, dense):
        for amplitude, expected in [(0., "planar"), (.3, "planar"), (5., "curved")]:
            points = ring(angles, amplitude*np.cos(2*angles)) @ rotation + [100, 40, -30]
            assert classify_interface_shape(points)["classification"] == expected
