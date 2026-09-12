"""Deterministic planar, equal-arc-length boundary reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_EPSILON = 1e-12
_BINOMIAL5 = np.asarray([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float64) / 16.0


class PlanarArcError(ValueError):
    """Raised when a boundary cannot be reconstructed without guessing."""


class CurveClarityRequired(PlanarArcError):
    """A new contour must be reviewed/remeshed, never assigned to old IDs."""
    def __init__(self, source, target, proposal, basis):
        self.source, self.target = source.copy(), target.copy()
        self.proposal, self.basis = proposal, basis
        super().__init__('Projected target has crossing lobes; clear-curve proposal '
                         'requires review and surface remeshing, not vertex-ID reassignment')


@dataclass(frozen=True)
class PlanarArcBoundary:
    source_vertex_ids: tuple[int, ...]
    source_points: np.ndarray
    target_points: np.ndarray
    guide_points: np.ndarray
    plane_origin: np.ndarray
    plane_u: np.ndarray
    plane_v: np.ndarray
    plane_normal: np.ndarray
    record: dict


def canonical_order(source_vertex_ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(source_vertex_ids, dtype=np.int64).reshape(-1)
    if len(ids) < 2:
        return np.arange(len(ids), dtype=np.int64)
    start = int(np.argmin(ids))
    forward = np.asarray([(start + offset) % len(ids) for offset in range(len(ids))])
    reverse = np.asarray([(start - offset) % len(ids) for offset in range(len(ids))])
    return forward if tuple(ids[forward]) <= tuple(ids[reverse]) else reverse


def _stable_sign(axis: np.ndarray) -> np.ndarray:
    result = np.asarray(axis, dtype=np.float64).copy()
    pivot = int(np.argmax(np.abs(result)))
    if result[pivot] < 0.0:
        result *= -1.0
    return result


def fit_stable_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 3:
        raise PlanarArcError("planar-arc boundary requires at least three 3-D points")
    origin = values.mean(axis=0)
    _left, singular_values, axes = np.linalg.svd(values - origin, full_matrices=False)
    if len(singular_values) < 2 or float(singular_values[1]) <= _EPSILON:
        raise PlanarArcError("planar-arc boundary is collinear or numerically unstable")
    u = _stable_sign(axes[0])
    normal = _stable_sign(axes[2])
    v = np.cross(normal, u)
    v /= max(float(np.linalg.norm(v)), _EPSILON)
    normal = np.cross(u, v)
    normal /= max(float(np.linalg.norm(normal)), _EPSILON)
    return origin, u, v, normal


def _closed_arc(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    edge_lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(edge_lengths)))
    total = float(cumulative[-1])
    if total <= _EPSILON:
        raise PlanarArcError("planar-arc boundary has zero perimeter")
    return edge_lengths, cumulative, total


def sample_closed_curve(points: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    _lengths, cumulative, total = _closed_arc(values)
    distances = np.mod(np.asarray(fractions, dtype=np.float64), 1.0) * total
    indices = np.searchsorted(cumulative, distances, side="right") - 1
    indices = np.clip(indices, 0, len(values) - 1)
    widths = cumulative[indices + 1] - cumulative[indices]
    blend = np.divide(
        distances - cumulative[indices],
        widths,
        out=np.zeros_like(distances),
        where=widths > _EPSILON,
    )
    return values[indices] + (
        values[(indices + 1) % len(values)] - values[indices]
    ) * blend[:, None]


def equal_arc_samples(points: np.ndarray, sample_count: int) -> np.ndarray:
    count = int(sample_count)
    if count < 16:
        raise PlanarArcError("boundary target sample count must be at least 16")
    return sample_closed_curve(points, np.arange(count, dtype=np.float64) / count)


def sample_values_on_curve(
    curve_points: np.ndarray,
    values: np.ndarray,
    fractions: np.ndarray,
) -> np.ndarray:
    """Interpolate arbitrary values by the physical arc of another curve."""

    curve = np.asarray(curve_points, dtype=np.float64)
    samples = np.asarray(values, dtype=np.float64)
    if len(curve) != len(samples):
        raise PlanarArcError("curve points and carried values must have equal length")
    _lengths, cumulative, total = _closed_arc(curve)
    distances = np.mod(np.asarray(fractions, dtype=np.float64), 1.0) * total
    indices = np.searchsorted(cumulative, distances, side="right") - 1
    indices = np.clip(indices, 0, len(curve) - 1)
    widths = cumulative[indices + 1] - cumulative[indices]
    blend = np.divide(
        distances - cumulative[indices],
        widths,
        out=np.zeros_like(distances),
        where=widths > _EPSILON,
    )
    return samples[indices] + (
        samples[(indices + 1) % len(samples)] - samples[indices]
    ) * blend[..., None] if samples.ndim > 1 else samples[indices] + (
        samples[(indices + 1) % len(samples)] - samples[indices]
    ) * blend


def cyclic_binomial_smooth(points: np.ndarray, passes: int) -> np.ndarray:
    result = np.asarray(points, dtype=np.float64).copy()
    for _index in range(max(int(passes), 0)):
        result = sum(
            np.roll(result, shift, axis=0) * weight
            for shift, weight in zip((-2, -1, 0, 1, 2), _BINOMIAL5)
        )
    return result


def _periodic_cubic_bspline_matrix(
    fractions: np.ndarray,
    control_count: int,
) -> np.ndarray:
    """Build the uniform periodic cubic B-spline design matrix."""

    count = int(control_count)
    if count < 4:
        raise PlanarArcError("periodic cubic B-spline requires at least four controls")
    phase = np.mod(np.asarray(fractions, dtype=np.float64), 1.0) * count
    spans = np.floor(phase).astype(np.int64)
    t = phase - spans
    weights = np.column_stack(
        (
            (1.0 - t) ** 3 / 6.0,
            (3.0 * t**3 - 6.0 * t**2 + 4.0) / 6.0,
            (-3.0 * t**3 + 3.0 * t**2 + 3.0 * t + 1.0) / 6.0,
            t**3 / 6.0,
        )
    )
    matrix = np.zeros((len(phase), count), dtype=np.float64)
    rows = np.arange(len(phase), dtype=np.int64)
    for offset, column_offset in enumerate((-1, 0, 1, 2)):
        matrix[rows, (spans + column_offset) % count] += weights[:, offset]
    return matrix


def fit_periodic_cubic_bspline(
    samples: np.ndarray,
    control_count: int,
) -> np.ndarray:
    """Least-squares fit a compact periodic cubic B-spline to cyclic samples."""

    values = np.asarray(samples, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    count = min(max(int(control_count), 4), len(values))
    fractions = np.arange(len(values), dtype=np.float64) / len(values)
    matrix = _periodic_cubic_bspline_matrix(fractions, count)
    controls, _residuals, _rank, _singular = np.linalg.lstsq(
        matrix,
        values,
        rcond=None,
    )
    return controls


def sample_periodic_cubic_bspline(
    controls: np.ndarray,
    fractions: np.ndarray,
) -> np.ndarray:
    """Evaluate a fitted periodic cubic B-spline at cyclic phases."""

    values = np.asarray(controls, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    return _periodic_cubic_bspline_matrix(fractions, len(values)) @ values


def _ripple_rms(points: np.ndarray) -> float:
    values = np.asarray(points, dtype=np.float64)
    residual = values - sum(
        np.roll(values, shift, axis=0) * weight
        for shift, weight in zip((-2, -1, 0, 1, 2), _BINOMIAL5)
    )
    return float(np.sqrt(np.mean(np.einsum("ij,ij->i", residual, residual))))


def build_planar_arc_boundary(
    points: np.ndarray,
    source_vertex_ids: np.ndarray,
    *,
    target_samples: int = 384,
    smooth_passes: int = 28,
    planar_control_points: int = 24,
    height_control_points: int = 8,
    maximum_target_offset_mm: float | None = None,
) -> PlanarArcBoundary:
    original = np.asarray(points, dtype=np.float64)
    source_ids = np.asarray(source_vertex_ids, dtype=np.int64)
    if len(original) != len(source_ids):
        raise PlanarArcError("boundary points and source ids must have equal length")
    order = canonical_order(source_ids)
    canonical = original[order]
    canonical_ids = source_ids[order]
    origin, u, v, normal = fit_stable_plane(canonical)
    relative = canonical - origin
    projected = np.column_stack((relative @ u, relative @ v))
    heights = relative @ normal
    equal_fractions = np.arange(int(target_samples), dtype=np.float64) / int(
        target_samples
    )
    equal = equal_arc_samples(projected, target_samples)
    equal_heights = sample_values_on_curve(projected, heights, equal_fractions)
    if len(canonical) == 3:
        # A three-edge closed loop is already the smallest valid polygon.  A
        # periodic cubic spline needs at least four controls and inventing a
        # fourth point would move a user-approved micro-feature.  Preserve the
        # exact triangle while still supplying a dense guide for downstream
        # diagnostics and keeping every topology/Boolean audit active.
        guide_2d = equal
        guide_heights = equal_heights
        guide_3d = (
            origin[None, :]
            + guide_2d[:, 0, None] * u[None, :]
            + guide_2d[:, 1, None] * v[None, :]
            + guide_heights[:, None] * normal[None, :]
        )
        return PlanarArcBoundary(
            source_vertex_ids=tuple(int(value) for value in canonical_ids),
            source_points=original.copy(),
            target_points=original.copy(),
            guide_points=guide_3d,
            plane_origin=origin,
            plane_u=u,
            plane_v=v,
            plane_normal=normal,
            record={
                "mode": "planar-arc-retopology",
                "status": "exact-minimal-loop",
                "source_vertices": 3,
                "target_samples": int(target_samples),
                "smooth_passes": 0,
                "maximum_target_offset_mm": 0.0,
                "rms_target_offset_mm": 0.0,
                "projected_ripple_before_mm": _ripple_rms(equal),
                "projected_ripple_after_mm": _ripple_rms(equal),
                "normal_height_ripple_before_mm": _ripple_rms(
                    equal_heights[:, None]
                ),
                "normal_height_ripple_after_mm": _ripple_rms(
                    equal_heights[:, None]
                ),
                "guide_model": "exact-minimal-triangle-loop",
                "planar_control_points": 3,
                "height_control_points": 3,
                "maximum_normal_height_change_mm": 0.0,
                "projected_extents_restored": False,
                "projected_extent_delta_mm": (0.0, 0.0),
                "source_order_preserved": True,
                "source_arc_parameters_preserved": True,
                "boundary_vertex_distribution": "exact-minimal-loop",
                "guide_sampling_parameterization": "equal-source-arc-phase",
                "guide_interpolation": "piecewise-linear-minimal-loop",
                "arc_parameter_smooth_passes": 0,
                "maximum_arc_parameter_change": 0.0,
                "surface_restore_policy": "exact-minimal-loop-preserved",
                "minimal_loop_preserved": True,
            },
        )
    planar_count = len(equal) if int(smooth_passes) <= 0 else planar_control_points
    height_count = (
        len(equal_heights)
        if int(smooth_passes) <= 0
        else height_control_points
    )
    planar_controls = fit_periodic_cubic_bspline(equal, planar_count)
    height_controls = fit_periodic_cubic_bspline(equal_heights, height_count)
    guide_2d = sample_periodic_cubic_bspline(
        planar_controls,
        equal_fractions,
    )
    guide_heights = sample_periodic_cubic_bspline(
        height_controls,
        equal_fractions,
    )[:, 0]
    # The reconstructed boundary must not reuse microscopic vendor rim edges.
    # Fully uniform redistribution can move semantic source ids several
    # millimetres around a deliberately nonuniform loop, so regularize the
    # physical edge fractions locally instead.  This removes isolated short
    # intervals while retaining the loop's large-scale sampling density.
    edge_lengths, cumulative, total = _closed_arc(projected)
    source_fractions = cumulative[:-1] / total
    parameter_pass_limit = 8 if len(canonical) <= 300 else 2
    parameter_smooth_passes = min(
        max(int(smooth_passes), 0), parameter_pass_limit
    )
    regularized_edges = cyclic_binomial_smooth(
        (edge_lengths / total)[:, None], parameter_smooth_passes
    )[:, 0]
    regularized_edges /= max(float(regularized_edges.sum()), _EPSILON)
    target_fractions = np.concatenate(
        ([0.0], np.cumsum(regularized_edges, dtype=np.float64))
    )[:-1]
    # ``guide_2d`` and ``guide_heights`` are indexed by equal physical arc on
    # the *source*.  Keep that native phase after smoothing.  Sampling them by
    # the guide's changed geometric arc length would apply a second,
    # unrelated parameterization and can pair a source vertex with a distant
    # location on an otherwise correct target curve.
    target_2d = sample_periodic_cubic_bspline(planar_controls, target_fractions)
    target_heights = sample_periodic_cubic_bspline(
        height_controls,
        target_fractions,
    )[:, 0]
    canonical_target = (
        origin[None, :]
        + target_2d[:, 0, None] * u[None, :]
        + target_2d[:, 1, None] * v[None, :]
        + target_heights[:, None] * normal[None, :]
    )
    target = np.empty_like(canonical_target)
    target[order] = canonical_target
    from .curve_clarity import propose_clear_curve
    clarity = propose_clear_curve(target, origin, u, v, normal)
    if clarity.record['status'] != 'clear_direct':
        raise CurveClarityRequired(original, target, clarity, (origin, u, v, normal))
    displacement = np.linalg.norm(target - original, axis=1)
    maximum_displacement = float(displacement.max(initial=0.0))
    if (
        maximum_target_offset_mm is not None
        and maximum_displacement > float(maximum_target_offset_mm) + 1e-9
    ):
        raise PlanarArcError(
            "planar-arc target leaves its retopology band: "
            f"required={maximum_displacement:.3f} mm, "
            f"allowed={float(maximum_target_offset_mm):.3f} mm"
        )
    guide_3d = (
        origin[None, :]
        + guide_2d[:, 0, None] * u[None, :]
        + guide_2d[:, 1, None] * v[None, :]
        + guide_heights[:, None] * normal[None, :]
    )
    return PlanarArcBoundary(
        source_vertex_ids=tuple(int(value) for value in canonical_ids),
        source_points=original.copy(),
        target_points=target,
        guide_points=guide_3d,
        plane_origin=origin,
        plane_u=u,
        plane_v=v,
        plane_normal=normal,
        record={
            "mode": "planar-arc-retopology",
            "status": "solved",
            "source_vertices": int(len(original)),
            "target_samples": int(target_samples),
            "smooth_passes": int(smooth_passes),
            "maximum_target_offset_mm": maximum_displacement,
            "rms_target_offset_mm": float(np.sqrt(np.mean(displacement * displacement))),
            "projected_ripple_before_mm": _ripple_rms(equal),
            "projected_ripple_after_mm": _ripple_rms(guide_2d),
            "normal_height_ripple_before_mm": _ripple_rms(
                equal_heights[:, None]
            ),
            "normal_height_ripple_after_mm": _ripple_rms(
                guide_heights[:, None]
            ),
            "guide_model": "least-squares-periodic-cubic-b-spline",
            "planar_control_points": int(len(planar_controls)),
            "height_control_points": int(len(height_controls)),
            "maximum_normal_height_change_mm": float(
                np.max(np.abs(target_heights - heights), initial=0.0)
            ),
            "projected_extents_restored": False,
            "projected_extent_delta_mm": tuple(
                float(value)
                for value in np.ptp(guide_2d, axis=0) - np.ptp(equal, axis=0)
            ),
            "source_order_preserved": True,
            "source_arc_parameters_preserved": bool(parameter_smooth_passes == 0),
            "boundary_vertex_distribution": (
                "locally-regularized-physical-arc"
                if parameter_smooth_passes > 0
                else "source-physical-arc"
            ),
            "guide_sampling_parameterization": "equal-source-arc-phase",
            "guide_interpolation": "direct-periodic-cubic-b-spline",
            "arc_parameter_smooth_passes": int(parameter_smooth_passes),
            "maximum_arc_parameter_change": float(
                np.max(np.abs(target_fractions - source_fractions), initial=0.0)
            ),
            "surface_restore_policy": "stable-plane-full-3d-smoothed-height",
        },
    )
