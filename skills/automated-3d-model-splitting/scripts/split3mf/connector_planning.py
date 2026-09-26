"""Pure planning policy for local painted-part connectors.

The meshing pipeline consumes the dictionaries returned here for backward
compatibility, while tests and harnesses can exercise depth policy without
loading the 3MF splitting state machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .common import (
    MAXIMUM_SAFE_INWARD_DEPTH_MM,
    PARENT_THICKNESS_CLEARANCE_MM,
)
from .hidden_interface import (
    MAX_BOUNDARY_THICKNESS_PROBES,
    boundary_screening_indices,
)
from .local_connectors import LocalConnectorSpec, plan_local_connector


class ThicknessProbe(Protocol):
    """Small interface needed by footprint safety planning."""

    def safety_limit(
        self,
        points: np.ndarray,
        directions: np.ndarray,
        requested_depth_mm: float,
    ) -> tuple[float, dict]: ...

    def first_hit_distances(
        self,
        points: np.ndarray,
        directions: np.ndarray,
        search_limit_mm: float,
    ) -> np.ndarray: ...


def measure_boundary_depth_angle_fan(
    *,
    boundary_points: np.ndarray,
    planar_center: np.ndarray,
    inward_normal: np.ndarray,
    parent_thickness_probe: ThicknessProbe,
    search_limit_mm: float,
    angles_degrees: np.ndarray | None = None,
) -> tuple[float, dict]:
    """Measure clearance along inward rays aimed at the loop center.

    Each ray direction lies between the inward normal and projected direction
    to the planar center. The fan angle is measured from the inward normal.
    A ray with no parent-shell hit within the finite search distance is safe
    through that complete requested distance.
    """

    points = np.asarray(boundary_points, dtype=np.float64)
    center = np.asarray(planar_center, dtype=np.float64)
    normal = np.asarray(inward_normal, dtype=np.float64).copy()
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("angular depth search requires at least three boundary samples")
    if center.shape != (3,) or normal.shape != (3,):
        raise ValueError("angular depth search center and normal must be 3-D vectors")
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    angles = np.asarray(
        np.linspace(30.0, 75.0, 10)
        if angles_degrees is None
        else angles_degrees,
        dtype=np.float64,
    ).reshape(-1)
    if not len(angles) or np.any((angles < 30.0) | (angles > 75.0)):
        raise ValueError("angular depth search angles must remain in 30-75 degrees")
    centerward = center[None, :] - points
    centerward -= (centerward @ normal)[:, None] * normal[None, :]
    lengths = np.linalg.norm(centerward, axis=1)
    if np.any(lengths <= 1e-9):
        raise ValueError("some boundary samples have no planar direction toward the center")
    centerward /= lengths[:, None]
    radians = np.deg2rad(angles)
    directions = (
        np.cos(radians)[None, :, None] * normal[None, None, :]
        + np.sin(radians)[None, :, None] * centerward[:, None, :]
    )
    origins = np.repeat(points[:, None, :], len(angles), axis=1)
    ray_points = origins.reshape((-1, 3))
    ray_directions = directions.reshape((-1, 3))
    ceiling = max(float(search_limit_mm), 0.0)
    hits = np.asarray(
        parent_thickness_probe.first_hit_distances(
            ray_points,
            ray_directions,
            ceiling,
        ),
        dtype=np.float64,
    ).reshape((len(points), len(angles)))
    hit_mask = hits < ceiling + PARENT_THICKNESS_CLEARANCE_MM - 1e-9
    safe_depths = np.where(
        hit_mask,
        np.maximum(0.0, hits - PARENT_THICKNESS_CLEARANCE_MM),
        ceiling,
    )
    best_angle_indices = np.argmax(safe_depths, axis=1)
    row_indices = np.arange(len(points))
    best_depths = safe_depths[row_indices, best_angle_indices]
    best_hit = hit_mask[row_indices, best_angle_indices]
    minimum_depth = float(best_depths.min(initial=ceiling))
    return minimum_depth, {
        "boundary_depth_search_policy": "equal_arc_boundary_centerward_inward_normal_angle_fan",
        "boundary_depth_search_sample_count": int(len(points)),
        "boundary_depth_search_angles_degrees": [float(value) for value in angles],
        "boundary_depth_search_reference": "angle_between_ray_and_local_inward_normal",
        "boundary_depth_search_center_direction": "projected_toward_planar_loop_center",
        "boundary_depth_search_distance_limit_mm": float(ceiling),
        "boundary_depth_search_unhit_is_safe_to_limit": True,
        "boundary_depth_search_ray_count": int(len(ray_points)),
        "boundary_depth_search_hit_ray_count": int(np.count_nonzero(hit_mask)),
        "boundary_depth_search_miss_ray_count": int(np.count_nonzero(~hit_mask)),
        "boundary_depth_search_selected_hit_count": int(np.count_nonzero(best_hit)),
        "boundary_depth_search_selected_miss_count": int(np.count_nonzero(~best_hit)),
        "boundary_depth_search_minimum_best_depth_mm": minimum_depth,
        "boundary_depth_search_selected_angles_degrees": [
            float(angles[index]) for index in best_angle_indices
        ],
        "boundary_depth_search_selected_depths_mm": [
            float(value) for value in best_depths
        ],
        "boundary_depth_search_selected_hit_mask": [bool(value) for value in best_hit],
    }


@dataclass(frozen=True)
class ConnectorDepthPolicy:
    """Manufacturing policy shared by production, tests, and harnesses."""

    maximum_engagement_depth_mm: float = 5.0
    maximum_total_depth_mm: float = 8.25
    printable_backing_depth_mm: float = 3.0
    minimum_engagement_depth_mm: float = 0.45
    minimum_elastic_backing_depth_mm: float = 0.45
    minimum_elastic_peg_width_mm: float = 1.20
    elastic_lateral_scale_exponent: float = 1.00
    robust_safety_percentile: float = 25.0


DEFAULT_DEPTH_POLICY = ConnectorDepthPolicy()


def local_connector_spec_for_interface(
    *,
    fit_clearance_mm: float,
    bottom_clearance_mm: float,
    lead_in_mm: float,
    safe_engagement_depth_mm: float,
    safe_backing_depth_mm: float | None = None,
    compact_peg_supported: bool = True,
    slope_validation_mode: str = "strict",
    surface_validation_mode: str = "strict",
    policy: ConnectorDepthPolicy = DEFAULT_DEPTH_POLICY,
) -> LocalConnectorSpec:
    """Convert separate rim/backing and compact-footprint budgets to one joint.

    The full-boundary 45-degree backing is limited by the thinnest painted rim.
    The compact peg is limited by a second thickness probe at its much smaller
    footprint.  Allocate the measured total depth in manufacturing-priority
    order: bottom clearance, the preferred three-millimetre backing, then up to
    five millimetres of compact-peg engagement.  Engagement may shrink to zero;
    backing shrinks only after that budget is exhausted or when the rim itself
    is thinner than the preferred backing.
    """

    nominal_backing_depth_mm = float(policy.printable_backing_depth_mm)
    total_safe_depth_mm = max(float(safe_engagement_depth_mm), 0.0)
    if compact_peg_supported:
        total_safe_depth_mm = min(
            total_safe_depth_mm,
            float(policy.maximum_total_depth_mm),
        )
    # A tiny interface can still carry a printable full-boundary backing even
    # when no protected compact peg footprint fits.  With no peg there is no
    # socket bottom, so reserving its clearance would only steal useful
    # backing thickness from the retained semantic detail.
    socket_bottom_clearance_mm = (
        max(float(bottom_clearance_mm), 0.0)
        if compact_peg_supported
        else 0.0
    )
    structural_depth_budget_mm = max(
        total_safe_depth_mm - socket_bottom_clearance_mm,
        0.0,
    )
    backing_safety_limit_mm = min(
        structural_depth_budget_mm,
        float(
            structural_depth_budget_mm
            if safe_backing_depth_mm is None
            else safe_backing_depth_mm
        ),
    )
    backing_depth_mm = (
        min(nominal_backing_depth_mm, backing_safety_limit_mm)
        if compact_peg_supported
        else backing_safety_limit_mm
    )
    backing_elastic_shrink_applied = bool(
        backing_depth_mm < nominal_backing_depth_mm - 1e-9
    )
    # The elastic minimum protects a load-bearing peg/backing joint.  When the
    # compact footprint cannot support a peg, the geometry is only a sealed
    # axial backing for retaining the painted semantic detail.  In that mode a
    # thinner, positively measured backing is safer than inventing unavailable
    # material or discarding the child altogether.
    effective_minimum_backing_depth_mm = (
        float(policy.minimum_elastic_backing_depth_mm)
        if compact_peg_supported
        else 0.0
    )
    if (
        backing_elastic_shrink_applied
        and backing_depth_mm
        < effective_minimum_backing_depth_mm - 1e-9
    ):
        raise ValueError(
            "local connector rim is too thin for the minimum elastic backing: "
            f"available={backing_depth_mm:.3f} mm, "
            f"minimum={float(policy.minimum_elastic_backing_depth_mm):.3f} mm"
        )
    nominal_engagement_depth_mm = float(policy.maximum_engagement_depth_mm)
    engagement_depth_mm = (
        min(
            nominal_engagement_depth_mm,
            max(structural_depth_budget_mm - backing_depth_mm, 0.0),
        )
        if compact_peg_supported
        else 0.0
    )
    # Sub-nozzle-depth pegs are numerically non-zero but mechanically fictional.
    # Omit both peg and socket, release the unused socket-bottom allowance, and
    # give that measured material back to the printable full-boundary backing.
    minimum_printable_engagement_mm = max(
        float(policy.minimum_engagement_depth_mm),
        0.0,
    )
    if engagement_depth_mm < minimum_printable_engagement_mm - 1e-9:
        compact_peg_supported = False
        socket_bottom_clearance_mm = 0.0
        structural_depth_budget_mm = total_safe_depth_mm
        backing_safety_limit_mm = min(
            structural_depth_budget_mm,
            float(
                structural_depth_budget_mm
                if safe_backing_depth_mm is None
                else safe_backing_depth_mm
            ),
        )
        backing_depth_mm = min(nominal_backing_depth_mm, backing_safety_limit_mm)
        backing_elastic_shrink_applied = bool(
            backing_depth_mm < nominal_backing_depth_mm - 1e-9
        )
        effective_minimum_backing_depth_mm = 0.0
        engagement_depth_mm = 0.0
    engagement_elastic_shrink_applied = bool(
        engagement_depth_mm < nominal_engagement_depth_mm - 1e-9
    )
    elastic_shrink_applied = bool(
        backing_elastic_shrink_applied or engagement_elastic_shrink_applied
    )
    backing_scale = min(
        1.0,
        backing_depth_mm / max(nominal_backing_depth_mm, 1e-12),
    )
    minimum_lateral_scale = max(
        float(policy.minimum_elastic_peg_width_mm) / 4.0,
        0.0,
    )
    lateral_scale = (
        1.0
        if not backing_elastic_shrink_applied
        else max(
            minimum_lateral_scale,
            backing_scale ** float(policy.elastic_lateral_scale_exponent),
        )
    )
    return LocalConnectorSpec(
        peg_width_mm=4.0 * lateral_scale,
        peg_length_mm=4.0 * lateral_scale,
        engagement_depth_mm=float(engagement_depth_mm),
        total_clearance_mm=max(float(fit_clearance_mm), 0.0),
        socket_bottom_clearance_mm=socket_bottom_clearance_mm,
        socket_mouth_chamfer_mm=max(float(lead_in_mm), 0.05),
        # The compact peg is vertical with a flat tip.  The 45-degree geometry
        # belongs to the full-boundary backing and female socket mouth.
        peg_tip_chamfer_mm=0.0,
        corner_radius_mm=0.45 * lateral_scale,
        full_boundary_backing_depth_mm=backing_depth_mm,
        elastic_shrink_applied=elastic_shrink_applied,
        backing_elastic_shrink_applied=backing_elastic_shrink_applied,
        engagement_elastic_shrink_applied=engagement_elastic_shrink_applied,
        elastic_backing_scale=backing_scale,
        elastic_engagement_scale=(
            engagement_depth_mm / max(nominal_engagement_depth_mm, 1e-12)
        ),
        elastic_lateral_scale=lateral_scale,
        nominal_full_boundary_backing_depth_mm=nominal_backing_depth_mm,
        nominal_engagement_depth_mm=nominal_engagement_depth_mm,
        minimum_elastic_backing_depth_mm=effective_minimum_backing_depth_mm,
        backing_safety_limit_mm=backing_safety_limit_mm,
        total_safety_limit_mm=total_safe_depth_mm,
        compact_peg_enabled=bool(
            compact_peg_supported and engagement_depth_mm > 1e-9
        ),
        slope_validation_mode=str(slope_validation_mode),
        surface_validation_mode=str(surface_validation_mode),
    )


def local_connector_safe_depth_from_field(
    distances: np.ndarray,
    *,
    percentile: float | None = None,
    policy: ConnectorDepthPolicy = DEFAULT_DEPTH_POLICY,
) -> dict:
    """Select a robust budget without letting one thin rim point veto it."""

    selected_percentile = (
        float(policy.robust_safety_percentile)
        if percentile is None
        else float(percentile)
    )
    values = np.asarray(distances, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values) & (values > 0.0)]
    if len(values) == 0:
        raise ValueError("local connector safety field has no positive finite values")
    minimum = float(values.min())
    robust = float(np.percentile(values, selected_percentile))
    selected = min(float(policy.maximum_total_depth_mm), max(minimum, robust))
    return {
        "boundary_safety_minimum_mm": minimum,
        "local_connector_safety_percentile": selected_percentile,
        "local_connector_safety_percentile_mm": robust,
        "local_connector_safety_budget_mm": selected,
        "thin_boundary_override_applied": bool(selected > minimum + 1e-9),
    }


def local_connector_safe_depth_at_boundary(
    *,
    boundary_points: np.ndarray,
    inward: np.ndarray,
    fit_clearance_mm: float,
    bottom_clearance_mm: float,
    lead_in_mm: float,
    boundary_distances: np.ndarray,
    parent_thickness_probe: ThicknessProbe | None,
    policy: ConnectorDepthPolicy = DEFAULT_DEPTH_POLICY,
) -> dict:
    """Measure axial depth on boundary samples and lateral clearance separately.

    ``boundary_distances`` remains useful evidence for the ordinary cap, but a
    single concave rim ray may hit a nearby side wall long before the true
    opposing parent shell. Probe equal-arc samples of the actual interface
    boundary along the stable assembly axis to measure the orange surface-to-
    interface depth. Keep projected polygon clearance as the independent
    lateral constraint; connector footprint points are not thickness origins.
    """

    boundary = np.asarray(boundary_points, dtype=np.float64)
    boundary_sample_indices = boundary_screening_indices(
        boundary,
        MAX_BOUNDARY_THICKNESS_PROBES,
    )
    sampled_boundary_distances = np.asarray(boundary_distances, dtype=np.float64)
    if len(sampled_boundary_distances) == len(boundary):
        sampled_boundary_distances = sampled_boundary_distances[boundary_sample_indices]
    elif len(sampled_boundary_distances) != len(boundary_sample_indices):
        raise ValueError(
            "boundary thickness field must align with either the source loop "
            "or its equal-arc sample points"
        )
    boundary_safety = local_connector_safe_depth_from_field(
        sampled_boundary_distances,
        policy=policy,
    )
    if parent_thickness_probe is None:
        return {
            **boundary_safety,
            "local_connector_boundary_probe_applied": False,
            "local_connector_compact_peg_supported": True,
        }
    provisional = local_connector_spec_for_interface(
        fit_clearance_mm=fit_clearance_mm,
        bottom_clearance_mm=bottom_clearance_mm,
        lead_in_mm=lead_in_mm,
        safe_engagement_depth_mm=float(policy.maximum_total_depth_mm),
        # This object exists only to locate the protected interior footprint.
        # Do not let one grazing source-rim ray fail the printable-backing gate
        # before that independent axial/lateral footprint is even measured.
        # The final allocator below still enforces the same 0.45 mm minimum
        # against the measured interior backing limit.
        safe_backing_depth_mm=None,
        policy=policy,
    )
    try:
        plan = plan_local_connector(
            boundary[boundary_sample_indices],
            inward,
            provisional,
            samples=32,
        )
        footprint_plan_fallback = False
    except ValueError:
        # A narrow interface may have enough room for the backing but not for a
        # provisional five-millimetre peg plus its protected land.  Engagement
        # is allowed to reach zero, so obtain a stable interior center without
        # inventing a compact footprint which the final allocator may omit.
        measurement_spec = LocalConnectorSpec(
            engagement_depth_mm=0.0,
            socket_bottom_clearance_mm=0.0,
            peg_tip_chamfer_mm=0.0,
            full_boundary_backing_depth_mm=0.0,
            compact_peg_enabled=False,
        )
        plan = plan_local_connector(
            boundary[boundary_sample_indices],
            inward,
            measurement_spec,
            samples=32,
        )
        footprint_plan_fallback = True
    axis = np.asarray(plan["inward"], dtype=np.float64)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    sampled_boundary = boundary[boundary_sample_indices]
    depth_ceiling = min(
        MAXIMUM_SAFE_INWARD_DEPTH_MM,
        max(
            float(policy.maximum_total_depth_mm),
            float(
                getattr(
                    parent_thickness_probe,
                    "maximum_probe_distance_mm",
                    policy.maximum_total_depth_mm,
                )
            ),
        ),
    )
    boundary_axis_safe, boundary_axis_record = measure_boundary_depth_angle_fan(
        boundary_points=sampled_boundary,
        planar_center=np.asarray(plan["center"], dtype=np.float64),
        inward_normal=axis,
        parent_thickness_probe=parent_thickness_probe,
        search_limit_mm=depth_ceiling,
    )
    selected = min(depth_ceiling, float(boundary_axis_safe))
    lateral_backing_limit = max(float(plan.get("edge_clearance_mm", 0.0)), 0.0)
    # Axial depth and lateral inset are different legs of the green sidewall.
    # Keep both limits independent; their jointly feasible slope is selected
    # later from the 30–75 degree boundary-sampled profile audit.
    backing_safety_limit = selected
    nominal_backing_depth = float(policy.printable_backing_depth_mm)
    limiting_tolerance = 1e-9
    if backing_safety_limit >= nominal_backing_depth - limiting_tolerance:
        backing_limiting_constraint = "nominal_printable_depth"
    elif selected <= 0.0:
        backing_limiting_constraint = "axial_boundary_depth"
    else:
        backing_limiting_constraint = "joint_slope_depth_profile"
    boundary_axis_record["connector_depth_origin"] = "ordered_interface_boundary"
    boundary_axis_record["connector_depth_path"] = (
        "interface_boundary_to_opposing_surface_along_shared_connector_axis"
    )
    return {
        **boundary_safety,
        "local_connector_safety_budget_mm": selected,
        "local_connector_backing_safety_limit_mm": backing_safety_limit,
        "local_connector_axial_safety_limit_mm": selected,
        "local_connector_lateral_backing_limit_mm": lateral_backing_limit,
        "local_connector_nominal_backing_depth_mm": nominal_backing_depth,
        "local_connector_backing_limiting_constraint": backing_limiting_constraint,
        "local_connector_parent_is_axially_thick_enough_for_nominal_backing": bool(
            selected >= nominal_backing_depth - limiting_tolerance
        ),
        "local_connector_interface_is_wide_enough_for_nominal_45_degree_backing": bool(
            lateral_backing_limit >= nominal_backing_depth - limiting_tolerance
        ),
        "local_connector_backing_safety_policy": (
            "sampled_interface_boundary_axis_depth_with_independent_lateral_clearance"
        ),
        "thin_boundary_override_applied": bool(
            backing_safety_limit
            > float(boundary_safety["boundary_safety_minimum_mm"]) + 1e-9
        ),
        "local_connector_boundary_probe_applied": True,
        "local_connector_footprint_plan_fallback": footprint_plan_fallback,
        "local_connector_compact_peg_supported": not footprint_plan_fallback,
        "local_connector_boundary_probe_input_point_count": int(len(boundary)),
        "local_connector_boundary_probe_sampled_point_count": int(
            boundary_axis_record.get("parent_thickness_boundary_sampled_vertices", len(boundary))
        ),
        "local_connector_boundary_safe_depth_mm": selected,
        "local_connector_boundary_depth_ceiling_mm": depth_ceiling,
        "local_connector_boundary_thickness_record": boundary_axis_record,
    }


class LocalConnectorPlanningService:
    """Stateless facade for dependency injection in workflows and harnesses."""

    spec_for_interface = staticmethod(local_connector_spec_for_interface)
    safe_depth_from_field = staticmethod(local_connector_safe_depth_from_field)
    safe_depth_at_boundary = staticmethod(local_connector_safe_depth_at_boundary)
