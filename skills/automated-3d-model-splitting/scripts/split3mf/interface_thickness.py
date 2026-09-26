from __future__ import annotations

from scipy.ndimage import convolve1d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from .cap_template import fit_affine_cap_inside_parent, progressive_boundary_deformation, refined_harmonic_heightfield_cap
import copy
import os
import time
from .common import *
from .print_tolerance import current as current_print_tolerance
from .project import *
from .recognition import *
from .mesh import *
from .package_io import *
from .selection import *
from .assembly import *
from .validation import *
from .interface_retopology import InterfaceRetopologyService, generated_geometry_boundary_vertices
from .domain import PlanarArcRetopologyContext, CapDecision
from .hidden_interface import HIDDEN_INTERFACE_MINIMUM_LOAD_BEARING_DEPTH_MM, MAX_BOUNDARY_THICKNESS_PROBES, HiddenInterfacePlanner, boundary_screening_indices
from .guided_internal_cut import GuidedInternalCutPlanner, GuidedInternalCutSpec, adaptive_guided_entry_ring
from .local_connectors import LocalConnectorSpec, build_socket_cutter_from_plan, plan_local_connector, subtract_socket_cutters
from .connector_planning import local_connector_safe_depth_at_boundary, local_connector_safe_depth_from_field, local_connector_spec_for_interface
from .connector_geometry import backing_taper_angle_audit as _backing_taper_angle_audit, line_preserving_inset_displacements as _line_preserving_inset_displacements, printable_backing_profile as _printable_backing_profile, printable_backing_rings as _printable_backing_rings, project_connector_points as _project_connector_points, user_reviewed_shallow_minimal_needle_advisory_is_eligible
from .connector_topology import orient_face_patch_consistently, triangulate_bounded_ring_strip, triangulate_connector_annulus
from .connector_surface import refine_connector_annulus_heightfield
from .mesh_finalization import finalize_source_preserving_mesh
from .reporting import runtime_log

def fit_plane_normal(points: np.ndarray, fallback_normal: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0)
    try:
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
        normal = np.asarray(vh[-1], dtype=np.float64)
    except np.linalg.LinAlgError:
        normal = np.asarray(fallback_normal, dtype=np.float64)
    if float(np.linalg.norm(normal)) <= 1e-12:
        normal = np.asarray(fallback_normal, dtype=np.float64)
    normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
    if float(np.dot(normal, fallback_normal)) < 0.0:
        normal = -normal
    return normal

def _expand_sampled_boundary_minima(
    sample_indices: np.ndarray,
    sampled_values: np.ndarray,
    boundary_count: int,
) -> np.ndarray:
    """Spread sampled safety limits over each intervening ordered boundary arc."""
    indices = np.asarray(sample_indices, dtype=np.int64)
    values = np.asarray(sampled_values, dtype=np.float64)
    expanded = np.full(int(boundary_count), np.inf, dtype=np.float64)
    if not len(indices):
        return expanded
    if len(indices) != len(values):
        raise ValueError("boundary sample indices and values must have matching lengths")
    if len(indices) == 1:
        expanded[:] = values[0]
        return expanded
    for position, start in enumerate(indices):
        next_position = (position + 1) % len(indices)
        end = int(indices[next_position])
        current = int(start)
        segment_limit = min(float(values[position]), float(values[next_position]))
        expanded[current] = min(expanded[current], segment_limit)
        while current != end:
            current = (current + 1) % int(boundary_count)
            expanded[current] = min(expanded[current], segment_limit)
    return expanded

def boundary_cap_distances(
    points: np.ndarray,
    fallback_inward: np.ndarray,
    inward_directions: np.ndarray,
    fixed_depth_mm: float,
    flat_clearance_mm: float,
    cap_mode: str,
    planar_extra_limit_mm: float,
    parent_thickness_probe: ParentThicknessProbe | None = None,
    safety_ceiling_mm: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Measure and generate depth along the shared interface inward axis."""
    points = np.asarray(points, dtype=np.float64)
    fallback = np.asarray(fallback_inward, dtype=np.float64)
    fallback /= max(float(np.linalg.norm(fallback)), 1e-12)
    local_plane_rays = np.asarray(inward_directions, dtype=np.float64)
    local_plane_rays /= np.maximum(
        np.linalg.norm(local_plane_rays, axis=1)[:, None],
        1e-12,
    )
    directions, smooth_direction_record = smooth_closed_inward_direction_field(
        local_plane_rays
    )
    if len(points) > MAX_BOUNDARY_THICKNESS_PROBES:
        boundary_sample_indices = boundary_screening_indices(
            points,
            MAX_BOUNDARY_THICKNESS_PROBES,
        )
    else:
        boundary_sample_indices = np.arange(len(points), dtype=np.int64)
    evaluation_points = points[boundary_sample_indices]
    evaluation_directions = directions[boundary_sample_indices]
    preferred_minimum = max(float(fixed_depth_mm), 0.4)
    global_ceiling = min(
        preferred_minimum + max(float(planar_extra_limit_mm), 0.0),
        MAXIMUM_SAFE_INWARD_DEPTH_MM,
    )
    if safety_ceiling_mm is not None:
        global_ceiling = min(
            global_ceiling,
            max(float(safety_ceiling_mm), 0.0),
        )

    def measured_limit(candidate_directions: np.ndarray) -> tuple[float, dict]:
        if parent_thickness_probe is None:
            return global_ceiling, {
                "parent_thickness_min_mm": global_ceiling
                + PARENT_THICKNESS_CLEARANCE_MM,
                "parent_thickness_hit_vertices": 0,
                "parent_thickness_probe_vertices": int(len(points)),
                "parent_thickness_is_lower_bound": True,
                "parent_thickness_clearance_mm": PARENT_THICKNESS_CLEARANCE_MM,
                "safe_maximum_inward_depth_mm": global_ceiling,
                "parent_thickness_depth_path": (
                    "interface_boundary_to_opposing_surface_along_shared_component_inward_axis"
                ),
                "parent_thickness_depth_direction": fallback.round(9).tolist(),
            }
        return parent_thickness_probe.safety_limit(
            evaluation_points,
            candidate_directions,
            global_ceiling,
        )

    candidate_specs: list[tuple[str, np.ndarray]] = [("global_inward", fallback)]
    best_fit_normal = fit_plane_normal(evaluation_points, fallback)
    best_fit_dot_global = float(np.dot(best_fit_normal, fallback))
    if best_fit_dot_global >= 0.15:
        candidate_specs.append(("loop_best_fit_normal", best_fit_normal))

    # Measure depth along the shared component-inward axis: this is the path
    # from the interface boundary to the opposite exposed surface (the orange
    # dimension in the cross-section), not along each locally varying rim
    # normal. Use the same axis for cap travel so measured and generated depth
    # refer to the same physical line.
    depth_directions = np.tile(fallback, (len(evaluation_points), 1))

    def materialize_sampled_boundary_field(
        sampled_distances: np.ndarray,
        record: dict,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Expand sampled safety decisions onto the source loop for geometry creation.

        Thickness is evaluated only at equal-arc boundary samples.  The cap
        mesh still has one vertex per source boundary vertex, so propagate the
        more conservative neighboring sample limit across each intervening
        arc before returning the planned geometry arrays.
        """
        full_distances = _expand_sampled_boundary_minima(
            boundary_sample_indices,
            np.asarray(sampled_distances, dtype=np.float64),
            len(points),
        )
        full_directions = np.tile(fallback, (len(points), 1))
        record.update(
            {
                "depth_field_sampled_vertex_count": int(len(evaluation_points)),
                "depth_field_materialized_vertex_count": int(len(points)),
                "depth_field_materialization_policy": (
                    "conservative_neighbor_equal_arc_sample_minimum"
                ),
                "minimum_generated_inward_travel_mm": float(full_distances.min()),
                "maximum_generated_inward_travel_mm": float(full_distances.max()),
            }
        )
        return full_distances, full_directions, record

    shared_safe_maximum, shared_thickness_record = measured_limit(
        depth_directions
    )
    shared_thickness_record["parent_thickness_depth_path"] = (
        "interface_boundary_to_opposing_surface_along_shared_component_inward_axis"
    )
    shared_thickness_record["parent_thickness_depth_direction"] = (
        fallback.round(9).tolist()
    )
    plane_candidates = []
    for orientation, plane_direction in candidate_specs:
        # Keep cap travel on the same shared axis used by the thickness probe;
        # local rim normals remain available for seam shaping, not depth.
        plane_directions = depth_directions.copy()
        ray_dot_plane = plane_directions @ plane_direction
        if len(ray_dot_plane) and float(ray_dot_plane.min()) <= 0.05:
            continue
        safe_maximum = shared_safe_maximum
        thickness_record = dict(shared_thickness_record)
        if safe_maximum <= 1e-6:
            effective_minimum = 0.0
        else:
            effective_minimum = min(preferred_minimum, safe_maximum)
        values = evaluation_points @ plane_direction
        plane_s = float(np.min(values + safe_maximum * ray_dot_plane))
        distances = (plane_s - values) / ray_dot_plane
        required_maximum = (
            float(distances.max()) if len(distances) else effective_minimum
        )
        minimum_travel = (
            float(distances.min()) if len(distances) else effective_minimum
        )
        regular_feasible = (
            effective_minimum > 0.0
            and minimum_travel >= effective_minimum - 1e-9
            and required_maximum <= safe_maximum + 1e-9
        )
        thin_parent_limited = bool(
            safe_maximum < preferred_minimum - 1e-9
        )
        thin_parent_planar_minimum = min(
            safe_maximum,
            MINIMUM_COHERENT_PLANAR_FLOOR_DEPTH_MM,
        )
        thin_parent_planar_feasible = bool(
            thin_parent_limited
            and thin_parent_planar_minimum > 0.0
            and minimum_travel >= thin_parent_planar_minimum - 1e-9
            and required_maximum <= safe_maximum + 1e-9
        )
        feasible = bool(regular_feasible or thin_parent_planar_feasible)
        forced_planar_feasible = feasible or (
            preferred_minimum <= 0.4 + 1e-9
            and safe_maximum > 0.0
            and minimum_travel >= PARENT_THICKNESS_CLEARANCE_MM - 1e-9
            and required_maximum <= safe_maximum + 1e-9
        )
        plane_candidates.append(
            {
                "orientation": orientation,
                "direction": plane_direction,
                "directions": plane_directions,
                "distances": distances,
                "plane_s": plane_s,
                "span_mm": float(values.max() - values.min()) if len(values) else 0.0,
                "required_maximum_mm": required_maximum,
                "minimum_travel_mm": minimum_travel,
                "effective_minimum_mm": effective_minimum,
                "safe_maximum_mm": safe_maximum,
                "minimum_ray_dot_plane": (
                    float(ray_dot_plane.min()) if len(ray_dot_plane) else 1.0
                ),
                "regular_feasible": regular_feasible,
                "thin_parent_limited": thin_parent_limited,
                "thin_parent_planar_minimum_mm": thin_parent_planar_minimum,
                "thin_parent_planar_feasible": thin_parent_planar_feasible,
                "feasible": feasible,
                "forced_planar_feasible": forced_planar_feasible,
                "thickness_record": thickness_record,
            }
        )

    if cap_mode == "tilted":
        selectable = [
            candidate
            for candidate in plane_candidates
            if candidate["orientation"] == "loop_best_fit_normal"
            and candidate["forced_planar_feasible"]
        ]
    elif cap_mode == "flat":
        selectable = [
            candidate
            for candidate in plane_candidates
            if candidate["orientation"] == "global_inward"
            and candidate["forced_planar_feasible"]
        ]
    elif cap_mode == "offset":
        selectable = []
    else:
        selectable = [candidate for candidate in plane_candidates if candidate["feasible"]]

    if selectable:
        selected = min(
            selectable,
            key=lambda candidate: (
                -candidate["minimum_travel_mm"],
                0 if candidate["orientation"] == "global_inward" else 1,
            ),
        )
        distances = np.asarray(selected["distances"], dtype=np.float64)
        record = {
            "requested_cap_mode": cap_mode,
            "cap_mode": "flat",
            "flat_orientation": selected["orientation"],
            "cap_depth_reference": "interface_boundary_to_opposing_surface_shared_axis",
            "inward_depth_policy": "deepest_safe_depth_along_shared_component_axis",
            "preferred_minimum_inward_depth_mm": preferred_minimum,
            "effective_minimum_inward_depth_mm": min(
                selected["effective_minimum_mm"],
                selected["minimum_travel_mm"],
            ),
            "safe_maximum_inward_depth_mm": selected["safe_maximum_mm"],
            "flat_span_mm": selected["span_mm"],
            "flat_bottom_required_max_extension_mm": selected["required_maximum_mm"],
            "flat_bottom_plane_s": selected["plane_s"],
            "flat_plane_extension_max_mm": selected["required_maximum_mm"],
            "minimum_generated_inward_travel_mm": float(distances.min()),
            "maximum_generated_inward_travel_mm": float(distances.max()),
            "thin_parent_planar_floor_applied": bool(
                selected["thin_parent_planar_feasible"]
                and not selected["regular_feasible"]
            ),
            "thin_parent_planar_minimum_depth_mm": float(
                selected["thin_parent_planar_minimum_mm"]
            ),
            "thin_parent_planar_achieved_minimum_depth_mm": float(
                selected["minimum_travel_mm"]
            ),
            "flat_priority_applied": True,
            "flat_feasibility_basis": (
                "thin_parent_printable_planar_floor"
                if selected["thin_parent_planar_feasible"]
                and not selected["regular_feasible"]
                else "required_plane_depth_within_parent_thickness_safety_limit"
            ),
            "flat_travel_within_limit": True,
            "target_exceeded": bool(
                selected["required_maximum_mm"] > preferred_minimum + 1e-9
            ),
            "target_exceeded_by_mm": float(
                max(0.0, selected["required_maximum_mm"] - preferred_minimum)
            ),
            "fixed_inward_depth_mm": preferred_minimum,
            "fixed_inward_depth_applied": bool(
                selected["minimum_travel_mm"] >= preferred_minimum - 1e-9
            ),
            "local_direction_correction_applied": False,
            "local_direction_corrected_vertices": 0,
            "minimum_direction_dot_global": 1.0,
            "best_fit_flat_normal": best_fit_normal.round(6).tolist(),
            "best_fit_flat_normal_dot_global": best_fit_dot_global,
            "plane_candidate_required_maxima_mm": {
                candidate["orientation"]: candidate["required_maximum_mm"]
                for candidate in plane_candidates
            },
            **smooth_direction_record,
            **selected["thickness_record"],
        }
        return materialize_sampled_boundary_field(distances, record)

    if cap_mode in {"flat", "tilted"}:
        if not plane_candidates:
            raise ValueError(
                f"{cap_mode} cap cannot fit: no usable plane candidate"
            )
        attempted = (
            plane_candidates[0]
            if cap_mode == "flat"
            else next(
                (
                    candidate
                    for candidate in plane_candidates
                    if candidate["orientation"] == "loop_best_fit_normal"
                ),
                plane_candidates[0],
            )
        )
        if (
            preferred_minimum <= 0.4 + 1e-9
            and parent_thickness_probe is not None
            and len(points) >= 3
        ):
            uniform_directions = depth_directions
            sampled_raw_thicknesses = parent_thickness_probe.first_hit_distances(
                evaluation_points,
                depth_directions,
                global_ceiling + PARENT_THICKNESS_CLEARANCE_MM,
            )
            # Keep the normal absolute reserve wherever the source shell is
            # thick enough.  For user-authorized source walls thinner than the
            # reserve itself, preserve half of the measured thickness so both
            # sides remain positive; the closed-loop Lipschitz pass then spreads
            # that exceptional shallow point with a <=45 degree transition.
            sampled_reserve = np.where(
                sampled_raw_thicknesses < PARENT_THICKNESS_CLEARANCE_MM,
                sampled_raw_thicknesses * 0.5,
                PARENT_THICKNESS_CLEARANCE_MM,
            )
            sampled_safe = np.minimum(
                global_ceiling,
                np.maximum(0.0, sampled_raw_thicknesses - sampled_reserve),
            )
            raw_thicknesses = sampled_raw_thicknesses
            per_vertex_reserve = sampled_reserve
            per_vertex_safe = sampled_safe
            # ``per_vertex_safe`` has already had the full parent-thickness
            # clearance subtracted above.  Requiring another clearance-sized
            # extrusion here double-counts that reserve and rejects valid,
            # very shallow local relief points.  A strictly positive travel
            # is sufficient to keep the generated wall non-degenerate while
            # preserving the requested clearance from every measured hit.
            if float(per_vertex_safe.min()) > 1e-6:
                distances = per_vertex_safe.copy()
                edge_lengths = np.linalg.norm(
                    np.roll(evaluation_points, -1, axis=0) - evaluation_points,
                    axis=1,
                )
                for _ in range(3):
                    for index in range(1, len(distances)):
                        distances[index] = min(
                            distances[index],
                            distances[index - 1] + edge_lengths[index - 1],
                        )
                    distances[0] = min(
                        distances[0],
                        distances[-1] + edge_lengths[-1],
                    )
                    for index in range(len(distances) - 2, -1, -1):
                        distances[index] = min(
                            distances[index],
                            distances[index + 1] + edge_lengths[index],
                        )
                    distances[-1] = min(
                        distances[-1],
                        distances[0] + edge_lengths[-1],
                    )
                depth_deltas = np.abs(
                    np.roll(distances, -1) - distances
                )
                slope_degrees = np.degrees(
                    np.arctan2(
                        depth_deltas,
                        np.maximum(edge_lengths, 1e-12),
                    )
                )
                record = {
                    "requested_cap_mode": cap_mode,
                    "cap_mode": "uniform-direction-lipschitz",
                    "flat_orientation": attempted["orientation"],
                    "cap_depth_reference": "interface_boundary_to_opposing_surface_shared_axis",
                    "inward_depth_policy": "maximum_safe_depth_with_planar_arc_interface",
                    "thickness_sampling_policy": "equal_arc_boundary_vertices_capped",
                    "thickness_boundary_input_vertex_count": int(len(points)),
                    "thickness_boundary_sampled_vertex_count": int(
                        len(boundary_sample_indices)
                    ),
                    "thickness_full_boundary_audit_applied": False,
                    "depth_slope_evaluation_policy": "equal_arc_boundary_samples_only",
                    "thickness_depth_path": (
                        "interface_boundary_to_opposing_surface_along_shared_component_inward_axis"
                    ),
                    "thickness_depth_direction": fallback.round(9).tolist(),
                    "preferred_minimum_inward_depth_mm": preferred_minimum,
                    "effective_minimum_inward_depth_mm": float(distances.min()),
                    "safe_maximum_inward_depth_mm": global_ceiling,
                    "minimum_generated_inward_travel_mm": float(distances.min()),
                    "maximum_generated_inward_travel_mm": float(distances.max()),
                    "per_vertex_raw_safe_minimum_mm": float(per_vertex_safe.min()),
                    "per_vertex_raw_safe_maximum_mm": float(per_vertex_safe.max()),
                    "per_vertex_reserve_minimum_mm": float(per_vertex_reserve.min()),
                    "per_vertex_half_thickness_reserve_count": int(
                        np.count_nonzero(
                            raw_thicknesses
                            < PARENT_THICKNESS_CLEARANCE_MM - 1e-9
                        )
                    ),
                    "per_vertex_limited_count": int(
                        np.count_nonzero(per_vertex_safe < global_ceiling - 1e-9)
                    ),
                    "per_vertex_faired_count": int(
                        np.count_nonzero(distances < per_vertex_safe - 1e-9)
                    ),
                    "depth_field_slope_max_degrees": float(slope_degrees.max()),
                    "flat_span_mm": attempted["span_mm"],
                    "flat_bottom_required_max_extension_mm": float(distances.max()),
                    "flat_bottom_plane_s": None,
                    "flat_plane_extension_max_mm": float(distances.max()),
                    "flat_priority_applied": False,
                    "flat_feasibility_basis": "common_plane_blocked_by_local_thickness",
                    "flat_travel_within_limit": True,
                    "target_exceeded": bool(
                        float(distances.max()) > preferred_minimum + 1e-9
                    ),
                    "target_exceeded_by_mm": float(
                        max(0.0, float(distances.max()) - preferred_minimum)
                    ),
                    "fixed_inward_depth_mm": preferred_minimum,
                    "fixed_inward_depth_applied": False,
                    "local_direction_correction_applied": False,
                    "local_direction_corrected_vertices": 0,
                    "minimum_direction_dot_global": float(
                        np.min(uniform_directions @ fallback)
                    ),
                    "best_fit_flat_normal": best_fit_normal.round(6).tolist(),
                    "best_fit_flat_normal_dot_global": best_fit_dot_global,
                    "plane_candidate_required_maxima_mm": {
                        candidate["orientation"]: candidate["required_maximum_mm"]
                        for candidate in plane_candidates
                    },
                    **smooth_direction_record,
                }
                return materialize_sampled_boundary_field(distances, record)
            zero_safe_indices = np.flatnonzero(per_vertex_safe <= 1e-6)
            raise ValueError(
                f"Forced {cap_mode} uniform-direction fallback has no positive "
                "travel after parent-thickness clearance: "
                f"raw_thickness_min={float(raw_thicknesses.min()):.9f}mm, "
                f"zero_safe_count={int(len(zero_safe_indices))}, "
                f"zero_safe_indices={zero_safe_indices[:16].tolist()}, "
                f"clearance={PARENT_THICKNESS_CLEARANCE_MM:.6f}mm"
            )
        raise ValueError(
            f"Forced {cap_mode} cap cannot fit inside the parent-thickness safety limit: "
            f"required={attempted['required_maximum_mm']:.6f}mm, "
            f"safe_maximum={attempted['safe_maximum_mm']:.6f}mm, "
            f"minimum_travel={attempted['minimum_travel_mm']:.6f}mm"
        )

    local_safe_maximum, local_thickness_record = measured_limit(
        depth_directions
    )
    local_thickness_record["parent_thickness_depth_path"] = (
        "interface_boundary_to_opposing_surface_along_shared_component_inward_axis"
    )
    local_thickness_record["parent_thickness_depth_direction"] = (
        fallback.round(9).tolist()
    )
    if local_safe_maximum <= 1e-6:
        raise ValueError(
            "No positive inward depth remains after the 0.05 mm parent-thickness clearance: "
            + json.dumps(local_thickness_record, ensure_ascii=False, sort_keys=True)
        )
    local_target_depth = min(preferred_minimum, local_safe_maximum)
    distances = np.full(len(points), local_target_depth, dtype=np.float64)
    required_plane_maximum = min(
        (
            float(candidate["required_maximum_mm"])
            for candidate in plane_candidates
        ),
        default=preferred_minimum,
    )
    direction_dot = np.clip(depth_directions @ fallback, -1.0, 1.0)
    record = {
        "requested_cap_mode": cap_mode,
        "cap_mode": "local-offset",
        "cap_depth_reference": "interface_boundary_to_opposing_surface_shared_axis",
        "preferred_minimum_inward_depth_mm": preferred_minimum,
        "effective_minimum_inward_depth_mm": min(
            preferred_minimum,
            local_safe_maximum,
        ),
        "safe_maximum_inward_depth_mm": local_safe_maximum,
        "local_offset_depth_policy": "requested_depth_clamped_to_shared_axis_surface_distance",
        "flat_span_mm": float(
            min(
                (candidate["span_mm"] for candidate in plane_candidates),
                default=0.0,
            )
        ),
        "flat_bottom_required_max_extension_mm": required_plane_maximum,
        "flat_bottom_plane_s": None,
        "flat_plane_extension_max_mm": required_plane_maximum,
        "minimum_generated_inward_travel_mm": local_target_depth,
        "maximum_generated_inward_travel_mm": local_target_depth,
        "flat_priority_applied": False,
        "flat_feasibility_basis": "required_plane_depth_exceeds_parent_thickness_safety_limit",
        "flat_travel_within_limit": False,
        "local_fallback_reason": "required_plane_depth_exceeds_safe_maximum",
        "target_exceeded": bool(required_plane_maximum > preferred_minimum + 1e-9),
        "target_exceeded_by_mm": float(
            max(0.0, required_plane_maximum - preferred_minimum)
        ),
        "fixed_inward_depth_mm": preferred_minimum,
        "fixed_inward_depth_applied": bool(
            abs(local_target_depth - preferred_minimum) <= 1e-9
        ),
        "local_direction_correction_applied": bool(
            np.any(direction_dot < 1.0 - 1e-8)
        ),
        "local_direction_corrected_vertices": int(
            np.count_nonzero(direction_dot < 1.0 - 1e-8)
        ),
        "minimum_direction_dot_global": (
            float(direction_dot.min()) if len(direction_dot) else 1.0
        ),
        "best_fit_flat_normal": best_fit_normal.round(6).tolist(),
        "best_fit_flat_normal_dot_global": best_fit_dot_global,
        "plane_candidate_required_maxima_mm": {
            candidate["orientation"]: candidate["required_maximum_mm"]
            for candidate in plane_candidates
        },
        **smooth_direction_record,
        **local_thickness_record,
    }
    record["thickness_depth_path"] = (
        "interface_boundary_to_opposing_surface_along_shared_component_inward_axis"
    )
    record["thickness_depth_direction"] = fallback.round(9).tolist()
    return materialize_sampled_boundary_field(distances, record)
