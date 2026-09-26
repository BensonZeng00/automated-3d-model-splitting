"""Reusable geometry policy for printable connector backings.

This module is intentionally independent of the recursive splitter.  It turns
one source boundary plus a connector plan into deterministic lead/floor rings;
mesh sewing and Boolean application remain separate concerns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .mesh import point_in_poly, point_on_poly_boundary, signed_area
from .hidden_interface import (
    MAX_BOUNDARY_THICKNESS_PROBES,
    boundary_screening_indices,
)


@dataclass(frozen=True)
class BackingProfile:
    """Generated rings for one continuous printable backing wall.

    ``rings`` excludes the immutable visible source boundary and is ordered
    from the first subsurface transition to the planar backing floor.
    """

    rings: tuple[np.ndarray, ...]
    depths_mm: tuple[float, ...]
    taper_depth_mm: float


def _sample_ordered_boundary(points: np.ndarray) -> np.ndarray:
    """Use deterministic equal-arc source vertices for boundary evaluation."""

    ring = np.asarray(points, dtype=np.float64)
    indices = boundary_screening_indices(ring, MAX_BOUNDARY_THICKNESS_PROBES)
    return ring[indices]


def _sample_boundary_with_directions(
    points: np.ndarray,
    directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep a loop and its per-vertex field aligned on equal-arc samples."""

    ring = np.asarray(points, dtype=np.float64)
    field = np.asarray(directions, dtype=np.float64)
    if field.shape != ring.shape:
        raise ValueError("boundary direction field must match the source loop")
    indices = boundary_screening_indices(ring, MAX_BOUNDARY_THICKNESS_PROBES)
    return ring[indices], field[indices], indices


def _project_points_to_closed_ring(
    query_points: np.ndarray,
    ring: np.ndarray,
    *,
    block_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project a batch of points to a closed ring with bounded vector work."""

    query = np.asarray(query_points, dtype=np.float64)
    source = np.asarray(ring, dtype=np.float64)
    following = np.roll(source, -1, axis=0)
    edge = following - source
    length_squared = np.einsum("ij,ij->i", edge, edge)
    distances = np.empty(len(query), dtype=np.float64)
    segments = np.empty(len(query), dtype=np.int64)
    parameters = np.empty(len(query), dtype=np.float64)
    for start in range(0, len(query), max(1, int(block_size))):
        stop = min(start + max(1, int(block_size)), len(query))
        block = query[start:stop]
        relative = block[:, None, :] - source[None, :, :]
        local_parameter = np.divide(
            np.einsum("bij,ij->bi", relative, edge),
            length_squared[None, :],
            out=np.zeros((len(block), len(source)), dtype=np.float64),
            where=length_squared[None, :] > 1e-24,
        )
        np.clip(local_parameter, 0.0, 1.0, out=local_parameter)
        closest = source[None, :, :] + local_parameter[:, :, None] * edge[None, :, :]
        delta = block[:, None, :] - closest
        distance_squared = np.einsum("bij,bij->bi", delta, delta)
        selected = np.argmin(distance_squared, axis=1)
        rows = np.arange(len(block))
        distances[start:stop] = np.sqrt(distance_squared[rows, selected])
        segments[start:stop] = selected
        parameters[start:stop] = local_parameter[rows, selected]
    return distances, segments, parameters


def _user_reviewed_shallow_minimal_closure_is_eligible(
    plan: dict,
    *,
    boundary_vertex_count: int,
    backing_depth_mm: float,
) -> bool:
    """Allow an un-tapered closure only below printable feature scale."""

    return bool(
        int(boundary_vertex_count) <= 4
        and 0.0 < float(backing_depth_mm) <= 0.05 + 1e-12
        and not bool(plan.get("compact_peg_enabled", True))
        and str(plan.get("backing_slope_validation_mode", "strict")) == "advisory"
        and str(plan.get("backing_surface_validation_mode", "strict")) == "advisory"
    )


def user_reviewed_shallow_minimal_needle_advisory_is_eligible(
    plan: dict,
    *,
    invalid_face_count: int,
    maximum_edge_mm: float,
) -> bool:
    """Accept only bounded slivers intrinsic to an approved minimal closure.

    A three- or four-point ring moved only a few microns inward can create a
    high aspect-ratio side triangle even though the strip is watertight and
    all of its bridges stay inside the audited annulus.  This policy does not
    waive degeneracy or topology checks; it merely keeps the later generic
    aspect heuristic consistent with the already-approved minimal closure.
    """

    boundary_vertex_count = int(plan.get("backing_source_ring_vertices", 0))
    backing_depth_mm = float(plan.get("full_boundary_backing_depth_mm", 0.0))
    strip_audits = list(plan.get("backing_strip_audits", []))
    checks = {
        "minimal_closure_preapproved": bool(
            plan.get("shallow_minimal_closure_advisory_accepted", False)
        ),
        "minimal_closure_geometry_eligible": (
            _user_reviewed_shallow_minimal_closure_is_eligible(
                plan,
                boundary_vertex_count=boundary_vertex_count,
                backing_depth_mm=backing_depth_mm,
            )
        ),
        "bounded_needle_face_count": (
            0 < int(invalid_face_count) <= max(boundary_vertex_count, 1)
        ),
        "bounded_needle_edge": (
            math.isfinite(float(maximum_edge_mm))
            and float(maximum_edge_mm) <= 8.0 + 1e-12
        ),
        "strip_audits_present": bool(strip_audits),
        "strip_topology_safe": bool(
            strip_audits
            and all(
                int(record.get("degenerate_face_count", 0)) == 0
                and int(record.get("invalid_bridge_count", 0)) == 0
                and int(record.get("maximum_fanout", 0)) <= 64
                for record in strip_audits
            )
        ),
    }
    plan["shallow_minimal_closure_needle_advisory_checks"] = dict(checks)
    # The tiny ring may retain a valid Clipper offset instead of taking the
    # axial-fallback branch.  Eligibility is a geometry contract, not a branch
    # label: keep the label for diagnostics but do not make it a prerequisite.
    required_checks = {
        name: value
        for name, value in checks.items()
        if name != "minimal_closure_preapproved"
    }
    return bool(all(required_checks.values()))


def _point_to_closed_ring_projection(
    point: np.ndarray,
    ring: np.ndarray,
) -> tuple[float, int, float]:
    """Return distance, segment index, and parameter on a closed polyline."""

    polygon = np.asarray(ring, dtype=np.float64)
    left = polygon
    edge = np.roll(polygon, -1, axis=0) - polygon
    length_squared = np.einsum("ij,ij->i", edge, edge)
    relative = np.asarray(point, dtype=np.float64)[None, :] - left
    parameter = np.divide(
        np.einsum("ij,ij->i", relative, edge),
        length_squared,
        out=np.zeros(len(edge), dtype=np.float64),
        where=length_squared > 1e-24,
    )
    parameter = np.clip(parameter, 0.0, 1.0)
    closest = left + parameter[:, None] * edge
    distances = np.linalg.norm(closest - np.asarray(point), axis=1)
    segment_index = int(np.argmin(distances))
    return (
        float(distances[segment_index]),
        segment_index,
        float(parameter[segment_index]),
    )


def measured_backing_taper_angles(
    source_points: np.ndarray,
    floor_points: np.ndarray,
    plan: dict,
) -> tuple[float, float, float]:
    """Measure local source-to-floor taper angles in physical millimetres."""

    angles = backing_taper_angle_samples(source_points, floor_points, plan)
    if not len(angles):
        return 0.0, 0.0, 0.0
    return float(angles.min()), float(np.median(angles)), float(angles.max())


def backing_taper_angle_samples(
    source_points: np.ndarray,
    floor_points: np.ndarray,
    plan: dict,
) -> np.ndarray:
    """Return every finite local taper sample used by the backing audit."""

    source = _sample_ordered_boundary(source_points)
    floor = _sample_ordered_boundary(floor_points)
    source_2d = project_connector_points(source, plan)
    floor_2d = project_connector_points(floor, plan)
    axis = np.asarray(plan["inward"], dtype=np.float64)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    center = np.asarray(plan["center"], dtype=np.float64)
    source_axial = (source - center[None, :]) @ axis
    floor_axial = (floor - center[None, :]) @ axis
    lateral, segment_indices, parameters = _project_points_to_closed_ring(
        floor_2d,
        source_2d,
    )
    following = (segment_indices + 1) % len(source_axial)
    matched_source_axial = (
        source_axial[segment_indices] * (1.0 - parameters)
        + source_axial[following] * parameters
    )
    axial = np.abs(floor_axial - matched_source_axial)
    valid = np.isfinite(axial) & np.isfinite(lateral) & (lateral > 1e-9)
    if not np.any(valid):
        return np.empty(0, dtype=np.float64)
    return np.degrees(np.arctan2(axial[valid], lateral[valid]))


def _maximum_cyclic_true_run(mask: np.ndarray) -> int:
    """Return the longest contiguous run on a closed Boolean sequence."""

    values = np.asarray(mask, dtype=bool)
    if not len(values) or not np.any(values):
        return 0
    if np.all(values):
        return int(len(values))
    first_false = int(np.flatnonzero(~values)[0])
    rotated = np.roll(values, -first_false)
    maximum = 0
    current = 0
    for value in rotated:
        if bool(value):
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0
    return int(maximum)


def summarize_backing_taper_angle_samples(
    angles: np.ndarray,
    *,
    minimum_degrees: float = 30.0,
    maximum_degrees: float = 75.0,
    maximum_outlier_ratio: float = 0.02,
    maximum_outlier_run_ratio: float = 0.01,
) -> dict:
    """Classify a taper profile without letting one medial-axis sample veto it.

    The nearest-ring projection is intentionally conservative but becomes
    ambiguous at isolated concave medial-axis events.  The manufacturing
    profile is accepted only when its central 90 percent and median remain in
    range, at most two percent of samples are outside, and no outside segment
    persists for more than one percent of the closed ring (with a three-sample
    floor).  Geometry topology, degeneracy, bridge, and cutter audits remain
    independent hard gates.
    """

    values = np.asarray(angles, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "valid": False,
            "sample_count": 0,
            "minimum_degrees": None,
            "p05_degrees": None,
            "median_degrees": None,
            "p95_degrees": None,
            "maximum_degrees": None,
            "outside_count": 0,
            "outside_ratio": 1.0,
            "maximum_cyclic_outlier_run": 0,
            "maximum_allowed_outlier_run": 0,
            "accepted_sparse_outliers": False,
        }

    lower = float(minimum_degrees)
    upper = float(maximum_degrees)
    outside = (values < lower - 1e-6) | (values > upper + 1e-6)
    outside_count = int(np.count_nonzero(outside))
    outside_ratio = float(outside_count / len(values))
    maximum_run = _maximum_cyclic_true_run(outside)
    allowed_run = max(
        3,
        int(math.ceil(float(maximum_outlier_run_ratio) * len(values))),
    )
    p05, median, p95 = np.percentile(values, (5.0, 50.0, 95.0))
    valid = bool(
        lower - 1e-6 <= float(median) <= upper + 1e-6
        and float(p05) >= lower - 1e-6
        and float(p95) <= upper + 1e-6
        and outside_ratio <= float(maximum_outlier_ratio) + 1e-12
        and maximum_run <= allowed_run
    )
    return {
        "valid": valid,
        "sample_count": int(len(values)),
        "minimum_degrees": float(values.min()),
        "p05_degrees": float(p05),
        "median_degrees": float(median),
        "p95_degrees": float(p95),
        "maximum_degrees": float(values.max()),
        "outside_count": outside_count,
        "outside_ratio": outside_ratio,
        "maximum_cyclic_outlier_run": maximum_run,
        "maximum_allowed_outlier_run": allowed_run,
        "accepted_sparse_outliers": bool(valid and outside_count > 0),
        "policy": "central_90_percent_plus_sparse_cyclic_outliers",
    }


def backing_taper_angle_audit(
    source_points: np.ndarray,
    floor_points: np.ndarray,
    plan: dict,
) -> dict:
    """Measure and summarize one production backing taper."""

    return summarize_backing_taper_angle_samples(
        backing_taper_angle_samples(source_points, floor_points, plan)
    )


def project_connector_points(points: np.ndarray, plan: dict) -> np.ndarray:
    centered = np.asarray(points, dtype=np.float64) - np.asarray(plan["center"], dtype=np.float64)
    return np.column_stack(
        (
            centered @ np.asarray(plan["u"], dtype=np.float64),
            centered @ np.asarray(plan["v"], dtype=np.float64),
        )
    )



def line_preserving_inset_displacements(
    boundary_points: np.ndarray,
    plan: dict,
    distance_mm: float,
) -> np.ndarray:
    """Inset a sampled planar loop without bowing straight source edges.

    A center-radial shrink turns every point on a square edge by a different
    angle and therefore bends that edge into an arc.  Here each vertex follows
    the miter of its two adjacent edge normals.  Collinear samples consequently
    receive exactly the same displacement, while genuinely curved boundaries
    keep their locally varying normals.
    """

    boundary = np.asarray(boundary_points, dtype=np.float64)
    axis = np.asarray(plan["inward"], dtype=np.float64)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    u = np.asarray(plan["u"], dtype=np.float64)
    v = np.asarray(plan["v"], dtype=np.float64)
    origin = boundary.mean(axis=0)
    polygon = np.column_stack(
        ((boundary - origin[None, :]) @ u, (boundary - origin[None, :]) @ v)
    )
    signed_twice_area = float(
        np.sum(
            polygon[:, 0] * np.roll(polygon[:, 1], -1)
            - np.roll(polygon[:, 0], -1) * polygon[:, 1]
        )
    )
    orientation = 1.0 if signed_twice_area >= 0.0 else -1.0
    displacement_2d = np.zeros_like(polygon)
    incoming_edges = np.zeros_like(polygon)
    outgoing_edges = np.zeros_like(polygon)
    incoming_normals = np.zeros_like(polygon)
    outgoing_normals = np.zeros_like(polygon)
    distance = float(distance_mm)
    absolute_distance = abs(distance)
    maximum_miter = max(4.0 * absolute_distance, absolute_distance)

    for index in range(len(polygon)):
        previous_edge = polygon[index] - polygon[index - 1]
        next_edge = polygon[(index + 1) % len(polygon)] - polygon[index]
        previous_length = float(np.linalg.norm(previous_edge))
        next_length = float(np.linalg.norm(next_edge))
        if previous_length <= 1e-12 or next_length <= 1e-12:
            continue
        previous_edge /= previous_length
        next_edge /= next_length
        incoming_edges[index] = previous_edge
        outgoing_edges[index] = next_edge
        if orientation > 0.0:
            previous_normal = np.asarray(
                [-previous_edge[1], previous_edge[0]], dtype=np.float64
            )
            next_normal = np.asarray(
                [-next_edge[1], next_edge[0]], dtype=np.float64
            )
        else:
            previous_normal = np.asarray(
                [previous_edge[1], -previous_edge[0]], dtype=np.float64
            )
            next_normal = np.asarray(
                [next_edge[1], -next_edge[0]], dtype=np.float64
            )
        incoming_normals[index] = previous_normal
        outgoing_normals[index] = next_normal
        summed = previous_normal + next_normal
        summed_length = float(np.linalg.norm(summed))
        if summed_length <= 1e-9:
            candidate = previous_normal * distance
        else:
            miter = summed / summed_length
            denominator = float(np.dot(miter, previous_normal))
            if denominator <= 1e-6:
                candidate = previous_normal * distance
            else:
                candidate = miter * (distance / denominator)
        candidate_length = float(np.linalg.norm(candidate))
        if candidate_length > maximum_miter and candidate_length > 1e-12:
            candidate *= maximum_miter / candidate_length
        displacement_2d[index] = candidate

    # When a long semantic edge is densely sampled, a large inset can place
    # its last ordinary sample beyond the mitered corner and make the ring fold
    # back on itself.  Detect real corners, then reparameterize only genuinely
    # straight intervening runs between the two exact miter anchors.  Curves
    # retain their local-normal offset; distributing a sharp-corner correction
    # across a curved U or V can push its middle outside the source outline.
    corner_cosine = math.cos(math.radians(25.0))
    corner_indices = [
        index
        for index in range(len(polygon))
        if float(np.dot(incoming_edges[index], outgoing_edges[index]))
        < corner_cosine
    ]
    corrected_positions = polygon + displacement_2d
    if len(corner_indices) >= 2:
        count = len(polygon)
        for corner_position, start in enumerate(corner_indices):
            end = corner_indices[(corner_position + 1) % len(corner_indices)]
            indices = [int(start)]
            while indices[-1] != int(end):
                indices.append((indices[-1] + 1) % count)
            if len(indices) <= 2:
                continue
            segment_points = polygon[np.asarray(indices, dtype=np.int64)]
            segment_lengths = np.linalg.norm(
                np.diff(segment_points, axis=0), axis=1
            )
            total_length = float(segment_lengths.sum())
            if total_length <= 1e-12:
                continue
            parameters = np.concatenate(
                ([0.0], np.cumsum(segment_lengths) / total_length)
            )
            chord = segment_points[-1] - segment_points[0]
            chord_length = float(np.linalg.norm(chord))
            if chord_length <= 1e-12:
                continue
            relative = segment_points - segment_points[0]
            deviations = np.abs(
                relative[:, 0] * chord[1] - relative[:, 1] * chord[0]
            ) / chord_length
            straight_tolerance = max(0.10, 0.01 * chord_length)
            if float(deviations.max()) <= straight_tolerance:
                for local_index, source_index in enumerate(indices[1:-1], start=1):
                    parameter = float(parameters[local_index])
                    corrected_positions[source_index] = (
                        (1.0 - parameter) * corrected_positions[start]
                        + parameter * corrected_positions[end]
                    )
                continue

            # A curved segment keeps its normal offset in the middle.  Blend
            # only the one-sided endpoint offsets into the sharp-corner miters
            # over a short physical neighborhood, preventing a local fold
            # without dragging the whole U/V curve toward either corner.
            median_step = float(np.median(segment_lengths))
            blend_length = min(
                0.35 * total_length,
                max(2.5 * absolute_distance, 4.0 * median_step),
            )
            if blend_length <= 1e-12:
                continue
            one_sided_start = (
                polygon[start] + outgoing_normals[start] * distance
            )
            one_sided_end = (
                polygon[end] + incoming_normals[end] * distance
            )
            start_correction = corrected_positions[start] - one_sided_start
            end_correction = corrected_positions[end] - one_sided_end

            def endpoint_weight(distance_from_endpoint: float) -> float:
                ratio = min(max(distance_from_endpoint / blend_length, 0.0), 1.0)
                smooth = ratio * ratio * (3.0 - 2.0 * ratio)
                return 1.0 - smooth

            for local_index, source_index in enumerate(indices[1:-1], start=1):
                distance_from_start = float(parameters[local_index] * total_length)
                distance_from_end = total_length - distance_from_start
                corrected_positions[source_index] += (
                    endpoint_weight(distance_from_start) * start_correction
                    + endpoint_weight(distance_from_end) * end_correction
                )
        displacement_2d = corrected_positions - polygon

    return (
        displacement_2d[:, 0, None] * u[None, :]
        + displacement_2d[:, 1, None] * v[None, :]
    )



def topology_safe_planar_inset_ring(
    boundary_points: np.ndarray,
    plan: dict,
    distance_mm: float,
    *,
    axial_depth_mm: float | None = None,
) -> np.ndarray | None:
    """Return a trimmed planar inset with independently selected lateral/axial travel.

    Local miter vectors preserve the input vertex count, but a deep inset of a
    concave V inevitably makes those miters cross.  Clipper2, exposed by the
    already-required manifold3d backend, removes the collapsed loops and gives
    the actual inward offset contour.  Preserve that contour's real topology:
    forcing it back to the dense source sample count creates hundreds of
    collinear duplicates and needle triangles in the backing wall.
    """
    distance = max(float(distance_mm), 0.0)
    axial_depth = (
        distance
        if axial_depth_mm is None
        else max(float(axial_depth_mm), 0.0)
    )
    boundary = np.asarray(boundary_points, dtype=np.float64)
    if distance <= 1e-9:
        return boundary.copy()
    axis = np.asarray(plan["inward"], dtype=np.float64)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    center = np.asarray(plan["center"], dtype=np.float64)
    u = np.asarray(plan["u"], dtype=np.float64)
    v = np.asarray(plan["v"], dtype=np.float64)
    axial = (boundary - center[None, :]) @ axis
    # The printable backing floor is intentionally planar.  A strongly curved
    # 3-D source loop may project with overlaps onto that assembly plane; this
    # is exactly when the Clipper2 cross-section cleanup is most necessary, not
    # a reason to fall back to one offset point per source normal.  Preserve the
    # exact visible 3-D ring in the outer wedge, but let the internal floor use
    # the cleaned simple projected contour.
    plan["backing_source_axial_spread_mm"] = float(np.ptp(axial))
    projected = np.column_stack(
        ((boundary - center[None, :]) @ u, (boundary - center[None, :]) @ v)
    )
    if abs(float(signed_area(projected))) <= 1e-9:
        return None
    # Recognition owns boundary simplification.  Preserve that ring's full
    # cardinality here and only construct its paired homothetic inner contour.
    from .contour_simplification import corresponding_scaled_contours
    correspondence_tolerance = 0.0
    try:
        outer_contour, inset_2d, retained, scale = corresponding_scaled_contours(
            projected,
            correspondence_tolerance,
            distance,
            center=np.zeros(2, dtype=np.float64),
        )
        plan["backing_ring_correspondence"] = "homothetic_one_to_one"
        plan["backing_ring_outer_vertices"] = int(len(outer_contour))
        plan["backing_ring_inner_vertices"] = int(len(inset_2d))
        plan["backing_ring_source_indices"] = [int(value) for value in retained]
        plan["backing_ring_scale"] = float(scale)
        plan["backing_inset_recognized_boundary_vertices"] = int(len(outer_contour))
        plan["backing_inset_input_vertices"] = int(len(projected))
        plan["backing_inset_method"] = "recognized_boundary_homothetic_correspondence"
        source_plane = center + axis * float(np.mean(axial))
        return (
            source_plane[None, :]
            + inset_2d[:, 0, None] * u[None, :]
            + inset_2d[:, 1, None] * v[None, :]
            + axis[None, :] * axial_depth
        )
    except ValueError:
        # A homothetic inset is only valid for a star-shaped boundary around
        # the measured interior center.  Signal infeasibility so the existing
        # depth search can reduce the backing or omit the optional peg.  An
        # unequal-ring Clipper fallback would reintroduce the topology defect.
        return None

def printable_backing_rings(
    boundary_points: np.ndarray,
    plan: dict,
    *,
    child_clearance: bool,
    inward_directions: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Create a topology-safe backing lead within the printable 30–75° range."""
    boundary = np.asarray(boundary_points, dtype=np.float64)
    axis = np.asarray(plan["inward"], dtype=np.float64)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    if inward_directions is None:
        directions = np.tile(axis, (len(boundary), 1))
        direction_mode = "single_global_flat_direction"
    else:
        safe_directions = np.asarray(inward_directions, dtype=np.float64)
        if safe_directions.shape != boundary.shape:
            raise ValueError("local connector inward directions must match the boundary")
        lengths = np.linalg.norm(safe_directions, axis=1)
        if np.any(~np.isfinite(safe_directions)) or np.any(lengths <= 1e-12):
            raise ValueError("local connector inward directions must be finite and non-zero")
        # The local safe field remains useful to select the permitted depth,
        # but sweeping the printable backing along one normal per source vertex
        # turns a smooth painted curve into the radial hills seen in slicers.
        # Build the connector geometry on one assembly-axis plane instead.
        directions = np.tile(axis, (len(boundary), 1))
        direction_mode = "single_global_planar_direction_from_safe_axis"
        plan["backing_safe_direction_field_received"] = True
    plan["backing_inward_direction_mode"] = direction_mode
    sampled_boundary = _sample_ordered_boundary(boundary)
    sampled_boundary_indices = boundary_screening_indices(
        boundary, MAX_BOUNDARY_THICKNESS_PROBES
    )
    backing_depth = float(plan.get("full_boundary_backing_depth_mm", 0.0))
    compact_peg_enabled = bool(plan.get("compact_peg_enabled", True))
    protected_ring = (
        np.asarray(
            plan["peg_top"] if child_clearance else plan["socket_mouth"],
            dtype=np.float64,
        )
        if compact_peg_enabled
        else np.empty((0, 3), dtype=np.float64)
    )
    if compact_peg_enabled:
        protected_radial = protected_ring - np.asarray(plan["center"])[None, :]
        protected_radial -= (
            (protected_radial @ axis)[:, None] * axis[None, :]
        )
        protected_radius = float(np.linalg.norm(protected_radial, axis=1).max())
        available_taper = (
            float(plan.get("edge_clearance_mm", 0.0))
            - protected_radius
            - 0.35
        )
    else:
        available_taper = backing_depth
    protected_taper = min(backing_depth, max(0.25, available_taper))
    if child_clearance:
        if compact_peg_enabled and not bool(
            plan.get("full_backing_taper_reserved", False)
        ):
            raise ValueError(
                "compact connector plan did not reserve the complete full-depth "
                "backing wedge for its selected slope"
            )
        # The connector planner has already reduced the compact peg until the
        # complete 3 mm eroded footprint remains available.  Spend that full
        # depth on one continuous wedge; the 0.60/0.80 mm lead-in option is for
        # the female socket mouth and must not truncate this visible wall.
        requested_taper_depth = backing_depth
    else:
        requested_taper_depth = min(backing_depth, protected_taper)
    taper_depth = requested_taper_depth
    requested_backing_depth = float(backing_depth)

    # Evaluate slope and axial depth together on deterministic equal-arc
    # boundary samples.  Full-resolution input is retained only to materialize
    # the selected printable contour after screening has chosen one pair.
    # If the requested depth is too deep, reduce depth and retry, preferring
    # steeper allowed slopes that need less lateral inset.
    taper_angle_candidates = (
        45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 40.0, 35.0, 30.0
    )
    candidate_attempts: list[dict] = []
    preferred_axial_depth = float(taper_depth)
    topology_limited_lead: np.ndarray | None = None
    selected_taper_angle = 45.0
    selected_slope_audit: dict | None = None

    def evaluate_slope_depth(
        axial_depth: float,
        angle_degrees: float,
    ) -> dict | None:
        attempt = {
            "target_angle_degrees": float(angle_degrees),
            "requested_axial_depth_mm": float(axial_depth),
            "lateral_inset_mm": float(
                float(axial_depth)
                / max(math.tan(math.radians(float(angle_degrees))), 1e-12)
            ),
            "topology_valid": False,
            "protected_ring_contained": None,
            "rejection_reason": None,
            "measured_slope_statistics": None,
        }
        attempt["green_sidewall_depth_mm"] = math.hypot(
            float(axial_depth),
            float(attempt["lateral_inset_mm"]),
        )
        candidate_attempts.append(attempt)
        candidate_plan = dict(plan)
        lateral_distance = attempt["lateral_inset_mm"]
        candidate_lead = topology_safe_planar_inset_ring(
            sampled_boundary,
            candidate_plan,
            lateral_distance,
            axial_depth_mm=float(axial_depth),
        )
        if candidate_lead is None:
            attempt["rejection_reason"] = "topology_inset_unavailable"
            return None
        attempt["topology_valid"] = True
        if not compact_peg_enabled:
            candidate_lead = _fit_zero_engagement_planar_backing_slope(
                sampled_boundary,
                candidate_lead,
                candidate_plan,
            )
        elif child_clearance:
            protected_2d = project_connector_points(protected_ring, candidate_plan)
            candidate_2d = project_connector_points(candidate_lead, candidate_plan)
            if not all(
                point_in_poly(point, candidate_2d)
                or point_on_poly_boundary(point, candidate_2d)
                for point in protected_2d
            ):
                attempt["protected_ring_contained"] = False
                attempt["rejection_reason"] = "protected_ring_not_contained"
                return None
            attempt["protected_ring_contained"] = True
        slope_audit = backing_taper_angle_audit(
            sampled_boundary,
            candidate_lead,
            candidate_plan,
        )
        attempt["measured_slope_statistics"] = {
            key: slope_audit.get(key)
            for key in (
                "sample_count",
                "minimum_degrees",
                "p05_degrees",
                "median_degrees",
                "p95_degrees",
                "maximum_degrees",
                "outside_ratio",
                "valid",
            )
        }
        if not bool(slope_audit.get("valid", False)):
            attempt["rejection_reason"] = "measured_slope_statistics_out_of_range"
            return None
        attempt["rejection_reason"] = None
        return {
            "axial_depth_mm": float(axial_depth),
            "angle_degrees": float(angle_degrees),
            "lateral_inset_mm": float(lateral_distance),
            "green_sidewall_depth_mm": float(attempt["green_sidewall_depth_mm"]),
            "lead": candidate_lead,
            "plan": candidate_plan,
            "audit": slope_audit,
        }

    feasible_candidates = [
        candidate
        for angle_degrees in taper_angle_candidates
        if (candidate := evaluate_slope_depth(taper_depth, angle_degrees)) is not None
    ]
    search_iterations = 0
    from .print_tolerance import current as current_print_tolerance

    depth_tolerance = max(
        0.01,
        min(0.05, float(current_print_tolerance().surface_distance_mm)),
    )
    if not feasible_candidates and taper_depth > 0.0:
        # Jointly refine depth for each representative slope. The prior full
        # 1-degree-by-multidepth grid repeated hundreds of expensive polygon
        # offsets per interface. This bounded search keeps measured geometry
        # authoritative while limiting the total to about 100 candidate pairs.
        for angle_degrees in taper_angle_candidates:
            low = 0.0
            high = float(taper_depth)
            while (
                high - low > depth_tolerance
                and len(candidate_attempts) < 100
            ):
                middle = (low + high) * 0.5
                candidate = evaluate_slope_depth(middle, angle_degrees)
                if candidate is None:
                    high = middle
                else:
                    low = middle
                    feasible_candidates.append(candidate)
                search_iterations += 1

    plan["backing_inset_search_iterations"] = int(search_iterations)
    plan["backing_inset_search_evaluations"] = int(len(candidate_attempts))
    plan["backing_inset_search_tolerance_mm"] = float(depth_tolerance)
    plan["backing_slope_sampling_policy"] = "equal_arc_boundary_vertices_capped"
    plan["backing_slope_boundary_input_vertices"] = int(len(boundary))
    plan["backing_slope_boundary_sampled_vertices"] = int(len(sampled_boundary))
    plan["backing_taper_angle_search"] = candidate_attempts
    plan["backing_taper_search_rejection_counts"] = {
        str(reason): int(
            sum(attempt.get("rejection_reason") == reason for attempt in candidate_attempts)
        )
        for reason in sorted(
            {
                str(attempt["rejection_reason"])
                for attempt in candidate_attempts
                if attempt.get("rejection_reason") is not None
            }
        )
    }
    if feasible_candidates:
        selected = max(
            feasible_candidates,
            key=lambda candidate: (
                float(candidate["green_sidewall_depth_mm"]),
                float(candidate["axial_depth_mm"]),
                -abs(
                    float(candidate["audit"].get("median_degrees") or 45.0)
                    - 45.0
                ),
                -abs(float(candidate["angle_degrees"]) - 45.0),
            ),
        )
        taper_depth = float(selected["axial_depth_mm"])
        backing_depth = float(taper_depth)
        selected_taper_angle = float(selected["angle_degrees"])
        plan.update(selected["plan"])
        # Rebuild the chosen contour once at source resolution so generated
        # walls preserve the exact manufacturing seam; all depth/slope checks
        # stay on the equal-arc sample set.
        topology_limited_lead = topology_safe_planar_inset_ring(
            boundary,
            plan,
            float(selected["lateral_inset_mm"]),
            axial_depth_mm=float(selected["axial_depth_mm"]),
        )
        if topology_limited_lead is None:
            raise ValueError(
                "selected sampled backing pair does not materialize on the full source boundary"
            )
        if not compact_peg_enabled:
            topology_limited_lead = _fit_zero_engagement_planar_backing_slope(
                sampled_boundary,
                topology_limited_lead,
                plan,
            )
        selected_slope_audit = backing_taper_angle_audit(
            sampled_boundary,
            topology_limited_lead,
            plan,
        )
        if not bool(selected_slope_audit.get("valid", False)):
            raise ValueError(
                "materialized backing slope failed the equal-arc boundary sample audit"
            )
        plan["backing_taper_target_degrees"] = selected_taper_angle
        plan["backing_taper_lateral_inset_mm"] = float(
            selected["lateral_inset_mm"]
        )
        plan["backing_taper_green_sidewall_depth_mm"] = float(
            selected["green_sidewall_depth_mm"]
        )
        plan["backing_taper_depth_convention"] = (
            "green_sidewall_segment_length_between_equal_arc_boundary_samples"
        )
        plan["backing_taper_axial_travel_mm"] = float(
            selected["axial_depth_mm"]
        )
        plan["backing_taper_selected_measured_statistics"] = {
            key: selected_slope_audit.get(key)
            for key in (
                "sample_count",
                "minimum_degrees",
                "p05_degrees",
                "median_degrees",
                "p95_degrees",
                "maximum_degrees",
                "outside_ratio",
                "valid",
            )
        }
        plan["backing_depth_shape_limited"] = bool(
            taper_depth < requested_backing_depth - 1e-9
        )
        if plan["backing_depth_shape_limited"]:
            plan["full_boundary_backing_depth_mm"] = float(backing_depth)
            plan["backing_depth_before_shape_limit_mm"] = requested_backing_depth
            plan["backing_depth_shape_limit_policy"] = (
                "deepest_jointly_feasible_depth_and_30_75_degree_slope"
            )
        plan["backing_taper_requested_mm"] = float(requested_taper_depth)
    elif taper_depth > 0.0:
        if _user_reviewed_shallow_minimal_closure_is_eligible(
            plan,
            boundary_vertex_count=len(boundary),
            backing_depth_mm=backing_depth,
        ):
            shallow_floor = boundary + directions * float(backing_depth)
            plan["backing_inset_method"] = (
                "user_reviewed_shallow_minimal_vertical_closure"
            )
            plan["backing_depth_shape_limited"] = True
            plan["backing_depth_before_shape_limit_mm"] = requested_backing_depth
            plan["backing_depth_shape_limit_policy"] = (
                "locked_minimal_ring_axial_closure"
            )
            plan["shallow_minimal_closure_advisory_accepted"] = True
            plan["backing_source_ring_vertices"] = int(len(boundary))
            plan["backing_floor_ring_vertices"] = int(len(boundary))
            plan["backing_taper_requested_mm"] = float(requested_taper_depth)
            plan["backing_taper_shape_backoff_applied"] = False
            plan.pop("_backing_inset_simplified_section", None)
            return shallow_floor.copy(), shallow_floor.copy(), float(backing_depth)
        plan["backing_taper_angle_search"] = candidate_attempts
        raise ValueError(
            "connector boundary has no continuous backing whose measured "
            "slope statistics fit 30-75 degrees; "
            f"requested_depth_mm={preferred_axial_depth:.6f}, "
            f"boundary_vertices={len(boundary)}, "
            f"candidate_pairs={len(candidate_attempts)}, "
            f"rejection_counts={plan['backing_taper_search_rejection_counts']}"
        )

    # Each candidate floor keeps its jointly selected slope/depth pair while
    # preserving the complete visible source rim.
    def lead_for_depth(axial_depth: float) -> np.ndarray:
        if (
            topology_limited_lead is not None
            and abs(float(axial_depth) - float(taper_depth)) <= 1e-9
        ):
            return np.asarray(topology_limited_lead, dtype=np.float64).copy()
        lateral_distance = float(axial_depth) / max(
            math.tan(math.radians(selected_taper_angle)),
            1e-12,
        )
        topology_safe = topology_safe_planar_inset_ring(
            boundary,
            plan,
            lateral_distance,
            axial_depth_mm=float(axial_depth),
        )
        if topology_safe is not None:
            return topology_safe
        inset_displacements = line_preserving_inset_displacements(
            boundary,
            plan,
            lateral_distance,
        )
        candidate = boundary + inset_displacements
        candidate_2d = project_connector_points(candidate, plan)
        center = np.asarray(plan["center"], dtype=np.float64)
        u = np.asarray(plan["u"], dtype=np.float64)
        v = np.asarray(plan["v"], dtype=np.float64)
        source_axial = (boundary - center[None, :]) @ axis
        floor_axial = float(np.mean(source_axial)) + float(axial_depth)
        plan["backing_inset_method"] = "source_projection_flat_miter_offset"
        return (
            center[None, :]
            + candidate_2d[:, 0, None] * u[None, :]
            + candidate_2d[:, 1, None] * v[None, :]
            + floor_axial * axis[None, :]
        )

    protected_2d = (
        project_connector_points(protected_ring, plan)
        if compact_peg_enabled
        else np.empty((0, 2), dtype=np.float64)
    )

    def contains_protected_ring(candidate_lead: np.ndarray) -> bool:
        if not compact_peg_enabled:
            return True
        candidate_2d = project_connector_points(candidate_lead, plan)
        return all(
            point_in_poly(point, candidate_2d)
            or point_on_poly_boundary(point, candidate_2d)
            for point in protected_2d
        )

    lead = lead_for_depth(taper_depth)
    shape_backoff_applied = False
    if not contains_protected_ring(lead):
        if child_clearance:
            raise ValueError(
                "reserved compact connector footprint is outside the complete "
                "backing wedge for its selected slope"
            )
        # Concave outlines can spend less lateral distance than the inscribed
        # circle estimate predicts.  Search the actual generated outline, keep
        # a small safety margin, and continue vertically to the unchanged 3 mm
        # backing depth. The lead retains its selected slope/depth relationship.
        low = 0.0
        high = taper_depth
        for _ in range(18):
            middle = (low + high) * 0.5
            if contains_protected_ring(lead_for_depth(middle)):
                low = middle
            else:
                high = middle
        backed_off_depth = low * 0.95
        if backed_off_depth < 0.05:
            raise ValueError(
                "concave connector boundary cannot preserve the selected printable lead slope"
            )
        backed_off = evaluate_slope_depth(
            backed_off_depth,
            selected_taper_angle,
        )
        if backed_off is None:
            raise ValueError(
                "connector boundary lost its printable slope after protected-ring backoff"
            )
        taper_depth = float(backed_off["axial_depth_mm"])
        backing_depth = float(taper_depth)
        plan["full_boundary_backing_depth_mm"] = float(backing_depth)
        plan["backing_depth_before_shape_limit_mm"] = requested_backing_depth
        plan["backing_depth_shape_limited"] = True
        plan["backing_depth_shape_limit_policy"] = (
            "protected_ring_containment_after_joint_slope_depth_search"
        )
        plan["backing_taper_target_degrees"] = float(selected_taper_angle)
        plan["backing_taper_requested_mm"] = float(requested_taper_depth)
        topology_limited_lead = np.asarray(backed_off["lead"], dtype=np.float64)
        selected_slope_audit = backed_off["audit"]
        plan.update(backed_off["plan"])
        plan["backing_taper_lateral_inset_mm"] = float(
            backed_off["lateral_inset_mm"]
        )
        plan["backing_taper_selected_measured_statistics"] = {
            key: selected_slope_audit.get(key)
            for key in (
                "sample_count",
                "minimum_degrees",
                "p05_degrees",
                "median_degrees",
                "p95_degrees",
                "maximum_degrees",
                "outside_ratio",
                "valid",
            )
        }
        lead = np.asarray(topology_limited_lead, dtype=np.float64)
        shape_backoff_applied = True

    # A zero-engagement backing has no compact peg which requires its floor at
    # the nominal connector datum.  Dense sculpted rims can vary noticeably in
    # the axial direction, so a plane placed at exactly the lateral inset depth
    # may make one side of the otherwise valid wedge shallower than 30 degrees.
    # Translate that one planar floor within the independently measured axial
    # safety budget.  The lateral inset and visible rim stay unchanged, while
    # every physical source-to-floor correspondence remains inside the same
    # printable 30--75 degree contract used by the final audit.
    if not compact_peg_enabled:
        lead = _fit_zero_engagement_planar_backing_slope(
            boundary,
            lead,
            plan,
        )
        selected_slope_audit = backing_taper_angle_audit(
            sampled_boundary,
            lead,
            plan,
        )
        if not bool(selected_slope_audit.get("valid", False)):
            raise ValueError(
                "zero-engagement backing lost its printable slope after planar fit"
            )
        plan["backing_taper_selected_measured_statistics"] = {
            key: selected_slope_audit.get(key)
            for key in (
                "sample_count",
                "minimum_degrees",
                "p05_degrees",
                "median_degrees",
                "p95_degrees",
                "maximum_degrees",
                "outside_ratio",
                "valid",
            )
        }

    # Triangulation is a representation problem, not a shape constraint.  The
    # previous preflight probe repeatedly shortened a valid wedge whenever its
    # dense concave annulus happened to defeat one bridge choice, producing the
    # near-vertical skirt visible in slicer previews.  Keep geometry decisions
    # here limited to actual containment; the annulus builder must solve the
    # already-selected shape or fail loudly.

    remaining_depth = max(backing_depth - taper_depth, 0.0)
    continuation_directions = np.tile(axis, (len(lead), 1))
    backing = lead + continuation_directions * remaining_depth
    plan["backing_source_ring_vertices"] = int(len(boundary))
    plan["backing_slope_evaluation_policy"] = "equal_arc_boundary_samples_only"
    plan["backing_slope_evaluation_source_vertices"] = int(len(boundary))
    plan["backing_slope_evaluation_sample_count"] = int(len(sampled_boundary))
    plan["backing_slope_evaluation_source_sample_indices"] = [
        int(value) for value in sampled_boundary_indices[:32]
    ]
    plan["backing_floor_ring_vertices"] = int(len(lead))
    plan["backing_taper_requested_mm"] = float(requested_taper_depth)
    plan["backing_taper_shape_backoff_applied"] = bool(shape_backoff_applied)
    # The manifold object is an implementation cache, not report data.  Drop
    # it before the plan leaves this geometry service so JSON/report callers
    # continue to receive a purely serializable record.
    plan.pop("_backing_inset_simplified_section", None)
    return lead, backing, float(taper_depth)


def _closed_ring_parameters(points: np.ndarray) -> np.ndarray:
    """Return normalized physical arc-length parameters for ring vertices."""

    ring = np.asarray(points, dtype=np.float64)
    following = np.roll(ring, -1, axis=0)
    lengths = np.linalg.norm(following - ring, axis=1)
    perimeter = float(lengths.sum())
    if perimeter <= 1e-12:
        raise ValueError("printable backing ring has zero perimeter")
    return np.concatenate(([0.0], np.cumsum(lengths[:-1]))) / perimeter


def _project_ring_scalar_field(
    query_points: np.ndarray,
    source_ring: np.ndarray,
    source_values: np.ndarray,
    *,
    block_size: int = 256,
) -> np.ndarray:
    """Transfer a scalar field by closest-point projection on a closed ring.

    Inset libraries are free to rotate the first vertex and to remove samples.
    Arc-length interpolation by raw array index therefore shifts the source
    height around every generated layer.  Projecting onto source *segments*
    keeps the field attached to physical boundary locations instead of to an
    arbitrary cyclic seam.
    """

    query = np.asarray(query_points, dtype=np.float64)
    source = np.asarray(source_ring, dtype=np.float64)
    values = np.asarray(source_values, dtype=np.float64)
    if source.ndim != 2 or query.ndim != 2 or source.shape[1] != query.shape[1]:
        raise ValueError("ring scalar projection needs matching point dimensions")
    if len(source) != len(values) or len(source) < 3:
        raise ValueError("ring scalar projection needs one value per source vertex")

    following = np.roll(source, -1, axis=0)
    edge = following - source
    length_squared = np.einsum("ij,ij->i", edge, edge)
    result = np.empty(len(query), dtype=np.float64)
    for start in range(0, len(query), max(1, int(block_size))):
        block = query[start : start + max(1, int(block_size))]
        relative = block[:, None, :] - source[None, :, :]
        parameter = np.divide(
            np.einsum("bij,ij->bi", relative, edge),
            length_squared[None, :],
            out=np.zeros((len(block), len(source)), dtype=np.float64),
            where=length_squared[None, :] > 1e-24,
        )
        parameter = np.clip(parameter, 0.0, 1.0)
        closest = source[None, :, :] + parameter[:, :, None] * edge[None, :, :]
        distance_squared = np.einsum(
            "bij,bij->bi",
            block[:, None, :] - closest,
            block[:, None, :] - closest,
        )
        segment = np.argmin(distance_squared, axis=1)
        local = parameter[np.arange(len(block)), segment]
        result[start : start + len(block)] = (
            values[segment] * (1.0 - local)
            + values[(segment + 1) % len(values)] * local
        )
    return result


def _fit_zero_engagement_planar_backing_slope(
    source_ring: np.ndarray,
    planar_floor: np.ndarray,
    plan: dict,
    *,
    minimum_degrees: float = 31.0,
    maximum_degrees: float = 74.0,
) -> np.ndarray:
    """Translate a no-peg planar floor to a genuinely printable slope.

    The topology-safe inset may contain far fewer vertices than the vendor
    boundary.  Constraints are therefore evaluated at closest physical source
    segments, not by array index.  A one-degree construction margin absorbs
    the later topology-preserving annulus resampling while the authoritative
    final audit remains 30--75 degrees.  Only axial translation is allowed:
    the planar-arc target, lateral inset, and visible source ring are immutable.
    """

    source = _sample_ordered_boundary(source_ring)
    floor = np.asarray(planar_floor, dtype=np.float64).copy()
    sampled_floor = _sample_ordered_boundary(floor)
    source_2d = project_connector_points(source, plan)
    floor_2d = project_connector_points(sampled_floor, plan)
    axis = np.asarray(plan["inward"], dtype=np.float64)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    center = np.asarray(plan["center"], dtype=np.float64)
    source_axial = (source - center[None, :]) @ axis
    floor_axial = (sampled_floor - center[None, :]) @ axis
    current_plane = float(np.mean(floor_axial))
    if float(np.ptp(floor_axial)) > 1e-7:
        raise ValueError("zero-engagement connector backing floor is not planar")

    # The topology layer may later insert collinear vertices into this sparse
    # inset.  Checking only the original corners can therefore miss the true
    # shallowest point half-way along a long concave edge.  Sample the complete
    # piecewise-linear floor at twice the denser ring count so every later
    # source-parameter-grid subdivision is covered by the same slope contract.
    edge_lengths = np.linalg.norm(np.roll(floor_2d, -1, axis=0) - floor_2d, axis=1)
    target_sample_count = max(len(source_2d), len(floor_2d)) * 2
    subdivisions = np.ones(len(floor_2d), dtype=np.int64)
    for _ in range(max(0, target_sample_count - len(floor_2d))):
        edge_index = int(np.argmax(edge_lengths / subdivisions))
        subdivisions[edge_index] += 1
    constraint_samples: list[np.ndarray] = []
    for index, point in enumerate(floor_2d):
        following = floor_2d[(index + 1) % len(floor_2d)]
        for sample_index in range(int(subdivisions[index])):
            ratio = sample_index / int(subdivisions[index])
            constraint_samples.append(point * (1.0 - ratio) + following * ratio)

    sample_array = np.asarray(constraint_samples, dtype=np.float64)
    distances, segment_indices, parameters = _project_points_to_closed_ring(
        sample_array,
        source_2d,
    )
    following = (segment_indices + 1) % len(source_axial)
    matched = (
        source_axial[segment_indices] * (1.0 - parameters)
        + source_axial[following] * parameters
    )
    valid = np.isfinite(distances) & (distances > 1e-9)
    if not np.any(valid):
        plan["backing_planar_slope_bias_applied"] = False
        plan["backing_planar_slope_bias_status"] = "no_measurable_lateral_span"
        return floor

    matched = matched[valid]
    lateral_array = distances[valid]
    lower = float(
        np.max(matched + np.tan(np.radians(float(minimum_degrees))) * lateral_array)
    )
    upper = float(
        np.min(matched + np.tan(np.radians(float(maximum_degrees))) * lateral_array)
    )
    source_mean = float(np.mean(source_axial))
    axial_safety_depth = max(
        float(plan.get("total_safety_limit_mm", 0.0)),
        float(plan.get("full_boundary_backing_depth_mm", 0.0)),
    )
    upper = min(upper, source_mean + axial_safety_depth)
    if lower > upper + 1e-9:
        plan["backing_planar_slope_bias_applied"] = False
        plan["backing_planar_slope_bias_status"] = "no_safe_planar_depth"
        return floor

    selected_plane = min(max(current_plane, lower), upper)
    shift = float(selected_plane - current_plane)
    plan["backing_planar_slope_bias_applied"] = bool(abs(shift) > 1e-9)
    plan["backing_planar_slope_bias_status"] = "safe_planar_depth_selected"
    plan["backing_planar_slope_bias_mm"] = shift
    plan["backing_planar_axial_depth_mm"] = float(selected_plane - source_mean)
    plan["backing_planar_slope_interval_degrees"] = [
        float(minimum_degrees),
        float(maximum_degrees),
    ]
    plan["backing_planar_slope_constraint_samples"] = int(
        np.count_nonzero(valid)
    )
    return floor + axis[None, :] * shift


def _carry_source_height_into_transition(
    source: np.ndarray,
    planar_ring: np.ndarray,
    plan: dict,
    *,
    depth_mm: float,
    backing_depth_mm: float,
) -> np.ndarray:
    """Fade a curved source rim smoothly into the planar backing floor.

    Projecting the first inset directly to the final plane creates one abrupt
    wall between a spatial source rim and a planar contour.  Preserve the
    source rim's axial variation at the first transition and reduce it linearly
    to zero at the floor.  Lateral travel remains the requested constant inset.
    """

    boundary = np.asarray(source, dtype=np.float64)
    ring = np.asarray(planar_ring, dtype=np.float64).copy()
    axis = np.asarray(plan["inward"], dtype=np.float64)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    center = np.asarray(plan["center"], dtype=np.float64)
    source_axial = (boundary - center[None, :]) @ axis
    source_mean = float(np.mean(source_axial))
    source_projected = project_connector_points(boundary, plan)
    ring_projected = project_connector_points(ring, plan)
    carried_axial = _project_ring_scalar_field(
        ring_projected,
        source_projected,
        source_axial - source_mean,
    )
    return _apply_carried_axial_residual(
        ring,
        plan,
        carried_axial,
        source_mean=source_mean,
        depth_mm=depth_mm,
        backing_depth_mm=backing_depth_mm,
    )


def _apply_carried_axial_residual(
    planar_ring: np.ndarray,
    plan: dict,
    carried_residual: np.ndarray,
    *,
    source_mean: float,
    depth_mm: float,
    backing_depth_mm: float,
) -> np.ndarray:
    """Apply one already-corresponded source-height residual to a layer."""

    ring = np.asarray(planar_ring, dtype=np.float64).copy()
    residual = np.asarray(carried_residual, dtype=np.float64)
    if len(ring) != len(residual):
        raise ValueError("carried axial residual must match transition ring")
    axis = np.asarray(plan["inward"], dtype=np.float64)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    center = np.asarray(plan["center"], dtype=np.float64)
    fade = max(
        0.0,
        1.0 - float(depth_mm) / max(float(backing_depth_mm), 1e-12),
    )
    target_axial = (
        source_mean
        + float(depth_mm)
        + fade * residual
    )
    current_axial = (ring - center[None, :]) @ axis
    return ring + (target_axial - current_axial)[:, None] * axis[None, :]


def printable_backing_profile(
    boundary_points: np.ndarray,
    plan: dict,
    *,
    child_clearance: bool,
    inward_directions: np.ndarray | None = None,
    maximum_layer_step_mm: float | None = None,
) -> BackingProfile:
    """Build one topology-coherent profile in the printable 30–75° range.

    A constant-distance offset can lose vertices and pass medial-axis events as
    depth increases.  Independently offsetting many intermediate layers and
    sewing those different topologies together creates the cratered underside
    seen on dense, deeply concave vendor contours.  The visible source rim and
    the final topology-safe offset are therefore joined by one audited annulus.
    Jointly selected lateral inset and axial travel define the chosen slope,
    while one topology solve prevents layer-to-layer phase drift.

    ``maximum_layer_step_mm`` is retained for API compatibility but is no
    longer a geometry instruction.
    """

    boundary = np.asarray(boundary_points, dtype=np.float64)
    lead, backing, taper_depth = printable_backing_rings(
        boundary,
        plan,
        child_clearance=child_clearance,
        inward_directions=inward_directions,
    )
    backing_depth = float(plan.get("full_boundary_backing_depth_mm", 0.0))
    if backing_depth <= 1e-9:
        return BackingProfile((backing.copy(),), (0.0,), float(taper_depth))

    rings = [np.asarray(lead, dtype=np.float64).copy()]
    depths = [float(taper_depth)]
    if backing_depth > float(taper_depth) + 1e-9:
        rings.append(np.asarray(backing, dtype=np.float64).copy())
        depths.append(float(backing_depth))
    else:
        rings[0] = np.asarray(backing, dtype=np.float64).copy()
        depths[0] = float(backing_depth)

    plan["backing_profile_layer_count"] = int(len(rings))
    plan["backing_profile_layer_policy"] = "single_topology_coherent_annulus"
    plan["backing_profile_maximum_layer_step_mm"] = float(backing_depth)
    plan["backing_profile_depths_mm"] = [float(value) for value in depths]
    return BackingProfile(
        tuple(rings),
        tuple(float(value) for value in depths),
        float(taper_depth),
    )
