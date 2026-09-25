from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .hidden_interface import (
    MAX_BOUNDARY_THICKNESS_PROBES,
    boundary_screening_indices,
)


GUIDED_CUT_MINIMUM_RAY_DOT = 0.05
GUIDED_CUT_MINIMUM_LOCAL_INWARD_DOT = 0.01
GUIDED_CUT_PARENT_CLEARANCE_MM = 0.05


class ThicknessProbe(Protocol):
    triangles: np.ndarray
    active_triangle_mask: np.ndarray

    def first_hit_distances(
        self,
        points: np.ndarray,
        directions: np.ndarray,
        global_ceiling_mm: float,
    ) -> np.ndarray: ...

    def safety_limit(
        self,
        points: np.ndarray,
        directions: np.ndarray,
        global_ceiling_mm: float,
    ) -> tuple[float, dict]: ...


def _unit_vector(value, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain three finite numbers")
    length = float(np.linalg.norm(vector))
    if length <= 1e-12:
        raise ValueError(f"{name} must not be zero length")
    return vector / length


def _unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    lengths = np.linalg.norm(values, axis=1)
    if np.any(lengths <= 1e-12) or not np.isfinite(values).all():
        raise ValueError("guided cut direction field contains an invalid vector")
    return values / lengths[:, None]


def adaptive_guided_entry_ring(
    *,
    boundary_points: np.ndarray,
    interior_conormals: np.ndarray,
    parent_thickness_probe: ThicknessProbe,
    spec: "GuidedInternalCutSpec",
    base_inset_mm: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Recess only thin entry arcs, with a smooth circular influence field.

    A large uniform polygon offset can collapse a concave vendor boundary even
    though only one or two samples touch a thin outer shell.  Probe the normal
    section entry from the ordinary fit ring, identify those local samples,
    and spread the requested extra inset over a physical-radius neighborhood.
    The visible source loop is never moved.
    """
    boundary = np.asarray(boundary_points, dtype=np.float64)
    conormals = _unit_rows(interior_conormals)
    if boundary.shape != conormals.shape or len(boundary) < 3:
        raise ValueError("guided entry boundary and conormals must match")
    base = max(float(base_inset_mm), 0.0)
    maximum = max(float(spec.entry_inset_mm), base)
    base_points = boundary + conormals * base
    if maximum <= base + 1e-9:
        return base_points, np.full(len(boundary), base), {
            "guided_entry_inset_policy": "uniform_base_clearance",
            "guided_entry_unsafe_seed_vertices": 0,
            "guided_entry_inset_min_mm": base,
            "guided_entry_inset_max_mm": base,
            "guided_entry_thickness_boundary_input_vertices": int(len(boundary)),
            "guided_entry_thickness_boundary_sampled_vertices": 0,
        }

    entry_directions = np.tile(spec.entry_direction, (len(boundary), 1))
    search_limit = min(max(float(spec.maximum_depth_mm), 0.0), 10.0) + 0.05
    thickness_sample_indices = boundary_screening_indices(
        base_points,
        MAX_BOUNDARY_THICKNESS_PROBES,
    )
    sampled_hits = parent_thickness_probe.first_hit_distances(
        base_points[thickness_sample_indices],
        entry_directions[thickness_sample_indices],
        search_limit,
    )
    sampled_unsafe = np.asarray(sampled_hits, dtype=np.float64) < (
        float(spec.minimum_depth_mm) + GUIDED_CUT_PARENT_CLEARANCE_MM
    )
    unsafe = np.zeros(len(boundary), dtype=bool)
    unsafe[thickness_sample_indices] = sampled_unsafe
    seed_count = int(np.count_nonzero(unsafe))
    if seed_count == 0:
        return base_points, np.full(len(boundary), base), {
            "guided_entry_inset_policy": "uniform_base_clearance",
            "guided_entry_unsafe_seed_vertices": 0,
            "guided_entry_inset_min_mm": base,
            "guided_entry_inset_max_mm": base,
            "guided_entry_thickness_boundary_input_vertices": int(len(boundary)),
            "guided_entry_thickness_boundary_sampled_vertices": int(
                len(thickness_sample_indices)
            ),
        }

    edge_lengths = np.linalg.norm(
        np.roll(boundary, -1, axis=0) - boundary,
        axis=1,
    )
    positive_edges = edge_lengths[edge_lengths > 1e-8]
    median_edge = float(np.median(positive_edges)) if len(positive_edges) else 1.0
    # A roughly two-inset-wide influence gives the local entry enough runway
    # to avoid sharp ring gradients while leaving remote arcs untouched.
    sigma_vertices = max((2.0 * maximum) / max(median_edge, 1e-6), 2.0)
    radius = min(
        max(int(np.ceil(3.0 * sigma_vertices)), 2),
        max((len(boundary) - 1) // 3, 2),
    )
    offsets = np.arange(-radius, radius + 1, dtype=np.int64)
    kernel = np.exp(-0.5 * (offsets.astype(np.float64) / sigma_vertices) ** 2)
    weights = np.zeros(len(boundary), dtype=np.float64)
    for seed in np.flatnonzero(unsafe):
        indices = (int(seed) + offsets) % len(boundary)
        weights[indices] = np.maximum(weights[indices], kernel)
    weights[unsafe] = 1.0
    insets = base + (maximum - base) * weights

    direction_sigma_vertices = max(1.0 / max(median_edge, 1e-6), 2.0)
    direction_radius = min(
        max(int(np.ceil(3.0 * direction_sigma_vertices)), 2),
        max((len(boundary) - 1) // 4, 2),
    )
    direction_offsets = np.arange(
        -direction_radius,
        direction_radius + 1,
        dtype=np.int64,
    )
    direction_kernel = np.exp(
        -0.5
        * (direction_offsets.astype(np.float64) / direction_sigma_vertices) ** 2
    )
    direction_kernel /= direction_kernel.sum()
    smooth_conormals = np.zeros_like(conormals)
    for offset, weight in zip(direction_offsets, direction_kernel):
        smooth_conormals += np.roll(conormals, int(offset), axis=0) * float(weight)
    weak = np.linalg.norm(smooth_conormals, axis=1) <= 1e-8
    smooth_conormals[weak] = conormals[weak]
    smooth_conormals = _unit_rows(smooth_conormals)
    fit_points = boundary + smooth_conormals * insets[:, None]
    direction_dot = np.einsum("ij,ij->i", smooth_conormals, conormals)
    return fit_points, insets, {
        "guided_entry_inset_policy": "localized_parent_thickness_avoidance",
        "guided_entry_thickness_boundary_input_vertices": int(len(boundary)),
        "guided_entry_thickness_boundary_sampled_vertices": int(
            len(thickness_sample_indices)
        ),
        "guided_entry_unsafe_seed_vertices": seed_count,
        "guided_entry_unsafe_seed_indices": [
            int(value) for value in np.flatnonzero(unsafe)[:32]
        ],
        "guided_entry_influence_radius_vertices": int(radius),
        "guided_entry_influence_sigma_vertices": float(sigma_vertices),
        "guided_entry_direction_smoothing_radius_vertices": int(
            direction_radius
        ),
        "guided_entry_direction_smoothing_sigma_vertices": float(
            direction_sigma_vertices
        ),
        "guided_entry_direction_original_dot_min": float(
            direction_dot.min(initial=1.0)
        ),
        "guided_entry_boundary_median_edge_mm": median_edge,
        "guided_entry_inset_min_mm": float(insets.min(initial=base)),
        "guided_entry_inset_max_mm": float(insets.max(initial=base)),
        "guided_entry_inset_mean_mm": float(np.mean(insets)),
    }


@dataclass(frozen=True)
class GuidedInternalCutSpec:
    """Side-view section guide for one hidden child/parent interface.

    The entry direction is the black front transition.  The target plane is the
    red main cut surface.  Their cross product is the view-depth axis that the
    two-dimensional annotation intentionally leaves unconstrained.
    """

    entry_direction: np.ndarray
    target_plane_normal: np.ndarray
    target_plane_point_mm: np.ndarray
    minimum_depth_mm: float
    maximum_depth_mm: float
    entry_inset_mm: float = 0.0
    maximum_parallel_shift_mm: float = 0.0
    require_planar: bool = True

    @classmethod
    def from_mapping(cls, value: dict) -> "GuidedInternalCutSpec":
        if not isinstance(value, dict):
            raise ValueError("guided_internal_cut must be an object")
        entry = _unit_vector(value.get("entry_direction"), "entry_direction")
        normal = _unit_vector(
            value.get("target_plane_normal"), "target_plane_normal"
        )
        point = np.asarray(value.get("target_plane_point_mm"), dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("target_plane_point_mm must contain three finite numbers")
        minimum = float(value.get("minimum_depth_mm", 0.45))
        maximum = float(value.get("maximum_depth_mm", 10.0))
        if not np.isfinite(minimum) or minimum <= 0.0:
            raise ValueError("minimum_depth_mm must be positive and finite")
        if not np.isfinite(maximum) or maximum < minimum:
            raise ValueError("maximum_depth_mm must be finite and >= minimum_depth_mm")
        entry_inset = float(value.get("entry_inset_mm", 0.0))
        if not np.isfinite(entry_inset) or entry_inset < 0.0:
            raise ValueError("entry_inset_mm must be non-negative and finite")
        parallel_shift = float(value.get("maximum_parallel_shift_mm", 0.0))
        if not np.isfinite(parallel_shift) or parallel_shift < 0.0:
            raise ValueError(
                "maximum_parallel_shift_mm must be non-negative and finite"
            )
        if float(np.dot(entry, normal)) < 0.0:
            normal = -normal
        if float(np.dot(entry, normal)) <= GUIDED_CUT_MINIMUM_RAY_DOT:
            raise ValueError("entry_direction is nearly parallel to the target plane")
        return cls(
            entry_direction=entry,
            target_plane_normal=normal,
            target_plane_point_mm=point,
            minimum_depth_mm=minimum,
            maximum_depth_mm=maximum,
            entry_inset_mm=entry_inset,
            maximum_parallel_shift_mm=parallel_shift,
            require_planar=bool(value.get("require_planar", True)),
        )

    def as_record(self) -> dict:
        return {
            "entry_direction": self.entry_direction.round(9).tolist(),
            "target_plane_normal": self.target_plane_normal.round(9).tolist(),
            "target_plane_point_mm": self.target_plane_point_mm.round(9).tolist(),
            "minimum_depth_mm": float(self.minimum_depth_mm),
            "maximum_depth_mm": float(self.maximum_depth_mm),
            "entry_inset_mm": float(self.entry_inset_mm),
            "maximum_parallel_shift_mm": float(self.maximum_parallel_shift_mm),
            "require_planar": bool(self.require_planar),
        }


@dataclass(frozen=True)
class GuidedInternalCutPlan:
    spec: GuidedInternalCutSpec
    directions: np.ndarray
    distances: np.ndarray
    bottom_points: np.ndarray
    effective_plane_point_mm: np.ndarray
    record: dict


def _plane_distances(
    points: np.ndarray,
    directions: np.ndarray,
    spec: GuidedInternalCutSpec,
) -> tuple[np.ndarray, np.ndarray]:
    ray_dot = directions @ spec.target_plane_normal
    if np.any(ray_dot <= GUIDED_CUT_MINIMUM_RAY_DOT):
        raise ValueError("guided cut contains a ray nearly parallel to the target plane")
    plane_s = float(np.dot(spec.target_plane_point_mm, spec.target_plane_normal))
    distances = (plane_s - points @ spec.target_plane_normal) / ray_dot
    return distances, ray_dot


def _smooth_closed_field(
    directions: np.ndarray,
    local_inward_normals: np.ndarray,
    iterations: int = 2,
) -> np.ndarray:
    directions = _unit_rows(directions)
    local = _unit_rows(local_inward_normals)
    for _ in range(max(int(iterations), 0)):
        directions = _unit_rows(
            np.roll(directions, 1, axis=0) * 0.15
            + directions * 0.70
            + np.roll(directions, -1, axis=0) * 0.15
        )
        local_dot = np.einsum("ij,ij->i", directions, local)
        unsafe = local_dot < GUIDED_CUT_MINIMUM_LOCAL_INWARD_DOT
        if np.any(unsafe):
            directions[unsafe] = _unit_rows(
                directions[unsafe]
                + local[unsafe]
                * (
                    GUIDED_CUT_MINIMUM_LOCAL_INWARD_DOT
                    - local_dot[unsafe]
                    + 0.01
                )[:, None]
            )
    return directions


def _active_parent_centroid(probe: ThicknessProbe) -> np.ndarray:
    triangles = np.asarray(probe.triangles, dtype=np.float64)
    active = np.asarray(probe.active_triangle_mask, dtype=bool)
    triangles = triangles[active]
    if not len(triangles):
        raise ValueError("guided cut requires active parent triangles")
    area2 = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    centroids = triangles.mean(axis=1)
    positive = area2 > 1e-12
    return (
        np.average(centroids[positive], axis=0, weights=area2[positive])
        if np.any(positive)
        else centroids.mean(axis=0)
    )


def _candidate_fields(
    points: np.ndarray,
    local_inward_normals: np.ndarray,
    parent_interior_point: np.ndarray,
    spec: GuidedInternalCutSpec,
) -> list[tuple[str, np.ndarray]]:
    entry = spec.entry_direction
    lateral = np.cross(entry, spec.target_plane_normal)
    lateral_length = float(np.linalg.norm(lateral))
    if lateral_length <= 1e-6:
        # A parallel entry/normal pair still leaves infinitely many view-depth
        # axes.  Pick the most stable axis perpendicular to the entry.
        basis = np.eye(3)[int(np.argmin(np.abs(entry)))]
        lateral = np.cross(entry, basis)
        lateral_length = float(np.linalg.norm(lateral))
    lateral /= lateral_length

    loop_center = points.mean(axis=0)
    target_lateral = float(np.dot(parent_interior_point, lateral))
    point_lateral = points @ lateral
    median_entry_depth = max(
        0.5 * (spec.minimum_depth_mm + spec.maximum_depth_mm),
        spec.minimum_depth_mm,
    )
    convergence_velocity = (target_lateral - point_lateral) / median_entry_depth
    local_lateral = local_inward_normals @ lateral

    raw: list[tuple[str, np.ndarray]] = [
        ("uniform_section_entry", np.tile(entry, (len(points), 1)))
    ]
    for lateral_velocity in (-1.50, -1.00, -0.60, -0.30, 0.30, 0.60, 1.00, 1.50):
        raw.append(
            (
                f"uniform_lateral_{lateral_velocity:+.2f}",
                np.tile(
                    entry + lateral * float(lateral_velocity),
                    (len(points), 1),
                ),
            )
        )
    for strength in (0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00):
        lateral_velocity = convergence_velocity * strength
        raw.append(
            (
                f"parent_lateral_convergence_{strength:.2f}",
                entry[None, :] + lateral_velocity[:, None] * lateral[None, :],
            )
        )
    for strength in (0.35, 0.70, 1.05, 1.40, 1.80, 2.20):
        lateral_velocity = local_lateral * strength
        raw.append(
            (
                f"local_lateral_field_{strength:.2f}",
                entry[None, :] + lateral_velocity[:, None] * lateral[None, :],
            )
        )
    for convergence_strength in (0.50, 0.75, 1.00):
        lateral_velocity = (
            convergence_velocity * convergence_strength + local_lateral * 0.35
        )
        raw.append(
            (
                f"hybrid_lateral_{convergence_strength:.2f}",
                entry[None, :] + lateral_velocity[:, None] * lateral[None, :],
            )
        )

    candidates: list[tuple[str, np.ndarray]] = []
    fingerprints: set[tuple] = set()
    for name, field in raw:
        smoothed = _smooth_closed_field(field, local_inward_normals)
        fingerprint = tuple(np.round(smoothed[:: max(len(smoothed) // 32, 1)], 4).reshape(-1))
        if fingerprint in fingerprints:
            continue
        fingerprints.add(fingerprint)
        candidates.append((name, smoothed))
    return candidates


def _guided_loft_quality(top: np.ndarray, bottom: np.ndarray) -> dict:
    """Rank section fields by the actual ruled wall between source and plane."""
    top = np.asarray(top, dtype=np.float64)
    bottom = np.asarray(bottom, dtype=np.float64)
    top_next = np.roll(top, -1, axis=0)
    bottom_next = np.roll(bottom, -1, axis=0)
    top_edges = np.linalg.norm(top_next - top, axis=1)
    bottom_edges = np.linalg.norm(bottom_next - bottom, axis=1)
    stretch = np.maximum(
        bottom_edges / np.maximum(top_edges, 1e-12),
        top_edges / np.maximum(bottom_edges, 1e-12),
    )
    conflicting = 0
    minimum_area = float("inf")
    for a0, a1, b0, b1 in zip(top, top_next, bottom, bottom_next):
        split_agreements: list[tuple[float, float]] = []
        for triangles in (
            ((a0, a1, b1), (a0, b1, b0)),
            ((a0, a1, b0), (a1, b1, b0)),
        ):
            normals = []
            areas = []
            for triangle in triangles:
                triangle = np.asarray(triangle, dtype=np.float64)
                normal = np.cross(
                    triangle[1] - triangle[0],
                    triangle[2] - triangle[0],
                )
                area = float(np.linalg.norm(normal))
                areas.append(area)
                normals.append(normal / area if area > 1e-14 else np.zeros(3))
            split_agreements.append(
                (float(np.dot(normals[0], normals[1])), min(areas))
            )
        agreement, area = max(split_agreements)
        minimum_area = min(minimum_area, area)
        conflicting += int(agreement < -0.25)
    return {
        "conflicting_quad_normals": int(conflicting),
        "maximum_edge_stretch": float(stretch.max(initial=1.0)),
        "maximum_bottom_edge_mm": float(bottom_edges.max(initial=0.0)),
        "minimum_triangle_double_area_mm2": float(minimum_area),
    }


class GuidedInternalCutPlanner:
    """Build a safe 3-D field while preserving a user-marked 2-D section."""

    @staticmethod
    def plan(
        *,
        points: np.ndarray,
        local_inward_normals: np.ndarray,
        parent_thickness_probe: ThicknessProbe,
        spec: GuidedInternalCutSpec,
        visible_top_points: np.ndarray | None = None,
    ) -> GuidedInternalCutPlan:
        points = np.asarray(points, dtype=np.float64)
        local = _unit_rows(local_inward_normals)
        if points.shape != local.shape or len(points) < 3:
            raise ValueError("guided cut points and local normals must be matching loops")
        if not np.isfinite(points).all():
            raise ValueError("guided cut points must be finite")
        visible_top = (
            points
            if visible_top_points is None
            else np.asarray(visible_top_points, dtype=np.float64)
        )
        if visible_top.shape != points.shape or not np.isfinite(visible_top).all():
            raise ValueError("guided cut visible top ring must match fit points")

        parent_interior = _active_parent_centroid(parent_thickness_probe)
        evaluations: list[dict] = []
        accepted: list[tuple[tuple, str, np.ndarray, np.ndarray, np.ndarray, dict]] = []
        for name, directions in _candidate_fields(
            points,
            local,
            parent_interior,
            spec,
        ):
            try:
                distances, ray_dot = _plane_distances(points, directions, spec)
            except ValueError as exc:
                evaluations.append({"name": name, "accepted": False, "reason": str(exc)})
                continue
            parallel_shift = max(
                0.0,
                float(
                    np.max(
                        (spec.minimum_depth_mm - distances) * ray_dot,
                        initial=0.0,
                    )
                ),
            )
            if parallel_shift > spec.maximum_parallel_shift_mm + 1e-9:
                evaluations.append(
                    {
                        "name": name,
                        "accepted": False,
                        "reason": "required_parallel_plane_shift_exceeds_limit",
                        "required_parallel_shift_mm": parallel_shift,
                    }
                )
                continue
            distances = distances + parallel_shift / ray_dot
            bottom = points + directions * distances[:, None]
            hit_ceiling = max(float(distances.max(initial=0.0)), spec.maximum_depth_mm)
            safe_depth, thickness_record = parent_thickness_probe.safety_limit(
                points,
                directions,
                hit_ceiling,
            )
            local_dot = np.einsum("ij,ij->i", directions, local)
            thickness_margin = float(safe_depth) - distances
            finite = bool(
                np.isfinite(distances).all()
                and np.isfinite(bottom).all()
                and np.isfinite(float(safe_depth))
            )
            within_depth = bool(
                finite
                and float(distances.min(initial=np.inf)) >= spec.minimum_depth_mm - 1e-9
                and float(distances.max(initial=-np.inf)) <= spec.maximum_depth_mm + 1e-9
            )
            safe = bool(
                within_depth
                and float(thickness_margin.min(initial=np.inf)) >= -1e-9
                and float(local_dot.min(initial=np.inf))
                >= GUIDED_CUT_MINIMUM_LOCAL_INWARD_DOT - 1e-9
            )
            record = {
                "name": name,
                "accepted": safe,
                "minimum_depth_mm": float(distances.min(initial=np.inf)),
                "maximum_depth_mm": float(distances.max(initial=-np.inf)),
                "safe_parent_depth_mm": float(safe_depth),
                "minimum_thickness_margin_mm": float(thickness_margin.min(initial=np.inf)),
                "minimum_local_inward_dot": float(local_dot.min(initial=np.inf)),
                "minimum_ray_dot_plane": float(ray_dot.min(initial=np.inf)),
                "parallel_plane_shift_mm": float(parallel_shift),
                "thickness_record": dict(thickness_record),
            }
            loft_quality = _guided_loft_quality(visible_top, bottom)
            record["guided_loft_quality"] = loft_quality
            evaluations.append(record)
            if safe:
                # Prefer clearance margin first, then continuity and fidelity to
                # the black entry direction.  Names are the deterministic tie-break.
                adjacent = np.linalg.norm(np.roll(bottom, -1, axis=0) - bottom, axis=1)
                entry_dot = float(np.mean(directions @ spec.entry_direction))
                score = (
                    int(loft_quality["conflicting_quad_normals"]),
                    float(loft_quality["maximum_edge_stretch"]),
                    -float(thickness_margin.min()),
                    float(adjacent.max(initial=0.0)),
                    -entry_dot,
                    name,
                )
                accepted.append((score, name, directions, distances, bottom, record))

        # A safe ray field can still pair nearby source samples with crossing
        # points on the marked plane.  Fair only the best few bottom rings in
        # that plane, reconstruct the rays, and rerun the thickness gate.  An
        # affine average of coplanar points remains exactly coplanar.
        refinement_seeds = sorted(accepted, key=lambda item: item[0])[:5]
        refined: list[
            tuple[tuple, str, np.ndarray, np.ndarray, np.ndarray, dict]
        ] = []
        for _seed_score, seed_name, _seed_directions, _seed_distances, seed_bottom, seed_record in refinement_seeds:
            if int(
                seed_record.get("guided_loft_quality", {}).get(
                    "conflicting_quad_normals", 0
                )
            ) == 0:
                continue
            smoothed_bottom = np.asarray(seed_bottom, dtype=np.float64).copy()
            completed_passes = 0
            for target_passes in (2, 4, 8, 16, 32):
                for _ in range(target_passes - completed_passes):
                    smoothed_bottom = (
                        np.roll(smoothed_bottom, 1, axis=0) * 0.25
                        + smoothed_bottom * 0.50
                        + np.roll(smoothed_bottom, -1, axis=0) * 0.25
                    )
                completed_passes = target_passes
                delta = smoothed_bottom - points
                variant_distances = np.linalg.norm(delta, axis=1)
                if np.any(variant_distances <= 1e-12):
                    continue
                variant_directions = delta / variant_distances[:, None]
                variant_local_dot = np.einsum(
                    "ij,ij->i", variant_directions, local
                )
                variant_ray_dot = (
                    variant_directions @ spec.target_plane_normal
                )
                finite = bool(
                    np.isfinite(variant_distances).all()
                    and np.isfinite(variant_directions).all()
                )
                within_depth = bool(
                    finite
                    and float(variant_distances.min(initial=np.inf))
                    >= spec.minimum_depth_mm - 1e-9
                    and float(variant_distances.max(initial=-np.inf))
                    <= spec.maximum_depth_mm + 1e-9
                    and float(variant_local_dot.min(initial=np.inf))
                    >= GUIDED_CUT_MINIMUM_LOCAL_INWARD_DOT - 1e-9
                    and float(variant_ray_dot.min(initial=np.inf))
                    >= GUIDED_CUT_MINIMUM_RAY_DOT - 1e-9
                )
                if not within_depth:
                    continue
                loft_quality = _guided_loft_quality(
                    visible_top, smoothed_bottom
                )
                if int(loft_quality["conflicting_quad_normals"]) >= int(
                    seed_record["guided_loft_quality"][
                        "conflicting_quad_normals"
                    ]
                ):
                    continue
                hit_ceiling = max(
                    float(variant_distances.max(initial=0.0)),
                    spec.maximum_depth_mm,
                )
                safe_depth, thickness_record = (
                    parent_thickness_probe.safety_limit(
                        points,
                        variant_directions,
                        hit_ceiling,
                    )
                )
                thickness_margin = float(safe_depth) - variant_distances
                safe = bool(
                    float(thickness_margin.min(initial=np.inf)) >= -1e-9
                )
                variant_name = f"{seed_name}_planar_fair_{target_passes}"
                variant_record = {
                    **dict(seed_record),
                    "name": variant_name,
                    "accepted": safe,
                    "minimum_depth_mm": float(variant_distances.min()),
                    "maximum_depth_mm": float(variant_distances.max()),
                    "safe_parent_depth_mm": float(safe_depth),
                    "minimum_thickness_margin_mm": float(
                        thickness_margin.min()
                    ),
                    "minimum_local_inward_dot": float(
                        variant_local_dot.min()
                    ),
                    "minimum_ray_dot_plane": float(variant_ray_dot.min()),
                    "thickness_record": dict(thickness_record),
                    "guided_planar_smoothing_passes": int(target_passes),
                    "guided_loft_quality": loft_quality,
                }
                evaluations.append(variant_record)
                if not safe:
                    continue
                entry_dot = float(
                    np.mean(variant_directions @ spec.entry_direction)
                )
                score = (
                    int(loft_quality["conflicting_quad_normals"]),
                    float(loft_quality["maximum_edge_stretch"]),
                    -float(thickness_margin.min()),
                    float(loft_quality["maximum_bottom_edge_mm"]),
                    -entry_dot,
                    variant_name,
                )
                refined.append(
                    (
                        score,
                        variant_name,
                        variant_directions.copy(),
                        variant_distances.copy(),
                        smoothed_bottom.copy(),
                        variant_record,
                    )
                )
                if int(loft_quality["conflicting_quad_normals"]) == 0:
                    break
        accepted.extend(refined)

        if not accepted:
            best_attempts = sorted(
                evaluations,
                key=lambda item: (
                    -float(item.get("safe_parent_depth_mm", -1.0)),
                    -float(item.get("minimum_thickness_margin_mm", -1e9)),
                    str(item.get("name", "")),
                ),
            )[:5]
            concise = [
                {
                    "name": item.get("name"),
                    "safe_parent_depth_mm": item.get("safe_parent_depth_mm"),
                    "minimum_depth_mm": item.get("minimum_depth_mm"),
                    "maximum_depth_mm": item.get("maximum_depth_mm"),
                    "minimum_thickness_margin_mm": item.get(
                        "minimum_thickness_margin_mm"
                    ),
                    "minimum_local_inward_dot": item.get(
                        "minimum_local_inward_dot"
                    ),
                }
                for item in best_attempts
            ]
            raise ValueError(
                "guided internal cut has no direction field satisfying the marked "
                f"plane, {spec.minimum_depth_mm:.3f}-{spec.maximum_depth_mm:.3f} mm "
                f"depth, and parent-thickness clearance; best_attempts={concise}"
            )
        accepted.sort(key=lambda item: item[0])
        _score, name, directions, distances, bottom, selected_record = accepted[0]
        effective_plane_point = (
            spec.target_plane_point_mm
            + spec.target_plane_normal
            * float(selected_record["parallel_plane_shift_mm"])
        )
        plane_residual = np.abs(
            (bottom - effective_plane_point[None, :]) @ spec.target_plane_normal
        )
        record = {
            "guided_internal_cut_policy": "section_projection_with_parent_lateral_search",
            "guided_internal_cut_selected_candidate": name,
            "guided_internal_cut_parent_interior_point_mm": parent_interior.round(6).tolist(),
            "guided_internal_cut_plane_residual_max_mm": float(plane_residual.max(initial=0.0)),
            "guided_internal_cut_effective_plane_point_mm": (
                effective_plane_point.round(9).tolist()
            ),
            "guided_internal_cut_candidates": evaluations,
            **spec.as_record(),
            **selected_record,
        }
        return GuidedInternalCutPlan(
            spec=spec,
            directions=directions.copy(),
            distances=distances.copy(),
            bottom_points=bottom.copy(),
            effective_plane_point_mm=effective_plane_point.copy(),
            record=record,
        )
