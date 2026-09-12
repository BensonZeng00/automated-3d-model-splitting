from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .common import *


HIDDEN_INTERFACE_MINIMUM_LOAD_BEARING_DEPTH_MM = 0.45
HIDDEN_INTERFACE_CANDIDATE_TARGET_FRACTIONS = (0.25, 0.50, 0.75, 1.00)
HIDDEN_INTERFACE_BLEND_FACTORS = (0.35, 0.65, 1.00)


class ThicknessProbe(Protocol):
    triangles: np.ndarray
    active_triangle_mask: np.ndarray

    def safety_limit(
        self,
        points: np.ndarray,
        directions: np.ndarray,
        global_ceiling_mm: float,
    ) -> tuple[float, dict]: ...


@dataclass(frozen=True)
class HiddenInterfaceCandidate:
    name: str
    axis: np.ndarray
    directions: np.ndarray
    safe_depth_mm: float
    minimum_local_inward_dot: float
    mean_initial_direction_dot: float
    thickness_record: dict


@dataclass(frozen=True)
class HiddenInterfacePlan:
    baseline_safe_depth_mm: float
    parent_interior_point: np.ndarray
    candidates: tuple[HiddenInterfaceCandidate, ...]
    record: dict


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1)[:, None], 1e-12)


def _active_parent_centroid(probe: ThicknessProbe) -> np.ndarray:
    triangles = np.asarray(probe.triangles, dtype=np.float64)
    active = np.asarray(probe.active_triangle_mask, dtype=bool)
    triangles = triangles[active]
    if not len(triangles):
        raise ValueError("hidden-interface planning requires active parent triangles")
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    weights = np.linalg.norm(cross, axis=1)
    centroids = triangles.mean(axis=1)
    positive = weights > 1e-12
    if np.any(positive):
        return np.average(centroids[positive], axis=0, weights=weights[positive])
    return centroids.mean(axis=0)


def _project_and_smooth_inward_field(
    field: np.ndarray,
    local_inward_normals: np.ndarray,
    iterations: int = 12,
    minimum_dot: float = 0.03,
) -> tuple[np.ndarray, float]:
    directions = _normalize_rows(field)
    local = _normalize_rows(local_inward_normals)
    if directions.shape != local.shape:
        raise ValueError("hidden-interface direction and local-normal fields must match")

    def project() -> None:
        dot = np.einsum("ij,ij->i", directions, local)
        unsafe = dot < float(minimum_dot)
        if not np.any(unsafe):
            return
        tangent = directions[unsafe] - dot[unsafe, None] * local[unsafe]
        tangent_length = np.linalg.norm(tangent, axis=1)
        valid = tangent_length > 1e-12
        projected = np.empty_like(tangent)
        tangent_scale = math.sqrt(max(0.0, 1.0 - minimum_dot * minimum_dot))
        projected[valid] = (
            tangent[valid] / tangent_length[valid, None] * tangent_scale
            + local[unsafe][valid] * minimum_dot
        )
        projected[~valid] = local[unsafe][~valid]
        directions[unsafe] = projected

    project()
    for _ in range(max(int(iterations), 0)):
        directions[:] = _normalize_rows(
            np.roll(directions, 1, axis=0) * 0.20
            + directions * 0.60
            + np.roll(directions, -1, axis=0) * 0.20
        )
        project()
    local_dot = np.einsum("ij,ij->i", directions, local)
    return directions, float(local_dot.min(initial=1.0))


def _candidate_fields(
    points: np.ndarray,
    initial_directions: np.ndarray,
    fallback_axis: np.ndarray,
    local_inward_normals: np.ndarray,
    parent_interior_point: np.ndarray,
) -> list[tuple[str, np.ndarray, np.ndarray, float]]:
    points = np.asarray(points, dtype=np.float64)
    initial = _normalize_rows(initial_directions)
    fallback = np.asarray(fallback_axis, dtype=np.float64)
    fallback /= max(float(np.linalg.norm(fallback)), 1e-12)
    loop_center = points.mean(axis=0)
    parent_delta = parent_interior_point - loop_center
    parent_axis = parent_delta / max(float(np.linalg.norm(parent_delta)), 1e-12)

    raw_fields: list[tuple[str, np.ndarray, np.ndarray]] = []
    raw_fields.append(
        (
            "uniform_parent_interior_axis",
            parent_axis,
            np.tile(parent_axis, (len(points), 1)),
        )
    )
    for target_fraction in HIDDEN_INTERFACE_CANDIDATE_TARGET_FRACTIONS:
        target = loop_center + parent_delta * float(target_fraction)
        target_field = _normalize_rows(target[None, :] - points)
        target_axis = target - loop_center
        target_axis /= max(float(np.linalg.norm(target_axis)), 1e-12)
        for blend_factor in HIDDEN_INTERFACE_BLEND_FACTORS:
            blend = float(blend_factor)
            field = _normalize_rows(initial * (1.0 - blend) + target_field * blend)
            raw_fields.append(
                (
                    f"parent_target_{target_fraction:.2f}_blend_{blend:.2f}",
                    target_axis,
                    field,
                )
            )

    candidates: list[tuple[str, np.ndarray, np.ndarray, float]] = []
    fingerprints: set[tuple] = set()
    for name, axis, raw_field in raw_fields:
        field, minimum_dot = _project_and_smooth_inward_field(
            raw_field,
            local_inward_normals,
        )
        mean_direction = field.mean(axis=0)
        if float(np.linalg.norm(mean_direction)) <= 1e-12:
            continue
        mean_direction /= float(np.linalg.norm(mean_direction))
        if float(np.dot(mean_direction, axis)) < 0.0:
            axis = -axis
        fingerprint = tuple(np.round(field[:: max(len(field) // 32, 1)], 4).reshape(-1))
        if fingerprint in fingerprints:
            continue
        fingerprints.add(fingerprint)
        mean_initial_dot = float(
            np.mean(np.einsum("ij,ij->i", field, initial))
        )
        candidates.append((name, axis.copy(), field, mean_initial_dot))
    return candidates


class HiddenInterfacePlanner:
    """Search deterministic inward fields toward the active parent interior."""

    @staticmethod
    def plan(
        *,
        points: np.ndarray,
        initial_directions: np.ndarray,
        fallback_axis: np.ndarray,
        local_inward_normals: np.ndarray,
        parent_thickness_probe: ThicknessProbe,
        baseline_safe_depth_mm: float,
        preferred_depth_mm: float,
    ) -> HiddenInterfacePlan:
        points = np.asarray(points, dtype=np.float64)
        initial = _normalize_rows(initial_directions)
        local = _normalize_rows(local_inward_normals)
        parent_interior_point = _active_parent_centroid(parent_thickness_probe)
        ceiling = min(
            max(float(preferred_depth_mm), HIDDEN_INTERFACE_MINIMUM_LOAD_BEARING_DEPTH_MM),
            MAXIMUM_SAFE_INWARD_DEPTH_MM,
        )
        evaluated: list[HiddenInterfaceCandidate] = []
        failures: list[dict] = []
        for name, axis, directions, mean_initial_dot in _candidate_fields(
            points,
            initial,
            fallback_axis,
            local,
            parent_interior_point,
        ):
            try:
                safe_depth, thickness_record = parent_thickness_probe.safety_limit(
                    points,
                    directions,
                    ceiling,
                )
            except (ValueError, RuntimeError) as exc:
                failures.append({"name": name, "error": str(exc)})
                continue
            local_dot = np.einsum("ij,ij->i", directions, local)
            evaluated.append(
                HiddenInterfaceCandidate(
                    name=name,
                    axis=np.asarray(axis, dtype=np.float64).copy(),
                    directions=np.asarray(directions, dtype=np.float64).copy(),
                    safe_depth_mm=float(safe_depth),
                    minimum_local_inward_dot=float(local_dot.min(initial=1.0)),
                    mean_initial_direction_dot=float(mean_initial_dot),
                    thickness_record=dict(thickness_record),
                )
            )
        evaluated.sort(
            key=lambda item: (
                -item.safe_depth_mm,
                -item.minimum_local_inward_dot,
                -item.mean_initial_direction_dot,
                item.name,
            )
        )
        improved = [
            item
            for item in evaluated
            if item.safe_depth_mm > float(baseline_safe_depth_mm) + 1e-6
        ]
        record = {
            "hidden_interface_policy": "deterministic_parent_interior_direction_search",
            "hidden_interface_baseline_safe_depth_mm": float(
                baseline_safe_depth_mm
            ),
            "hidden_interface_minimum_load_bearing_depth_mm": (
                HIDDEN_INTERFACE_MINIMUM_LOAD_BEARING_DEPTH_MM
            ),
            "hidden_interface_parent_interior_point_mm": (
                parent_interior_point.round(6).tolist()
            ),
            "hidden_interface_candidate_count": int(len(evaluated)),
            "hidden_interface_improved_candidate_count": int(len(improved)),
            "hidden_interface_candidates": [
                {
                    "name": item.name,
                    "safe_depth_mm": item.safe_depth_mm,
                    "minimum_local_inward_dot": item.minimum_local_inward_dot,
                    "mean_initial_direction_dot": item.mean_initial_direction_dot,
                    "axis": item.axis.round(6).tolist(),
                }
                for item in evaluated
            ],
            "hidden_interface_failures": failures,
        }
        return HiddenInterfacePlan(
            baseline_safe_depth_mm=float(baseline_safe_depth_mm),
            parent_interior_point=parent_interior_point.copy(),
            candidates=tuple(improved),
            record=record,
        )


