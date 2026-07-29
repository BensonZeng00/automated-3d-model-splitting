from __future__ import annotations

import time

from .common import *
from .project import *
from .recognition import *
from .mesh import *
from .package_io import *
from .selection import *
from .assembly import *
from .validation import *
from .boundary_fairing import BoundaryFairingService
from .domain import BoundaryFairingContext, CapDecision
from .reporting import runtime_log


def recursive_face_geometry_key(
    vertex_array: np.ndarray,
    face: np.ndarray,
) -> tuple:
    points = np.round(
        vertex_array[np.asarray(face, dtype=np.int64)],
        decimals=6,
    )
    return tuple(
        sorted(tuple(float(value) for value in point) for point in points)
    )


def finalize_recursive_colored_mesh(
    mesh: trimesh.Trimesh,
    face_color_codes: list[str],
    face_filament_slots: list[int | None],
    default_color_code: str,
) -> trimesh.Trimesh:
    """Finalize geometry while preserving each surviving triangle's material."""
    if len(face_color_codes) != len(mesh.faces):
        raise ValueError("recursive face-color count does not match mesh faces")
    if len(face_filament_slots) != len(mesh.faces):
        raise ValueError("recursive face-slot count does not match mesh faces")
    vertex_array = np.asarray(mesh.vertices, dtype=np.float64)
    color_meaning_by_geometry = {
        recursive_face_geometry_key(vertex_array, face): (str(code), slot)
        for face, code, slot in zip(
            np.asarray(mesh.faces),
            face_color_codes,
            face_filament_slots,
        )
    }
    mesh.visual.face_colors = np.asarray(
        [
            COLOR_INFO.get(code, {"rgba": [200, 200, 200, 255]})["rgba"]
            for code in face_color_codes
        ],
        dtype=np.uint8,
    )
    mesh = finalize_mesh(mesh)
    final_vertex_array = np.asarray(mesh.vertices, dtype=np.float64)
    default_slot = COLOR_INFO.get(default_color_code, {}).get("filament_slot")
    final_codes = []
    final_slots = []
    for face in np.asarray(mesh.faces):
        code, slot = color_meaning_by_geometry.get(
            recursive_face_geometry_key(final_vertex_array, face),
            (str(default_color_code), default_slot),
        )
        final_codes.append(str(code))
        final_slots.append(None if slot is None else int(slot))
    mesh.visual.face_colors = np.asarray(
        [
            COLOR_INFO.get(code, {"rgba": [200, 200, 200, 255]})["rgba"]
            for code in final_codes
        ],
        dtype=np.uint8,
    )
    mesh.metadata["face_color_codes"] = final_codes
    mesh.metadata["face_filament_slot_indices"] = final_slots
    return mesh


def component_inward_direction(
    vertices: np.ndarray,
    faces: np.ndarray,
    component: Component,
    model_center: np.ndarray,
) -> np.ndarray:
    local_vertices, local_faces, _, _ = build_local_mesh(vertices, faces, component)
    component_center = local_vertices[local_faces.reshape(-1)].mean(axis=0)
    return -average_outward_normal(local_vertices, local_faces, component_center, model_center)


def mesh_vertex_inward_normals(local_vertices: np.ndarray, local_faces: np.ndarray) -> np.ndarray:
    """Return area-weighted local inward normals for every source vertex."""
    triangles = local_vertices[local_faces]
    face_cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    accumulated = np.zeros_like(local_vertices, dtype=np.float64)
    for column in range(3):
        np.add.at(accumulated, local_faces[:, column], face_cross)
    lengths = np.linalg.norm(accumulated, axis=1)
    valid = lengths > 1e-12
    accumulated[valid] /= lengths[valid, None]
    accumulated[~valid] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return -accumulated


def safe_boundary_inward_directions(
    vertex_inward_normals: np.ndarray,
    loop: list[int],
    fallback_inward: np.ndarray,
    loop_points: np.ndarray | None = None,
    full_local_below_dot: float = 0.0,
    keep_global_above_dot: float = 0.50,
    flat_safe_dot: float = 0.05,
    smoothing_iterations: int = 32,
    smoothing_radius_mm: float = 1.20,
) -> tuple[np.ndarray, dict]:
    """Prefer one flat direction, otherwise build a continuous safe direction field."""
    fallback = np.asarray(fallback_inward, dtype=np.float64)
    fallback /= max(float(np.linalg.norm(fallback)), 1e-12)
    local = np.asarray(vertex_inward_normals[np.asarray(loop, dtype=np.int64)], dtype=np.float64)
    local /= np.maximum(np.linalg.norm(local, axis=1)[:, None], 1e-12)
    initial_dot = local @ fallback
    flat_direction_safe = bool(len(loop) == 0 or float(initial_dot.min()) >= float(flat_safe_dot))
    if flat_direction_safe:
        directions = np.tile(fallback, (len(loop), 1))
        adjacent_angles = np.zeros(len(loop), dtype=np.float64)
        return directions, {
            "vertices": int(len(loop)),
            "global_outward_vertices_before": int(np.count_nonzero(initial_dot < 0.0)),
            "global_outward_fraction_before": float(np.mean(initial_dot < 0.0)) if len(loop) else 0.0,
            "outward_vertices_after": 0,
            "minimum_global_dot_local_inward_before": float(initial_dot.min()) if len(loop) else 1.0,
            "minimum_safe_dot_local_inward_after": float(initial_dot.min()) if len(loop) else 1.0,
            "median_safe_dot_local_inward_after": float(np.median(initial_dot)) if len(loop) else 1.0,
            "corrected_vertices": 0,
            "maximum_direction_correction_degrees": 0.0,
            "maximum_adjacent_direction_angle_degrees": float(adjacent_angles.max()) if len(adjacent_angles) else 0.0,
            "p95_adjacent_direction_angle_degrees": 0.0,
            "flat_direction_safe": True,
            "flat_safe_dot_threshold": float(flat_safe_dot),
            "direction_mode": "single_global_flat_direction",
        }

    smoothing_half_window = 0
    smoothing_edge_median_mm = 0.0
    smooth_local = local.copy()
    if len(local) >= 4 and loop_points is not None:
        points = np.asarray(loop_points, dtype=np.float64)
        edge_lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
        positive_edges = edge_lengths[edge_lengths > 1e-8]
        if len(positive_edges):
            smoothing_edge_median_mm = float(np.median(positive_edges))
            maximum_window = max(1, min((len(local) - 1) // 4, 128))
            smoothing_half_window = min(
                maximum_window,
                max(2, int(math.ceil(max(float(smoothing_radius_mm), 0.0) / smoothing_edge_median_mm))),
            )
            offsets = np.arange(-smoothing_half_window, smoothing_half_window + 1, dtype=np.int64)
            sigma = max(float(smoothing_half_window) * 0.45, 1.0)
            weights = np.exp(-0.5 * (offsets.astype(np.float64) / sigma) ** 2)
            weights /= weights.sum()

            def circular_smooth(values: np.ndarray) -> np.ndarray:
                result = np.zeros_like(values, dtype=np.float64)
                for offset, weight in zip(offsets, weights):
                    result += np.roll(values, int(offset), axis=0) * float(weight)
                result /= np.maximum(np.linalg.norm(result, axis=1)[:, None], 1e-12)
                return result

            smooth_local = circular_smooth(local)
        else:
            circular_smooth = None
    else:
        circular_smooth = None

    smooth_initial_dot = smooth_local @ fallback
    lower = float(full_local_below_dot)
    upper = max(float(keep_global_above_dot), lower + 1e-6)
    blend = np.clip((smooth_initial_dot - lower) / (upper - lower), 0.0, 1.0)
    blend = blend * blend * (3.0 - 2.0 * blend)
    directions = smooth_local * (1.0 - blend[:, None]) + fallback[None, :] * blend[:, None]
    directions /= np.maximum(np.linalg.norm(directions, axis=1)[:, None], 1e-12)
    if circular_smooth is not None:
        directions = circular_smooth(directions)

    # Smooth on the closed boundary, then project minimally back into each
    # vertex's local inward hemisphere.  Repeating both operations prevents
    # the abrupt per-vertex normal changes which twist the bottom ring into
    # radial folds while retaining a strictly inward first-order direction.
    minimum_safe_dot = 0.02
    for _ in range(max(int(smoothing_iterations), 0)):
        directions = (
            np.roll(directions, 1, axis=0) * 0.25
            + directions * 0.50
            + np.roll(directions, -1, axis=0) * 0.25
        )
        directions /= np.maximum(np.linalg.norm(directions, axis=1)[:, None], 1e-12)
        current_dot = np.einsum("ij,ij->i", smooth_local, directions)
        unsafe = current_dot < minimum_safe_dot
        if np.any(unsafe):
            unsafe_directions = directions[unsafe]
            unsafe_local = smooth_local[unsafe]
            unsafe_dot = current_dot[unsafe]
            tangent = unsafe_directions - unsafe_dot[:, None] * unsafe_local
            tangent_length = np.linalg.norm(tangent, axis=1)
            valid_tangent = tangent_length > 1e-12
            projected = np.empty_like(unsafe_directions)
            projected[valid_tangent] = (
                tangent[valid_tangent]
                / tangent_length[valid_tangent, None]
                * math.sqrt(max(0.0, 1.0 - minimum_safe_dot * minimum_safe_dot))
                + unsafe_local[valid_tangent] * minimum_safe_dot
            )
            projected[~valid_tangent] = unsafe_local[~valid_tangent]
            directions[unsafe] = projected
        directions /= np.maximum(np.linalg.norm(directions, axis=1)[:, None], 1e-12)

    final_dot = np.einsum("ij,ij->i", smooth_local, directions)
    unsafe = final_dot <= 1e-6
    if np.any(unsafe):
        directions[unsafe] = local[unsafe]
        final_dot[unsafe] = 1.0
    angular_change = np.degrees(
        np.arccos(np.clip(directions @ fallback, -1.0, 1.0))
    )
    adjacent_angles = np.degrees(
        np.arccos(np.clip(np.einsum("ij,ij->i", directions, np.roll(directions, -1, axis=0)), -1.0, 1.0))
    )
    corrected = initial_dot < upper
    return directions, {
        "vertices": int(len(loop)),
        "global_outward_vertices_before": int(np.count_nonzero(initial_dot < 0.0)),
        "global_outward_fraction_before": float(np.mean(initial_dot < 0.0)) if len(loop) else 0.0,
        "outward_vertices_after": int(np.count_nonzero(final_dot <= 0.0)),
        "minimum_global_dot_local_inward_before": float(initial_dot.min()) if len(loop) else 1.0,
        "minimum_safe_dot_local_inward_after": float(final_dot.min()) if len(loop) else 1.0,
        "median_safe_dot_local_inward_after": float(np.median(final_dot)) if len(loop) else 1.0,
        "corrected_vertices": int(np.count_nonzero(corrected)),
        "maximum_direction_correction_degrees": float(angular_change.max()) if len(loop) else 0.0,
        "maximum_adjacent_direction_angle_degrees": float(adjacent_angles.max()) if len(adjacent_angles) else 0.0,
        "p95_adjacent_direction_angle_degrees": float(np.percentile(adjacent_angles, 95.0)) if len(adjacent_angles) else 0.0,
        "flat_direction_safe": False,
        "flat_safe_dot_threshold": float(flat_safe_dot),
        "direction_smoothing_iterations": int(max(int(smoothing_iterations), 0)),
        "direction_smoothing_radius_mm": float(max(float(smoothing_radius_mm), 0.0)),
        "direction_smoothing_half_window_vertices": int(smoothing_half_window),
        "direction_smoothing_edge_median_mm": float(smoothing_edge_median_mm),
        "raw_local_outward_vertices_after": int(
            np.count_nonzero(np.einsum("ij,ij->i", local, directions) <= 0.0)
        ),
        "direction_mode": "smoothed_boundary_local_hemisphere",
    }


def reference_loop_inward_directions(
    ref: dict | None,
    global_loop: list[int],
    fallback_inward: np.ndarray,
) -> np.ndarray:
    mapping = (ref or {}).get("inward_by_global", {})
    fallback = np.asarray(fallback_inward, dtype=np.float64)
    fallback /= max(float(np.linalg.norm(fallback)), 1e-12)
    return np.asarray(
        [mapping.get(int(global_id), fallback) for global_id in global_loop],
        dtype=np.float64,
    )


def effective_feature_clearance(
    component: Component,
    requested_mm: float,
    profile: str = "feature-adaptive",
    feature_ratio: float = 0.04,
    minimum_mm: float = 0.05,
) -> tuple[float, dict]:
    """Clamp clearance for small painted details without changing the visible top surface."""
    requested = max(float(requested_mm), 0.0)
    extents = np.asarray(component.bbox_max, dtype=np.float64) - np.asarray(
        component.bbox_min, dtype=np.float64
    )
    positive_extents = extents[extents > 1e-6]
    feature_width = float(positive_extents.min()) if len(positive_extents) else 0.0
    if profile == "fixed" or requested <= 0.0 or feature_width <= 0.0:
        effective = requested
        reason = "fixed_profile" if profile == "fixed" else "no_positive_feature_extent"
    else:
        feature_limit = max(float(minimum_mm), feature_width * max(float(feature_ratio), 1e-9))
        effective = min(requested, feature_limit)
        reason = "feature_scale_clamp" if effective < requested - 1e-12 else "requested_within_feature_limit"
    return effective, {
        "profile": str(profile),
        "requested_fit_clearance_mm": requested,
        "effective_fit_clearance_mm": float(effective),
        "feature_width_mm": feature_width,
        "feature_ratio": float(feature_ratio),
        "minimum_target_mm": float(minimum_mm),
        "reason": reason,
    }


def build_component_cut_references(
    vertices: np.ndarray,
    faces: np.ndarray,
    components: list[Component],
    model_center: np.ndarray,
    inward_overrides: dict[int, np.ndarray] | None = None,
    fit_clearance_by_part: dict[int, float] | None = None,
    clearance_mode: str = "insert-shrink",
) -> list[dict]:
    inward_overrides = inward_overrides or {}
    fit_clearance_by_part = fit_clearance_by_part or {}
    refs = []
    for index, component in enumerate(components, start=1):
        local_vertices, local_faces, _, global_vertex_ids = build_local_mesh(vertices, faces, component)
        loops = boundary_loops(local_faces)
        component_center = local_vertices[local_faces.reshape(-1)].mean(axis=0)
        inward = inward_overrides.get(index)
        if inward is None:
            inward = -average_outward_normal(local_vertices, local_faces, component_center, model_center)
        inward = inward / max(float(np.linalg.norm(inward)), 1e-12)
        vertex_inward_normals = mesh_vertex_inward_normals(local_vertices, local_faces)
        color_info = COLOR_INFO.get(component.color_code, {"name": component.color_code})
        for loop_index, loop in enumerate(loops):
            loop_global = [int(global_vertex_ids[i]) for i in loop]
            loop_directions, direction_record = safe_boundary_inward_directions(
                vertex_inward_normals,
                loop,
                inward,
                loop_points=local_vertices[np.asarray(loop, dtype=np.int64)],
            )
            refs.append(
                {
                    "component_index": index,
                    "color_code": component.color_code,
                    "color_name": color_info["name"],
                    "loop_index": loop_index,
                    "global_vertices": set(loop_global),
                    "global_loop": loop_global,
                    "global_edges": {
                        tuple(sorted((loop_global[position], loop_global[(position + 1) % len(loop_global)])))
                        for position in range(len(loop_global))
                    },
                    "inward": inward,
                    "inward_by_global": {
                        int(global_id): direction.copy()
                        for global_id, direction in zip(loop_global, loop_directions)
                    },
                    "local_inward_direction_record": direction_record,
                    "fit_clearance_mm": float(fit_clearance_by_part.get(index, 0.0)),
                    "socket_overcut_mm": float(
                        clearance_offsets(clearance_mode, fit_clearance_by_part.get(index, 0.0))[1]
                    ),
                }
            )
    return refs


def build_boundary_cut_references(
    vertices: np.ndarray,
    faces: np.ndarray,
    components: list[Component],
    body_component: Component,
    model_center: np.ndarray,
) -> list[dict]:
    return [
        ref
        for ref in build_component_cut_references(vertices, faces, components, model_center)
        if components[int(ref["component_index"]) - 1] is not body_component
    ]


def subtree_component_indices(root_index: int, children: dict[int, list[int]]) -> list[int]:
    indices = [int(root_index)]
    for child_index in children.get(int(root_index), []):
        indices.extend(subtree_component_indices(int(child_index), children))
    return sorted(indices)


def build_subassembly_component(
    vertices: np.ndarray,
    faces: np.ndarray,
    components: list[Component],
    indices: list[int],
    color_code: str,
) -> Component:
    global_faces = np.concatenate([components[int(index) - 1].global_faces for index in indices])
    group_faces = faces[global_faces]
    points = vertices[group_faces.reshape(-1)]
    areas = triangle_areas(vertices, faces)
    return Component(
        color_code=color_code,
        global_faces=global_faces,
        face_count=int(len(global_faces)),
        area=float(areas[global_faces].sum()),
        bbox_min=points.min(axis=0),
        bbox_max=points.max(axis=0),
        center=points.mean(axis=0),
    )


def boundary_loop_parent_contact(
    loop: list[int],
    global_vertex_ids: np.ndarray,
    boundary_neighbor_lookup: dict[tuple[int, int], set[int]],
    parent_index: int,
    subtree_indices: set[int],
) -> dict:
    neighbor_counts: collections.Counter[int] = collections.Counter()
    parent_edges = 0
    external_edges = 0
    for position, local_a in enumerate(loop):
        local_b = int(loop[(position + 1) % len(loop)])
        edge = tuple(sorted((int(global_vertex_ids[int(local_a)]), int(global_vertex_ids[local_b]))))
        neighbors = {int(value) for value in boundary_neighbor_lookup.get(edge, set())}
        for neighbor_index in neighbors:
            if neighbor_index not in subtree_indices:
                neighbor_counts[neighbor_index] += 1
        if int(parent_index) in neighbors:
            parent_edges += 1
        if any(neighbor_index not in subtree_indices for neighbor_index in neighbors):
            external_edges += 1
    return {
        "parent_edges": int(parent_edges),
        "external_edges": int(external_edges),
        "neighbor_counts": {str(key): int(value) for key, value in sorted(neighbor_counts.items())},
    }


def build_layer_child_cut_references(
    vertices: np.ndarray,
    faces: np.ndarray,
    components: list[Component],
    parent_index: int,
    direct_child_indices: list[int],
    assembly_children: dict[int, list[int]],
    model_center: np.ndarray,
    boundary_neighbor_lookup: dict[tuple[int, int], set[int]],
    inward_overrides: dict[int, np.ndarray],
    effective_cap_mode,
    effective_planar_extra_limit=None,
    fit_clearance_by_part: dict[int, float] | None = None,
    clearance_mode: str = "insert-shrink",
    boundary_fairing: BoundaryFairingContext | None = None,
    max_extension_mm: float | None = None,
    flat_clearance_mm: float | None = None,
    bottom_clearance_mm: float = 0.0,
) -> tuple[list[dict], dict[int, Component], dict[int, list[int]]]:
    context_started_at = time.perf_counter()
    fit_clearance_by_part = fit_clearance_by_part or {}
    cap_planning_enabled = (
        boundary_fairing is not None
        and max_extension_mm is not None
        and flat_clearance_mm is not None
    )
    face_owner_indices = np.zeros(len(faces), dtype=np.int32)
    for component_index, component in enumerate(components, start=1):
        component_faces = np.asarray(component.global_faces, dtype=np.int64)
        if len(component_faces):
            face_owner_indices[component_faces] = int(component_index)
    parent_thickness_probe = (
        ParentThicknessProbe(
            vertices,
            faces,
            triangle_owner_indices=face_owner_indices,
        )
        if cap_planning_enabled
        else None
    )
    runtime_log(
        "递归预计算",
        "layer_child_context_start",
        "开始预计算直属子件帽底和母槽上下文",
        parent_index=int(parent_index),
        direct_child_indices=sorted(int(index) for index in direct_child_indices),
        source_face_count=int(len(faces)),
        cap_planning_enabled=bool(cap_planning_enabled),
    )
    refs: list[dict] = []
    union_by_child: dict[int, Component] = {}
    subtree_by_child: dict[int, list[int]] = {}
    for child_index in sorted(int(index) for index in direct_child_indices):
        child_started_at = time.perf_counter()
        subtree = subtree_component_indices(child_index, assembly_children)
        runtime_log(
            "递归预计算",
            "layer_child_start",
            "开始预计算直属子件",
            parent_index=int(parent_index),
            child_index=int(child_index),
            subtree_indices=[int(index) for index in subtree],
        )
        subtree_by_child[child_index] = subtree
        union_component = build_subassembly_component(
            vertices,
            faces,
            components,
            subtree,
            components[child_index - 1].color_code,
        )
        union_by_child[child_index] = union_component
        child_parent_thickness_probe = parent_thickness_probe
        if parent_thickness_probe is not None:
            excluded_subtree_faces = np.unique(
                np.asarray(union_component.global_faces, dtype=np.int64)
            )
            child_parent_thickness_probe = (
                parent_thickness_probe.excluding_triangles(
                    excluded_subtree_faces,
                    filter_context={
                        "parent_part_index": int(parent_index),
                        "child_part_index": int(child_index),
                        "excluded_subtree_part_indices": [
                            int(index) for index in subtree
                        ],
                    },
                )
            )
            runtime_log(
                "递归预计算",
                "parent_thickness_probe_subtree_excluded",
                "父体厚度探针已排除当前子件整棵子树",
                parent_index=int(parent_index),
                child_index=int(child_index),
                subtree_indices=[int(index) for index in subtree],
                total_face_count=int(len(faces)),
                excluded_subtree_face_count=int(len(excluded_subtree_faces)),
                included_probe_face_count=int(
                    child_parent_thickness_probe.active_triangle_count
                ),
            )
        root_component = components[child_index - 1]
        local_vertices, local_faces, _global_to_local, global_vertex_ids = build_local_mesh(vertices, faces, root_component)
        loops = boundary_loops(local_faces)
        runtime_log(
            "递归预计算",
            "layer_child_boundary_done",
            "直属子件局部网格和边界环已建立",
            parent_index=int(parent_index),
            child_index=int(child_index),
            local_face_count=int(len(local_faces)),
            boundary_loop_count=int(len(loops)),
            stage_elapsed_seconds=round(
                float(time.perf_counter() - child_started_at),
                3,
            ),
        )
        inward = inward_overrides.get(child_index)
        if inward is None:
            inward = component_inward_direction(vertices, faces, root_component, model_center)
        inward = inward / max(float(np.linalg.norm(inward)), 1e-12)
        vertex_inward_normals = mesh_vertex_inward_normals(local_vertices, local_faces)
        color_info = COLOR_INFO.get(union_component.color_code, {"name": union_component.color_code})
        current_layer_component_set = {int(child_index)}
        selected_loop_records: list[dict] = []
        for loop_index, loop in enumerate(loops):
            contact = boundary_loop_parent_contact(loop, global_vertex_ids, boundary_neighbor_lookup, parent_index, current_layer_component_set)
            if int(contact["parent_edges"]) < 3:
                continue
            loop_global = [int(global_vertex_ids[i]) for i in loop]
            loop_directions, direction_record = safe_boundary_inward_directions(
                vertex_inward_normals,
                loop,
                inward,
                loop_points=local_vertices[np.asarray(loop, dtype=np.int64)],
            )
            selected_loop_records.append(
                {
                    "loop_index": int(loop_index),
                    "loop": [int(value) for value in loop],
                    "global_loop": loop_global,
                    "loop_directions": loop_directions,
                    "direction_record": direction_record,
                    "contact": contact,
                }
            )

        planned_vertices = local_vertices
        if cap_planning_enabled and selected_loop_records:
            fairing_started_at = time.perf_counter()
            planned_vertices, _fairing_records = BoundaryFairingService.fair_local_loops(
                local_vertices,
                [record["loop"] for record in selected_loop_records],
                global_vertex_ids,
                boundary_fairing,
            )
            runtime_log(
                "递归预计算",
                "layer_child_fairing_done",
                "直属子件边界平顺预计算完成",
                parent_index=int(parent_index),
                child_index=int(child_index),
                selected_loop_count=int(len(selected_loop_records)),
                stage_elapsed_seconds=round(
                    float(time.perf_counter() - fairing_started_at),
                    3,
                ),
            )

        requested_cap_mode = str(effective_cap_mode(child_index))
        planar_extra_limit_mm = (
            None
            if effective_planar_extra_limit is None
            else float(effective_planar_extra_limit(child_index))
        )
        fit_clearance_mm = float(fit_clearance_by_part.get(child_index, 0.0))
        insert_shrink_mm, socket_overcut_mm = clearance_offsets(
            clearance_mode,
            fit_clearance_mm,
        )
        for selected in selected_loop_records:
            loop_started_at = time.perf_counter()
            loop = selected["loop"]
            loop_global = selected["global_loop"]
            loop_directions = selected["loop_directions"]
            contact = selected["contact"]
            ref = {
                "component_index": int(child_index),
                "color_code": union_component.color_code,
                "color_name": color_info["name"],
                "loop_index": int(selected["loop_index"]),
                "global_vertices": set(loop_global),
                "global_loop": loop_global,
                "global_edges": {
                    tuple(
                        sorted(
                            (
                                loop_global[position],
                                loop_global[(position + 1) % len(loop_global)],
                            )
                        )
                    )
                    for position in range(len(loop_global))
                },
                "inward": inward,
                "inward_by_global": {
                    int(global_id): direction.copy()
                    for global_id, direction in zip(loop_global, loop_directions)
                },
                "local_inward_direction_record": selected["direction_record"],
                "requested_cap_mode": requested_cap_mode,
                "cap_mode": requested_cap_mode,
                "planar_extra_limit_mm": planar_extra_limit_mm,
                "stage_subtree_indices": sorted(int(index) for index in subtree),
                "contact_source_component_index": int(child_index),
                "parent_contact_edges": int(contact["parent_edges"]),
                "parent_contact_neighbor_counts": contact["neighbor_counts"],
                "fit_clearance_mm": fit_clearance_mm,
                "socket_overcut_mm": float(socket_overcut_mm),
            }
            if cap_planning_enabled:
                runtime_log(
                    "递归预计算",
                    "layer_child_cap_start",
                    "开始测量直属子件帽底安全深度",
                    parent_index=int(parent_index),
                    child_index=int(child_index),
                    loop_index=int(selected["loop_index"]),
                    boundary_vertex_count=int(len(loop)),
                )
                original_points = planned_vertices[
                    np.asarray(loop, dtype=np.int64)
                ]
                u, v = orthonormal_basis(inward)
                fit_points = radial_offset_points(
                    original_points,
                    original_points.mean(axis=0),
                    u,
                    v,
                    inward,
                    -max(float(insert_shrink_mm), 0.0),
                )
                socket_points = radial_offset_points(
                    original_points,
                    original_points.mean(axis=0),
                    u,
                    v,
                    inward,
                    max(float(socket_overcut_mm), 0.0),
                )
                cap_decision = plan_cap_decision(
                    points=fit_points,
                    source_vertex_ids=loop_global,
                    fallback_inward=inward,
                    inward_directions=loop_directions,
                    fixed_depth_mm=float(max_extension_mm),
                    flat_clearance_mm=float(flat_clearance_mm),
                    cap_mode=requested_cap_mode,
                    planar_extra_limit_mm=max(
                        float(planar_extra_limit_mm or 0.0)
                        - max(float(bottom_clearance_mm), 0.0),
                        0.0,
                    ),
                    parent_thickness_probe=child_parent_thickness_probe,
                )
                reserved_distances, reserved_record = reserve_flat_socket_travel_budget(
                    child_fit_points=fit_points,
                    child_distances=cap_decision.distances,
                    child_directions=cap_decision.directions,
                    socket_top_points=socket_points,
                    bottom_clearance_mm=bottom_clearance_mm,
                    maximum_socket_travel_mm=float(max_extension_mm)
                    + max(float(planar_extra_limit_mm or 0.0), 0.0),
                    plane_record=cap_decision.record,
                )
                cap_decision = CapDecision(
                    mode=str(reserved_record["cap_mode"]),
                    source_vertex_ids=cap_decision.source_vertex_ids,
                    fit_points=np.asarray(
                        cap_decision.fit_points,
                        dtype=np.float64,
                    ).copy(),
                    directions=np.asarray(
                        cap_decision.directions,
                        dtype=np.float64,
                    ).copy(),
                    distances=np.asarray(
                        reserved_distances,
                        dtype=np.float64,
                    ).copy(),
                    record=dict(reserved_record),
                )
                ref["cap_decision"] = cap_decision
                ref["cap_mode"] = cap_decision.mode
                runtime_log(
                    "递归预计算",
                    "layer_child_cap_done",
                    "直属子件帽底安全深度测量完成",
                    parent_index=int(parent_index),
                    child_index=int(child_index),
                    loop_index=int(selected["loop_index"]),
                    boundary_vertex_count=int(len(loop)),
                    selected_cap_mode=str(cap_decision.mode),
                    stage_elapsed_seconds=round(
                        float(time.perf_counter() - loop_started_at),
                        3,
                    ),
                )
            refs.append(ref)
        runtime_log(
            "递归预计算",
            "layer_child_done",
            "直属子件帽底和母槽上下文预计算完成",
            parent_index=int(parent_index),
            child_index=int(child_index),
            selected_loop_count=int(len(selected_loop_records)),
            stage_elapsed_seconds=round(
                float(time.perf_counter() - child_started_at),
                3,
            ),
        )
    runtime_log(
        "递归预计算",
        "layer_child_context_done",
        "全部直属子件帽底和母槽上下文预计算完成",
        parent_index=int(parent_index),
        child_count=int(len(direct_child_indices)),
        cut_reference_count=int(len(refs)),
        stage_elapsed_seconds=round(
            float(time.perf_counter() - context_started_at),
            3,
        ),
    )
    return refs, union_by_child, subtree_by_child


def best_cut_reference(
    body_loop_global_vertices: set[int],
    body_loop_global_edges: set[tuple[int, int]],
    cut_refs: list[dict],
) -> dict | None:
    best_ref = None
    best_rank = None
    for ref in cut_refs:
        ref_edges = ref.get("global_edges", set())
        edge_overlap = len(body_loop_global_edges.intersection(ref_edges))
        vertex_overlap = len(body_loop_global_vertices.intersection(ref["global_vertices"]))
        minimum_edge_overlap = max(3, int(math.ceil(0.1 * max(len(body_loop_global_edges), 1))))
        if edge_overlap < minimum_edge_overlap:
            continue
        rank = (
            edge_overlap / max(len(body_loop_global_edges), 1),
            edge_overlap,
            vertex_overlap,
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_ref = ref
    return best_ref


def classify_body_cut_loop_references(
    loops: list[list[int]],
    global_vertex_ids: np.ndarray,
    cut_refs: list[dict],
    *,
    preserve_unmatched_source_geometry: bool,
) -> tuple[list[dict | None], list[dict]]:
    """Match body boundary loops and identify inherited loops that must stay immutable.

    A reloaded recursive body already owns the parent-contact shell emitted by
    its parent step. Only loops matching a current direct-child reference may
    be changed; every unmatched loop belongs to that validated input state.
    """
    loop_refs: list[dict | None] = []
    immutable_records: list[dict] = []
    for loop_index, loop in enumerate(loops):
        ordered_global_ids = [
            int(global_vertex_ids[int(local_index)]) for local_index in loop
        ]
        global_edges = {
            tuple(
                sorted(
                    (
                        ordered_global_ids[position],
                        ordered_global_ids[(position + 1) % len(ordered_global_ids)],
                    )
                )
            )
            for position in range(len(ordered_global_ids))
        }
        ref = best_cut_reference(
            set(ordered_global_ids),
            global_edges,
            cut_refs,
        )
        loop_refs.append(ref)
        if preserve_unmatched_source_geometry and ref is None:
            immutable_records.append(
                {
                    "loop_index": int(loop_index),
                    "vertices": int(len(loop)),
                    "reason": "inherited_parent_contact_shell_or_validated_source_boundary",
                }
            )
    return loop_refs, immutable_records


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


class ParentThicknessProbe:
    """Measure the first opposite-surface hit along inward boundary rays."""

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        triangle_source_face_indices: np.ndarray | None = None,
        triangle_owner_indices: np.ndarray | None = None,
    ) -> None:
        self.triangles = np.asarray(vertices, dtype=np.float64)[
            np.asarray(faces, dtype=np.int64)
        ]
        triangle_count = int(len(self.triangles))
        self.triangle_source_face_indices = (
            np.arange(triangle_count, dtype=np.int64)
            if triangle_source_face_indices is None
            else np.asarray(
                triangle_source_face_indices,
                dtype=np.int64,
            ).copy()
        )
        self.triangle_owner_indices = (
            np.zeros(triangle_count, dtype=np.int32)
            if triangle_owner_indices is None
            else np.asarray(
                triangle_owner_indices,
                dtype=np.int32,
            ).copy()
        )
        if self.triangle_source_face_indices.shape != (triangle_count,):
            raise ValueError(
                "triangle_source_face_indices must match the probe triangle count"
            )
        if self.triangle_owner_indices.shape != (triangle_count,):
            raise ValueError(
                "triangle_owner_indices must match the probe triangle count"
            )
        self.active_triangle_mask = np.ones(triangle_count, dtype=bool)
        self.active_triangle_count = triangle_count
        self.filter_context: dict = {
            "excluded_triangle_count": 0,
            "included_triangle_count": triangle_count,
        }
        self.centroids = self.triangles.mean(axis=1)
        self.radii = np.linalg.norm(
            self.triangles - self.centroids[:, None, :],
            axis=2,
        ).max(axis=1)
        self.normals = np.cross(
            self.triangles[:, 1] - self.triangles[:, 0],
            self.triangles[:, 2] - self.triangles[:, 0],
        )
        self.normals /= np.maximum(
            np.linalg.norm(self.normals, axis=1)[:, None],
            1e-12,
        )
        self.maximum_radius = float(self.radii.max()) if len(self.radii) else 0.0
        self.radius_buckets: list[tuple[np.ndarray, cKDTree, float]] = []
        if len(self.centroids):
            radius_scale_keys = np.ceil(
                np.log2(np.maximum(self.radii, 1e-9))
            ).astype(np.int16)
            for key in np.unique(radius_scale_keys):
                triangle_ids = np.flatnonzero(
                    radius_scale_keys == key
                ).astype(np.int64)
                bucket_centroids = self.centroids[triangle_ids]
                bucket_maximum_radius = float(
                    self.radii[triangle_ids].max()
                )
                self.radius_buckets.append(
                    (
                        triangle_ids,
                        cKDTree(bucket_centroids),
                        bucket_maximum_radius,
                    )
                )

    def excluding_triangles(
        self,
        triangle_indices: np.ndarray,
        filter_context: dict | None = None,
    ) -> "ParentThicknessProbe":
        """Return a lightweight probe view that shares acceleration data."""
        excluded = np.unique(
            np.asarray(triangle_indices, dtype=np.int64)
        )
        if len(excluded) and (
            int(excluded.min()) < 0
            or int(excluded.max()) >= len(self.triangles)
        ):
            raise ValueError("excluded triangle index is outside the probe")
        active_mask = np.asarray(
            self.active_triangle_mask,
            dtype=bool,
        ).copy()
        active_mask[excluded] = False
        view = object.__new__(type(self))
        for attribute in (
            "triangles",
            "triangle_source_face_indices",
            "triangle_owner_indices",
            "centroids",
            "radii",
            "normals",
            "maximum_radius",
            "radius_buckets",
        ):
            setattr(view, attribute, getattr(self, attribute))
        view.active_triangle_mask = active_mask
        view.active_triangle_count = int(np.count_nonzero(active_mask))
        excluded_owner_indices = sorted(
            int(value)
            for value in np.unique(
                self.triangle_owner_indices[excluded]
            )
            if int(value) > 0
        )
        view.filter_context = {
            **dict(filter_context or {}),
            "excluded_triangle_count": int(
                len(active_mask) - view.active_triangle_count
            ),
            "included_triangle_count": int(view.active_triangle_count),
            "excluded_owner_indices": excluded_owner_indices,
        }
        return view

    def _owner_counts(self, triangle_ids: np.ndarray) -> dict[str, int]:
        triangle_ids = np.asarray(triangle_ids, dtype=np.int64)
        owner_indices = np.asarray(
            getattr(
                self,
                "triangle_owner_indices",
                np.empty(0, dtype=np.int32),
            ),
            dtype=np.int32,
        )
        triangle_ids = triangle_ids[
            (triangle_ids >= 0)
            & (triangle_ids < len(owner_indices))
        ]
        if not len(triangle_ids):
            return {}
        owners, counts = np.unique(
            owner_indices[triangle_ids],
            return_counts=True,
        )
        return {
            str(int(owner)): int(count)
            for owner, count in zip(owners, counts)
            if int(owner) > 0
        }

    def first_hit_distances(
        self,
        points: np.ndarray,
        directions: np.ndarray,
        search_limit_mm: float,
    ) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        directions = np.asarray(directions, dtype=np.float64)
        directions /= np.maximum(
            np.linalg.norm(directions, axis=1)[:, None],
            1e-12,
        )
        search_limit = max(float(search_limit_mm), 0.0)
        misses = search_limit + PARENT_THICKNESS_CLEARANCE_MM
        hits = np.full(len(points), misses, dtype=np.float64)
        self.last_hit_had_preceding_entry = np.zeros(
            len(points),
            dtype=bool,
        )
        if not self.radius_buckets or search_limit <= 0.0:
            return hits

        query_started_at = time.perf_counter()
        runtime_log(
            "厚度测量",
            "thickness_candidates_start",
            "开始查询父体厚度射线候选三角形",
            probe_vertex_count=int(len(points)),
            source_triangle_count=int(len(self.triangles)),
            search_limit_mm=round(float(search_limit), 6),
            maximum_triangle_radius_mm=round(float(self.maximum_radius), 6),
            radius_bucket_count=int(len(self.radius_buckets)),
        )
        midpoints = points + directions * (search_limit * 0.5)
        candidate_count = 0
        query_elapsed_seconds = 0.0
        intersection_elapsed_seconds = 0.0
        exit_hits: list[list[np.ndarray]] = [[] for _ in range(len(points))]
        entry_hits: list[list[np.ndarray]] = [[] for _ in range(len(points))]
        selected_exit_distances = np.full(len(points), np.nan, dtype=np.float64)
        selected_entry_distances = np.full(len(points), np.nan, dtype=np.float64)
        selected_exit_triangle_ids = np.full(len(points), -1, dtype=np.int64)
        selected_entry_triangle_ids = np.full(len(points), -1, dtype=np.int64)
        for (
            bucket_triangle_ids,
            bucket_tree,
            bucket_maximum_radius,
        ) in self.radius_buckets:
            bucket_query_started_at = time.perf_counter()
            candidate_groups = bucket_tree.query_ball_point(
                midpoints,
                search_limit * 0.5 + bucket_maximum_radius + 1e-8,
                workers=-1 if len(points) >= 64 else 1,
            )
            query_elapsed_seconds += float(
                time.perf_counter() - bucket_query_started_at
            )
            candidate_count += int(
                sum(len(group) for group in candidate_groups)
            )

            bucket_intersection_started_at = time.perf_counter()
            for index, (origin, direction, candidates) in enumerate(
                zip(points, directions, candidate_groups)
            ):
                if not candidates:
                    continue
                candidate_ids = bucket_triangle_ids[
                    np.asarray(candidates, dtype=np.int64)
                ]
                candidate_ids = candidate_ids[
                    self.active_triangle_mask[candidate_ids]
                ]
                if not len(candidate_ids):
                    continue

                centroid_delta = self.centroids[candidate_ids] - origin
                ray_projection = centroid_delta @ direction
                clamped_projection = np.clip(
                    ray_projection,
                    0.0,
                    search_limit,
                )
                closest = origin + clamped_projection[:, None] * direction
                sphere_distance = np.linalg.norm(
                    self.centroids[candidate_ids] - closest,
                    axis=1,
                )
                candidate_ids = candidate_ids[
                    sphere_distance <= self.radii[candidate_ids] + 1e-7
                ]
                if not len(candidate_ids):
                    continue

                triangles = self.triangles[candidate_ids]
                edge_1 = triangles[:, 1] - triangles[:, 0]
                edge_2 = triangles[:, 2] - triangles[:, 0]
                h = np.cross(
                    np.broadcast_to(direction, edge_2.shape),
                    edge_2,
                )
                determinant = np.einsum("ij,ij->i", edge_1, h)
                active = np.abs(determinant) > 1e-12
                inverse = np.zeros_like(determinant)
                inverse[active] = 1.0 / determinant[active]
                s = origin - triangles[:, 0]
                u = inverse * np.einsum("ij,ij->i", s, h)
                q = np.cross(s, edge_1)
                v = inverse * (q @ direction)
                distance = inverse * np.einsum("ij,ij->i", edge_2, q)
                geometric_hit = (
                    active
                    & (u >= -1e-9)
                    & (v >= -1e-9)
                    & (u + v <= 1.0 + 1e-9)
                    & (distance > 1e-4)
                    & (distance <= search_limit + 1e-9)
                )
                if not np.any(geometric_hit):
                    continue
                hit_distances = distance[geometric_hit]
                hit_triangle_ids = candidate_ids[geometric_hit]
                hit_facing = (
                    self.normals[candidate_ids][geometric_hit] @ direction
                )
                exiting_mask = hit_facing > 1e-6
                entering_mask = hit_facing < -1e-6
                exiting = hit_distances[exiting_mask]
                entering = hit_distances[entering_mask]
                if len(exiting):
                    exit_hits[index].append(
                        (
                            exiting,
                            hit_triangle_ids[exiting_mask],
                        )
                    )
                if len(entering):
                    entry_hits[index].append(
                        (
                            entering,
                            hit_triangle_ids[entering_mask],
                        )
                    )
            intersection_elapsed_seconds += float(
                time.perf_counter() - bucket_intersection_started_at
            )

        runtime_log(
            "厚度测量",
            "thickness_candidates_ready",
            "父体厚度射线候选三角形查询完成",
            probe_vertex_count=int(len(points)),
            source_triangle_count=int(len(self.triangles)),
            candidate_count=candidate_count,
            average_candidates_per_vertex=round(
                float(candidate_count / max(len(points), 1)),
                3,
            ),
            maximum_triangle_radius_mm=round(float(self.maximum_radius), 6),
            radius_bucket_count=int(len(self.radius_buckets)),
            query_elapsed_seconds=round(float(query_elapsed_seconds), 3),
        )
        for index, exiting_groups in enumerate(exit_hits):
            if not exiting_groups:
                continue
            exit_distance = math.inf
            exit_triangle_id = -1
            for group_distances, group_triangle_ids in exiting_groups:
                group_index = int(np.argmin(group_distances))
                candidate_distance = float(group_distances[group_index])
                if candidate_distance < exit_distance:
                    exit_distance = candidate_distance
                    exit_triangle_id = int(group_triangle_ids[group_index])
            selected_exit_distances[index] = exit_distance
            selected_exit_triangle_ids[index] = exit_triangle_id
            entry_distance = 0.0
            entry_triangle_id = -1
            for group_distances, group_triangle_ids in entry_hits[index]:
                eligible_indices = np.flatnonzero(
                    group_distances < exit_distance - 1e-5
                )
                if len(eligible_indices):
                    group_index = int(
                        eligible_indices[
                            np.argmax(group_distances[eligible_indices])
                        ]
                    )
                    candidate_distance = float(
                        group_distances[group_index]
                    )
                    if candidate_distance > entry_distance:
                        entry_distance = candidate_distance
                        entry_triangle_id = int(
                            group_triangle_ids[group_index]
                        )
            if entry_distance > 0.0:
                selected_entry_distances[index] = entry_distance
                selected_entry_triangle_ids[index] = entry_triangle_id
            hits[index] = exit_distance - entry_distance
        near_hit_mask = hits <= PARENT_THICKNESS_CLEARANCE_MM + 1e-9
        paired_hit_mask = np.isfinite(selected_entry_distances)
        self.last_hit_had_preceding_entry = paired_hit_mask.copy()
        self.last_selected_exit_triangle_ids = (
            selected_exit_triangle_ids.copy()
        )
        self.last_selected_entry_triangle_ids = (
            selected_entry_triangle_ids.copy()
        )
        self.last_hit_diagnostics = {
            "near_hit_vertices": int(np.count_nonzero(near_hit_mask)),
            "near_hit_with_preceding_entry_vertices": int(
                np.count_nonzero(near_hit_mask & paired_hit_mask)
            ),
            "near_hit_without_preceding_entry_vertices": int(
                np.count_nonzero(near_hit_mask & ~paired_hit_mask)
            ),
            "selected_exit_distance_min_mm": (
                float(np.nanmin(selected_exit_distances))
                if np.any(np.isfinite(selected_exit_distances))
                else None
            ),
            "selected_preceding_entry_distance_max_mm": (
                float(np.nanmax(selected_entry_distances))
                if np.any(np.isfinite(selected_entry_distances))
                else None
            ),
            "selected_exit_owner_counts": self._owner_counts(
                selected_exit_triangle_ids
            ),
            "selected_preceding_entry_owner_counts": self._owner_counts(
                selected_entry_triangle_ids
            ),
            "probe_filter": dict(
                getattr(self, "filter_context", {})
            ),
        }
        runtime_log(
            "厚度测量",
            "thickness_intersections_done",
            "父体厚度射线相交计算完成",
            probe_vertex_count=int(len(points)),
            candidate_count=candidate_count,
            measured_hit_count=int(np.count_nonzero(hits < misses)),
            intersection_elapsed_seconds=round(
                float(intersection_elapsed_seconds),
                3,
            ),
            total_elapsed_seconds=round(
                float(time.perf_counter() - query_started_at),
                3,
            ),
            **self.last_hit_diagnostics,
        )
        return hits

    def safety_limit(
        self,
        points: np.ndarray,
        directions: np.ndarray,
        global_ceiling_mm: float,
    ) -> tuple[float, dict]:
        global_ceiling = min(
            max(float(global_ceiling_mm), 0.0),
            MAXIMUM_SAFE_INWARD_DEPTH_MM,
        )
        search_limit = global_ceiling + PARENT_THICKNESS_CLEARANCE_MM
        thicknesses = self.first_hit_distances(
            points,
            directions,
            search_limit,
        )
        raw_minimum = float(thicknesses.min()) if len(thicknesses) else search_limit
        coincident = thicknesses <= PARENT_THICKNESS_CLEARANCE_MM + 1e-9
        coincident_count = int(np.count_nonzero(coincident))
        coincident_ratio = float(coincident_count / max(len(thicknesses), 1))
        preceding_entry_mask = getattr(
            self,
            "last_hit_had_preceding_entry",
            None,
        )
        has_entry_classification = bool(
            isinstance(preceding_entry_mask, np.ndarray)
            and preceding_entry_mask.shape == thicknesses.shape
        )
        unpaired_coincident = (
            coincident & ~preceding_entry_mask
            if has_entry_classification
            else np.zeros_like(coincident)
        )
        unpaired_coincident_count = int(
            np.count_nonzero(unpaired_coincident)
        )
        usable_mask = ~unpaired_coincident
        after_unpaired = thicknesses[usable_mask]
        after_unpaired_coincident = (
            after_unpaired
            <= PARENT_THICKNESS_CLEARANCE_MM + 1e-9
        )
        after_unpaired_coincident_count = int(
            np.count_nonzero(after_unpaired_coincident)
        )
        after_unpaired_coincident_ratio = float(
            after_unpaired_coincident_count
            / max(len(thicknesses), 1)
        )
        discard_isolated_coincident = bool(
            after_unpaired_coincident_count
            and after_unpaired_coincident_ratio < 0.01
            and np.any(~after_unpaired_coincident)
        )
        usable_thicknesses = (
            after_unpaired[~after_unpaired_coincident]
            if discard_isolated_coincident
            else after_unpaired
        )
        if discard_isolated_coincident:
            usable_mask &= ~coincident
        measured_minimum = (
            float(usable_thicknesses.min())
            if len(usable_thicknesses)
            else search_limit
        )
        thickness_quantiles = (
            np.quantile(usable_thicknesses, [0.0, 0.01, 0.05, 0.50, 0.95, 1.0])
            if len(usable_thicknesses)
            else np.full(6, search_limit, dtype=np.float64)
        )
        safe_maximum = min(
            global_ceiling,
            max(0.0, measured_minimum - PARENT_THICKNESS_CLEARANCE_MM),
        )
        measured_hits = thicknesses <= search_limit + 1e-9
        limiting_vertex_indices = np.flatnonzero(
            usable_mask
            & np.isclose(
                thicknesses,
                measured_minimum,
                rtol=0.0,
                atol=1e-9,
            )
        )
        selected_exit_triangle_ids = getattr(
            self,
            "last_selected_exit_triangle_ids",
            np.full(len(thicknesses), -1, dtype=np.int64),
        )
        selected_entry_triangle_ids = getattr(
            self,
            "last_selected_entry_triangle_ids",
            np.full(len(thicknesses), -1, dtype=np.int64),
        )
        limiting_exit_triangle_ids = selected_exit_triangle_ids[
            limiting_vertex_indices
        ]
        limiting_entry_triangle_ids = selected_entry_triangle_ids[
            limiting_vertex_indices
        ]
        triangle_source_face_indices = np.asarray(
            getattr(
                self,
                "triangle_source_face_indices",
                np.empty(0, dtype=np.int64),
            ),
            dtype=np.int64,
        )
        valid_limiting_exit_triangle_ids = limiting_exit_triangle_ids[
            (limiting_exit_triangle_ids >= 0)
            & (
                limiting_exit_triangle_ids
                < len(triangle_source_face_indices)
            )
        ]
        valid_limiting_entry_triangle_ids = limiting_entry_triangle_ids[
            (limiting_entry_triangle_ids >= 0)
            & (
                limiting_entry_triangle_ids
                < len(triangle_source_face_indices)
            )
        ]
        limiting_exit_source_faces_all = sorted(
            {
            int(value)
            for value in triangle_source_face_indices[
                valid_limiting_exit_triangle_ids
            ]
            }
        )
        limiting_entry_source_faces_all = sorted(
            {
            int(value)
            for value in triangle_source_face_indices[
                valid_limiting_entry_triangle_ids
            ]
            }
        )
        limiting_vertex_sample = [
            int(value) for value in limiting_vertex_indices[:16]
        ]
        limiting_exit_source_faces = limiting_exit_source_faces_all[:16]
        limiting_entry_source_faces = limiting_entry_source_faces_all[:16]
        result = {
            "parent_thickness_min_mm": measured_minimum,
            "parent_thickness_raw_min_mm": raw_minimum,
            "parent_thickness_coincident_hit_vertices": coincident_count,
            "parent_thickness_coincident_hit_ratio": coincident_ratio,
            "parent_thickness_unpaired_surface_hit_vertices_discarded": (
                unpaired_coincident_count
            ),
            "parent_thickness_remaining_coincident_hit_vertices": (
                after_unpaired_coincident_count
            ),
            "parent_thickness_remaining_coincident_hit_ratio": (
                after_unpaired_coincident_ratio
            ),
            "parent_thickness_isolated_coincident_hits_discarded": discard_isolated_coincident,
            "parent_thickness_hit_vertices": int(np.count_nonzero(measured_hits)),
            "parent_thickness_probe_vertices": int(len(points)),
            "parent_thickness_is_lower_bound": bool(not np.all(measured_hits)),
            "parent_thickness_quantiles_mm": {
                key: float(value)
                for key, value in zip(
                    ("minimum", "p01", "p05", "median", "p95", "maximum"),
                    thickness_quantiles,
                )
            },
            "parent_thickness_clearance_mm": PARENT_THICKNESS_CLEARANCE_MM,
            "safe_maximum_inward_depth_mm": safe_maximum,
            "parent_thickness_limiting_probe_vertex_indices": [
                int(value) for value in limiting_vertex_sample
            ],
            "parent_thickness_limiting_probe_vertex_count": int(
                len(limiting_vertex_indices)
            ),
            "parent_thickness_limiting_exit_face_indices": (
                limiting_exit_source_faces
            ),
            "parent_thickness_limiting_exit_face_count": int(
                len(limiting_exit_source_faces_all)
            ),
            "parent_thickness_limiting_entry_face_indices": (
                limiting_entry_source_faces
            ),
            "parent_thickness_limiting_entry_face_count": int(
                len(limiting_entry_source_faces_all)
            ),
            "parent_thickness_limiting_exit_owner_counts": (
                self._owner_counts(limiting_exit_triangle_ids)
            ),
            "parent_thickness_limiting_entry_owner_counts": (
                self._owner_counts(limiting_entry_triangle_ids)
            ),
            "parent_thickness_probe_filter": dict(
                getattr(self, "filter_context", {})
            ),
            "parent_thickness_hit_diagnostics": dict(
                getattr(self, "last_hit_diagnostics", {})
            ),
        }
        runtime_log(
            "厚度测量",
            "parent_thickness_limit_selected",
            "父体厚度安全上限及限制命中面已确定",
            measured_parent_thickness_mm=float(measured_minimum),
            safe_maximum_inward_depth_mm=float(safe_maximum),
            limiting_probe_vertex_indices=[
                int(value) for value in limiting_vertex_sample
            ],
            limiting_probe_vertex_count=int(len(limiting_vertex_indices)),
            limiting_exit_face_indices=limiting_exit_source_faces,
            limiting_exit_face_count=int(
                len(limiting_exit_source_faces_all)
            ),
            limiting_entry_face_indices=limiting_entry_source_faces,
            limiting_entry_face_count=int(
                len(limiting_entry_source_faces_all)
            ),
            limiting_exit_owner_counts=self._owner_counts(
                limiting_exit_triangle_ids
            ),
            limiting_entry_owner_counts=self._owner_counts(
                limiting_entry_triangle_ids
            ),
            probe_filter=dict(
                getattr(self, "filter_context", {})
            ),
        )
        return safe_maximum, result


def planar_cap_distances(
    points: np.ndarray,
    inward: np.ndarray,
    fixed_depth_mm: float,
    flat_clearance_mm: float,
    cap_mode: str,
    planar_extra_limit_mm: float,
) -> tuple[np.ndarray, dict]:
    s_values = points @ inward
    s_min = float(s_values.min())
    s_max = float(s_values.max())
    clearance = max(float(flat_clearance_mm), 0.0)
    minimum_flat_depth_mm = MINIMUM_INWARD_DEPTH_MM
    fixed_depth = max(float(fixed_depth_mm), minimum_flat_depth_mm)
    extra_limit = max(float(planar_extra_limit_mm), 0.0)
    flat_plane_s = s_max + fixed_depth
    flat_distances = flat_plane_s - s_values
    flat_extra = float(flat_distances.max() - fixed_depth) if len(flat_distances) else 0.0

    tilted_record: dict | None = None
    tilted_distances: np.ndarray | None = None
    tilted_normal = fit_plane_normal(points, inward)
    tilted_dot = float(np.dot(tilted_normal, inward))
    if tilted_dot > 0.20:
        tilted_values = points @ tilted_normal
        tilted_plane_s = float(tilted_values.max() + fixed_depth * tilted_dot)
        tilted_distances = (tilted_plane_s - tilted_values) / tilted_dot
        tilted_record = {
            "tilted_plane_normal": tilted_normal.round(6).tolist(),
            "tilted_plane_dot_inward": tilted_dot,
            "tilted_plane_span_mm": float(tilted_values.max() - tilted_values.min()),
            "tilted_plane_s": tilted_plane_s,
            "tilted_plane_extension_max_mm": float(tilted_distances.max()),
        }

    selected_mode = "flat"
    selected_distances = flat_distances
    plane_s: float | None = flat_plane_s
    cap_depth_reference = "max_boundary_dot_inward"

    if cap_mode == "offset":
        selected_mode = "offset"
        selected_distances = np.full_like(s_values, fixed_depth, dtype=np.float64)
        plane_s = None
        cap_depth_reference = "per_boundary_vertex_offset"
    elif cap_mode == "tilted":
        if tilted_distances is not None:
            selected_mode = "tilted"
            selected_distances = tilted_distances
            plane_s = float(tilted_record["tilted_plane_s"]) if tilted_record else None
            cap_depth_reference = "tilted_best_fit_plane"
    elif cap_mode == "adaptive":
        if flat_extra <= extra_limit:
            selected_mode = "flat"
        elif tilted_distances is not None and float(tilted_distances.max() - fixed_depth) <= extra_limit:
            selected_mode = "tilted"
            selected_distances = tilted_distances
            plane_s = float(tilted_record["tilted_plane_s"]) if tilted_record else None
            cap_depth_reference = "tilted_best_fit_plane"
        else:
            selected_mode = "offset"
            selected_distances = np.full_like(s_values, fixed_depth, dtype=np.float64)
            plane_s = None
            cap_depth_reference = "per_boundary_vertex_offset"

    record = {
        "requested_cap_mode": cap_mode,
        "cap_mode": selected_mode,
        "target_extension_mm": fixed_depth,
        "fixed_inward_depth_mm": fixed_depth,
        "minimum_flat_bottom_depth_mm": minimum_flat_depth_mm,
        "flat_clearance_mm": clearance,
        "flat_span_mm": float(s_max - s_min),
        "flat_bottom_required_max_extension_mm": float(selected_distances.max()) if len(selected_distances) else fixed_depth,
        "flat_bottom_plane_s": None if plane_s is None else float(plane_s),
        "flat_plane_extension_max_mm": float(flat_distances.max()) if len(flat_distances) else fixed_depth,
        "planar_extra_limit_mm": extra_limit,
        "target_exceeded": bool(
            len(selected_distances)
            and float(selected_distances.max()) > fixed_depth + 1e-9
        ),
        "target_exceeded_by_mm": float(
            max(0.0, float(selected_distances.max()) - fixed_depth)
            if len(selected_distances)
            else 0.0
        ),
        "fixed_inward_depth_applied": True,
        "cap_depth_reference": cap_depth_reference,
    }
    if tilted_record is not None:
        record.update(tilted_record)
    return selected_distances, record


def boundary_cap_distances(
    points: np.ndarray,
    fallback_inward: np.ndarray,
    inward_directions: np.ndarray,
    fixed_depth_mm: float,
    flat_clearance_mm: float,
    cap_mode: str,
    planar_extra_limit_mm: float,
    parent_thickness_probe: ParentThicknessProbe | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Use a common bottom plane when it fits; otherwise use safe local offset."""
    points = np.asarray(points, dtype=np.float64)
    fallback = np.asarray(fallback_inward, dtype=np.float64)
    fallback /= max(float(np.linalg.norm(fallback)), 1e-12)
    directions = np.asarray(inward_directions, dtype=np.float64)
    directions /= np.maximum(np.linalg.norm(directions, axis=1)[:, None], 1e-12)
    preferred_minimum = max(
        float(fixed_depth_mm),
        DEFAULT_EFFECTIVE_MINIMUM_INWARD_DEPTH_MM,
    )
    global_ceiling = min(
        preferred_minimum + max(float(planar_extra_limit_mm), 0.0),
        MAXIMUM_SAFE_INWARD_DEPTH_MM,
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
            }
        return parent_thickness_probe.safety_limit(
            points,
            candidate_directions,
            global_ceiling,
        )

    candidate_specs: list[tuple[str, np.ndarray]] = [("global_inward", fallback)]
    best_fit_normal = fit_plane_normal(points, fallback)
    best_fit_dot_global = float(np.dot(best_fit_normal, fallback))
    if best_fit_dot_global >= 0.15:
        candidate_specs.append(("loop_best_fit_normal", best_fit_normal))

    plane_candidates = []
    for orientation, plane_direction in candidate_specs:
        plane_directions = np.tile(plane_direction, (len(points), 1))
        safe_maximum, thickness_record = measured_limit(plane_directions)
        if safe_maximum <= 1e-6:
            effective_minimum = 0.0
        else:
            effective_minimum = min(preferred_minimum, safe_maximum)
        values = points @ plane_direction
        plane_s = float(values.max() + effective_minimum)
        distances = plane_s - values
        required_maximum = (
            float(distances.max()) if len(distances) else effective_minimum
        )
        feasible = (
            effective_minimum > 0.0
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
                "effective_minimum_mm": effective_minimum,
                "safe_maximum_mm": safe_maximum,
                "feasible": feasible,
                "thickness_record": thickness_record,
            }
        )

    if cap_mode == "tilted":
        selectable = [
            candidate
            for candidate in plane_candidates
            if candidate["orientation"] == "loop_best_fit_normal"
            and candidate["feasible"]
        ]
    elif cap_mode == "flat":
        selectable = [
            candidate
            for candidate in plane_candidates
            if candidate["orientation"] == "global_inward"
            and candidate["feasible"]
        ]
    elif cap_mode == "offset":
        selectable = []
    else:
        selectable = [candidate for candidate in plane_candidates if candidate["feasible"]]

    if selectable:
        selected = min(
            selectable,
            key=lambda candidate: (
                candidate["required_maximum_mm"],
                0 if candidate["orientation"] == "global_inward" else 1,
            ),
        )
        distances = np.asarray(selected["distances"], dtype=np.float64)
        record = {
            "requested_cap_mode": cap_mode,
            "cap_mode": "flat",
            "flat_orientation": selected["orientation"],
            "cap_depth_reference": "common_plane_with_variable_point_depth",
            "preferred_minimum_inward_depth_mm": preferred_minimum,
            "effective_minimum_inward_depth_mm": selected["effective_minimum_mm"],
            "safe_maximum_inward_depth_mm": selected["safe_maximum_mm"],
            "flat_span_mm": selected["span_mm"],
            "flat_bottom_required_max_extension_mm": selected["required_maximum_mm"],
            "flat_bottom_plane_s": selected["plane_s"],
            "flat_plane_extension_max_mm": selected["required_maximum_mm"],
            "minimum_generated_inward_travel_mm": float(distances.min()),
            "maximum_generated_inward_travel_mm": float(distances.max()),
            "flat_priority_applied": True,
            "flat_feasibility_basis": "required_plane_depth_within_parent_thickness_safety_limit",
            "flat_travel_within_limit": True,
            "target_exceeded": bool(
                selected["required_maximum_mm"] > preferred_minimum + 1e-9
            ),
            "target_exceeded_by_mm": float(
                max(0.0, selected["required_maximum_mm"] - preferred_minimum)
            ),
            "fixed_inward_depth_mm": preferred_minimum,
            "fixed_inward_depth_applied": bool(
                abs(selected["effective_minimum_mm"] - preferred_minimum) <= 1e-9
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
            **selected["thickness_record"],
        }
        return distances, np.asarray(selected["directions"]), record

    if cap_mode in {"flat", "tilted"}:
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
        raise ValueError(
            f"Forced {cap_mode} cap cannot fit inside the parent-thickness safety limit: "
            f"required={attempted['required_maximum_mm']:.6f}mm, "
            f"safe_maximum={attempted['safe_maximum_mm']:.6f}mm"
        )

    local_safe_maximum, local_thickness_record = measured_limit(directions)
    if local_safe_maximum <= 1e-6:
        raise ValueError(
            "No positive inward depth remains after the 0.05 mm parent-thickness clearance: "
            + json.dumps(local_thickness_record, ensure_ascii=False, sort_keys=True)
        )
    distances = np.full(len(points), local_safe_maximum, dtype=np.float64)
    required_plane_maximum = min(
        (
            float(candidate["required_maximum_mm"])
            for candidate in plane_candidates
        ),
        default=preferred_minimum,
    )
    direction_dot = np.clip(directions @ fallback, -1.0, 1.0)
    record = {
        "requested_cap_mode": cap_mode,
        "cap_mode": "local-offset",
        "cap_depth_reference": "per_boundary_vertex_safe_local_inward_direction",
        "preferred_minimum_inward_depth_mm": preferred_minimum,
        "effective_minimum_inward_depth_mm": min(
            preferred_minimum,
            local_safe_maximum,
        ),
        "safe_maximum_inward_depth_mm": local_safe_maximum,
        "flat_span_mm": float(
            min(
                (candidate["span_mm"] for candidate in plane_candidates),
                default=0.0,
            )
        ),
        "flat_bottom_required_max_extension_mm": required_plane_maximum,
        "flat_bottom_plane_s": None,
        "flat_plane_extension_max_mm": required_plane_maximum,
        "minimum_generated_inward_travel_mm": local_safe_maximum,
        "maximum_generated_inward_travel_mm": local_safe_maximum,
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
            abs(local_safe_maximum - preferred_minimum) <= 1e-9
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
        **local_thickness_record,
    }
    return distances, directions, record


def plan_cap_decision(
    points: np.ndarray,
    source_vertex_ids: list[int] | tuple[int, ...],
    fallback_inward: np.ndarray,
    inward_directions: np.ndarray,
    fixed_depth_mm: float,
    flat_clearance_mm: float,
    cap_mode: str,
    planar_extra_limit_mm: float,
    parent_thickness_probe: ParentThicknessProbe | None = None,
) -> CapDecision:
    """Plan one reusable child cap decision keyed by source vertex id."""
    vertex_ids = tuple(int(value) for value in source_vertex_ids)
    if len(vertex_ids) != len(set(vertex_ids)):
        raise ValueError("cap decision source vertex ids must be unique")
    if len(vertex_ids) != len(points):
        raise ValueError("cap decision vertex-id count does not match boundary points")
    distances, directions, record = boundary_cap_distances(
        points,
        fallback_inward,
        inward_directions,
        fixed_depth_mm,
        flat_clearance_mm,
        cap_mode,
        planar_extra_limit_mm,
        parent_thickness_probe,
    )
    return CapDecision(
        mode=str(record["cap_mode"]),
        source_vertex_ids=vertex_ids,
        fit_points=np.asarray(points, dtype=np.float64).copy(),
        directions=np.asarray(directions, dtype=np.float64).copy(),
        distances=np.asarray(distances, dtype=np.float64).copy(),
        record=dict(record),
    )


def remap_cap_decision(
    decision: CapDecision,
    source_vertex_ids: list[int] | tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return a shared cap decision in the caller's boundary order."""
    requested_ids = tuple(int(value) for value in source_vertex_ids)
    if len(requested_ids) != len(set(requested_ids)):
        raise ValueError("matching cap boundary contains duplicate source vertex ids")
    requested_set = set(requested_ids)
    source_index = {
        int(global_id): index
        for index, global_id in enumerate(decision.source_vertex_ids)
    }
    missing = [global_id for global_id in requested_ids if global_id not in source_index]
    extra = [
        global_id
        for global_id in decision.source_vertex_ids
        if global_id not in requested_set
    ]
    if missing or extra:
        raise ValueError(
            "matching parent socket and child cap source vertices differ: "
            f"missing={missing}, extra={extra}"
        )
    order = np.asarray(
        [source_index[global_id] for global_id in requested_ids],
        dtype=np.int64,
    )
    fit_points = np.asarray(decision.fit_points, dtype=np.float64)[order].copy()
    directions = np.asarray(decision.directions, dtype=np.float64)[order].copy()
    distances = np.asarray(decision.distances, dtype=np.float64)[order].copy()
    record = dict(decision.record)
    if str(record.get("cap_mode")) != str(decision.mode):
        raise ValueError("cap decision mode and record disagree")
    record["cap_decision_source"] = "prevalidated_shared_child"
    return fit_points, distances, directions, record


def add_side_faces_between_rings(output_faces: list[list[int]], first_ids: list[int], second_ids: list[int]) -> int:
    if len(first_ids) != len(second_ids):
        raise ValueError("Side rings must have the same vertex count")
    side_faces = 0
    count = len(first_ids)
    for index in range(count):
        a0 = int(first_ids[index])
        a1 = int(first_ids[(index + 1) % count])
        b0 = int(second_ids[index])
        b1 = int(second_ids[(index + 1) % count])
        output_faces.append([a0, a1, b1])
        output_faces.append([a0, b1, b0])
        side_faces += 2
    return side_faces


def add_loop_extrusion_and_cap(
    output_vertices: list[np.ndarray],
    output_faces: list[list[int]],
    top_ids: list[int],
    top_points: np.ndarray,
    inward: np.ndarray,
    origin: np.ndarray,
    max_extension_mm: float,
    flat_clearance_mm: float,
    cap_mode: str,
    planar_extra_limit_mm: float,
    inward_directions: np.ndarray | None = None,
    bottom_points_override: np.ndarray | None = None,
    plane_record_override: dict | None = None,
    parent_thickness_probe: ParentThicknessProbe | None = None,
) -> tuple[int, int, dict]:
    u, v = orthonormal_basis(inward)
    if inward_directions is None:
        inward_directions = np.tile(np.asarray(inward, dtype=np.float64), (len(top_points), 1))
    inward_directions = np.asarray(inward_directions, dtype=np.float64)
    inward_directions /= np.maximum(np.linalg.norm(inward_directions, axis=1)[:, None], 1e-12)
    if bottom_points_override is None:
        distances, inward_directions, plane_record = boundary_cap_distances(
            top_points,
            inward,
            inward_directions,
            max_extension_mm,
            flat_clearance_mm,
            cap_mode,
            planar_extra_limit_mm,
            parent_thickness_probe,
        )
        bottom_points = top_points + inward_directions * distances[:, None]
    else:
        bottom_points = np.asarray(bottom_points_override, dtype=np.float64)
        if bottom_points.shape != top_points.shape:
            raise ValueError("Socket bottom override must match the top boundary shape")
        displacement = bottom_points - top_points
        distances = np.linalg.norm(displacement, axis=1)
        inward_directions = displacement / np.maximum(distances[:, None], 1e-12)
        plane_record = dict(plane_record_override or {})

    bottom_ids = []
    for bottom_point in bottom_points:
        bottom_ids.append(len(output_vertices))
        output_vertices.append(bottom_point)

    side_faces = add_side_faces_between_rings(output_faces, top_ids, bottom_ids)

    bottom_points_2d = [project_points(np.array([output_vertices[i] for i in bottom_ids]), origin, u, v)]
    cap_faces = triangulate_cap(output_vertices, output_faces, [bottom_ids], bottom_points_2d, [[0]], inward)
    extension_record = {
        "vertices": len(top_ids),
        "extension_min_mm": float(distances.min()),
        "extension_max_mm": float(distances.max()),
        **plane_record,
    }
    return side_faces, cap_faces, extension_record


def matched_socket_bottom_geometry(
    source_boundary_points: np.ndarray,
    socket_top_points: np.ndarray,
    inward: np.ndarray,
    inward_directions: np.ndarray,
    child_insert_shrink_mm: float,
    max_extension_mm: float,
    flat_clearance_mm: float,
    cap_mode: str,
    planar_extra_limit_mm: float,
    bottom_clearance_mm: float,
    parent_thickness_probe: ParentThicknessProbe | None = None,
    cap_decision: CapDecision | None = None,
    source_vertex_ids: list[int] | tuple[int, ...] | None = None,
) -> tuple[np.ndarray, dict]:
    """Reuse the child's actual cap field, then place the socket bottom behind it."""
    inward = np.asarray(inward, dtype=np.float64)
    inward /= max(float(np.linalg.norm(inward)), 1e-12)
    u, v = orthonormal_basis(inward)
    source_boundary_points = np.asarray(source_boundary_points, dtype=np.float64)
    socket_top_points = np.asarray(socket_top_points, dtype=np.float64)
    child_fit_points = radial_offset_points(
        source_boundary_points,
        source_boundary_points.mean(axis=0),
        u,
        v,
        inward,
        -max(float(child_insert_shrink_mm), 0.0),
    )
    if cap_decision is not None:
        if source_vertex_ids is None:
            raise ValueError("a shared cap decision requires source vertex ids")
        (
            child_fit_points,
            child_distances,
            child_directions,
            child_record,
        ) = remap_cap_decision(cap_decision, source_vertex_ids)
    else:
        child_distances, child_directions, child_record = boundary_cap_distances(
            child_fit_points,
            inward,
            inward_directions,
            max_extension_mm,
            flat_clearance_mm,
            cap_mode,
            max(
                float(planar_extra_limit_mm)
                - max(float(bottom_clearance_mm), 0.0),
                0.0,
            ),
            parent_thickness_probe,
        )
        child_distances, child_record = reserve_flat_socket_travel_budget(
            child_fit_points=child_fit_points,
            child_distances=child_distances,
            child_directions=child_directions,
            socket_top_points=socket_top_points,
            bottom_clearance_mm=bottom_clearance_mm,
            maximum_socket_travel_mm=max_extension_mm
            + max(float(planar_extra_limit_mm), 0.0),
            plane_record=child_record,
        )
    child_bottom_points = child_fit_points + child_directions * child_distances[:, None]
    clearance = max(float(bottom_clearance_mm), 0.0)
    if str(child_record.get("cap_mode")) == "flat":
        plane_normal = np.asarray(child_directions[0], dtype=np.float64)
        plane_normal /= max(float(np.linalg.norm(plane_normal)), 1e-12)
        child_plane_s = float(np.mean(child_bottom_points @ plane_normal))
        socket_distances = (
            child_plane_s
            + clearance
            - socket_top_points @ plane_normal
        )
        socket_bottom_points = socket_top_points + socket_distances[:, None] * plane_normal
    else:
        socket_bottom_points = socket_top_points + child_directions * (
            child_distances + clearance
        )[:, None]
    record = dict(child_record)
    record.update(
        {
            "socket_cap_field_source": (
                "prevalidated_shared_child_cap"
                if cap_decision is not None
                else "matched_child_cap"
            ),
            "socket_bottom_clearance_applied_mm": clearance,
            "child_insert_shrink_mm": max(float(child_insert_shrink_mm), 0.0),
        }
    )
    return socket_bottom_points, record


def reserve_flat_socket_travel_budget(
    child_fit_points: np.ndarray,
    child_distances: np.ndarray,
    child_directions: np.ndarray,
    socket_top_points: np.ndarray,
    bottom_clearance_mm: float,
    maximum_socket_travel_mm: float,
    plane_record: dict,
) -> tuple[np.ndarray, dict]:
    """Shift a child plane outward so its matching socket, including clearance, stays in budget."""
    record = dict(plane_record)
    if str(record.get("cap_mode")) != "flat" or not len(child_distances):
        return child_distances, record
    plane_normal = np.asarray(child_directions[0], dtype=np.float64)
    plane_normal /= max(float(np.linalg.norm(plane_normal)), 1e-12)
    child_bottom_points = (
        np.asarray(child_fit_points, dtype=np.float64)
        + np.asarray(child_directions, dtype=np.float64)
        * np.asarray(child_distances, dtype=np.float64)[:, None]
    )
    child_plane_s = float(np.mean(child_bottom_points @ plane_normal))
    required_socket_distances = (
        child_plane_s
        + max(float(bottom_clearance_mm), 0.0)
        - np.asarray(socket_top_points, dtype=np.float64) @ plane_normal
    )
    socket_budget_shift = max(
        0.0,
        float(required_socket_distances.max()) - float(maximum_socket_travel_mm),
    )
    shifted_distances = np.asarray(child_distances, dtype=np.float64) - socket_budget_shift
    if float(shifted_distances.min()) < MINIMUM_INWARD_DEPTH_MM - 1e-9:
        record.update(
            {
                "socket_budget_shift_mm": socket_budget_shift,
                "socket_budget_shift_applied": False,
                "socket_budget_shift_reason": "minimum_inward_depth_would_be_violated",
            }
        )
        return child_distances, record
    record.update(
        {
            "socket_budget_shift_mm": socket_budget_shift,
            "socket_budget_shift_applied": bool(socket_budget_shift > 0.0),
            "socket_budget_shift_reason": "matching_socket_total_travel_limit",
            "maximum_matching_socket_travel_mm": float(maximum_socket_travel_mm),
            "maximum_generated_inward_travel_mm": float(shifted_distances.max()),
            "minimum_generated_inward_travel_mm": float(shifted_distances.min()),
        }
    )
    return shifted_distances, record


def make_part_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    component: Component,
    component_index: int,
    assembly_parent_index: int | None,
    part_id: str,
    max_extension_mm: float,
    boundary_fairing: BoundaryFairingContext,
    flat_clearance_mm: float,
    fit_clearance_mm: float,
    insert_shrink_mm: float,
    lead_in_mm: float,
    top_edge_clearance_mm: float,
    clearance_mode: str,
    child_cut_refs: list[dict] | None,
    boundary_neighbor_lookup: dict[tuple[int, int], set[int]],
    component_centers: dict[int, np.ndarray],
    sibling_clearance_mm: float,
    socket_overcut_mm: float,
    bottom_clearance_mm: float,
    inward_override: np.ndarray | None,
    model_center: np.ndarray,
    cap_mode: str,
    planar_extra_limit_mm: float,
    parent_contact_only: bool = False,
) -> tuple[trimesh.Trimesh, dict]:
    parent_thickness_probe = ParentThicknessProbe(vertices, faces)
    local_vertices, local_faces, _, global_vertex_ids = build_local_mesh(vertices, faces, component)
    loops = boundary_loops(local_faces)
    boundary_vertex_count = int(sum(len(loop) for loop in loops))

    component_center = local_vertices[local_faces.reshape(-1)].mean(axis=0)
    outward = average_outward_normal(local_vertices, local_faces, component_center, model_center)
    inward = inward_override if inward_override is not None else -outward
    inward = inward / max(float(np.linalg.norm(inward)), 1e-12)
    vertex_inward_normals = mesh_vertex_inward_normals(local_vertices, local_faces)
    loop_inward_directions: dict[int, np.ndarray] = {}
    local_inward_direction_records: list[dict] = []
    for loop_index, loop in enumerate(loops):
        directions, direction_record = safe_boundary_inward_directions(
            vertex_inward_normals,
            loop,
            inward,
            loop_points=local_vertices[np.asarray(loop, dtype=np.int64)],
        )
        loop_inward_directions[int(loop_index)] = directions
        local_inward_direction_records.append({"loop_index": int(loop_index), **direction_record})

    local_vertices, boundary_fairing_records = BoundaryFairingService.fair_local_loops(
        local_vertices,
        loops,
        global_vertex_ids,
        boundary_fairing,
    )

    component_center = local_vertices[local_faces.reshape(-1)].mean(axis=0)
    u, v = orthonormal_basis(inward)
    insert_shrink_mm = max(float(insert_shrink_mm), 0.0)
    lead_in_mm = max(float(lead_in_mm), 0.0)
    top_edge_clearance_mm = min(max(float(top_edge_clearance_mm), 0.0), insert_shrink_mm)
    socket_overcut_mm = max(float(socket_overcut_mm), 0.0)
    parent_socket_overcut_mm = clearance_offsets(clearance_mode, fit_clearance_mm)[1]
    bottom_clearance_mm = max(float(bottom_clearance_mm), 0.0)
    insert_planar_extra_limit_mm = max(
        float(planar_extra_limit_mm) - bottom_clearance_mm,
        0.0,
    )
    child_cut_refs = child_cut_refs or []
    child_component_indices = {int(ref["component_index"]) for ref in child_cut_refs}
    socket_flat_clearance_mm = max(float(flat_clearance_mm), 0.0)
    sibling_clearance_mm = max(float(sibling_clearance_mm), 0.0)

    loop_fit_points: list[np.ndarray] = []
    loop_clearance_records = []
    insert_loop_records: list[dict] = []
    sibling_clearance_records: list[dict] = []
    socket_extension_records: list[dict] = []
    skipped_socket_records: list[dict] = []
    skipped_non_parent_loop_records: list[dict] = []
    dropped_non_parent_face_indices: set[int] = set()
    socket_side_faces = 0
    socket_cap_faces = 0

    output_vertices: list[np.ndarray] = [p.copy() for p in local_vertices]
    output_faces: list[list[int]] = local_faces.astype(int).tolist()

    def sibling_cleared_points(loop: list[int], points: np.ndarray) -> tuple[np.ndarray, dict]:
        if sibling_clearance_mm <= 1e-9:
            return points, {"sibling_edges": 0, "sibling_vertices": 0, "sibling_neighbor_indices": []}
        offsets = np.zeros_like(points)
        counts = np.zeros(len(points), dtype=np.int64)
        sibling_neighbors: set[int] = set()
        loop_position = {int(local_index): position for position, local_index in enumerate(loop)}
        for position, local_a in enumerate(loop):
            local_b = int(loop[(position + 1) % len(loop)])
            global_edge = tuple(sorted((int(global_vertex_ids[local_a]), int(global_vertex_ids[local_b]))))
            side_neighbors = [
                int(neighbor)
                for neighbor in boundary_neighbor_lookup.get(global_edge, set())
                if int(neighbor) != component_index
                and int(neighbor) != int(assembly_parent_index or -1)
                and int(neighbor) not in child_component_indices
            ]
            if not side_neighbors:
                continue
            for neighbor_index in side_neighbors:
                neighbor_center = component_centers.get(neighbor_index)
                if neighbor_center is None:
                    continue
                sibling_neighbors.add(neighbor_index)
                for local_index in (int(local_a), local_b):
                    point_position = loop_position[int(local_index)]
                    direction = points[point_position] - neighbor_center
                    direction = direction - inward * float(np.dot(direction, inward))
                    length = float(np.linalg.norm(direction))
                    if length <= 1e-9:
                        continue
                    offsets[point_position] += direction / length * sibling_clearance_mm
                    counts[point_position] += 1
        shifted = points.copy()
        active = counts > 0
        if np.any(active):
            shifted[active] = shifted[active] + offsets[active] / counts[active, None]
        return shifted, {
            "sibling_edges": int(np.sum(counts) // 2),
            "sibling_vertices": int(np.sum(active)),
            "sibling_neighbor_indices": sorted(sibling_neighbors),
        }

    for loop_index, loop in enumerate(loops):
        loop_array = np.array(loop, dtype=np.int64)
        original_points = local_vertices[loop_array]
        original_points, sibling_record = sibling_cleared_points(loop, original_points)
        loop_global_vertices = set(int(global_vertex_ids[i]) for i in loop)
        loop_global_ordered = [int(global_vertex_ids[i]) for i in loop]
        loop_global_edges = {
            tuple(sorted((loop_global_ordered[position], loop_global_ordered[(position + 1) % len(loop_global_ordered)])))
            for position in range(len(loop_global_ordered))
        }
        socket_ref = best_cut_reference(loop_global_vertices, loop_global_edges, child_cut_refs)
        loop_parent_edges = 0
        loop_neighbor_counts: collections.Counter[int] = collections.Counter()
        if assembly_parent_index is not None:
            for position, local_a in enumerate(loop):
                local_b = loop[(position + 1) % len(loop)]
                global_edge = tuple(sorted((int(global_vertex_ids[local_a]), int(global_vertex_ids[local_b]))))
                neighbors = boundary_neighbor_lookup.get(global_edge, set())
                for neighbor_index in neighbors:
                    if int(neighbor_index) != component_index:
                        loop_neighbor_counts[int(neighbor_index)] += 1
                if int(assembly_parent_index) in neighbors:
                    loop_parent_edges += 1
        shared_loop_parent_edges = loop_parent_edges
        shared_loop_child_edges = 0
        shared_parent_child_loop = False
        if socket_ref is not None and assembly_parent_index is not None:
            child_component_index = int(socket_ref["component_index"])
            for position, local_a in enumerate(loop):
                local_b = loop[(position + 1) % len(loop)]
                global_edge = tuple(sorted((int(global_vertex_ids[local_a]), int(global_vertex_ids[local_b]))))
                neighbors = boundary_neighbor_lookup.get(global_edge, set())
                if child_component_index in neighbors:
                    shared_loop_child_edges += 1
            shared_parent_child_loop = shared_loop_parent_edges >= 3 and shared_loop_child_edges >= 3
        if socket_ref is not None:
            socket_inward = socket_ref["inward"]
            socket_cap_mode = str(socket_ref.get("cap_mode", cap_mode))
            socket_planar_extra_limit_mm = float(
                socket_ref.get("planar_extra_limit_mm")
                if socket_ref.get("planar_extra_limit_mm") is not None
                else planar_extra_limit_mm
            )
            socket_processing_mode = str(socket_ref.get("processing_mode", "inward"))
            if shared_parent_child_loop:
                skipped_socket_records.append(
                    {
                        "loop_index": loop_index,
                        "matched_component_index": socket_ref["component_index"],
                        "matched_cap_mode": socket_cap_mode,
                        "matched_processing_mode": socket_processing_mode,
                        "matched_color_code": socket_ref["color_code"],
                        "matched_color_name": socket_ref["color_name"],
                        "skip_reason": "shared_parent_child_loop",
                        "shared_loop_parent_edges": shared_loop_parent_edges,
                        "shared_loop_child_edges": shared_loop_child_edges,
                    }
                )
            else:
                socket_u, socket_v = orthonormal_basis(socket_inward)
                socket_points = original_points
                effective_socket_overcut_mm = max(
                    float(socket_ref.get("socket_overcut_mm", socket_overcut_mm)), 0.0
                )
                if effective_socket_overcut_mm > 1e-9:
                    socket_points = radial_offset_points(
                        original_points,
                        original_points.mean(axis=0),
                        socket_u,
                        socket_v,
                        socket_inward,
                        effective_socket_overcut_mm,
                    )
                    for local_index, point in zip(loop, socket_points):
                        output_vertices[int(local_index)] = point
                socket_directions = reference_loop_inward_directions(
                    socket_ref,
                    loop_global_ordered,
                    socket_inward,
                )
                matched_insert_shrink_mm, _ = clearance_offsets(
                    clearance_mode,
                    float(socket_ref.get("fit_clearance_mm", fit_clearance_mm)),
                )
                socket_bottom_points, matched_plane_record = matched_socket_bottom_geometry(
                    source_boundary_points=original_points,
                    socket_top_points=socket_points,
                    inward=socket_inward,
                    inward_directions=socket_directions,
                    child_insert_shrink_mm=matched_insert_shrink_mm,
                    max_extension_mm=max_extension_mm,
                    flat_clearance_mm=socket_flat_clearance_mm,
                    cap_mode=socket_cap_mode,
                    planar_extra_limit_mm=socket_planar_extra_limit_mm,
                    bottom_clearance_mm=bottom_clearance_mm,
                    parent_thickness_probe=parent_thickness_probe,
                    cap_decision=socket_ref.get("cap_decision"),
                    source_vertex_ids=loop_global_ordered,
                )
                added_side, added_cap, extension_record = add_loop_extrusion_and_cap(
                    output_vertices=output_vertices,
                    output_faces=output_faces,
                    top_ids=[int(i) for i in loop],
                    top_points=socket_points,
                    inward=socket_inward,
                    origin=socket_points.mean(axis=0),
                    max_extension_mm=max_extension_mm,
                    flat_clearance_mm=socket_flat_clearance_mm,
                    cap_mode=socket_cap_mode,
                    planar_extra_limit_mm=socket_planar_extra_limit_mm,
                    inward_directions=socket_directions,
                    bottom_points_override=socket_bottom_points,
                    plane_record_override=matched_plane_record,
                    parent_thickness_probe=parent_thickness_probe,
                )
                socket_side_faces += added_side
                socket_cap_faces += added_cap
                extension_record["loop_index"] = loop_index
                extension_record["matched_component_index"] = socket_ref["component_index"]
                extension_record["matched_cap_mode"] = socket_cap_mode
                extension_record["matched_color_code"] = socket_ref["color_code"]
                extension_record["matched_color_name"] = socket_ref["color_name"]
                extension_record["socket_overcut_mm"] = effective_socket_overcut_mm
                extension_record["bottom_clearance_mm"] = bottom_clearance_mm
                extension_record["shared_loop_parent_edges"] = shared_loop_parent_edges
                extension_record["shared_loop_child_edges"] = shared_loop_child_edges
                extension_record["shared_parent_child_loop"] = shared_parent_child_loop
                socket_extension_records.append(extension_record)
                continue

        if parent_contact_only and assembly_parent_index is not None and loop_parent_edges < 3:
            dropped_source_faces: list[int] = []
            if len(loop) <= 8 and not loop_neighbor_counts:
                loop_vertex_set = {int(value) for value in loop}
                loop_edges = {
                    tuple(sorted((int(loop[position]), int(loop[(position + 1) % len(loop)]))))
                    for position in range(len(loop))
                }
                for face_index, face in enumerate(local_faces):
                    face_values = [int(value) for value in face]
                    if not set(face_values).issubset(loop_vertex_set):
                        continue
                    face_edges = {
                        tuple(sorted((face_values[0], face_values[1]))),
                        tuple(sorted((face_values[1], face_values[2]))),
                        tuple(sorted((face_values[2], face_values[0]))),
                    }
                    if face_edges.intersection(loop_edges):
                        dropped_non_parent_face_indices.add(int(face_index))
                        dropped_source_faces.append(int(face_index))
            skipped_non_parent_loop_records.append(
                {
                    "loop_index": loop_index,
                    "vertices": len(loop),
                    "parent_edges": int(loop_parent_edges),
                    "neighbor_counts": {str(key): int(value) for key, value in sorted(loop_neighbor_counts.items())},
                    "dropped_source_faces": dropped_source_faces,
                    "skip_reason": "non_parent_non_child_boundary",
                }
            )
            continue

        loop_origin = original_points.mean(axis=0)
        top_points = radial_offset_points(original_points, loop_origin, u, v, inward, -top_edge_clearance_mm)
        fit_points = radial_offset_points(original_points, loop_origin, u, v, inward, -insert_shrink_mm)
        for local_index, point in zip(loop, top_points):
            output_vertices[int(local_index)] = point
        loop_fit_points.append(fit_points)
        insert_loop_records.append(
            {
                "loop_index": loop_index,
                "loop": [int(i) for i in loop],
                "top_points": top_points,
                "source_boundary_points": original_points,
                "fit_points": fit_points,
                "inward_directions": loop_inward_directions[int(loop_index)],
            }
        )
        loop_clearance_records.append(
            {
                "loop_index": loop_index,
                "vertices": len(loop),
                "insert_shrink_mm": insert_shrink_mm,
                "top_edge_clearance_mm": top_edge_clearance_mm,
                "lead_in_mm": lead_in_mm,
                "sibling_clearance_mm": sibling_clearance_mm,
                **sibling_record,
            }
        )
        if sibling_record["sibling_edges"] > 0:
            record = dict(sibling_record)
            record["loop_index"] = loop_index
            record["sibling_clearance_mm"] = sibling_clearance_mm
            sibling_clearance_records.append(record)

    if dropped_non_parent_face_indices:
        output_faces = [
            face
            for face_index, face in enumerate(output_faces)
            if face_index not in dropped_non_parent_face_indices
        ]

    loop_points_2d_for_grouping = [project_points(record["top_points"], component_center, u, v) for record in insert_loop_records]
    loop_groups = group_loops(loop_points_2d_for_grouping)

    loops_bottom: list[list[int]] = []
    loops_lead: list[list[int]] = []
    bottom_loop_points_2d: list[np.ndarray] = []
    group_extension_records = []

    for insert_index, record in enumerate(insert_loop_records):
        loop_index = int(record["loop_index"])
        top_ids = record["loop"]
        fit_points = record["fit_points"]
        inward_directions = record["inward_directions"]
        distances, inward_directions, plane_record = boundary_cap_distances(
            fit_points,
            inward,
            inward_directions,
            max_extension_mm,
            flat_clearance_mm,
            cap_mode,
            insert_planar_extra_limit_mm,
            parent_thickness_probe,
        )
        source_boundary_points = record["source_boundary_points"]
        expected_parent_socket_top = radial_offset_points(
            source_boundary_points,
            source_boundary_points.mean(axis=0),
            u,
            v,
            inward,
            parent_socket_overcut_mm,
        )
        distances, plane_record = reserve_flat_socket_travel_budget(
            child_fit_points=fit_points,
            child_distances=distances,
            child_directions=inward_directions,
            socket_top_points=expected_parent_socket_top,
            bottom_clearance_mm=bottom_clearance_mm,
            maximum_socket_travel_mm=max_extension_mm + max(float(planar_extra_limit_mm), 0.0),
            plane_record=plane_record,
        )
        lead_ids = []
        if lead_in_mm > 1e-9 and insert_shrink_mm > top_edge_clearance_mm + 1e-9:
            for fit_point, direction, distance in zip(fit_points, inward_directions, distances):
                lead_depth = min(lead_in_mm, max(0.0, float(distance) * 0.5))
                lead_ids.append(len(output_vertices))
                output_vertices.append(fit_point + direction * lead_depth)
        bottom_ids = []
        for fit_point, direction, distance in zip(fit_points, inward_directions, distances):
            bottom_point = fit_point + direction * float(distance)
            bottom_ids.append(len(output_vertices))
            output_vertices.append(bottom_point)
        loops_lead.append(lead_ids)
        loops_bottom.append(bottom_ids)
        bottom_loop_points_2d.append(project_points(np.array([output_vertices[i] for i in bottom_ids]), component_center, u, v))
        group_extension_records.append(
            {
                "loop_index": loop_index,
                "vertices": len(top_ids),
                "extension_min_mm": float(distances.min()),
                "extension_max_mm": float(distances.max()),
                "lead_in_applied": bool(lead_ids),
                "lead_in_depth_min_mm": float(
                    min((np.linalg.norm(np.array(output_vertices[i]) - fit_points[pos]) for pos, i in enumerate(lead_ids)), default=0.0)
                ),
                "lead_in_depth_max_mm": float(
                    max((np.linalg.norm(np.array(output_vertices[i]) - fit_points[pos]) for pos, i in enumerate(lead_ids)), default=0.0)
                ),
                **plane_record,
            }
        )

    side_faces = 0
    for record, lead_ids, bottom_ids in zip(insert_loop_records, loops_lead, loops_bottom):
        top_ring = record["loop"]
        if lead_ids:
            side_faces += add_side_faces_between_rings(output_faces, top_ring, lead_ids)
            side_faces += add_side_faces_between_rings(output_faces, lead_ids, bottom_ids)
        else:
            side_faces += add_side_faces_between_rings(output_faces, top_ring, bottom_ids)

    cap_faces = triangulate_cap(output_vertices, output_faces, loops_bottom, bottom_loop_points_2d, loop_groups, inward)

    mesh = trimesh.Trimesh(
        vertices=np.array(output_vertices, dtype=np.float64),
        faces=np.array(output_faces, dtype=np.int64),
        process=False,
        metadata={"name": part_id},
    )
    mesh = finalize_large_partition_mesh(mesh) if len(output_faces) > 250000 else finalize_mesh(mesh)

    color_info = COLOR_INFO.get(component.color_code, {"name": component.color_code, "hex": "", "rgba": [200, 200, 200, 255]})
    mesh.visual.face_colors = np.tile(np.array(color_info["rgba"], dtype=np.uint8), (len(mesh.faces), 1))

    bbox = mesh.bounds
    selected_cap_modes = sorted(
        {str(record.get("cap_mode", cap_mode)) for record in group_extension_records + socket_extension_records}
    )
    geometry_cap_mode = selected_cap_modes[0] if len(selected_cap_modes) == 1 else "mixed"
    stats = {
        "part_id": part_id,
        "processing_mode": "insert_inward_adaptive_bottom",
        "selected_processing_mode": "inward",
        "color_code": component.color_code,
        "color_name": color_info["name"],
        "color_hex": color_info["hex"],
        "source_faces": component.face_count,
        "output_faces": int(len(mesh.faces)),
        "output_vertices": int(len(mesh.vertices)),
        "boundary_loops": len(loops),
        "boundary_vertices": boundary_vertex_count,
        "boundary_fairing_records": boundary_fairing_records,
        "loop_groups": loop_groups,
        "side_faces_added": side_faces + socket_side_faces,
        "cap_faces_added": cap_faces + socket_cap_faces,
        "insert_side_faces_added": side_faces,
        "insert_cap_faces_added": cap_faces,
        "child_socket_side_faces_added": socket_side_faces,
        "child_socket_cap_faces_added": socket_cap_faces,
        "max_extension_limit_mm": max_extension_mm,
        "cap_mode": cap_mode,
        "geometry_cap_mode": geometry_cap_mode,
        "planar_extra_limit_mm": planar_extra_limit_mm,
        "selected_cap_modes": selected_cap_modes,
        "flat_clearance_mm": flat_clearance_mm,
        "clearance_mode": clearance_mode,
        "fit_clearance_mm": max(float(fit_clearance_mm), 0.0),
        "insert_shrink_mm": insert_shrink_mm,
        "socket_overcut_mm": 0.0,
        "child_socket_overcut_mm": socket_overcut_mm,
        "bottom_clearance_mm": 0.0,
        "child_socket_bottom_clearance_mm": bottom_clearance_mm,
        "lead_in_mm": lead_in_mm,
        "top_edge_clearance_mm": top_edge_clearance_mm,
        "sibling_clearance_mm": sibling_clearance_mm,
        "parent_contact_only": bool(parent_contact_only),
        "inward_direction": inward.round(6).tolist(),
        "inward_direction_source": "assembly_parent" if inward_override is not None else "model_center",
        "local_inward_direction_records": local_inward_direction_records,
        "local_inward_outward_vertices_before": int(
            sum(record["global_outward_vertices_before"] for record in local_inward_direction_records)
        ),
        "local_inward_outward_vertices_after": int(
            sum(record["outward_vertices_after"] for record in local_inward_direction_records)
        ),
        "extension_max_observed_mm": float(max((r["extension_max_mm"] for r in group_extension_records), default=0.0)),
        "maximum_generated_inward_travel_mm": float(
            max(
                (r["extension_max_mm"] for r in group_extension_records + socket_extension_records),
                default=0.0,
            )
        ),
        "extension_min_observed_mm": float(min((r["extension_min_mm"] for r in group_extension_records), default=0.0)),
        "flat_bottom_required_max_extension_mm": float(max((r["flat_bottom_required_max_extension_mm"] for r in group_extension_records), default=0.0)),
        "flat_bottom_target_exceeded": bool(any(r["target_exceeded"] for r in group_extension_records)),
        "fixed_inward_depth_mm": max_extension_mm,
        "boundary_span_max_mm": float(max((r["flat_span_mm"] for r in group_extension_records), default=0.0)),
        "fixed_inward_depth_applied": bool(all(r.get("fixed_inward_depth_applied", False) for r in group_extension_records)) if group_extension_records else True,
        "child_socket_count": len(socket_extension_records),
        "child_socket_extensions": socket_extension_records,
        "child_socket_skipped_count": len(skipped_socket_records),
        "child_socket_skipped": skipped_socket_records,
        "skipped_non_parent_loop_count": len(skipped_non_parent_loop_records),
        "skipped_non_parent_loops": skipped_non_parent_loop_records,
        "dropped_non_parent_source_faces": int(len(dropped_non_parent_face_indices)),
        "loop_clearances": loop_clearance_records,
        "sibling_clearances": sibling_clearance_records,
        "loop_extensions": group_extension_records,
        "bbox_min": bbox[0].round(6).tolist(),
        "bbox_max": bbox[1].round(6).tolist(),
        "bbox_size_mm": (bbox[1] - bbox[0]).round(6).tolist(),
        **mesh_runtime_stats(mesh),
    }
    return mesh, stats


def make_body_cut_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    body_component: Component,
    part_id: str,
    cut_refs: list[dict],
    max_extension_mm: float,
    boundary_fairing: BoundaryFairingContext,
    flat_clearance_mm: float,
    fit_clearance_mm: float,
    socket_overcut_mm: float,
    bottom_clearance_mm: float,
    clearance_mode: str,
    model_center: np.ndarray,
    cap_mode: str,
    planar_extra_limit_mm: float,
    preserve_unmatched_source_geometry: bool = False,
) -> tuple[trimesh.Trimesh, dict]:
    parent_thickness_probe = ParentThicknessProbe(vertices, faces)
    local_vertices, local_faces, _global_to_local, global_vertex_ids = build_local_mesh(vertices, faces, body_component)
    loops = boundary_loops(local_faces)
    boundary_vertex_count = int(sum(len(loop) for loop in loops))
    component_center = local_vertices[local_faces.reshape(-1)].mean(axis=0)
    fallback_inward = -average_outward_normal(local_vertices, local_faces, component_center, model_center)
    fallback_inward /= max(float(np.linalg.norm(fallback_inward)), 1e-12)
    vertex_inward_normals = mesh_vertex_inward_normals(local_vertices, local_faces)
    fallback_loop_directions: dict[int, np.ndarray] = {}
    fallback_direction_records: list[dict] = []
    for loop_index, loop in enumerate(loops):
        directions, direction_record = safe_boundary_inward_directions(
            vertex_inward_normals,
            loop,
            fallback_inward,
            loop_points=local_vertices[np.asarray(loop, dtype=np.int64)],
        )
        fallback_loop_directions[int(loop_index)] = directions
        fallback_direction_records.append({"loop_index": int(loop_index), **direction_record})

    loop_cut_refs, immutable_source_loop_records = classify_body_cut_loop_references(
        loops,
        global_vertex_ids,
        cut_refs,
        preserve_unmatched_source_geometry=preserve_unmatched_source_geometry,
    )
    mutable_loop_indices = [
        int(loop_index)
        for loop_index, ref in enumerate(loop_cut_refs)
        if not preserve_unmatched_source_geometry or ref is not None
    ]
    mutable_loops = [loops[loop_index] for loop_index in mutable_loop_indices]
    local_vertices, boundary_fairing_records = BoundaryFairingService.fair_local_loops(
        local_vertices,
        mutable_loops,
        global_vertex_ids,
        boundary_fairing,
    )
    for fairing_record, loop_index in zip(
        boundary_fairing_records,
        mutable_loop_indices,
    ):
        fairing_record["loop_index"] = int(loop_index)
    if immutable_source_loop_records:
        runtime_log(
            "递归几何",
            "inherited_parent_contact_shell_preserved",
            "上阶段父接触外壳的未匹配边界保持不可变，仅处理本阶段直属子件接口",
            part_id=str(part_id),
            immutable_loop_count=int(len(immutable_source_loop_records)),
            immutable_vertex_count=int(
                sum(record["vertices"] for record in immutable_source_loop_records)
            ),
            immutable_loops=immutable_source_loop_records,
            mutable_child_interface_loop_indices=mutable_loop_indices,
        )

    output_vertices: list[np.ndarray] = [p.copy() for p in local_vertices]
    output_faces: list[list[int]] = local_faces.astype(int).tolist()
    component_center = local_vertices[local_faces.reshape(-1)].mean(axis=0)

    side_faces = 0
    cap_faces = 0
    loop_extension_records = []
    matched_refs = []
    applied_direction_records = []
    socket_overcut_mm = max(float(socket_overcut_mm), 0.0)
    bottom_clearance_mm = max(float(bottom_clearance_mm), 0.0)
    body_flat_clearance_mm = max(float(flat_clearance_mm), 0.0)

    for loop_index, loop in enumerate(loops):
        body_loop_global = set(int(global_vertex_ids[i]) for i in loop)
        body_loop_global_ordered = [int(global_vertex_ids[i]) for i in loop]
        body_loop_global_edges = {
            tuple(sorted((body_loop_global_ordered[position], body_loop_global_ordered[(position + 1) % len(body_loop_global_ordered)])))
            for position in range(len(body_loop_global_ordered))
        }
        ref = loop_cut_refs[int(loop_index)]
        if preserve_unmatched_source_geometry and ref is None:
            continue
        inward = ref["inward"] if ref else fallback_inward
        inward_directions = (
            reference_loop_inward_directions(ref, body_loop_global_ordered, inward)
            if ref is not None
            else fallback_loop_directions[int(loop_index)]
        )
        direction_record = dict(
            ref.get("local_inward_direction_record", {})
            if ref is not None
            else fallback_direction_records[int(loop_index)]
        )
        direction_record["loop_index"] = int(loop_index)
        direction_record["matched_component_index"] = ref.get("component_index") if ref else None
        applied_direction_records.append(direction_record)
        loop_cap_mode = str(ref.get("cap_mode", cap_mode)) if ref else cap_mode
        loop_planar_extra_limit_mm = float(
            ref.get("planar_extra_limit_mm")
            if ref and ref.get("planar_extra_limit_mm") is not None
            else planar_extra_limit_mm
        )
        u, v = orthonormal_basis(inward)
        loop_array = np.array(loop, dtype=np.int64)
        source_boundary_points = local_vertices[loop_array]
        top_points = source_boundary_points
        effective_socket_overcut_mm = max(
            float(ref.get("socket_overcut_mm", socket_overcut_mm)) if ref else socket_overcut_mm,
            0.0,
        )
        if effective_socket_overcut_mm > 1e-9:
            top_points = radial_offset_points(
                top_points, top_points.mean(axis=0), u, v, inward, effective_socket_overcut_mm
            )
            for local_index, point in zip(loop, top_points):
                output_vertices[int(local_index)] = point
        shared_cap_decision = ref.get("cap_decision") if ref else None
        if shared_cap_decision is None:
            added_side, added_cap, extension_record = add_loop_extrusion_and_cap(
                output_vertices=output_vertices,
                output_faces=output_faces,
                top_ids=[int(i) for i in loop],
                top_points=top_points,
                inward=inward,
                origin=top_points.mean(axis=0),
                max_extension_mm=max_extension_mm,
                flat_clearance_mm=body_flat_clearance_mm,
                cap_mode=loop_cap_mode,
                planar_extra_limit_mm=loop_planar_extra_limit_mm,
                inward_directions=inward_directions,
                parent_thickness_probe=parent_thickness_probe,
            )
        else:
            child_insert_shrink_mm, _unused_socket_overcut_mm = clearance_offsets(
                clearance_mode,
                float(ref.get("fit_clearance_mm", fit_clearance_mm)),
            )
            socket_bottom_points, shared_record = matched_socket_bottom_geometry(
                source_boundary_points=source_boundary_points,
                socket_top_points=top_points,
                inward=inward,
                inward_directions=inward_directions,
                child_insert_shrink_mm=child_insert_shrink_mm,
                max_extension_mm=max_extension_mm,
                flat_clearance_mm=body_flat_clearance_mm,
                cap_mode=loop_cap_mode,
                planar_extra_limit_mm=loop_planar_extra_limit_mm,
                bottom_clearance_mm=bottom_clearance_mm,
                parent_thickness_probe=parent_thickness_probe,
                cap_decision=shared_cap_decision,
                source_vertex_ids=body_loop_global_ordered,
            )
            added_side, added_cap, extension_record = add_loop_extrusion_and_cap(
                output_vertices=output_vertices,
                output_faces=output_faces,
                top_ids=[int(i) for i in loop],
                top_points=top_points,
                inward=inward,
                origin=top_points.mean(axis=0),
                max_extension_mm=max_extension_mm,
                flat_clearance_mm=body_flat_clearance_mm,
                cap_mode=loop_cap_mode,
                planar_extra_limit_mm=loop_planar_extra_limit_mm,
                inward_directions=inward_directions,
                bottom_points_override=socket_bottom_points,
                plane_record_override=shared_record,
                parent_thickness_probe=parent_thickness_probe,
            )
        side_faces += added_side
        cap_faces += added_cap
        extension_record["loop_index"] = loop_index
        extension_record["matched_cap_mode"] = loop_cap_mode
        extension_record["socket_overcut_mm"] = effective_socket_overcut_mm
        extension_record["bottom_clearance_mm"] = bottom_clearance_mm
        extension_record["matched_component_index"] = ref["component_index"] if ref else None
        extension_record["matched_color_code"] = ref["color_code"] if ref else None
        extension_record["matched_color_name"] = ref["color_name"] if ref else None
        loop_extension_records.append(extension_record)
        matched_refs.append(ref)

    mesh = trimesh.Trimesh(
        vertices=np.array(output_vertices, dtype=np.float64),
        faces=np.array(output_faces, dtype=np.int64),
        process=False,
        metadata={"name": part_id},
    )
    mesh = finalize_large_partition_mesh(mesh) if len(output_faces) > 250000 else finalize_mesh(mesh)

    color_info = COLOR_INFO.get(body_component.color_code, {"name": body_component.color_code, "hex": "", "rgba": [200, 200, 200, 255]})
    mesh.visual.face_colors = np.tile(np.array(color_info["rgba"], dtype=np.uint8), (len(mesh.faces), 1))
    bbox = mesh.bounds
    selected_cap_modes = sorted({str(record.get("cap_mode", cap_mode)) for record in loop_extension_records})
    geometry_cap_mode = selected_cap_modes[0] if len(selected_cap_modes) == 1 else "mixed"
    stats = {
        "part_id": part_id,
        "processing_mode": "body_cut_from_adjacent_insert_boundaries",
        "selected_processing_mode": "body",
        "color_code": body_component.color_code,
        "color_name": color_info["name"],
        "color_hex": color_info["hex"],
        "source_faces": body_component.face_count,
        "output_faces": int(len(mesh.faces)),
        "output_vertices": int(len(mesh.vertices)),
        "boundary_loops": len(loops),
        "boundary_vertices": boundary_vertex_count,
        "boundary_fairing_records": boundary_fairing_records,
        "preserve_unmatched_source_geometry": bool(
            preserve_unmatched_source_geometry
        ),
        "immutable_source_loop_count": int(len(immutable_source_loop_records)),
        "immutable_source_loops": immutable_source_loop_records,
        "processed_body_cut_loop_indices": mutable_loop_indices,
        "side_faces_added": side_faces,
        "cap_faces_added": cap_faces,
        "max_extension_limit_mm": max_extension_mm,
        "cap_mode": cap_mode,
        "geometry_cap_mode": geometry_cap_mode,
        "planar_extra_limit_mm": planar_extra_limit_mm,
        "selected_cap_modes": selected_cap_modes,
        "flat_clearance_mm": flat_clearance_mm,
        "clearance_mode": clearance_mode,
        "fit_clearance_mm": max(float(fit_clearance_mm), 0.0),
        "insert_shrink_mm": 0.0,
        "socket_overcut_mm": socket_overcut_mm,
        "bottom_clearance_mm": bottom_clearance_mm,
        "body_flat_clearance_mm": body_flat_clearance_mm,
        "lead_in_mm": 0.0,
        "top_edge_clearance_mm": 0.0,
        "local_inward_direction_records": applied_direction_records,
        "local_inward_outward_vertices_before": int(
            sum(record.get("global_outward_vertices_before", 0) for record in applied_direction_records)
        ),
        "local_inward_outward_vertices_after": int(
            sum(record.get("outward_vertices_after", 0) for record in applied_direction_records)
        ),
        "extension_max_observed_mm": float(max((r["extension_max_mm"] for r in loop_extension_records), default=0.0)),
        "maximum_generated_inward_travel_mm": float(max((r["extension_max_mm"] for r in loop_extension_records), default=0.0)),
        "extension_min_observed_mm": float(min((r["extension_min_mm"] for r in loop_extension_records), default=0.0)),
        "flat_bottom_required_max_extension_mm": float(max((r["flat_bottom_required_max_extension_mm"] for r in loop_extension_records), default=0.0)),
        "flat_bottom_target_exceeded": bool(any(r["target_exceeded"] for r in loop_extension_records)),
        "fixed_inward_depth_mm": max_extension_mm,
        "boundary_span_max_mm": float(max((r["flat_span_mm"] for r in loop_extension_records), default=0.0)),
        "fixed_inward_depth_applied": bool(all(r.get("fixed_inward_depth_applied", False) for r in loop_extension_records)) if loop_extension_records else True,
        "child_socket_count": len(loop_extension_records),
        "child_socket_extensions": loop_extension_records,
        "loop_extensions": loop_extension_records,
        "bbox_min": bbox[0].round(6).tolist(),
        "bbox_max": bbox[1].round(6).tolist(),
        "bbox_size_mm": (bbox[1] - bbox[0]).round(6).tolist(),
        **mesh_runtime_stats(mesh),
    }
    return mesh, stats


def make_layer_child_subassembly_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    source_colors: list[str],
    components: list[Component],
    component: Component,
    root_child_index: int,
    subtree_indices: list[int],
    parent_index: int,
    part_id: str,
    max_extension_mm: float,
    boundary_fairing: BoundaryFairingContext,
    flat_clearance_mm: float,
    fit_clearance_mm: float,
    insert_shrink_mm: float,
    lead_in_mm: float,
    top_edge_clearance_mm: float,
    boundary_neighbor_lookup: dict[tuple[int, int], set[int]],
    inward_override: np.ndarray | None,
    model_center: np.ndarray,
    cap_mode: str,
    planar_extra_limit_mm: float,
    cap_decisions_by_loop: dict[int, CapDecision] | None = None,
) -> tuple[trimesh.Trimesh, dict]:
    cap_decisions_by_loop = cap_decisions_by_loop or {}
    parent_thickness_probe = ParentThicknessProbe(vertices, faces)
    local_vertices, local_faces, _global_to_local, global_vertex_ids = build_local_mesh(vertices, faces, component)
    union_global_to_local = {int(global_id): int(local_index) for local_index, global_id in enumerate(global_vertex_ids)}
    loops = boundary_loops(local_faces)
    root_component = components[root_child_index - 1]
    root_local_vertices, root_local_faces, _root_map, root_global_vertex_ids = build_local_mesh(vertices, faces, root_component)
    root_loops = boundary_loops(root_local_faces)
    current_layer_component_set = {int(root_child_index)}
    root_center = root_local_vertices[root_local_faces.reshape(-1)].mean(axis=0)
    inward = inward_override
    if inward is None:
        inward = component_inward_direction(vertices, faces, root_component, model_center)
    inward = inward / max(float(np.linalg.norm(inward)), 1e-12)
    root_vertex_inward_normals = mesh_vertex_inward_normals(root_local_vertices, root_local_faces)

    selected_loop_records = []
    ignored_loop_records = []
    for loop_index, root_loop in enumerate(root_loops):
        contact = boundary_loop_parent_contact(root_loop, root_global_vertex_ids, boundary_neighbor_lookup, parent_index, current_layer_component_set)
        mapped_loop = [union_global_to_local[int(root_global_vertex_ids[int(local_index)])] for local_index in root_loop]
        record = {
            "loop_index": int(loop_index),
            "vertices": int(len(mapped_loop)),
            "contact_source_component_index": int(root_child_index),
            **contact,
        }
        loop_directions, direction_record = safe_boundary_inward_directions(
            root_vertex_inward_normals,
            root_loop,
            inward,
            loop_points=root_local_vertices[np.asarray(root_loop, dtype=np.int64)],
        )
        if int(contact["parent_edges"]) >= 3:
            selected_loop_records.append(
                {
                    **record,
                    "loop": [int(value) for value in mapped_loop],
                    "global_loop": [
                        int(root_global_vertex_ids[int(local_index)])
                        for local_index in root_loop
                    ],
                    "inward_directions": loop_directions,
                    "local_inward_direction_record": direction_record,
                }
            )
        else:
            ignored_loop_records.append(record)

    selected_loops = [record["loop"] for record in selected_loop_records]
    local_vertices, boundary_fairing_records = BoundaryFairingService.fair_local_loops(
        local_vertices,
        selected_loops,
        global_vertex_ids,
        boundary_fairing,
    )
    for fairing_record, selected_record in zip(boundary_fairing_records, selected_loop_records):
        fairing_record["loop_index"] = int(selected_record["loop_index"])

    u, v = orthonormal_basis(inward)

    output_vertices: list[np.ndarray] = [point.copy() for point in local_vertices]
    output_faces: list[list[int]] = local_faces.astype(int).tolist()
    color_info = COLOR_INFO.get(component.color_code, {"name": component.color_code, "hex": "", "rgba": [200, 200, 200, 255]})
    source_face_codes = [
        str(source_colors[int(global_face)])
        for global_face in component.global_faces
    ]
    loop_extension_records = []
    insert_loop_records = []
    loops_lead: list[list[int]] = []
    loops_bottom: list[list[int]] = []
    bottom_loop_points_2d: list[np.ndarray] = []

    insert_shrink_mm = max(float(insert_shrink_mm), 0.0)
    lead_in_mm = max(float(lead_in_mm), 0.0)
    top_edge_clearance_mm = min(max(float(top_edge_clearance_mm), 0.0), insert_shrink_mm)

    for record in selected_loop_records:
        loop = record["loop"]
        loop_array = np.array(loop, dtype=np.int64)
        original_points = local_vertices[loop_array]
        loop_origin = original_points.mean(axis=0)
        top_points = radial_offset_points(original_points, loop_origin, u, v, inward, -top_edge_clearance_mm)
        fit_points = radial_offset_points(original_points, loop_origin, u, v, inward, -insert_shrink_mm)
        inward_directions = record["inward_directions"]
        for local_index, point in zip(loop, top_points):
            output_vertices[int(local_index)] = point
        cap_decision = cap_decisions_by_loop.get(int(record["loop_index"]))
        if cap_decision is None:
            distances, inward_directions, plane_record = boundary_cap_distances(
                fit_points,
                inward,
                inward_directions,
                max_extension_mm,
                flat_clearance_mm,
                cap_mode,
                planar_extra_limit_mm,
                parent_thickness_probe,
            )
        else:
            (
                planned_fit_points,
                distances,
                inward_directions,
                plane_record,
            ) = remap_cap_decision(cap_decision, record["global_loop"])
            if not np.allclose(
                fit_points,
                planned_fit_points,
                atol=1e-9,
                rtol=0.0,
            ):
                raise ValueError(
                    f"P{int(root_child_index):02d} boundary fit points changed "
                    f"after cap planning on loop {int(record['loop_index'])}"
                )
        lead_ids = []
        if lead_in_mm > 1e-9 and insert_shrink_mm > top_edge_clearance_mm + 1e-9:
            for fit_point, direction, distance in zip(fit_points, inward_directions, distances):
                lead_depth = min(lead_in_mm, max(0.0, float(distance) * 0.5))
                lead_ids.append(len(output_vertices))
                output_vertices.append(fit_point + direction * lead_depth)
        bottom_ids = []
        for fit_point, direction, distance in zip(fit_points, inward_directions, distances):
            bottom_ids.append(len(output_vertices))
            output_vertices.append(fit_point + direction * float(distance))
        loops_lead.append(lead_ids)
        loops_bottom.append(bottom_ids)
        bottom_loop_points_2d.append(project_points(np.array([output_vertices[i] for i in bottom_ids]), root_center, u, v))
        insert_loop_records.append({"loop": loop, "top_points": top_points})
        loop_extension_records.append(
            {
                "loop_index": int(record["loop_index"]),
                "vertices": int(len(loop)),
                "parent_edges": int(record["parent_edges"]),
                "extension_min_mm": float(distances.min()),
                "extension_max_mm": float(distances.max()),
                "lead_in_applied": bool(lead_ids),
                **plane_record,
            }
        )

    side_faces = 0
    for record, lead_ids, bottom_ids in zip(insert_loop_records, loops_lead, loops_bottom):
        top_ring = record["loop"]
        if lead_ids:
            side_faces += add_side_faces_between_rings(output_faces, top_ring, lead_ids)
            side_faces += add_side_faces_between_rings(output_faces, lead_ids, bottom_ids)
        else:
            side_faces += add_side_faces_between_rings(output_faces, top_ring, bottom_ids)

    loop_groups = group_loops(bottom_loop_points_2d) if bottom_loop_points_2d else []
    cap_faces = triangulate_cap(output_vertices, output_faces, loops_bottom, bottom_loop_points_2d, loop_groups, inward)
    added_face_count = max(0, len(output_faces) - len(source_face_codes))
    face_color_codes = source_face_codes + [
        str(root_component.color_code) for _ in range(added_face_count)
    ]

    mesh = trimesh.Trimesh(
        vertices=np.array(output_vertices, dtype=np.float64),
        faces=np.array(output_faces, dtype=np.int64),
        process=False,
        metadata={"name": part_id},
    )
    face_filament_slots = [
        COLOR_INFO.get(code, {}).get("filament_slot")
        for code in face_color_codes
    ]
    mesh = finalize_recursive_colored_mesh(
        mesh,
        face_color_codes,
        face_filament_slots,
        str(root_component.color_code),
    )

    bbox = mesh.bounds
    selected_cap_modes = sorted({str(record.get("cap_mode", cap_mode)) for record in loop_extension_records})
    geometry_cap_mode = selected_cap_modes[0] if len(selected_cap_modes) == 1 else "mixed"
    stats = {
        "part_id": part_id,
        "processing_mode": "layer_subassembly_parent_contact_only",
        "color_code": component.color_code,
        "color_name": color_info["name"],
        "color_hex": color_info["hex"],
        "source_faces": component.face_count,
        "output_faces": int(len(mesh.faces)),
        "output_vertices": int(len(mesh.vertices)),
        "boundary_loops": len(loops),
        "boundary_loops_total": len(loops),
        "union_boundary_loops_total": len(loops),
        "root_boundary_loops_total": len(root_loops),
        "parent_contact_source_component_index": int(root_child_index),
        "selected_parent_contact_loops": len(selected_loop_records),
        "boundary_fairing_records": boundary_fairing_records,
        "ignored_non_parent_loops": len(ignored_loop_records),
        "parent_contact_edges": int(sum(record["parent_edges"] for record in selected_loop_records)),
        "selected_loop_summary": [
            {
                key: value
                for key, value in record.items()
                if key not in {"loop", "global_loop", "inward_directions"}
            }
            for record in selected_loop_records
        ],
        "ignored_loop_summary": ignored_loop_records,
        "side_faces_added": side_faces,
        "cap_faces_added": cap_faces,
        "child_socket_count": 0,
        "loop_groups": loop_groups,
        "max_extension_limit_mm": max_extension_mm,
        "cap_mode": cap_mode,
        "geometry_cap_mode": geometry_cap_mode,
        "planar_extra_limit_mm": planar_extra_limit_mm,
        "selected_cap_modes": selected_cap_modes,
        "shared_prevalidated_cap_decision_loops": sorted(
            int(loop_index) for loop_index in cap_decisions_by_loop
        ),
        "flat_clearance_mm": flat_clearance_mm,
        "fit_clearance_mm": max(float(fit_clearance_mm), 0.0),
        "insert_shrink_mm": insert_shrink_mm,
        "socket_overcut_mm": 0.0,
        "bottom_clearance_mm": 0.0,
        "lead_in_mm": lead_in_mm,
        "top_edge_clearance_mm": top_edge_clearance_mm,
        "inward_direction": inward.round(6).tolist(),
        "local_inward_direction_records": [
            {"loop_index": int(record["loop_index"]), **record["local_inward_direction_record"]}
            for record in selected_loop_records
        ],
        "local_inward_outward_vertices_before": int(
            sum(
                record["local_inward_direction_record"]["global_outward_vertices_before"]
                for record in selected_loop_records
            )
        ),
        "local_inward_outward_vertices_after": int(
            sum(
                record["local_inward_direction_record"]["outward_vertices_after"]
                for record in selected_loop_records
            )
        ),
        "extension_max_observed_mm": float(max((record["extension_max_mm"] for record in loop_extension_records), default=0.0)),
        "maximum_generated_inward_travel_mm": float(max((record["extension_max_mm"] for record in loop_extension_records), default=0.0)),
        "extension_min_observed_mm": float(min((record["extension_min_mm"] for record in loop_extension_records), default=0.0)),
        "boundary_span_max_mm": float(max((record["flat_span_mm"] for record in loop_extension_records), default=0.0)),
        "fixed_inward_depth_mm": max_extension_mm,
        "fixed_inward_depth_applied": bool(all(record.get("fixed_inward_depth_applied", False) for record in loop_extension_records)) if loop_extension_records else True,
        "loop_extensions": loop_extension_records,
        "bbox_min": bbox[0].round(6).tolist(),
        "bbox_max": bbox[1].round(6).tolist(),
        "bbox_size_mm": (bbox[1] - bbox[0]).round(6).tolist(),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "open_edges": open_edge_count(mesh),
        "volume_mm3": float(mesh.volume) if mesh.is_watertight else None,
        "root_child_index": int(root_child_index),
        "parent_index": int(parent_index),
        "contains": [int(index) for index in subtree_indices],
    }
    return mesh, stats






class InwardDirectionPlanner:
    """Build a smooth, locally inward-safe direction field for one boundary."""

    plan = staticmethod(safe_boundary_inward_directions)


class AdaptiveCapPlanner:
    """Choose coherent flat or smooth local-offset cap geometry."""

    choose = staticmethod(boundary_cap_distances)


class BoundaryTriangulator:
    """Triangulate simple, spatial, and holed boundary caps without center fans."""

    polygon = staticmethod(triangulate_polygon_ear_clip)
    spatial_loop = staticmethod(triangulate_ordered_loop_3d)
    cap = staticmethod(triangulate_cap)


class PartMeshBuilder:
    """Build final insert, body-cut, and recursive subassembly meshes."""

    build_part = staticmethod(make_part_mesh)
    build_body_cut = staticmethod(make_body_cut_mesh)
    build_layer_subassembly = staticmethod(make_layer_child_subassembly_mesh)
