from __future__ import annotations

from .common import *
from .project import *
from .recognition import *
from .mesh import *
from .package_io import *



def component_identity_index(components: list[Component], target: Component | None) -> int | None:
    """Return a one-based index without comparing NumPy-backed dataclass fields."""
    if target is None:
        return None
    return next((index for index, component in enumerate(components, start=1) if component is target), None)


def component_recognition_records(
    vertices: np.ndarray,
    faces: np.ndarray,
    components: list[Component],
    source_colors: list[str],
) -> list[dict]:
    records = []
    for index, component in enumerate(components, start=1):
        color_info = COLOR_INFO.get(component.color_code, {"name": component.color_code, "hex": ""})
        mapping_source = color_info.get("mapping_source", "unmapped")
        resolution_status = color_resolution_status(color_info)
        _, local_faces, _, _ = build_local_mesh(vertices, faces, component)
        loops = boundary_loops(local_faces)
        bbox_size = component.bbox_max - component.bbox_min
        raw_token_counts = collections.Counter(
            str(source_colors[int(face_id)]) for face_id in component.global_faces
        )
        raw_color_tokens = [
            {"token": token, "faces": int(count)}
            for token, count in sorted(raw_token_counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
        ]
        records.append(
            {
                "part_index": index,
                "role": "part",
                "color_code": component.color_code,
                "raw_color_token": "|".join(item["token"] for item in raw_color_tokens),
                "raw_color_tokens": raw_color_tokens,
                "color_name": color_info.get("name", component.color_code),
                "color_hex": color_info.get("hex", ""),
                "filament_slot_index": color_info.get("filament_slot"),
                "color_mapping_source": mapping_source,
                "color_resolution_status": resolution_status,
                "color_is_fallback": resolution_status == "fallback_estimate",
                "faces": component.face_count,
                "area_mm2": component.area,
                "bbox_min": component.bbox_min.round(6).tolist(),
                "bbox_max": component.bbox_max.round(6).tolist(),
                "bbox_size_mm": bbox_size.round(6).tolist(),
                "center": component.center.round(6).tolist(),
                "boundary_loops": len(loops),
            }
        )
    return records


def point_in_triangle_2d_strict(point: np.ndarray, triangle: np.ndarray, epsilon: float) -> bool:
    """Return whether a 2D point lies strictly inside a triangle."""

    def orient(left: np.ndarray, right: np.ndarray, sample: np.ndarray) -> float:
        return float(
            (right[0] - left[0]) * (sample[1] - left[1])
            - (right[1] - left[1]) * (sample[0] - left[0])
        )

    values = [
        orient(triangle[0], triangle[1], point),
        orient(triangle[1], triangle[2], point),
        orient(triangle[2], triangle[0], point),
    ]
    return bool(all(value > epsilon for value in values) or all(value < -epsilon for value in values))


def segments_intersect_2d_strict(
    first_left: np.ndarray,
    first_right: np.ndarray,
    second_left: np.ndarray,
    second_right: np.ndarray,
    epsilon: float,
) -> bool:
    """Return whether two 2D segments cross away from their endpoints."""

    def orient(left: np.ndarray, right: np.ndarray, sample: np.ndarray) -> float:
        return float(
            (right[0] - left[0]) * (sample[1] - left[1])
            - (right[1] - left[1]) * (sample[0] - left[0])
        )

    first_a = orient(first_left, first_right, second_left)
    first_b = orient(first_left, first_right, second_right)
    second_a = orient(second_left, second_right, first_left)
    second_b = orient(second_left, second_right, first_right)
    return bool(
        first_a * first_b < -(epsilon * epsilon)
        and second_a * second_b < -(epsilon * epsilon)
    )


def coplanar_triangles_overlap_interior(
    first: np.ndarray,
    second: np.ndarray,
    normal: np.ndarray,
    epsilon: float,
) -> bool:
    """Detect positive-area coplanar overlap while excluding boundary-only contact."""
    drop_axis = int(np.argmax(np.abs(normal)))
    first_2d = np.delete(first, drop_axis, axis=1)
    second_2d = np.delete(second, drop_axis, axis=1)
    first_center = first_2d.mean(axis=0)
    second_center = second_2d.mean(axis=0)
    if point_in_triangle_2d_strict(first_center, second_2d, epsilon):
        return True
    if point_in_triangle_2d_strict(second_center, first_2d, epsilon):
        return True
    for point in first_2d:
        if point_in_triangle_2d_strict(point, second_2d, epsilon):
            return True
    for point in second_2d:
        if point_in_triangle_2d_strict(point, first_2d, epsilon):
            return True
    first_edges = ((0, 1), (1, 2), (2, 0))
    second_edges = ((0, 1), (1, 2), (2, 0))
    return any(
        segments_intersect_2d_strict(
            first_2d[first_left],
            first_2d[first_right],
            second_2d[second_left],
            second_2d[second_right],
            epsilon,
        )
        for first_left, first_right in first_edges
        for second_left, second_right in second_edges
    )


def segment_intersects_triangle_interior(
    start: np.ndarray,
    end: np.ndarray,
    triangle: np.ndarray,
    epsilon: float,
) -> bool:
    """Moller-Trumbore segment test that excludes all boundary-only hits."""
    direction = end - start
    edge_1 = triangle[1] - triangle[0]
    edge_2 = triangle[2] - triangle[0]
    cross_direction = np.cross(direction, edge_2)
    determinant = float(np.dot(edge_1, cross_direction))
    if abs(determinant) <= epsilon:
        return False
    inverse = 1.0 / determinant
    offset = start - triangle[0]
    u = inverse * float(np.dot(offset, cross_direction))
    if u <= epsilon or u >= 1.0 - epsilon:
        return False
    cross_offset = np.cross(offset, edge_1)
    v = inverse * float(np.dot(direction, cross_offset))
    if v <= epsilon or u + v >= 1.0 - epsilon:
        return False
    travel = inverse * float(np.dot(edge_2, cross_offset))
    return bool(epsilon < travel < 1.0 - epsilon)


def triangles_intersect_strict(first: np.ndarray, second: np.ndarray, epsilon: float) -> bool:
    """Detect face-interior intersection; shared vertices or edges do not qualify."""
    first_normal = np.cross(first[1] - first[0], first[2] - first[0])
    second_normal = np.cross(second[1] - second[0], second[2] - second[0])
    first_norm = float(np.linalg.norm(first_normal))
    second_norm = float(np.linalg.norm(second_normal))
    if first_norm <= epsilon or second_norm <= epsilon:
        return False
    unit_first = first_normal / first_norm
    unit_second = second_normal / second_norm
    parallel = float(np.linalg.norm(np.cross(unit_first, unit_second))) <= 1e-7
    plane_distance = max(abs(float(np.dot(point - first[0], unit_first))) for point in second)
    if parallel and plane_distance <= epsilon:
        return coplanar_triangles_overlap_interior(first, second, unit_first, epsilon)
    edges = ((0, 1), (1, 2), (2, 0))
    return bool(
        any(
            segment_intersects_triangle_interior(first[left], first[right], second, epsilon)
            for left, right in edges
        )
        or any(
            segment_intersects_triangle_interior(second[left], second[right], first, epsilon)
            for left, right in edges
        )
    )


def build_other_part_intersection_context(
    vertices: np.ndarray,
    faces: np.ndarray,
    components: list[Component],
) -> dict:
    """Build a dependency-light broad-phase index for strict face intersections."""
    face_to_component = np.zeros(len(faces), dtype=np.int32)
    for component_index, component in enumerate(components, start=1):
        face_to_component[np.asarray(component.global_faces, dtype=np.int64)] = int(component_index)
    active_face_ids = np.flatnonzero(face_to_component > 0).astype(np.int64)
    triangles = vertices[faces[active_face_ids]]
    centers = triangles.mean(axis=1)
    radii = np.linalg.norm(triangles - centers[:, None, :], axis=2).max(axis=1)
    bbox_min = triangles.min(axis=1)
    bbox_max = triangles.max(axis=1)
    model_diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    epsilon = max(model_diagonal * 1e-9, 1e-9)
    positive_radii = radii[radii > epsilon]
    base_radius = float(np.percentile(positive_radii, 10.0)) if len(positive_radii) else epsilon
    base_radius = max(base_radius, epsilon)
    radius_levels = np.maximum(
        0,
        np.floor(np.log2(np.maximum(radii, base_radius) / base_radius)).astype(np.int32),
    )
    radius_buckets = []
    for level in sorted(int(value) for value in np.unique(radius_levels)):
        positions = np.flatnonzero(radius_levels == level).astype(np.int64)
        radius_buckets.append(
            {
                "tree": cKDTree(centers[positions]),
                "active_positions": positions,
                "max_radius": float(radii[positions].max()),
            }
        )
    return {
        "radius_buckets": radius_buckets,
        "active_face_ids": active_face_ids,
        "face_to_component": face_to_component,
        "centers": centers,
        "radii": radii,
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "epsilon": epsilon,
    }


def component_other_part_intersection_evidence(
    vertices: np.ndarray,
    faces: np.ndarray,
    component: Component,
    component_index: int,
    context: dict,
    minimum_faces: int = 2,
) -> dict:
    """Count candidate faces that strictly intersect a different recognized part."""
    hit_faces: list[int] = []
    hit_parts: set[int] = set()
    checked_pairs = 0
    face_to_component = context["face_to_component"]
    active_face_ids = context["active_face_ids"]
    epsilon = float(context["epsilon"])
    for face_index in np.asarray(component.global_faces, dtype=np.int64):
        first = vertices[faces[int(face_index)]]
        first_center = first.mean(axis=0)
        first_radius = float(np.linalg.norm(first - first_center, axis=1).max())
        nearby_positions = []
        for bucket in context["radius_buckets"]:
            local_positions = bucket["tree"].query_ball_point(
                first_center,
                first_radius + float(bucket["max_radius"]) + epsilon,
            )
            nearby_positions.extend(
                int(bucket["active_positions"][int(local_position)])
                for local_position in local_positions
            )
        first_vertex_ids = set(int(value) for value in faces[int(face_index)])
        found = False
        for nearby_position in nearby_positions:
            other_face_index = int(active_face_ids[int(nearby_position)])
            other_component_index = int(face_to_component[other_face_index])
            if other_component_index <= 0 or other_component_index == int(component_index):
                continue
            if first_vertex_ids.intersection(int(value) for value in faces[other_face_index]):
                continue
            if np.any(first.max(axis=0) < context["bbox_min"][int(nearby_position)] - epsilon):
                continue
            if np.any(first.min(axis=0) > context["bbox_max"][int(nearby_position)] + epsilon):
                continue
            checked_pairs += 1
            second = vertices[faces[other_face_index]]
            if triangles_intersect_strict(first, second, epsilon):
                hit_faces.append(int(face_index))
                hit_parts.add(other_component_index)
                found = True
                break
        if found and len(hit_faces) >= int(minimum_faces):
            break
    return {
        "intersecting_other_part_face_count": int(len(hit_faces)),
        "intersecting_other_part_face_indices": hit_faces,
        "intersected_other_part_indices": sorted(hit_parts),
        "minimum_intersecting_faces_for_through": int(minimum_faces),
        "other_part_intersection_gate": bool(len(hit_faces) >= int(minimum_faces)),
        "other_part_intersection_test": "strict_triangle_interior_excluding_shared_vertices_and_edges",
        "intersection_pairs_narrow_phase_tested": int(checked_pairs),
        "intersection_scan_stopped_at_minimum": bool(len(hit_faces) >= int(minimum_faces)),
    }






def component_shared_edges(faces: np.ndarray, components: list[Component]) -> dict[tuple[int, int], dict]:
    face_to_component: dict[int, int] = {}
    for index, component in enumerate(components, start=1):
        for face_index in component.global_faces:
            face_to_component[int(face_index)] = index

    edge_to_faces: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
    for face_index, tri in enumerate(faces):
        if face_index not in face_to_component:
            continue
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge_to_faces[tuple(sorted((int(a), int(b))))].append(face_index)

    adjacency: dict[tuple[int, int], dict] = {}
    for edge, face_indices in edge_to_faces.items():
        comp_ids = sorted({face_to_component[face_index] for face_index in face_indices if face_index in face_to_component})
        if len(comp_ids) < 2:
            continue
        for left_index, left in enumerate(comp_ids):
            for right in comp_ids[left_index + 1 :]:
                key = (left, right)
                record = adjacency.setdefault(key, {"shared_edges": 0, "shared_vertices": set()})
                record["shared_edges"] += 1
                record["shared_vertices"].update(edge)

    for record in adjacency.values():
        record["shared_vertex_count"] = len(record["shared_vertices"])
        del record["shared_vertices"]
    return adjacency
