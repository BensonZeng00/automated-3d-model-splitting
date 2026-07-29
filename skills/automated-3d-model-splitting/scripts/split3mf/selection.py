from __future__ import annotations

from .common import *
from .project import *
from .recognition import *
from .mesh import *
from .package_io import *

def choose_body_component(
    components: list[Component],
    strategy: str,
    body_color: str | None,
    body_index: int | None,
    excluded_auto_indices: set[int] | None = None,
    auto_selection_evidence: dict[int, dict] | None = None,
) -> Component | None:
    if not components or strategy == "none":
        return None
    if body_index is not None:
        if body_index < 1 or body_index > len(components):
            raise ValueError(f"--body-index must be between 1 and {len(components)}")
        return components[body_index - 1]
    if body_color:
        candidates = [component for component in components if component.color_code == body_color]
        if not candidates:
            raise ValueError(f"No effective component found with color code {body_color!r}")
        return max(candidates, key=lambda component: component.face_count)
    excluded = {int(index) for index in (excluded_auto_indices or set())}
    eligible = [
        component
        for index, component in enumerate(components, start=1)
        if index not in excluded
    ]
    if not eligible:
        eligible = list(components)
    if strategy == "largest":
        return max(eligible, key=lambda component: component.face_count)
    if strategy == "auto-score":
        candidates = eligible
        candidate_indices = {
            id(component): index
            for index, component in enumerate(components, start=1)
        }
        evidence = auto_selection_evidence or {}
        return max(
            candidates,
            key=lambda component: (
                float(evidence.get(candidate_indices[id(component)], {}).get("body_candidate_score", 0.0)),
                component.face_count,
                component.area,
            ),
        )
    raise ValueError(f"Unknown body strategy: {strategy}")


def component_identity_index(components: list[Component], target: Component | None) -> int | None:
    """Return a one-based index without comparing NumPy-backed dataclass fields."""
    if target is None:
        return None
    return next((index for index, component in enumerate(components, start=1) if component is target), None)


def component_recognition_records(
    vertices: np.ndarray,
    faces: np.ndarray,
    components: list[Component],
    body_component: Component | None,
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
                "role": "body" if component is body_component else "insert",
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


def component_processing_metrics(
    vertices: np.ndarray,
    faces: np.ndarray,
    component: Component,
    model_center: np.ndarray,
    model_bbox_diagonal: float,
) -> dict:
    """Measure whether a painted surface behaves like a one-sided insert or a through region."""
    local_vertices, local_faces, _, _ = build_local_mesh(vertices, faces, component)
    triangles = local_vertices[local_faces]
    raw_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_areas = np.linalg.norm(raw_normals, axis=1)
    safe = np.where(double_areas > 1e-15, double_areas, 1.0)
    normals = raw_normals / safe[:, None]
    centroids = triangles.mean(axis=1)

    radial = centroids - model_center
    flip = np.einsum("ij,ij->i", normals, radial) < 0.0
    normals[flip] *= -1.0
    weights = np.maximum(double_areas, 1e-15)
    total_weight = float(weights.sum())
    weighted_mean = (normals * weights[:, None]).sum(axis=0)
    normal_resultant = float(np.linalg.norm(weighted_mean) / max(total_weight, 1e-15))

    covariance = (normals.T * weights) @ normals / max(total_weight, 1e-15)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        dominant = np.asarray(eigenvectors[:, int(np.argmax(eigenvalues))], dtype=np.float64)
    except np.linalg.LinAlgError:
        dominant = weighted_mean.copy()
    if float(np.linalg.norm(dominant)) <= 1e-12:
        dominant = component.center - model_center
    dominant = dominant / max(float(np.linalg.norm(dominant)), 1e-12)
    if float(np.dot(dominant, weighted_mean)) < 0.0:
        dominant = -dominant

    dots = normals @ dominant
    positive = dots >= 0.35
    negative = dots <= -0.35
    positive_weight = float(weights[positive].sum())
    negative_weight = float(weights[negative].sum())
    directional_weight = positive_weight + negative_weight
    opposing_balance = (
        float(2.0 * min(positive_weight, negative_weight) / directional_weight)
        if directional_weight > 1e-15
        else 0.0
    )

    opposing_separation_mm = 0.0
    if positive_weight > 1e-15 and negative_weight > 1e-15:
        positive_center = (centroids[positive] * weights[positive, None]).sum(axis=0) / positive_weight
        negative_center = (centroids[negative] * weights[negative, None]).sum(axis=0) / negative_weight
        opposing_separation_mm = abs(float(np.dot(positive_center - negative_center, dominant)))

    loops = boundary_loops(local_faces)
    loop_spans = []
    for loop in loops:
        points = local_vertices[np.asarray(loop, dtype=np.int64)]
        if len(points):
            loop_spans.append(float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))))
    max_loop_span_mm = max(loop_spans, default=0.0)
    model_diagonal = max(float(model_bbox_diagonal), 1e-9)
    bbox_diagonal_mm = float(np.linalg.norm(component.bbox_max - component.bbox_min))
    bbox_diagonal_ratio = bbox_diagonal_mm / model_diagonal
    max_loop_span_ratio = max_loop_span_mm / model_diagonal
    opposing_separation_ratio = opposing_separation_mm / model_diagonal

    span_evidence = min(1.0, max_loop_span_ratio / 0.22)
    separation_evidence = min(1.0, opposing_separation_ratio / 0.14)
    size_evidence = min(1.0, bbox_diagonal_ratio / 0.30)
    through_score = float(
        0.34 * opposing_balance
        + 0.22 * (1.0 - normal_resultant)
        + 0.20 * span_evidence
        + 0.14 * separation_evidence
        + 0.10 * size_evidence
    )
    through_gate = bool(
        opposing_balance >= 0.30
        and bbox_diagonal_ratio >= 0.18
        and max_loop_span_ratio >= 0.12
        and opposing_separation_ratio >= 0.05
    )
    return {
        "normal_resultant": normal_resultant,
        "opposing_normal_balance": opposing_balance,
        "opposing_separation_mm": opposing_separation_mm,
        "opposing_separation_ratio": opposing_separation_ratio,
        "bbox_diagonal_mm": bbox_diagonal_mm,
        "bbox_diagonal_ratio": bbox_diagonal_ratio,
        "max_boundary_loop_span_mm": max_loop_span_mm,
        "max_boundary_loop_span_ratio": max_loop_span_ratio,
        "through_score": through_score,
        "through_gate": through_gate,
    }


def classify_component_processing_modes(
    vertices: np.ndarray,
    faces: np.ndarray,
    components: list[Component],
    body_component: Component | None,
    model_center: np.ndarray,
    global_mode: str,
    overrides: dict[int, str],
    accept_ambiguous_inward: bool,
    precomputed_structural_evidence: dict[int, dict] | None = None,
) -> dict[int, dict]:
    model_bbox_diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    structural_by_part = (
        precomputed_structural_evidence
        if precomputed_structural_evidence is not None
        else structural_interface_evidence(faces, components)
    )
    classifications: dict[int, dict] = {}
    processing_metric_keys = {
        "normal_resultant",
        "opposing_normal_balance",
        "opposing_separation_mm",
        "opposing_separation_ratio",
        "bbox_diagonal_mm",
        "bbox_diagonal_ratio",
        "max_boundary_loop_span_mm",
        "max_boundary_loop_span_ratio",
        "through_score",
        "through_gate",
    }
    for index, component in enumerate(components, start=1):
        precomputed = structural_by_part.get(index, {})
        if processing_metric_keys.issubset(precomputed):
            metrics = {
                key: precomputed[key]
                for key in processing_metric_keys
            }
        else:
            metrics = component_processing_metrics(
                vertices,
                faces,
                component,
                model_center,
                model_bbox_diagonal,
            )
        metrics.update(precomputed)
        if component is body_component:
            selected_mode = "body"
            suggested_mode = "body"
            status = "body_root"
            confidence = 1.0
        else:
            score = float(metrics["through_score"])
            metrics["structural_separator_evidence"] = bool(
                metrics["through_gate"]
                and score >= 0.58
                and metrics.get("large_structural_interface_gate", False)
            )
            selected_mode = "inward"
            suggested_mode = "inward"
            status = (
                "explicit_inward_override"
                if index in overrides
                else (
                    "explicit_global_inward"
                    if global_mode == "inward"
                    else "inward_only_structural_evidence_recorded"
                )
            )
            confidence = 1.0
        classifications[index] = {
            "part_index": index,
            "selected_processing_mode": selected_mode,
            "suggested_processing_mode": suggested_mode,
            "processing_mode_status": status,
            "processing_mode_confidence": float(max(0.0, min(1.0, confidence))),
            "processing_mode_evidence": metrics,
        }
    return classifications


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


def structural_interface_evidence(
    faces: np.ndarray,
    components: list[Component],
    min_shared_edges: int = 32,
    min_neighbor_face_ratio: float = 0.25,
) -> dict[int, dict]:
    """Find large interfaces to structural peers instead of decorative inserts."""
    adjacency = component_shared_edges(faces, components)
    evidence: dict[int, dict] = {}
    for component_index, component in enumerate(components, start=1):
        minimum_neighbor_faces = max(1, int(math.ceil(component.face_count * float(min_neighbor_face_ratio))))
        interfaces = []
        for (left, right), record in sorted(adjacency.items()):
            if component_index not in (left, right):
                continue
            neighbor_index = right if left == component_index else left
            neighbor = components[int(neighbor_index) - 1]
            shared_edges = int(record.get("shared_edges", 0))
            qualifies = bool(
                shared_edges >= int(min_shared_edges)
                and neighbor.face_count >= minimum_neighbor_faces
            )
            interfaces.append(
                {
                    "neighbor_index": int(neighbor_index),
                    "neighbor_faces": int(neighbor.face_count),
                    "neighbor_face_ratio": float(neighbor.face_count / max(component.face_count, 1)),
                    "shared_edges": shared_edges,
                    "shared_vertices": int(record.get("shared_vertex_count", 0)),
                    "qualifies_as_large_structural_interface": qualifies,
                }
            )
        large_interfaces = [record for record in interfaces if record["qualifies_as_large_structural_interface"]]
        evidence[component_index] = {
            "large_structural_interface_count": int(len(large_interfaces)),
            "minimum_large_structural_interfaces_for_through": 2,
            "large_structural_interface_gate": bool(len(large_interfaces) >= 2),
            "large_structural_interfaces": large_interfaces,
            "all_shared_interfaces": interfaces,
            "large_interface_min_shared_edges": int(min_shared_edges),
            "large_interface_min_neighbor_face_ratio": float(min_neighbor_face_ratio),
            "large_interface_min_neighbor_faces": int(minimum_neighbor_faces),
        }
    return evidence


def shell_removal_evidence(
    vertices: np.ndarray,
    faces: np.ndarray,
    component: Component,
    min_faces: int,
    balanced_region_ratio: float = 0.50,
) -> dict:
    """Measure whether removing one component separates meaningful shell bodies."""
    all_faces = np.arange(len(faces), dtype=np.int64)
    remainder = all_faces[~np.isin(all_faces, component.global_faces, assume_unique=False)]
    groups = connected_face_regions(vertices, faces, remainder)
    meaningful_min = max(32, min(max(int(min_faces), 32), int(max(len(faces), 1) * 0.01)))
    meaningful = [group for group in groups if len(group) >= meaningful_min]
    sizes = sorted((int(len(group)) for group in meaningful), reverse=True)
    two_largest_balance = float(sizes[1] / max(sizes[0], 1)) if len(sizes) >= 2 else 0.0
    return {
        "shell_split_gate": bool(len(sizes) >= 2),
        "shell_split_meaningful_region_count": int(len(sizes)),
        "shell_split_meaningful_region_faces": sizes,
        "shell_split_meaningful_min_faces": int(meaningful_min),
        "two_largest_region_balance": two_largest_balance,
        "minimum_balanced_region_ratio_for_body_exclusion": float(balanced_region_ratio),
        "balanced_two_region_separator": bool(
            len(sizes) == 2 and two_largest_balance >= float(balanced_region_ratio)
        ),
    }


def body_selection_separator_evidence(
    vertices: np.ndarray,
    faces: np.ndarray,
    components: list[Component],
    model_center: np.ndarray,
    min_faces: int,
) -> dict[int, dict]:
    """Identify strong separators before an automatic body is selected."""
    model_bbox_diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    maximum_component_faces = max((component.face_count for component in components), default=1)
    interface_by_part = structural_interface_evidence(faces, components)
    evidence: dict[int, dict] = {}
    for index, component in enumerate(components, start=1):
        metrics = component_processing_metrics(
            vertices,
            faces,
            component,
            model_center,
            model_bbox_diagonal,
        )
        record = {
            **metrics,
            **interface_by_part[index],
            "shell_split_gate": False,
            "shell_split_meaningful_region_count": 0,
            "shell_split_meaningful_region_faces": [],
            "two_largest_region_balance": 0.0,
            "balanced_two_region_separator": False,
        }
        if metrics["through_gate"] and float(metrics["through_score"]) >= 0.58:
            record.update(shell_removal_evidence(vertices, faces, component, min_faces))
        record["exclude_from_automatic_body"] = bool(
            record["through_gate"]
            and float(record["through_score"]) >= 0.58
            and record["balanced_two_region_separator"]
            and record["large_structural_interface_gate"]
        )
        size_score = float(component.face_count / max(maximum_component_faces, 1))
        through_likelihood = float(record["through_score"]) if record["through_gate"] else 0.0
        separator_strength = float(
            through_likelihood
            * float(record.get("two_largest_region_balance", 0.0))
            * min(float(record.get("large_structural_interface_count", 0)) / 3.0, 1.0)
        )
        record["body_candidate_size_score"] = size_score
        record["body_candidate_through_penalty"] = through_likelihood
        record["body_candidate_separator_penalty"] = separator_strength
        record["body_candidate_score"] = float(
            size_score
            - 2.0 * separator_strength
        )
        evidence[index] = record
    return evidence




class BodySelector:
    """Select the root body while keeping structural evidence separate from geometry mode."""

    choose = staticmethod(choose_body_component)
    separator_evidence = staticmethod(body_selection_separator_evidence)
    classify_processing = staticmethod(classify_component_processing_modes)
