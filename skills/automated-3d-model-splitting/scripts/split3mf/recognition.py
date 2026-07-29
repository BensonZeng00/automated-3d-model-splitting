from __future__ import annotations

from .common import *
from .project import *

def representative_color_token(global_face_ids: np.ndarray, display_colors: list[str]) -> str:
    counts = collections.Counter(str(display_colors[int(face_id)]) for face_id in global_face_ids)
    return min(
        counts,
        key=lambda token: (-int(counts[token]), COLOR_ORDER.get(token, 99), token),
    )


def triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tris = vertices[faces]
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    return np.linalg.norm(normals, axis=1) * 0.5


def fibonacci_view_directions(count: int) -> np.ndarray:
    """Return deterministic approximately uniform directions on a sphere."""
    count = max(int(count), 6)
    indices = np.arange(count, dtype=np.float64)
    z = 1.0 - 2.0 * (indices + 0.5) / float(count)
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = indices * (np.pi * (3.0 - np.sqrt(5.0)))
    return np.column_stack((radius * np.cos(theta), radius * np.sin(theta), z))


def exterior_visibility_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(reference, direction))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    u = np.cross(reference, direction)
    u /= max(float(np.linalg.norm(u)), 1e-12)
    v = np.cross(direction, u)
    v /= max(float(np.linalg.norm(v)), 1e-12)
    return u, v


def exterior_visible_face_mask(
    vertices: np.ndarray,
    faces: np.ndarray,
    view_count: int = 32,
    depth_map_resolution: int = 768,
    depth_tolerance_mm: float = 0.08,
) -> tuple[np.ndarray, dict]:
    """Approximate direct exterior visibility with deterministic depth maps.

    Dense vendor-painted meshes have triangles at or below depth-map pixel
    scale, so centroid z-buffering is sufficient for recognition while avoiding
    ray acceleration dependencies. A face is exterior-visible when it is at the
    front depth of at least one outside view, within the configured tolerance.
    """
    view_count = max(int(view_count), 6)
    resolution = max(int(depth_map_resolution), 64)
    tolerance = max(float(depth_tolerance_mm), 0.0)
    centroids = vertices[faces].mean(axis=1)
    visible = np.zeros(len(faces), dtype=bool)
    per_view_visible = []
    for direction in fibonacci_view_directions(view_count):
        u, v = exterior_visibility_basis(direction)
        projected_x = centroids @ u
        projected_y = centroids @ v
        depth = centroids @ direction
        min_x = float(projected_x.min())
        min_y = float(projected_y.min())
        span = max(
            float(projected_x.max()) - min_x,
            float(projected_y.max()) - min_y,
            1e-9,
        )
        scale = (resolution - 1) / span
        pixel_x = np.clip(
            ((projected_x - min_x) * scale).astype(np.int64),
            0,
            resolution - 1,
        )
        pixel_y = np.clip(
            ((projected_y - min_y) * scale).astype(np.int64),
            0,
            resolution - 1,
        )
        keys = pixel_y * resolution + pixel_x
        front_depth = np.full(resolution * resolution, -np.inf, dtype=np.float64)
        np.maximum.at(front_depth, keys, depth)
        view_visible = depth >= front_depth[keys] - tolerance
        visible |= view_visible
        per_view_visible.append(int(np.count_nonzero(view_visible)))
    return visible, {
        "profile": "exterior-visible",
        "method": "multi_view_centroid_depth_map",
        "view_count": view_count,
        "depth_map_resolution": resolution,
        "depth_tolerance_mm": tolerance,
        "visible_faces": int(np.count_nonzero(visible)),
        "occluded_faces": int(np.count_nonzero(~visible)),
        "visible_ratio": float(np.mean(visible)) if len(visible) else 0.0,
        "per_view_visible_faces": per_view_visible,
    }


def recognition_colors_from_exterior(
    colors: list[str],
    visible_faces: np.ndarray,
    body_color_override: str | None = None,
) -> tuple[list[str], dict]:
    """Keep exterior paint and neutralize occluded paint for recognition only."""
    color_array = np.asarray(colors, dtype=object)
    if len(color_array) != len(visible_faces):
        raise ValueError("exterior visibility mask does not match source face count")
    if body_color_override:
        base_color = str(body_color_override)
        base_source = "explicit_body_color"
    elif np.any(color_array == "DEFAULT"):
        base_color = "DEFAULT"
        base_source = "source_default_token"
    else:
        visible_colors = color_array[visible_faces]
        candidates = visible_colors if len(visible_colors) else color_array
        values, counts = np.unique(candidates, return_counts=True)
        base_color = str(values[int(np.argmax(counts))])
        base_source = "largest_exterior_color"
    recognition = color_array.copy()
    reassigned = (~visible_faces) & (recognition != base_color)
    original_tokens, original_counts = np.unique(
        recognition[reassigned], return_counts=True
    )
    recognition[~visible_faces] = base_color
    return recognition.tolist(), {
        "base_color_token": base_color,
        "base_color_source": base_source,
        "reassigned_occluded_faces": int(np.count_nonzero(reassigned)),
        "already_base_occluded_faces": int(
            np.count_nonzero((~visible_faces) & (color_array == base_color))
        ),
        "reassigned_by_source_token": [
            {"raw_color_token": str(token), "faces": int(count)}
            for token, count in sorted(
                zip(original_tokens.tolist(), original_counts.tolist()),
                key=lambda item: (-int(item[1]), str(item[0])),
            )
        ],
    }


def connected_components_by_color(faces: np.ndarray, colors: list[str]) -> list[np.ndarray]:
    dsu = DSU(len(faces))
    first_edge_face: dict[tuple[str, int, int], int] = {}
    for face_index, (face, color) in enumerate(zip(faces, colors)):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            if a > b:
                a, b = b, a
            key = (color, int(a), int(b))
            previous = first_edge_face.get(key)
            if previous is None:
                first_edge_face[key] = face_index
            else:
                dsu.union(face_index, previous)

    groups: dict[int, list[int]] = collections.defaultdict(list)
    for face_index in range(len(faces)):
        groups[dsu.find(face_index)].append(face_index)
    return [np.array(indices, dtype=np.int64) for indices in groups.values()]


def material_identity(color_code: str) -> tuple[str, object]:
    """Return the physical material identity, not the vendor paint token."""
    info = COLOR_INFO.get(str(color_code), {})
    slot = info.get("filament_slot")
    if slot is not None:
        return ("filament_slot", int(slot))
    color_hex = str(info.get("hex", "")).upper()
    if color_hex:
        return ("color_hex", color_hex)
    return ("paint_token", str(color_code))


def connected_face_regions(
    vertices: np.ndarray,
    faces: np.ndarray,
    global_face_ids: np.ndarray,
) -> list[np.ndarray]:
    """Split an arbitrary source-face subset without considering paint color."""
    global_face_ids = np.asarray(global_face_ids, dtype=np.int64)
    if not len(global_face_ids):
        return []
    shell = trimesh.Trimesh(
        vertices=vertices,
        faces=faces[global_face_ids],
        process=False,
    )
    local_groups = trimesh.graph.connected_components(
        shell.face_adjacency,
        nodes=np.arange(len(global_face_ids), dtype=np.int64),
        min_len=1,
        engine="scipy",
    )
    groups = [global_face_ids[np.asarray(group, dtype=np.int64)] for group in local_groups]
    groups.sort(key=lambda group: (-len(group), int(group.min()) if len(group) else -1))
    return groups




def summarize_components(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: list[str],
    groups: Iterable[np.ndarray],
    min_faces: int,
    display_colors: list[str] | None = None,
) -> tuple[list[Component], list[dict]]:
    areas = triangle_areas(vertices, faces)
    components: list[Component] = []
    ignored: list[dict] = []
    for group in groups:
        color = representative_color_token(group, display_colors or colors)
        group_faces = faces[group]
        points = vertices[group_faces.reshape(-1)]
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        center = points.mean(axis=0)
        area = float(areas[group].sum())
        record = {
            "color_code": color,
            "faces": int(len(group)),
            "area_mm2": area,
            "bbox_min": bbox_min.round(6).tolist(),
            "bbox_max": bbox_max.round(6).tolist(),
        }
        if len(group) >= min_faces:
            components.append(
                Component(
                    color_code=color,
                    global_faces=group,
                    face_count=int(len(group)),
                    area=area,
                    bbox_min=bbox_min,
                    bbox_max=bbox_max,
                    center=center,
                )
            )
        else:
            ignored.append(record)

    components.sort(key=lambda c: (COLOR_ORDER.get(c.color_code, 99), -c.face_count))
    ignored.sort(key=lambda r: (COLOR_ORDER.get(r["color_code"], 99), -r["faces"]))
    return components, ignored


def make_component_from_global_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    areas: np.ndarray,
    global_faces: np.ndarray,
    color_code: str,
) -> Component:
    group_faces = faces[global_faces]
    points = vertices[group_faces.reshape(-1)]
    return Component(
        color_code=color_code,
        global_faces=np.array(global_faces, dtype=np.int64),
        face_count=int(len(global_faces)),
        area=float(areas[global_faces].sum()),
        bbox_min=points.min(axis=0),
        bbox_max=points.max(axis=0),
        center=points.mean(axis=0),
    )


def tiny_group_record(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: list[str],
    areas: np.ndarray,
    group: np.ndarray,
    fragment_id: int,
) -> dict:
    group_faces = faces[group]
    points = vertices[group_faces.reshape(-1)]
    color = colors[int(group[0])]
    return {
        "fragment_id": int(fragment_id),
        "color_code": color,
        "faces": int(len(group)),
        "area_mm2": float(areas[group].sum()),
        "bbox_min": points.min(axis=0).round(6).tolist(),
        "bbox_max": points.max(axis=0).round(6).tolist(),
        "center": points.mean(axis=0).round(6).tolist(),
    }


def bbox_distance_sq(a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray) -> float:
    gap = np.maximum(0.0, np.maximum(a_min - b_max, b_min - a_max))
    return float(np.dot(gap, gap))


def edge_key_from_vertices(a: int, b: int) -> tuple[int, int]:
    a = int(a)
    b = int(b)
    return (a, b) if a <= b else (b, a)


def merge_tiny_groups_into_components(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: list[str],
    groups: Iterable[np.ndarray],
    min_faces: int,
    display_colors: list[str] | None = None,
) -> tuple[list[Component], list[dict], list[dict]]:
    display_colors = display_colors or colors
    areas = triangle_areas(vertices, faces)
    effective_groups: list[np.ndarray] = []
    tiny_groups: list[np.ndarray] = []
    for group in groups:
        if len(group) >= min_faces:
            effective_groups.append(np.array(group, dtype=np.int64))
        else:
            tiny_groups.append(np.array(group, dtype=np.int64))

    components = [
        make_component_from_global_faces(
            vertices,
            faces,
            areas,
            group,
            representative_color_token(group, display_colors),
        )
        for group in effective_groups
    ]
    components.sort(key=lambda c: (COLOR_ORDER.get(c.color_code, 99), -c.face_count))
    tiny_groups.sort(
        key=lambda group: (COLOR_ORDER.get(representative_color_token(group, display_colors), 99), -len(group))
    )
    if not components:
        ignored = [
            tiny_group_record(vertices, faces, display_colors, areas, group, fragment_id)
            for fragment_id, group in enumerate(tiny_groups, start=1)
        ]
        return [], ignored, []

    edge_to_component_slots: dict[tuple[int, int], set[int]] = collections.defaultdict(set)
    for slot, component in enumerate(components):
        for face_index in component.global_faces:
            tri = faces[int(face_index)]
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                edge_to_component_slots[edge_key_from_vertices(a, b)].add(slot)

    assigned_faces: list[list[np.ndarray]] = [[] for _ in components]
    merged_records: list[dict] = []
    ignored: list[dict] = []

    for fragment_id, group in enumerate(tiny_groups, start=1):
        record = tiny_group_record(vertices, faces, display_colors, areas, group, fragment_id)
        shared_counts: collections.Counter[int] = collections.Counter()
        for face_index in group:
            tri = faces[int(face_index)]
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                for slot in edge_to_component_slots.get(edge_key_from_vertices(a, b), set()):
                    shared_counts[int(slot)] += 1

        if shared_counts:
            assigned_slot, shared_edges = max(
                shared_counts.items(),
                key=lambda item: (int(item[1]), components[int(item[0])].face_count),
            )
            method = "shared_edge"
            nearest_distance_mm = 0.0
        else:
            fragment_min = np.array(record["bbox_min"], dtype=np.float64)
            fragment_max = np.array(record["bbox_max"], dtype=np.float64)
            assigned_slot, nearest_component = min(
                enumerate(components),
                key=lambda item: (
                    bbox_distance_sq(fragment_min, fragment_max, item[1].bbox_min, item[1].bbox_max),
                    -item[1].face_count,
                ),
            )
            nearest_distance_sq = bbox_distance_sq(fragment_min, fragment_max, nearest_component.bbox_min, nearest_component.bbox_max)
            shared_edges = 0
            method = "nearest_bbox"
            nearest_distance_mm = math.sqrt(float(nearest_distance_sq))

        assigned_faces[int(assigned_slot)].append(group)
        assigned_component = components[int(assigned_slot)]
        record.update(
            {
                "assigned_component_index": int(assigned_slot) + 1,
                "assigned_component_color_code": assigned_component.color_code,
                "assigned_component_faces_before_merge": int(assigned_component.face_count),
                "assignment_method": method,
                "shared_edges_to_assigned_component": int(shared_edges),
                "nearest_bbox_distance_mm": float(nearest_distance_mm),
            }
        )
        merged_records.append(record)

    merged_components = []
    for slot, component in enumerate(components):
        if assigned_faces[slot]:
            merged_global_faces = np.concatenate([component.global_faces, *assigned_faces[slot]])
        else:
            merged_global_faces = component.global_faces
        merged_components.append(
            make_component_from_global_faces(
                vertices,
                faces,
                areas,
                merged_global_faces,
                component.color_code,
            )
        )

    return merged_components, ignored, merged_records


def component_owned_face_colors(
    source_colors: list[str],
    components: list[Component],
) -> list[str]:
    """Use effective component ownership for recursive geometry materials.

    Raw paint tokens remain in recognition provenance. Once a tiny region is
    merged into an effective component, however, its recursive mesh faces must
    carry that component's resolved material so later parent-emitted 3MF inputs
    can be rebound to the same component tree without inventing an extra slot.
    """
    result = np.asarray(source_colors, dtype=object).copy()
    ownership = np.full(len(result), -1, dtype=np.int64)
    for component_index, component in enumerate(components, start=1):
        face_ids = np.asarray(component.global_faces, dtype=np.int64)
        if np.any(face_ids < 0) or np.any(face_ids >= len(result)):
            raise ValueError(
                f"P{component_index:02d} contains an out-of-range source face"
            )
        overlapping = ownership[face_ids] >= 0
        if np.any(overlapping):
            face_id = int(face_ids[np.flatnonzero(overlapping)[0]])
            raise ValueError(
                f"source face {face_id} belongs to more than one effective component"
            )
        ownership[face_ids] = int(component_index)
        result[face_ids] = str(component.color_code)
    return [str(value) for value in result.tolist()]




class PartRecognizer:
    """Expose the deterministic exterior-paint recognition operations as one service."""

    visible_mask = staticmethod(exterior_visible_face_mask)
    exterior_colors = staticmethod(recognition_colors_from_exterior)
    connected_components = staticmethod(connected_components_by_color)
    merge_tiny = staticmethod(merge_tiny_groups_into_components)
    summarize = staticmethod(summarize_components)
