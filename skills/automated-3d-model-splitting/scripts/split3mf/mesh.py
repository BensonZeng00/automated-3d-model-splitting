from __future__ import annotations

from .common import *
from .project import *


def orient_mesh_faces_consistently(mesh: trimesh.Trimesh) -> dict:
    """Make adjacent face winding consistent without changing mesh geometry.

    Face order, vertex order, and triangle membership remain stable; only the
    order of vertices inside affected triangles may change. This is safe for
    generated single-body inserts and preserves every fit-critical coordinate.
    """
    faces_before = np.asarray(mesh.faces, dtype=np.int64).copy()
    consistent_before = bool(mesh.is_winding_consistent)
    if not consistent_before:
        trimesh.repair.fix_winding(mesh)
    if bool(mesh.is_watertight) and float(mesh.volume) < 0.0:
        mesh.invert()
    faces_after = np.asarray(mesh.faces, dtype=np.int64)
    if not np.array_equal(np.sort(faces_before, axis=1), np.sort(faces_after, axis=1)):
        raise RuntimeError("Winding repair changed face membership or face order")
    consistent_after = bool(mesh.is_winding_consistent)
    return {
        "winding_consistent_before": consistent_before,
        "winding_consistent_after": consistent_after,
        "changed_triangle_winding": int(np.any(faces_before != faces_after, axis=1).sum()),
        "vertices_unchanged": True,
        "triangle_membership_unchanged": True,
    }

def build_local_mesh(
    vertices: np.ndarray, faces: np.ndarray, component: Component
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    global_face_array = faces[component.global_faces]
    global_vertex_ids = np.unique(global_face_array.reshape(-1))
    local_vertices = vertices[global_vertex_ids].copy()
    # Source vertex ids are dense package indices.  A NumPy lookup avoids one
    # Python dictionary call per triangle corner, which dominates large
    # recursive subassemblies.  The returned lookup preserves the old
    # global-to-local information for internal callers without Python objects.
    global_to_local = np.full(len(vertices), -1, dtype=np.int64)
    global_to_local[global_vertex_ids] = np.arange(len(global_vertex_ids), dtype=np.int64)
    remapped = global_to_local[global_face_array]
    if np.any(remapped < 0):
        raise ValueError("local mesh remapping lost a referenced source vertex")
    return local_vertices, remapped, global_to_local, global_vertex_ids


def face_edge_set(face_array: np.ndarray) -> set[tuple[int, int]]:
    """Return undirected indexed edges used by a face array."""
    return {
        tuple(sorted((int(left), int(right))))
        for face in np.asarray(face_array, dtype=np.int64)
        for left, right in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))
    }


def face_edges_among_vertices(
    face_array: np.ndarray,
    vertex_ids: set[int],
    vertex_count: int,
) -> set[tuple[int, int]]:
    """Find source edges whose two endpoints both lie on one cut loop."""
    if not vertex_ids:
        return set()
    allowed = np.zeros(int(vertex_count), dtype=bool)
    allowed[np.fromiter((int(index) for index in vertex_ids), dtype=np.int64)] = True
    result: set[tuple[int, int]] = set()
    faces_array = np.asarray(face_array, dtype=np.int64)
    for left_column, right_column in ((0, 1), (1, 2), (2, 0)):
        edges = faces_array[:, [left_column, right_column]]
        selected = edges[allowed[edges[:, 0]] & allowed[edges[:, 1]]]
        result.update(tuple(sorted((int(left), int(right)))) for left, right in selected)
    return result


def boundary_loops(faces: np.ndarray) -> list[list[int]]:
    """Return edge-complete simple boundary cycles.

    Painted regions can have several boundary cycles touching at one vertex, so
    the boundary graph may have degree 4 or higher. A greedy previous/next walk
    turns that graph into open paths and then falsely caps the path endpoints.
    Decompose every even boundary graph into Euler circuits first, then split
    circuits at repeated vertices into genuine simple cycles.
    """
    edge_count: collections.Counter[tuple[int, int]] = collections.Counter()
    for face in faces:
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            a = int(a)
            b = int(b)
            if a > b:
                a, b = b, a
            edge_count[(a, b)] += 1

    adjacency: dict[int, set[int]] = collections.defaultdict(set)
    unused_edges: set[tuple[int, int]] = set()
    for (a, b), count in edge_count.items():
        if count == 1:
            adjacency[a].add(b)
            adjacency[b].add(a)
            unused_edges.add((a, b))

    loops: list[list[int]] = []

    def split_simple_cycles(closed_trail: list[int]) -> None:
        if len(closed_trail) < 4 or closed_trail[0] != closed_trail[-1]:
            return
        seen: dict[int, int] = {}
        for position, vertex in enumerate(closed_trail[:-1]):
            if vertex not in seen:
                seen[vertex] = position
                continue
            start_position = seen[vertex]
            first = closed_trail[start_position : position + 1]
            second = closed_trail[: start_position + 1] + closed_trail[position + 1 :]
            split_simple_cycles(first)
            split_simple_cycles(second)
            return
        cycle = closed_trail[:-1]
        if len(cycle) >= 3:
            loops.append(cycle)

    while unused_edges:
        start_edge = min(unused_edges)
        start = int(start_edge[0])
        stack = [start]
        circuit: list[int] = []
        while stack:
            current = int(stack[-1])
            candidates = sorted(
                neighbor
                for neighbor in adjacency.get(current, set())
                if tuple(sorted((current, int(neighbor)))) in unused_edges
            )
            if candidates:
                next_vertex = int(candidates[0])
                unused_edges.remove(tuple(sorted((current, next_vertex))))
                stack.append(next_vertex)
            else:
                circuit.append(int(stack.pop()))
        circuit.reverse()
        split_simple_cycles(circuit)

    loops.sort(key=lambda loop: (-len(loop), tuple(loop)))
    return loops


def taubin_smooth_loop(points: np.ndarray, iterations: int, lambda_factor: float, mu_factor: float) -> np.ndarray:
    smoothed = points.astype(np.float64).copy()
    if len(smoothed) < 4:
        return smoothed
    for _ in range(iterations):
        lap = (np.roll(smoothed, 1, axis=0) + np.roll(smoothed, -1, axis=0)) * 0.5 - smoothed
        smoothed = smoothed + lambda_factor * lap
        lap = (np.roll(smoothed, 1, axis=0) + np.roll(smoothed, -1, axis=0)) * 0.5 - smoothed
        smoothed = smoothed + mu_factor * lap
    return smoothed


def average_outward_normal(local_vertices: np.ndarray, local_faces: np.ndarray, component_center: np.ndarray, model_center: np.ndarray) -> np.ndarray:
    tris = local_vertices[local_faces]
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    average = normals.sum(axis=0)
    length = float(np.linalg.norm(average))
    if length < 1e-9:
        average = component_center - model_center
        length = float(np.linalg.norm(average))
    if length < 1e-9:
        average = np.array([0.0, 0.0, 1.0])
        length = 1.0
    average = average / length
    if float(np.dot(average, component_center - model_center)) < 0.0:
        average = -average
    return average


def orthonormal_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = normal / np.linalg.norm(normal)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(reference, normal))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, reference)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)
    return u, v


def project_points(points: np.ndarray, origin: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    rel = points - origin
    return np.column_stack((rel @ u, rel @ v))


def radial_offset_points(
    points: np.ndarray,
    origin: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    normal: np.ndarray,
    offset_mm: float,
) -> np.ndarray:
    if abs(float(offset_mm)) < 1e-9 or len(points) == 0:
        return points.copy()
    rel = points - origin
    xy = np.column_stack((rel @ u, rel @ v))
    normal_s = rel @ normal
    centroid = xy.mean(axis=0)
    directions = xy - centroid
    lengths = np.linalg.norm(directions, axis=1)
    safe_lengths = np.where(lengths > 1e-9, lengths, 1.0)
    directions = directions / safe_lengths[:, None]
    directions[lengths <= 1e-9] = 0.0
    shifted = xy + directions * float(offset_mm)
    return origin + shifted[:, 0, None] * u + shifted[:, 1, None] * v + normal_s[:, None] * normal


def clearance_offsets(clearance_mode: str, fit_clearance_mm: float) -> tuple[float, float]:
    clearance = max(float(fit_clearance_mm), 0.0)
    if clearance_mode == "insert-shrink":
        return clearance, 0.0
    if clearance_mode == "socket-overcut":
        return 0.0, clearance
    if clearance_mode == "split":
        half = clearance * 0.5
        return half, half
    raise ValueError(f"Unsupported clearance mode: {clearance_mode}")


def signed_area(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def point_in_poly(point: np.ndarray, poly: np.ndarray) -> bool:
    x, y = float(point[0]), float(point[1])
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = float(poly[i, 0]), float(poly[i, 1])
        xj, yj = float(poly[j, 0]), float(poly[j, 1])
        crosses = (yi > y) != (yj > y)
        if crosses:
            x_at_y = (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi
            if x < x_at_y:
                inside = not inside
        j = i
    return inside


def point_on_poly_boundary(point: np.ndarray, poly: np.ndarray, eps: float = 1e-7) -> bool:
    p = np.asarray(point, dtype=np.float64)
    for index in range(len(poly)):
        a = poly[index]
        b = poly[(index + 1) % len(poly)]
        ab = b - a
        ap = p - a
        length_sq = float(np.dot(ab, ab))
        if length_sq < eps:
            continue
        cross = abs(float(ab[0] * ap[1] - ab[1] * ap[0]))
        if cross > eps * max(1.0, math.sqrt(length_sq)):
            continue
        dot = float(np.dot(ap, ab))
        if -eps <= dot <= length_sq + eps:
            return True
    return False


def points_in_or_on_poly_batch(points: np.ndarray, poly: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Vectorized polygon membership for boundary-cap triangle samples."""
    points = np.asarray(points, dtype=np.float64)
    poly = np.asarray(poly, dtype=np.float64)
    result = np.zeros(len(points), dtype=bool)
    if not len(points) or len(poly) < 3:
        return result
    a = poly
    b = np.roll(poly, -1, axis=0)
    edge_scale = np.maximum(np.linalg.norm(b - a, axis=1), 1.0)
    for start in range(0, len(points), 512):
        block = points[start : start + 512]
        px = block[:, 0, None]
        py = block[:, 1, None]
        ax = a[None, :, 0]
        ay = a[None, :, 1]
        bx = b[None, :, 0]
        by = b[None, :, 1]
        cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
        dot = (px - ax) * (px - bx) + (py - ay) * (py - by)
        on_boundary = np.any((np.abs(cross) <= eps * edge_scale[None, :]) & (dot <= eps), axis=1)
        denominator = np.where(np.abs(by - ay) > 1e-15, by - ay, 1e-15)
        crossing = ((ay > py) != (by > py)) & (px < (bx - ax) * (py - ay) / denominator + ax)
        inside = np.bitwise_xor.reduce(crossing, axis=1)
        result[start : start + len(block)] = inside | on_boundary
    return result


def triangulate_polygon_ear_clip(points_2d: np.ndarray) -> list[tuple[int, int, int]]:
    """Triangulate a simple polygon with short, distributed boundary ears."""
    points_2d = np.asarray(points_2d, dtype=np.float64)
    count = len(points_2d)
    if count < 3:
        return []
    orientation = 1.0 if signed_area(points_2d) >= 0.0 else -1.0
    previous = np.arange(count, dtype=np.int64) - 1
    previous[0] = count - 1
    following = np.arange(count, dtype=np.int64) + 1
    following[-1] = 0
    alive = np.ones(count, dtype=bool)
    version = np.zeros(count, dtype=np.int64)
    heap: list[tuple[float, float, int, int]] = []
    triangles: list[tuple[int, int, int]] = []
    remaining = count
    scale = max(float(np.ptp(points_2d[:, 0])), float(np.ptp(points_2d[:, 1])), 1.0)
    eps = scale * scale * 1e-12

    def push(index: int) -> None:
        if not alive[index]:
            return
        left = int(previous[index])
        right = int(following[index])
        diagonal_sq = float(np.sum((points_2d[right] - points_2d[left]) ** 2))
        first_edge = points_2d[index] - points_2d[left]
        second_edge = points_2d[right] - points_2d[left]
        local_area = abs(float(first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0]))
        heapq.heappush(heap, (diagonal_sq, -local_area, int(index), int(version[index])))

    for index in range(count):
        push(index)

    while remaining > 3 and heap:
        _diagonal_sq, _negative_area, current, queued_version = heapq.heappop(heap)
        if not alive[current] or queued_version != int(version[current]):
            continue
        left = int(previous[current])
        right = int(following[current])
        a, b, c = points_2d[[left, current, right]]
        cross = float(
            (b[0] - a[0]) * (c[1] - b[1])
            - (b[1] - a[1]) * (c[0] - b[0])
        ) * orientation
        if cross <= eps:
            continue
        candidates = np.flatnonzero(alive)
        candidates = candidates[(candidates != left) & (candidates != current) & (candidates != right)]
        if len(candidates):
            p = points_2d[candidates]
            ab = float(orientation) * ((b[0] - a[0]) * (p[:, 1] - a[1]) - (b[1] - a[1]) * (p[:, 0] - a[0]))
            bc = float(orientation) * ((c[0] - b[0]) * (p[:, 1] - b[1]) - (c[1] - b[1]) * (p[:, 0] - b[0]))
            ca = float(orientation) * ((a[0] - c[0]) * (p[:, 1] - c[1]) - (a[1] - c[1]) * (p[:, 0] - c[0]))
            # Points exactly on an ear edge must not block that ear. Painted
            # 3MF boundaries commonly contain long runs of collinear vertices;
            # treating those points as interior can leave the queue with no
            # removable ear even though the polygon itself is valid.
            if np.any((ab > eps) & (bc > eps) & (ca > eps)):
                continue
        triangles.append((left, current, right))
        alive[current] = False
        following[left] = right
        previous[right] = left
        remaining -= 1
        for neighbor in (left, right):
            version[neighbor] += 1
            push(neighbor)

    if remaining == 3:
        first = int(np.flatnonzero(alive)[0])
        second = int(following[first])
        third = int(following[second])
        triangles.append((first, second, third))
    if len(triangles) != count - 2:
        return []
    return triangles


def triangulate_ordered_loop_3d(
    points: np.ndarray,
    fallback_normal: np.ndarray,
    forbidden_edges: set[tuple[int, int]] | None = None,
) -> tuple[list[tuple[int, int, int]], np.ndarray, dict]:
    """Find a non-self-overlapping projection for a non-planar 3D loop."""
    points = np.asarray(points, dtype=np.float64)
    fallback = np.asarray(fallback_normal, dtype=np.float64)
    centered = points - points.mean(axis=0)
    candidates: list[tuple[str, np.ndarray]] = []

    following = np.roll(points, -1, axis=0)
    newell = np.array(
        [
            np.sum((points[:, 1] - following[:, 1]) * (points[:, 2] + following[:, 2])),
            np.sum((points[:, 2] - following[:, 2]) * (points[:, 0] + following[:, 0])),
            np.sum((points[:, 0] - following[:, 0]) * (points[:, 1] + following[:, 1])),
        ],
        dtype=np.float64,
    )
    candidates.append(("newell", newell))
    try:
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
        for rank, vector in enumerate(vh[::-1]):
            candidates.append((f"pca_axis_{rank}", np.asarray(vector, dtype=np.float64)))
    except np.linalg.LinAlgError:
        pass
    candidates.extend(
        [
            ("fallback", fallback),
            ("axis_x", np.array([1.0, 0.0, 0.0])),
            ("axis_y", np.array([0.0, 1.0, 0.0])),
            ("axis_z", np.array([0.0, 0.0, 1.0])),
        ]
    )

    tried: list[dict] = []
    successes: list[tuple[tuple[float, float], list[tuple[int, int, int]], np.ndarray, str]] = []
    seen_normals: list[np.ndarray] = []
    for name, candidate in candidates:
        length = float(np.linalg.norm(candidate))
        if length <= 1e-12:
            continue
        normal = candidate / length
        if float(np.dot(normal, fallback)) < 0.0:
            normal = -normal
        if any(abs(float(np.dot(normal, previous_normal))) > 0.999999 for previous_normal in seen_normals):
            continue
        seen_normals.append(normal)
        u, v = orthonormal_basis(normal)
        points_2d = project_points(points, points.mean(axis=0), u, v)
        triangles = triangulate_polygon_ear_clip(points_2d)
        if triangles and forbidden_edges:
            triangle_edges = {
                tuple(sorted(edge))
                for triangle in triangles
                for edge in (
                    (triangle[0], triangle[1]),
                    (triangle[1], triangle[2]),
                    (triangle[2], triangle[0]),
                )
            }
            if triangle_edges.intersection(forbidden_edges):
                triangles = []
        record = {"projection": name, "triangles": len(triangles)}
        tried.append(record)
        if len(triangles) != len(points) - 2:
            continue
        triangle_array = np.asarray(triangles, dtype=np.int64)
        triangle_points = points[triangle_array]
        edge_lengths = np.linalg.norm(
            np.concatenate(
                [
                    triangle_points[:, 1] - triangle_points[:, 0],
                    triangle_points[:, 2] - triangle_points[:, 1],
                    triangle_points[:, 0] - triangle_points[:, 2],
                ],
                axis=0,
            ),
            axis=1,
        )
        # Prefer the triangulation without long cross-body chords; those are
        # the direct geometric cause of the former plate/spike artifacts.
        score = (float(edge_lengths.max()), float(np.sum(edge_lengths * edge_lengths)))
        successes.append((score, triangles, normal, name))

    if not successes:
        triangles, greedy_record = triangulate_nonplanar_loop_greedy(points, forbidden_edges)
        return triangles, fit_plane_normal(points, fallback), {
            "status": "ready" if triangles else "projection_and_greedy_failed",
            "projection": "nonplanar_shortest_diagonal",
            "tried": tried,
            **greedy_record,
        }
    successes.sort(key=lambda item: item[0])
    score, triangles, normal, name = successes[0]
    return triangles, normal, {
        "status": "ready",
        "projection": name,
        "max_triangle_edge_mm": score[0],
        "edge_square_sum": score[1],
        "successful_projection_count": len(successes),
        "tried": tried,
    }


def triangulate_nonplanar_loop_greedy(
    points: np.ndarray,
    forbidden_edges: set[tuple[int, int]] | None = None,
) -> tuple[list[tuple[int, int, int]], dict]:
    """Triangulate a spatial loop by repeatedly removing its shortest ear.

    This is the fail-safe for loops whose every planar projection self-crosses.
    It creates no hub vertex and no point outside the source boundary.
    """
    points = np.asarray(points, dtype=np.float64)
    forbidden_edges = forbidden_edges or set()
    count = len(points)
    if count < 3:
        return [], {"greedy_status": "skipped_short_loop"}
    previous = np.arange(count, dtype=np.int64) - 1
    previous[0] = count - 1
    following = np.arange(count, dtype=np.int64) + 1
    following[-1] = 0
    alive = np.ones(count, dtype=bool)
    version = np.zeros(count, dtype=np.int64)
    heap: list[tuple[float, float, int, int]] = []

    def push(index: int) -> None:
        if not alive[index]:
            return
        left = int(previous[index])
        right = int(following[index])
        if tuple(sorted((left, right))) in forbidden_edges:
            return
        diagonal = float(np.linalg.norm(points[right] - points[left]))
        area2 = float(np.linalg.norm(np.cross(points[index] - points[left], points[right] - points[left])))
        # Short diagonals dominate. Prefer a non-degenerate ear on ties.
        heapq.heappush(heap, (diagonal, -area2, int(index), int(version[index])))

    for index in range(count):
        push(index)
    triangles: list[tuple[int, int, int]] = []
    remaining = count
    while remaining > 3 and heap:
        _diagonal, _negative_area, current, queued_version = heapq.heappop(heap)
        if not alive[current] or queued_version != int(version[current]):
            continue
        left = int(previous[current])
        right = int(following[current])
        triangles.append((left, current, right))
        alive[current] = False
        following[left] = right
        previous[right] = left
        remaining -= 1
        for neighbor in (left, right):
            version[neighbor] += 1
            push(neighbor)
    if remaining == 3:
        first = int(np.flatnonzero(alive)[0])
        second = int(following[first])
        third = int(following[second])
        triangles.append((first, second, third))
    if len(triangles) != count - 2:
        return [], {"greedy_status": "incomplete", "triangles": len(triangles)}
    triangle_points = points[np.asarray(triangles, dtype=np.int64)]
    edge_lengths = np.linalg.norm(
        np.concatenate(
            [
                triangle_points[:, 1] - triangle_points[:, 0],
                triangle_points[:, 2] - triangle_points[:, 1],
                triangle_points[:, 0] - triangle_points[:, 2],
            ],
            axis=0,
        ),
        axis=1,
    )
    return triangles, {
        "greedy_status": "ready",
        "triangles": len(triangles),
        "max_triangle_edge_mm": float(edge_lengths.max()) if len(edge_lengths) else 0.0,
        "edge_square_sum": float(np.sum(edge_lengths * edge_lengths)),
        "synthetic_vertex_count": 0,
    }


def group_loops(loop_points_2d: list[np.ndarray]) -> list[list[int]]:
    centroids = [poly.mean(axis=0) for poly in loop_points_2d]
    areas = [abs(signed_area(poly)) for poly in loop_points_2d]
    depths = []
    for index, centroid in enumerate(centroids):
        depth = 0
        for other, poly in enumerate(loop_points_2d):
            if other == index:
                continue
            if point_in_poly(centroid, poly):
                depth += 1
        depths.append(depth)

    groups: list[list[int]] = []
    for index, depth in enumerate(depths):
        if depth % 2 != 0:
            continue
        children = []
        for other, other_depth in enumerate(depths):
            if other == index:
                continue
            if other_depth == depth + 1 and point_in_poly(centroids[other], loop_points_2d[index]):
                children.append(other)
        children.sort(key=lambda i: areas[i], reverse=True)
        groups.append([index] + children)

    if not groups and loop_points_2d:
        largest = int(np.argmax(np.array(areas)))
        groups.append([largest])
    groups.sort(key=lambda group: areas[group[0]], reverse=True)
    return groups


def triangulate_loop_group_with_bridges(
    loops_bottom: list[list[int]],
    loop_points_2d: list[np.ndarray],
    group: list[int],
) -> list[tuple[int, int, int]]:
    """Triangulate one outer loop plus holes while preserving every ring edge.

    An unconstrained Delaunay triangulation can cross a concave painted boundary.
    Filtering those triangles by centroid then leaves missing boundary edges (or
    gives a boundary edge two cap faces).  Bridge each hole to a visible outer
    vertex first, then use the ordered ear clipper, whose contract preserves all
    input boundary edges.
    """
    if not group:
        return []

    outer_index = int(group[0])
    outer_ids = [int(value) for value in loops_bottom[outer_index]]
    outer_points = np.asarray(loop_points_2d[outer_index], dtype=np.float64)
    if len(outer_ids) < 3 or len(outer_ids) != len(outer_points):
        return []
    if signed_area(outer_points) < 0.0:
        outer_ids.reverse()
        outer_points = outer_points[::-1].copy()

    hole_records: list[tuple[list[int], np.ndarray]] = []
    for loop_index in group[1:]:
        hole_ids = [int(value) for value in loops_bottom[int(loop_index)]]
        hole_points = np.asarray(loop_points_2d[int(loop_index)], dtype=np.float64)
        if len(hole_ids) < 3 or len(hole_ids) != len(hole_points):
            return []
        if signed_area(hole_points) > 0.0:
            hole_ids.reverse()
            hole_points = hole_points[::-1].copy()
        hole_records.append((hole_ids, hole_points))

    current_ids = list(outer_ids)
    current_points = [np.asarray(point, dtype=np.float64) for point in outer_points]
    all_rings = [outer_points] + [points for _ids, points in hole_records]
    scale = max(
        float(np.ptp(outer_points[:, 0])),
        float(np.ptp(outer_points[:, 1])),
        1.0,
    )
    epsilon = scale * 1e-10

    def segment_crosses_ring(left: np.ndarray, right: np.ndarray, ring: np.ndarray) -> bool:
        for edge_index in range(len(ring)):
            edge_left = ring[edge_index]
            edge_right = ring[(edge_index + 1) % len(ring)]
            if segments_intersect_2d_strict(left, right, edge_left, edge_right, epsilon):
                return True
        return False

    # Process rightmost holes first so later bridges are less likely to cross an
    # already inserted bridge in clusters of small painted openings.
    hole_records.sort(key=lambda item: float(np.max(item[1][:, 0])), reverse=True)
    for hole_ids, hole_points in hole_records:
        hole_start = int(np.lexsort((hole_points[:, 1], -hole_points[:, 0]))[0])
        bridge_hole_point = hole_points[hole_start]
        candidates: list[tuple[float, int]] = []
        for current_index, candidate in enumerate(current_points):
            distance_sq = float(np.dot(candidate - bridge_hole_point, candidate - bridge_hole_point))
            if distance_sq <= epsilon * epsilon:
                continue
            midpoint = (candidate + bridge_hole_point) * 0.5
            if not (point_in_poly(midpoint, outer_points) or point_on_poly_boundary(midpoint, outer_points)):
                continue
            if any(
                point_in_poly(midpoint, ring) and not point_on_poly_boundary(midpoint, ring)
                for ring in all_rings[1:]
            ):
                continue
            if any(segment_crosses_ring(candidate, bridge_hole_point, ring) for ring in all_rings):
                continue
            current_ring = np.asarray(current_points, dtype=np.float64)
            if segment_crosses_ring(candidate, bridge_hole_point, current_ring):
                continue
            candidates.append((distance_sq, current_index))
        if not candidates:
            return []
        _distance_sq, bridge_outer_index = min(candidates)
        rotated_ids = hole_ids[hole_start:] + hole_ids[:hole_start]
        rotated_points = np.concatenate((hole_points[hole_start:], hole_points[:hole_start]), axis=0)
        outer_id = int(current_ids[bridge_outer_index])
        outer_point = np.asarray(current_points[bridge_outer_index], dtype=np.float64)
        current_ids = (
            current_ids[: bridge_outer_index + 1]
            + rotated_ids
            + [int(rotated_ids[0]), outer_id]
            + current_ids[bridge_outer_index + 1 :]
        )
        current_points = (
            current_points[: bridge_outer_index + 1]
            + [np.asarray(point, dtype=np.float64) for point in rotated_points]
            + [np.asarray(rotated_points[0], dtype=np.float64), outer_point]
            + current_points[bridge_outer_index + 1 :]
        )

    polygon_points = np.asarray(current_points, dtype=np.float64)
    triangles = triangulate_polygon_ear_clip(polygon_points)
    if len(triangles) != len(polygon_points) - 2:
        return []
    mapped: list[tuple[int, int, int]] = []
    for triangle in triangles:
        face = tuple(int(current_ids[int(index)]) for index in triangle)
        if len(set(face)) == 3:
            mapped.append(face)
    return mapped


def triangulate_cap(
    vertices: list[np.ndarray],
    faces: list[list[int]],
    loops_bottom: list[list[int]],
    loop_points_2d: list[np.ndarray],
    loop_groups: list[list[int]],
    inward: np.ndarray,
) -> int:
    cap_faces = 0
    for group in loop_groups:
        all_bottom_ids: list[int] = []
        all_points_2d: list[np.ndarray] = []
        loop_id_ranges = []
        for loop_index in group:
            start = len(all_bottom_ids)
            ids = loops_bottom[loop_index]
            all_bottom_ids.extend(ids)
            all_points_2d.extend(loop_points_2d[loop_index])
            loop_id_ranges.append((loop_index, start, len(all_bottom_ids)))

        if len(all_bottom_ids) < 3:
            continue

        if len(group) == 1:
            # A centroid fan turns even a mildly non-planar local-offset ring
            # into a visible radial starburst.  Triangulate only between the
            # ordered boundary vertices so no synthetic hub or long fan of
            # folded triangles can be introduced.
            added, _record = triangulate_boundary_cap_without_center(
                vertices,
                faces,
                [int(index) for index in all_bottom_ids],
                inward,
            )
            cap_faces += int(added)
            continue

        bridged_triangles = triangulate_loop_group_with_bridges(loops_bottom, loop_points_2d, group)
        for tri in bridged_triangles:
            face = [int(tri[0]), int(tri[1]), int(tri[2])]
            p = np.array([vertices[face[0]], vertices[face[1]], vertices[face[2]]])
            normal = np.cross(p[1] - p[0], p[2] - p[0])
            if float(np.dot(normal, inward)) < 0.0:
                face = [face[0], face[2], face[1]]
            faces.append(face)
            cap_faces += 1
    return cap_faces


def triangulate_boundary_cap_without_center(
    vertices: list[np.ndarray],
    faces: list[list[int]],
    loop: list[int],
    fallback_normal: np.ndarray,
    occupied_edges: set[tuple[int, int]] | None = None,
) -> tuple[int, dict]:
    """Close one source boundary directly without adding inward side walls."""
    if len(loop) < 3:
        return 0, {"method": "skipped_short_loop", "vertices": len(loop)}
    points = np.asarray([vertices[int(index)] for index in loop], dtype=np.float64)
    origin = points.mean(axis=0)
    loop_positions = {int(vertex): position for position, vertex in enumerate(loop)}
    loop_boundary_edges = {
        tuple(sorted((int(loop[position]), int(loop[(position + 1) % len(loop)]))))
        for position in range(len(loop))
    }
    forbidden_edges = {
        tuple(sorted((loop_positions[int(left)], loop_positions[int(right)])))
        for left, right in (occupied_edges or set())
        if int(left) in loop_positions
        and int(right) in loop_positions
        and tuple(sorted((int(left), int(right)))) not in loop_boundary_edges
    }
    valid_triangles, normal, projection_record = triangulate_ordered_loop_3d(
        points,
        fallback_normal,
        forbidden_edges,
    )
    added = 0
    method = "ordered_polygon_ear_clip_projection_search"
    try:
        if not valid_triangles:
            method = "ear_clip_failed"
        for triangle in valid_triangles:
            face = [int(loop[int(triangle[0])]), int(loop[int(triangle[1])]), int(loop[int(triangle[2])])]
            tri_points = np.asarray([vertices[index] for index in face], dtype=np.float64)
            tri_normal = np.cross(tri_points[1] - tri_points[0], tri_points[2] - tri_points[0])
            if float(np.dot(tri_normal, normal)) < 0.0:
                face = [face[0], face[2], face[1]]
            faces.append(face)
            added += 1
    except Exception:
        added = 0
        method = "triangulation_failed"

    plane_offsets = np.abs((points - origin) @ normal)
    return added, {
        "method": method,
        "vertices": len(loop),
        "planarity_max_error_mm": float(plane_offsets.max()) if len(plane_offsets) else 0.0,
        "planarity_mean_error_mm": float(plane_offsets.mean()) if len(plane_offsets) else 0.0,
        "cap_faces": int(added),
        "unsafe_center_fan_used": False,
        "projection_search": projection_record,
    }
