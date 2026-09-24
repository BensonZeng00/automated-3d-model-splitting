from __future__ import annotations

from .cap_template import progressive_boundary_deformation
from .common import *
from .project import *
from .winding import fix_winding_indexed


def fit_orientation_preserving_affine(
    source_points: np.ndarray,
    target_points: np.ndarray,
    determinant_epsilon: float = 1e-9,
) -> tuple[np.ndarray, dict]:
    """Fit a stable row-vector affine map without allowing reflection.

    Painted interfaces are commonly planar or nearly planar.  An unconstrained
    3-D least-squares fit then leaves the transform's normal-axis coefficient
    underdetermined and may return an arbitrary negative determinant even when
    the requested boundary motion is a harmless inward translation.  Centered
    ridge fits preserve the identity transform in that unobserved direction;
    the smallest regularization that restores positive orientation is chosen.
    """
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("affine source and target points must be matching Nx3 arrays")
    if len(source) < 3:
        raise ValueError("affine fitting requires at least three point pairs")

    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    centered_source = source - source_center
    centered_target = target - target_center
    covariance = centered_source.T @ centered_source
    cross_covariance = centered_source.T @ centered_target
    scale = max(float(np.trace(covariance)) / 3.0, 1.0)

    candidates: list[tuple[float, np.ndarray, float, float]] = []
    for relative_weight in (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0):
        weight = float(relative_weight * scale)
        if weight <= 0.0:
            linear, _residuals, _rank, _singular = np.linalg.lstsq(
                centered_source,
                centered_target,
                rcond=None,
            )
        else:
            linear = np.linalg.solve(
                covariance + np.eye(3, dtype=np.float64) * weight,
                cross_covariance + np.eye(3, dtype=np.float64) * weight,
            )
        translation = target_center - source_center @ linear
        matrix = np.vstack((linear, translation))
        determinant = float(np.linalg.det(linear))
        predicted = np.column_stack(
            (source, np.ones(len(source), dtype=np.float64))
        ) @ matrix
        maximum_error = float(np.linalg.norm(predicted - target, axis=1).max())
        candidates.append((relative_weight, matrix, determinant, maximum_error))
        if determinant > float(determinant_epsilon):
            return matrix, {
                "strategy": "centered_orientation_preserving_affine",
                "relative_regularization": float(relative_weight),
                "determinant": determinant,
                "maximum_fit_error_mm": maximum_error,
                "candidate_count": int(len(candidates)),
            }

    best = max(candidates, key=lambda item: item[2])
    raise ValueError(
        "orientation-preserving affine fit could not be found: "
        f"best_determinant={best[2]:.9f}, "
        f"best_fit_error_mm={best[3]:.9f}"
    )


def _point_inside_closed_triangle_shell(
    point: np.ndarray,
    triangles: np.ndarray,
) -> bool:
    """Classify one point by the shell's absolute generalized winding number."""

    vectors = np.asarray(triangles, dtype=np.float64) - np.asarray(
        point, dtype=np.float64
    ).reshape((1, 1, 3))
    a, b, c = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    lengths = np.linalg.norm(vectors, axis=2)
    numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
    denominator = (
        lengths[:, 0] * lengths[:, 1] * lengths[:, 2]
        + np.einsum("ij,ij->i", a, b) * lengths[:, 2]
        + np.einsum("ij,ij->i", b, c) * lengths[:, 0]
        + np.einsum("ij,ij->i", c, a) * lengths[:, 1]
    )
    solid_angle = float(np.sum(2.0 * np.arctan2(numerator, denominator)))
    return bool(abs(solid_angle) > 2.0 * np.pi)


def watertight_component_orientation_audit(mesh: trimesh.Trimesh) -> dict:
    """Measure material-correct orientation for every disconnected closed shell.

    ``mesh.volume`` sums all shells, so one large positive body can hide a
    small inverted tetrahedral fragment. Slicers then render that fragment as
    a dark crack even though the complete object is watertight and has
    consistent shared-edge winding. A negative shell nested inside a positive
    outer shell is different: it is a correctly oriented enclosed cavity, not
    an inverted fragment. Expected winding therefore alternates with shell
    containment depth.
    """

    faces = np.asarray(mesh.faces, dtype=np.int64)
    if not len(faces):
        return {
            "component_count": 0,
            "inward_closed_component_count": 0,
            "inward_closed_component_ids": [],
            "component_signed_volumes_mm3": [],
            "all_closed_components_outward": True,
        }
    if not bool(mesh.is_watertight):
        return {
            "component_count": None,
            "inward_closed_component_count": None,
            "inward_closed_component_ids": [],
            "component_signed_volumes_mm3": [],
            "all_closed_components_outward": None,
            "skipped_reason": "mesh_not_watertight",
        }
    components = trimesh.graph.connected_components(
        np.asarray(mesh.face_adjacency, dtype=np.int64),
        nodes=np.arange(len(faces), dtype=np.int64),
        min_len=1,
    )
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = vertices[faces]
    signed_face_volumes = np.einsum(
        "ij,ij->i",
        triangles[:, 0],
        np.cross(triangles[:, 1], triangles[:, 2]),
    ) / 6.0
    signed_volumes = np.asarray(
        [float(np.sum(signed_face_volumes[np.asarray(group, dtype=np.int64)])) for group in components],
        dtype=np.float64,
    )
    scale = max(float(np.linalg.norm(np.ptp(vertices, axis=0))), 1.0)
    tolerance = max(scale ** 3 * 1e-15, 1e-15)
    component_triangles = [
        triangles[np.asarray(group, dtype=np.int64)] for group in components
    ]
    component_bounds = [
        np.asarray(
            [item.reshape((-1, 3)).min(axis=0), item.reshape((-1, 3)).max(axis=0)],
            dtype=np.float64,
        )
        for item in component_triangles
    ]
    from .print_tolerance import current
    component_areas = np.asarray([
        np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum() / 2
        for t in component_triangles
    ])
    micro_ids = set(np.flatnonzero(component_areas <= current().micro_area_mm2).tolist())
    # Always inspect the principal shell, even for a deliberately tiny part.
    micro_ids.discard(int(np.argmax(component_areas)))
    containment_depths: list[int] = []
    for component_id, shell_triangles in enumerate(component_triangles):
        if component_id in micro_ids:
            containment_depths.append(-1)
            continue
        cross = np.cross(
            shell_triangles[:, 1] - shell_triangles[:, 0],
            shell_triangles[:, 2] - shell_triangles[:, 0],
        )
        largest_face = int(np.argmax(np.linalg.norm(cross, axis=1)))
        sample = np.mean(shell_triangles[largest_face], axis=0)
        # Containment depth must not change merely because this shell's faces
        # are flipped.  Nudging along the signed face normal made a repair
        # oscillate when a disconnected Boolean cavity sat nearly coincident
        # with another shell.  Use an orientation-invariant geometric ray from
        # the component bounds centre to the selected surface point instead.
        shell_bounds = component_bounds[component_id]
        direction = sample - np.mean(shell_bounds, axis=0)
        direction_length = float(np.linalg.norm(direction))
        if direction_length <= 1e-15:
            direction = np.asarray(cross[largest_face], dtype=np.float64)
            dominant_axis = int(np.argmax(np.abs(direction)))
            if direction[dominant_axis] < 0.0:
                direction = -direction
            direction_length = float(np.linalg.norm(direction))
        if direction_length > 0.0:
            sample = sample + direction / direction_length * max(
                scale * 1e-9,
                1e-9,
            )
        depth = 0
        for container_id, container_triangles in enumerate(component_triangles):
            if container_id == component_id or container_id in micro_ids:
                continue
            bounds = component_bounds[container_id]
            bounds_tolerance = max(scale * 1e-10, 1e-10)
            if np.any(sample < bounds[0] - bounds_tolerance) or np.any(
                sample > bounds[1] + bounds_tolerance
            ):
                continue
            if _point_inside_closed_triangle_shell(sample, container_triangles):
                depth += 1
        containment_depths.append(int(depth))

    expected_signs = np.asarray(
        [-1.0 if depth % 2 else 1.0 for depth in containment_depths],
        dtype=np.float64,
    )
    checked = np.asarray([i not in micro_ids for i in range(len(components))])
    expected_signs[~checked] = np.where(signed_volumes[~checked] < 0, -1.0, 1.0)
    incorrect_ids = np.flatnonzero(
        checked & (signed_volumes * expected_signs < -tolerance)
    ).astype(int).tolist()
    negative_ids = np.flatnonzero(signed_volumes < -tolerance).astype(int).tolist()
    return {
        "component_count": int(len(components)),
        "inward_closed_component_count": int(len(incorrect_ids)),
        "inward_closed_component_ids": incorrect_ids,
        "negative_signed_component_count": int(len(negative_ids)),
        "negative_signed_component_ids": negative_ids,
        "component_containment_depths": containment_depths,
        "component_expected_signed_volume_signs": expected_signs.astype(int).tolist(),
        "orientation_policy": "containment_depth_with_micro_shell_advisories",
        "print_tolerance_ignored_component_ids": sorted(micro_ids),
        "component_areas_mm2": component_areas.tolist(),
        "component_signed_volumes_mm3": signed_volumes.tolist(),
        "minimum_component_signed_volume_mm3": (
            float(np.min(signed_volumes)) if len(signed_volumes) else 0.0
        ),
        "orientation_volume_tolerance_mm3": float(tolerance),
        "all_closed_components_outward": bool(not incorrect_ids),
    }


def remove_new_redundant_boolean_micro_shells(
    mesh: trimesh.Trimesh,
    reference_mesh: trimesh.Trimesh,
    *,
    maximum_faces: int = 100,
    maximum_volume_mm3: float = 1e-5,
    maximum_equivalent_thickness_mm: float = 0.01,
    maximum_cover_distance_mm: float = 0.05,
    reference_vertex_tolerance_mm: float = 1e-7,
    minimum_reference_vertex_fraction: float = 0.80,
    include_face_indices: bool = False,
) -> tuple[trimesh.Trimesh, dict]:
    """Remove only new, near-coplanar Boolean dust already covered by the body.

    A closed Boolean result can contain a detached sliver with almost zero
    volume whose vertices sit a few microns above an otherwise complete main
    shell.  Topology and volume checks accept that sliver, while slicers show
    the two near-coincident surfaces as a dark crack.  Source-supported shells
    are protected by comparing candidate vertices with the immediate pre-cut
    parent; this is deliberately not a generic "keep largest component" rule.
    """
    result = mesh.copy()
    faces = np.asarray(result.faces, dtype=np.int64)
    vertices = np.asarray(result.vertices, dtype=np.float64)
    empty_record = {
        "applied": False,
        "policy": "new_redundant_near_coplanar_boolean_micro_shells",
        "component_count_before": 0 if not len(faces) else 1,
        "removed_component_count": 0,
        "removed_face_count": 0,
        "removed_volume_mm3": 0.0,
        "removed_components": [],
    }
    if not len(faces) or not bool(result.is_watertight):
        empty_record["skipped_reason"] = (
            "empty_mesh" if not len(faces) else "mesh_not_watertight"
        )
        return result, empty_record

    components = trimesh.graph.connected_components(
        np.asarray(result.face_adjacency, dtype=np.int64),
        nodes=np.arange(len(faces), dtype=np.int64),
        min_len=1,
    )
    empty_record["component_count_before"] = int(len(components))
    if len(components) <= 1:
        empty_record["skipped_reason"] = "single_component"
        return result, empty_record

    main_id = int(np.argmax([len(group) for group in components]))
    main_faces = np.asarray(components[main_id], dtype=np.int64)
    main_vertex_ids = np.unique(faces[main_faces].reshape(-1))
    main_tree = cKDTree(vertices[main_vertex_ids])
    reference_vertices = np.asarray(reference_mesh.vertices, dtype=np.float64)
    reference_tree = cKDTree(reference_vertices) if len(reference_vertices) else None

    triangles = vertices[faces]
    double_areas = np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )
    remove_mask = np.zeros(len(faces), dtype=bool)
    removed: list[dict] = []
    for component_id, group in enumerate(components):
        if component_id == main_id or len(group) > int(maximum_faces):
            continue
        face_ids = np.asarray(group, dtype=np.int64)
        component_vertex_ids = np.unique(faces[face_ids].reshape(-1))
        component_vertices = vertices[component_vertex_ids]
        area = float(0.5 * np.sum(double_areas[face_ids]))
        if area <= 0.0:
            continue
        origin = component_vertices.mean(axis=0)
        local_triangles = triangles[face_ids] - origin
        volume = abs(
            float(
                np.sum(
                    np.einsum(
                        "ij,ij->i",
                        local_triangles[:, 0],
                        np.cross(local_triangles[:, 1], local_triangles[:, 2]),
                    )
                )
                / 6.0
            )
        )
        equivalent_thickness = float(2.0 * volume / area)
        cover_distances, _nearest_main = main_tree.query(component_vertices, k=1)
        maximum_cover_distance = float(np.max(cover_distances))
        if reference_tree is None:
            reference_fraction = 0.0
        else:
            reference_distances, _nearest_reference = reference_tree.query(
                component_vertices,
                k=1,
            )
            reference_fraction = float(
                np.mean(
                    reference_distances
                    <= float(reference_vertex_tolerance_mm) + 1e-15
                )
            )
        removable = bool(
            volume <= float(maximum_volume_mm3) + 1e-15
            and equivalent_thickness
            <= float(maximum_equivalent_thickness_mm) + 1e-15
            and maximum_cover_distance
            <= float(maximum_cover_distance_mm) + 1e-15
            and reference_fraction
            < float(minimum_reference_vertex_fraction) - 1e-15
        )
        if not removable:
            continue
        remove_mask[face_ids] = True
        removed.append(
            {
                "component_id": int(component_id),
                "faces": int(len(face_ids)),
                "vertices": int(len(component_vertex_ids)),
                "area_mm2": area,
                "volume_mm3": volume,
                "equivalent_thickness_mm": equivalent_thickness,
                "maximum_cover_distance_mm": maximum_cover_distance,
                "reference_vertex_fraction": reference_fraction,
                "bbox_min_mm": component_vertices.min(axis=0).tolist(),
                "bbox_max_mm": component_vertices.max(axis=0).tolist(),
            }
        )

    if not removed:
        empty_record["skipped_reason"] = "no_redundant_micro_shell"
        return result, empty_record

    result.faces = faces[~remove_mask]
    result.remove_unreferenced_vertices()
    if not bool(result.is_watertight) or not bool(result.is_winding_consistent):
        raise ValueError(
            "redundant Boolean micro-shell cleanup damaged closed topology"
        )
    record = {
        **empty_record,
        "applied": True,
        "component_count_after": int(len(components) - len(removed)),
        "removed_component_count": int(len(removed)),
        "removed_face_count": int(np.count_nonzero(remove_mask)),
        "removed_volume_mm3": float(sum(item["volume_mm3"] for item in removed)),
        "maximum_faces": int(maximum_faces),
        "maximum_volume_mm3": float(maximum_volume_mm3),
        "maximum_equivalent_thickness_mm": float(
            maximum_equivalent_thickness_mm
        ),
        "maximum_cover_distance_mm": float(maximum_cover_distance_mm),
        "reference_vertex_tolerance_mm": float(reference_vertex_tolerance_mm),
        "minimum_reference_vertex_fraction": float(
            minimum_reference_vertex_fraction
        ),
        "removed_components": removed,
    }
    result.metadata["boolean_redundant_micro_shell_cleanup"] = record
    if include_face_indices:
        # Return provenance separately: do not serialize a face-sized metadata array.
        record = {**record, 'retained_face_indices': np.flatnonzero(~remove_mask).tolist()}
    return result, record


def orient_mesh_faces_consistently(mesh: trimesh.Trimesh) -> dict:
    """Make adjacent face winding consistent without changing mesh geometry.

    Face order, vertex order, and triangle membership remain stable; only the
    order of vertices inside affected triangles may change. This is safe for
    generated single-body inserts and preserves every fit-critical coordinate.
    """
    faces_before = np.asarray(mesh.faces, dtype=np.int64).copy()
    consistent_before = bool(mesh.is_winding_consistent)
    if not consistent_before:
        fix_winding_indexed(mesh)
    component_orientation_before = watertight_component_orientation_audit(mesh)
    inward_component_ids = list(
        component_orientation_before.get("inward_closed_component_ids", [])
    )
    if inward_component_ids:
        face_groups = trimesh.graph.connected_components(
            np.asarray(mesh.face_adjacency, dtype=np.int64),
            nodes=np.arange(len(mesh.faces), dtype=np.int64),
            min_len=1,
        )
        repaired_faces = np.asarray(mesh.faces, dtype=np.int64).copy()
        for component_id in inward_component_ids:
            face_ids = np.asarray(face_groups[int(component_id)], dtype=np.int64)
            repaired_faces[face_ids] = repaired_faces[face_ids][:, ::-1]
        mesh.faces = repaired_faces
    faces_after = np.asarray(mesh.faces, dtype=np.int64)
    if not np.array_equal(np.sort(faces_before, axis=1), np.sort(faces_after, axis=1)):
        raise RuntimeError("Winding repair changed face membership or face order")
    consistent_after = bool(mesh.is_winding_consistent)
    component_orientation_after = watertight_component_orientation_audit(mesh)
    return {
        "winding_consistent_before": consistent_before,
        "winding_consistent_after": consistent_after,
        "changed_triangle_winding": int(np.any(faces_before != faces_after, axis=1).sum()),
        "vertices_unchanged": True,
        "triangle_membership_unchanged": True,
        "component_orientation_before": component_orientation_before,
        "component_orientation_after": component_orientation_after,
        "inverted_closed_component_count": int(len(inward_component_ids)),
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
        pending = [closed_trail]
        while pending:
            trail = pending.pop()
            if len(trail) < 4 or trail[0] != trail[-1]:
                continue
            seen: dict[int, int] = {}
            for position, vertex in enumerate(trail[:-1]):
                if vertex not in seen:
                    seen[vertex] = position
                    continue
                start_position = seen[vertex]
                # Stack order preserves the previous depth-first traversal.
                pending.append(trail[:start_position + 1] + trail[position + 1:])
                pending.append(trail[start_position:position + 1])
                break
            else:
                loops.append(trail[:-1])

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


def average_outward_normal(local_vertices: np.ndarray, local_faces: np.ndarray, component_center: np.ndarray, model_center: np.ndarray) -> np.ndarray:
    """Use oriented surface normals; the model center is only a fallback.

    A recessed surface can face toward the model center. Flipping its valid
    normal to match a radial vector reverses the material side of the cut.
    Local vertex inward normals already use this same winding convention.
    """
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
    """Offset a closed loop using its local projected conormal.

    Positive offsets move away from the owning polygon and negative offsets
    move into it.  A loop-centroid ray reverses at concave boundaries, so the
    local incoming/outgoing tangent is the only safe default for fit rings.
    """
    if abs(float(offset_mm)) < 1e-9 or len(points) == 0:
        return points.copy()
    rel = points - origin
    xy = np.column_stack((rel @ u, rel @ v))
    normal_s = rel @ normal
    incoming = xy - np.roll(xy, 1, axis=0)
    outgoing = np.roll(xy, -1, axis=0) - xy
    incoming /= np.maximum(np.linalg.norm(incoming, axis=1)[:, None], 1e-12)
    outgoing /= np.maximum(np.linalg.norm(outgoing, axis=1)[:, None], 1e-12)
    tangents = incoming + outgoing
    tangent_lengths = np.linalg.norm(tangents, axis=1)
    valid = tangent_lengths > 1e-9
    tangents[valid] /= tangent_lengths[valid, None]

    area = signed_area(xy)
    if abs(float(area)) <= 1e-12 or not np.all(valid):
        centroid = xy.mean(axis=0)
        outward = xy - centroid
        outward /= np.maximum(np.linalg.norm(outward, axis=1)[:, None], 1e-12)
    elif area > 0.0:
        outward = np.column_stack((tangents[:, 1], -tangents[:, 0]))
    else:
        outward = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    shifted = xy + outward * float(offset_mm)
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


def _triangulate_polygon_ear_clip_global_scan(
    points_2d: np.ndarray,
) -> list[tuple[int, int, int]]:
    """Correctness fallback which re-evaluates every remaining ear.

    A vertex blocked by a distant interior point can become an ear after that
    point is removed.  The fast neighbor-only heap does not observe that state
    change, so deeply concave polygons need this bounded global rescan.
    """

    points = np.asarray(points_2d, dtype=np.float64)
    count = int(len(points))
    if count < 3:
        return []
    orientation = 1.0 if signed_area(points) >= 0.0 else -1.0
    scale = max(
        float(np.ptp(points[:, 0])),
        float(np.ptp(points[:, 1])),
        1.0,
    )
    epsilon = scale * scale * 1e-12
    remaining = [int(index) for index in range(count)]
    triangles: list[tuple[int, int, int]] = []
    while len(remaining) > 3:
        ears: list[tuple[float, float, int, tuple[int, int, int]]] = []
        for position, current in enumerate(remaining):
            left = int(remaining[(position - 1) % len(remaining)])
            right = int(remaining[(position + 1) % len(remaining)])
            a, b, c = points[[left, int(current), right]]
            cross = orientation * float(
                (b[0] - a[0]) * (c[1] - b[1])
                - (b[1] - a[1]) * (c[0] - b[0])
            )
            if cross <= epsilon:
                continue
            candidate_ids = np.asarray(
                [
                    int(value)
                    for value in remaining
                    if int(value) not in {left, int(current), right}
                ],
                dtype=np.int64,
            )
            if len(candidate_ids):
                samples = points[candidate_ids]
                ab = orientation * (
                    (b[0] - a[0]) * (samples[:, 1] - a[1])
                    - (b[1] - a[1]) * (samples[:, 0] - a[0])
                )
                bc = orientation * (
                    (c[0] - b[0]) * (samples[:, 1] - b[1])
                    - (c[1] - b[1]) * (samples[:, 0] - b[0])
                )
                ca = orientation * (
                    (a[0] - c[0]) * (samples[:, 1] - c[1])
                    - (a[1] - c[1]) * (samples[:, 0] - c[0])
                )
                if np.any((ab > epsilon) & (bc > epsilon) & (ca > epsilon)):
                    continue
            diagonal_sq = float(np.sum((c - a) ** 2))
            ears.append(
                (
                    diagonal_sq,
                    -cross,
                    int(position),
                    (left, int(current), right),
                )
            )
        if not ears:
            return []
        ears.sort(key=lambda item: (item[0], item[1], item[2]))
        _diagonal, _area, position, face = ears[0]
        triangles.append(face)
        remaining.pop(int(position))
    triangles.append(tuple(int(value) for value in remaining))
    return triangles if len(triangles) == count - 2 else []


def _triangulate_polygon_ear_clip_core(
    points_2d: np.ndarray,
    allow_global_scan_fallback: bool = True,
) -> list[tuple[int, int, int]]:
    """Triangulate a simple polygon whose boundary has no redundant collinear vertices."""
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
    waiting_on: dict[int, set[int]] = {}
    blocked_waiters: dict[int, set[int]] = collections.defaultdict(set)
    triangles: list[tuple[int, int, int]] = []
    remaining = count
    point_tree = cKDTree(points_2d)
    scale = max(float(np.ptp(points_2d[:, 0])), float(np.ptp(points_2d[:, 1])), 1.0)
    eps = scale * scale * 1e-12

    def clear_waiting(index: int) -> None:
        blockers = waiting_on.pop(int(index), set())
        for blocker in blockers:
            waiters = blocked_waiters.get(int(blocker))
            if waiters is None:
                continue
            waiters.discard(int(index))
            if not waiters:
                blocked_waiters.pop(int(blocker), None)

    def push(index: int) -> None:
        clear_waiting(int(index))
        if not alive[index]:
            return
        left = int(previous[index])
        right = int(following[index])
        diagonal_sq = float(np.sum((points_2d[right] - points_2d[left]) ** 2))
        first_edge = points_2d[index] - points_2d[left]
        second_edge = points_2d[right] - points_2d[left]
        local_area = abs(float(first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0]))
        heapq.heappush(heap, (diagonal_sq, -local_area, int(index), int(version[index])))

    def is_reflex(index: int) -> bool:
        if not alive[int(index)]:
            return False
        left = int(previous[int(index)])
        right = int(following[int(index)])
        a, b, c = points_2d[[left, int(index), right]]
        cross = float(
            (b[0] - a[0]) * (c[1] - b[1])
            - (b[1] - a[1]) * (c[0] - b[0])
        ) * orientation
        return bool(cross <= eps)

    def release_blocker(index: int) -> None:
        waiters = tuple(blocked_waiters.pop(int(index), set()))
        for waiter in waiters:
            blockers = waiting_on.get(int(waiter))
            if blockers is None:
                continue
            blockers.discard(int(index))
            if not blockers:
                waiting_on.pop(int(waiter), None)
                if alive[int(waiter)]:
                    push(int(waiter))

    # Reflex status changes only for the two neighbors of a removed ear.  The
    # former implementation recomputed it for every surviving vertex on every
    # heap pop, which turns a two-thousand-point painted boundary into millions
    # of redundant orientation tests.
    reflex = np.asarray(
        [is_reflex(int(index)) for index in range(count)],
        dtype=bool,
    )

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
        bounds_min = np.minimum(np.minimum(a, b), c)
        bounds_max = np.maximum(np.maximum(a, b), c)
        query_center = (bounds_min + bounds_max) * 0.5
        query_radius = float(np.linalg.norm(bounds_max - bounds_min)) * 0.5
        candidates = np.asarray(
            point_tree.query_ball_point(
                query_center,
                query_radius + math.sqrt(max(eps, 0.0)) + 1e-15,
            ),
            dtype=np.int64,
        )
        if len(candidates):
            candidates = candidates[
                alive[candidates] & reflex[candidates]
            ]
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
            inside = (ab > eps) & (bc > eps) & (ca > eps)
            if np.any(inside):
                blockers = set(int(value) for value in candidates[inside])
                waiting_on[int(current)] = blockers
                for blocker in blockers:
                    blocked_waiters[int(blocker)].add(int(current))
                continue
        triangles.append((left, current, right))
        neighbor_was_reflex = {
            int(neighbor): bool(reflex[int(neighbor)])
            for neighbor in (left, right)
        }
        clear_waiting(int(current))
        alive[current] = False
        reflex[current] = False
        following[left] = right
        previous[right] = left
        remaining -= 1
        release_blocker(int(current))
        for neighbor in (left, right):
            version[neighbor] += 1
            reflex[int(neighbor)] = is_reflex(int(neighbor))
            if neighbor_was_reflex[int(neighbor)] and not reflex[int(neighbor)]:
                release_blocker(int(neighbor))
            push(neighbor)

    if remaining == 3:
        first = int(np.flatnonzero(alive)[0])
        second = int(following[first])
        third = int(following[second])
        triangles.append((first, second, third))
    if len(triangles) != count - 2:
        if allow_global_scan_fallback:
            return _triangulate_polygon_ear_clip_global_scan(points_2d)
        return []
    return triangles


def triangulate_polygon_ear_clip(
    points_2d: np.ndarray,
) -> list[tuple[int, int, int]]:
    """Triangulate a simple polygon while preserving collinear source edges.

    Painted meshes commonly sample a visually straight boundary hundreds of
    times.  A strict ear clipper cannot remove a zero-area collinear ear, but
    dropping that vertex also drops two source boundary edges.  Triangulate a
    topology-equivalent reduced polygon first, then split the triangle incident
    to each reduced boundary edge into a fan over the original collinear chain.
    Every source edge survives and every emitted triangle has finite area.
    """

    points = np.asarray(points_2d, dtype=np.float64)
    count = int(len(points))
    if count < 3:
        return []
    scale = max(
        float(np.ptp(points[:, 0])),
        float(np.ptp(points[:, 1])),
        1.0,
    )
    # Output validation welds coordinates at six decimal places.  A curve can
    # therefore be mathematically non-collinear yet serialize to one straight
    # line, leaving a zero-area cap ear after the weld.  Classify collinearity
    # on the exact serialization grid instead of enlarging a scale-dependent
    # tolerance, which would flatten legitimate densely sampled curvature.
    classification_points = np.round(points, 6)
    area_epsilon = scale * scale * 1e-12
    kept = list(range(count))

    # Remove only a point which lies between its current neighbors.  A
    # zero-cross-product reversal or duplicated endpoint is not a harmless
    # sampling point and must remain a hard triangulation failure.
    changed = True
    while changed and len(kept) > 3:
        changed = False
        for position, current in enumerate(tuple(kept)):
            left = int(kept[(position - 1) % len(kept)])
            right = int(kept[(position + 1) % len(kept)])
            first = (
                classification_points[int(current)]
                - classification_points[left]
            )
            second = (
                classification_points[right]
                - classification_points[int(current)]
            )
            cross = abs(
                float(first[0] * second[1] - first[1] * second[0])
            )
            if cross > area_epsilon:
                continue
            if float(np.dot(first, second)) < -area_epsilon:
                continue
            kept.pop(position)
            changed = True
            break

    reduced = points[np.asarray(kept, dtype=np.int64)]
    reduced_faces = _triangulate_polygon_ear_clip_core(
        reduced,
        allow_global_scan_fallback=len(reduced) <= 512,
    )
    if len(reduced_faces) != len(kept) - 2:
        return []
    faces = [
        tuple(int(kept[int(index)]) for index in face)
        for face in reduced_faces
    ]

    def cyclic_chain(start: int, stop: int) -> list[int]:
        chain = [int(start)]
        cursor = int(start)
        while cursor != int(stop):
            cursor = (cursor + 1) % count
            chain.append(cursor)
            if len(chain) > count + 1:
                raise RuntimeError("polygon boundary chain did not close")
        return chain

    # Expand every reduced boundary edge exactly once.  The opposite triangle
    # vertex is already inside the polygon, so connecting each original edge
    # in the collinear chain to it yields non-overlapping finite-area faces.
    for position, start in enumerate(kept):
        stop = int(kept[(position + 1) % len(kept)])
        chain = cyclic_chain(int(start), stop)
        if len(chain) <= 2:
            continue
        target = None
        target_direction = 0
        for face_index, face in enumerate(faces):
            directed_edges = (
                (int(face[0]), int(face[1])),
                (int(face[1]), int(face[2])),
                (int(face[2]), int(face[0])),
            )
            if (int(start), stop) in directed_edges:
                target = int(face_index)
                target_direction = 1
                break
            if (stop, int(start)) in directed_edges:
                target = int(face_index)
                target_direction = -1
                break
        if target is None:
            return []
        old_face = faces.pop(target)
        opposite = next(
            int(vertex)
            for vertex in old_face
            if int(vertex) not in {int(start), stop}
        )
        replacements: list[tuple[int, int, int]] = []
        for left, right in zip(chain, chain[1:]):
            candidate = (
                (int(left), int(right), opposite)
                if target_direction > 0
                else (int(right), int(left), opposite)
            )
            triangle = classification_points[
                np.asarray(candidate, dtype=np.int64)
            ]
            double_area = abs(
                float(
                    (triangle[1, 0] - triangle[0, 0])
                    * (triangle[2, 1] - triangle[0, 1])
                    - (triangle[1, 1] - triangle[0, 1])
                    * (triangle[2, 0] - triangle[0, 0])
                )
            )
            if double_area <= area_epsilon:
                return []
            replacements.append(candidate)
        faces.extend(replacements)

    if len(faces) != count - 2:
        return []
    return faces


def inward_cap_surface_quality(
    points: np.ndarray,
    triangles: list[tuple[int, int, int]] | np.ndarray,
    reference_normal: np.ndarray,
) -> dict:
    """Audit whether a bottom cap is one coherent printable floor.

    A topologically closed triangulation can still fold a non-planar boundary
    into thin internal plates.  Cap geometry therefore needs a 3-D quality
    contract in addition to the ordinary edge-incidence audit.
    """
    point_array = np.asarray(points, dtype=np.float64)
    triangle_array = np.asarray(triangles, dtype=np.int64)
    normal = np.asarray(reference_normal, dtype=np.float64)
    normal_length = float(np.linalg.norm(normal))
    if normal_length <= 1e-12 and len(point_array) >= 3:
        normal = fit_plane_normal(point_array, np.array([0.0, 0.0, 1.0]))
    else:
        normal /= max(normal_length, 1e-12)
    origin = (
        point_array.mean(axis=0)
        if len(point_array)
        else np.zeros(3, dtype=np.float64)
    )
    plane_offsets = (
        np.abs((point_array - origin) @ normal)
        if len(point_array)
        else np.empty(0, dtype=np.float64)
    )
    maximum_planarity_error = (
        float(plane_offsets.max()) if len(plane_offsets) else 0.0
    )
    mean_planarity_error = (
        float(plane_offsets.mean()) if len(plane_offsets) else 0.0
    )

    minimum_normal_cosine = 1.0
    maximum_triangle_edge = 0.0
    inconsistent_normal_faces = 0
    long_folded_faces = 0
    if triangle_array.size:
        triangle_array = triangle_array.reshape((-1, 3))
        triangle_points = point_array[triangle_array]
        triangle_normals = np.cross(
            triangle_points[:, 1] - triangle_points[:, 0],
            triangle_points[:, 2] - triangle_points[:, 0],
        )
        double_areas = np.linalg.norm(triangle_normals, axis=1)
        valid_area = double_areas > 1e-15
        normal_cosines = np.zeros(len(triangle_points), dtype=np.float64)
        normal_cosines[valid_area] = np.abs(
            triangle_normals[valid_area] @ normal
        ) / double_areas[valid_area]
        minimum_normal_cosine = float(normal_cosines.min())
        inconsistent_normal_faces = int(
            np.count_nonzero(
                normal_cosines < MINIMUM_INWARD_CAP_NORMAL_COSINE - 1e-9
            )
        )
        edge_lengths = np.stack(
            [
                np.linalg.norm(
                    triangle_points[:, 1] - triangle_points[:, 0], axis=1
                ),
                np.linalg.norm(
                    triangle_points[:, 2] - triangle_points[:, 1], axis=1
                ),
                np.linalg.norm(
                    triangle_points[:, 0] - triangle_points[:, 2], axis=1
                ),
            ],
            axis=1,
        )
        maximum_edges = edge_lengths.max(axis=1)
        maximum_triangle_edge = float(maximum_edges.max())
        long_folded_faces = int(
            np.count_nonzero(
                (maximum_edges > 1.0)
                & (
                    normal_cosines
                    < MINIMUM_INWARD_CAP_NORMAL_COSINE - 1e-9
                )
            )
        )

    planarity_valid = bool(
        maximum_planarity_error
        <= MAXIMUM_INWARD_CAP_PLANARITY_ERROR_MM + 1e-9
    )
    normal_field_valid = bool(
        minimum_normal_cosine
        >= MINIMUM_INWARD_CAP_NORMAL_COSINE - 1e-9
    )
    return {
        "valid": bool(planarity_valid and normal_field_valid),
        "planarity_valid": planarity_valid,
        "normal_field_valid": normal_field_valid,
        "maximum_planarity_error_mm": maximum_planarity_error,
        "mean_planarity_error_mm": mean_planarity_error,
        "maximum_planarity_error_limit_mm": (
            MAXIMUM_INWARD_CAP_PLANARITY_ERROR_MM
        ),
        "minimum_normal_cosine": minimum_normal_cosine,
        "minimum_normal_cosine_limit": MINIMUM_INWARD_CAP_NORMAL_COSINE,
        "inconsistent_normal_faces": inconsistent_normal_faces,
        "long_folded_faces": long_folded_faces,
        "maximum_triangle_edge_mm": maximum_triangle_edge,
        "triangle_count": int(len(triangle_array)) if triangle_array.size else 0,
    }


def harmonic_patch_deformation(
    source_points: np.ndarray,
    source_faces: np.ndarray,
    boundary_vertex_ids: list[int] | tuple[int, ...] | np.ndarray,
    target_boundary_points: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Deform a source patch smoothly while matching its target rim exactly.

    Solving the displacement field, rather than absolute coordinates, retains
    the vendor patch's dense local topology and prevents the long cross-polygon
    diagonals that formed blade-like cap triangles on curved concave rims.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    points = np.asarray(source_points, dtype=np.float64)
    faces = np.asarray(source_faces, dtype=np.int64)
    boundary = np.asarray(boundary_vertex_ids, dtype=np.int64)
    targets = np.asarray(target_boundary_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("harmonic patch points must be an Nx3 array")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError("harmonic patch faces must be a non-empty Mx3 array")
    if targets.shape != (len(boundary), 3):
        raise ValueError("harmonic patch target rim does not match boundary ids")
    if len(boundary) != len(np.unique(boundary)):
        raise ValueError("harmonic patch boundary ids must be unique")
    if np.any(boundary < 0) or np.any(boundary >= len(points)):
        raise ValueError("harmonic patch boundary ids are out of range")

    neighbors: list[set[int]] = [set() for _ in range(len(points))]
    for a, b, c in faces:
        a, b, c = int(a), int(b), int(c)
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    boundary_mask = np.zeros(len(points), dtype=bool)
    boundary_mask[boundary] = True
    interior = np.flatnonzero(~boundary_mask)
    # Remove the large coherent part of the motion first.  A direct harmonic
    # displacement makes interior vertices lag behind a strongly shrunken rim
    # and can fold thousands of otherwise healthy source triangles.  The
    # least-squares affine baseline carries translation, rotation, shear and
    # global scale together; harmonic propagation then handles only the small
    # non-affine rim residual.
    boundary_homogeneous = np.column_stack(
        (points[boundary], np.ones(len(boundary), dtype=np.float64))
    )
    affine_matrix, _residuals, _rank, _singular = np.linalg.lstsq(
        boundary_homogeneous,
        targets,
        rcond=None,
    )
    affine_points = np.column_stack(
        (points, np.ones(len(points), dtype=np.float64))
    ) @ affine_matrix
    displacement = np.zeros_like(points)
    displacement[boundary] = targets - affine_points[boundary]
    if len(interior):
        interior_lookup = np.full(len(points), -1, dtype=np.int64)
        interior_lookup[interior] = np.arange(len(interior), dtype=np.int64)
        rows: list[int] = []
        columns: list[int] = []
        values: list[float] = []
        rhs = np.zeros((len(interior), 3), dtype=np.float64)
        for row, vertex_id in enumerate(interior):
            adjacent = neighbors[int(vertex_id)]
            if not adjacent:
                raise ValueError("harmonic patch contains an isolated interior vertex")
            rows.append(int(row))
            columns.append(int(row))
            values.append(float(len(adjacent)))
            for neighbor_id in adjacent:
                if boundary_mask[int(neighbor_id)]:
                    rhs[row] += displacement[int(neighbor_id)]
                else:
                    rows.append(int(row))
                    columns.append(int(interior_lookup[int(neighbor_id)]))
                    values.append(-1.0)
        matrix = coo_matrix(
            (values, (rows, columns)),
            shape=(len(interior), len(interior)),
        ).tocsr()
        for axis in range(3):
            displacement[interior, axis] = spsolve(matrix, rhs[:, axis])

    deformed = affine_points + displacement
    deformed[boundary] = targets
    source_triangles = affine_points[faces]
    deformed_triangles = deformed[faces]
    source_normals = np.cross(
        source_triangles[:, 1] - source_triangles[:, 0],
        source_triangles[:, 2] - source_triangles[:, 0],
    )
    deformed_normals = np.cross(
        deformed_triangles[:, 1] - deformed_triangles[:, 0],
        deformed_triangles[:, 2] - deformed_triangles[:, 0],
    )
    source_areas = np.linalg.norm(source_normals, axis=1)
    deformed_areas = np.linalg.norm(deformed_normals, axis=1)
    valid_source = source_areas > 1e-15
    valid_deformed = deformed_areas > 1e-12
    cosine = np.ones(len(faces), dtype=np.float64)
    comparable = valid_source & valid_deformed
    cosine[comparable] = np.einsum(
        "ij,ij->i",
        source_normals[comparable],
        deformed_normals[comparable],
    ) / (source_areas[comparable] * deformed_areas[comparable])
    source_edges = np.stack(
        [
            np.linalg.norm(source_triangles[:, 1] - source_triangles[:, 0], axis=1),
            np.linalg.norm(source_triangles[:, 2] - source_triangles[:, 1], axis=1),
            np.linalg.norm(source_triangles[:, 0] - source_triangles[:, 2], axis=1),
        ],
        axis=1,
    )
    deformed_edges = np.stack(
        [
            np.linalg.norm(deformed_triangles[:, 1] - deformed_triangles[:, 0], axis=1),
            np.linalg.norm(deformed_triangles[:, 2] - deformed_triangles[:, 1], axis=1),
            np.linalg.norm(deformed_triangles[:, 0] - deformed_triangles[:, 2], axis=1),
        ],
        axis=1,
    )
    edge_stretch = deformed_edges / np.maximum(source_edges, 0.02)
    boundary_error = float(
        np.linalg.norm(deformed[boundary] - targets, axis=1).max()
    )
    degenerate_faces = int(np.count_nonzero(~valid_deformed))
    reversed_faces = int(np.count_nonzero(cosine < 0.05))
    maximum_edge_stretch = float(edge_stretch.max())
    maximum_source_edge = float(source_edges.max())
    maximum_deformed_edge = float(deformed_edges.max())
    long_edge_valid = bool(
        maximum_deformed_edge <= max(2.5, 4.0 * maximum_source_edge)
    )
    quality = {
        "valid": bool(
            boundary_error <= 1e-9
            and degenerate_faces == 0
            and reversed_faces == 0
            and long_edge_valid
        ),
        "strategy": "source_patch_harmonic_displacement",
        "vertex_count": int(len(points)),
        "face_count": int(len(faces)),
        "boundary_vertex_count": int(len(boundary)),
        "boundary_match_error_mm": boundary_error,
        "degenerate_face_count": degenerate_faces,
        "reversed_face_count": reversed_faces,
        "minimum_source_normal_cosine": float(cosine.min()),
        "maximum_edge_stretch_ratio": maximum_edge_stretch,
        "maximum_affine_baseline_edge_mm": maximum_source_edge,
        "maximum_triangle_edge_mm": maximum_deformed_edge,
        "long_edge_valid": long_edge_valid,
    }
    return deformed, quality


def append_harmonic_cap_template(
    output_vertices: list[np.ndarray],
    output_faces: list[list[int]],
    template_points: np.ndarray,
    template_faces: np.ndarray,
    template_boundary_ids: tuple[int, ...] | list[int],
    bottom_ids: list[int],
    target_boundary_points: np.ndarray | None = None,
) -> tuple[int, dict]:
    """Append a reversed cap patch while welding its rim to existing bottom ids."""
    points = np.asarray(template_points, dtype=np.float64)
    faces = np.asarray(template_faces, dtype=np.int64)
    boundary = tuple(int(value) for value in template_boundary_ids)
    if len(boundary) != len(bottom_ids):
        raise ValueError("harmonic cap template rim does not match bottom ring")
    quality = {"valid": True, "strategy": "prevalidated_harmonic_template"}
    if target_boundary_points is not None:
        points, quality = harmonic_patch_deformation(
            points,
            faces,
            boundary,
            np.asarray(target_boundary_points, dtype=np.float64),
        )
        if not bool(quality["valid"]):
            rejected_harmonic_quality = quality
            points, quality = progressive_boundary_deformation(
                source_points=template_points,
                source_faces=faces,
                boundary_indices=boundary,
                target_boundary_points=target_boundary_points,
                deform=harmonic_patch_deformation,
            )
            if bool(quality["valid"]):
                quality["rejected_one_step_quality"] = rejected_harmonic_quality
        if not bool(quality["valid"]):
            rejected_progressive_quality = quality
            target = np.asarray(target_boundary_points, dtype=np.float64)
            template = np.asarray(template_points, dtype=np.float64)
            affine_matrix, affine_fit = fit_orientation_preserving_affine(
                template[np.asarray(boundary, dtype=np.int64)],
                target,
            )
            determinant = float(np.linalg.det(affine_matrix[:3, :]))
            if determinant <= 1e-9:
                return 0, {
                    **quality,
                    "valid": False,
                    "affine_fallback_rejected": True,
                    "affine_determinant": determinant,
                }
            points = np.column_stack(
                (template, np.ones(len(template), dtype=np.float64))
            ) @ affine_matrix
            deviation = np.linalg.norm(
                points[np.asarray(boundary, dtype=np.int64)] - target,
                axis=1,
            )
            quality = {
                "valid": True,
                "strategy": "source_patch_affine_socket_clearance",
                "vertex_count": int(len(points)),
                "face_count": int(len(faces)),
                "boundary_vertex_count": int(len(boundary)),
                "affine_determinant": determinant,
                "affine_fit": affine_fit,
                "target_boundary_deviation_max_mm": float(deviation.max()),
                "target_boundary_deviation_mean_mm": float(deviation.mean()),
                "rejected_harmonic_quality": rejected_harmonic_quality,
                "rejected_progressive_quality": rejected_progressive_quality,
            }
    mapping: dict[int, int] = {
        int(template_id): int(bottom_id)
        for template_id, bottom_id in zip(boundary, bottom_ids)
    }
    for template_id, bottom_id in zip(boundary, bottom_ids):
        output_vertices[int(bottom_id)] = points[int(template_id)].copy()
    for template_id, point in enumerate(points):
        if int(template_id) in mapping:
            continue
        mapping[int(template_id)] = len(output_vertices)
        output_vertices.append(np.asarray(point, dtype=np.float64))
    for a, b, c in faces:
        output_faces.append(
            [mapping[int(a)], mapping[int(c)], mapping[int(b)]]
        )
    return int(len(faces)), quality


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
    if len(points) > 512:
        # Dense vendor caps are already validated again after the authoritative
        # solid is built.  Three independent representative normals cover the
        # fitted surface without paying for four redundant world-axis views.
        dense_names = {"newell", "pca_axis_0", "fallback"}
        candidates = [item for item in candidates if item[0] in dense_names]

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
    scale = max(float(np.ptp(points, axis=0).max()), 1.0)
    minimum_double_area = scale * scale * 1e-14
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
        # Repeated and nearly collinear vendor samples can otherwise be chosen
        # as the shortest ears.  They close the edge graph but create a
        # zero-area triangle that invalidates the Boolean parent before the
        # mortise cutter runs.
        if area2 <= minimum_double_area:
            return
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
        final_points = points[[first, second, third]]
        final_area2 = float(
            np.linalg.norm(
                np.cross(
                    final_points[1] - final_points[0],
                    final_points[2] - final_points[0],
                )
            )
        )
        if final_area2 > minimum_double_area:
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
        "minimum_double_area_mm2": float(minimum_double_area),
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
        edge_left = np.asarray(ring, dtype=np.float64)
        edge_right = np.roll(edge_left, -1, axis=0)
        direction = np.asarray(right, dtype=np.float64) - np.asarray(left, dtype=np.float64)
        first_a = (
            direction[0] * (edge_left[:, 1] - float(left[1]))
            - direction[1] * (edge_left[:, 0] - float(left[0]))
        )
        first_b = (
            direction[0] * (edge_right[:, 1] - float(left[1]))
            - direction[1] * (edge_right[:, 0] - float(left[0]))
        )
        edge_direction = edge_right - edge_left
        second_a = (
            edge_direction[:, 0] * (float(left[1]) - edge_left[:, 1])
            - edge_direction[:, 1] * (float(left[0]) - edge_left[:, 0])
        )
        second_b = (
            edge_direction[:, 0] * (float(right[1]) - edge_left[:, 1])
            - edge_direction[:, 1] * (float(right[0]) - edge_left[:, 0])
        )
        return bool(
            np.any(
                (first_a * first_b < -(epsilon * epsilon))
                & (second_a * second_b < -(epsilon * epsilon))
            )
        )

    def segments_cross_ring_batch(
        left_points: np.ndarray,
        right_points: np.ndarray,
        ring: np.ndarray,
    ) -> np.ndarray:
        """Vectorized strict intersection test for many bridge candidates."""
        left_points = np.asarray(left_points, dtype=np.float64)
        right_points = np.asarray(right_points, dtype=np.float64)
        ring_left = np.asarray(ring, dtype=np.float64)
        ring_right = np.roll(ring_left, -1, axis=0)
        ring_direction = ring_right - ring_left
        result = np.zeros(len(left_points), dtype=bool)
        for start in range(0, len(left_points), 256):
            left = left_points[start : start + 256]
            right = right_points[start : start + 256]
            direction = right - left
            first_a = (
                direction[:, None, 0] * (ring_left[None, :, 1] - left[:, None, 1])
                - direction[:, None, 1] * (ring_left[None, :, 0] - left[:, None, 0])
            )
            first_b = (
                direction[:, None, 0] * (ring_right[None, :, 1] - left[:, None, 1])
                - direction[:, None, 1] * (ring_right[None, :, 0] - left[:, None, 0])
            )
            second_a = (
                ring_direction[None, :, 0] * (left[:, None, 1] - ring_left[None, :, 1])
                - ring_direction[None, :, 1] * (left[:, None, 0] - ring_left[None, :, 0])
            )
            second_b = (
                ring_direction[None, :, 0] * (right[:, None, 1] - ring_left[None, :, 1])
                - ring_direction[None, :, 1] * (right[:, None, 0] - ring_left[None, :, 0])
            )
            result[start : start + len(left)] = np.any(
                (first_a * first_b < -(epsilon * epsilon))
                & (second_a * second_b < -(epsilon * epsilon)),
                axis=1,
            )
        return result

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

    point_by_id = {
        int(vertex_id): np.asarray(point, dtype=np.float64)
        for ring_ids, ring_points in zip(loops_bottom, loop_points_2d)
        for vertex_id, point in zip(ring_ids, ring_points)
    }
    required_boundary_edges = {
        tuple(sorted((int(ring[position]), int(ring[(position + 1) % len(ring)]))))
        for ring in loops_bottom
        for position in range(len(ring))
    }

    def validate_mapped_faces(
        candidate_faces: list[tuple[int, int, int]],
    ) -> list[tuple[int, int, int]]:
        mapped: list[tuple[int, int, int]] = []
        face_keys: set[tuple[int, int, int]] = set()
        area_epsilon = scale * scale * 1e-13
        for candidate_face in candidate_faces:
            face = tuple(int(index) for index in candidate_face)
            if len(set(face)) != 3:
                continue
            points = np.asarray([point_by_id[index] for index in face])
            area2 = abs(
                float(
                    (points[1, 0] - points[0, 0])
                    * (points[2, 1] - points[0, 1])
                    - (points[1, 1] - points[0, 1])
                    * (points[2, 0] - points[0, 0])
                )
            )
            if area2 <= area_epsilon:
                continue
            key = tuple(sorted(face))
            if key in face_keys:
                continue
            face_keys.add(key)
            mapped.append(face)
        if not mapped:
            return []
        edge_counts: collections.Counter[tuple[int, int]] = collections.Counter(
            tuple(sorted(edge))
            for face in mapped
            for edge in (
                (face[0], face[1]),
                (face[1], face[2]),
                (face[2], face[0]),
            )
        )
        if any(int(edge_counts[edge]) != 1 for edge in required_boundary_edges):
            return []
        if any(int(count) not in (1, 2) for count in edge_counts.values()):
            return []
        if {edge for edge, count in edge_counts.items() if int(count) == 1} != required_boundary_edges:
            return []
        return mapped

    def map_and_validate(
        candidate_triangles: list[tuple[int, int, int]],
    ) -> list[tuple[int, int, int]]:
        return validate_mapped_faces([
            tuple(int(current_ids[int(index)]) for index in triangle)
            for triangle in candidate_triangles
        ])

    triangles = triangulate_polygon_ear_clip(polygon_points)
    mapped = map_and_validate(triangles)
    if mapped:
        return mapped

    # A one-hole annulus can be partitioned into two genuinely simple
    # polygons with two non-crossing visible bridges.  Unlike the conventional
    # one-bridge representation above, neither bridge endpoint is duplicated,
    # so dense concave vendor rings do not leave the ear clipper with a final
    # zero-area ear.  This is also preferable to a fan: both source boundaries
    # remain exact and no synthetic hub can create the familiar corner tents.
    if len(hole_records) == 1:
        hole_ids, hole_points = hole_records[0]

        def cyclic_arc(values: list, start: int, stop: int) -> list:
            result = [values[int(start)]]
            position = int(start)
            while position != int(stop):
                position = (position + 1) % len(values)
                result.append(values[position])
            return result

        outer_positions = np.tile(np.arange(len(outer_points), dtype=np.int64), len(hole_points))
        hole_positions = np.repeat(np.arange(len(hole_points), dtype=np.int64), len(outer_points))
        candidate_outer = outer_points[outer_positions]
        candidate_hole = hole_points[hole_positions]
        candidate_delta = candidate_outer - candidate_hole
        distance_sq = np.sum(candidate_delta * candidate_delta, axis=1)
        midpoints = (candidate_outer + candidate_hole) * 0.5
        valid = distance_sq > epsilon * epsilon
        valid &= points_in_or_on_poly_batch(midpoints, outer_points)
        valid &= ~points_in_or_on_poly_batch(midpoints, hole_points)
        valid &= ~segments_cross_ring_batch(candidate_outer, candidate_hole, outer_points)
        valid &= ~segments_cross_ring_batch(candidate_outer, candidate_hole, hole_points)

        visible_bridges: list[tuple[float, int, int]] = []
        for hole_position in range(len(hole_points)):
            candidate_indices = np.flatnonzero(valid & (hole_positions == hole_position))
            if not len(candidate_indices):
                continue
            order = candidate_indices[np.argsort(distance_sq[candidate_indices], kind="stable")]
            # Four nearest visible bridges per hole retain alternate routes at
            # concave corners while bounding the opposite-bridge pair search.
            visible_bridges.extend(
                (
                    float(distance_sq[index]),
                    int(outer_positions[index]),
                    int(hole_positions[index]),
                )
                for index in order[:4]
            )

        bridge_pairs: list[
            tuple[float, float, tuple[float, int, int], tuple[float, int, int]]
        ] = []
        for left_index, left in enumerate(visible_bridges):
            for right in visible_bridges[left_index + 1 :]:
                if left[1] == right[1] or left[2] == right[2]:
                    continue
                left_outer = outer_points[int(left[1])]
                left_hole = hole_points[int(left[2])]
                right_outer = outer_points[int(right[1])]
                right_hole = hole_points[int(right[2])]
                if segments_intersect_2d_strict(
                    left_outer,
                    left_hole,
                    right_outer,
                    right_hole,
                    epsilon,
                ):
                    continue
                outer_delta = abs(int(left[1]) - int(right[1]))
                outer_separation = min(outer_delta, len(outer_ids) - outer_delta) / float(
                    len(outer_ids)
                )
                hole_delta = abs(int(left[2]) - int(right[2]))
                hole_separation = min(hole_delta, len(hole_ids) - hole_delta) / float(
                    len(hole_ids)
                )
                balance = min(outer_separation, hole_separation)
                bridge_pairs.append(
                    (-float(balance), float(left[0] + right[0]), left, right)
                )
        # Opposite bridges make two balanced simple polygons.  Two nearby
        # bridges instead leave one almost-degenerate sliver and one enormous
        # collinear polygon, which is both slower and less numerically stable.
        bridge_pairs.sort(key=lambda item: (item[0], item[1]))

        for _negative_balance, _distance_score, left, right in bridge_pairs[:8]:
            outer_left, hole_left = int(left[1]), int(left[2])
            outer_right, hole_right = int(right[1]), int(right[2])
            polygon_specs = (
                (
                    cyclic_arc(outer_ids, outer_left, outer_right)
                    + cyclic_arc(hole_ids, hole_right, hole_left),
                    cyclic_arc(
                        [np.asarray(point, dtype=np.float64) for point in outer_points],
                        outer_left,
                        outer_right,
                    )
                    + cyclic_arc(
                        [np.asarray(point, dtype=np.float64) for point in hole_points],
                        hole_right,
                        hole_left,
                    ),
                ),
                (
                    cyclic_arc(outer_ids, outer_right, outer_left)
                    + cyclic_arc(hole_ids, hole_left, hole_right),
                    cyclic_arc(
                        [np.asarray(point, dtype=np.float64) for point in outer_points],
                        outer_right,
                        outer_left,
                    )
                    + cyclic_arc(
                        [np.asarray(point, dtype=np.float64) for point in hole_points],
                        hole_left,
                        hole_right,
                    ),
                ),
            )
            split_faces: list[tuple[int, int, int]] = []
            split_failed = False
            for split_ids, split_points in polygon_specs:
                split_triangles = triangulate_polygon_ear_clip(
                    np.asarray(split_points, dtype=np.float64)
                )
                if len(split_triangles) != len(split_ids) - 2:
                    split_failed = True
                    break
                split_faces.extend(
                    tuple(int(split_ids[int(index)]) for index in triangle)
                    for triangle in split_triangles
                )
            if split_failed:
                continue
            mapped = validate_mapped_faces(split_faces)
            if mapped:
                return mapped

    # Bridging a hole creates a mathematically valid *weakly simple* polygon:
    # both bridge endpoints occur twice and the two bridge edges coincide.
    # Dense vendor boundaries can leave the ordinary ear clipper with those
    # duplicate occurrences as its final zero-area ears.  Open only the 2-D
    # triangulation slit by a microscopic amount, then map both occurrences
    # back to their original 3-D vertex ids.  The topology audit above accepts
    # the result only when every original outer/hole edge occurs exactly once
    # and every synthetic interior edge exactly twice, so this cannot turn a
    # folded shortcut into a successful annulus.
    rounded_groups: dict[tuple[float, float], list[int]] = collections.defaultdict(list)
    for index, point in enumerate(polygon_points):
        rounded_groups[tuple(float(value) for value in np.round(point, 12))].append(index)
    duplicate_groups = [indices for indices in rounded_groups.values() if len(indices) > 1]
    if not duplicate_groups:
        return []
    for relative_delta in (1e-10, 1e-9, 1e-8, 1e-7, 1e-6):
        for global_sign in (1.0, -1.0):
            opened = polygon_points.copy()
            delta = scale * relative_delta
            for indices in duplicate_groups:
                for occurrence, index in enumerate(indices):
                    previous_index = (int(index) - 1) % len(opened)
                    next_index = (int(index) + 1) % len(opened)
                    tangent = polygon_points[next_index] - polygon_points[previous_index]
                    tangent_length = float(np.linalg.norm(tangent))
                    if tangent_length <= 1e-15:
                        continue
                    normal = np.asarray([-tangent[1], tangent[0]]) / tangent_length
                    side = 1.0 if occurrence % 2 == 0 else -1.0
                    opened[int(index)] += normal * delta * side * global_sign
            candidate = triangulate_polygon_ear_clip(opened)
            mapped = map_and_validate(candidate)
            if mapped:
                return mapped
    return []


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
                require_planar_quality=True,
            )
            expected = len(all_bottom_ids) - 2
            if int(added) != int(expected):
                raise ValueError(
                    "single-loop inward cap triangulation did not preserve the "
                    f"complete boundary: added={added}, expected={expected}"
                )
            cap_faces += int(added)
            continue

        bridged_triangles = triangulate_loop_group_with_bridges(loops_bottom, loop_points_2d, group)
        local_index = {
            int(vertex_id): position
            for position, vertex_id in enumerate(all_bottom_ids)
        }
        local_triangles = [
            tuple(local_index[int(vertex_id)] for vertex_id in triangle)
            for triangle in bridged_triangles
        ]
        cap_quality = inward_cap_surface_quality(
            np.asarray([vertices[index] for index in all_bottom_ids]),
            local_triangles,
            fit_plane_normal(
                np.asarray([vertices[index] for index in all_bottom_ids]),
                inward,
            ),
        )
        if not cap_quality["valid"]:
            raise ValueError(
                "multi-loop inward cap would form a non-planar blade surface: "
                + json.dumps(cap_quality, ensure_ascii=False, sort_keys=True)
            )
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
    require_planar_quality: bool = False,
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
    valid_triangles, projection_normal, projection_record = triangulate_ordered_loop_3d(
        points,
        fallback_normal,
        forbidden_edges,
    )
    # The projection normal belongs to the 2-D triangulation search.  A
    # successful world-axis view may produce shorter diagonals than the actual
    # cap plane, especially for concave or oblique connector rings.  It must
    # therefore never be reused as the physical plane normal for planarity or
    # winding checks: doing so reports a perfectly planar tilted peg tip as a
    # folded blade.  Fit the authoritative cap plane directly from the 3-D
    # ring, consistent with the multi-loop path above.
    normal = fit_plane_normal(points, fallback_normal)
    surface_quality = inward_cap_surface_quality(
        points,
        valid_triangles,
        normal,
    )
    if require_planar_quality and not surface_quality["valid"]:
        raise ValueError(
            "inward cap would form a non-planar blade surface: "
            + json.dumps(surface_quality, ensure_ascii=False, sort_keys=True)
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
        "occupied_edges_considered": int(len(occupied_edges or set())),
        "forbidden_internal_edges": int(len(forbidden_edges)),
        "planarity_max_error_mm": float(plane_offsets.max()) if len(plane_offsets) else 0.0,
        "planarity_mean_error_mm": float(plane_offsets.mean()) if len(plane_offsets) else 0.0,
        "cap_faces": int(added),
        "unsafe_center_fan_used": False,
        "projection_search": projection_record,
        "projection_normal": np.asarray(projection_normal, dtype=np.float64).tolist(),
        "cap_plane_normal": np.asarray(normal, dtype=np.float64).tolist(),
        "surface_quality": surface_quality,
    }
