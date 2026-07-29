from __future__ import annotations

from .common import *
from .mesh import *
from .package_io import *

def validate_exported_mesh(path: Path) -> dict:
    mesh = trimesh.load_mesh(path, process=True)
    if len(mesh.edges_unique_inverse):
        counts = np.bincount(mesh.edges_unique_inverse)
        open_edges = int(np.sum(counts == 1))
        over_edges = int(np.sum(counts > 2))
        unique_edges = int(len(counts))
    else:
        open_edges = 0
        over_edges = 0
        unique_edges = 0
    defect_edges = int(open_edges + over_edges)
    return {
        "reload_watertight": bool(mesh.is_watertight),
        "reload_winding_consistent": bool(mesh.is_winding_consistent),
        "reload_open_edges": open_edges,
        "reload_over_shared_edges": over_edges,
        "reload_unique_edges": unique_edges,
        "reload_defect_edges": defect_edges,
        "reload_topology_defect_ratio": float(defect_edges / max(unique_edges, 1)),
        "reload_faces": int(len(mesh.faces)),
        "reload_vertices": int(len(mesh.vertices)),
    }


def validate_mesh_in_memory(mesh: trimesh.Trimesh) -> dict:
    # Validate the topology that will actually be serialized. Vendor meshes
    # may intentionally keep coincident vertices distinct at paint junctions;
    # welding them for validation invents false three-face edge defects.
    check = mesh
    faces = np.asarray(check.faces, dtype=np.int64)
    directed_edges = (
        np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
        if len(faces)
        else np.empty((0, 2), dtype=np.int64)
    )
    if len(directed_edges):
        edge_min = np.minimum(directed_edges[:, 0], directed_edges[:, 1])
        edge_max = np.maximum(directed_edges[:, 0], directed_edges[:, 1])
        maximum_vertex_id = int(max(edge_min.max(), edge_max.max()))
        if maximum_vertex_id < (1 << 32):
            edge_keys = (
                edge_min.astype(np.uint64) << np.uint64(32)
            ) | edge_max.astype(np.uint64)
            unique_keys, inverse_edges, counts = np.unique(
                edge_keys,
                return_inverse=True,
                return_counts=True,
            )
            unique_edges = np.column_stack(
                (
                    (unique_keys >> np.uint64(32)).astype(np.int64),
                    (unique_keys & np.uint64(0xFFFFFFFF)).astype(np.int64),
                )
            )
        else:
            edges = np.column_stack((edge_min, edge_max))
            unique_edges, inverse_edges, counts = np.unique(
                edges,
                axis=0,
                return_inverse=True,
                return_counts=True,
            )
        direction_sign = np.where(
            directed_edges[:, 0] == edge_min,
            1,
            -1,
        )
    else:
        unique_edges = np.empty((0, 2), dtype=np.int64)
        inverse_edges = np.array([], dtype=np.int64)
        counts = np.array([], dtype=np.int64)
        direction_sign = np.array([], dtype=np.int8)
    open_edge_ids = unique_edges[counts == 1]
    over_edge_ids = unique_edges[counts > 2]
    direction_balance = (
        np.bincount(inverse_edges, weights=direction_sign, minlength=len(unique_edges))
        if len(inverse_edges)
        else np.array([], dtype=np.float64)
    )
    inconsistent_edge_ids = unique_edges[(counts == 2) & (np.abs(direction_balance) == 2)]
    unique_edge_count = int(len(unique_edges))
    defect_edge_count = int(len(open_edge_ids) + len(over_edge_ids))
    vertices = np.asarray(check.vertices, dtype=np.float64)

    def edge_metrics(selected: np.ndarray) -> dict:
        if not len(selected):
            return {"count": 0, "total_length_mm": 0.0, "max_length_mm": 0.0, "bbox_min": None, "bbox_max": None}
        segments = vertices[selected]
        lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
        points = segments.reshape(-1, 3)
        return {
            "count": int(len(selected)),
            "total_length_mm": float(lengths.sum()),
            "max_length_mm": float(lengths.max()),
            "bbox_min": points.min(axis=0).round(6).tolist(),
            "bbox_max": points.max(axis=0).round(6).tolist(),
        }
    return {
        "watertight": bool(
            len(faces)
            and not len(open_edge_ids)
            and not len(over_edge_ids)
        ),
        "winding_consistent": not bool(len(inconsistent_edge_ids)),
        "open_edges": int(len(open_edge_ids)),
        "over_shared_edges": int(len(over_edge_ids)),
        "inconsistent_shared_edges": int(len(inconsistent_edge_ids)),
        "unique_edges": unique_edge_count,
        "defect_edges": defect_edge_count,
        "topology_defect_ratio": float(defect_edge_count / max(unique_edge_count, 1)),
        "open_edge_ratio": float(len(open_edge_ids) / max(unique_edge_count, 1)),
        "over_shared_edge_ratio": float(len(over_edge_ids) / max(unique_edge_count, 1)),
        "inconsistent_orientation_ratio": float(len(inconsistent_edge_ids) / max(unique_edge_count, 1)),
        "open_edge_metrics": edge_metrics(open_edge_ids),
        "over_shared_edge_metrics": edge_metrics(over_edge_ids),
        "inconsistent_edge_metrics": edge_metrics(inconsistent_edge_ids),
        "faces": int(len(check.faces)),
        "vertices": int(len(check.vertices)),
    }


def finalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh.merge_vertices(digits_vertex=6)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(faces):
        _unique_faces, keep_indices = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
        if len(keep_indices) != len(faces):
            mesh.update_faces(np.sort(keep_indices))
    if hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()
    else:
        mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fill_holes(mesh)
    if open_edge_count(mesh) > 0:
        mesh = close_residual_boundaries(mesh)
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh)
    orientation_record = orient_mesh_faces_consistently(mesh)
    mesh.metadata["orientation_repair"] = orientation_record
    return mesh


def finalize_large_partition_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Linear cleanup for large source-preserving recursive bodies.

    Large bodies usually inherit consistently oriented source faces, but generated
    caps and sockets can introduce locally reversed triangles. Keep the linear
    cleanup, then run winding repair only when the fast consistency check fails.
    """
    # Do not weld coincident vendor vertices here. Paint junctions may use
    # separate vertex identities even when their coordinates match exactly.
    if hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()
    else:
        mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if open_edge_count(mesh) > 0:
        mesh = close_residual_boundaries(mesh)
    orientation_record = orient_mesh_faces_consistently(mesh)
    mesh.metadata["orientation_repair"] = orientation_record
    return mesh


def mesh_runtime_stats(mesh: trimesh.Trimesh) -> dict:
    """Avoid duplicate million-face edge scans before common validation."""
    if len(mesh.faces) > 250000:
        return {
            "watertight": None,
            "winding_consistent": None,
            "open_edges": None,
            "volume_mm3": None,
        }
    watertight = bool(mesh.is_watertight)
    return {
        "watertight": watertight,
        "winding_consistent": bool(mesh.is_winding_consistent),
        "open_edges": open_edge_count(mesh),
        "volume_mm3": float(mesh.volume) if watertight else None,
    }


def localized_short_open_edge_acceptance(
    record: dict,
    standard_ratio_limit: float,
) -> tuple[bool, dict]:
    """Accept only small, localized, consistently wound open-edge findings."""
    bbox_min = np.asarray(
        record.get("bbox_min") or [0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    bbox_max = np.asarray(
        record.get("bbox_max") or [0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    bbox_diagonal = float(np.linalg.norm(bbox_max - bbox_min))
    metrics = record.get("open_edge_metrics") or {}
    extended_ratio_limit = max(float(standard_ratio_limit), 0.002)
    maximum_edge_limit_mm = max(0.05, bbox_diagonal * 0.006)
    total_length_limit_mm = bbox_diagonal * 0.075
    accepted = bool(
        record["open_edges"]
        and not record["over_shared_edges"]
        and not record["inconsistent_shared_edges"]
        and record["winding_consistent"]
        and float(record["topology_defect_ratio"]) <= extended_ratio_limit
        and float(metrics.get("max_length_mm", float("inf")))
        <= maximum_edge_limit_mm
        and float(metrics.get("total_length_mm", float("inf")))
        <= total_length_limit_mm
    )
    return accepted, {
        "extended_ratio_limit": extended_ratio_limit,
        "maximum_open_edge_length_limit_mm": maximum_edge_limit_mm,
        "total_open_edge_length_limit_mm": total_length_limit_mm,
        "bbox_diagonal_mm": bbox_diagonal,
    }


def validate_multiview_visual_consistency(
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    source_part_by_face: np.ndarray,
    generated_parts: list[dict],
    view_count: int = 32,
    resolution: int = 384,
    depth_tolerance_mm: float = 0.12,
    max_intrusion_ratio: float = 0.04,
    max_material_mismatch_ratio: float = 0.03,
    max_local_material_mismatch_ratio: float = 0.10,
    min_coverage_ratio: float = 0.65,
    minimum_front_facing_dot: float = 0.05,
) -> dict:
    """Compare source paint ownership with the assembled split using deterministic depth maps.

    This check is intentionally offline. It does not open or control a slicer. It catches
    generated caps or sockets that move in front of the source surface, including valid but
    visually inverted nested details.
    """
    source_vertices = np.asarray(source_vertices, dtype=np.float64)
    source_faces = np.asarray(source_faces, dtype=np.int64)
    source_labels = np.asarray(source_part_by_face, dtype=np.int32)
    if len(source_labels) != len(source_faces):
        raise ValueError("source visual labels do not match source face count")
    assigned = source_labels > 0
    if not np.any(assigned):
        return {
            "valid": False,
            "errors": ["no source faces were assigned to recognized parts"],
            "method": "multi_view_centroid_surface_consistency",
        }

    def face_centroids_and_normals(
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        triangles = vertices[faces]
        normals = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-12)
        return triangles.mean(axis=1), normals

    source_centroids_all, source_normals_all = face_centroids_and_normals(
        source_vertices,
        source_faces,
    )
    source_centroids = source_centroids_all[assigned]
    source_normals = source_normals_all[assigned]
    source_labels = source_labels[assigned]
    generated_centroids = []
    generated_normals = []
    generated_labels = []
    generated_synthetic = []
    for part_index, part in enumerate(generated_parts, start=1):
        mesh = part["mesh"]
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if not len(faces):
            continue
        part_centroids, part_normals = face_centroids_and_normals(vertices, faces)
        generated_centroids.append(part_centroids)
        generated_normals.append(part_normals)
        generated_labels.append(np.full(len(faces), part_index, dtype=np.int32))
        source_surface_face_count = min(
            max(int(part.get("source_surface_face_count", 0)), 0),
            len(faces),
        )
        face_is_synthetic = np.ones(len(faces), dtype=np.bool_)
        face_is_synthetic[:source_surface_face_count] = False
        generated_synthetic.append(face_is_synthetic)
    if not generated_centroids:
        return {
            "valid": False,
            "errors": ["generated assembly has no faces"],
            "method": "multi_view_centroid_surface_consistency",
        }
    generated_centroids_array = np.vstack(generated_centroids)
    generated_normals_array = np.vstack(generated_normals)
    generated_labels_array = np.concatenate(generated_labels)
    generated_synthetic_array = np.concatenate(generated_synthetic)

    def front_map(
        points: np.ndarray,
        labels: np.ndarray,
        direction: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        min_x: float,
        min_y: float,
        scale: float,
        synthetic: np.ndarray | None = None,
        normals: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if normals is not None:
            visible = (normals @ direction) > max(
                float(minimum_front_facing_dot),
                0.0,
            )
            points = points[visible]
            labels = labels[visible]
            if synthetic is not None:
                synthetic = synthetic[visible]
        if not len(points):
            return (
                np.full(resolution * resolution, -np.inf, dtype=np.float64),
                np.zeros(resolution * resolution, dtype=np.int32),
                np.zeros(resolution * resolution, dtype=np.bool_),
            )
        px = np.clip(((points @ u - min_x) * scale).astype(np.int64), 0, resolution - 1)
        py = np.clip(((points @ v - min_y) * scale).astype(np.int64), 0, resolution - 1)
        keys = py * resolution + px
        depth = points @ direction
        front_depth = np.full(resolution * resolution, -np.inf, dtype=np.float64)
        np.maximum.at(front_depth, keys, depth)
        candidate_indices = np.flatnonzero(depth == front_depth[keys])
        front_indices = np.full(resolution * resolution, -1, dtype=np.int64)
        np.maximum.at(front_indices, keys[candidate_indices], candidate_indices)
        winner_pixels = np.flatnonzero(front_indices >= 0)
        winners = front_indices[winner_pixels]
        front_label = np.zeros(resolution * resolution, dtype=np.int32)
        front_synthetic = np.zeros(resolution * resolution, dtype=np.bool_)
        front_label[winner_pixels] = labels[winners]
        if synthetic is not None:
            front_synthetic[winner_pixels] = synthetic[winners]
        return front_depth, front_label, front_synthetic

    def neighborhood_front_map(
        depth: np.ndarray,
        labels: np.ndarray,
        radius: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Use the nearest raster neighborhood as a conservative source envelope.

        Triangle centroids are sparse point samples rather than exact ray/triangle
        intersections.  Comparing only the identical pixel makes an ordinary inward
        cap look external whenever its centroid lands beside, rather than on, the
        source triangle centroid.  A one-pixel max-depth envelope removes that
        rasterization artifact while remaining much narrower than printable facial
        details at the default 384-pixel resolution.
        """
        side = int(resolution)
        source_depth = depth.reshape((side, side))
        source_labels = labels.reshape((side, side))
        padded_depth = np.pad(
            source_depth,
            ((radius, radius), (radius, radius)),
            mode="constant",
            constant_values=-np.inf,
        )
        padded_labels = np.pad(
            source_labels,
            ((radius, radius), (radius, radius)),
            mode="constant",
            constant_values=0,
        )
        envelope_depth = np.full_like(source_depth, -np.inf)
        envelope_labels = np.zeros_like(source_labels)
        for offset_y in range(2 * radius + 1):
            for offset_x in range(2 * radius + 1):
                candidate_depth = padded_depth[
                    offset_y : offset_y + side,
                    offset_x : offset_x + side,
                ]
                candidate_labels = padded_labels[
                    offset_y : offset_y + side,
                    offset_x : offset_x + side,
                ]
                replace = candidate_depth > envelope_depth
                envelope_depth[replace] = candidate_depth[replace]
                envelope_labels[replace] = candidate_labels[replace]
        return envelope_depth.reshape(-1), envelope_labels.reshape(-1)

    per_view = []
    total_source_pixels = 0
    total_common_pixels = 0
    total_intrusion_pixels = 0
    total_mismatch_pixels = 0
    source_pixels_by_part: collections.Counter[int] = collections.Counter()
    intrusion_pixels_by_generated_part: collections.Counter[int] = collections.Counter()
    mismatch_pair_pixels: collections.Counter[tuple[int, int]] = collections.Counter()
    for view_index, direction in enumerate(fibonacci_view_directions(view_count)):
        u, v = exterior_visibility_basis(direction)
        source_x = source_centroids @ u
        source_y = source_centroids @ v
        min_x = float(source_x.min())
        min_y = float(source_y.min())
        span = max(
            float(source_x.max()) - min_x,
            float(source_y.max()) - min_y,
            1e-9,
        )
        scale = (resolution - 1) / span
        source_depth, source_part, _source_synthetic = front_map(
            source_centroids,
            source_labels,
            direction,
            u,
            v,
            min_x,
            min_y,
            scale,
            normals=source_normals,
        )
        source_envelope_depth, source_envelope_part = neighborhood_front_map(
            source_depth, source_part
        )
        generated_depth, generated_part, generated_front_is_synthetic = front_map(
            generated_centroids_array,
            generated_labels_array,
            direction,
            u,
            v,
            min_x,
            min_y,
            scale,
            generated_synthetic_array,
            generated_normals_array,
        )
        source_pixels = np.isfinite(source_depth)
        common_pixels = source_pixels & np.isfinite(generated_depth)
        comparable_pixels = np.isfinite(generated_depth) & np.isfinite(source_envelope_depth)
        intrusion_pixels = comparable_pixels & generated_front_is_synthetic & (
            generated_depth > source_envelope_depth + max(float(depth_tolerance_mm), 0.0)
        )
        # A material disagreement is visually relevant only when the generated
        # material is actually in front of the local source envelope.  Counting
        # equal-depth neighboring paint regions turns normal antialiased boundaries
        # into false failures.
        mismatch_pixels = intrusion_pixels & (generated_part != source_envelope_part)
        source_count = int(np.count_nonzero(source_pixels))
        common_count = int(np.count_nonzero(common_pixels))
        intrusion_count = int(np.count_nonzero(intrusion_pixels))
        mismatch_count = int(np.count_nonzero(mismatch_pixels))
        total_source_pixels += source_count
        total_common_pixels += common_count
        total_intrusion_pixels += intrusion_count
        total_mismatch_pixels += mismatch_count
        source_values, source_value_counts = np.unique(source_part[source_pixels], return_counts=True)
        for value, count in zip(source_values, source_value_counts):
            if int(value) > 0:
                source_pixels_by_part[int(value)] += int(count)
        intrusion_values, intrusion_value_counts = np.unique(
            generated_part[intrusion_pixels], return_counts=True
        )
        for value, count in zip(intrusion_values, intrusion_value_counts):
            if int(value) > 0:
                intrusion_pixels_by_generated_part[int(value)] += int(count)
        if mismatch_count:
            source_mismatch = source_envelope_part[mismatch_pixels].astype(np.uint64)
            generated_mismatch = generated_part[mismatch_pixels].astype(np.uint64)
            pair_keys = (
                source_mismatch << np.uint64(32)
            ) | generated_mismatch
            unique_pairs, pair_counts = np.unique(pair_keys, return_counts=True)
            for pair_key, pair_count in zip(unique_pairs, pair_counts):
                source_value = int(pair_key >> np.uint64(32))
                generated_value = int(pair_key & np.uint64(0xFFFFFFFF))
                mismatch_pair_pixels[(source_value, generated_value)] += int(pair_count)
        per_view.append(
            {
                "view_index": int(view_index),
                "source_pixels": source_count,
                "common_pixels": common_count,
                "coverage_ratio": float(common_count / max(source_count, 1)),
                "intrusion_pixels": intrusion_count,
                "material_mismatch_pixels": mismatch_count,
                "synthetic_front_pixels": int(np.count_nonzero(generated_front_is_synthetic)),
            }
        )

    coverage_ratio = float(total_common_pixels / max(total_source_pixels, 1))
    intrusion_ratio = float(total_intrusion_pixels / max(total_source_pixels, 1))
    mismatch_ratio = float(total_mismatch_pixels / max(total_common_pixels, 1))
    per_part = [
        {
            "part_index": int(part_index),
            "source_pixels": int(source_pixels_by_part.get(part_index, 0)),
            "generated_intrusion_pixels": int(
                intrusion_pixels_by_generated_part.get(part_index, 0)
            ),
            "generated_intrusion_to_source_ratio": float(
                intrusion_pixels_by_generated_part.get(part_index, 0)
                / max(source_pixels_by_part.get(part_index, 0), 1)
            ),
        }
        for part_index in range(1, len(generated_parts) + 1)
    ]
    material_mismatch_pairs = [
        {
            "source_part_index": int(source_part_index),
            "generated_part_index": int(generated_part_index),
            "pixels": int(pixel_count),
            "ratio_of_source_part": float(
                pixel_count / max(source_pixels_by_part.get(source_part_index, 0), 1)
            ),
        }
        for (source_part_index, generated_part_index), pixel_count in sorted(
            mismatch_pair_pixels.items(), key=lambda item: (-item[1], item[0])
        )
        if source_part_index > 0 and generated_part_index > 0
    ]
    blocking_local_mismatch_pairs = sorted(
        [
        pair
        for pair in material_mismatch_pairs
        if pair["ratio_of_source_part"]
        > float(max_local_material_mismatch_ratio)
        ],
        key=lambda pair: -float(pair["ratio_of_source_part"]),
    )
    errors = []
    advisories = []
    if coverage_ratio < float(min_coverage_ratio):
        errors.append(
            f"visual coverage ratio {coverage_ratio:.6f} is below {float(min_coverage_ratio):.6f}"
        )
    if intrusion_ratio > float(max_intrusion_ratio):
        errors.append(
            f"generated-surface intrusion ratio {intrusion_ratio:.6f} exceeds {float(max_intrusion_ratio):.6f}"
        )
    elif intrusion_ratio > min(0.02, float(max_intrusion_ratio)):
        advisories.append(
            f"generated-surface intrusion ratio {intrusion_ratio:.6f} exceeds advisory level 0.020000"
        )
    if mismatch_ratio > float(max_material_mismatch_ratio):
        errors.append(
            f"front-material mismatch ratio {mismatch_ratio:.6f} exceeds {float(max_material_mismatch_ratio):.6f}"
        )
    if blocking_local_mismatch_pairs:
        worst_pair = blocking_local_mismatch_pairs[0]
        errors.append(
            "local material mismatch P{source:02d}<-P{generated:02d} ratio {ratio:.6f} "
            "exceeds {limit:.6f}".format(
                source=int(worst_pair["source_part_index"]),
                generated=int(worst_pair["generated_part_index"]),
                ratio=float(worst_pair["ratio_of_source_part"]),
                limit=float(max_local_material_mismatch_ratio),
            )
        )
    return {
        "valid": not errors,
        "errors": errors,
        "advisories": advisories,
        "method": "multi_view_front_facing_neighborhood_surface_consistency",
        "view_count": int(max(view_count, 6)),
        "depth_map_resolution": int(max(resolution, 64)),
        "depth_tolerance_mm": float(depth_tolerance_mm),
        "source_envelope_radius_pixels": 1,
        "backface_and_grazing_culling": True,
        "minimum_front_facing_dot": float(minimum_front_facing_dot),
        "audited_surface": "generated_side_and_cap_faces_only",
        "coverage_ratio": coverage_ratio,
        "generated_surface_intrusion_ratio": intrusion_ratio,
        "front_material_mismatch_ratio": mismatch_ratio,
        "max_intrusion_ratio": float(max_intrusion_ratio),
        "advisory_intrusion_ratio": min(0.02, float(max_intrusion_ratio)),
        "max_material_mismatch_ratio": float(max_material_mismatch_ratio),
        "max_local_material_mismatch_ratio": float(max_local_material_mismatch_ratio),
        "min_coverage_ratio": float(min_coverage_ratio),
        "per_view": per_view,
        "per_part": per_part,
        "material_mismatch_pairs": material_mismatch_pairs,
        "blocking_local_material_mismatch_pairs": blocking_local_mismatch_pairs,
        "requires_computer_use": False,
        "slicer_screenshot_policy": "ask_user_to_open_final_3mf_and_provide_screenshots",
    }


def close_residual_boundaries(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Close residual loops without a centroid fan or synthetic hub vertex."""
    loops = boundary_loops(np.asarray(mesh.faces, dtype=np.int64))
    if not loops:
        return mesh
    vertices = [np.array(v, dtype=np.float64) for v in np.asarray(mesh.vertices)]
    faces = np.asarray(mesh.faces, dtype=np.int64).tolist()
    for loop in loops:
        if len(loop) < 3:
            continue
        loop_points = np.array([vertices[i] for i in loop], dtype=np.float64)
        if len(loop) == 3:
            edge_pairs = [(0, 1), (1, 2), (2, 0)]
            edge_lengths = [
                float(np.linalg.norm(loop_points[left] - loop_points[right]))
                for left, right in edge_pairs
            ]
            longest_index = int(np.argmax(edge_lengths))
            endpoint_left, endpoint_right = edge_pairs[longest_index]
            middle_index = next(index for index in range(3) if index not in (endpoint_left, endpoint_right))
            long_start = int(loop[endpoint_left])
            long_end = int(loop[endpoint_right])
            middle = int(loop[middle_index])
            long_vector = loop_points[endpoint_right] - loop_points[endpoint_left]
            long_length = float(np.linalg.norm(long_vector))
            line_distance = (
                float(np.linalg.norm(np.cross(loop_points[middle_index] - loop_points[endpoint_left], long_vector)))
                / long_length
                if long_length > 0.0
                else float("inf")
            )
            # A zero-area three-edge loop is usually a T-junction: one boundary
            # edge spans two collinear boundary edges. Split the face containing
            # that long edge at the middle vertex instead of adding a degenerate
            # cap triangle which cleanup would immediately remove again.
            if line_distance <= max(1e-9, long_length * 1e-7):
                split_face_index = None
                split_directed_edge = None
                split_third = None
                for face_index, face in enumerate(faces):
                    for edge_start, edge_end, third in (
                        (face[0], face[1], face[2]),
                        (face[1], face[2], face[0]),
                        (face[2], face[0], face[1]),
                    ):
                        if {int(edge_start), int(edge_end)} == {long_start, long_end}:
                            split_face_index = face_index
                            split_directed_edge = (int(edge_start), int(edge_end))
                            split_third = int(third)
                            break
                    if split_face_index is not None:
                        break
                if split_face_index is not None and split_directed_edge is not None and split_third is not None:
                    directed_start, directed_end = split_directed_edge
                    faces[split_face_index] = [directed_start, middle, split_third]
                    faces.append([middle, directed_end, split_third])
                    continue
        fallback = np.zeros(3, dtype=np.float64)
        shifted = np.roll(loop_points, -1, axis=0)
        fallback = np.sum(np.cross(loop_points, shifted), axis=0)
        if float(np.linalg.norm(fallback)) <= 1e-12:
            fallback = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        triangulate_boundary_cap_without_center(
            vertices,
            faces,
            [int(index) for index in loop],
            fallback,
            face_edge_set(np.asarray(faces, dtype=np.int64)),
        )
    repaired = trimesh.Trimesh(
        vertices=np.array(vertices, dtype=np.float64),
        faces=np.array(faces, dtype=np.int64),
        process=False,
        metadata=mesh.metadata.copy(),
    )
    repaired.merge_vertices(digits_vertex=6)
    if hasattr(repaired, "remove_degenerate_faces"):
        repaired.remove_degenerate_faces()
    else:
        repaired.update_faces(repaired.nondegenerate_faces())
    repaired.remove_unreferenced_vertices()
    return repaired




class ValidationService:
    """Validate generated meshes and the atomically reloaded 3MF package."""

    validate_mesh = staticmethod(validate_mesh_in_memory)
    validate_exported_mesh = staticmethod(validate_exported_mesh)

    @staticmethod
    def validate_package(
        path: Path,
        expected_parts: list[dict],
        source_filament_colors: list[str] | None = None,
        source_application: str | None = None,
        source_project_settings: dict | None = None,
        output_layout: str = "assembly",
    ) -> dict:
        return validate_colored_parts_3mf(
            path,
            expected_parts,
            source_filament_colors=source_filament_colors,
            source_application=source_application,
            source_project_settings=source_project_settings,
            output_layout=output_layout,
        )
