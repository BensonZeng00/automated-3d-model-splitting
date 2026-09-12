"""Boolean-only cutter proxies for recursive local connectors.

Printable child solids intentionally preserve their visible painted source
surface.  That surface is coincident with the restored parent patch, which is
the worst possible input for a solid difference.  This module derives a
topologically equivalent cutter by moving the complete source patch, including
its shared rim vertices, a tiny distance outside the parent.  Moving the rim
is essential: an exact-rim collar merely moves the coplanarity problem onto a
curve and Manifold emits hundreds of collapsed seam triangles there.  The
original child is never mutated or serialized from here.
"""

from __future__ import annotations

import numpy as np
import trimesh


_MAX_ISOLATED_NORMAL_RECOVERIES = 8
_MAX_ISOLATED_NORMAL_RECOVERY_RATIO = 1e-4
_MINIMUM_REMOTE_REENTRY_MM = 0.05


def _source_patch_boundary_edges(
    source_faces: np.ndarray,
) -> list[tuple[int, int]]:
    """Return oriented edges used once by the source-face prefix."""

    counts: dict[tuple[int, int], int] = {}
    oriented: dict[tuple[int, int], tuple[int, int]] = {}
    for face in np.asarray(source_faces, dtype=np.int64).reshape((-1, 3)):
        for left, right in (
            (int(face[0]), int(face[1])),
            (int(face[1]), int(face[2])),
            (int(face[2]), int(face[0])),
        ):
            key = tuple(sorted((left, right)))
            counts[key] = counts.get(key, 0) + 1
            oriented.setdefault(key, (left, right))
    return [oriented[key] for key in sorted(counts) if counts[key] == 1]


def _area_weighted_source_vertex_normals(
    vertices: np.ndarray,
    source_faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Compute normalized outward directions for source-patch vertices."""

    vertex_ids = np.unique(np.asarray(source_faces, dtype=np.int64).reshape(-1))
    accumulated = np.zeros_like(np.asarray(vertices, dtype=np.float64))
    triangles = np.asarray(vertices, dtype=np.float64)[source_faces]
    weighted_normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    for corner in range(3):
        np.add.at(accumulated, source_faces[:, corner], weighted_normals)
    lengths = np.linalg.norm(accumulated[vertex_ids], axis=1)
    bad = vertex_ids[lengths <= 1e-12]
    recovered: list[int] = []
    if len(bad):
        allowed = max(
            1,
            min(
                _MAX_ISOLATED_NORMAL_RECOVERIES,
                int(np.ceil(len(vertex_ids) * _MAX_ISOLATED_NORMAL_RECOVERY_RATIO)),
            ),
        )
        bad_flags = np.zeros(len(accumulated), dtype=bool)
        bad_flags[bad] = True
        clustered = bool(np.any(np.sum(bad_flags[source_faces], axis=1) >= 2))
        if len(bad) > allowed or clustered:
            reason = "clustered" if clustered else "count_limit"
            raise ValueError(
                "Boolean cutter source patch has undefined vertex normals: "
                f"count={int(len(bad))}, allowed={allowed}, reason={reason}, "
                f"samples={bad[:8].astype(int).tolist()}"
            )

        # A single source-mesh pole may cancel under area weighting even when
        # its surrounding patch is valid. Recover isolated cases from their
        # valid one-ring, then from the largest oriented incident face. A bad
        # cluster remains blocking because it indicates a damaged patch.
        valid = np.linalg.norm(accumulated, axis=1) > 1e-12
        for vertex_id in bad:
            incident = np.flatnonzero(np.any(source_faces == vertex_id, axis=1))
            adjacent = np.unique(source_faces[incident].reshape(-1))
            adjacent = adjacent[(adjacent != vertex_id) & valid[adjacent]]
            replacement = (
                np.sum(
                    accumulated[adjacent]
                    / np.linalg.norm(accumulated[adjacent], axis=1)[:, None],
                    axis=0,
                )
                if len(adjacent)
                else np.zeros(3, dtype=np.float64)
            )
            replacement_length = float(np.linalg.norm(replacement))
            if replacement_length <= 1e-12 and len(incident):
                incident_normals = weighted_normals[incident]
                incident_lengths = np.linalg.norm(incident_normals, axis=1)
                largest = int(np.argmax(incident_lengths))
                replacement = incident_normals[largest]
                replacement_length = float(incident_lengths[largest])
            if replacement_length <= 1e-12:
                raise ValueError(
                    "Boolean cutter source patch has an unrecoverable isolated "
                    f"vertex normal: vertex={int(vertex_id)}"
                )
            accumulated[int(vertex_id)] = replacement / replacement_length
            recovered.append(int(vertex_id))

        lengths = np.linalg.norm(accumulated[vertex_ids], axis=1)
    if np.any(lengths <= 1e-12):
        bad = vertex_ids[lengths <= 1e-12]
        raise ValueError(
            "Boolean cutter source patch has undefined vertex normals: "
            f"count={int(len(bad))}, samples={bad[:8].astype(int).tolist()}"
        )
    directions = accumulated[vertex_ids] / lengths[:, None]
    return vertex_ids, directions, {
        "undefined_normal_recovery_method": (
            "isolated_one_ring_then_largest_incident_face" if recovered else "none"
        ),
        "recovered_undefined_normal_count": int(len(recovered)),
        "recovered_undefined_normal_vertex_samples": recovered[:8],
    }


def assess_source_patch_reentry(
    vertices: np.ndarray,
    source_faces: np.ndarray,
    generated_faces: np.ndarray,
    *,
    overshoot_mm: float,
) -> dict[str, object]:
    """Detect source geometry which returns behind its own interface plane.

    A complete painted child is normally a safe Boolean proxy because its
    visible source patch stays outside the parent.  On a wraparound or oblique
    cut, however, a remote part of that patch can cross back to the generated
    backing side.  Subtracting the complete patch then punches a second exit
    through a thin parent wall while still producing a perfectly watertight
    solid.  The generated boundary attachment gives us the authoritative
    inward half-space without relying on a model-wide axis.
    """

    boundary_edges = _source_patch_boundary_edges(source_faces)
    boundary_ids = np.unique(np.asarray(boundary_edges, dtype=np.int64).reshape(-1))
    boundary_set = {int(vertex_id) for vertex_id in boundary_ids}
    inward_samples: list[np.ndarray] = []
    for face in np.asarray(generated_faces, dtype=np.int64).reshape((-1, 3)):
        boundary_vertices = [int(value) for value in face if int(value) in boundary_set]
        generated_vertices = [int(value) for value in face if int(value) not in boundary_set]
        for boundary_vertex in boundary_vertices:
            for generated_vertex in generated_vertices:
                delta = vertices[generated_vertex] - vertices[boundary_vertex]
                length = float(np.linalg.norm(delta))
                if length > 1e-9:
                    inward_samples.append(delta / length)
    inward_sample_array = np.asarray(inward_samples, dtype=np.float64).reshape((-1, 3))
    if len(inward_sample_array):
        inward = np.sum(inward_sample_array, axis=0)
    else:
        generated_ids = np.unique(np.asarray(generated_faces, dtype=np.int64).reshape(-1))
        inward = (
            vertices[generated_ids].mean(axis=0)
            - vertices[boundary_ids].mean(axis=0)
        )
    inward_length = float(np.linalg.norm(inward))
    cancellation_ratio = (
        inward_length / float(len(inward_sample_array))
        if len(inward_sample_array)
        else 0.0
    )
    inference_method = "generated_attachment_vector_sum"
    if inward_length <= 1e-12 or (
        len(inward_sample_array) and cancellation_ratio <= 1e-6
    ):
        # A rotationally symmetric backing can have perfectly valid inward
        # edge vectors whose vector sum is zero.  The oriented source-patch
        # rim still carries a stable normal by Stokes' theorem.  Its area
        # vector follows the visible patch's outward winding, so the opposite
        # direction is the attachment's inward half-space.
        boundary_center = vertices[boundary_ids].mean(axis=0)
        oriented_area = np.zeros(3, dtype=np.float64)
        for left, right in boundary_edges:
            oriented_area += np.cross(
                vertices[int(left)] - boundary_center,
                vertices[int(right)] - boundary_center,
            )
        oriented_area_length = float(np.linalg.norm(oriented_area))
        boundary_scale = max(
            float(np.linalg.norm(np.ptp(vertices[boundary_ids], axis=0))),
            1.0,
        )
        if oriented_area_length > boundary_scale * boundary_scale * 1e-12:
            inward = -oriented_area
            inward_length = oriented_area_length
            inference_method = "oriented_source_rim_area_fallback"
    if inward_length <= 1e-12:
        raise ValueError(
            "Boolean cutter cannot infer the generated attachment's inward half-space"
        )
    inward /= inward_length

    source_ids = np.unique(np.asarray(source_faces, dtype=np.int64).reshape(-1))
    remote_ids = np.asarray(
        [int(vertex_id) for vertex_id in source_ids if int(vertex_id) not in boundary_set],
        dtype=np.int64,
    )
    boundary_center = vertices[boundary_ids].mean(axis=0)
    signed_depths = (
        (vertices[remote_ids] - boundary_center) @ inward
        if len(remote_ids)
        else np.zeros(0, dtype=np.float64)
    )
    tolerance = max(
        float(_MINIMUM_REMOTE_REENTRY_MM),
        5.0 * float(overshoot_mm),
    )
    reentered = signed_depths > tolerance
    maximum = float(np.max(signed_depths)) if len(signed_depths) else 0.0
    samples = remote_ids[np.flatnonzero(reentered)[:8]].astype(int).tolist()
    return {
        "method": "generated_attachment_inward_half_space",
        "inward_axis_inference": inference_method,
        "inward_vector_cancellation_ratio": float(cancellation_ratio),
        "boundary_vertex_count": int(len(boundary_ids)),
        "remote_source_vertex_count": int(len(remote_ids)),
        "inward_axis": np.asarray(inward, dtype=np.float64).round(9).tolist(),
        "reentry_tolerance_mm": float(tolerance),
        "maximum_remote_inward_reentry_mm": maximum,
        "reentered_source_vertex_count": int(np.count_nonzero(reentered)),
        "reentered_source_vertex_samples": samples,
        "complete_source_patch_proxy_safe": bool(not np.any(reentered)),
    }


def build_interface_cap_overshoot_proxy(
    printable_child: trimesh.Trimesh,
    *,
    source_face_count: int,
    overshoot_mm: float = 0.01,
) -> tuple[trimesh.Trimesh, dict[str, object]]:
    """Replace a re-entering source patch with its interface cap.

    The printable child remains untouched.  Only the disposable Boolean proxy
    is changed: generated backing/connector faces are retained, while the
    potentially wraparound visible patch is replaced by a boundary-preserving
    cap.  This opens the intended parent preclosure but cannot reach a remote
    exterior wall.
    """

    from .mesh import boundary_loops, triangulate_boundary_cap_without_center

    distance = float(overshoot_mm)
    if not np.isfinite(distance) or distance <= 0.0:
        raise ValueError("Boolean cutter exterior overshoot must be positive")
    child = printable_child.copy()
    child.remove_unreferenced_vertices()
    trimesh.repair.fix_winding(child)
    trimesh.repair.fix_normals(child, multibody=True)
    if not child.is_watertight or not child.is_winding_consistent:
        raise ValueError(
            "Printable child must be closed and consistently wound before "
            "building an interface-cap Boolean proxy"
        )

    faces = np.asarray(child.faces, dtype=np.int64).reshape((-1, 3))
    vertices = np.asarray(child.vertices, dtype=np.float64).reshape((-1, 3))
    prefix = int(source_face_count)
    if prefix <= 0 or prefix >= len(faces):
        raise ValueError(
            "Interface-cap proxy needs a nonempty source-face prefix smaller "
            f"than the complete child: source_faces={prefix}, faces={len(faces)}"
        )
    source_faces = faces[:prefix]
    generated_faces = faces[prefix:]
    loops = boundary_loops(source_faces)
    if not loops:
        raise ValueError("Boolean cutter source patch has no open interface rim")

    reentry_record = assess_source_patch_reentry(
        vertices,
        source_faces,
        generated_faces,
        overshoot_mm=distance,
    )
    inward = np.asarray(reentry_record["inward_axis"], dtype=np.float64)
    source_vertex_ids, source_directions, normal_record = (
        _area_weighted_source_vertex_normals(vertices, source_faces)
    )
    direction_by_vertex = {
        int(vertex_id): direction
        for vertex_id, direction in zip(source_vertex_ids, source_directions)
    }
    boundary_ids = sorted({int(vertex_id) for loop in loops for vertex_id in loop})
    output_array = vertices.copy()
    for vertex_id in boundary_ids:
        output_array[vertex_id] = (
            output_array[vertex_id] + distance * direction_by_vertex[vertex_id]
        )

    # Triangulate the replacement cap at its final proxy coordinates.  Choosing
    # cap diagonals before the source rim is shifted can turn an otherwise valid
    # ear into a zero-area triangle on dense, curved interfaces.
    output_vertices = output_array.tolist()
    output_faces = generated_faces.astype(int).tolist()
    occupied_edges = {
        tuple(sorted((int(left), int(right))))
        for face in output_faces
        for left, right in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        )
    }
    cap_records: list[dict[str, object]] = []
    for loop_index, loop in enumerate(loops):
        cap_face_start = len(output_faces)
        added, cap_record = triangulate_boundary_cap_without_center(
            output_vertices,
            output_faces,
            [int(vertex_id) for vertex_id in loop],
            -inward,
            occupied_edges=occupied_edges,
        )
        if int(added) != len(loop) - 2:
            raise ValueError(
                "Interface-cap Boolean proxy could not close its source rim: "
                f"loop={loop_index}, vertices={len(loop)}, faces={added}"
            )
        cap_records.append({"loop_index": int(loop_index), **cap_record})
        for face in output_faces[cap_face_start:]:
            occupied_edges.update(
                tuple(sorted((int(left), int(right))))
                for left, right in (
                    (face[0], face[1]),
                    (face[1], face[2]),
                    (face[2], face[0]),
                )
            )

    output_array = np.asarray(output_vertices, dtype=np.float64)

    proxy = trimesh.Trimesh(
        vertices=output_array,
        faces=np.asarray(output_faces, dtype=np.int64),
        process=False,
        metadata=dict(child.metadata),
    )
    proxy.remove_unreferenced_vertices()
    trimesh.repair.fix_winding(proxy)
    trimesh.repair.fix_normals(proxy, multibody=True)
    triangles = np.asarray(proxy.vertices, dtype=np.float64)[
        np.asarray(proxy.faces, dtype=np.int64)
    ]
    double_areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    scale = max(float(np.linalg.norm(np.ptp(proxy.vertices, axis=0))), 1.0)
    degenerate_faces = int(np.count_nonzero(double_areas <= scale * scale * 1e-14))
    edge_counts = np.bincount(
        np.asarray(proxy.edges_unique_inverse, dtype=np.int64),
        minlength=len(proxy.edges_unique),
    )
    if (
        not proxy.is_watertight
        or not proxy.is_winding_consistent
        or np.any(edge_counts != 2)
        or degenerate_faces
    ):
        raise ValueError(
            "Interface-cap exterior proxy is invalid: "
            f"watertight={bool(proxy.is_watertight)}, "
            f"winding_consistent={bool(proxy.is_winding_consistent)}, "
            f"boundary_edges={int(np.count_nonzero(edge_counts == 1))}, "
            f"over_shared_edges={int(np.count_nonzero(edge_counts > 2))}, "
            f"degenerate_faces={degenerate_faces}"
        )
    record = {
        "proxy_method": "interface_cap_and_generated_attachment_overshoot",
        "proxy_selection_reason": "remote_source_patch_reentry",
        "printable_child_mutated": False,
        "source_face_count": int(prefix),
        "generated_face_count": int(len(generated_faces)),
        "source_patch_replaced_by_interface_cap": True,
        "source_patch_boundary_loop_count": int(len(loops)),
        "source_patch_boundary_edge_count": int(sum(len(loop) for loop in loops)),
        "interface_cap_face_count": int(sum(int(item["cap_faces"]) for item in cap_records)),
        "interface_cap_records": cap_records,
        "interface_cap_occupied_edge_guard": True,
        **normal_record,
        "exterior_overshoot_mm": distance,
        "exact_source_rim_preserved": False,
        "proxy_source_rim_shifted_outward": True,
        "interface_cap_triangulated_after_rim_overshoot": True,
        "collar_face_count": 0,
        "proxy_faces": int(len(proxy.faces)),
        "proxy_vertices": int(len(proxy.vertices)),
        "proxy_watertight": bool(proxy.is_watertight),
        "proxy_winding_consistent": bool(proxy.is_winding_consistent),
        "proxy_degenerate_faces": degenerate_faces,
        "proxy_volume_mm3": float(abs(proxy.volume)),
        "printable_child_volume_mm3": float(abs(child.volume)),
        "source_patch_reentry": reentry_record,
    }
    proxy.metadata["boolean_cutter_proxy"] = dict(record)
    return proxy, record


def build_protected_exterior_overshoot_proxy(
    printable_child: trimesh.Trimesh,
    *,
    source_face_count: int,
    overshoot_mm: float = 0.01,
) -> tuple[trimesh.Trimesh, dict[str, object]]:
    """Select the broad or interface-cap proxy from geometry evidence."""

    faces = np.asarray(printable_child.faces, dtype=np.int64).reshape((-1, 3))
    vertices = np.asarray(printable_child.vertices, dtype=np.float64).reshape((-1, 3))
    prefix = int(source_face_count)
    if prefix <= 0 or prefix >= len(faces):
        raise ValueError(
            "Boolean cutter proxy needs a valid protected source-face prefix"
        )
    reentry_record = assess_source_patch_reentry(
        vertices,
        faces[:prefix],
        faces[prefix:],
        overshoot_mm=overshoot_mm,
    )
    if bool(reentry_record["complete_source_patch_proxy_safe"]):
        proxy, proxy_record = build_exterior_overshoot_proxy(
            printable_child,
            source_face_count=prefix,
            overshoot_mm=overshoot_mm,
        )
        proxy_record = {
            **proxy_record,
            "proxy_selection_reason": "complete_source_patch_stays_exterior",
            "source_patch_reentry": reentry_record,
        }
        proxy.metadata["boolean_cutter_proxy"] = dict(proxy_record)
        return proxy, proxy_record
    return build_interface_cap_overshoot_proxy(
        printable_child,
        source_face_count=prefix,
        overshoot_mm=overshoot_mm,
    )


def build_exterior_overshoot_proxy(
    printable_child: trimesh.Trimesh,
    *,
    source_face_count: int,
    overshoot_mm: float = 0.01,
) -> tuple[trimesh.Trimesh, dict[str, object]]:
    """Move the child's complete source patch just outside the parent.

    Every source-patch vertex is displaced along its area-weighted local
    outward normal.  Boundary vertices are shared by generated backing faces,
    so their top edge moves by the same tiny amount and the proxy remains
    closed without a collar.  The printable child and its visible rim remain
    untouched; only this disposable Boolean operand changes.
    """

    distance = float(overshoot_mm)
    if not np.isfinite(distance) or distance <= 0.0:
        raise ValueError("Boolean cutter exterior overshoot must be positive")
    child = printable_child.copy()
    child.remove_unreferenced_vertices()
    trimesh.repair.fix_winding(child)
    trimesh.repair.fix_normals(child, multibody=True)
    if not child.is_watertight or not child.is_winding_consistent:
        raise ValueError(
            "Printable child must be a closed, consistently wound solid before "
            "building its Boolean cutter proxy"
        )

    faces = np.asarray(child.faces, dtype=np.int64).reshape((-1, 3))
    vertices = np.asarray(child.vertices, dtype=np.float64).reshape((-1, 3))
    prefix = int(source_face_count)
    if prefix <= 0 or prefix >= len(faces):
        raise ValueError(
            "Boolean cutter proxy needs a nonempty source-face prefix smaller "
            f"than the complete child: source_faces={prefix}, faces={len(faces)}"
        )
    source_faces = faces[:prefix]
    generated_faces = faces[prefix:]
    boundary_edges = _source_patch_boundary_edges(source_faces)
    if not boundary_edges:
        raise ValueError("Boolean cutter source patch has no open interface rim")

    # Every source-patch rim edge must already be closed by one generated child
    # face.  Otherwise the collar would hide an upstream topology error.
    all_edge_counts = np.bincount(
        np.asarray(child.edges_unique_inverse, dtype=np.int64),
        minlength=len(child.edges_unique),
    )
    if np.any(all_edge_counts != 2):
        raise ValueError("Printable child topology changed before cutter derivation")

    source_vertex_ids, directions, normal_record = _area_weighted_source_vertex_normals(
        vertices,
        source_faces,
    )
    output_vertices = vertices.copy()
    for vertex_id, direction in zip(source_vertex_ids, directions):
        output_vertices[int(vertex_id)] = (
            vertices[int(vertex_id)] + distance * direction
        )
    output_faces = faces.copy()
    proxy = trimesh.Trimesh(
        vertices=np.asarray(output_vertices, dtype=np.float64),
        faces=output_faces,
        process=False,
        metadata=dict(child.metadata),
    )
    proxy.remove_unreferenced_vertices()
    trimesh.repair.fix_winding(proxy)
    trimesh.repair.fix_normals(proxy, multibody=True)
    proxy_faces = np.asarray(proxy.faces, dtype=np.int64)
    proxy_triangles = np.asarray(proxy.vertices, dtype=np.float64)[proxy_faces]
    proxy_double_areas = np.linalg.norm(
        np.cross(
            proxy_triangles[:, 1] - proxy_triangles[:, 0],
            proxy_triangles[:, 2] - proxy_triangles[:, 0],
        ),
        axis=1,
    )
    scale = max(float(np.linalg.norm(np.ptp(proxy.vertices, axis=0))), 1.0)
    minimum_double_area = scale * scale * 1e-14
    degenerate_faces = int(
        np.count_nonzero(proxy_double_areas <= minimum_double_area)
    )
    edge_counts = np.bincount(
        np.asarray(proxy.edges_unique_inverse, dtype=np.int64),
        minlength=len(proxy.edges_unique),
    )
    if (
        not proxy.is_watertight
        or not proxy.is_winding_consistent
        or np.any(edge_counts != 2)
        or degenerate_faces
    ):
        raise ValueError(
            "Exterior overshoot Boolean proxy is invalid: "
            f"watertight={bool(proxy.is_watertight)}, "
            f"winding_consistent={bool(proxy.is_winding_consistent)}, "
            f"boundary_edges={int(np.count_nonzero(edge_counts == 1))}, "
            f"over_shared_edges={int(np.count_nonzero(edge_counts > 2))}, "
            f"degenerate_faces={degenerate_faces}"
        )

    record = {
        "proxy_method": "source_patch_and_shared_rim_exterior_overshoot",
        "printable_child_mutated": False,
        "source_face_count": int(prefix),
        "generated_face_count": int(len(generated_faces)),
        "source_patch_vertex_count": int(len(source_vertex_ids)),
        **normal_record,
        "source_patch_boundary_edge_count": int(len(boundary_edges)),
        "collar_face_count": 0,
        "exterior_overshoot_mm": distance,
        "minimum_source_patch_displacement_mm": distance,
        "maximum_source_patch_displacement_mm": distance,
        "source_patch_outward_direction_span": np.ptp(
            directions,
            axis=0,
        ).round(9).tolist(),
        "exact_source_rim_preserved": False,
        "proxy_source_rim_shifted_outward": True,
        "proxy_faces": int(len(proxy.faces)),
        "proxy_vertices": int(len(proxy.vertices)),
        "proxy_watertight": bool(proxy.is_watertight),
        "proxy_winding_consistent": bool(proxy.is_winding_consistent),
        "proxy_degenerate_faces": degenerate_faces,
        "proxy_volume_mm3": float(abs(proxy.volume)),
        "printable_child_volume_mm3": float(abs(child.volume)),
    }
    proxy.metadata["boolean_cutter_proxy"] = dict(record)
    return proxy, record
