from __future__ import annotations

import collections
from dataclasses import dataclass

from .common import *
from .mesh import orient_mesh_faces_consistently
from .validation import close_residual_boundaries, validate_mesh_in_memory


@dataclass(frozen=True)
class SourcePreservingFinalizationPolicy:
    """Immutable cleanup policy for a generated shell with a source-face prefix.

    Vendor meshes legitimately contain extremely small, non-zero triangles.
    They are topology, not numerical litter, so generic vertex welding or
    Trimesh's scale-dependent degenerate-face removal must never touch the
    protected prefix.  Generated geometry is expected to arrive prevalidated;
    this service audits it and closes only residual identity-level seams.
    """

    protected_source_face_count: int
    coordinate_digits: int = 6
    allow_residual_boundary_closure: bool = True
    require_watertight: bool = True


@dataclass
class SourcePreservingFinalizationResult:
    mesh: trimesh.Trimesh
    audit: dict


def _face_coordinate_key(vertices: np.ndarray, face: np.ndarray) -> tuple:
    points = np.round(vertices[np.asarray(face, dtype=np.int64)], decimals=10)
    return tuple(sorted(tuple(float(value) for value in point) for point in points))


def _source_face_inventory(
    vertices: np.ndarray,
    faces: np.ndarray,
    protected_count: int,
) -> collections.Counter:
    return collections.Counter(
        _face_coordinate_key(vertices, face)
        for face in faces[:protected_count]
    )


def weld_coincident_open_boundary_vertices(
    mesh: trimesh.Trimesh,
    digits_vertex: int = 6,
) -> tuple[trimesh.Trimesh, dict]:
    """Weld only coordinate aliases which participate in an open boundary.

    A global merge can collapse valid thin vendor triangles.  This operation
    is deliberately narrower: it considers only coordinate-equal groups that
    touch an open edge, rejects any remap that repeats a vertex inside a face,
    and accepts the candidate only when topology strictly improves.
    """

    before = validate_mesh_in_memory(mesh)
    record = {
        "attempted": False,
        "accepted": False,
        "open_edges_before": int(before["open_edges"]),
        "open_edges_after": int(before["open_edges"]),
        "over_shared_edges_before": int(before["over_shared_edges"]),
        "over_shared_edges_after": int(before["over_shared_edges"]),
        "welded_vertex_groups": 0,
    }
    if not int(before["open_edges"]):
        return mesh, record

    faces = np.asarray(mesh.faces, dtype=np.int64)
    directed_edges = np.vstack(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])
    )
    undirected_edges = np.sort(directed_edges, axis=1)
    unique_edges, edge_counts = np.unique(
        undirected_edges,
        axis=0,
        return_counts=True,
    )
    open_edges = unique_edges[edge_counts == 1]
    open_vertices = set(int(value) for value in open_edges.reshape(-1))
    if not open_vertices:
        return mesh, record

    rounded = np.round(np.asarray(mesh.vertices, dtype=np.float64), digits_vertex)
    coordinate_groups: dict[tuple[float, float, float], list[int]] = (
        collections.defaultdict(list)
    )
    for vertex_index, point in enumerate(rounded):
        coordinate_groups[tuple(float(value) for value in point)].append(
            int(vertex_index)
        )

    vertex_faces: dict[int, set[int]] = collections.defaultdict(set)
    for face_index, face in enumerate(faces):
        for vertex_index in face:
            vertex_faces[int(vertex_index)].add(int(face_index))

    remap = np.arange(len(mesh.vertices), dtype=np.int64)
    welded_groups = 0
    for group in coordinate_groups.values():
        if len(group) < 2 or not any(index in open_vertices for index in group):
            continue
        group_set = set(group)
        affected_faces: set[int] = set()
        for vertex_index in group:
            affected_faces.update(vertex_faces.get(vertex_index, set()))
        if any(
            sum(int(vertex_index) in group_set for vertex_index in faces[face_index]) > 1
            for face_index in affected_faces
        ):
            continue
        representative = min(group)
        remap[np.asarray(group, dtype=np.int64)] = representative
        welded_groups += 1

    if not welded_groups:
        return mesh, record
    record["attempted"] = True
    record["welded_vertex_groups"] = int(welded_groups)
    candidate = mesh.copy()
    candidate.faces = remap[faces]
    candidate.remove_unreferenced_vertices()
    after = validate_mesh_in_memory(candidate)
    record["open_edges_after"] = int(after["open_edges"])
    record["over_shared_edges_after"] = int(after["over_shared_edges"])
    improves = int(after["open_edges"]) < int(before["open_edges"])
    safe = (
        int(after["over_shared_edges"]) <= int(before["over_shared_edges"])
        and int(after["inconsistent_shared_edges"])
        <= int(before["inconsistent_shared_edges"])
        and len(candidate.faces) == len(mesh.faces)
    )
    if improves and safe:
        record["accepted"] = True
        return candidate, record
    return mesh, record


class SourcePreservingMeshFinalizer:
    """Finalize generated parts without deleting valid vendor source faces."""

    @staticmethod
    def finalize(
        mesh: trimesh.Trimesh,
        policy: SourcePreservingFinalizationPolicy,
    ) -> SourcePreservingFinalizationResult:
        result = trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices, dtype=np.float64).copy(),
            faces=np.asarray(mesh.faces, dtype=np.int64).copy(),
            process=False,
            metadata=mesh.metadata.copy(),
        )
        from .finalization_case import FinalizationCase
        case = FinalizationCase(result, policy)
        cached = case.load(result)
        if cached is not None:
            return SourcePreservingFinalizationResult(mesh=cached[0],audit=cached[1])
        protected_count = int(policy.protected_source_face_count)
        if not 0 <= protected_count <= len(result.faces):
            raise ValueError("protected source-face count is outside the mesh")

        initial_vertices = np.asarray(result.vertices, dtype=np.float64)
        initial_faces = np.asarray(result.faces, dtype=np.int64)
        source_inventory = _source_face_inventory(
            initial_vertices,
            initial_faces,
            protected_count,
        )
        initial_validation = validate_mesh_in_memory(result)

        # Geometry builders own triangle membership.  Do not run a generic
        # merge/remove-degenerate pass here: it can erase a narrow but valid
        # source triangle and turn its three edges into a hole.
        result.remove_unreferenced_vertices()
        result, weld_record = weld_coincident_open_boundary_vertices(
            result,
            digits_vertex=int(policy.coordinate_digits),
        )
        case.save('after_weld', result)
        preclosure_validation = validate_mesh_in_memory(result)
        source_ear_restoration = {'applied': False}
        if preclosure_validation['over_shared_edges']:
            from .source_ear_repair import restore_source_ears
            result, source_ear_restoration = restore_source_ears(result, protected_count)
            if source_ear_restoration['applied']:
                case.save('after_source_ear_restoration', result)
                preclosure_validation = validate_mesh_in_memory(result)
        faces_before_closure = int(len(result.faces))
        closure_applied = False
        if (
            int(preclosure_validation["open_edges"]) > 0
            and bool(policy.allow_residual_boundary_closure)
        ):
            result = close_residual_boundaries(result, merge_and_clean=False)
            closure_applied = int(len(result.faces)) > faces_before_closure

        case.save('after_closure', result)
        orientation_record = orient_mesh_faces_consistently(result)
        final_validation = validate_mesh_in_memory(result)
        if policy.allow_residual_boundary_closure and any(final_validation[key] for key in ('open_edges','inconsistent_shared_edges')) and not final_validation['over_shared_edges']:
            from .micro_mesh_repair import repair_micro_mesh
            result, micro_repair = repair_micro_mesh(result)
            case.save('after_micro_repair',result)
            orientation_record = orient_mesh_faces_consistently(result)
            final_validation = validate_mesh_in_memory(result)
        final_vertices = np.asarray(result.vertices, dtype=np.float64)
        final_faces = np.asarray(result.faces, dtype=np.int64)
        final_inventory = collections.Counter(
            _face_coordinate_key(final_vertices, face)
            for face in final_faces
        )
        missing_source_faces = int(
            sum(
                max(int(count) - int(final_inventory.get(key, 0)), 0)
                for key, count in source_inventory.items()
            )
        )
        from .surface_preservation import audit_replaced_surface
        preservation = audit_replaced_surface(source_inventory, final_inventory, result)
        if missing_source_faces and not preservation['accepted']:
            raise ValueError(
                "source-preserving finalization removed vendor source faces: "
                f"missing={missing_source_faces}"
            )

        audit = {
            "strategy": "source_face_prefix_preserving",
            "protected_source_face_count": protected_count,
            "generated_face_count_before": int(len(initial_faces) - protected_count),
            "input_faces": int(len(initial_faces)),
            "output_faces": int(len(final_faces)),
            "source_faces_missing": missing_source_faces,
            "source_surface_preservation": preservation,
            "print_micro_mesh_repair": result.metadata.get("print_micro_mesh_repair", {}),
            "selective_open_boundary_weld": weld_record,
            "source_ear_restoration": source_ear_restoration,
            "residual_boundary_closure_applied": bool(closure_applied),
            "residual_boundary_faces_added": int(len(final_faces) - faces_before_closure),
            "initial_topology": initial_validation,
            "preclosure_topology": preclosure_validation,
            "final_topology": final_validation,
            "orientation_repair": orientation_record,
        }
        result.metadata["source_preserving_finalization"] = audit
        result.metadata["orientation_repair"] = orientation_record

        if bool(policy.require_watertight) and (
            int(final_validation["open_edges"]) > 0
            or int(final_validation["over_shared_edges"]) > 0
            or int(final_validation["inconsistent_shared_edges"]) > 0
        ):
            raise ValueError(
                "source-preserving finalization did not produce strict topology: "
                + json.dumps(final_validation, ensure_ascii=False, sort_keys=True)
            )
        case.accept(result,audit)
        return SourcePreservingFinalizationResult(mesh=result, audit=audit)


def finalize_source_preserving_mesh(
    mesh: trimesh.Trimesh,
    protected_source_face_count: int,
) -> trimesh.Trimesh:
    return SourcePreservingMeshFinalizer.finalize(
        mesh,
        SourcePreservingFinalizationPolicy(
            protected_source_face_count=int(protected_source_face_count),
        ),
    ).mesh
