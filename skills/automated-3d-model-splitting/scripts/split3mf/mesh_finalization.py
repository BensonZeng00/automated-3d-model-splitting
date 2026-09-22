from __future__ import annotations

import collections
from dataclasses import dataclass

from .common import *
from .validation import validate_mesh_in_memory


@dataclass(frozen=True)
class SourcePreservingFinalizationPolicy:
    """Audit policy for a generated shell with an immutable source prefix."""

    protected_source_face_count: int


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

        # Builders own interface geometry. Source defects are evidence, not
        # finalization work: never weld, cap, flip, or remove source here.
        preclosure_validation = validate_mesh_in_memory(result)
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
            "source_geometry_mutation": "none",
            "initial_topology": initial_validation,
            "preclosure_topology": preclosure_validation,
            "final_topology": final_validation,
        }
        result.metadata["source_preserving_finalization"] = audit
        result.metadata["source_topology_diagnostics"] = final_validation
        result.metadata["source_geometry_mutation"] = "none"

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
