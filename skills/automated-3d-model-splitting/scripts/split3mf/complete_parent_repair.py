"""Explicit recovery of an intact current parent; never a socket reconstruction."""
from collections import Counter
import numpy as np
from .local_connectors import _manifold64
from .mesh_finalization import finalize_source_preserving_mesh
from .print_tolerance import current
from .surface_preservation import face_key
from .validation import validate_mesh_in_memory


def repair_complete_parent(mesh):
    """Reuse source-preserving cleanup and reject broad added closure surfaces."""
    before = validate_mesh_in_memory(mesh)
    candidate = finalize_source_preserving_mesh(mesh, protected_source_face_count=len(mesh.faces))
    available = Counter(face_key(triangle) for triangle in mesh.triangles)
    added_area = 0.
    for triangle, area in zip(candidate.triangles, candidate.area_faces):
        key = face_key(triangle)
        if available[key]:
            available[key] -= 1
        else:
            added_area += float(area)
    if added_area > current().micro_area_mm2 + 1e-12:
        raise ValueError(f'Complete-parent recovery adds {added_area:.9g} mm2 of surface; '
                         f'local budget is {current().micro_area_mm2:.9g} mm2')
    after = validate_mesh_in_memory(candidate)
    solid = _manifold64(candidate)
    if (not after['watertight'] or not after['winding_consistent']
            or not after['all_closed_components_outward']
            or str(solid.status()) != 'Error.NoError' or solid.is_empty()):
        raise ValueError('Repaired complete parent failed topology or kernel validation')
    return candidate, dict(valid=True, before=before, after=after,
                           added_or_retriangulated_area_mm2=added_area,
                           kernel_status=str(solid.status()),
                           source_finalization=candidate.metadata['source_preserving_finalization'],
                           scope='intact_current_parent_only; interfaces_and_assembly_not_validated')
