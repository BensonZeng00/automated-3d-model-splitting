"""Preserve the complete current source solid before exact child subtraction."""
from __future__ import annotations

import numpy as np
import trimesh

from .interface_retopology import InterfaceRetopologyService
from .mesh_finalization import finalize_source_preserving_mesh


def build_complete_boolean_parent(vertices, faces, *, part_id, cut_refs,
                                  interface_retopology):
    """Keep curved child patches and inherited contact shells until subtraction.

    Closing holes in the body component with rim-only caps replaces the child's
    curved exterior. On a recessed patch that creates material outside the
    cutter, leaving an opaque membrane even after a valid solid difference.
    The current recursive input already supplies the authoritative closed shell.
    """
    points = np.asarray(vertices, dtype=np.float64).copy()
    triangles = np.asarray(faces, dtype=np.int64).copy()
    loops = [list(ref['global_loop']) for ref in cut_refs]
    points, records = InterfaceRetopologyService.retopologize_local_loops(
        points, loops, np.arange(len(points), dtype=np.int64),
        interface_retopology, local_faces=triangles,
    )
    mesh = trimesh.Trimesh(points, triangles, process=False,
                           metadata={'name': part_id})
    if not mesh.is_watertight:
        raise ValueError('complete recursive source must be closed before exact subtraction')
    mesh = finalize_source_preserving_mesh(
        mesh, protected_source_face_count=len(triangles),
    )
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        raise ValueError('complete recursive source must be closed before exact subtraction')
    return mesh, {
        'part_id': part_id,
        'interface_geometry': 'local-connector',
        'selected_processing_mode': 'body',
        'processing_mode': 'complete_source_then_exact_child_difference',
        'parent_closure_source': 'complete_current_recursive_input',
        'source_faces': len(triangles),
        'output_faces': len(mesh.faces),
        'output_vertices': len(mesh.vertices),
        'interface_retopology_records': records,
        'side_faces_added': 0,
        'cap_faces_added': 0,
        'geometry_cap_mode': 'exact_child_subtraction',
        'fit_clearance_mm': 0.0,
        'socket_overcut_mm': 0.0,
        'bottom_clearance_mm': 0.0,
        'bbox_min': mesh.bounds[0].tolist(),
        'bbox_max': mesh.bounds[1].tolist(),
        'bbox_size_mm': mesh.extents.tolist(),
    }
