"""Collect every participating child rim before any child cap is precomputed."""
from dataclasses import replace
import hashlib
import numpy as np
from .layer_surface_plan import build_layer_surface_plan


_BOUNDARY_TOPOLOGY_IMPLEMENTATION = "boundary-loops-v1"


def layer_child_boundary_topology(faces, components, subtree, context):
    """Return cached local connectivity for one recursive subtree.

    Vertex positions are deliberately not cached: seam planning may move them.
    The expensive face extraction, dense remap, and boundary walk depend only
    on connectivity, which is authenticated by the key on every lookup.
    """
    from .mesh import build_local_mesh_from_faces, boundary_loops

    subtree_key = tuple(sorted(int(index) for index in subtree))
    face_array = np.asarray(faces, dtype=np.int64)
    global_faces = np.concatenate(
        [components[index - 1].global_faces for index in subtree_key]
    ).astype(np.int64, copy=False)
    selected_faces = np.ascontiguousarray(face_array[global_faces], dtype=np.int64)
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(global_faces).tobytes())
    digest.update(selected_faces.tobytes())
    digest.update(np.asarray(selected_faces.shape, dtype=np.int64).tobytes())
    key = (_BOUNDARY_TOPOLOGY_IMPLEMENTATION, digest.hexdigest(), subtree_key)
    cached = context.layer_boundary_topologies.get(key) if context is not None else None
    if cached is not None:
        return cached

    # Vertex ids can exceed 3*face_count only for malformed/sparse synthetic
    # inputs, so size from the actual maximum rather than assuming density.
    vertex_count = int(face_array.max()) + 1 if face_array.size else 0
    placeholder_vertices = np.empty((vertex_count, 3), dtype=np.float64)
    _, local_faces, _, global_vertex_ids = build_local_mesh_from_faces(
        placeholder_vertices, face_array, global_faces
    )
    result = (local_faces, global_vertex_ids, boundary_loops(local_faces))
    if context is not None:
        context.layer_boundary_topologies[key] = result
    return result


def prepare_layer_seams(vertices, faces, components, parent_index, children,
                        assembly_children, neighbor_lookup, context):
    if context is None:
        return vertices, faces, context
    # Lazy import avoids the inward builder/module dependency cycle.
    from .inward import subtree_component_indices, boundary_loop_parent_contact
    from .micro_interfaces import filter_micro_interface_loops
    loops = []
    for child in sorted(map(int, children)):
        subtree = subtree_component_indices(child, assembly_children)
        triangles, ids, child_loops = layer_child_boundary_topology(
            faces, components, subtree, context
        )
        points = np.asarray(vertices)[ids]
        records = []
        for index, loop in enumerate(child_loops):
            contact = boundary_loop_parent_contact(loop, ids, neighbor_lookup, parent_index, set(subtree))
            if int(contact['parent_edges']) >= 3:
                records.append(dict(loop_index=index, loop=loop))
        records, _ = filter_micro_interface_loops(points, records, part_index=child)
        loops.extend(ids[np.asarray(record['loop'], dtype=np.int64)].tolist() for record in records)
    groups = np.zeros(len(faces), dtype=np.int64)
    for index, component in enumerate(components, 1):
        groups[np.asarray(component.global_faces, dtype=np.int64)] = index
    plan = build_layer_surface_plan(vertices, faces, loops, context, groups)
    context.layer_seams.pop(int(parent_index), None)
    if plan is None:
        return vertices, faces, replace(context, active_layer_seam=None)
    context.layer_seams[int(parent_index)] = plan
    return plan.vertices, plan.faces, replace(context, active_layer_seam=plan)


def activate_layer_seams(context, parent_index, vertices, faces):
    plan = context.layer_seams.get(int(parent_index))
    if plan is None:
        return vertices, faces, replace(context, active_layer_seam=None)
    if plan.config != context.config:
        raise ValueError('Layer seam settings changed after planning')
    import hashlib
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(vertices, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(faces, dtype=np.int64).tobytes())
    if digest.hexdigest() != plan.report['source_geometry_sha256']:
        raise ValueError('Layer seam cache does not match the current recursive input')
    return plan.vertices, plan.faces, replace(context, active_layer_seam=plan)
