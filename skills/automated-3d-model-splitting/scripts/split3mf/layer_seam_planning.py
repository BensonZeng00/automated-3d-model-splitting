"""Collect every participating child rim before any child cap is precomputed."""
from dataclasses import replace
import hashlib
import numpy as np
from .layer_surface_plan import build_layer_surface_plan


_BOUNDARY_TOPOLOGY_IMPLEMENTATION = "boundary-loops-v1"


def layer_child_boundary_topology(faces, components, subtree, context,
                                  neighbor_lookup=None, recognized_boundaries=None):
    """Return cached local connectivity for one recursive subtree.

    Vertex positions are deliberately not cached: seam planning may move them.
    The expensive face extraction, dense remap, and boundary walk depend only
    on connectivity, which is authenticated by the key on every lookup.
    """
    from .mesh import build_local_mesh_from_faces, boundary_loops, boundary_cycles_from_edges

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
    if recognized_boundaries is not None:
        key += (recognized_boundaries.fingerprint,)
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
    if recognized_boundaries is None:
        loops = boundary_loops(local_faces)
    else:
        subtree_set = set(subtree_key)
        boundary_edges = set()
        for component_index in subtree_key:
            for loop in recognized_boundaries.source_loops_for_component(component_index):
                for position, left in enumerate(loop):
                    right = loop[(position + 1) % len(loop)]
                    edge = tuple(sorted((int(left), int(right))))
                    neighbors = (neighbor_lookup or {}).get(edge, {component_index})
                    neighbor_ids = set(map(int, neighbors))
                    if neighbor_ids - subtree_set or len(neighbor_ids) <= 1:
                        boundary_edges.add(edge)
        global_loops = boundary_cycles_from_edges(np.asarray(sorted(boundary_edges), dtype=np.int64).reshape((-1, 2)))
        global_to_local = np.full(vertex_count, -1, dtype=np.int64)
        global_to_local[global_vertex_ids] = np.arange(len(global_vertex_ids), dtype=np.int64)
        loops = [global_to_local[np.asarray(loop, dtype=np.int64)] for loop in global_loops]
        if any(np.any(loop < 0) for loop in loops):
            raise ValueError("recognized subtree boundary references an absent source vertex")
    result = (local_faces, global_vertex_ids, loops)
    if context is not None:
        context.layer_boundary_topologies[key] = result
    return result


def prepare_layer_seams(vertices, faces, components, parent_index, children,
                        assembly_children, neighbor_lookup, context,
                        recognized_boundaries=None):
    if context is None:
        return vertices, faces, context
    # Lazy import avoids the inward builder/module dependency cycle.
    from .part_geometry import subtree_component_indices, boundary_loop_parent_contact
    from .micro_interfaces import filter_micro_interface_loops
    loops = []
    for child in sorted(map(int, children)):
        subtree = subtree_component_indices(child, assembly_children)
        triangles, ids, child_loops = layer_child_boundary_topology(
            faces, components, subtree, context, neighbor_lookup,
            recognized_boundaries,
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
