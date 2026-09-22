"""Collect every participating child rim before any child cap is precomputed."""
from dataclasses import replace
import numpy as np
from .layer_surface_plan import build_layer_surface_plan


def prepare_layer_seams(vertices, faces, components, parent_index, children,
                        assembly_children, neighbor_lookup, context):
    if context is None:
        return vertices, faces, context
    # Lazy import avoids the inward builder/module dependency cycle.
    from .inward import (subtree_component_indices, build_subassembly_component,
                         build_local_mesh, boundary_loops, boundary_loop_parent_contact)
    from .micro_interfaces import filter_micro_interface_loops
    loops = []
    for child in sorted(map(int, children)):
        subtree = subtree_component_indices(child, assembly_children)
        component = build_subassembly_component(vertices, faces, components, subtree,
                                                 components[child-1].color_code)
        points, triangles, _, ids = build_local_mesh(vertices, faces, component)
        records = []
        for index, loop in enumerate(boundary_loops(triangles)):
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
