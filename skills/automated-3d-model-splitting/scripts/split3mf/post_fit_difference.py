"""Opt-in parent-priority trimming of already scaled final assembly inserts."""
from copy import deepcopy
from pathlib import Path

import manifold3d
import numpy as np
import trimesh

from .assembly_seating import SeatingError
from .local_connectors import _mesh64_arrays, _manifold64
from .post_fit_audit import audit_cut_surface, nearest_faces, audit_volume_partition
from .reporting import runtime_log
from .assembly_overlap import measure_patch_bounds
from .overlap_policy import (overlap_is_ignored, validate_ignore_overlap_threshold,
                             ratio_volume_limit, ratio_measurement, validate_ignore_overlap_ratio)
from .cutting_reference import pair_cutting_volume
from .mesh import remove_new_redundant_boolean_micro_shells


FACE_FIELDS = ('face_color_hexes', 'face_filament_slot_indices', 'face_color_codes')


def _tagged_solid(mesh, offset=0):
    vertices, faces = _mesh64_arrays(mesh)
    solid = manifold3d.Manifold(manifold3d.Mesh64(
        vertices, faces, face_id=np.arange(offset, offset + len(faces), dtype=np.uint64)))
    if str(solid.status()) != 'Error.NoError':
        raise ValueError(f'post-fit input is not a valid solid: {solid.status()}')
    return solid


def subtract_parent(insert, parent, *, volume_tolerance_mm3=1e-8,
                    depth_tolerance_mm=0.0, area_budget_mm2=1.0,
                    ignore_overlap_below_mm3=0.0, cutting_volume_mm3=None,
                    ignore_overlap_ratio=0.0):
    """Pure Boolean with exact retained-face provenance and child-owned cut faces."""
    validate_ignore_overlap_threshold(ignore_overlap_below_mm3)
    ratio_limit = ratio_volume_limit(cutting_volume_mm3, ignore_overlap_ratio)
    before = insert['mesh']
    # Shared local coordinates condition both operands without changing their
    # relative pose. The symmetric center is deterministic, never searched.
    origin = (before.bounds.mean(axis=0) + parent.bounds.mean(axis=0)) * .5
    local_child, local_parent = before.copy(), parent.copy()
    local_child.apply_translation(-origin)
    local_parent.apply_translation(-origin)
    child = _tagged_solid(local_child)
    cutter = _tagged_solid(local_parent, len(before.faces))
    overlap = child ^ cutter
    if str(overlap.status()) != 'Error.NoError':
        raise ValueError('post-fit intersection failed')
    removed = abs(float(overlap.volume()))
    if overlap_is_ignored(removed, ratio_limit):
        return insert, dict(changed=False, removed_volume_mm3=0.,
                            retained_intersection_mm3=removed,
                            **ratio_measurement(removed, cutting_volume_mm3, ignore_overlap_ratio))
    if overlap_is_ignored(removed, ignore_overlap_below_mm3):
        return insert, dict(changed=False, removed_volume_mm3=0.,
                            retained_intersection_mm3=removed,
                            ignored_by_volume_policy=True,
                            ignore_overlap_below_mm3=ignore_overlap_below_mm3)
    if removed <= volume_tolerance_mm3:
        return insert, dict(changed=False, removed_volume_mm3=removed)
    if depth_tolerance_mm > 0:
        overlap_raw = overlap.to_mesh64()
        overlap_mesh = trimesh.Trimesh(np.asarray(overlap_raw.vert_properties)[:, :3],
                                      np.asarray(overlap_raw.tri_verts), process=False)
        bounds = measure_patch_bounds(overlap_mesh, depth_tolerance_mm=depth_tolerance_mm,
                                      area_budget_mm2=area_budget_mm2)
        if bounds['accepted']:
            return insert, dict(changed=False, removed_volume_mm3=0.,
                                retained_intersection_mm3=removed,
                                accepted_physical_overlap=bounds)
    result = child - cutter
    if str(result.status()) != 'Error.NoError' or result.is_empty():
        raise ValueError('post-fit difference failed or erased the insert')
    raw = result.to_mesh64()
    mesh = trimesh.Trimesh(np.asarray(raw.vert_properties)[:, :3] + origin,
                           np.asarray(raw.tri_verts), process=False)
    owners = np.asarray(raw.face_id, dtype=np.int64)
    raw_export_volume = float(mesh.volume)
    before_shells = before.split(only_watertight=False)
    after_shells = mesh.split(only_watertight=False)
    cleanup = None
    if len(after_shells) > len(before_shells):
        cleaned, cleanup = remove_new_redundant_boolean_micro_shells(
            mesh, before, include_face_indices=True)
        if cleanup['applied']:
            retained = np.asarray(cleanup.pop('retained_face_indices'), dtype=np.int64)
            np.testing.assert_array_equal(cleaned.triangles, mesh.triangles[retained])
            owners = owners[retained]
            mesh = cleaned
            after_shells = mesh.split(only_watertight=False)
    # A negative oriented cavity is not a detached printable fragment.
    before_materials = sum(bool(shell.volume > 0) for shell in before_shells)
    after_materials = sum(bool(shell.volume > 0) for shell in after_shells)
    record = dict(changed=True, removed_volume_mm3=removed,
                  numerical_common_origin_mm=origin.tolist(),
                  volume_before_mm3=float(before.volume), volume_after_mm3=float(mesh.volume),
                  shells_before=len(before_shells), shells_after=len(after_shells),
                  material_components_before=before_materials,
                  material_components_after=after_materials)
    if cleanup is not None:
        record['micro_shell_cleanup'] = cleanup
    if (not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0
            or before_materials != after_materials):
        raise SeatingError('post-fit difference changed solid connectivity or topology', record)
    mass_balance = audit_volume_partition(child, result, overlap,
        depth_tolerance_mm=depth_tolerance_mm, area_budget_mm2=area_budget_mm2)
    record['volume_balance'] = mass_balance
    # Partition audit applies to the exact Boolean before measured cleanup.
    # Account for removed dust explicitly rather than loosening conservation.
    export_delta = abs(float(raw_export_volume - result.volume()))
    record['cleanup_signed_volume_change_mm3'] = raw_export_volume-float(mesh.volume)
    record['kernel_export_volume_error_mm3'] = export_delta
    if not mass_balance['valid'] or export_delta > max(1e-6, abs(mesh.volume) * 1e-9):
        raise SeatingError('post-fit difference failed volume conservation', record)
    residual_solid = _manifold64(mesh) ^ _manifold64(parent)
    if str(residual_solid.status()) != 'Error.NoError':
        raise SeatingError('post-fit residual intersection failed', record)
    residual = abs(float(residual_solid.volume()))
    record['residual_intersection_mm3'] = residual
    record['residual_ignored_by_volume_policy'] = overlap_is_ignored(
        residual, ignore_overlap_below_mm3)
    record['residual_cut_volume_policy'] = ratio_measurement(
        residual, cutting_volume_mm3, ignore_overlap_ratio)
    if (residual > volume_tolerance_mm3 and not record['residual_ignored_by_volume_policy']
            and not record['residual_cut_volume_policy']['ignored_by_cut_volume_ratio']):
        raise SeatingError('post-fit difference left residual intersection', record)
    generated = owners >= len(before.faces)
    # The visual validator uses a source-face prefix. Preserve that contract
    # explicitly rather than treating Manifold's reordered triangles as source.
    source_mask = (~generated) & (owners < int(insert.get('source_surface_face_count', 0)))
    order = np.argsort(~source_mask, kind='stable')
    mesh.faces = np.asarray(mesh.faces)[order]
    owners, generated = owners[order], generated[order]
    if generated.any():
        owners[generated] = nearest_faces(before, mesh.triangles_center[generated])[0]
    candidate = dict(insert, mesh=mesh, annotation=deepcopy(insert.get('annotation', {})))
    candidate['source_surface_face_count'] = int(source_mask.sum())
    for field in FACE_FIELDS:
        if insert.get(field) is not None:
            values = np.asarray(insert[field])
            if len(values) != len(before.faces):
                raise ValueError(f'{field}: source color count does not match faces')
            candidate[field] = values[owners].tolist()
    # Old mesh metadata is face-indexed and is no longer safe after a Boolean.
    mesh.metadata = {key: deepcopy(value) for key, value in before.metadata.items()
                     if not key.startswith('face_')}
    record.update(new_cut_faces=int(generated.sum()),
                  color_provenance='retained_face_id; new_faces_nearest_insert_surface')
    return candidate, record


def trim_assembly(parts, source_vertices, source_faces, source_labels, *,
                  volume_tolerance_mm3=1e-8, area_budget_mm2=1.0,
                  surface_tolerance_mm=0.05, recovery_dir=None, depth_tolerance_mm=0.0,
                  defer_failed_difference=False, ignore_overlap_below_mm3=0.0,
                  ignore_overlap_ratio=0.0):
    """Return a private candidate assembly. Never mutate source or caller parts."""
    validate_ignore_overlap_threshold(ignore_overlap_below_mm3)
    validate_ignore_overlap_ratio(ignore_overlap_ratio)
    candidates = [dict(p, mesh=p['mesh'].copy(), annotation=deepcopy(p.get('annotation', {})))
                  for p in parts]
    by_id = {p['part_id'].split('_')[0]: p for p in candidates}
    if len(by_id) != len(parts):
        raise ValueError('duplicate part identities')

    def ancestors(key):
        chain, seen = [], {key}
        parent = by_id[key]['annotation'].get('parent_part')
        while parent is not None:
            if parent not in by_id or parent in seen:
                raise ValueError('missing or cyclic assembly parent')
            chain.append(parent)
            seen.add(parent)
            parent = by_id[parent]['annotation'].get('parent_part')
        return list(reversed(chain))

    chains = {key: ancestors(key) for key in by_id}
    records = []
    source = trimesh.Trimesh(source_vertices, source_faces, process=False)
    for key in sorted(by_id, key=lambda k: (len(chains[k]), k)):
        part = by_id[key]
        if not chains[key]:
            continue
        before, steps = part['mesh'], []
        for parent_id in chains[key]:
            runtime_log('配合', 'post_fit_difference_start', '检查缩小公件与最终母件的差集',
                        part_id=key, parent_part=parent_id)
            try:
                part, step = subtract_parent(part, by_id[parent_id]['mesh'],
                                            volume_tolerance_mm3=volume_tolerance_mm3,
                                            depth_tolerance_mm=depth_tolerance_mm,
                                            ignore_overlap_below_mm3=ignore_overlap_below_mm3,
                                            ignore_overlap_ratio=ignore_overlap_ratio,
                                            cutting_volume_mm3=pair_cutting_volume(part, by_id[parent_id]),
                                            area_budget_mm2=area_budget_mm2)
            except ValueError as exc:
                if defer_failed_difference:
                    # Never commit the rejected Boolean. Leave this valid input
                    # for the caller's bounded rigid seating + all-pair checks.
                    failure = dict(parent_part=parent_id, error=str(exc),
                                   detail=getattr(exc, 'record', {}),
                                   rejected_candidate_committed=False)
                    part['annotation']['post_fit_difference_deferred_to_seating'] = failure
                    runtime_log('配合', 'rejected_difference_deferred_to_seating',
                                '差集候选未采用，保留有效输入交给受限装配检查',
                                part_id=key, **failure)
                    break
                raise SeatingError(f'{key}: {exc}', dict(part_id=key, parent_part=parent_id,
                                   detail=getattr(exc, 'record', {}))) from exc
            if step['changed']:
                steps.append(dict(parent_part=parent_id, **step))
            elif step.get('accepted_physical_overlap'):
                accepted = dict(parent_part=parent_id, **step)
                part['annotation'].setdefault('tolerance_accepted_contacts', []).append(accepted)
                runtime_log('配合', 'post_fit_overlap_within_physical_tolerance',
                            '微小相交已满足物理容差，保留原实体', part_id=key, **accepted)
        if steps:
            patch = source.submesh([np.flatnonzero(np.asarray(source_labels) == int(key[1:]))],
                                   append=True, repair=False)
            scaling = part['annotation'].get('post_split_uniform_scaling')
            if scaling:
                center = np.asarray(scaling['center_mm'])
                patch.vertices = center + np.asarray(scaling['factor_xyz']) * (patch.vertices-center)
            audit = audit_cut_surface(patch, before, part['mesh'], area_budget_mm2=area_budget_mm2,
                                      surface_tolerance_mm=surface_tolerance_mm)
            record = dict(part_id=key, method='scaled_insert_minus_final_ancestors',
                          shape_changed=True, steps=steps, surface_and_thickness=audit)
            if recovery_dir:
                directory = Path(recovery_dir) / 'post_fit_difference'
                directory.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(directory / f'{key}_unvalidated_candidate.npz',
                                    vertices=part['mesh'].vertices, faces=part['mesh'].faces)
            if not audit['valid']:
                raise SeatingError(f'{key}: post-fit difference failed surface/thickness audit', record)
            part['annotation']['post_fit_difference'] = record
            part['annotation']['fit_strategy'] = 'exact_subtract_then_uniform_scale_then_parent_difference'
            by_id[key] = part
            records.append(record)
            runtime_log('配合', 'post_fit_difference_done', '差集完成，表面和厚度抽检通过', **record)
    return [by_id[p['part_id'].split('_')[0]] for p in parts], records
