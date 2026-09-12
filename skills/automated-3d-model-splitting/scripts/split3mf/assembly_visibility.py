"""Validate compact insert visibility and correct bounded final seating poses."""
from __future__ import annotations

import numpy as np
import trimesh
import json
from pathlib import Path
from itertools import combinations
from .assembly_overlap import measure_overlap, union_mesh
from .overlap_policy import (DEFAULT_IGNORE_OVERLAP_RATIO, validate_ignore_overlap_threshold,
                             validate_ignore_overlap_ratio)
from .cutting_reference import pair_cutting_volume

from .assembly_seating import seat_insert_outward, SeatingError
from .insert_visibility import assess_insert_visibility


def validate_and_seat_assembly(parts, source_vertices, source_faces, source_labels,
                               *, maximum_travel_mm=0.1, area_budget_mm2=1.0,
                               allow_coupled_seating=False,
                               overlap_tolerance_mm3=1e-8,
                               penetration_tolerance_mm=0.0, recovery_dir=None,
                               post_fit_difference=False, surface_tolerance_mm=0.05,
                               allow_manual_adjustment=False,
                               ignore_overlap_below_mm3=0.0,
                               ignore_overlap_ratio=DEFAULT_IGNORE_OVERLAP_RATIO):
    """Capture every failure, preserving caller meshes until all gates pass."""
    validate_ignore_overlap_threshold(ignore_overlap_below_mm3)
    validate_ignore_overlap_ratio(ignore_overlap_ratio)
    if recovery_dir:
        from .assembly_case import save_assembly_case
        save_assembly_case(Path(recovery_dir) / 'assembly_case', parts, source_vertices,
                           source_faces, source_labels, dict(
            maximum_travel_mm=maximum_travel_mm, area_budget_mm2=area_budget_mm2,
            allow_coupled_seating=allow_coupled_seating,
            overlap_tolerance_mm3=overlap_tolerance_mm3,
            ignore_overlap_below_mm3=ignore_overlap_below_mm3,
            ignore_overlap_ratio=ignore_overlap_ratio,
            penetration_tolerance_mm=penetration_tolerance_mm,
            post_fit_difference=post_fit_difference, surface_tolerance_mm=surface_tolerance_mm,
            allow_manual_adjustment=allow_manual_adjustment))
    try:
        candidates, trimming = parts, []
        if post_fit_difference:
            from .post_fit_difference import trim_assembly
            candidates, trimming = trim_assembly(
                parts, source_vertices, source_faces, source_labels,
                volume_tolerance_mm3=overlap_tolerance_mm3,
                ignore_overlap_below_mm3=ignore_overlap_below_mm3,
                ignore_overlap_ratio=ignore_overlap_ratio,
                depth_tolerance_mm=penetration_tolerance_mm,
                area_budget_mm2=area_budget_mm2, surface_tolerance_mm=surface_tolerance_mm,
                recovery_dir=recovery_dir, defer_failed_difference=True)
        try:
            result = _validate_and_seat(candidates, source_vertices, source_faces, source_labels,
                                  maximum_travel_mm=maximum_travel_mm,
                                  area_budget_mm2=area_budget_mm2,
                                  allow_coupled_seating=allow_coupled_seating,
                                  overlap_tolerance_mm3=overlap_tolerance_mm3,
                                  ignore_overlap_below_mm3=ignore_overlap_below_mm3,
                                  ignore_overlap_ratio=ignore_overlap_ratio,
                                  penetration_tolerance_mm=penetration_tolerance_mm,
                                      recovery_dir=recovery_dir)
        except SeatingError as exc:
            if not allow_manual_adjustment:
                raise
            from .assembly_review import measure_manual_adjustment
            result = measure_manual_adjustment(candidates, source_vertices, source_faces,
                source_labels, failure=dict(error=str(exc), detail=exc.record),
                volume_tolerance_mm3=overlap_tolerance_mm3,
                ignore_overlap_below_mm3=ignore_overlap_below_mm3,
                ignore_overlap_ratio=ignore_overlap_ratio,
                depth_tolerance_mm=penetration_tolerance_mm, area_budget_mm2=area_budget_mm2)
        from .assembly_review import annotate_manual_adjustment
        annotate_manual_adjustment(candidates, result)
        if post_fit_difference:
            for original, candidate in zip(parts, candidates):
                original.update(candidate)
        result['post_fit_difference'] = trimming
        if recovery_dir:
            (Path(recovery_dir) / 'assembly_result.json').write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        return result
    except ValueError as exc:
        evidence = dict(error=str(exc), detail=getattr(exc, 'record', {}),
                        penetration_tolerance_mm=penetration_tolerance_mm,
                        overlap_tolerance_mm3=overlap_tolerance_mm3)
        evidence['ignore_overlap_below_mm3'] = ignore_overlap_below_mm3
        evidence['ignore_overlap_ratio'] = ignore_overlap_ratio
        if recovery_dir:
            directory = Path(recovery_dir) / 'assembly_failure'
            directory.mkdir(parents=True, exist_ok=True)
            (directory / 'summary.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
            for part in parts:
                key = part['part_id'].split('_')[0]
                mesh = part['mesh']
                np.savez_compressed(directory / f'{key}_input_scaled.npz',
                                    vertices=mesh.vertices, faces=mesh.faces)
        raise


def _validate_and_seat(parts, source_vertices, source_faces, source_labels,
                      *, maximum_travel_mm, area_budget_mm2, allow_coupled_seating,
                      overlap_tolerance_mm3, penetration_tolerance_mm, recovery_dir,
                      ignore_overlap_below_mm3, ignore_overlap_ratio):
    if not np.isfinite(penetration_tolerance_mm) or penetration_tolerance_mm < 0:
        raise ValueError('penetration tolerance must be finite and nonnegative')
    if not np.isfinite(overlap_tolerance_mm3) or overlap_tolerance_mm3 <= 0:
        raise ValueError('overlap tolerance must be finite and positive')
    reference = trimesh.Trimesh(source_vertices, source_faces, process=False)
    by_id = {part['part_id'].split('_')[0]: part for part in parts}
    if len(by_id) != len(parts):
        raise ValueError('duplicate part identities')
    parent_ids = {part.get('annotation', {}).get('parent_part') for part in parts}
    children_by_parent = {}
    for part in parts:
        parent_id = part.get('annotation', {}).get('parent_part')
        children_by_parent.setdefault(parent_id, []).append(part['part_id'].split('_')[0])

    def descendants(part_id):
        result, pending = set(), [part_id]
        while pending:
            current = pending.pop()
            if current in result:
                continue
            result.add(current)
            pending.extend(children_by_parent.get(current, ()))
        return result

    translations = {part_id: np.zeros(3, dtype=np.float64) for part_id in by_id}

    def translated_mesh(part):
        part_id = part['part_id'].split('_')[0]
        mesh = part['mesh'].copy()
        mesh.vertices = np.asarray(mesh.vertices) + translations[part_id]
        return mesh

    def depth(part):
        seen = set()
        parent = part.get('annotation', {}).get('parent_part')
        while parent in by_id:
            if parent in seen:
                raise ValueError('assembly parent cycle')
            seen.add(parent)
            parent = by_id[parent].get('annotation', {}).get('parent_part')
        return len(seen)

    def collision(first, second, cutting_volume):
        if np.any(first.bounds[1] <= second.bounds[0]) or np.any(second.bounds[1] <= first.bounds[0]):
            return {'accepted': True, 'intersection_mm3': 0.0}
        return measure_overlap(first, second, depth_tolerance_mm=penetration_tolerance_mm,
                               area_budget_mm2=area_budget_mm2,
                               ignore_overlap_below_mm3=ignore_overlap_below_mm3,
                               cutting_volume_mm3=cutting_volume,
                               ignore_overlap_ratio=ignore_overlap_ratio,
                               volume_tolerance_mm3=overlap_tolerance_mm3)

    updates, records = [], []
    for part in sorted(parts, key=depth):
        index = int(part['part_id'].split('_')[0][1:])
        annotation = part.get('annotation', {})
        parent_id = annotation.get('parent_part')
        if annotation.get('selected_processing_mode') != 'inward' or parent_id not in by_id:
            continue
        source_ids = np.flatnonzero(np.asarray(source_labels) == index)
        if not len(source_ids):
            continue
        patch = reference.submesh([source_ids], append=True, repair=False)
        current_mesh = translated_mesh(part)
        parent = translated_mesh(by_id[parent_id])
        before = assess_insert_visibility(patch, current_mesh, parent, reference=reference)
        record = {'part_id': part['part_id'], 'parent_part': parent_id, 'before': before}
        collision_record = collision(current_mesh, parent, pair_cutting_volume(part, by_id[parent_id]))
        overlap = collision_record['intersection_mm3']
        record['collision'] = collision_record
        record['intersection_before_mm3'] = overlap

        def obstructed(measurement):
            return (measurement.get('covered_fraction', 0) > 0.01
                    and measurement.get('covered_area_estimate_mm2', 0) > area_budget_mm2)

        if obstructed(before) or not collision_record['accepted']:
            part_id = part['part_id'].split('_')[0]
            is_subassembly = part_id in parent_ids
            if is_subassembly and not allow_coupled_seating:
                raise SeatingError(f"{part['part_id']}: interfering or obstructed subassembly requires a coupled seating review", record)
            if 'inward_axis' not in before:
                raise SeatingError(f"{part['part_id']}: overlap {overlap} mm3 without a coherent seating axis", record)
            try:
                group_ids = descendants(part_id) if is_subassembly else {part_id}
                combined = union_mesh([translated_mesh(by_id[k]) for k in sorted(group_ids)])
                obstacles = union_mesh([translated_mesh(by_id[k]) for k in sorted(set(by_id) - group_ids)])
                seated, seating = seat_insert_outward(
                    combined, obstacles, before['inward_axis'],
                    maximum_travel_mm=maximum_travel_mm,
                    volume_tolerance_mm3=overlap_tolerance_mm3,
                    ignore_overlap_below_mm3=ignore_overlap_below_mm3,
                    ignore_overlap_ratio=ignore_overlap_ratio,
                    cutting_volumes_mm3=[
                        [pair_cutting_volume(by_id[k], by_id[o])
                         for o in sorted(set(by_id)-group_ids)] for k in sorted(group_ids)],
                    penetration_tolerance_mm=penetration_tolerance_mm,
                    area_budget_mm2=area_budget_mm2,
                    obstacles=[translated_mesh(by_id[k]) for k in sorted(set(by_id)-group_ids)],
                    moving_parts=[translated_mesh(by_id[k]) for k in sorted(group_ids)],
                )
            except SeatingError as exc:
                exc.record.update(part_id=part['part_id'], parent_part=parent_id,
                                  visibility_before=before)
                if recovery_dir:
                    directory = Path(recovery_dir) / 'assembly_failure'
                    directory.mkdir(parents=True, exist_ok=True)
                    (directory / 'failure.json').write_text(json.dumps(exc.record, indent=2), encoding='utf-8')
                    for key, value in by_id.items():
                        mesh = translated_mesh(value)
                        np.savez_compressed(directory / f'{key}.npz', vertices=mesh.vertices, faces=mesh.faces)
                raise SeatingError(f"{part['part_id']}: {exc}", exc.record) from exc
            delta = np.asarray(seating['translation_mm'])
            seated = current_mesh.copy()
            seated.apply_translation(delta)
            after = assess_insert_visibility(patch, seated, parent, reference=reference)
            if obstructed(after):
                raise SeatingError(f"{part['part_id']}: parent still covers the seated insert", dict(**record, after=after))
            record.update(seating=seating, after=after)
            translation = np.asarray(seated.vertices) - np.asarray(current_mesh.vertices)
            if len(translation):
                translation = translation[0]
            if is_subassembly:
                group_ids = descendants(part_id)
                for group_id in group_ids:
                    translations[group_id] += translation
                seating = dict(seating, method='rigid_coupled_subassembly_seating_after_uniform_scaling',
                               coupled_part_ids=sorted(group_ids))
                record['coupled_part_ids'] = sorted(group_ids)
            else:
                group_ids = {part_id}
                translations[part_id] += translation
            updates.append((group_ids, seating))
        records.append(record)
    final_pairs = []
    for first, second in combinations(parts, 2):
        result = collision(translated_mesh(first), translated_mesh(second), pair_cutting_volume(first, second))
        final_pairs.append(dict(first=first['part_id'], second=second['part_id'], **result))
        if not result['accepted']:
            raise SeatingError('final pair collision', final_pairs[-1])
    for part in parts:
        key = part['part_id'].split('_')[0]
        if np.linalg.norm(translations[key]) > maximum_travel_mm + 1e-12:
            raise SeatingError('cumulative seating travel exceeded', {'part_id': key})
        parent_id = part.get('annotation', {}).get('parent_part')
        if parent_id in by_id:
            ids = np.flatnonzero(np.asarray(source_labels) == int(key[1:]))
            patch = reference.submesh([ids], append=True, repair=False)
            final = assess_insert_visibility(patch, translated_mesh(part), translated_mesh(by_id[parent_id]), reference=reference)
            if obstructed(final):
                raise SeatingError('final ownership check failed', dict(part_id=key, visibility=final))
    # Commit only after every interface passes. The source, cutters, and parent
    # meshes remain unchanged; moving a non-leaf independently is prohibited.
    for part_id, part in by_id.items():
        translation = translations[part_id]
        if np.any(translation):
            part['mesh'].vertices = np.asarray(part['mesh'].vertices) + translation
    for part_id, translation in translations.items():
        if np.any(translation):
            by_id[part_id]['annotation']['post_fit_seating'] = dict(
                method='tree_ordered_coupled_seating', translation_mm=translation.tolist(),
                total_travel_mm=float(np.linalg.norm(translation)), shape_changed=False,
                steps=[seating for group, seating in updates if part_id in group])
    return {'valid': True, 'method': 'exact_insert_parent_ray_ownership',
            'maximum_seating_travel_mm': maximum_travel_mm,
            'corrected_parts': sum(bool(np.any(t)) for t in translations.values()),
            'coupled_seating_confirmed': bool(allow_coupled_seating),
            'overlap_tolerance_mm3': float(overlap_tolerance_mm3),
            'ignore_overlap_below_mm3': ignore_overlap_below_mm3,
            'ignore_overlap_ratio': ignore_overlap_ratio,
            'penetration_tolerance_mm': penetration_tolerance_mm,
            'final_pairs': final_pairs,
            'interfaces': records}
