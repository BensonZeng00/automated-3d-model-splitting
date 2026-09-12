"""Measured assembly handoff; independent mesh and package gates stay strict."""
from itertools import combinations

import numpy as np
import trimesh

from .assembly_overlap import measure_overlap
from .overlap_policy import (validate_ignore_overlap_threshold, validate_ignore_overlap_ratio,
                             DEFAULT_IGNORE_OVERLAP_RATIO)
from .cutting_reference import pair_cutting_volume
from .insert_visibility import assess_insert_visibility


def measure_manual_adjustment(parts, vertices, faces, labels, *, failure=None,
                              volume_tolerance_mm3, depth_tolerance_mm, area_budget_mm2,
                              ignore_overlap_below_mm3=0.0,
                              ignore_overlap_ratio=DEFAULT_IGNORE_OVERLAP_RATIO):
    validate_ignore_overlap_threshold(ignore_overlap_below_mm3)
    validate_ignore_overlap_ratio(ignore_overlap_ratio)
    reference = trimesh.Trimesh(vertices, faces, process=False)
    by_id = {part['part_id'].split('_')[0]: part for part in parts}
    pairs, interfaces = [], []
    affected = set()
    for first, second in combinations(parts, 2):
        a, b = first['mesh'], second['mesh']
        if np.any(a.bounds[1] <= b.bounds[0]) or np.any(b.bounds[1] <= a.bounds[0]):
            overlap = dict(accepted=True, intersection_mm3=0.0)
        else:
            overlap = measure_overlap(a, b, volume_tolerance_mm3=volume_tolerance_mm3,
                                      depth_tolerance_mm=depth_tolerance_mm,
                                      ignore_overlap_below_mm3=ignore_overlap_below_mm3,
                                      ignore_overlap_ratio=ignore_overlap_ratio,
                                      cutting_volume_mm3=pair_cutting_volume(first, second),
                                      area_budget_mm2=area_budget_mm2)
        pairs.append(dict(first=first['part_id'], second=second['part_id'], **overlap))
        if not overlap['accepted']:
            affected.update((first['part_id'], second['part_id']))
    for key, part in by_id.items():
        parent_id = part.get('annotation', {}).get('parent_part')
        if parent_id not in by_id:
            continue
        ids = np.flatnonzero(np.asarray(labels) == int(key[1:]))
        if not len(ids):
            continue
        patch = reference.submesh([ids], append=True, repair=False)
        visibility = assess_insert_visibility(patch, part['mesh'], by_id[parent_id]['mesh'],
                                              reference=reference)
        covered = (visibility.get('covered_fraction', 0) > .01
                   and visibility.get('covered_area_estimate_mm2', 0) > area_budget_mm2)
        interfaces.append(dict(part_id=part['part_id'], parent_part=parent_id,
                               visibility=visibility, manual_adjustment_required=covered))
        if covered:
            affected.update((part['part_id'], by_id[parent_id]['part_id']))
    required = bool(affected or failure)
    return dict(valid=not required, status='manual_adjustment_required' if required else 'validated',
                manual_adjustment_required=required, failure=failure,
                method='all_pair_intersection_and_source_front_ownership',
                affected_parts=sorted(affected), final_pairs=pairs, interfaces=interfaces,
                corrected_parts=0, failed_seating_transforms_committed=False,
                overlap_tolerance_mm3=volume_tolerance_mm3,
                ignore_overlap_below_mm3=ignore_overlap_below_mm3,
                ignore_overlap_ratio=ignore_overlap_ratio,
                penetration_tolerance_mm=depth_tolerance_mm)


def annotate_manual_adjustment(parts, assembly, visual=None):
    """Keep raw validation failures visible inside every delivered 3MF object."""
    required = bool(assembly.get('manual_adjustment_required')
                    or (visual is not None and not visual.get('valid', False)))
    if not required:
        for part in parts:
            part.get('annotation', {}).pop('assembly_fit', None)
        return False
    for part in parts:
        part.setdefault('annotation', {})['assembly_fit'] = dict(
            status='manual_adjustment_required', automatically_fitted=False,
            affected_parts=assembly.get('affected_parts', []),
            unresolved_pairs=[p for p in assembly.get('final_pairs', []) if not p['accepted']],
            interfaces=[p for p in assembly.get('interfaces', [])
                        if p.get('manual_adjustment_required')],
            seating_failure=assembly.get('failure'),
            visual_errors=[] if visual is None else visual.get('errors', []),
            instruction='请观察模型并判断这些装配差异是否影响打印，必要时手动调整。')
    return True
