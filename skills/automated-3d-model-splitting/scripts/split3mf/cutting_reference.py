"""Frozen actual socket-removal volumes, before final insert fitting."""
from .overlap_policy import validate_ignore_overlap_threshold


def attach_cutting_references(parts, parent_records):
    """Map existing sequential Boolean measurements onto their emitted children."""
    by_id = {p['part_id'].split('_')[0]: p for p in parts}
    for parent_id, record in parent_records.items():
        children = record.get('child_indices', [])
        seen = set()
        for step in record.get('cutter_steps', []):
            index = int(step['cutter_index'])
            if not 0 <= index < len(children) or index in seen:
                raise ValueError('invalid or duplicate cutting reference index')
            seen.add(index)
            child_id = f'P{int(children[index]):02d}'
            part = by_id.get(child_id)
            if part is None or part.get('annotation', {}).get('parent_part') != parent_id:
                raise ValueError('cutting reference does not match the assembly tree')
            volume = float(step['intersection_volume_mm3'])
            validate_ignore_overlap_threshold(volume)
            part['annotation']['assembly_cutting_reference'] = dict(
                parent_part=parent_id, cutting_volume_mm3=volume,
                source='actual_sequential_full_size_socket_intersection',
                cutter_index=index, fixed_before_uniform_scaling=True)


def _cutting_volume(part):
    annotation = part.get('annotation', {})
    reference = annotation.get('assembly_cutting_reference', {})
    if not reference or reference.get('parent_part') != annotation.get('parent_part'):
        return None
    volume = reference.get('cutting_volume_mm3')
    if volume is not None:
        validate_ignore_overlap_threshold(volume)
    return volume


def pair_cutting_volume(first, second):
    """Direct pair: its cut; other pairs: the smaller available cut reference.

    A root has no incoming cut. Missing references on non-root parts do not
    imply permission to ignore collisions. No model volume or bbox fallback.
    """
    first_id, second_id = (p['part_id'].split('_')[0] for p in (first, second))
    first_parent = first.get('annotation', {}).get('parent_part')
    second_parent = second.get('annotation', {}).get('parent_part')
    if second_parent == first_id:
        return _cutting_volume(second)
    if first_parent == second_id:
        return _cutting_volume(first)
    volumes = []
    for part, parent in ((first, first_parent), (second, second_parent)):
        if parent is None:
            continue
        volume = _cutting_volume(part)
        if volume is None:
            return None
        volumes.append(volume)
    return min(volumes) if volumes else None
