"""Bounded rigid seating correction for a scaled insert in its exact socket."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .local_connectors import _manifold64
from .assembly_overlap import measure_overlap
from .overlap_policy import validate_ignore_overlap_threshold, validate_ignore_overlap_ratio


class SeatingError(ValueError):
    """A failed bounded search with inspectable measurements."""

    def __init__(self, message, record):
        super().__init__(message)
        self.record = record


def _contact_pose_search(overlap_at, axis, maximum_travel, tolerance, start, accepts=None):
    """Search a rigid pose in the outward cone, without changing either solid."""
    trials = []
    # A small finite-difference step is for optimization only. Acceptance still
    # uses an exact solid intersection and the original volume threshold.
    for seed in (start, -axis * maximum_travel * .9):
        objective_scale = max(tolerance * 1000., overlap_at(seed), 1e-12)
        outcome = minimize(
            lambda value: overlap_at(value) / objective_scale,
            seed, method='SLSQP', bounds=[(-maximum_travel, maximum_travel)]*3,
            constraints=[
                {'type': 'ineq', 'fun': lambda value: maximum_travel**2 - value @ value},
                {'type': 'ineq', 'fun': lambda value: -value @ axis - .5*np.linalg.norm(value)},
            ], options={'maxiter': 80, 'ftol': 1e-9, 'eps': min(1e-5, maximum_travel/1000)},
        )
        translation = np.asarray(outcome.x, dtype=float)
        length = float(np.linalg.norm(translation))
        if np.isfinite(length) and length > maximum_travel:
            translation *= maximum_travel*(1-1e-10)/length
        volume = overlap_at(translation)
        norm = float(np.linalg.norm(translation))
        outward = float(-translation @ axis)
        accepted = (np.isfinite(translation).all() and norm <= maximum_travel
                    and outward >= .5*norm-1e-12
                    and (accepts(translation) if accepts else volume <= tolerance))
        trials.append({'translation_mm': translation.tolist(), 'travel_mm': norm,
                       'intersection_mm3': volume, 'accepted': bool(accepted),
                       'optimizer_success': bool(outcome.success)})
        if accepted:
            return translation, trials
    return None, trials


def seat_insert_outward(insert, parent, inward, *, maximum_travel_mm=0.1,
                        volume_tolerance_mm3=1e-8, penetration_tolerance_mm=0.0,
                        area_budget_mm2=1.0, obstacles=None, moving_parts=None,
                        ignore_overlap_below_mm3=0.0, ignore_overlap_ratio=0.0,
                        cutting_volumes_mm3=None):
    """Translate without changing either solid or adding a clearance cutter.

    Bbox-centered scaling need not stay inside a concave socket. Search only
    along the measured assembly axis first, then inside a bounded outward
    cone. Only a measured collision-free endpoint is accepted.
    """
    validate_ignore_overlap_threshold(ignore_overlap_below_mm3)
    validate_ignore_overlap_ratio(ignore_overlap_ratio)
    axis = np.asarray(inward, dtype=np.float64)
    length = np.linalg.norm(axis)
    if not np.isfinite(length) or length <= 1e-12:
        raise ValueError('seating requires a finite nonzero inward axis')
    if not np.isfinite(maximum_travel_mm) or maximum_travel_mm <= 0:
        raise ValueError('seating travel bound must be positive and finite')
    if not np.isfinite(volume_tolerance_mm3) or volume_tolerance_mm3 <= 0:
        raise ValueError('seating volume tolerance must be positive and finite')
    if not np.isfinite(penetration_tolerance_mm) or penetration_tolerance_mm < 0:
        raise ValueError('penetration tolerance must be finite and nonnegative')
    if not np.isfinite(area_budget_mm2) or area_budget_mm2 < 0:
        raise ValueError('contact area budget must be finite and nonnegative')
    axis = axis / length
    parent_solid = _manifold64(parent)
    insert_solid = _manifold64(insert)
    obstacle_meshes = [parent] if obstacles is None else list(obstacles)
    moving_meshes = [insert] if moving_parts is None else list(moving_parts)
    if cutting_volumes_mm3 is None:
        cutting_volumes_mm3 = [[None] * len(obstacle_meshes) for _ in moving_meshes]
    if (len(cutting_volumes_mm3) != len(moving_meshes)
            or any(len(row) != len(obstacle_meshes) for row in cutting_volumes_mm3)):
        raise ValueError('cutting reference matrix must match moving/obstacle pairs')

    def contact_at(translation):
        contacts = []
        for index, mesh in enumerate(moving_meshes):
            candidate = mesh.copy()
            candidate.apply_translation(translation)
            contacts.extend(measure_overlap(candidate, obstacle, depth_tolerance_mm=penetration_tolerance_mm,
                                area_budget_mm2=area_budget_mm2,
                                volume_tolerance_mm3=volume_tolerance_mm3,
                                ignore_overlap_below_mm3=ignore_overlap_below_mm3,
                                ignore_overlap_ratio=ignore_overlap_ratio,
                                cutting_volume_mm3=cutting_volumes_mm3[index][other])
                            for other, obstacle in enumerate(obstacle_meshes))
        return contacts

    def accepts(translation):
        return all(record['accepted'] for record in contact_at(translation))

    def overlap_at(translation):
        intersection = parent_solid ^ insert_solid.translate(np.asarray(translation).tolist())
        if str(intersection.status()) != 'Error.NoError':
            raise ValueError(f'seating intersection failed: {intersection.status()}')
        return abs(float(intersection.volume()))

    def overlap(distance):
        return overlap_at(-distance*axis)

    before = overlap(0.0)
    axial_trials, pose_trials = [], []
    translation = np.zeros(3)
    lower, upper = 0.0, 0.0
    if not accepts(translation):
        for upper in np.linspace(maximum_travel_mm / 10, maximum_travel_mm, 10):
            volume = overlap(float(upper))
            axial_trials.append({'distance_mm': float(upper), 'intersection_mm3': volume})
            if accepts(-float(upper)*axis):
                break
            lower = float(upper)
        else:
            best = min(axial_trials, key=lambda item: item['intersection_mm3'])
            translation, pose_trials = _contact_pose_search(
                overlap_at, axis, maximum_travel_mm, volume_tolerance_mm3,
                -axis*best['distance_mm'], accepts=accepts)
            if translation is None:
                raise SeatingError('no collision-free seating inside the requested travel bound', {
                    'intersection_before_mm3': before, 'maximum_travel_mm': maximum_travel_mm,
                    'volume_tolerance_mm3': volume_tolerance_mm3,
                    'ignore_overlap_below_mm3': ignore_overlap_below_mm3,
                    'ignore_overlap_ratio': ignore_overlap_ratio,
                    'axial_trials': axial_trials, 'contact_pose_trials': pose_trials})
        if not pose_trials:
            for _ in range(12):
                middle = (lower + upper) * .5
                if accepts(-middle*axis):
                    upper = middle
                else:
                    lower = middle
            translation = -float(upper)*axis
    result = insert.copy()
    result.vertices = np.asarray(insert.vertices) + translation
    after = abs(float((parent_solid ^ _manifold64(result)).volume()))
    final_contacts = contact_at(translation)
    if not all(record['accepted'] for record in final_contacts):
        raise ValueError('seating result failed the final collision check')
    return result, {
        'method': 'rigid_outward_seating_after_uniform_scaling',
        'translation_mm': translation.tolist(),
        'outward_travel_mm': float(-translation @ axis),
        'total_travel_mm': float(np.linalg.norm(translation)),
        'source_inward_axis': axis.tolist(),
        'search_strategy': 'bounded_contact_pose' if pose_trials else 'source_axis',
        'axial_trials': axial_trials,
        'contact_pose_trials': pose_trials,
        'maximum_travel_mm': float(maximum_travel_mm),
        'intersection_before_mm3': before,
        'intersection_after_mm3': after,
        'volume_tolerance_mm3': float(volume_tolerance_mm3),
        'ignore_overlap_below_mm3': ignore_overlap_below_mm3,
        'ignore_overlap_ratio': ignore_overlap_ratio,
        'penetration_tolerance_mm': float(penetration_tolerance_mm),
        'area_budget_mm2': float(area_budget_mm2),
        'final_contacts': final_contacts,
        'shape_changed': False, 'parent_modified': False,
    }
