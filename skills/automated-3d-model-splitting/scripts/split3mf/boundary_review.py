"""Conditional boundary review and fingerprint-bound ownership approval."""
from __future__ import annotations

import json
import time
from pathlib import Path
import numpy as np

from .boundary_clarity import BoundaryGraph
from .reporting import runtime_log


class BoundaryReviewRequired(Exception):
    def __init__(self, directory, report):
        self.directory, self.report = Path(directory), report
        super().__init__(f'Boundary review required: {directory}')


class BoundaryDecisionError(ValueError):
    """Invalid or stale explicit approval; never a reason to silently continue."""


class BoundaryReviewService:
    """Clear boundaries allocate no candidates, pictures, or review files.

    Endpoint-only boundaries use the general nearest-ownership advisory policy.
    Branched boundaries get at most three reviewed local searches. Geometry
    and per-face paint stay untouched in both paths.
    """
    def __init__(self, directory: Path, decisions: Path | None = None, *, preserve_source_branches=False):
        self.directory = Path(directory)
        self.decisions = Path(decisions) if decisions else None
        self.records = []
        self.preserve_source_branches = bool(preserve_source_branches)

    def review_curve(self, failure):
        """Geometry proposals have a separate approval scope from face labels."""
        from .curve_preview import export_curve_review
        import hashlib
        fingerprint = hashlib.sha256(failure.source.astype('<f8').tobytes()
            + failure.target.astype('<f8').tobytes()).hexdigest()
        directory = self.directory / ('curve_' + fingerprint[:12])
        report = export_curve_review(failure.source, failure.target, failure.proposal,
            failure.basis, directory, record=dict(fingerprint=fingerprint,
                geometry_change=False, mesh_untouched=True,
                previous_ownership_approval_reused=False))
        self.records.append(report)
        raise BoundaryReviewRequired(directory, report)

    def prepare(self, vertices, faces, owners, *, context='input', display_colors=None):
        started = time.perf_counter()
        graph = BoundaryGraph(vertices, faces)
        labels = np.asarray(owners).astype(str)
        assessment = graph.assess(labels)
        report = dict(context=context, check_seconds=round(time.perf_counter()-started, 4),
                      **assessment.as_record())
        self.records.append(report)
        if assessment.clear:
            runtime_log('分界', 'boundary_clear_direct', '边界清晰，直接进入原拆分流程', context=context)
            return labels, report
        if all(not item['branching_vertices'] for item in assessment.reasons):
            from .nearest_boundary import merge_nearest_ownership
            labels, nearest = merge_nearest_ownership(graph, labels)
            report.update(status='endpoint_advisory', nearest_merge=nearest,
                          requires_semantic_confirmation=False)
            runtime_log('警告' if nearest['warning'] else '分界', 'endpoint_nearest_merge',
                        '已按就近原则处理边界端点；残留按当前接口统计', context=context, **nearest)
            return labels, report
        if self.preserve_source_branches:
            from .print_tolerance import triangle_areas
            affected_area = float(triangle_areas(graph.vertices[graph.faces[assessment.uncertain_faces]]).sum())
            report.update(status='source_boundary_preserved', affected_area_mm2=affected_area,
                          requires_semantic_confirmation=False, geometry_change=False,
                          material_change=False, ownership_change=False)
            runtime_log('分界', 'source_boundary_preserved',
                        '按原边界继续；分叉记录为提示，实体生成另行检查',
                        context=context, affected_area_mm2=affected_area)
            return labels, report
        fingerprint = graph.fingerprint(labels, context)
        candidates = graph.propose(labels, assessment, max_candidates=3)
        report.update(fingerprint=fingerprint, search_attempt_limit=3,
            check_and_search_seconds=round(time.perf_counter()-started, 4),
            acceptance_scope='local seam topology only; not semantic correctness or print/assembly validation',
            requires_semantic_confirmation=True,
            method='bounded_local_ownership_votes_with_fixed_outside_anchors',
            material_change=False, geometry_change=False,
            provenance='ownership proposal only; original/generated face origin not inferred from color',
            candidates=[dict(id=c['id'], admissible=c['admissible'],
                fingerprint=graph.fingerprint(c['owners'], context),
                changed_face_ids=c['changed_faces'].tolist(),
                proposed_owners=c['owners'][c['changed_faces']].tolist(),
                crease_weight=c['crease_weight'], assessment=c['assessment'].as_record()) for c in candidates])
        if self.decisions:
            try:
                payload = json.loads(self.decisions.read_text(encoding='utf-8'))
                decisions = payload.get('decisions', [payload])
                if not isinstance(decisions, list) or not all(isinstance(d, dict) for d in decisions):
                    raise ValueError('Expected a list of boundary decisions')
            except (OSError, ValueError, AttributeError) as exc:
                raise BoundaryDecisionError(f'Cannot read boundary approvals: {exc}') from exc
            matching = [d for d in decisions if d.get('fingerprint') == fingerprint]
            if len(matching) > 1:
                raise BoundaryDecisionError('Duplicate boundary approvals for one stage')
            decision = matching[0] if matching else {}
            # A decision for one recursive stage cannot silently approve another.
            if decision.get('fingerprint') == fingerprint:
                if decision.get('user_confirmed') is not True:
                    raise BoundaryDecisionError('Boundary decision is not user-confirmed')
                selected = next((c for c in candidates if c['id'] == decision.get('selected_candidate')), None)
                if selected is None or not selected['admissible']:
                    raise BoundaryDecisionError('Selected boundary candidate is absent or still ambiguous')
                if decision.get('candidate_fingerprint') != graph.fingerprint(selected['owners'], context):
                    raise BoundaryDecisionError('Boundary candidate fingerprint mismatch')
                report.update(status='user_confirmed', selected_candidate=selected['id'])
                report['preserve_visible_boundary'] = decision.get('preserve_visible_boundary') is True
                runtime_log('分界', 'boundary_review_applied', '已应用用户确认的局部归属；颜色和顶点不变',
                    context=context, changed_faces=len(selected['changed_faces']))
                return selected['owners'], report
        directory = self.directory / (context.replace('/', '_')+'_'+fingerprint[:12])
        directory.mkdir(parents=True, exist_ok=True)
        from .boundary_preview import render_boundary_review
        try:
            previews = render_boundary_review(graph, labels, assessment, candidates, directory, display_colors)
        except ImportError as exc:
            previews = []
            report['preview_error'] = str(exc)
        report['previews'] = [str(p) for p in previews]
        report['total_review_seconds'] = round(time.perf_counter()-started, 4)
        (directory/'review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        template = directory/'user_decision.json'
        if not template.exists():
            template.write_text(json.dumps(dict(fingerprint=fingerprint, user_confirmed=False,
                selected_candidate=None, candidate_fingerprint=None), indent=2), encoding='utf-8')
        runtime_log('分界', 'boundary_review_required', '边界不清晰，已生成局部候选预览；确认前不切割',
                    context=context, review_dir=str(directory), candidates=len(candidates))
        raise BoundaryReviewRequired(directory, report)


def owners_from_components(face_count, components):
    owners = np.full(face_count, '', dtype=object)
    for index, component in enumerate(components, 1):
        owners[np.asarray(component.global_faces, dtype=int)] = str(index)
    return owners.astype(str)


def apply_component_ownership(vertices, faces, components, owners):
    """Rebind only changed ownership; all original material arrays remain external."""
    from .recognition import make_component_from_global_faces, triangle_areas
    import copy
    owners = np.asarray(owners).astype(str)
    original_owners = owners_from_components(len(faces), components)
    if owners.shape != original_owners.shape or not set(owners).issubset(set(original_owners)):
        raise ValueError('Boundary proposal contains invalid ownership labels')
    if np.any((owners == '') != (original_owners == '')):
        raise ValueError('Boundary proposal changes unassigned faces')
    areas = triangle_areas(vertices, faces)
    result = list(components)
    for index, original in enumerate(components, 1):
        ids = np.flatnonzero(owners == str(index))
        if np.array_equal(ids, original.global_faces):
            continue
        if not len(ids) and original.face_count:
            raise ValueError(f'Boundary proposal removes P{index:02d}')
        if not len(ids):
            continue
        updated = make_component_from_global_faces(vertices, faces, areas, ids, original.color_code)
        result[index-1] = copy.copy(original)
        for key in ('global_faces', 'face_count', 'area', 'bbox_min', 'bbox_max', 'center'):
            setattr(result[index-1], key, getattr(updated, key))
    return result
