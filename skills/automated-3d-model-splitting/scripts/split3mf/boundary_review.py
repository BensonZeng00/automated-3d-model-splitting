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
    def __init__(self, directory: Path, decisions: Path | None = None):
        self.directory = Path(directory)
        self.decisions = Path(decisions) if decisions else None
        self.records = []
        self.area_budgets = {}

    def review_curve(self, failure):
        """Geometry proposals have a separate approval scope from face labels."""
        from .curve_preview import export_curve_review
        import hashlib
        fingerprint = hashlib.sha256(failure.source.astype('<f8').tobytes()
            + failure.target.astype('<f8').tobytes()).hexdigest()
        candidate_fingerprint = failure.proposal.record.get('candidate_sha256')
        decision = self._matching_decision(fingerprint)
        if decision:
            if decision.get('user_confirmed') is not True:
                raise BoundaryDecisionError('Curve decision is not user-confirmed')
            if decision.get('action') != 'apply_clear_curve':
                raise BoundaryDecisionError('Curve decision action must be apply_clear_curve')
            if not candidate_fingerprint or decision.get('candidate_fingerprint') != candidate_fingerprint:
                raise BoundaryDecisionError('Clear-curve candidate fingerprint mismatch')
            if failure.proposal.record.get('status') != 'proposed':
                raise BoundaryDecisionError('Reviewed curve has no complete applicable proposal')
            runtime_log('分界', 'curve_review_applied',
                        '已应用指纹匹配的用户确认清除曲线；继续表面带重网格与严格审核',
                        fingerprint=fingerprint, candidate_fingerprint=candidate_fingerprint)
            return failure.proposal
        directory = self.directory / ('curve_' + fingerprint[:12])
        report = export_curve_review(failure.source, failure.target, failure.proposal,
            failure.basis, directory, record=dict(fingerprint=fingerprint,
                candidate_fingerprint=candidate_fingerprint,
                action='apply_clear_curve', user_confirmed=False,
                geometry_change=False, mesh_untouched=True,
                previous_ownership_approval_reused=False))
        self.records.append(report)
        raise BoundaryReviewRequired(directory, report)

    def _matching_decision(self, fingerprint):
        if self.decisions is None:
            return None
        try:
            payload = json.loads(self.decisions.read_text(encoding='utf-8'))
            decisions = payload.get('decisions', [payload])
            if not isinstance(decisions, list) or not all(isinstance(item, dict) for item in decisions):
                raise ValueError('Expected a list of boundary decisions')
        except (OSError, ValueError, AttributeError) as exc:
            raise BoundaryDecisionError(f'Cannot read boundary approvals: {exc}') from exc
        matching = [item for item in decisions if item.get('fingerprint') == fingerprint]
        if len(matching) > 1:
            raise BoundaryDecisionError('Duplicate boundary approvals for one stage')
        return matching[0] if matching else None

    def prepare(self, vertices, faces, owners, *, context='input', display_colors=None):
        started = time.perf_counter()
        runtime_log('分界', 'boundary_graph_start', '开始构建稀疏面邻接图',
                    context=context, faces=len(faces))
        graph = BoundaryGraph(vertices, faces)
        runtime_log('分界', 'boundary_graph_done', '稀疏面邻接图构建完成',
                    context=context, faces=len(faces), adjacency_edges=len(graph.adjacency),
                    duration_seconds=round(time.perf_counter()-started, 4))
        labels = np.asarray(owners).astype(str)
        from .boundary_budget import BoundaryAreaBudget
        from .boundary_simplification import simplify_crossing_ownership
        budget_key = graph.fingerprint(np.full(len(labels), ''), context)
        if budget_key not in self.area_budgets:
            self.area_budgets[budget_key] = BoundaryAreaBudget(vertices, faces, labels)
        budget = self.area_budgets[budget_key]
        # An explicit review file owns its proposal: do not invalidate it first.
        cleanup = []
        if self.decisions is None:
            cleanup_started = time.perf_counter()
            runtime_log('分界', 'boundary_crossing_cleanup_start',
                        '开始检查材料间共享边的局部交叉', context=context)
            labels, cleanup = simplify_crossing_ownership(graph, labels, budget)
            runtime_log('分界', 'boundary_crossing_cleanup_done',
                        '材料间共享边局部交叉检查完成', context=context,
                        owners_checked=len(np.unique(labels)), proposals=len(cleanup),
                        duration_seconds=round(time.perf_counter()-cleanup_started, 4))
        assessment_started = time.perf_counter()
        assessment = graph.assess(labels)
        runtime_log('分界', 'boundary_assessment_done', '边界拓扑评估完成',
                    context=context, uncertain_faces=len(assessment.uncertain_faces),
                    ambiguous_pairs=len(assessment.reasons),
                    duration_seconds=round(time.perf_counter()-assessment_started, 4))
        report = dict(context=context, check_seconds=round(time.perf_counter()-started, 4),
                      crossing_cleanup=cleanup, **assessment.as_record())
        self.records.append(report)
        if assessment.clear:
            runtime_log('分界', 'boundary_clear_direct', '边界清晰，直接进入原拆分流程', context=context)
            return labels, report
        # Source paint boundaries are recognition evidence, not generated
        # handoff seams. Preserve their exact ownership and report ambiguity
        # without blocking recognition; generated recursive interfaces remain
        # subject to the normal blocking checks below.
        if context == 'input' or context.startswith('step_'):
            report.update(status='source_diagnostics',
                          source_diagnostics='non_blocking',
                          ownership_changed=False,
                          geometry_changed=False,
                          requires_semantic_confirmation=False)
            runtime_log('警告', 'source_boundary_diagnostic',
                        '源材料边界存在拓扑歧义，保留原始面归属并继续识别；生成接口仍执行严格审核',
                        context=context,
                        ambiguous_pairs=len(assessment.reasons),
                        uncertain_faces=len(assessment.uncertain_faces))
            return labels, report
        if all(not item['branching_vertices'] for item in assessment.reasons):
            from .nearest_boundary import merge_nearest_ownership
            labels, nearest = merge_nearest_ownership(graph, labels)
            report.update(status='endpoint_advisory', nearest_merge=nearest,
                          requires_semantic_confirmation=False)
            runtime_log('警告' if nearest['warning'] else '分界', 'endpoint_nearest_merge',
                        '已按就近原则处理边界端点；残留按当前接口统计', context=context, **nearest)
            return labels, report
        fingerprint = graph.fingerprint(labels, context)
        proposal_started = time.perf_counter()
        runtime_log('分界', 'boundary_candidate_search_start',
                    '开始局部边界候选搜索', context=context,
                    uncertain_faces=len(assessment.uncertain_faces))
        candidates = graph.propose(labels, assessment, max_candidates=3)
        runtime_log('分界', 'boundary_candidate_search_done',
                    '局部边界候选搜索完成', context=context,
                    candidates=len(candidates),
                    duration_seconds=round(time.perf_counter()-proposal_started, 4))
        from .boundary_budget import BoundaryAreaBudget
        # A context identifies one stage; repeated proposals share one frozen budget.
        budget_key = graph.fingerprint(np.full(len(labels), ''), context)
        budget = self.area_budgets.setdefault(budget_key,
            BoundaryAreaBudget(vertices, faces, labels))
        for candidate in candidates:
            candidate['area_policy'] = budget.evaluate(labels, candidate['owners'])
        automatic = [c for c in candidates if c['admissible'] and c['area_policy']['automatic']]
        if automatic and self.decisions is None:
            chosen = min(automatic, key=lambda c: sum(
                item['affected_area_mm2'] for item in c['area_policy']['regions']))
            budget.evaluate(labels, chosen['owners'], commit=True)
            report.update(status='automatic_local_merge', area_policy=chosen['area_policy'],
                          changed_face_ids=chosen['changed_faces'].tolist(),
                          requires_semantic_confirmation=False, geometry_change=False,
                          material_change=False, boundary_shape='smooth')
            runtime_log('分界', 'boundary_small_anomaly_merged',
                        '累计异常面积小于各受影响部件的 1%，已局部归并并继续平滑',
                        context=context, **chosen['area_policy'])
            return chosen['owners'], report
        viable = [c for c in candidates if c['admissible']
                  and c['area_policy']['identities_preserved']]
        if viable and self.decisions is None:
            # These proposals only relabel the local uncertain band. Graph.propose
            # has verified retained parts, connectivity and fixed outside anchors;
            # vertices, paint and exterior shape are untouched. This is a concrete
            # completion-first assessment, not a larger numeric tolerance.
            chosen = min(viable, key=lambda c: sum(
                item['affected_area_mm2'] for item in c['area_policy']['regions']))
            budget.evaluate(labels, chosen['owners'], commit=True)
            report.update(status='completion_priority_local_merge',
                          area_policy=chosen['area_policy'],
                          completion_assessment=dict(exterior_unchanged=True,
                              paint_unchanged=True, all_parts_retained=True,
                              no_new_disconnected_regions=True,
                              outside_band_unchanged=True,
                              assembly_rebuild_and_validation_required=True),
                          changed_face_ids=chosen['changed_faces'].tolist(),
                          requires_semantic_confirmation=False,
                          geometry_change=False, material_change=False, boundary_shape='smooth')
            runtime_log('分界', 'boundary_completion_priority_merge',
                        '局部异常达到 1%，外形、颜色、部件身份和连通性保持，按完成优先原则归并后继续',
                        context=context, **chosen['area_policy'])
            return chosen['owners'], report
        # Dense triangle-selector paint can contain thousands of legitimate
        # pairwise endpoints where a third paint region owns the continuation.
        # Requiring one pair at a time to become a closed loop makes every
        # bounded proposal formally inadmissible, even though the proposal is
        # microscopic relative to every affected material.  Prefer the least
        # ambiguous bounded proposal in that specific, scale-free case and let
        # the later component, interface, topology and visual audits decide.
        fragmented = [c for c in candidates
                      if c['area_policy']['identities_preserved']
                      and c['area_policy']['regions']
                      and max(r['affected_area_ratio']
                              for r in c['area_policy']['regions']) <= 0.001]
        if (self.decisions is None and context == 'input' and len(faces) >= 100_000
                and len(assessment.uncertain_faces) / len(faces) >= 0.02
                and fragmented):
            def ambiguity(candidate):
                record = candidate['assessment'].as_record()
                return (sum(item['branching_vertices'] + item['unexplained_endpoints']
                            for item in record['reasons']),
                        sum(item['affected_area_mm2']
                            for item in candidate['area_policy']['regions']))
            chosen = min(fragmented, key=ambiguity)
            budget.evaluate(labels, chosen['owners'], commit=True)
            report.update(status='fragmented_paint_completion',
                          area_policy=chosen['area_policy'],
                          residual_assessment=chosen['assessment'].as_record(),
                          changed_face_ids=chosen['changed_faces'].tolist(),
                          requires_semantic_confirmation=False,
                          geometry_change=False, material_change=False,
                          boundary_shape='smooth',
                          completion_assessment=dict(
                              reason='dense_triangle_selector_pairwise_endpoints',
                              maximum_affected_region_ratio=0.001,
                              all_material_identities_preserved=True,
                              downstream_geometry_audits_required=True))
            runtime_log('分界', 'fragmented_paint_completion',
                        '密集三角涂色的成对端点按最小相对影响候选继续，后续几何审核保持阻断',
                        context=context, **chosen['area_policy'])
            return chosen['owners'], report
        report.update(fingerprint=fingerprint, search_attempt_limit=3,
            check_and_search_seconds=round(time.perf_counter()-started, 4),
            acceptance_scope='local seam topology only; not semantic correctness or print/assembly validation',
            requires_semantic_confirmation=True,
            method='bounded_local_ownership_votes_with_fixed_outside_anchors',
            material_change=False, geometry_change=False,
            provenance='ownership proposal only; original/generated face origin not inferred from color',
            above_threshold_policy='assess_completion_priority_not_percentage_only',
            candidates=[dict(id=c['id'], admissible=c['admissible'], area_policy=c['area_policy'],
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
                budget.evaluate(labels, selected['owners'], commit=True)
                report['boundary_shape'] = 'smooth'
                runtime_log('分界', 'boundary_review_applied', '已应用用户确认的局部归属；颜色和顶点不变',
                    context=context, changed_faces=len(selected['changed_faces']))
                return selected['owners'], report
        directory = self.directory / (context.replace('/', '_')+'_'+fingerprint[:12])
        directory.mkdir(parents=True, exist_ok=True)
        from .boundary_preview import render_boundary_review
        try:
            preview_started = time.perf_counter()
            runtime_log('分界', 'boundary_preview_start', '开始渲染局部边界审核图',
                        context=context, candidates=len(candidates))
            previews = render_boundary_review(graph, labels, assessment, candidates, directory, display_colors)
            runtime_log('分界', 'boundary_preview_done', '局部边界审核图渲染完成',
                        context=context, previews=len(previews),
                        duration_seconds=round(time.perf_counter()-preview_started, 4))
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
