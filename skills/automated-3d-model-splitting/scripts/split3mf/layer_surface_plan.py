"""A single validated surface snapshot consumed by every builder in one layer."""
import hashlib
import numpy as np
from .planar_arc import PlanarArcError
from .shared_seam_topology import canonical_loop, classify_shared_seams, validate_region_boundary_preservation
from .print_tolerance import current


def face_keys(faces):
    values = np.ascontiguousarray(np.sort(np.asarray(faces, dtype=np.int64), axis=1))
    return values.view(np.dtype((np.void, 24))).ravel()


class LayerSurfacePlan:
    def __init__(self, source_vertices, source_faces, vertices, faces, records, quality, report, config):
        self.vertices, self.faces = vertices, faces
        self.records, self.quality, self.report = records, quality, report
        self.config = config
        self.keys = np.sort(face_keys(faces))
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(source_vertices).tobytes())
        digest.update(np.ascontiguousarray(source_faces).tobytes())
        self.report['source_geometry_sha256'] = digest.hexdigest()
        self.vertices.setflags(write=False); self.faces.setflags(write=False)

    def apply(self, vertices, faces, global_ids, loops):
        ids = np.asarray(global_ids, dtype=np.int64)
        if np.any(ids < 0) or np.any(ids >= len(self.vertices)):
            raise PlanarArcError('Layer seam source IDs do not match current input')
        if not np.allclose(vertices, self.vertices[ids], atol=1e-9, rtol=0):
            raise PlanarArcError('Layer seam requires the authoritative current-layer surface')
        if faces is not None:
            wanted = face_keys(ids[np.asarray(faces, dtype=np.int64)])
            positions = np.searchsorted(self.keys, wanted)
            if np.any(positions >= len(self.keys)) or np.any(self.keys[positions] != wanted):
                raise PlanarArcError('Layer seam surface triangles changed after planning')
        records = []
        for index, loop in enumerate(loops):
            key = canonical_loop(ids[np.asarray(loop, dtype=np.int64)])
            if key not in self.records:
                raise PlanarArcError('Unplanned boundary loop in shared layer surface')
            records.append(dict(self.records[key], loop_index=index,
                                shared_layer_surface=self.report,
                                visible_surface_band_quality=self.quality))
        return np.asarray(vertices).copy(), records


def build_layer_surface_plan(vertices, faces, loops, context, face_groups=None):
    from .interface_retopology import InterfaceRetopologyService, _surface_band_deformation
    from .surface_quality import directed_edge_topology_issues
    source = np.asarray(vertices, dtype=np.float64)
    source_faces = np.asarray(faces, dtype=np.int64)
    canonical, neighbors, report = classify_shared_seams(source, source_faces, loops)
    if not report['shared_vertex_count'] and not report['duplicate_loop_records']:
        return None
    if not context.inherited_surface_provenance_known:
        raise PlanarArcError('Shared smoothing requires inherited contact-surface provenance')
    proposals, records = {}, {}
    for loop in canonical:
        ids = np.asarray(loop)
        target, record = InterfaceRetopologyService.retopologize_loop(source[ids], ids, context)
        records[loop] = record
        for vertex, point in zip(loop, target):
            proposals.setdefault(vertex, []).append(point)
    frozen = set(map(int, context.inherited_frozen_vertex_ids)) if context.inherited_frozen_vertex_ids is not None else set()
    for vertex in frozen:
        proposals.setdefault(vertex, [source[vertex]])
    boundary = np.asarray(sorted(proposals), dtype=np.int64)
    targets = []
    disagreement = 0.
    for vertex in boundary:
        values = np.asarray(proposals[int(vertex)])
        if len(values) > 1:
            disagreement = max(disagreement, float(np.linalg.norm(values[:,None]-values[None,:], axis=2).max()))
        # Equal-confidence least squares under one shared positional constraint.
        # Junctions remain fixed to avoid changing the region adjacency layout.
        targets.append(source[vertex] if vertex in frozen or len(neighbors[int(vertex)]) != 2 else values.mean(axis=0))
    targets = np.asarray(targets)
    changed_faces = source_faces.copy()
    result, quality = _surface_band_deformation(
        source, changed_faces, boundary, targets, context.config.retopology_band_mm,
        failure_sink=context.failure_sink,
        minimum_sparse_inversion_angle_degrees=1. if context.config.surface_band_validation == 'advisory' else 3.,
        validation_mode=context.config.surface_band_validation)
    if not quality['valid']:
        raise PlanarArcError(f'Shared layer surface failed quality audit: {quality}')
    if not np.allclose(result[boundary], targets, atol=1e-9, rtol=0):
        raise PlanarArcError('Shared layer solver moved its fixed seam targets')
    if face_groups is not None:
        groups = np.asarray(face_groups)
        changed = np.flatnonzero(np.any(changed_faces != source_faces, axis=1))
        for group in np.unique(groups[changed]):
            allowed = np.unique(source_faces[groups == group])
            selected = changed[groups[changed] == group]
            if not np.isin(changed_faces[selected], allowed).all():
                raise PlanarArcError('Shared surface retriangulation crosses source region ownership')
        if len(changed):
            validate_region_boundary_preservation(source_faces, changed_faces, canonical, groups)
    if frozen:
        frozen_ids = np.asarray(sorted(frozen), dtype=np.int64)
        inherited_faces = np.all(np.isin(source_faces, frozen_ids), axis=1)
        if (not np.array_equal(result[frozen_ids], source[frozen_ids])
                or not np.array_equal(changed_faces[inherited_faces], source_faces[inherited_faces])):
            raise PlanarArcError('Shared surface changed inherited parent-contact geometry')
    # Retriangulation must not erase a paint boundary or change its incidence.
    classify_shared_seams(result, changed_faces, canonical)
    for loop, record in records.items():
        offsets = np.linalg.norm(result[np.asarray(loop)] - source[np.asarray(loop)], axis=1)
        record.update(visible_boundary_retopologized=True,
                      visible_top_source_preserved=False,
                      joint_actual_maximum_offset_mm=float(offsets.max()),
                      joint_actual_rms_offset_mm=float(np.sqrt(np.mean(offsets ** 2))),
                      interface_retopology_surface_role='shared_current_layer_surface')
    report.update(maximum_proposal_disagreement_mm=disagreement,
                  residual_shared_target_disagreement_mm=0.,
                  shared_target_strategy='joint_least_squares_with_source_locked_junctions',
                  affected_area_within_micro_budget=report['affected_source_area_mm2'] <= current().micro_area_mm2,
                  inherited_frozen_vertex_count=len(frozen),
                  tolerance_waiver_applied=False)
    report.pop('shared_edges', None)
    report.pop('shared_source_ids', None)
    return LayerSurfacePlan(source, source_faces, result, changed_faces, records, quality, report, context.config)
