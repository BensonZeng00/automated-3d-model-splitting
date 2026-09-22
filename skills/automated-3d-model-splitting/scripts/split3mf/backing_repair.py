"""Pre-Boolean backing validation and explicitly enabled source-following repair."""
import json
import numpy as np
import trimesh
from .backing_thickness import audit_local_backing
from .curved_backing import build
from .local_connectors import _manifold64
from .print_tolerance import current
from .reporting import runtime_log


RECOVERY_TAPER_SLOPE = float(np.tan(np.deg2rad(60.)))
PRINTABLE_THIN_BACKING_ADVISORY_AREA_MM2 = 2.0


def audit_matching_socket(patch, child, parent, scale_factor):
    """Exercise the real fit transform before accepting a replacement backing."""
    from .uniform_fit import scale_finished_insert
    from .post_fit_difference import subtract_parent
    from .post_fit_audit import audit_cut_surface
    socket_solid = _manifold64(parent)-_manifold64(child)
    if str(socket_solid.status()) != 'Error.NoError' or socket_solid.is_empty():
        raise ValueError('Rebuilt backing failed exact matching socket subtraction')
    raw = socket_solid.to_mesh64()
    socket = trimesh.Trimesh(raw.vert_properties[:, :3], raw.tri_verts, process=False)
    scaled, scaling = scale_finished_insert(child, scale_factor)
    candidate, record = subtract_parent({'mesh':scaled}, socket)
    scaled_patch = patch.copy()
    center = np.asarray(scaling['center_mm'])
    scaled_patch.vertices = center+scale_factor*(scaled_patch.vertices-center)
    audit = audit_cut_surface(scaled_patch, scaled, candidate['mesh'],
                              area_budget_mm2=current().micro_area_mm2)
    if not audit['valid']:
        raise BackingRepairError('Rebuilt backing failed matching-socket surface/thickness', audit)
    return dict(valid=True, subtraction=record, surface_and_thickness=audit,
                scope='isolated_complete_parent_contact; final_assembly_still_required')


class BackingRepairError(ValueError):
    def __init__(self, message, record):
        super().__init__(message)
        self.record = record


def ensure_backing(patch, mesh, parent, source_codes, inward, *, part_id):
    """Keep valid builds; reconstruct only failed hidden backing, never the front."""
    policy = current()
    kwargs = dict(scale_factor=policy.backing_validation_scale,
                  area_budget_mm2=policy.micro_area_mm2)
    before = audit_local_backing(patch, mesh, **kwargs)
    record = dict(before=before, repaired=False)
    runtime_log('背衬验收', 'actual_backing_thickness',
                '正在核验生成后实际局部厚度', part_id=part_id, **before)
    if before['valid']:
        return mesh, record, None
    if (
        float(before.get('thin_interior_area_mm2', float('inf')))
        <= PRINTABLE_THIN_BACKING_ADVISORY_AREA_MM2 + 1e-12
    ):
        record.update(
            accepted_without_repair=True,
            acceptance='bounded_hidden_thin_backing_patch',
            advisory_area_limit_mm2=PRINTABLE_THIN_BACKING_ADVISORY_AREA_MM2,
        )
        runtime_log(
            '背衬验收', 'thin_backing_patch_accepted',
            '隐藏背衬薄区不超过打印尺度面积预算；保留原生成几何并继续',
            part_id=part_id, **record,
        )
        return mesh, record, None
    candidate = None
    try:
        if not policy.repair_thin_backing:
            raise ValueError('Actual backing is too thin; explicit --repair-thin-backing required')
        candidate, geometry = build(patch, parent, inward, direction_mode='local-normal',
                                    taper_slope=RECOVERY_TAPER_SLOPE)
        owners = geometry.pop('back_face_source_indices')
        record['construction'] = geometry
        np.testing.assert_array_equal(candidate.triangles[:len(patch.faces)], patch.triangles)
        record['source_front_triangles_preserved'] = True
        if len(candidate.split(only_watertight=False)) != len(mesh.split(only_watertight=False)):
            raise ValueError('Backing reconstruction changed material connectivity')
        solid, container = _manifold64(candidate), _manifold64(parent)
        outside = solid-container
        record.update(kernel_status=str(solid.status()), outside_status=str(outside.status()),
                      outside_volume_mm3=abs(float(outside.volume())))
        if (str(solid.status()) != 'Error.NoError' or str(container.status()) != 'Error.NoError'
                or str(outside.status()) != 'Error.NoError'
                or record['outside_volume_mm3'] > 1e-8):
            raise ValueError('Source-following backing is not contained by the complete parent')
        after = audit_local_backing(patch, candidate, **kwargs)
        record['after'] = after
        if not after['valid']:
            raise ValueError('Source-following backing still fails local thickness')
        record['matching_socket_fit'] = audit_matching_socket(
            patch, candidate, parent, policy.backing_validation_scale)
        record['repaired'] = True
        colors = list(source_codes)+[source_codes[i] for i in owners]
        candidate.metadata = dict(mesh.metadata)
        # Caller's normal source-preserving finalizer regenerates face metadata.
        for key in list(candidate.metadata):
            if key.startswith('face_'):
                del candidate.metadata[key]
        runtime_log('背衬修复', 'source_following_backing_rebuilt',
                    '已重建隐藏背衬，后续使用新实体重新扣母件', part_id=part_id, **record)
        return candidate, record, colors
    except (ValueError, AssertionError) as exc:
        record['error'] = str(exc)
        if getattr(exc, 'record', None):
            record['failure_detail'] = exc.record
        if policy.recovery_dir is not None:
            folder = policy.recovery_dir/'backing_failure'/part_id
            folder.mkdir(parents=True, exist_ok=True)
            for name, value in [('input', mesh), ('source_patch', patch), ('candidate', candidate)]:
                if value is not None:
                    np.savez_compressed(folder/f'{name}_unvalidated.npz', vertices=value.vertices, faces=value.faces)
            np.savez_compressed(folder/'complete_parent.npz', vertices=parent.vertices, faces=parent.faces)
            (folder/'replay_options.json').write_text(json.dumps(dict(
                source_codes=list(source_codes), inward=np.asarray(inward).tolist(), part_id=part_id,
                scale_factor=policy.backing_validation_scale, area_budget_mm2=policy.micro_area_mm2),
                ensure_ascii=False, indent=2), encoding='utf-8')
            (folder/'report.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        raise BackingRepairError(f'{part_id}: {exc}', record) from exc
