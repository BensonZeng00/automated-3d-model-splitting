"""Geometry findings must preserve outputs without falsifying validation results."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from split3mf.common import Component, load_core_dependencies
load_core_dependencies()
from split3mf.quality_report import report_scope, quality_check, best_effort, published
from split3mf import interface_assembly as assembly
from split3mf.stage_cache import fingerprint_payload
from split3mf.package_io import ThreeMFWriter, load_colored_mesh_objects_3mf
from split3mf.interface_package_validation import validate_reloaded_interface_parts


def test_warning_keeps_evidence_and_status_and_context_resets(tmp_path):
    path = tmp_path / 'split_report.md'
    with report_scope(path) as report:
        quality_check(False, '开放边', stage='05', identity='P01', measured=3, threshold=0)
        published(tmp_path / 'result.3mf')
        assert len(report.findings) == 1
    data = json.loads(path.with_suffix('.json').read_text())
    assert data['status'] == 'completed_with_warnings'
    assert data['printability'] == 'user_review_required'
    assert data['visual_consistency'] == 'not_checked'
    assert '开放边' in path.read_text()
    assert not best_effort()
    with pytest.raises(ValueError):
        quality_check(False, 'strict', stage='05', identity='P01')


def test_strict_mode_and_fatal_error_preserve_report(tmp_path):
    path = tmp_path / 'report.md'
    with pytest.raises(ValueError):
        with report_scope(path, strict=True):
            quality_check(False, 'invalid geometry', stage='05', identity='P01')
    assert json.loads(path.with_suffix('.json').read_text())['status'] == 'stopped'
    with pytest.raises(OSError):
        with report_scope(path):
            raise OSError('disk unavailable')
    assert 'disk unavailable' in path.read_text()
    assert not best_effort()


def plan():
    relations = [dict(interface_id=f'I00{i}', parts=['P01', 'P02'], tenon_part='P01',
                      mortise_part='P02', contact={'shared_boundary_loops': [{}]}) for i in (1, 2)]
    return dict(schema='contact-interface-plan/v1', interfaces=relations,
                relation_table={'rows': [[r['interface_id'], 'P01', 'P02'] for r in relations]})


def test_one_interface_failure_does_not_discard_other_interfaces(tmp_path, monkeypatch):
    original = plan()
    calls = []
    def builder(single, **kwargs):
        # Verify the retry still uses the real role-table integrity checker.
        roles = assembly.interface_role_index(single)
        calls.append(set(roles))
        if 'I001' in roles:
            raise ValueError('I001: minimum connector depth exceeds host thickness')
        return {'retained': np.array([7])}, dict(interfaces=[dict(interface_id='I002', loop_position=0)],
                                               interface_role_index_sha256=fingerprint_payload(roles))
    monkeypatch.setattr(assembly, '_build_pairwise_interface_surfaces_strict', builder)
    with report_scope(tmp_path / 'report.md') as report:
        arrays, summary = assembly.build_pairwise_interface_surfaces(
            original, scale_ratio=.5, clearance_mm=.2)
        assert arrays['retained'][0] == 7
        assert summary['skipped_interfaces'] == ['I001']
        assert summary['interface_role_index_sha256'] == fingerprint_payload(assembly.interface_role_index(original))
        assert report.interfaces == dict(planned=2, generated=1, skipped=['I001'])
        assert report.findings[0]['identity'] == 'I001'
    assert len(original['interfaces']) == 2
    assert {'I002'} in calls


def test_failed_part_is_preserved_and_other_parts_continue(tmp_path, monkeypatch):
    first = trimesh.creation.box()
    second = trimesh.creation.box()
    second.apply_translation([3, 0, 0])
    mesh = trimesh.util.concatenate([first, second])
    components = [Component('A', ids, len(ids), 6., np.zeros(3), np.ones(3), np.zeros(3))
                  for ids in (np.arange(12), np.arange(12, 24))]
    original = assembly.compose_part_with_interfaces
    def compose(vertices, faces, attachments):
        if vertices[:, 0].mean() < 1:
            raise ValueError('source boundary did not close')
        return original(vertices, faces, attachments)
    monkeypatch.setattr(assembly, 'compose_part_with_interfaces', compose)
    empty = dict(schema='contact-interface-plan/v1', interfaces=[], relation_table={'rows': []})
    summary = dict(interfaces=[], interface_role_index_sha256=fingerprint_payload({}))
    with report_scope(tmp_path / 'report.md') as report:
        parts, records = assembly.build_pairwise_part_meshes(empty, {}, summary,
            vertices=mesh.vertices, faces=mesh.faces, components=components)
        assert len(parts) == 2
        assert records[0]['build_status'] == 'reference_source_mesh'
        assert records[1]['build_status'] == 'constructed'
        assert Path(records[0]['preserved_mesh']).exists()
        output = tmp_path / 'result.3mf'
        ThreeMFWriter().write(output, parts, 'test')
        assert len(load_colored_mesh_objects_3mf(output)) == 2
        published(output)
        assert report.parts[0]['status'] == 'reference_source_mesh'
    assert 'reference_source_mesh' in (tmp_path / 'report.md').read_text()


def test_readback_geometry_warning_retains_actual_audit(tmp_path):
    mesh = trimesh.creation.box()
    part = dict(part_id='P01', mesh=mesh, color_hex='#FFFFFF')
    path = tmp_path / 'result.3mf'
    ThreeMFWriter().write(path, [part], 'test')
    moved = mesh.copy()
    moved.apply_translation([0.01, 0, 0])
    expected = [{**part, 'mesh': moved}]
    records = [dict(part_id='P01', source_degenerate_face_count=0)]
    with pytest.raises(ValueError, match='readback'):
        validate_reloaded_interface_parts(path, expected, records)
    with report_scope(tmp_path / 'report.md') as report:
        audits = validate_reloaded_interface_parts(path, expected, records)
        assert audits[0]['maximum_export_vertex_shift_mm'] > 1e-4
        assert report.findings[0]['details']['thresholds']['maximum_vertex_shift_mm'] == 1e-4
    assert path.exists()


def test_palette_error_falls_back_to_readable_basic_package(tmp_path):
    part = dict(part_id='P01', mesh=trimesh.creation.box(), color_hex='#FFFFFF',
                color_resolution_status='source_metadata', filament_slot_index=99)
    with report_scope(tmp_path / 'report.md') as report:
        path = tmp_path / 'result.3mf'
        ThreeMFWriter().write(path, [part], 'test', source_filament_colors=['#000000'])
        loaded = load_colored_mesh_objects_3mf(path)
        assert len(loaded) == 1
        np.testing.assert_array_equal(loaded[0]['mesh'].triangles, part['mesh'].triangles)
        assert report.findings[0]['stage'] == '08'
    assert part['filament_slot_index'] == 99


def test_empty_mesh_is_still_fatal(tmp_path):
    with report_scope(tmp_path / 'report.md'):
        with pytest.raises(ValueError, match='no exportable'):
            ThreeMFWriter().write(tmp_path / 'empty.3mf', [], 'test')


@pytest.mark.parametrize('strict, expected_exit', [(False, 0), (True, 3)])
def test_full_cli_selects_policy_and_reports_warning(tmp_path, monkeypatch, strict, expected_exit):
    from split3mf import cli
    target = tmp_path / 'model.3mf'
    def execute(args, parser):
        quality_check(False, 'open edge count exceeds zero', stage='05', identity='P01', measured=3, threshold=0)
        mesh = trimesh.Trimesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], process=False)
        ThreeMFWriter().write(target, [dict(part_id='P01', mesh=mesh, color_hex='#FFFFFF')], 'open surface')
        assert len(load_colored_mesh_objects_3mf(target)) == 1
        published(target)
    monkeypatch.setattr(cli, '_run', execute)
    argv = ['--input', str(tmp_path / 'source.3mf'), '--output', str(target),
            '--preflight-only', '--log-file', str(tmp_path / 'run.jsonl')]
    if strict:
        argv.append('--strict-quality')
    with pytest.raises(SystemExit) as caught:
        cli.main(argv)
    assert caught.value.code == expected_exit
    data = json.loads((tmp_path / 'model_split_report.json').read_text())
    assert data['status'] == ('stopped' if strict else 'completed_with_warnings')
    assert target.exists() is not strict


def test_corrupt_package_readback_still_stops(tmp_path):
    path = tmp_path / 'broken.3mf'
    path.write_bytes(b'not a ZIP package')
    with report_scope(tmp_path / 'report.md'):
        with pytest.raises(Exception):
            validate_reloaded_interface_parts(path, [], [])


def test_normal_construction_uses_existing_strict_recovery_first(tmp_path, monkeypatch):
    mesh = trimesh.creation.box()
    component = Component('A', np.arange(12), 12, 6., np.zeros(3), np.ones(3), np.zeros(3))
    original = assembly.compose_part_with_interfaces
    observed = []
    def compose(*args):
        observed.append(best_effort())
        return original(*args)
    monkeypatch.setattr(assembly, 'compose_part_with_interfaces', compose)
    empty = dict(schema='contact-interface-plan/v1', interfaces=[], relation_table={'rows': []})
    summary = dict(interfaces=[], interface_role_index_sha256=fingerprint_payload({}))
    with report_scope(tmp_path / 'report.md') as report:
        parts, _ = assembly.build_pairwise_part_meshes(empty, {}, summary,
            vertices=mesh.vertices, faces=mesh.faces, components=[component])
        np.testing.assert_array_equal(parts[0]['mesh'].triangles, mesh.triangles)
        assert report.findings == []
        assert best_effort()
    assert observed == [False]


def test_shared_full_and_resume_topology_audit_warns_on_open_mesh(tmp_path):
    from split3mf.validation import validate_part_meshes
    mesh = trimesh.Trimesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], process=False)
    parts = [dict(part_id='P01', mesh=mesh)]
    with report_scope(tmp_path / 'report.md') as report:
        audits = validate_part_meshes(parts)
        assert audits[0]['open_edges'] == 3
        assert report.findings[0]['details']['thresholds']['open_edges'] == 0
    with report_scope(tmp_path / 'strict.md', strict=True):
        with pytest.raises(ValueError, match='topology defects'):
            validate_part_meshes(parts)
