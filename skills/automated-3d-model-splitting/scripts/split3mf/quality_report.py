"""Run-scoped, non-blocking geometry findings and durable Markdown reports.

Library calls remain strict unless a caller explicitly enters a report scope.
Input identity, review consent, resource budgets and I/O errors stay blocking.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import json
from pathlib import Path

_ACTIVE = ContextVar('quality_report', default=None)
_STRICT_CONSTRUCTION = ContextVar('strict_construction_checks', default=False)
_IDENTITY = ContextVar('quality_identity', default=None)


def current_report():
    return _ACTIVE.get()


def best_effort():
    report = current_report()
    return report is not None and not report.strict and not _STRICT_CONSTRUCTION.get()


@contextmanager
def strict_construction_checks():
    """Try existing finite recovery before accepting an out-of-tolerance candidate."""
    token = _STRICT_CONSTRUCTION.set(True)
    try:
        yield
    finally:
        _STRICT_CONSTRUCTION.reset(token)


@contextmanager
def quality_identity(identity):
    token = _IDENTITY.set(identity)
    try:
        yield
    finally:
        _IDENTITY.reset(token)


@dataclass
class QualityReport:
    path: Path
    strict: bool = False
    findings: list = field(default_factory=list)
    parts: list = field(default_factory=list)
    interfaces: dict = field(default_factory=dict)
    output: str | None = None
    status: str = 'in_progress'

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = dict(status=self.status, quality_policy='strict' if self.strict else 'best-effort',
                    output=self.output, findings=self.findings, parts=self.parts, interfaces=self.interfaces,
                    printability='user_review_required',
                    visual_consistency='not_checked')
        self.path.with_suffix('.json').write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
        lines = ['# 拆件检查报告', '', f'运行状态：{self.status}',
                 f'输出文件：{self.output or "尚未输出"}', '',
                 '可打印性由用户检查确认。告警不代表检查通过；参考网格不代表已完成接口。', '',
                 '多视角视觉一致性：未检查。', '',
                 f'接口统计：{json.dumps(self.interfaces, ensure_ascii=False)}', '', '## 零件', '',
                 '| 零件 | 结果 | 面数 | 水密 | 绕向一致 | 保留网格 |', '| --- | --- | --- | --- | --- | --- |']
        def cell(value):
            return str(value).replace('|', '\\|').replace('\n', '<br>')
        for part in self.parts:
            audit = part.get('audit', {})
            mesh_path = audit.get('preserved_mesh')
            mesh_link = f'[PLY](<{Path(mesh_path).resolve().as_posix()}>)' if mesh_path else '无'
            lines.append(f'| {cell(part["part_id"])} | {cell(part["status"])} | '
                         f'{audit.get("face_count", "未检查")} | {audit.get("watertight", "未检查")} | '
                         f'{audit.get("winding_consistent", "未检查")} | {mesh_link} |')
        lines += ['', '## 问题与处理', '']
        if not self.findings:
            lines += ['暂无已记录问题；未执行的检查不计为通过。', '']
        for i, finding in enumerate(self.findings, 1):
            lines += [f'### {i}. {finding["stage"]} / {finding["identity"]}', '',
                      finding['message'], '', f'处理：{finding["action"]}', '',
                      '```json', json.dumps(finding['details'], ensure_ascii=False, indent=2, default=str),
                      '```', '']
        self.path.write_text('\n'.join(lines), encoding='utf-8')


def warn(stage, identity, message, *, action='保留问题网格并继续；由用户检查可打印性', **details):
    report = current_report()
    if report is None:
        return
    identity = _IDENTITY.get() if identity == 'source_cap' and _IDENTITY.get() else identity
    report.findings.append(dict(stage=stage, identity=identity, message=str(message),
                                action=action, details=details))
    report.save()
    from .execution_log import emit
    emit(stage, 'quality_warning', str(message), level='WARNING', identity=identity,
         action=action, diagnostic=details)


def quality_check(condition, message, *, stage, identity, **details):
    if condition:
        return
    if not best_effort():
        raise ValueError(message)
    warn(stage, identity, message, **details)


def record_parts(records):
    report = current_report()
    if report:
        report.parts = [dict(part_id=r['part_id'], status=r.get('build_status', 'constructed'),
                                 audit={k: v for k, v in r.items() if k != 'composition'}) for r in records]
        report.save()


def record_interfaces(plan, interface_summary):
    report = current_report()
    if report:
        skipped = interface_summary.get('skipped_interfaces', [])
        report.interfaces = dict(planned=len(plan['interfaces']),
                                 generated=len({r['interface_id'] for r in interface_summary['interfaces']}),
                                 skipped=skipped)
        report.save()


def preserve_part(part, record):
    report = current_report()
    if report:
        directory = report.path.with_name(report.path.stem + '_partial_parts')
        directory.mkdir(parents=True, exist_ok=True)
        # PLY preserves open/non-manifold reference surfaces without welding or repairing.
        part['mesh'].export(directory / (part['part_id'] + '.ply'))
        record['preserved_mesh'] = str(directory / (part['part_id'] + '.ply'))


def published(path):
    report = current_report()
    if report:
        report.output = str(Path(path).resolve())
        report.save()


@contextmanager
def report_scope(path, *, strict=False):
    report = QualityReport(Path(path), strict=strict)
    token = _ACTIVE.set(report)
    try:
        report.save()
        from .execution_log import summary
        summary({'split_report': str(report.path.resolve()), 'quality_policy': 'strict' if strict else 'best-effort'})
        yield report
    except BaseException as exc:
        report.status = 'review_required' if isinstance(exc, SystemExit) and exc.code == 4 else 'stopped'
        report.findings.append(dict(stage='execution', identity='run', message=str(exc),
                                    action='保留已生成的阶段文件和报告', details={}))
        raise
    else:
        report.status = ('completed_with_warnings' if report.findings else 'completed') if report.output else 'stage_only'
    finally:
        try:
            report.save()
        finally:
            _ACTIVE.reset(token)
