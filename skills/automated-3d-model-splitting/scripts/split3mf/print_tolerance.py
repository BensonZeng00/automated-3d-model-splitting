"""Physical, per-patch tolerances for ordinary filament-printing workflows."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class PrintTolerance:
    micro_area_mm2: float = 1.0
    surface_distance_mm: float = 0.05
    recovery_dir: Path | None = None
    preserve_audited_hidden_surfaces: bool = True
    repair_thin_backing: bool = False
    backing_validation_scale: float = .99


_ACTIVE = ContextVar('split3mf_print_tolerance', default=PrintTolerance())


def current():
    return _ACTIVE.get()


@contextmanager
def tolerance_scope(policy):
    token = _ACTIVE.set(policy)
    try:
        yield policy
    finally:
        _ACTIVE.reset(token)


def triangle_areas(triangles):
    triangles = np.asarray(triangles, dtype=np.float64)
    return np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                   triangles[:, 2]-triangles[:, 0]), axis=1) * 0.5


def small_patch_report(vertices, faces, *, maximum_span_mm=None, separate_components=False):
    """Measure the complete affected patch, never grant one waiver per facet."""
    points = np.asarray(vertices, dtype=np.float64)
    indexed = np.asarray(faces, dtype=np.int64).reshape((-1, 3))
    triangles = points[indexed]
    areas = triangle_areas(triangles)
    area = float(areas.sum())
    span = float(np.linalg.norm(np.ptp(triangles.reshape((-1, 3)), axis=0))) if len(indexed) else 0.
    component_spans = []
    if separate_components and maximum_span_mm is not None and span > maximum_span_mm:
        # Shared vertices define connected affected patches. Keep one total
        # area budget, while unrelated patches do not inflate each other's span.
        parents = list(range(len(indexed)))
        def root(index):
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index
        owners = {}
        for index, face in enumerate(indexed):
            for vertex in face:
                if int(vertex) in owners:
                    parents[root(index)] = root(owners[int(vertex)])
                else:
                    owners[int(vertex)] = index
        groups = {}
        for index in range(len(indexed)):
            groups.setdefault(root(index), []).append(index)
        component_spans = [float(np.linalg.norm(np.ptp(triangles[ids].reshape(-1, 3), axis=0)))
                           for ids in groups.values()]
    checked_span = max(component_spans, default=span)
    accepted = (current().micro_area_mm2 > 0 and np.isfinite(triangles).all()
                and area <= current().micro_area_mm2 + 1e-12
                and (maximum_span_mm is None or checked_span <= maximum_span_mm))
    return dict(accepted=bool(accepted), affected_area_mm2=area, affected_span_mm=span,
                area_limit_mm2=current().micro_area_mm2, face_count=len(indexed),
                checked_span_mm=checked_span, connected_patch_spans_mm=component_spans)
