"""Structural admission and separate resource budgets; no automatic repairs."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
import json
from pathlib import Path


class AdmissionError(ValueError):
    def __init__(self, stage, metrics, limits, status="unsupported", **context):
        self.record = {"status": status, "stage": stage, "metrics": metrics, "limits": limits, **context}
        super().__init__(f"{stage}: admission {status}: " + ", ".join(
            f"{key}={value} (limit {limits.get(key, 'unknown')})" for key, value in metrics.items()))


@dataclass(frozen=True)
class ComplexityLimits:
    # Provisional conservative budgets, not a claim of universal printability.
    paint_token_chars: int = 16_384
    model_xml_bytes: int = 268_435_456

    @classmethod
    def load(cls, path=None):
        if path is None:
            return cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or set(data) - set(asdict(cls())):
            raise ValueError("complexity config must contain only documented limit names")
        if any(type(v) is not int or v <= 0 for v in data.values()):
            raise ValueError("complexity limits must be positive integers")
        return cls(**data)


_LIMITS = ContextVar("complexity_limits", default=ComplexityLimits())


@contextmanager
def limits_scope(limits):
    token = _LIMITS.set(limits)
    try:
        yield
    finally:
        _LIMITS.reset(token)


def check(stage, **metrics):
    policy = _LIMITS.get()
    unknown = set(metrics) - set(ComplexityLimits.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown complexity metrics: {sorted(unknown)}")
    limits = {key: getattr(policy, key) for key in metrics}
    exceeded = {key: int(value) for key, value in metrics.items() if value > limits[key]}
    if exceeded:
        raise AdmissionError(stage, exceeded, {key: limits[key] for key in exceeded}, status="budget_exceeded", reason="resource_budget")
    return {"status": "supported", "stage": stage, "metrics": metrics,
            "limits": {key: limits[key] for key in metrics}, "provisional_limits": True}


def audit_frozen_boundary(points):
    """Apply the established bounded contact policy without changing frozen input."""
    from .spatial_intersections import remove_true_self_intersections_3d, clean_tiny_boundary_contacts_3d
    import numpy as np
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3 or not np.isfinite(points).all():
        raise AdmissionError("boundary", {"invalid_loop": 1}, {"invalid_loop": 0}, status="undetermined")
    cleaned, _, audit = remove_true_self_intersections_3d(points)
    if not np.array_equal(cleaned, points):
        _, _, tolerance = clean_tiny_boundary_contacts_3d(points, np.arange(len(points), dtype=np.int64))
        if tolerance["remaining_intersection_count"] or not tolerance["removed_crossing_loop_count"]:
            raise AdmissionError("boundary", {"unsafe_self_contact": 1}, {"unsafe_self_contact": 0})
        audit["tolerated_point_contacts"] = tolerance
    return audit


def assess_mesh(vertices, faces, colors):
    """Report fragmentation; structural checks require later-stage evidence."""
    from .recognition import connected_components_by_color
    import numpy as np
    groups = connected_components_by_color(faces, list(colors))
    triangles = vertices[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1)*0.5
    effective = sum(float(areas[group].sum()) >= 1.0 for group in groups)
    return {"status": "undetermined", "stage": "connectivity",
            "metrics": {"raw_regions": len(groups), "effective_regions": effective,
                        "noise_regions": len(groups) - effective},
            "pending_checks": ["frozen_boundaries", "interface_relationships", "interface_fit"]}
