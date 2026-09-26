from __future__ import annotations

from dataclasses import dataclass
from typing import Any

@dataclass
class CapDecision:
    mode: str
    source_vertex_ids: tuple[int, ...]
    fit_points: Any
    directions: Any
    distances: Any
    record: dict[str, Any]
    source_points: Any | None = None
    cap_template_points: Any | None = None
    cap_template_faces: Any | None = None
    cap_template_boundary_ids: tuple[int, ...] | None = None
