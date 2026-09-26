from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
from .planar_arc_retopology_config import PlanarArcRetopologyConfig

@dataclass(frozen=True)
class PlanarArcRetopologyContext:
    config: PlanarArcRetopologyConfig
    failure_sink: Callable[[dict[str, Any]], None] | None = None
    curve_review_sink: Callable[[Any], None] | None = None
    layer_seams: dict[int, Any] = field(default_factory=dict, compare=False, repr=False)
    # Topology-only recursive boundary data.  Keys include the complete face
    # connectivity digest and subtree ids, so a reloaded/changed artifact
    # cannot reuse another artifact's local indexing.
    layer_boundary_topologies: dict[Any, Any] = field(
        default_factory=dict, compare=False, repr=False
    )
    active_layer_seam: Any | None = field(default=None, compare=False, repr=False)
    inherited_frozen_vertex_ids: Any | None = field(default=None, compare=False, repr=False)
    inherited_surface_provenance_known: bool = True
