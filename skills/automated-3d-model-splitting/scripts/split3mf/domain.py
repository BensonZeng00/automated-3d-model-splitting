from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .common import Component


@dataclass(frozen=True)
class SplitConfig:
    namespace: argparse.Namespace
    input_path: Path
    preflight_checks: dict[str, Any]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.namespace, name)


@dataclass(frozen=True)
class PlanarArcRetopologyConfig:
    """One production boundary policy; there are intentionally no legacy modes."""

    target_samples: int = 384
    smooth_passes: int = 28
    retopology_band_mm: float = 3.0
    target_slope_degrees: float = 45.0
    minimum_slope_degrees: float = 30.0
    maximum_slope_degrees: float = 75.0
    maximum_band_fraction: float = 0.45
    connector_slope_validation: str = "advisory"
    surface_band_validation: str = "strict"
    connector_surface_validation: str = "strict"
    preserve_confirmed_seam: bool = False

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace) -> "PlanarArcRetopologyConfig":
        surface_band_validation = str(
            getattr(namespace, "surface_band_validation", "strict")
        )
        return cls(
            preserve_confirmed_seam=getattr(namespace, 'boundary_shape', 'source') == 'source',
            target_samples=int(namespace.boundary_target_samples),
            smooth_passes=int(namespace.boundary_smooth_passes),
            retopology_band_mm=float(namespace.boundary_retopology_band_mm),
            connector_slope_validation=str(
                getattr(namespace, "connector_slope_validation", "advisory")
            ),
            # A visually reviewed target may use more of the explicitly
            # requested band, but it must still remain inside that real
            # surface neighborhood. Strict mode keeps the original margin.
            maximum_band_fraction=(
                0.60 if surface_band_validation == "advisory" else 0.45
            ),
            surface_band_validation=surface_band_validation,
            connector_surface_validation=str(
                getattr(namespace, "connector_surface_validation", "strict")
            ),
        )

    @property
    def maximum_safe_target_offset_mm(self) -> float:
        return float(self.retopology_band_mm * self.maximum_band_fraction)


@dataclass(frozen=True)
class PlanarArcRetopologyContext:
    config: PlanarArcRetopologyConfig
    failure_sink: Callable[[dict[str, Any]], None] | None = None
    curve_review_sink: Callable[[Any], None] | None = None


@dataclass
class LoadedProject:
    vertices: Any
    faces: Any
    paint_tokens: list[str]
    settings: dict[str, Any]


@dataclass
class RecognitionResult:
    components: list[Component]
    ignored: list[dict[str, Any]] = field(default_factory=list)
    merged_tiny: list[dict[str, Any]] = field(default_factory=list)
    semantic_preserved_tiny: list[dict[str, Any]] = field(default_factory=list)
    exterior_record: dict[str, Any] = field(default_factory=dict)


@dataclass
class AssemblyPlan:
    body_index: int
    parents: dict[int, int | None]
    children: dict[int, list[int]]
    depths: dict[int, int]
    layers: list[dict[str, Any]]


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


@dataclass
class PartBuildResult:
    mesh: Any
    stats: dict[str, Any]


@dataclass
class ValidationReport:
    valid: bool
    details: dict[str, Any]
