from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
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
class SeamSmoothingPolicy:
    """Physical and topology budgets for printable seam smoothing."""

    profile: str
    p95_displacement_mm: float
    maximum_displacement_mm: float
    maximum_bbox_diagonal_ratio: float
    maximum_affected_face_ratio: float
    maximum_affected_area_ratio: float
    maximum_affected_vertex_ratio: float
    maximum_affected_vertices: int
    maximum_topology_layers: int
    maximum_introduced_reversed_ratio: float
    maximum_reversed_cluster_faces: int
    maximum_reversed_cluster_ratio: float
    minimum_reversed_angle_degrees: float
    maximum_edge_stretch_ratio: float
    allow_sparse_seam_reversals: bool

    @classmethod
    def named(cls, profile: str) -> "SeamSmoothingPolicy":
        policies = {
            "source-conservative": cls(
                "source-conservative", 10.0, 10.0, 1.0, 0.02, 0.15, 0.15, 5000, 8,
                0.0, 0, 0.0, 3.0, 8.0, False,
            ),
            "print-balanced": cls(
                "print-balanced", 10.0, 10.0, 1.0, 0.02, 0.15, 0.15, 5000, 8,
                0.15, 32, 0.02, 0.01, 16.0, True,
            ),
            "print-smooth": cls(
                "print-smooth", 10.0, 10.0, 1.0, 0.02, 0.15, 0.15, 5000, 8,
                0.15, 64, 0.02, 0.001, 64.0, True,
            ),
        }
        try:
            return policies[str(profile)]
        except KeyError as exc:
            raise ValueError(f"unsupported seam smoothing profile: {profile}") from exc


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
    visible_interface_simplification_tolerance: float = 1.0
    smoothing_policy: SeamSmoothingPolicy = field(
        default_factory=lambda: SeamSmoothingPolicy.named("print-balanced")
    )

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace) -> "PlanarArcRetopologyConfig":
        surface_band_validation = str(
            getattr(namespace, "surface_band_validation", "strict")
        )
        smoothing_policy = SeamSmoothingPolicy.named(
            getattr(namespace, "seam_smoothing_profile", "print-balanced")
        )
        maximum_boundary_displacement_mm = float(
            getattr(namespace, "maximum_boundary_displacement_mm", 10.0)
        )
        smoothing_policy = replace(
            smoothing_policy,
            maximum_displacement_mm=maximum_boundary_displacement_mm,
            p95_displacement_mm=maximum_boundary_displacement_mm,
        )
        if getattr(namespace, 'boundary_shape', 'smooth') != 'smooth':
            raise ValueError('Only smooth boundary mode is supported')
        return cls(
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
            visible_interface_simplification_tolerance=float(
                getattr(namespace, "visible_interface_simplification_tolerance", 1.0)
            ),
            smoothing_policy=smoothing_policy,
        )

    @property
    def maximum_safe_target_offset_mm(self) -> float:
        return float(self.retopology_band_mm * self.maximum_band_fraction)


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
