from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .common import Component


@dataclass(frozen=True)
class SplitConfig:
    namespace: argparse.Namespace
    input_path: Path
    preflight_checks: dict[str, Any]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.namespace, name)


@dataclass(frozen=True)
class BoundaryFairingConfig:
    mode: str
    radius_mm: float
    max_displacement_mm: float
    feature_angle_degrees: float
    fidelity_weight: float
    legacy_iterations: int
    legacy_lambda: float
    legacy_mu: float

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace) -> "BoundaryFairingConfig":
        return cls(
            mode=str(namespace.boundary_fairing_mode),
            radius_mm=float(namespace.boundary_fairing_radius_mm),
            max_displacement_mm=float(namespace.boundary_max_displacement_mm),
            feature_angle_degrees=float(namespace.boundary_feature_angle_deg),
            fidelity_weight=float(namespace.boundary_fidelity_weight),
            legacy_iterations=int(namespace.smooth_iterations),
            legacy_lambda=float(namespace.lambda_factor),
            legacy_mu=float(namespace.mu_factor),
        )


@dataclass(frozen=True)
class BoundaryFairingContext:
    config: BoundaryFairingConfig
    source_surface_normals: Any


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


@dataclass
class PartBuildResult:
    mesh: Any
    stats: dict[str, Any]


@dataclass
class ValidationReport:
    valid: bool
    details: dict[str, Any]
