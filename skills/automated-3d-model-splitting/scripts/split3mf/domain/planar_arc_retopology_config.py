from __future__ import annotations

from dataclasses import dataclass, field, replace
import argparse
from .seam_smoothing_policy import SeamSmoothingPolicy

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
            smoothing_policy=smoothing_policy,
        )

    @property
    def maximum_safe_target_offset_mm(self) -> float:
        return float(self.retopology_band_mm * self.maximum_band_fraction)
