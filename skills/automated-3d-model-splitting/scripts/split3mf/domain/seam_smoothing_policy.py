from __future__ import annotations

from dataclasses import dataclass

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
