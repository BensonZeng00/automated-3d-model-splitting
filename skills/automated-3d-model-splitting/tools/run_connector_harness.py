#!/usr/bin/env python3
"""Run small deterministic local-connector geometry cases.

This harness deliberately avoids vendor files.  It exercises the same planner,
45-degree backing, annulus triangulator, and topology audit used by production
so a geometry change can be reviewed in seconds before a large painted 3MF is
processed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from split3mf import common  # noqa: E402

common.load_core_dependencies()

from split3mf.connector_topology import (  # noqa: E402
    triangulate_bounded_ring_strip,
    triangulate_connector_annulus,
    triangulate_star_shaped_annulus,
)
from split3mf.connector_geometry import (  # noqa: E402
    printable_backing_profile,
)
from split3mf.connector_planning import (  # noqa: E402
    local_connector_spec_for_interface,
)
from split3mf.local_connectors import plan_local_connector  # noqa: E402
from split3mf.part_geometry import add_local_male_connector_and_backing  # noqa: E402
from split3mf.mesh import point_in_poly, point_on_poly_boundary  # noqa: E402


@dataclass(frozen=True)
class HarnessResult:
    case: str
    passed: bool
    boundary_vertices: int
    backing_vertices: int
    connector_vertices: int
    backing_depth_mm: float
    connector_edges_outside_backing: int
    topology: dict
    plan: dict
    error: str = ""


def _cubic(p0, p1, p2, p3, samples: int) -> np.ndarray:
    parameter = np.linspace(0.0, 1.0, int(samples), endpoint=False)
    remaining = 1.0 - parameter
    return (
        (remaining**3)[:, None] * np.asarray(p0, dtype=np.float64)[None, :]
        + (3.0 * remaining**2 * parameter)[:, None]
        * np.asarray(p1, dtype=np.float64)[None, :]
        + (3.0 * remaining * parameter**2)[:, None]
        * np.asarray(p2, dtype=np.float64)[None, :]
        + (parameter**3)[:, None] * np.asarray(p3, dtype=np.float64)[None, :]
    )


def boundary_fixture(name: str) -> np.ndarray:
    if name == "square":
        return np.asarray(
            [
                [-8.0, -8.0, 0.0],
                [8.0, -8.0, 0.0],
                [8.0, 8.0, 0.0],
                [-8.0, 8.0, 0.0],
            ],
            dtype=np.float64,
        )
    if name == "curved":
        angles = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
        return np.column_stack(
            (
                9.0 * np.cos(angles),
                7.5 * np.sin(angles),
                0.18 * np.sin(3.0 * angles),
            )
        )
    if name == "smooth-ripple":
        # A smooth but strongly non-planar rim whose inset rings may choose a
        # different cyclic start vertex.  This catches height transfer by raw
        # array index, the source of cratered production backing walls.
        angles = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
        radius = 10.0 + 1.10 * np.cos(3.0 * angles)
        return np.column_stack(
            (
                radius * np.cos(angles),
                0.78 * radius * np.sin(angles),
                0.75 * np.sin(2.0 * angles) + 0.20 * np.cos(5.0 * angles),
            )
        )
    if name == "concave-v":
        left_top = (-7.5, 4.0, 0.0)
        bottom = (0.0, -7.0, 0.0)
        right_top = (7.5, 4.0, 0.0)
        return np.vstack(
            (
                _cubic(left_top, (-5.6, 0.0, 0.0), (-2.4, -7.0, 0.0), bottom, 128),
                _cubic(bottom, (2.4, -7.0, 0.0), (5.6, 0.0, 0.0), right_top, 128),
                _cubic(right_top, (3.2, 1.0, 0.0), (-3.2, 1.0, 0.0), left_top, 128),
            )
        )
    raise ValueError(f"unknown harness case {name!r}")


def run_mismatched_ripple_strip_case() -> dict:
    """Reproduce the 996-to-433 nonplanar Lulu backing regression."""

    outer_count = 996
    inner_count = 433
    outer_angle = np.linspace(0.0, 2.0 * np.pi, outer_count, endpoint=False)
    inner_angle = np.linspace(0.0, 2.0 * np.pi, inner_count, endpoint=False)
    outer = np.column_stack(
        (
            18.0 * np.cos(outer_angle),
            14.0 * np.sin(outer_angle),
            0.825 * np.sin(3.0 * outer_angle),
        )
    )
    inner = np.column_stack(
        (
            17.5 * np.cos(inner_angle),
            13.5 * np.sin(inner_angle),
            -0.5 + 0.4125 * np.sin(3.0 * inner_angle),
        )
    )
    strip = triangulate_bounded_ring_strip(
        list(range(outer_count)),
        outer,
        list(range(outer_count, outer_count + inner_count)),
        inner,
        outer[:, :2],
        inner[:, :2],
        maximum_fanout=4,
    )
    return {
        "case": "mismatched-ripple-996-to-433",
        "passed": bool(
            strip.audit.valid
            and len(strip.faces) == outer_count + len(strip.inner_ids)
            and strip.audit.maximum_fanout <= 4
            and strip.audit.maximum_cross_edge_mm < 2.0
        ),
        "outer_vertices": outer_count,
        "inner_vertices": inner_count,
        "source_axial_spread_mm": float(np.ptp(outer[:, 2])),
        "audit": asdict(strip.audit),
    }


def _project(points: np.ndarray, plan: dict) -> np.ndarray:
    centered = np.asarray(points, dtype=np.float64) - np.asarray(
        plan["center"], dtype=np.float64
    )
    return np.column_stack(
        (
            centered @ np.asarray(plan["u"], dtype=np.float64),
            centered @ np.asarray(plan["v"], dtype=np.float64),
        )
    )


def _connector_edges_outside(outer: np.ndarray, inner: np.ndarray) -> int:
    outside = 0
    for left, right in zip(inner, np.roll(inner, -1, axis=0)):
        for ratio in np.linspace(0.0, 1.0, 17):
            point = left * (1.0 - ratio) + right * ratio
            if not (
                point_in_poly(point, outer)
                or point_on_poly_boundary(point, outer)
            ):
                outside += 1
                break
    return int(outside)


def run_case(name: str, *, strategy: str = "auto") -> HarnessResult:
    boundary = boundary_fixture(name)
    spec = local_connector_spec_for_interface(
        fit_clearance_mm=0.0,
        bottom_clearance_mm=0.0,
        lead_in_mm=0.80,
        safe_engagement_depth_mm=5.0,
        surface_validation_mode="advisory",
    )
    try:
        plan = plan_local_connector(
            boundary,
            np.asarray([0.0, 0.0, -1.0]),
            spec,
            samples=64,
        )
        backing_profile = printable_backing_profile(
            boundary,
            plan,
            child_clearance=True,
        )
        backing = np.asarray(backing_profile.rings[-1], dtype=np.float64)
        depth = float(backing_profile.taper_depth_mm)
        outer = _project(backing, plan)
        inner = _project(np.asarray(plan["peg_top"]), plan)
        outer_ids = list(range(len(outer)))
        inner_ids = list(range(len(outer), len(outer) + len(inner)))
        triangulator = (
            triangulate_star_shaped_annulus
            if strategy == "star"
            else triangulate_connector_annulus
        )
        faces, audit = triangulator(
            outer_ids,
            outer,
            inner_ids,
            inner,
        )
        outside = _connector_edges_outside(outer, inner)
        production_vertices = [point.copy() for point in boundary]
        production_faces: list[list[int]] = []
        boolean_cutters = []
        production_record = add_local_male_connector_and_backing(
            output_vertices=production_vertices,
            output_faces=production_faces,
            boundary_ids=list(range(len(boundary))),
            boundary_points=boundary,
            inward=np.asarray([0.0, 0.0, -1.0]),
            spec=spec,
            boolean_cutters=boolean_cutters,
        )
        cutter_roles = [
            str(cutter.metadata.get("local_connector_cutter_role", ""))
            for cutter in boolean_cutters
        ]
        passed = bool(
            audit.valid
            and faces
            and outside == 0
            and int(production_record["backing_profile_layer_count"]) >= 1
            and int(production_record["backing_wedge_dense_sampling_faces"]) >= 0
            and (production_record["backing_wedge_smooth_seam_p95_dihedral_degrees"] is None
                 or production_record["backing_wedge_smooth_seam_p95_dihedral_degrees"] <= 30.0)
            and float(
                production_record["backing_wedge_smooth_seam_over_60_ratio"]
            )
            <= 0.01
            and float(production_record["backing_taper_depth_mm"]) == 3.0
            and 30.0
            <= float(production_record["backing_taper_measured_minimum_degrees"])
            <= 60.0
            and 40.0
            <= float(production_record["backing_taper_measured_median_degrees"])
            <= 50.0
            and 30.0
            <= float(production_record["backing_taper_measured_maximum_degrees"])
            <= 60.0
            and cutter_roles == []
            and all(
                cutter.is_watertight and cutter.is_winding_consistent
                for cutter in boolean_cutters
            )
            and abs(
                float(production_record["backing_clearance_per_side_mm"])
                - 0.0
            )
            <= 1e-9
            and abs(
                float(production_record["backing_bottom_clearance_mm"])
                - 0.0
            )
            <= 1e-9
            and abs(
                float(production_record["backing_clearance_shift_mm"])
                - 0.0
            )
            <= 1e-9
        )
        return HarnessResult(
            case=name,
            passed=passed,
            boundary_vertices=int(len(boundary)),
            backing_vertices=int(len(backing)),
            connector_vertices=int(len(inner)),
            backing_depth_mm=float(depth),
            connector_edges_outside_backing=outside,
            topology=asdict(audit),
            plan={
                "backing_inset_method": str(
                    plan.get("backing_inset_method", "")
                ),
                "fit_scale": float(plan["fit_scale"]),
                "full_backing_taper_reserved": bool(
                    plan["full_backing_taper_reserved"]
                ),
                "boolean_cutter_roles": cutter_roles,
                "backing_clearance_per_side_mm": float(
                    production_record["backing_clearance_per_side_mm"]
                ),
                "backing_bottom_clearance_mm": float(
                    production_record["backing_bottom_clearance_mm"]
                ),
                "backing_clearance_shift_mm": float(
                    production_record["backing_clearance_shift_mm"]
                ),
                "backing_clearance_strategy": str(
                    production_record["backing_clearance_strategy"]
                ),
                "production_backing_profile_layer_count": int(
                    production_record["backing_profile_layer_count"]
                ),
                "production_backing_strip_maximum_fanout": int(
                    production_record["backing_strip_maximum_fanout"]
                ),
                "production_backing_strip_maximum_cross_edge_mm": float(
                    production_record["backing_strip_maximum_cross_edge_mm"]
                ),
                "production_backing_smooth_seam_p95_dihedral_degrees": production_record[
                    "backing_wedge_smooth_seam_p95_dihedral_degrees"],
                "production_backing_smooth_seam_maximum_dihedral_degrees": production_record[
                    "backing_wedge_smooth_seam_maximum_dihedral_degrees"],
                "production_backing_smooth_seam_status": production_record[
                    "backing_wedge_smooth_seam_status"],
                "production_backing_smooth_seam_over_60_ratio": float(
                    production_record[
                        "backing_wedge_smooth_seam_over_60_ratio"
                    ]
                ),
                "production_backing_taper_measured_minimum_degrees": float(
                    production_record["backing_taper_measured_minimum_degrees"]
                ),
                "production_backing_taper_measured_median_degrees": float(
                    production_record["backing_taper_measured_median_degrees"]
                ),
                "production_backing_taper_measured_maximum_degrees": float(
                    production_record["backing_taper_measured_maximum_degrees"]
                ),
            },
        )
    except Exception as error:  # harness reports; production still fails loudly
        return HarnessResult(
            case=name,
            passed=False,
            boundary_vertices=int(len(boundary)),
            backing_vertices=0,
            connector_vertices=0,
            backing_depth_mm=0.0,
            connector_edges_outside_backing=0,
            topology={},
            plan={},
            error=str(error),
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=[
            "all",
            "square",
            "curved",
            "smooth-ripple",
            "concave-v",
            "mismatched-ripple",
        ],
        default="all",
    )
    parser.add_argument(
        "--strategy",
        choices=["auto", "star"],
        default="auto",
    )
    args = parser.parse_args()
    if args.case == "mismatched-ripple":
        result = run_mismatched_ripple_strip_case()
        print(json.dumps([result], indent=2))
        return 0 if result["passed"] else 1
    names = (
        ["square", "curved", "smooth-ripple", "concave-v"]
        if args.case == "all"
        else [str(args.case)]
    )
    results = [run_case(name, strategy=str(args.strategy)) for name in names]
    serialized = [asdict(result) for result in results]
    if args.case == "all":
        serialized.append(run_mismatched_ripple_strip_case())
    print(json.dumps(serialized, indent=2))
    return 0 if all(result["passed"] for result in serialized) else 1


if __name__ == "__main__":
    raise SystemExit(main())
