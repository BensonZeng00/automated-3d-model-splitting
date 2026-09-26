#!/usr/bin/env python3
"""Guard against folded, blade-like bottom caps on dense curved boundaries."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from split3mf import common  # noqa: E402

common.load_core_dependencies()

from split3mf.domain import CapDecision  # noqa: E402
from split3mf.part_geometry import (  # noqa: E402
    boundary_cap_distances,
    cap_decision_patch_quality,
    cap_decision_patch_quality_preflight,
)


SAFE_MAXIMUM_MM = 2.191061965836216


class ThinCurvedParentProbe:
    """Deterministic stand-in for the thin parent behind the real P01 loop."""

    def safety_limit(self, points, directions, global_ceiling_mm):
        safe = min(float(global_ceiling_mm), SAFE_MAXIMUM_MM)
        return safe, {
            "parent_thickness_min_mm": safe + 0.05,
            "parent_thickness_hit_vertices": int(len(points)),
            "parent_thickness_probe_vertices": int(len(points)),
            "parent_thickness_is_lower_bound": False,
            "parent_thickness_clearance_mm": 0.05,
            "safe_maximum_inward_depth_mm": safe,
        }


def curved_loop(point_count: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, point_count, endpoint=False)
    return np.column_stack(
        (
            8.0 * np.cos(angles),
            4.0 * np.sin(angles),
            0.64875 * np.sin(2.0 * angles),
        )
    )


def run(point_count: int, max_seconds: float) -> dict:
    points = curved_loop(point_count)
    inward = np.asarray([0.0, 0.0, 1.0])
    input_directions = np.tile(inward, (point_count, 1))
    started = time.perf_counter()
    distances, directions, record = boundary_cap_distances(
        points=points,
        fallback_inward=inward,
        inward_directions=input_directions,
        fixed_depth_mm=3.0,
        flat_clearance_mm=0.0,
        cap_mode="adaptive",
        planar_extra_limit_mm=7.0,
        parent_thickness_probe=ThinCurvedParentProbe(),
    )
    decision = CapDecision(
        mode=str(record["cap_mode"]),
        source_vertex_ids=tuple(range(point_count)),
        fit_points=points,
        directions=directions,
        distances=distances,
        record=record,
        source_points=points.copy(),
    )
    selected_quality = cap_decision_patch_quality(
        decision,
        inward,
        points,
        input_directions,
        lead_in_mm=0.60,
        lead_in_applied=False,
    )

    old_folded_decision = CapDecision(
        mode="local-offset",
        source_vertex_ids=tuple(range(point_count)),
        fit_points=points,
        directions=input_directions,
        distances=np.full(point_count, SAFE_MAXIMUM_MM),
        record={"cap_mode": "local-offset"},
        source_points=points.copy(),
    )
    rejected_quality = cap_decision_patch_quality_preflight(
        old_folded_decision,
        inward,
        points,
        input_directions,
        lead_in_mm=0.60,
        lead_in_applied=False,
        defer_cap_triangulation=True,
    )
    elapsed = time.perf_counter() - started

    if record["cap_mode"] != "flat":
        raise AssertionError(f"expected flat cap, got {record['cap_mode']!r}")
    if not record.get("thin_parent_planar_floor_applied"):
        raise AssertionError("thin-parent planar floor was not applied")
    if float(distances.min()) < 0.08 - 1e-9:
        raise AssertionError("selected common plane is shallower than 0.08 mm")
    if not selected_quality["valid"]:
        raise AssertionError("selected common-plane cap failed the quality gate")
    if rejected_quality["valid"]:
        raise AssertionError("old translated nonplanar blade cap was not rejected")
    if elapsed > max_seconds:
        raise AssertionError(
            f"cap harness took {elapsed:.3f}s; budget is {max_seconds:.3f}s"
        )

    return {
        "passed": True,
        "point_count": point_count,
        "selected_cap_mode": record["cap_mode"],
        "minimum_depth_mm": round(float(distances.min()), 6),
        "maximum_depth_mm": round(float(distances.max()), 6),
        "selected_cap_planarity_error_mm": selected_quality[
            "cap_surface_quality"
        ]["maximum_planarity_error_mm"],
        "old_folded_cap_planarity_error_mm": rejected_quality[
            "cap_surface_quality"
        ]["maximum_planarity_error_mm"],
        "old_folded_cap_rejected": True,
        "elapsed_seconds": round(elapsed, 6),
        "time_budget_seconds": max_seconds,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=2089)
    parser.add_argument("--max-seconds", type=float, default=8.0)
    arguments = parser.parse_args()
    if arguments.points < 3:
        parser.error("--points must be at least 3")
    if arguments.max_seconds <= 0.0:
        parser.error("--max-seconds must be positive")
    print(
        json.dumps(
            run(arguments.points, arguments.max_seconds),
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
