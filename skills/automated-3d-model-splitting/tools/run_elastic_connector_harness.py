"""Exercise the thin-rim elastic local-connector policy without a vendor 3MF."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from split3mf import common  # noqa: E402

common.load_core_dependencies()

from split3mf.connector_planning import (  # noqa: E402
    local_connector_spec_for_interface,
)
from split3mf.local_connectors import (  # noqa: E402
    build_synthetic_local_connector_pair,
    plan_local_connector,
)
from split3mf.validation import validate_mesh_in_memory  # noqa: E402


def run_case(name: str, rim_safety_mm: float, footprint_safety_mm: float) -> dict:
    spec = local_connector_spec_for_interface(
        fit_clearance_mm=0.50,
        bottom_clearance_mm=0.25,
        lead_in_mm=0.60,
        safe_engagement_depth_mm=footprint_safety_mm,
        safe_backing_depth_mm=rim_safety_mm,
    )
    pair = build_synthetic_local_connector_pair(spec, samples=64)
    child_audit = validate_mesh_in_memory(pair["child"])
    parent_audit = validate_mesh_in_memory(pair["parent"])
    generated_total_depth_mm = (
        spec.full_boundary_backing_depth_mm
        + spec.engagement_depth_mm
        + spec.socket_bottom_clearance_mm
    )
    passed = bool(
        spec.full_boundary_backing_depth_mm <= rim_safety_mm + 1e-9
        and abs(generated_total_depth_mm - footprint_safety_mm) <= 1e-9
        and child_audit["watertight"]
        and parent_audit["watertight"]
        and child_audit["open_edges"] == 0
        and parent_audit["open_edges"] == 0
    )
    return {
        "name": name,
        "passed": passed,
        "rim_safety_mm": rim_safety_mm,
        "footprint_safety_mm": footprint_safety_mm,
        "backing_depth_mm": spec.full_boundary_backing_depth_mm,
        "engagement_depth_mm": spec.engagement_depth_mm,
        "bottom_clearance_mm": spec.socket_bottom_clearance_mm,
        "generated_total_depth_mm": generated_total_depth_mm,
        "peg_width_mm": spec.peg_width_mm,
        "elastic_backing_scale": spec.elastic_backing_scale,
        "elastic_engagement_scale": spec.elastic_engagement_scale,
        "elastic_lateral_scale": spec.elastic_lateral_scale,
        "compact_peg_enabled": spec.compact_peg_enabled,
        "child_topology": child_audit,
        "parent_topology": parent_audit,
    }


def main() -> int:
    cases = [
        run_case("full_3_plus_5", 8.25, 8.25),
        run_case("engagement_shrinks_first", 6.0, 6.0),
        run_case("engagement_reaches_zero", 5.0, 3.25),
        run_case("backing_shrinks_after_zero", 5.0, 2.0),
        run_case("thin_rim_separate_budget", 1.095, 5.0),
    ]
    expected = {
        "full_3_plus_5": (3.0, 5.0, True),
        "engagement_shrinks_first": (3.0, 2.75, True),
        "engagement_reaches_zero": (3.0, 0.0, False),
        "backing_shrinks_after_zero": (1.75, 0.0, False),
        "thin_rim_separate_budget": (1.095, 3.655, True),
    }
    for case in cases:
        backing, engagement, peg_enabled = expected[str(case["name"])]
        case["passed"] = bool(
            case["passed"]
            and abs(float(case["backing_depth_mm"]) - backing) <= 1e-9
            and abs(float(case["engagement_depth_mm"]) - engagement) <= 1e-9
            and bool(case["compact_peg_enabled"]) == peg_enabled
        )
    sample_count = 2089
    angles = np.linspace(0.0, 2.0 * np.pi, sample_count, endpoint=False)
    dense_boundary = np.column_stack(
        (
            8.0 * np.cos(angles),
            6.0 * np.sin(angles),
            0.08 * np.sin(7.0 * angles),
        )
    )
    dense_spec = local_connector_spec_for_interface(
        fit_clearance_mm=0.50,
        bottom_clearance_mm=0.25,
        lead_in_mm=0.60,
        safe_engagement_depth_mm=5.0,
        safe_backing_depth_mm=1.095,
    )
    dense_started = time.perf_counter()
    dense_plan = plan_local_connector(
        dense_boundary,
        np.asarray([0.0, 0.0, -1.0]),
        dense_spec,
        samples=64,
    )
    dense_elapsed = float(time.perf_counter() - dense_started)
    dense_case = {
        "name": "dense_2089_point_planning",
        "passed": bool(
            dense_elapsed <= 5.0
            and dense_plan["full_backing_taper_reserved"]
            and dense_plan["compact_peg_enabled"]
        ),
        "boundary_vertices": sample_count,
        "elapsed_seconds": dense_elapsed,
        "time_budget_seconds": 5.0,
        "fit_scale": float(dense_plan["fit_scale"]),
    }
    result = {
        "passed": all(case["passed"] for case in cases) and dense_case["passed"],
        "cases": cases,
        "dense_planning": dense_case,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if not result["passed"]:
        raise AssertionError("elastic local-connector harness failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
