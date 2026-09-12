#!/usr/bin/env python3
"""Run deterministic performance guards for inward connector primitives.

The harness intentionally uses synthetic geometry.  It catches the two costly
regressions that previously made a single interface take several minutes:
unordered taper candidates and quadratic dense-polygon ear clipping.
"""

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

from split3mf.inward import tapered_profile_inset_candidates  # noqa: E402
from split3mf.mesh import triangulate_polygon_ear_clip  # noqa: E402


def dense_wavy_loop(point_count: int) -> np.ndarray:
    """Return a deterministic simple loop with many locally reflex vertices."""
    angles = np.linspace(0.0, 2.0 * np.pi, point_count, endpoint=False)
    radii = 50.0 + 3.0 * np.sin(7.0 * angles) + 1.5 * np.sin(19.0 * angles)
    return np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))


def run(point_count: int, max_seconds: float) -> dict:
    candidates = tapered_profile_inset_candidates(0.4, 2.25)
    inset_values = [
        float(candidate["effective_insert_shrink_mm"])
        for candidate in candidates
    ]
    if inset_values != sorted(inset_values, reverse=True):
        raise AssertionError("taper candidates are not ordered largest-first")
    if len(inset_values) != len(set(inset_values)):
        raise AssertionError("taper candidates contain duplicate inset values")

    points = dense_wavy_loop(point_count)
    started = time.perf_counter()
    faces = triangulate_polygon_ear_clip(points)
    elapsed = time.perf_counter() - started
    expected_faces = point_count - 2
    if len(faces) != expected_faces:
        raise AssertionError(
            f"dense triangulation emitted {len(faces)} faces; expected {expected_faces}"
        )
    if elapsed > max_seconds:
        raise AssertionError(
            f"dense triangulation took {elapsed:.3f}s; budget is {max_seconds:.3f}s"
        )
    return {
        "passed": True,
        "point_count": point_count,
        "face_count": len(faces),
        "triangulation_seconds": round(elapsed, 6),
        "time_budget_seconds": max_seconds,
        "candidate_count": len(candidates),
        "candidates_descending": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=2089)
    parser.add_argument("--max-seconds", type=float, default=5.0)
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
