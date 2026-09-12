"""Fast deterministic smoke harness for the production boundary algorithm."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from split3mf.planar_arc import build_planar_arc_boundary


def main() -> None:
    angle = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
    ripple = 0.12 * np.sin(37.0 * angle)
    points = np.column_stack(
        (
            (9.0 + ripple) * np.cos(angle),
            (6.0 + ripple) * np.sin(angle),
            0.05 * np.sin(5.0 * angle),
        )
    )
    result = build_planar_arc_boundary(
        points,
        np.arange(len(points), dtype=np.int64),
        target_samples=384,
        smooth_passes=28,
        maximum_target_offset_mm=0.54,
    )
    record = dict(result.record)
    record["passed"] = bool(
        record["projected_ripple_after_mm"]
        < record["projected_ripple_before_mm"]
        and len(result.guide_points) == 384
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    if not record["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
