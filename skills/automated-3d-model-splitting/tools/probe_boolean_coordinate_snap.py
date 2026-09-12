from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from split3mf import common  # noqa: E402

common.load_core_dependencies()

from split3mf.local_connectors import (  # noqa: E402
    _manifold64,
    _trimesh_from_manifold64,
)


def audit(mesh):
    triangles = np.asarray(mesh.vertices, dtype=np.float64)[np.asarray(mesh.faces)]
    areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    scale = max(float(np.linalg.norm(np.ptp(mesh.vertices, axis=0))), 1.0)
    threshold = scale * scale * 1e-14
    edge_counts = np.bincount(mesh.edges_unique_inverse)
    bad = np.flatnonzero(areas <= threshold)
    details = []
    for face_id in bad[:8]:
        face = np.asarray(mesh.faces[int(face_id)], dtype=np.int64)
        points = np.asarray(mesh.vertices, dtype=np.float64)[face]
        lengths = [
            float(np.linalg.norm(points[(index + 1) % 3] - points[index]))
            for index in range(3)
        ]
        details.append(
            {
                "face_id": int(face_id),
                "indices": face.tolist(),
                "points": points.tolist(),
                "edge_lengths_mm": lengths,
                "double_area_mm2": float(areas[int(face_id)]),
            }
        )
    return {
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "winding": bool(mesh.is_winding_consistent),
        "boundary_edges": int(np.count_nonzero(edge_counts == 1)),
        "over_shared_edges": int(np.count_nonzero(edge_counts > 2)),
        "degenerate_faces": int(len(bad)),
        "minimum_double_area_mm2": float(areas.min()),
        "threshold_mm2": float(threshold),
        "bad_face_details": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-npz", type=Path, required=True)
    args = parser.parse_args()
    cache = np.load(args.cache_npz)
    parent = common.trimesh.Trimesh(
        vertices=cache["parent_vertices"], faces=cache["parent_faces"], process=False
    )
    cutter = common.trimesh.Trimesh(
        vertices=cache["cutter_vertices"], faces=cache["cutter_faces"], process=False
    )
    difference = _manifold64(parent) - _manifold64(cutter)
    simplified = difference.as_original().simplify(1e-8)
    exported = _trimesh_from_manifold64(simplified)
    result = {"before_snap": audit(exported), "snap_probes": []}
    source_volume = float(abs(exported.volume))
    for grid_mm in (1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5):
        snapped = exported.copy()
        snapped.vertices = (
            np.rint(np.asarray(snapped.vertices, dtype=np.float64) / grid_mm) * grid_mm
        )
        manifold = _manifold64(snapped, tolerance_mm=grid_mm)
        if manifold.is_empty():
            result["snap_probes"].append(
                {"grid_mm": grid_mm, "passed": False, "status": str(manifold.status())}
            )
            continue
        roundtrip = _trimesh_from_manifold64(manifold.as_original().simplify(grid_mm))
        record = audit(roundtrip)
        record.update(
            {
                "grid_mm": grid_mm,
                "passed": bool(
                    record["watertight"]
                    and record["winding"]
                    and record["boundary_edges"] == 0
                    and record["over_shared_edges"] == 0
                    and record["degenerate_faces"] == 0
                ),
                "volume_delta_mm3": abs(float(abs(roundtrip.volume)) - source_volume),
            }
        )
        result["snap_probes"].append(record)
        if record["passed"]:
            break
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
