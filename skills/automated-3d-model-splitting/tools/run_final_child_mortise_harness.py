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
    _validated_boolean_manifold64,
    build_clearance_cutter_from_final_part,
    subtract_socket_cutters,
)
from split3mf.mesh import (  # noqa: E402
    boundary_loops,
    face_edges_among_vertices,
    orient_mesh_faces_consistently,
    triangulate_boundary_cap_without_center,
)
from split3mf.package_io import load_colored_mesh_objects_3mf  # noqa: E402


def _newell(points: np.ndarray) -> np.ndarray:
    following = np.roll(points, -1, axis=0)
    return np.asarray(
        [
            np.sum((points[:, 1] - following[:, 1]) * (points[:, 2] + following[:, 2])),
            np.sum((points[:, 2] - following[:, 2]) * (points[:, 0] + following[:, 0])),
            np.sum((points[:, 0] - following[:, 0]) * (points[:, 1] + following[:, 1])),
        ],
        dtype=np.float64,
    )


def _load_one(path: Path):
    objects = load_colored_mesh_objects_3mf(path)
    if len(objects) != 1:
        raise ValueError(f"expected one mesh object in {path}, got {len(objects)}")
    return objects[0]["mesh"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild one real closed parent and subtract one final-child clearance cutter."
    )
    parser.add_argument("--parent-preview", type=Path, required=True)
    parser.add_argument("--parent-source-faces", type=int, required=True)
    parser.add_argument("--final-child", type=Path, required=True)
    parser.add_argument("--radial-clearance-mm", type=float, default=0.05)
    parser.add_argument("--cache-npz", type=Path)
    parser.add_argument(
        "--probe-input-tolerances",
        action="store_true",
        help="Try multiple Mesh64 input tolerances against the cached real pair.",
    )
    args = parser.parse_args()

    if args.cache_npz is not None and args.cache_npz.exists():
        cache = np.load(args.cache_npz)
        parent = common.trimesh.Trimesh(
            vertices=cache["parent_vertices"],
            faces=cache["parent_faces"],
            process=False,
        )
        cutter = common.trimesh.Trimesh(
            vertices=cache["cutter_vertices"],
            faces=cache["cutter_faces"],
            process=False,
        )
        child_faces = int(cache["child_faces"][0])
        loop_sizes = [int(value) for value in cache["loop_sizes"]]
        cutter_record = {"source": "cached_real_pair"}
    else:
        preview = _load_one(args.parent_preview)
        source_faces = np.asarray(
            preview.faces[: int(args.parent_source_faces)], dtype=np.int64
        )
        vertices = np.asarray(preview.vertices, dtype=np.float64).tolist()
        faces = source_faces.astype(int).tolist()
        loop_sizes = []
        for loop in boundary_loops(source_faces):
            loop_ids = [int(value) for value in loop]
            points = np.asarray([vertices[index] for index in loop_ids], dtype=np.float64)
            added, _record = triangulate_boundary_cap_without_center(
                vertices,
                faces,
                loop_ids,
                _newell(points),
                occupied_edges=face_edges_among_vertices(
                    source_faces,
                    set(loop_ids),
                    len(vertices),
                ),
            )
            if int(added) != len(loop_ids) - 2:
                raise ValueError(f"parent loop closure failed: loop={len(loop_ids)}, added={added}")
            loop_sizes.append(len(loop_ids))
        parent = common.trimesh.Trimesh(
            vertices=np.asarray(vertices, dtype=np.float64),
            faces=np.asarray(faces, dtype=np.int64),
            process=False,
        )
        orient_mesh_faces_consistently(parent)
        if not parent.is_watertight or not parent.is_winding_consistent:
            raise ValueError("rebuilt parent is not a closed, consistently wound solid")
        child = _load_one(args.final_child)
        child_faces = int(len(child.faces))
        cutter, cutter_record = build_clearance_cutter_from_final_part(
            child,
            radial_clearance_mm=float(args.radial_clearance_mm),
        )
        if args.cache_npz is not None:
            args.cache_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.cache_npz,
                parent_vertices=np.asarray(parent.vertices, dtype=np.float64),
                parent_faces=np.asarray(parent.faces, dtype=np.int64),
                cutter_vertices=np.asarray(cutter.vertices, dtype=np.float64),
                cutter_faces=np.asarray(cutter.faces, dtype=np.int64),
                child_faces=np.asarray([child_faces], dtype=np.int64),
                loop_sizes=np.asarray(loop_sizes, dtype=np.int64),
            )
    if args.probe_input_tolerances:
        probes = []
        for tolerance_mm in (0.0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5):
            difference = _manifold64(
                parent, tolerance_mm=tolerance_mm
            ) - _manifold64(cutter, tolerance_mm=tolerance_mm)
            try:
                candidate, cleanup = _validated_boolean_manifold64(difference)
                probes.append(
                    {
                        "input_tolerance_mm": tolerance_mm,
                        "passed": True,
                        "faces": int(len(candidate.faces)),
                        "cleanup": cleanup,
                    }
                )
                break
            except ValueError as error:
                probes.append(
                    {
                        "input_tolerance_mm": tolerance_mm,
                        "passed": False,
                        "error": str(error),
                    }
                )
        print(json.dumps({"input_tolerance_probes": probes}, ensure_ascii=False, indent=2))
        return 0 if any(item["passed"] for item in probes) else 1
    result, boolean_record = subtract_socket_cutters(parent, [cutter])
    edge_counts = np.bincount(result.edges_unique_inverse)
    triangles = np.asarray(result.vertices)[np.asarray(result.faces)]
    double_areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    output = {
        "parent_source_faces": int(args.parent_source_faces),
        "closed_parent_faces": int(len(parent.faces)),
        "parent_loop_sizes": loop_sizes,
        "parent_watertight": bool(parent.is_watertight),
        "child_faces": child_faces,
        "cutter": cutter_record,
        "result_faces": int(len(result.faces)),
        "result_watertight": bool(result.is_watertight),
        "result_winding_consistent": bool(result.is_winding_consistent),
        "boundary_edges": int(np.count_nonzero(edge_counts == 1)),
        "over_shared_edges": int(np.count_nonzero(edge_counts > 2)),
        "minimum_double_area_mm2": float(double_areas.min()),
        "boolean": boolean_record,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
