from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from split3mf import common  # noqa: E402

common.load_core_dependencies()

from split3mf.mesh import (  # noqa: E402
    boundary_loops,
    face_edges_among_vertices,
    orient_mesh_faces_consistently,
    triangulate_boundary_cap_without_center,
)
from split3mf.package_io import load_colored_mesh_objects_3mf  # noqa: E402


def topology(mesh) -> dict[str, object]:
    counts = np.bincount(mesh.edges_unique_inverse)
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "boundary_edges": int(np.count_nonzero(counts == 1)),
        "over_shared_edges": int(np.count_nonzero(counts > 2)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replace preview preclosure caps while forbidding existing parent chords."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-faces", type=int, required=True)
    parser.add_argument(
        "--loop-index",
        type=int,
        default=None,
        help="Inspect only one boundary loop; useful for a fast real-model regression.",
    )
    args = parser.parse_args()

    loaded = load_colored_mesh_objects_3mf(args.input)
    if len(loaded) != 1:
        raise ValueError("preclosure harness requires one mesh object")
    preview = loaded[0]["mesh"]
    source_face_count = int(args.source_faces)
    if not 0 < source_face_count < len(preview.faces):
        raise ValueError("source face count must exclude at least one generated cap face")

    vertices = np.asarray(preview.vertices, dtype=np.float64).tolist()
    source_faces = np.asarray(preview.faces[:source_face_count], dtype=np.int64)
    output_faces = source_faces.astype(int).tolist()
    loops = boundary_loops(source_faces)
    records: list[dict[str, object]] = []
    for loop_index, loop in enumerate(loops):
        if args.loop_index is not None and int(loop_index) != int(args.loop_index):
            continue
        loop_set = {int(index) for index in loop}
        occupied = face_edges_among_vertices(
            source_faces,
            loop_set,
            len(vertices),
        )
        points = np.asarray([vertices[int(index)] for index in loop], dtype=np.float64)
        following = np.roll(points, -1, axis=0)
        newell = np.array(
            [
                np.sum((points[:, 1] - following[:, 1]) * (points[:, 2] + following[:, 2])),
                np.sum((points[:, 2] - following[:, 2]) * (points[:, 0] + following[:, 0])),
                np.sum((points[:, 0] - following[:, 0]) * (points[:, 1] + following[:, 1])),
            ],
            dtype=np.float64,
        )
        added, record = triangulate_boundary_cap_without_center(
            vertices,
            output_faces,
            [int(index) for index in loop],
            newell,
            occupied_edges=occupied,
        )
        if int(added) != len(loop) - 2:
            raise ValueError(
                f"loop {loop_index} expected {len(loop) - 2} cap faces, got {added}"
            )
        cap_triangles = np.asarray(vertices, dtype=np.float64)[
            np.asarray(output_faces[-int(added):], dtype=np.int64)
        ]
        cap_double_areas = np.linalg.norm(
            np.cross(
                cap_triangles[:, 1] - cap_triangles[:, 0],
                cap_triangles[:, 2] - cap_triangles[:, 0],
            ),
            axis=1,
        )
        degenerate_indices = np.flatnonzero(cap_double_areas <= 1e-12)
        degenerate_details = []
        for local_index in degenerate_indices[:8]:
            face = np.asarray(output_faces[-int(added) + int(local_index)], dtype=np.int64)
            tri_points = np.asarray(vertices, dtype=np.float64)[face]
            degenerate_details.append(
                {
                    "face": face.astype(int).tolist(),
                    "points": np.round(tri_points, 9).tolist(),
                    "edge_lengths_mm": np.round(
                        np.linalg.norm(
                            np.roll(tri_points, -1, axis=0) - tri_points,
                            axis=1,
                        ),
                        12,
                    ).tolist(),
                }
            )
        records.append(
            {
                "loop_index": int(loop_index),
                "vertices": int(len(loop)),
                "minimum_cap_double_area_mm2": float(cap_double_areas.min()),
                "degenerate_cap_faces": int(np.count_nonzero(cap_double_areas <= 1e-12)),
                "degenerate_details": degenerate_details,
                **record,
            }
        )

    candidate = common.trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(output_faces, dtype=np.int64),
        process=False,
    )
    before_winding = topology(candidate)
    winding_record = orient_mesh_faces_consistently(candidate)
    after_winding = topology(candidate)
    print(
        {
            "input_topology": topology(preview),
            "source_faces": source_face_count,
            "generated_cap_faces": int(len(candidate.faces) - source_face_count),
            "loop_records": records,
            "candidate_topology_before_winding": before_winding,
            "winding_record": winding_record,
            "candidate_topology": after_winding,
        }
    )
    no_degenerate_caps = all(
        int(record["degenerate_cap_faces"]) == 0 for record in records
    )
    if args.loop_index is not None:
        return 0 if records and no_degenerate_caps else 1
    return 0 if candidate.is_watertight and candidate.is_winding_consistent and no_degenerate_caps else 1


if __name__ == "__main__":
    raise SystemExit(main())
