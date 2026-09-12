from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from split3mf import common  # noqa: E402

common.load_core_dependencies()

from split3mf.connector_planning import local_connector_spec_for_interface  # noqa: E402
from split3mf.inward import add_local_female_boolean_closure  # noqa: E402
from split3mf.local_connectors import subtract_socket_cutters  # noqa: E402
from split3mf.mesh import (  # noqa: E402
    boundary_loops,
    face_edges_among_vertices,
    orient_mesh_faces_consistently,
    triangulate_boundary_cap_without_center,
)
from split3mf.package_io import load_colored_mesh_objects_3mf  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one real Lulu parent-loop mortise Boolean without full recognition."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-faces", type=int, required=True)
    parser.add_argument("--loop-index", type=int, default=0)
    parser.add_argument("--overcut-mm", type=float, required=True)
    args = parser.parse_args()

    loaded = load_colored_mesh_objects_3mf(args.input)
    if len(loaded) != 1:
        raise ValueError("single-interface harness requires one mesh object")
    preview = loaded[0]["mesh"]
    source_faces = np.asarray(
        preview.faces[: int(args.source_faces)], dtype=np.int64
    )
    vertices = np.asarray(preview.vertices, dtype=np.float64).tolist()
    faces = source_faces.astype(int).tolist()
    loops = boundary_loops(source_faces)
    target_loop_index = int(args.loop_index)
    loop = loops[target_loop_index]
    boundary = np.asarray([vertices[int(index)] for index in loop], dtype=np.float64)
    following = np.roll(boundary, -1, axis=0)
    newell = np.asarray(
        [
            np.sum((boundary[:, 1] - following[:, 1]) * (boundary[:, 2] + following[:, 2])),
            np.sum((boundary[:, 2] - following[:, 2]) * (boundary[:, 0] + following[:, 0])),
            np.sum((boundary[:, 0] - following[:, 0]) * (boundary[:, 1] + following[:, 1])),
        ],
        dtype=np.float64,
    )
    inward = -newell / max(float(np.linalg.norm(newell)), 1e-12)
    spec = local_connector_spec_for_interface(
        fit_clearance_mm=0.50,
        bottom_clearance_mm=0.25,
        lead_in_mm=0.60,
        safe_engagement_depth_mm=5.0,
    )
    record, cutters = add_local_female_boolean_closure(
        output_vertices=vertices,
        output_faces=faces,
        boundary_ids=[int(index) for index in loop],
        boundary_points=boundary,
        inward=inward,
        spec=spec,
        occupied_parent_edges=face_edges_among_vertices(
            source_faces,
            {int(index) for index in loop},
            len(vertices),
        ),
    )
    for other_index, other_loop in enumerate(loops):
        if int(other_index) == target_loop_index:
            continue
        other_boundary = np.asarray(
            [vertices[int(index)] for index in other_loop], dtype=np.float64
        )
        other_following = np.roll(other_boundary, -1, axis=0)
        other_newell = np.asarray(
            [
                np.sum((other_boundary[:, 1] - other_following[:, 1]) * (other_boundary[:, 2] + other_following[:, 2])),
                np.sum((other_boundary[:, 2] - other_following[:, 2]) * (other_boundary[:, 0] + other_following[:, 0])),
                np.sum((other_boundary[:, 0] - other_following[:, 0]) * (other_boundary[:, 1] + other_following[:, 1])),
            ],
            dtype=np.float64,
        )
        added, _other_record = triangulate_boundary_cap_without_center(
            vertices,
            faces,
            [int(index) for index in other_loop],
            other_newell,
            occupied_edges=face_edges_among_vertices(
                source_faces,
                {int(index) for index in other_loop},
                len(vertices),
            ),
        )
        if int(added) != len(other_loop) - 2:
            raise ValueError(f"failed to close non-target loop {other_index}")
    if abs(float(args.overcut_mm) - 0.02) > 1e-12:
        from split3mf.inward import build_local_male_attachment_cutter

        cutters[0] = build_local_male_attachment_cutter(
            boundary_points=boundary,
            inward=inward,
            spec=spec,
            outside_extension_mm=4.0,
            boundary_overcut_mm=float(args.overcut_mm),
        )
    parent = common.trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    orient_mesh_faces_consistently(parent)
    result, boolean_record = subtract_socket_cutters(parent, cutters)
    print(
        {
            "loop_vertices": int(len(loop)),
            "overcut_mm": float(args.overcut_mm),
            "parent_watertight": bool(parent.is_watertight),
            "result_watertight": bool(result.is_watertight),
            "result_winding_consistent": bool(result.is_winding_consistent),
            "faces": int(len(result.faces)),
            "record": record,
            "boolean": boolean_record,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
