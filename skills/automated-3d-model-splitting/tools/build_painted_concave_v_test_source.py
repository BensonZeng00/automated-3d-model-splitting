from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tools"))

from split3mf import common  # noqa: E402

common.load_core_dependencies()

from split3mf.mesh import (  # noqa: E402
    triangulate_loop_group_with_bridges,
    triangulate_polygon_ear_clip,
)
from build_painted_boolean_test_source import (  # noqa: E402
    content_types_xml,
    model_xml,
    relationships_xml,
)


def cubic_curve(p0, p1, p2, p3, samples: int) -> np.ndarray:
    parameter = np.linspace(0.0, 1.0, int(samples), endpoint=False)
    one_minus = 1.0 - parameter
    return (
        (one_minus**3)[:, None] * np.asarray(p0)[None, :]
        + (3.0 * one_minus**2 * parameter)[:, None] * np.asarray(p1)[None, :]
        + (3.0 * one_minus * parameter**2)[:, None] * np.asarray(p2)[None, :]
        + (parameter**3)[:, None] * np.asarray(p3)[None, :]
    )


def shield_loop(samples_per_curve: int = 32) -> np.ndarray:
    left_top = (-7.5, 4.0)
    bottom = (0.0, -7.0)
    right_top = (7.5, 4.0)
    left_side = cubic_curve(
        left_top,
        (-5.6, 0.0),
        (-2.4, -7.0),
        bottom,
        samples_per_curve,
    )
    right_side = cubic_curve(
        bottom,
        (2.4, -7.0),
        (5.6, 0.0),
        right_top,
        samples_per_curve,
    )
    concave_top = cubic_curve(
        right_top,
        (3.2, 1.0),
        (-3.2, 1.0),
        left_top,
        samples_per_curve,
    )
    return np.vstack((left_side, right_side, concave_top))


def subdivide(
    vertices: np.ndarray,
    faces: np.ndarray,
    paint: list[str],
    iterations: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    points = [point.copy() for point in np.asarray(vertices, dtype=np.float64)]
    triangles = [list(map(int, face)) for face in np.asarray(faces, dtype=np.int64)]
    tokens = list(paint)
    for _ in range(int(iterations)):
        edge_midpoint: dict[tuple[int, int], int] = {}

        def midpoint(left: int, right: int) -> int:
            edge = tuple(sorted((int(left), int(right))))
            if edge not in edge_midpoint:
                edge_midpoint[edge] = len(points)
                points.append((points[edge[0]] + points[edge[1]]) * 0.5)
            return edge_midpoint[edge]

        next_triangles: list[list[int]] = []
        next_tokens: list[str] = []
        for (a, b, c), token in zip(triangles, tokens, strict=True):
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            next_triangles.extend(
                ([a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca])
            )
            next_tokens.extend((token, token, token, token))
        triangles = next_triangles
        tokens = next_tokens
    return (
        np.asarray(points, dtype=np.float64),
        np.asarray(triangles, dtype=np.int64),
        tokens,
    )


def build_mesh() -> tuple[np.ndarray, np.ndarray, list[str]]:
    outer = np.asarray(
        [(-18.0, -13.0), (18.0, -13.0), (18.0, 13.0), (-18.0, 13.0)],
        dtype=np.float64,
    )
    shield = shield_loop()
    top_z = 10.0
    bottom_z = 0.0
    vertices = [np.asarray([x, y, top_z]) for x, y in outer]
    shield_ids = list(range(len(vertices), len(vertices) + len(shield)))
    vertices.extend(np.asarray([x, y, top_z]) for x, y in shield)
    bottom_ids = list(range(len(vertices), len(vertices) + len(outer)))
    vertices.extend(np.asarray([x, y, bottom_z]) for x, y in outer)
    outer_ids = list(range(len(outer)))

    faces: list[list[int]] = []
    paint: list[str] = []
    top_body = triangulate_loop_group_with_bridges(
        [outer_ids, shield_ids],
        [outer, shield],
        [0, 1],
    )
    if not top_body:
        raise RuntimeError("failed to triangulate the rectangular top around the shield")
    faces.extend([list(face) for face in top_body])
    paint.extend("4" for _ in top_body)

    insert_triangles = triangulate_polygon_ear_clip(shield)
    if not insert_triangles:
        raise RuntimeError("failed to triangulate the concave shield insert")
    for face in insert_triangles:
        mapped = [shield_ids[int(index)] for index in face]
        triangle = np.asarray(vertices)[np.asarray(mapped)]
        if np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])[2] < 0.0:
            mapped = [mapped[0], mapped[2], mapped[1]]
        faces.append(mapped)
        paint.append("8")

    # Bottom and vertical outside walls complete one watertight rectangular box.
    faces.extend(
        (
            [bottom_ids[0], bottom_ids[2], bottom_ids[1]],
            [bottom_ids[0], bottom_ids[3], bottom_ids[2]],
        )
    )
    paint.extend(("4", "4"))
    for index in range(4):
        nxt = (index + 1) % 4
        faces.extend(
            (
                [bottom_ids[index], bottom_ids[nxt], outer_ids[nxt]],
                [bottom_ids[index], outer_ids[nxt], outer_ids[index]],
            )
        )
        paint.extend(("4", "4"))

    mesh_vertices, mesh_faces, mesh_paint = subdivide(
        np.asarray(vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int64),
        paint,
        iterations=2,
    )
    mesh = common.trimesh.Trimesh(
        vertices=mesh_vertices,
        faces=mesh_faces,
        process=False,
    )
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        raise RuntimeError("generated concave-V source must be watertight and consistently wound")
    return mesh_vertices, mesh_faces, mesh_paint


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: build_painted_concave_v_test_source.py OUTPUT.3mf")
    output = Path(sys.argv[1]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    vertices, faces, paint = build_mesh()
    project_settings = {
        "filament_colour": ["#EDC148", "#E86825"],
        "filament_type": ["PLA", "PLA"],
        "filament_vendor": ["Generic", "Generic"],
        "filament_settings_id": ["Generic PLA", "Generic PLA"],
    }
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types_xml())
        archive.writestr("_rels/.rels", relationships_xml())
        archive.writestr("3D/3dmodel.model", model_xml(vertices, faces, paint))
        archive.writestr(
            "Metadata/project_settings.config",
            json.dumps(project_settings, ensure_ascii=False, separators=(",", ":")),
        )
    print(
        json.dumps(
            {
                "output": str(output),
                "vertices": int(len(vertices)),
                "faces": int(len(faces)),
                "yellow_faces": int(paint.count("4")),
                "orange_faces": int(paint.count("8")),
                "bbox_mm": [36.0, 26.0, 10.0],
                "insert_shape": "concave_top_rounded_v",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

