"""Plot an actual representative 45-degree backing section from a split 3MF."""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "split3mf-matplotlib"),
)
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import numpy as np
import trimesh


CORE_NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
ANNOTATION_KEY = "automated-3d-model-splitting:annotation"


def load_part(path: Path, prefix: str) -> tuple[np.ndarray, np.ndarray, dict]:
    namespace = {"m": CORE_NS}
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("3D/3dmodel.model"))
    part = next(
        item
        for item in root.findall(".//m:object", namespace)
        if (item.get("name") or "").startswith(prefix)
    )
    vertices = np.asarray(
        [
            [float(vertex.get(axis)) for axis in ("x", "y", "z")]
            for vertex in part.findall(".//m:vertex", namespace)
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [int(face.get(axis)) for axis in ("v1", "v2", "v3")]
            for face in part.findall(".//m:triangle", namespace)
        ],
        dtype=np.int64,
    )
    metadata = {
        item.get("name"): item.text or ""
        for item in part.findall("m:metadata", namespace)
    }
    return vertices, faces, json.loads(metadata[ANNOTATION_KEY])


def source_face_count(annotation: dict) -> int:
    return sum(int(item["faces"]) for item in annotation["raw_color_tokens"])


def stable_inward_axis(vertices: np.ndarray, source_faces: np.ndarray) -> np.ndarray:
    triangles = vertices[source_faces]
    inward = -np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    ).sum(axis=0)
    return inward / np.linalg.norm(inward)


def planar_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray([0.0, 0.0, 1.0])
    if abs(float(reference @ axis)) > 0.9:
        reference = np.asarray([0.0, 1.0, 0.0])
    u = np.cross(axis, reference)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    return u, v / np.linalg.norm(v)


def boundary_edges(faces: np.ndarray) -> list[tuple[int, int]]:
    owners = collections.Counter(
        tuple(sorted((int(left), int(right))))
        for face in faces
        for left, right in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        )
    )
    return [edge for edge, count in owners.items() if count == 1]


def nearest_source_point(
    point: np.ndarray,
    ring: np.ndarray,
    source_axial: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    following = np.roll(ring, -1, axis=0)
    edges = following - ring
    length_squared = np.einsum("ij,ij->i", edges, edges)
    relative = point[None, :] - ring
    blend = np.divide(
        np.einsum("ij,ij->i", relative, edges),
        length_squared,
        out=np.zeros(len(ring)),
        where=length_squared > 1e-24,
    )
    blend = np.clip(blend, 0.0, 1.0)
    closest = ring + blend[:, None] * edges
    distances = np.linalg.norm(point[None, :] - closest, axis=1)
    index = int(np.argmin(distances))
    source_point = closest[index]
    matched_axial = (
        source_axial[index] * (1.0 - blend[index])
        + source_axial[(index + 1) % len(source_axial)] * blend[index]
    )
    return source_point, float(distances[index]), float(matched_axial)


def representative_profile(
    vertices: np.ndarray,
    faces: np.ndarray,
    source_count: int,
    loop: np.ndarray,
    inward: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    origin = vertices[faces[:source_count].reshape(-1)].mean(axis=0)
    axial = (vertices - origin[None, :]) @ inward
    first_generated_vertex = int(faces[:source_count].max()) + 1
    quantized = np.round(axial[first_generated_vertex:], 4)
    floor_axial = float(collections.Counter(quantized).most_common(1)[0][0])

    generated_faces = faces[source_count:]
    floor_face_mask = np.all(
        np.abs(axial[generated_faces] - floor_axial) < 8e-5,
        axis=1,
    )
    floor_faces = generated_faces[floor_face_mask]
    floor_ids = sorted(
        {vertex for edge in boundary_edges(floor_faces) for vertex in edge}
    )
    if not floor_ids:
        # Regular heightfield refinement does not guarantee a triangle whose
        # three vertices lie on the same axial slice.  The slice vertices are
        # still present, so recover the representative ring directly instead
        # of relying on the old coplanar-face topology.
        generated_ids = np.unique(generated_faces.reshape(-1))
        floor_ids = [
            int(vertex_id)
            for vertex_id in generated_ids
            if abs(float(axial[int(vertex_id)]) - floor_axial) < 8e-5
        ]
    if not floor_ids:
        raise ValueError("could not recover a generated backing-profile ring")

    u, v = planar_basis(inward)
    source_3d = vertices[loop]
    source_2d = np.column_stack(
        ((source_3d - origin) @ u, (source_3d - origin) @ v)
    )
    floor_3d = vertices[np.asarray(floor_ids, dtype=np.int64)]
    floor_2d = np.column_stack(
        ((floor_3d - origin) @ u, (floor_3d - origin) @ v)
    )
    source_axial = (source_3d - origin) @ inward

    candidates = []
    for floor_index, floor_point_2d in enumerate(floor_2d):
        source_point_2d, lateral, matched_source_axial = nearest_source_point(
            floor_point_2d,
            source_2d,
            source_axial,
        )
        depth = abs(floor_axial - matched_source_axial)
        angle = float(np.degrees(np.arctan2(depth, lateral)))
        candidates.append(
            (abs(angle - 45.0), angle, floor_index, source_point_2d, lateral, depth)
        )
    candidates.sort(key=lambda item: item[0])
    _, angle, floor_index, source_point_2d, lateral, depth = candidates[0]
    floor_point = floor_3d[floor_index]
    source_point = (
        origin
        + source_point_2d[0] * u
        + source_point_2d[1] * v
        + (floor_axial - depth) * inward
    )
    all_angles = np.asarray([item[1] for item in candidates])
    record = {
        "angle": angle,
        "lateral": lateral,
        "depth": depth,
        "minimum_angle": float(all_angles.min()),
        "median_angle": float(np.median(all_angles)),
        "maximum_angle": float(all_angles.max()),
    }
    return origin, source_point, floor_point, record


def plot_section(
    vertices: np.ndarray,
    faces: np.ndarray,
    source_count: int,
    source_point: np.ndarray,
    floor_point: np.ndarray,
    inward: np.ndarray,
    record: dict,
    output: Path,
) -> None:
    delta = floor_point - source_point
    lateral_vector = delta - inward * float(delta @ inward)
    lateral_vector /= np.linalg.norm(lateral_vector)
    cut_normal = np.cross(lateral_vector, inward)
    cut_normal /= np.linalg.norm(cut_normal)
    segments, face_ids = trimesh.intersections.mesh_plane(
        trimesh.Trimesh(vertices=vertices, faces=faces, process=False),
        plane_normal=cut_normal,
        plane_origin=source_point,
        return_faces=True,
    )
    section = np.stack(
        [
            np.column_stack(
                (
                    (segment - source_point) @ lateral_vector,
                    (segment - source_point) @ inward,
                )
            )
            for segment in segments
        ]
    )

    figure, axis = plt.subplots(figsize=(10.4, 7.4))
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    axis.grid(True, color="#D1D5DB", linewidth=0.7, alpha=0.7)

    source_label_used = False
    generated_label_used = False
    for segment, face_id in zip(section, face_ids):
        generated = int(face_id) >= source_count
        axis.plot(
            segment[:, 0],
            segment[:, 1],
            color="#EA580C" if generated else "#2563EB",
            linewidth=1.8 if generated else 1.2,
            alpha=0.95,
            label=(
                "Generated backing surface"
                if generated and not generated_label_used
                else "Visible source surface"
                if not generated and not source_label_used
                else None
            ),
        )
        generated_label_used |= generated
        source_label_used |= not generated

    lateral = float(record["lateral"])
    depth = float(record["depth"])
    axis.plot(
        [0.0, lateral],
        [0.0, depth],
        color="#111827",
        linestyle="--",
        linewidth=2.0,
        label="45-degree design line",
        zorder=5,
    )
    axis.plot([0.0, lateral], [0.0, 0.0], color="#6B7280", linestyle=":")
    axis.plot([lateral, lateral], [0.0, depth], color="#6B7280", linestyle=":")
    axis.scatter([0.0, lateral], [0.0, depth], s=42, color="#111827", zorder=6)

    arc_radius = min(lateral, depth) * 0.42
    axis.add_patch(
        Arc(
            (0.0, 0.0),
            2.0 * arc_radius,
            2.0 * arc_radius,
            theta1=0.0,
            theta2=float(record["angle"]),
            color="#111827",
            linewidth=1.3,
        )
    )
    axis.text(
        arc_radius * 0.78,
        arc_radius * 0.30,
        f"{record['angle']:.3f}°",
        fontsize=12,
        color="#111827",
    )
    axis.text(lateral * 0.5, -0.10, f"lateral inset  {lateral:.3f} mm", ha="center")
    axis.text(
        lateral + 0.10,
        depth * 0.5,
        f"inward depth  {depth:.3f} mm",
        va="center",
        rotation=-90,
    )

    padding = 0.65
    axis.set_xlim(-padding, lateral + padding)
    axis.set_ylim(-padding, depth + padding)
    axis.invert_yaxis()
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Lateral inset from visible boundary (mm)")
    axis.set_ylabel("Inward depth (mm)")
    axis.set_title("Actual 45° backing cut — representative section", fontsize=16)
    axis.legend(loc="upper left", frameon=False)
    figure.text(
        0.5,
        0.025,
        (
            f"Measured around floor ring: {record['minimum_angle']:.2f}°–"
            f"{record['maximum_angle']:.2f}°  |  median {record['median_angle']:.2f}°"
        ),
        ha="center",
        color="#374151",
    )
    figure.subplots_adjust(left=0.12, right=0.96, top=0.90, bottom=0.15)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_3mf", type=Path)
    parser.add_argument("output_png", type=Path)
    parser.add_argument("--object-prefix", default="P01_")
    args = parser.parse_args()

    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    from split3mf.mesh import boundary_loops

    vertices, faces, annotation = load_part(args.input_3mf, args.object_prefix)
    count = source_face_count(annotation)
    loops = boundary_loops(faces[:count])
    if len(loops) != 1:
        raise ValueError(f"expected one source boundary loop, found {len(loops)}")
    loop = np.asarray(loops[0], dtype=np.int64)
    inward = stable_inward_axis(vertices, faces[:count])
    _origin, source_point, floor_point, record = representative_profile(
        vertices,
        faces,
        count,
        loop,
        inward,
    )
    plot_section(
        vertices,
        faces,
        count,
        source_point,
        floor_point,
        inward,
        record,
        args.output_png,
    )
    print(json.dumps(record, indent=2))
    print(args.output_png.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
