from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from split3mf import common  # noqa: E402

common.load_core_dependencies()

from split3mf.part_geometry import (  # noqa: E402
    add_local_female_boolean_closure,
    add_local_male_connector_and_backing,
    local_connector_spec_for_interface,
)
from split3mf.local_connectors import subtract_socket_cutters  # noqa: E402
from split3mf.package_io import export_colored_parts_3mf  # noqa: E402


def build_child(samples: int, spec):
    angles = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    rings: list[np.ndarray] = []
    for level in range(7):
        theta = (math.pi / 2.0) * (7 - level) / 7
        radius = 8.0 * math.sin(theta)
        radius *= 1.0 + 0.035 * np.sin(3.0 * angles + 0.3)
        rings.append(
            np.column_stack(
                (radius * np.cos(angles), radius * np.sin(angles), 7.0 * np.cos(theta) * np.ones(samples))
            )
        )
    vertices = [point.copy() for ring in rings for point in ring]
    top_id = len(vertices)
    vertices.append(np.asarray([0.0, 0.0, 7.1]))
    faces: list[list[int]] = []
    for level in range(len(rings) - 1):
        first = level * samples
        second = (level + 1) * samples
        for index in range(samples):
            nxt = (index + 1) % samples
            faces.extend(
                ([first + index, first + nxt, second + nxt], [first + index, second + nxt, second + index])
            )
    last = (len(rings) - 1) * samples
    for index in range(samples):
        faces.append([last + index, last + (index + 1) % samples, top_id])
    record = add_local_male_connector_and_backing(
        output_vertices=vertices,
        output_faces=faces,
        boundary_ids=list(range(samples)),
        boundary_points=rings[0],
        inward=np.asarray([0.0, 0.0, -1.0]),
        spec=spec,
    )
    mesh = common.trimesh.Trimesh(
        vertices=np.asarray(vertices), faces=np.asarray(faces), process=True
    )
    common.trimesh.repair.fix_winding(mesh)
    common.trimesh.repair.fix_normals(mesh)
    return mesh, record


def build_parent(samples: int, spec):
    angles = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    top = np.column_stack((10.0 * np.cos(angles), 10.0 * np.sin(angles), np.zeros(samples)))
    bottom = top.copy()
    bottom[:, 2] = -7.0
    vertices = [point.copy() for point in top] + [point.copy() for point in bottom]
    bottom_center = len(vertices)
    vertices.append(np.asarray([0.0, 0.0, -7.0]))
    faces: list[list[int]] = []
    for index in range(samples):
        nxt = (index + 1) % samples
        faces.extend(
            ([index, samples + nxt, nxt], [index, samples + index, samples + nxt])
        )
        faces.append([samples + index, bottom_center, samples + nxt])
    record, cutter = add_local_female_boolean_closure(
        output_vertices=vertices,
        output_faces=faces,
        boundary_ids=list(range(samples)),
        boundary_points=top,
        inward=np.asarray([0.0, 0.0, -1.0]),
        spec=spec,
    )
    closed = common.trimesh.Trimesh(
        vertices=np.asarray(vertices), faces=np.asarray(faces), process=True
    )
    common.trimesh.repair.fix_winding(closed)
    common.trimesh.repair.fix_normals(closed)
    socketed, boolean_record = subtract_socket_cutters(closed, [cutter])
    return socketed, record, boolean_record


def rotation_matrix() -> np.ndarray:
    yaw = math.radians(-26.0)
    pitch = math.radians(62.0)
    rz = np.asarray(
        [[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]]
    )
    rx = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, math.cos(pitch), -math.sin(pitch)], [0.0, math.sin(pitch), math.cos(pitch)]]
    )
    return rx @ rz


def shade(rgb: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    factor = 0.48 + 0.52 * max(0.0, min(1.0, amount))
    return tuple(int(max(0, min(255, channel * factor))) for channel in rgb)


def draw_meshes(meshes, output_path: Path) -> None:
    width, height = 1600, 920
    image = Image.new("RGB", (width, height), (241, 243, 246))
    draw = ImageDraw.Draw(image)
    matrix = rotation_matrix()
    transformed = []
    all_xy = []
    for mesh, color, offset in meshes:
        vertices = (mesh.vertices + np.asarray(offset)[None, :]) @ matrix.T
        transformed.append((mesh, vertices, color))
        all_xy.append(vertices[:, :2])
    xy = np.concatenate(all_xy, axis=0)
    scale = min(820.0 / max(np.ptp(xy[:, 0]), 1e-9), 700.0 / max(np.ptp(xy[:, 1]), 1e-9))
    center = (xy.min(axis=0) + xy.max(axis=0)) * 0.5
    origin = np.asarray([520.0, 500.0])
    triangles = []
    light = np.asarray([-0.25, -0.45, 0.86])
    light /= np.linalg.norm(light)
    for mesh, vertices, color in transformed:
        projected = (vertices[:, :2] - center[None, :]) * scale
        projected[:, 1] *= -1.0
        projected += origin[None, :]
        for face in mesh.faces:
            points3 = vertices[np.asarray(face)]
            normal = np.cross(points3[1] - points3[0], points3[2] - points3[0])
            length = np.linalg.norm(normal)
            if length <= 1e-12:
                continue
            normal /= length
            brightness = 0.28 + 0.72 * abs(float(normal @ light))
            triangles.append(
                (float(points3[:, 2].mean()), projected[np.asarray(face)], shade(color, brightness))
            )
    for _depth, points, color in sorted(triangles, key=lambda item: item[0]):
        polygon = [(float(x), float(y)) for x, y in points]
        draw.polygon(polygon, fill=color)

    font = ImageFont.load_default(size=24)
    small = ImageFont.load_default(size=19)
    draw.text((52, 34), "3 mm printable backing + 45 degree taper + boolean female socket", fill=(25, 30, 40), font=font)
    draw.text((105, 790), "exploded child: orange male peg", fill=(130, 62, 4), font=small)
    draw.text((510, 836), "parent: real boolean cavity (not a pyramid)", fill=(21, 93, 65), font=small)

    # Exact axial section diagram, derived from the same production dimensions.
    x0, y0 = 1080, 160
    sx, sy = 46.0, 82.0
    outer = 8.0
    inner = 5.0
    peg = 2.0
    z_surface, z_back, z_tip = 0.0, -3.0, -4.75
    def point(x, z):
        return (x0 + x * sx, y0 - z * sy)
    section = [point(-outer, z_surface), point(-inner, z_back), point(-peg, z_back), point(-peg, z_tip), point(peg, z_tip), point(peg, z_back), point(inner, z_back), point(outer, z_surface)]
    draw.polygon(section, fill=(238, 128, 35), outline=(117, 55, 5), width=3)
    draw.line([point(-outer, z_surface), point(-inner, z_back)], fill=(218, 35, 35), width=6)
    draw.line([point(outer, z_surface), point(inner, z_back)], fill=(218, 35, 35), width=6)
    draw.line([point(-9.2, z_surface), point(-9.2, z_back)], fill=(30, 45, 60), width=2)
    draw.line([point(-9.5, z_surface), point(-8.9, z_surface)], fill=(30, 45, 60), width=2)
    draw.line([point(-9.5, z_back), point(-8.9, z_back)], fill=(30, 45, 60), width=2)
    draw.text((1180, 505), "3.00 mm", fill=(30, 45, 60), font=small)
    draw.text((1160, 250), "45 deg", fill=(205, 35, 35), font=small)
    draw.text((1055, 660), "outer large -> inner small", fill=(30, 45, 60), font=small)
    draw.text((1080, 735), "peg engagement: 1.75 mm", fill=(30, 45, 60), font=small)
    draw.text((1080, 775), "total parent depth: 5.00 mm", fill=(30, 45, 60), font=small)
    image.save(output_path)


def main() -> None:
    output_dir = ROOT.parent / "local_connector_boolean_backing_demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = local_connector_spec_for_interface(
        fit_clearance_mm=0.50,
        bottom_clearance_mm=0.25,
        lead_in_mm=0.80,
        safe_engagement_depth_mm=5.0,
    )
    child, child_record = build_child(64, spec)
    parent, parent_record, boolean_record = build_parent(64, spec)
    if not (child.is_watertight and parent.is_watertight):
        raise RuntimeError("demo parts must both be watertight")
    exploded_child = child.copy()
    exploded_child.apply_translation([-7.0, 0.0, 8.0])
    path_3mf = output_dir / "local_connector_3mm_45deg_boolean_demo.3mf"
    export_colored_parts_3mf(
        path_3mf,
        [
            {
                "part_id": "demo_child_male",
                "mesh": exploded_child,
                "color_hex": "#F58220FF",
                "color_code": "orange",
                "color_name": "orange",
                "annotation": child_record,
            },
            {
                "part_id": "demo_parent_female",
                "mesh": parent,
                "color_hex": "#2AA876FF",
                "color_code": "green",
                "color_name": "green",
                "annotation": {**parent_record, **boolean_record},
            },
        ],
        title="3 mm 45 degree local connector boolean demo",
        output_layout="separate-items",
    )
    preview = output_dir / "local_connector_3mm_45deg_boolean_demo_preview.png"
    draw_meshes(
        [(child, (245, 130, 32), (-7.0, 0.0, 8.0)), (parent, (42, 168, 118), (0.0, 0.0, 0.0))],
        preview,
    )
    print(path_3mf)
    print(preview)
    print({"child": child_record, "parent": parent_record, "boolean": boolean_record})


if __name__ == "__main__":
    main()
