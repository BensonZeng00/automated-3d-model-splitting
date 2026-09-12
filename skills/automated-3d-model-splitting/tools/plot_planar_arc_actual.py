"""Plot the actual source and planar-arc target curves from a split 3MF."""

from __future__ import annotations

import argparse
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
import numpy as np


CORE_NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
ANNOTATION_KEY = "automated-3d-model-splitting:annotation"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_3mf", type=Path)
    parser.add_argument("output_png", type=Path)
    parser.add_argument("--object-prefix", default="P01_")
    return parser


def _load_part(path: Path, object_prefix: str) -> tuple[np.ndarray, np.ndarray, dict]:
    namespace = {"m": CORE_NS}
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("3D/3dmodel.model"))

    objects = root.findall(".//m:object", namespace)
    part = next(
        (item for item in objects if (item.get("name") or "").startswith(object_prefix)),
        None,
    )
    if part is None:
        raise ValueError(f"no object starts with {object_prefix!r}")

    vertices = np.asarray(
        [
            [float(vertex.get(axis)) for axis in ("x", "y", "z")]
            for vertex in part.findall(".//m:vertex", namespace)
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [int(triangle.get(axis)) for axis in ("v1", "v2", "v3")]
            for triangle in part.findall(".//m:triangle", namespace)
        ],
        dtype=np.int64,
    )
    metadata = {
        item.get("name"): item.text or ""
        for item in part.findall("m:metadata", namespace)
    }
    annotation = json.loads(metadata[ANNOTATION_KEY])
    return vertices, faces, annotation


def _source_face_count(annotation: dict) -> int:
    color_groups = annotation.get("raw_color_tokens", [])
    count = sum(int(item.get("faces", 0)) for item in color_groups)
    if count <= 0:
        raise ValueError("annotation does not identify the preserved source faces")
    return count


def _project(points: np.ndarray, origin: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    basis = np.column_stack((u, v))
    return (points - origin[None, :]) @ basis


def _closed(points: np.ndarray) -> np.ndarray:
    return np.vstack((points, points[:1]))


def plot_curves(source: np.ndarray, target: np.ndarray, record: dict, output: Path) -> None:
    displacement = np.linalg.norm(target - source, axis=1)
    maximum_index = int(np.argmax(displacement))
    maximum = float(displacement[maximum_index])

    source_color = "#2563EB"
    target_color = "#EA580C"
    marker_color = "#111827"

    figure, axes = plt.subplots(1, 2, figsize=(13.2, 6.7))
    figure.patch.set_facecolor("white")
    figure.suptitle(
        "planar-arc-retopology: actual interface curves",
        fontsize=17,
        fontweight="semibold",
    )

    for axis in axes:
        axis.set_facecolor("white")
        axis.grid(True, color="#D1D5DB", linewidth=0.7, alpha=0.7)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("Stable-plane U (mm)")
        axis.set_ylabel("Stable-plane V (mm)")
        for spine in axis.spines.values():
            spine.set_color("#9CA3AF")

    full_source = _closed(source)
    full_target = _closed(target)
    axes[0].plot(
        full_source[:, 0],
        full_source[:, 1],
        color=source_color,
        linewidth=1.15,
        label=f"Original boundary ({len(source)} vertices)",
    )
    axes[0].plot(
        full_target[:, 0],
        full_target[:, 1],
        color=target_color,
        linewidth=1.7,
        label=f"Actual retopology target ({len(target)} vertices)",
    )
    axes[0].plot(
        [source[maximum_index, 0], target[maximum_index, 0]],
        [source[maximum_index, 1], target[maximum_index, 1]],
        color=marker_color,
        linewidth=1.2,
        marker="o",
        markersize=3.5,
    )
    axes[0].set_title("Full boundary")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)

    axes[1].plot(
        full_source[:, 0],
        full_source[:, 1],
        color=source_color,
        linewidth=1.6,
    )
    axes[1].plot(
        full_target[:, 0],
        full_target[:, 1],
        color=target_color,
        linewidth=2.2,
    )
    axes[1].annotate(
        "",
        xy=target[maximum_index],
        xytext=source[maximum_index],
        arrowprops={"arrowstyle": "->", "color": marker_color, "lw": 1.4},
    )
    midpoint = (source[maximum_index] + target[maximum_index]) * 0.5
    axes[1].annotate(
        f"max offset  {maximum:.3f} mm",
        xy=midpoint,
        xytext=(-145, 35),
        textcoords="offset points",
        fontsize=11,
        color=marker_color,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#9CA3AF"},
    )
    axes[1].scatter(*source[maximum_index], s=23, color=source_color, zorder=4)
    axes[1].scatter(*target[maximum_index], s=23, color=target_color, zorder=4)
    local_points = np.vstack((source[maximum_index], target[maximum_index]))
    local_center = local_points.mean(axis=0)
    local_span = np.ptp(local_points, axis=0)
    half_width = max(float(local_span.max()) * 1.4, 0.82)
    axes[1].set_xlim(local_center[0] - half_width, local_center[0] + half_width)
    axes[1].set_ylim(local_center[1] - half_width, local_center[1] + half_width)
    axes[1].set_title("Zoom at maximum separation")

    figure.text(
        0.5,
        0.025,
        (
            f"Source: {record['source_vertices']} vertices  |  "
            f"equal-arc guide: {record['target_samples']} points  |  "
            f"smoothing: {record['smooth_passes']} passes  |  "
            f"RMS offset: {record['rms_target_offset_mm']:.3f} mm"
        ),
        ha="center",
        fontsize=10.5,
        color="#374151",
    )

    figure.subplots_adjust(left=0.07, right=0.985, top=0.84, bottom=0.22, wspace=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> int:
    args = _parser().parse_args()
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    from split3mf.mesh import boundary_loops
    from split3mf.planar_arc import build_planar_arc_boundary

    vertices, faces, annotation = _load_part(args.input_3mf, args.object_prefix)
    source_faces = faces[: _source_face_count(annotation)]
    loops = boundary_loops(source_faces)
    if len(loops) != 1:
        raise ValueError(f"expected one preserved source boundary, found {len(loops)}")

    loop = np.asarray(loops[0], dtype=np.int64)
    config = annotation["interface_retopology_records"][0]
    result = build_planar_arc_boundary(
        vertices[loop],
        loop,
        target_samples=int(config["target_samples"]),
        smooth_passes=int(config["smooth_passes"]),
        maximum_target_offset_mm=(
            float(config["retopology_band_mm"])
            * float(config["maximum_band_fraction"])
        ),
    )
    source_2d = _project(
        result.source_points,
        result.plane_origin,
        result.plane_u,
        result.plane_v,
    )
    target_2d = _project(
        result.target_points,
        result.plane_origin,
        result.plane_u,
        result.plane_v,
    )
    plot_curves(source_2d, target_2d, result.record, args.output_png)
    print(json.dumps(result.record, ensure_ascii=False, indent=2))
    print(args.output_png.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
