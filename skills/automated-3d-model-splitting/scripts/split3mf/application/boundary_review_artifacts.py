from __future__ import annotations

from pathlib import Path

import numpy as np
from ..boundary_preview import _even_sample
from ..boundary_matching import matching_seam

MAX_RECOGNITION_PREVIEW_FACES = 750_000


def write_boundary_review_artifacts(
    directory: Path,
    boundaries,
    recognition_records: list[dict],
    region_review: dict,
    components: list,
    recognition_review_path: Path | None = None,
    *,
    source_vertices: np.ndarray | None = None,
    source_faces: np.ndarray | None = None,
    source_face_color_tokens: np.ndarray | None = None,
    source_color_map: dict[str, str] | None = None,
) -> dict[str, str | int]:
    """Write a color-coded boundary preview and a compact human review report."""
    recognition_records = list(recognition_records)
    semantic_labels = _required_semantic_labels(recognition_records)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    image_path = directory / "03_recognition_boundaries.png"
    opposite_image_path = directory / "03_recognition_boundaries_opposite.png"
    side_image_path = directory / "03_recognition_boundaries_side.png"
    report_path = directory / "03_recognition_boundary_review.md"
    records = list(boundaries.simplification_records)
    plotted_boundary_count = _write_boundary_plot(
        image_path,
        boundaries,
        source_vertices=source_vertices,
        source_faces=source_faces,
        source_face_color_tokens=source_face_color_tokens,
        source_color_map=source_color_map,
        components=components,
        recognition_records=recognition_records,
        azimuth=-90,
        view_label="current view",
    )
    _write_boundary_plot(
        opposite_image_path,
        boundaries,
        source_vertices=source_vertices,
        source_faces=source_faces,
        source_face_color_tokens=source_face_color_tokens,
        source_color_map=source_color_map,
        components=components,
        recognition_records=recognition_records,
        azimuth=90,
        view_label="opposite view",
    )
    _write_boundary_plot(
        side_image_path,
        boundaries,
        source_vertices=source_vertices,
        source_faces=source_faces,
        source_face_color_tokens=source_face_color_tokens,
        source_color_map=source_color_map,
        components=components,
        recognition_records=recognition_records,
        azimuth=0,
        view_label="side view",
    )
    _write_summary(
        report_path, records, recognition_records, region_review, components,
        recognition_review_path, semantic_labels,
    )
    return {
        "boundary_image": str(image_path),
        "boundary_image_opposite": str(opposite_image_path),
        "boundary_image_side": str(side_image_path),
        "boundary_review_report": str(report_path),
        "boundary_count": sum(
            len(loops) for loops in boundaries.component_loops
        ),
        "component_count": len(boundaries.component_loops),
        "plotted_boundary_count": plotted_boundary_count,
        "simplification_records": len(records),
    }


def _write_boundary_plot(
    path: Path,
    boundaries,
    *,
    source_vertices: np.ndarray | None = None,
    source_faces: np.ndarray | None = None,
    source_face_color_tokens: np.ndarray | None = None,
    source_color_map: dict[str, str] | None = None,
    components: list | None = None,
    recognition_records: list[dict] | None = None,
    azimuth: float = -90,
    view_label: str = "current view",
) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    figure = plt.figure(figsize=(13, 10))
    figure.patch.set_facecolor("#20242A")
    axis = figure.add_subplot(111, projection="3d")
    axis.computed_zorder = False
    axis.set_facecolor("#20242A")
    for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
        pane.set_facecolor((0.12, 0.14, 0.16, 1.0))
        pane.set_edgecolor("#697078")
    axis.tick_params(colors="#E6E8EB")
    all_points = []
    render_source_mesh = all(
        value is not None
        for value in (
            source_vertices, source_faces, source_face_color_tokens, source_color_map
        )
    )
    if render_source_mesh:
        vertices = np.asarray(source_vertices, dtype=np.float64)
        faces = np.asarray(source_faces, dtype=np.int64)
        tokens = np.asarray(source_face_color_tokens).astype(str)
        if len(tokens) != len(faces):
            raise ValueError("source face color tokens must align with source faces")
        face_ids = _sample_component_faces(components or [], len(faces))
        triangles = vertices[faces[face_ids]]
        rgb = np.asarray([
            _hex_to_rgb(source_color_map.get(tokens[index], "#A8A8AC"))
            for index in face_ids
        ], dtype=np.float64)
        normals = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        lengths = np.linalg.norm(normals, axis=1)
        valid = lengths > 1e-12
        normals[valid] /= lengths[valid, None]
        light = np.asarray([-0.35, -0.45, 0.82], dtype=np.float64)
        light /= np.linalg.norm(light)
        intensity = 0.82 + 0.18 * np.abs(normals @ light)
        rgb = np.clip(rgb * intensity[:, None], 0.0, 1.0)
        axis.add_collection3d(Poly3DCollection(
            triangles, facecolors=rgb, edgecolors="none", linewidths=0, zorder=1
        ))
        all_points.append(vertices)
    records_by_part = {
        int(record["part_index"]): record
        for record in (recognition_records or [])
    }
    visible_loops, seam_colors = _boundary_display_groups(boundaries.component_loop_points)
    plotted_boundary_count = 0
    for component_index, component_loops in visible_loops.items():
        for points, color in component_loops:
            halo = "#101318"
            closed = np.vstack((points, points[0]))
            segments = np.stack((closed[:-1], closed[1:]), axis=1)
            axis.add_collection3d(Line3DCollection(
                segments, colors=halo, linewidths=6.0, alpha=1.0, zorder=8
            ))
            axis.add_collection3d(Line3DCollection(
                segments, colors=color, linewidths=4.0, alpha=1.0, zorder=10
            ))
            all_points.append(points)
            plotted_boundary_count += 1
    if all_points:
        points = np.vstack(all_points)
        center = (points.min(axis=0) + points.max(axis=0)) * 0.5
        extent = np.maximum(points.max(axis=0) - points.min(axis=0), 1e-6)
        margin = max(float(extent.max()) * 0.04, 0.5)
        axis.set_xlim(center[0] - extent[0] / 2 - margin, center[0] + extent[0] / 2 + margin)
        axis.set_ylim(center[1] - extent[1] / 2 - margin, center[1] + extent[1] / 2 + margin)
        axis.set_zlim(center[2] - extent[2] / 2 - margin, center[2] + extent[2] / 2 + margin)
        axis.set_box_aspect(extent)
    axis.set_title(
        f"Original painted model with simplified and smoothed boundaries ({view_label})"
        if render_source_mesh
        else f"Simplified and smoothed recognized boundaries ({view_label})"
    )
    axis.set_xlabel("X (mm)")
    axis.set_ylabel("Y (mm)")
    axis.set_zlabel("Z (mm)")
    axis.xaxis.label.set_color("#E6E8EB")
    axis.yaxis.label.set_color("#E6E8EB")
    axis.zaxis.label.set_color("#E6E8EB")
    axis.view_init(elev=20, azim=float(azimuth))
    axis.title.set_color("#FFFFFF")
    figure.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
    _draw_boundary_legend(
        figure, len(boundaries.component_loop_points), records_by_part,
        source_color_map, seam_colors,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return plotted_boundary_count


def _component_paint_color(
    component_index: int,
    records_by_part: dict[int, dict],
    source_color_map: dict[str, str] | None,
) -> str:
    record = records_by_part.get(component_index, {})
    source_code = str(record.get("color_code", ""))
    return str(
        record.get("color_hex")
        or (source_color_map or {}).get(source_code)
        or "#A8A8AC"
    )


BOUNDARY_COLORS = (
    "#FF5C5C", "#2DE2E6", "#FFD166", "#9B7BFF", "#59E071",
    "#FF8F3D", "#F77BD3", "#6CB6FF", "#D8F05A", "#B57BFF",
    "#20C997", "#FF6B8B", "#A0C4FF", "#E43CFF", "#B8F28B",
    "#F4A7D5", "#80CBC4", "#FFA85C", "#C3A6FF", "#E8E85A",
)


def _seam_color(seam_index: int) -> str:
    if seam_index < len(BOUNDARY_COLORS):
        return BOUNDARY_COLORS[seam_index]
    import colorsys

    hue = ((seam_index - len(BOUNDARY_COLORS)) * 0.618033988749895 + 0.13) % 1.0
    rgb = colorsys.hsv_to_rgb(hue, 0.72, 1.0)
    return "#" + "".join(f"{round(channel * 255):02X}" for channel in rgb)


def _boundary_display_groups(component_loop_points):
    """Give a shared seam one visible color and list that color for both owners."""
    selected: list[np.ndarray] = []
    visible: dict[int, list[tuple[np.ndarray, str]]] = {}
    colors_by_part: dict[int, list[str]] = {}
    for component_index, loops in enumerate(component_loop_points, start=1):
        for loop in loops:
            if len(loop) < 3:
                continue
            points = np.asarray(loop, dtype=np.float64)
            seam_index = next(
                (index for index, earlier in enumerate(selected)
                 if matching_seam(points, earlier)), None
            )
            if seam_index is None:
                seam_index = len(selected)
                selected.append(points)
                color = _seam_color(seam_index)
                visible.setdefault(component_index, []).append((points, color))
            color = _seam_color(seam_index)
            owner_colors = colors_by_part.setdefault(component_index, [])
            if color not in owner_colors:
                owner_colors.append(color)
    return visible, colors_by_part


def _draw_boundary_legend(figure, part_count, records_by_part, source_color_map, seam_colors):
    """Show paint as a swatch and the actual boundary stroke in the final column."""
    from matplotlib.patches import Rectangle

    legend = figure.add_axes((0.82, 0.29, 0.17, 0.62))
    legend.set_facecolor("#30363D")
    legend.set_xlim(0, 1)
    legend.set_ylim(0, part_count + 1.5)
    legend.set_xticks([])
    legend.set_yticks([])
    for spine in legend.spines.values():
        spine.set_color("#697078")
    legend.text(0.06, part_count + 0.65, "Part", color="white", fontsize=9)
    legend.text(0.36, part_count + 0.65, "Paint", color="white", fontsize=9)
    legend.text(0.66, part_count + 0.65, "Boundary", color="white", fontsize=9)
    for index in range(1, part_count + 1):
        y = part_count + 0.25 - index
        legend.text(0.06, y, f"P{index:02d}", color="white", fontsize=9, va="center")
        legend.add_patch(Rectangle(
            (0.39, y - 0.17), 0.17, 0.34,
            facecolor=_component_paint_color(index, records_by_part, source_color_map),
            edgecolor="#AAAAAA", linewidth=0.5,
        ))
        colors = seam_colors.get(index, [])
        for color_index, color in enumerate(colors):
            x0 = 0.66 + 0.28 * color_index / max(len(colors), 1)
            x1 = x0 + 0.28 / max(len(colors), 1)
            legend.plot((x0, x1), (y, y), color="#101318", linewidth=6, solid_capstyle="round")
            legend.plot((x0, x1), (y, y), color=color, linewidth=3, solid_capstyle="round")


def _sample_component_faces(components: list, face_count: int) -> np.ndarray:
    """Systematically sample source triangles while representing each region."""
    component_faces = [
        np.asarray(component.global_faces, dtype=np.int64)
        for component in components
        if len(component.global_faces)
    ]
    if not component_faces:
        return _even_sample(np.arange(face_count, dtype=np.int64), MAX_RECOGNITION_PREVIEW_FACES)
    total_component_faces = sum(len(values) for values in component_faces)
    target = min(MAX_RECOGNITION_PREVIEW_FACES, face_count)
    selected = []
    selected_count = 0
    for values in component_faces:
        quota = max(1, int(round(target * len(values) / max(total_component_faces, 1))))
        quota = min(quota, target - selected_count)
        if quota <= 0:
            break
        chosen = _even_sample(values, quota)
        selected.append(chosen)
        selected_count += len(chosen)
    selected_ids = np.unique(np.concatenate(selected)) if selected else np.empty(0, dtype=np.int64)
    if len(selected_ids) < target:
        unselected = np.ones(face_count, dtype=bool)
        unselected[selected_ids] = False
        selected_ids = np.sort(np.r_[
            selected_ids,
            _even_sample(np.flatnonzero(unselected), target - len(selected_ids)),
        ])
    return selected_ids


def _hex_to_rgb(color_hex: str) -> tuple[float, float, float]:
    token = str(color_hex).strip().lstrip("#")
    try:
        return tuple(int(token[index:index + 2], 16) / 255.0 for index in (0, 2, 4))
    except (ValueError, IndexError):
        return (0.66, 0.66, 0.66)


def _write_summary(
    path: Path,
    simplification_records: list[dict],
    recognition_records: list[dict],
    region_review: dict,
    components: list,
    recognition_review_path: Path | None,
    semantic_labels: dict[int, str],
) -> None:
    recognition_by_index = {
        int(record["part_index"]): record for record in recognition_records
    }
    index_by_min_face = {
        int(np.min(component.global_faces)): index
        for index, component in enumerate(components, start=1)
        if len(component.global_faces)
    }
    classification_by_index = {}
    for record in region_review.get("classifications", []):
        try:
            component_index = index_by_min_face[int(record["source_min_face_index"])]
            classification_by_index[component_index] = record
        except (KeyError, TypeError, ValueError):
            continue
    lines = [
        "# 识别边界人工判断摘要",
        "",
        f"## 区域判断（{len(recognition_records)} 个有效零件）",
        "",
        "共用简化边界只绘制一次；相接零件共用该边界线色。图例分别显示原模型涂色和边界线色。",
        "",
        "| 区域 | 区域语义 / 颜色 | 面数 / 面积 mm² | 判断建议 |",
        "|---|---|---:|---|",
    ]
    for component_index, part in sorted(recognition_by_index.items()):
        review = classification_by_index.get(component_index, {})
        classification = str(
            review.get("classification") or part.get("source_region_classification") or ""
        )
        if classification == "noise":
            suggestion = "保留源面（复核标为噪声）"
        elif classification == "part":
            suggestion = "建议保留（复核为独立部件）"
        elif classification == "uncertain":
            suggestion = "需人工判断；检查是否应与相邻区域合并"
        else:
            suggestion = "检查外观与语义；不独立时可考虑合并"
        semantic = semantic_labels[component_index]
        semantic_confidence = str(part.get("visual_semantic_confidence") or "").upper()
        confidence_label = {
            "HIGH": "高置信度",
            "MED": "中置信度",
            "MEDIUM": "中置信度",
            "LOW": "低置信度",
            "VERYLOW": "极低置信度",
        }.get(semantic_confidence)
        if confidence_label:
            semantic = f"{semantic}（{confidence_label}）"
        color = str(part.get("color_hex") or part.get("color_code") or "未知")
        color_name = _chinese_color_name(color)
        lines.append(
            f"| P{component_index:02d} | {semantic} · {color_name}（{color}） | "
            f"{int(part.get('faces', 0))} / {float(part.get('area_mm2', 0.0)):.2f} | "
            f"{suggestion} |"
        )
    lines.extend([
        "",
        "## 识别复核",
        "",
        "请检查上表和边界预览：若有识别错误，可在复核 JSON 的 `actions` 中填写 `delete` 或 `merge`。",
        "应用修改后会重新生成报告；再次确认无误后，将该 JSON 的 `user_confirmed` 设为 `true`，流程才会进入装配。",
    ])
    if recognition_review_path is not None:
        lines.append(f"复核文件：`{recognition_review_path}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _semantic_label(part: dict) -> str:
    """Return a normalized non-empty visual label for one recognized region."""
    for key in (
        "region_review_label",
        "visual_semantic_label",
        "label",
        "semantic_label",
    ):
        value = str(part.get(key) or "").strip()
        if value:
            return value
    return ""


def _required_semantic_labels(recognition_records: list[dict]) -> dict[int, str]:
    labels = {
        int(part["part_index"]): _semantic_label(part) or f"待确认区域 P{int(part['part_index']):02d}"
        for part in recognition_records
    }
    return labels


def _chinese_color_name(color_hex: str) -> str:
    """Map common exact colors or the nearest basic hue to a Chinese name."""
    try:
        red, green, blue = (int(color_hex[index:index + 2], 16) for index in (1, 3, 5))
    except (ValueError, IndexError):
        return "未知色"
    if max(red, green, blue) - min(red, green, blue) < 28:
        return "白色" if min(red, green, blue) > 220 else "灰色"
    hue = __import__("colorsys").rgb_to_hsv(red / 255, green / 255, blue / 255)[0] * 360
    if hue < 15 or hue >= 345:
        return "红色"
    if hue < 45:
        value = max(red, green, blue) / 255
        return "棕色" if value < 0.4 else "橙色"
    if hue < 70:
        return "黄色"
    if hue < 165:
        return "绿色"
    if hue < 195:
        return "青色"
    if hue < 255:
        return "蓝色"
    if hue < 300:
        return "紫色"
    return "粉色"
