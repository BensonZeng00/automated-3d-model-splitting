from __future__ import annotations

from pathlib import Path

import numpy as np
from ..boundary_preview import _even_sample

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
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    image_path = directory / "03_recognition_boundaries.png"
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
    )
    _write_summary(
        report_path, records, recognition_records, region_review, components,
        recognition_review_path,
    )
    return {
        "boundary_image": str(image_path),
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
    palette = [
        ("red", "#E31A1C"), ("white", "#FFFFFF"), ("blue", "#1F78B4"),
        ("yellow", "#FFD92F"), ("purple", "#984EA3"), ("orange", "#FF7F00"),
        ("cyan", "#00BFC4"), ("pink", "#F781BF"), ("green", "#33A02C"),
        ("lime", "#B2DF8A"), ("magenta", "#E7298A"), ("teal", "#66C2A5"),
        ("gold", "#E6AB02"),
    ]
    legend_handles = []
    plotted_boundary_count = 0
    for component_index, loops in enumerate(boundaries.component_loop_points, start=1):
        color_name, color = palette[(component_index - 1) % len(palette)]
        component_points = [np.asarray(loop, dtype=np.float64) for loop in loops if len(loop) >= 3]
        if not component_points:
            continue
        plotted_for_component = 0
        for loop in loops:
            if len(loop) < 3:
                continue
            points = np.asarray(loop, dtype=np.float64)
            closed = np.vstack((points, points[0]))
            segments = np.stack((closed[:-1], closed[1:]), axis=1)
            axis.add_collection3d(Line3DCollection(
                segments, colors="#20242A", linewidths=4.2, alpha=1.0, zorder=8
            ))
            axis.add_collection3d(Line3DCollection(
                segments, colors=color, linewidths=2.6, alpha=1.0, zorder=10
            ))
            all_points.append(points)
            plotted_for_component += 1
            plotted_boundary_count += 1
        if plotted_for_component:
            legend_handles.append(plt.Line2D(
                [0], [0], color=color, linewidth=2,
                label=f"P{component_index:02d}  {color_name}",
            ))
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
        "Original painted model with recognized boundaries (5% arc sampling)"
        if render_source_mesh
        else "Recognized region boundaries (5% arc sampling)"
    )
    axis.set_xlabel("X (mm)")
    axis.set_ylabel("Y (mm)")
    axis.set_zlabel("Z (mm)")
    axis.xaxis.label.set_color("#E6E8EB")
    axis.yaxis.label.set_color("#E6E8EB")
    axis.zaxis.label.set_color("#E6E8EB")
    if render_source_mesh:
        axis.view_init(elev=20, azim=-90)
    axis.title.set_color("#FFFFFF")
    if legend_handles:
        legend = axis.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0))
        legend.get_frame().set_facecolor("#30363D")
        legend.get_frame().set_edgecolor("#697078")
        for text in legend.get_texts():
            text.set_color("#FFFFFF")
    figure.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return plotted_boundary_count


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
        "边界重合时会互相遮盖，预览中的重合处只显示一种颜色。",
        "",
        "| 区域 | 颜色 | 面数 / 面积 mm² | 判断建议 |",
        "|---|---|---:|---|",
    ]
    for component_index, part in sorted(recognition_by_index.items()):
        review = classification_by_index.get(component_index, {})
        classification = str(
            review.get("classification") or part.get("source_region_classification") or ""
        )
        if classification == "noise":
            suggestion = "建议删除（复核标为噪声）"
        elif classification == "part":
            suggestion = "建议保留（复核为独立部件）"
        elif classification == "uncertain":
            suggestion = "需人工判断；检查是否应与相邻区域合并"
        else:
            suggestion = "检查外观与语义；不独立时可考虑合并"
        semantic = str(
            part.get("region_review_label") or part.get("visual_semantic_label")
            or part.get("label") or part.get("semantic_label") or "未标注"
        )
        semantic_confidence = str(part.get("visual_semantic_confidence") or "").upper()
        confidence_label = {
            "HIGH": "高置信度",
            "MED": "中置信度",
            "MEDIUM": "中置信度",
            "LOW": "低置信度",
            "VERYLOW": "极低置信度",
        }.get(semantic_confidence)
        if semantic != "未标注" and confidence_label:
            semantic = f"{semantic}（{confidence_label}）"
        color = str(part.get("color_hex") or part.get("color_code") or "未知")
        color_name = _chinese_color_name(color)
        lines.append(
            f"| P{component_index:02d} | {color_name}（{color}；{semantic}） | "
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
