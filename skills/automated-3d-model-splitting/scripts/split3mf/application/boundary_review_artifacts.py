from __future__ import annotations

from pathlib import Path

import numpy as np

from ..planar_arc import (
    cyclic_binomial_smooth,
    fit_periodic_cubic_bspline,
    sample_closed_curve,
    sample_periodic_cubic_bspline,
)


def write_boundary_review_artifacts(
    directory: Path,
    boundaries,
    recognition_records: list[dict],
    region_review: dict,
    components: list,
    recognition_review_path: Path | None = None,
) -> dict[str, str | int]:
    """Write a color-coded boundary preview and a compact human review report."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    image_path = directory / "03_recognition_boundaries.png"
    report_path = directory / "03_recognition_boundary_review.md"
    records = list(boundaries.simplification_records)
    plotted_boundary_count = _write_boundary_plot(image_path, boundaries)
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


def _write_boundary_plot(path: Path, boundaries) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(13, 10))
    figure.patch.set_facecolor("#20242A")
    axis = figure.add_subplot(111, projection="3d")
    axis.set_facecolor("#20242A")
    for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
        pane.set_facecolor((0.12, 0.14, 0.16, 1.0))
        pane.set_edgecolor("#697078")
    axis.tick_params(colors="#E6E8EB")
    all_points = []
    palette = [
        ("red", "#E31A1C"), ("white", "#FFFFFF"), ("blue", "#1F78B4"),
        ("yellow", "#FFD92F"), ("purple", "#984EA3"), ("orange", "#FF7F00"),
        ("cyan", "#00BFC4"), ("pink", "#F781BF"), ("green", "#33A02C"),
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
            display_count = max(96, min(512, len(points) * 2))
            arc_samples = sample_closed_curve(
                points, np.arange(display_count, dtype=np.float64) / display_count
            )
            if len(arc_samples) >= 12:
                controls = fit_periodic_cubic_bspline(
                    arc_samples, control_count=max(4, min(32, len(arc_samples) // 8))
                )
                fractions = np.arange(display_count, dtype=np.float64) / display_count
                smoothed = sample_periodic_cubic_bspline(controls, fractions)
            else:
                smoothed = cyclic_binomial_smooth(arc_samples, passes=8)
            closed = np.vstack((smoothed, smoothed[0]))
            axis.plot(*closed.T, color=color, linewidth=1.4, alpha=0.95)
            all_points.append(smoothed)
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
        axis.set_xlim(center[0] - extent.max() / 2, center[0] + extent.max() / 2)
        axis.set_ylim(center[1] - extent.max() / 2, center[1] + extent.max() / 2)
        axis.set_zlim(center[2] - extent.max() / 2, center[2] + extent.max() / 2)
    axis.set_title("Filtered and smoothed region boundaries (5% arc sampling)")
    axis.set_xlabel("X (mm)")
    axis.set_ylabel("Y (mm)")
    axis.set_zlabel("Z (mm)")
    axis.xaxis.label.set_color("#E6E8EB")
    axis.yaxis.label.set_color("#E6E8EB")
    axis.zaxis.label.set_color("#E6E8EB")
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
            part.get("region_review_label") or part.get("label")
            or part.get("semantic_label") or "未标注"
        )
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
        return "橙色"
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
