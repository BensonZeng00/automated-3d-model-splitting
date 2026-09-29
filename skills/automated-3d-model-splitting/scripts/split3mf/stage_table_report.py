"""Small, inspectable tables for recognition and interface-plan artifacts."""

from __future__ import annotations

from pathlib import Path
import json


def recognition_table(summary: dict) -> dict:
    loop_counts = summary.get("recognized_boundaries", {}).get("loop_vertex_counts", [])
    rows = []
    for region in summary.get("regions", []):
        index = int(region["part_index"])
        rows.append([
            f"P{index:02d}",
            str(region.get("visual_semantic_label") or "待确认"),
            str(region.get("color_code") or ""),
            str(region.get("color_hex") or ""),
            int(region.get("faces", 0)),
            round(float(region.get("area_mm2", 0.0)), 3),
            len(loop_counts[index - 1]) if index <= len(loop_counts) else 0,
            str(region.get("source_region_classification") or "有效零件"),
        ])
    return {
        "columns": ["零件", "区域", "材料码", "模型颜色", "面数", "面积 mm²", "简化边界数", "分类"],
        "rows": rows,
    }


def interface_table(plan: dict) -> dict:
    names = {
        str(part["part"]): str(part.get("visual_semantic_label") or "")
        for part in plan.get("parts", [])
    }

    def part_label(part_id: str) -> str:
        label = names.get(part_id, "")
        return f"{part_id} {label}" if label else part_id

    rows = []
    for interface in plan.get("interfaces", []):
        contact = interface.get("contact", {})
        rows.append([
            str(interface["interface_id"]),
            part_label(str(interface["tenon_part"])),
            part_label(str(interface["mortise_part"])),
            len(contact.get("shared_boundary_loops", [])),
            int(contact.get("shared_boundary_sample_count", 0)),
            {
                "inward_alignment": "内向法线",
                "large_area_difference": "面积悬殊，小面积者优先",
                "smaller_region_tiebreak": "面积较小者优先",
                "user_confirmed_part_center_axis": "已确认的零件中心接合方向",
                "user_confirmed_local_surface_axis": "已确认的局部接合面插入方向",
            }.get(
                str(interface.get("direction_evidence", {}).get("method") or ""),
                str(interface.get("direction_evidence", {}).get("method") or "待确认"),
            ),
        ])
    return {
        "columns": ["接口", "榫方", "卯方", "共用简化边界数", "边界采样点数", "方向依据"],
        "rows": rows,
    }


def markdown_table(title: str, table: dict) -> str:
    columns = table["columns"]
    rows = table["rows"]

    def cell(value: object) -> str:
        return str(value if value is not None else "").replace("|", "\\|").replace("\n", " ")

    lines = [f"# {title}", "", "| " + " | ".join(map(cell, columns)) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(map(cell, row)) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def write_markdown_table(path: Path, title: str, table: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_table(title, table), encoding="utf-8")
    return path


def publish_stage_tables(run_dir: Path) -> dict[str, Path]:
    """Build user-facing tables from completed 03/04 JSON without the source mesh."""
    from .application.stage_artifacts import StageArtifactStore

    run_dir = Path(run_dir).expanduser().resolve()

    def result(stage: str) -> dict:
        path = run_dir / f"{stage}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("stage") != stage or record.get("status") != "completed":
            raise ValueError(f"stage artifact is not completed: {path}")
        return record["result"]

    tables = {
        "03_recognition_table": ("03 识别零件表", recognition_table(result("03_recognition_summary"))),
        "04_interface_table": ("04 榫卯接口表", interface_table(result("04_assembly_plan"))),
    }
    store = StageArtifactStore(run_dir.parent, run_id=run_dir.name)
    paths = {}
    for stage, (title, table) in tables.items():
        store.write_json(stage, table)
        paths[stage] = write_markdown_table(run_dir / f"{stage}.md", title, table)
    return paths
