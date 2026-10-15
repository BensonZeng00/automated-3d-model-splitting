"""Render image-grounded semantic proposals from saved recognition artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from .recognition_metadata import meaningful_semantic_label
from .stage_table_report import markdown_table, recognition_table


def _validate_proposal(run_dir: Path, proposal: dict, summary: dict) -> tuple[dict, dict, set[str]]:
    if proposal.get("schema") != "painted-3mf-semantic-proposals/v1":
        raise ValueError("unsupported semantic review schema")
    if Path(proposal.get("source_run_dir", "")).resolve() != Path(run_dir).resolve():
        raise ValueError("semantic review belongs to a different Stage 03 run")
    expected_parts = {f"P{int(item['part_index']):02d}" for item in summary["regions"]}
    expected_fragments = {
        f"F{int(item['fragment_id']):03d}"
        for item in summary["source_region_classifications"]
    }
    parts, fragments = proposal.get("parts", {}), proposal.get("fragments", {})
    if set(parts) != expected_parts or set(fragments) != expected_fragments:
        raise ValueError("semantic review must cover every saved F/P region")
    for identifier, item in {**parts, **fragments}.items():
        if (not meaningful_semantic_label(item.get("label"))
                or item.get("confidence") not in {"LOW", "MED", "HIGH"}
                or not str(item.get("evidence") or "").strip()):
            raise ValueError(f"{identifier} lacks an image-grounded semantic decision")
    for identifier, item in fragments.items():
        if item.get("suggested_classification") not in {"noise", "part", "uncertain"}:
            raise ValueError(f"{identifier} needs a classification proposal")
    return parts, fragments, expected_fragments


def ensure_semantic_review_confirmed(
    run_dir: Path, proposal_path: Path | None, summary: dict
) -> dict:
    """Guard artifact-only Stage 05 against skipped F/P image interpretation."""
    if proposal_path is None:
        raise ValueError("confirmed --semantic-review-json is required before Stage 05")
    proposal = json.loads(Path(proposal_path).read_text(encoding="utf-8"))
    _parts, fragments, _expected = _validate_proposal(run_dir, proposal, summary)
    if proposal.get("user_confirmed") is not True:
        raise ValueError("semantic review still awaits user confirmation")
    expected_fragments = {
        f"F{int(item['fragment_id']):03d}": str(item["classification"])
        for item in summary["source_region_classifications"]
    }
    for identifier, classification in expected_fragments.items():
        if fragments[identifier].get("suggested_classification") != classification:
            raise ValueError(
                f"{identifier} classification differs from saved recognition; "
                "rebuild Stage 03 from saved Stage 02 geometry before Stage 05"
            )
    return proposal


def render_semantic_review_from_artifacts(
    run_dir: Path, proposal_path: Path, output_dir: Path
) -> dict[str, Path]:
    run_dir, proposal_path, output_dir = map(Path, (run_dir, proposal_path, output_dir))
    summary_path = run_dir / "03_recognition_summary.json"
    stage = json.loads(summary_path.read_text(encoding="utf-8"))
    if stage.get("stage") != "03_recognition_summary" or stage.get("status") != "completed":
        raise ValueError("completed Stage 03 summary is required")
    summary = stage["result"]
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    parts, fragments, expected_fragments = _validate_proposal(run_dir, proposal, summary)

    annotated = dict(summary)
    annotated["regions"] = []
    for region in summary["regions"]:
        identifier = f"P{int(region['part_index']):02d}"
        item = parts[identifier]
        annotated["regions"].append({
            **region,
            "visual_semantic_label": item["label"],
            "visual_semantic_confidence": item["confidence"],
            "visual_semantic_evidence": item["evidence"],
        })
    part_table = recognition_table(annotated)
    part_table["columns"].extend(["语义置信度", "图像依据"])
    for row, region in zip(part_table["rows"], annotated["regions"]):
        row.extend([region["visual_semantic_confidence"], region["visual_semantic_evidence"]])

    fragment_rows = []
    review_dir = Path(proposal["region_review_dir"])
    for record in summary["source_region_classifications"]:
        identifier = f"F{int(record['fragment_id']):03d}"
        item = fragments[identifier]
        image = review_dir / f"{identifier}_context_and_zoom.png"
        if not image.is_file():
            raise ValueError(f"missing saved review image: {image}")
        fragment_rows.append([
            identifier, item["label"], item["evidence"], item["confidence"],
            item["suggested_classification"],
            str(record.get("classification") or "未确认"),
            f"[六视图]({image.as_posix()})",
        ])
    fragment_table = {
        "columns": ["区域", "图像语义", "视觉依据", "置信度", "本次建议", "已有分类", "预览"],
        "rows": fragment_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    part_path = output_dir / "03_recognition_semantic_review.md"
    fragment_path = output_dir / "source_region_semantic_review.md"
    part_path.write_text(markdown_table("零件语义识别提案（待用户确认）", part_table), encoding="utf-8")
    fragment_path.write_text(markdown_table("小区域语义识别提案（待用户确认）", fragment_table), encoding="utf-8")
    manifest_path = review_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_ids = {
        f"F{int(item['fragment_id']):03d}": int(item["source_min_face_index"])
        for item in manifest["items"]
    }
    if set(source_ids) != expected_fragments:
        raise ValueError("saved region manifest differs from Stage 03 fragments")
    decisions_path = output_dir / "region_decisions_proposed.json"
    decisions_path.write_text(json.dumps({
        "schema_version": manifest["schema_version"],
        "source": manifest["source"],
        "source_face_count": manifest["source_face_count"],
        "noise_max_faces": manifest["noise_max_faces"],
        "small_region_max_faces": manifest["small_region_max_faces"],
        "max_review_candidates": manifest["max_review_candidates"],
        "user_confirmed": False,
        "items": [
            {
                "fragment_id": int(identifier[1:]),
                "source_min_face_index": source_ids[identifier],
                "semantic_label": item["label"],
                "visual_confidence": item["confidence"],
                "classification": item["suggested_classification"],
            }
            for identifier, item in sorted(fragments.items())
        ],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    part_semantics_path = output_dir / "part_semantics_proposed.json"
    part_semantics_path.write_text(json.dumps({
        "parts": {
            identifier: {
                "label": item["label"],
                "confidence": item["confidence"],
                "visual_evidence": item["evidence"],
            }
            for identifier, item in sorted(parts.items())
        },
        "physical_partitions": [],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "parts": part_path, "fragments": fragment_path,
        "region_decisions": decisions_path,
        "part_semantics": part_semantics_path,
    }
