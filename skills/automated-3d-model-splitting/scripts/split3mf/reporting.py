from __future__ import annotations

import time

from .common import *


_RUNTIME_LOG_STARTED_AT = time.perf_counter()


def machine_record(name: str, payload: dict, *, blocking: bool = False) -> None:
    """Emit a machine-readable record on the stream matching its severity."""
    stream = sys.stderr if blocking else sys.stdout
    print(
        f"{name}=" + json.dumps(payload, ensure_ascii=False),
        file=stream,
        flush=True,
    )


def runtime_log(stage: str, event: str, message: str, **details) -> None:
    """Emit one readable checkpoint and one machine-readable JSON log line."""
    record = {
        "stage": str(stage),
        "event": str(event),
        "message": str(message),
        "elapsed_seconds": round(
            float(time.perf_counter() - _RUNTIME_LOG_STARTED_AT),
            3,
        ),
        **details,
    }
    readable_details = (
        " " + json.dumps(details, ensure_ascii=False, sort_keys=True)
        if details
        else ""
    )
    print(f"[{stage}] {message}{readable_details}", flush=True)
    print(
        "runtime_step="
        + json.dumps(record, ensure_ascii=False, sort_keys=True),
        flush=True,
    )


def write_markdown_report(path: Path, report: dict) -> None:
    lines = [
        f"# {Path(report['source']).name} Split Parts Report",
        "",
        f"- Source: `{report['source']}`",
        f"- Selected model entry: `{report.get('selected_model_entry', '')}`",
        f"- Vertex count: `{report['source_vertex_count']}`",
        f"- Triangle count: `{report['source_face_count']}`",
        f"- Region review policy: `{report.get('region_review_policy', 'preserve-source')}`",
        f"- Noise review threshold: `<= {report.get('noise_review_max_faces', 100)}` faces",
        f"- Small-region review threshold: `<= {report.get('small_region_review_max_faces', 999)}` faces",
        f"- Reviewed source regions: `{report.get('reviewed_source_region_count', 0)}`",
        f"- Interface retopology: `{report.get('interface_retopology_mode', 'planar-arc-retopology')}`",
        f"- Equal arc samples: `{report.get('boundary_target_samples', 384)}`",
        f"- Cyclic binomial passes: `{report.get('boundary_smooth_passes', 28)}`",
        f"- Retopology band: `{report.get('boundary_retopology_band_mm', 3.0)} mm`",
        (
            f"- Interface slope: target `{report.get('boundary_target_slope_deg', 45.0)}°`, "
            f"allowed `{report.get('boundary_min_slope_deg', 30.0)}–"
            f"{report.get('boundary_max_slope_deg', 75.0)}°`"
        ),
        f"- Requested inward cap depth: `{report.get('requested_max_extension_mm', report['max_extension_mm'])} mm`",
        f"- Minimum flat-bottom depth: `{report.get('minimum_flat_bottom_depth_mm', MINIMUM_INWARD_DEPTH_MM)} mm`",
        f"- Fixed inward cap depth: `{report['max_extension_mm']} mm`",
        f"- Cap mode: `{report.get('cap_mode', 'flat')}`",
        f"- Nested cap mode: `{report.get('nested_cap_mode', 'flat')}`",
        f"- Planar extra limit: `{report.get('planar_extra_limit_mm', 0.0)} mm`",
        f"- Force-flat parts: `{','.join(f'P{int(index):02d}' for index in report.get('force_flat_parts', [])) or 'none'}`",
        f"- Flat-bottom clearance: `{report['flat_clearance_mm']} mm` (reported only; not added to fixed inward cap depth)",
        f"- Fit clearance: `{report['fit_clearance_mm']} mm`",
        f"- Clearance mode: `{report['clearance_mode']}`",
        f"- Insert shrink: `{report['insert_shrink_mm']} mm`",
        f"- Socket overcut: `{report['socket_overcut_mm']} mm`",
        f"- Bottom clearance: `{report['bottom_clearance_mm']} mm` (not added to fixed inward cap depth)",
        f"- Lead-in: `{report['lead_in_mm']} mm`",
        f"- Sibling clearance: `{report.get('sibling_clearance_mm', 0.0)} mm`",
        f"- Assembly mode: `{report.get('assembly_mode', 'flat')}`",
        f"- Assembly tree strategy: `{report.get('assembly_tree_strategy', 'recursive-minimal')}`",
        f"- Assembly direction override dot threshold: `{report.get('assembly_direction_override_dot', '')}`",
        f"- Visual semantics source: `{report.get('visual_semantics_source') or 'none'}`",
        f"- Visual semantic min confidence: `{report.get('visual_semantic_min_confidence', 'MED')}`",
        f"- Body part: `{report.get('body_part_id', 'none')}`",
        f"- Explicit multi-material body merge: `{report.get('explicit_body_merge') or 'none'}`",
        f"- Exported parts: `{len(report['parts'])}`",
        f"- Recursive layer stage output: `{report.get('recursive_layer_stage_output_dir') or 'none'}`",
        f"- Ignored tiny components: `{len(report['ignored_tiny_components'])}`",
        "",
        "## Recognition",
        "",
        "| Part | Role | Visual semantic | Color | Faces | Size mm | Center | Loops |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for record in report.get("recognition", []):
        size = ", ".join(f"{value:.2f}" for value in record["bbox_size_mm"])
        center = ", ".join(f"{value:.2f}" for value in record["center"])
        semantic = record.get("visual_semantic_label") or ""
        semantic_confidence = record.get("visual_semantic_confidence") or ""
        semantic_text = f"{semantic} `{semantic_confidence}`" if semantic else ""
        lines.append(
            f"| P{record['part_index']:02d} | {record['role']} | {semantic_text} | {record['color_name']} `{record['color_code']}` `{record['color_hex']}` | "
            f"{record['faces']} | {size} | {center} | {record['boundary_loops']} |"
        )
    part_semantics = report.get("visual_part_semantics", [])
    if part_semantics:
        lines.extend(["", "## Visual Part Semantics", ""])
        for item in part_semantics:
            description = item.get("description", "")
            evidence = item.get("visual_evidence", "")
            suffix = f" - {description}" if description else ""
            if evidence:
                suffix += f" Evidence: {evidence}"
            lines.append(
                f"- P{int(item['part_index']):02d}: `{item.get('label', '')}` "
                f"confidence=`{item.get('confidence', 'UNKNOWN')}`{suffix}"
            )
    lines.extend(
        [
            "",
            "## Exported Parts",
            "",
            "| Part | Parent | Children | Depth | Recursive role | Mode | Selected cap | Forced flat | Color | Source faces | Output faces | Loops | Child sockets | Cap faces | Fixed depth mm | Max travel mm | Boundary span mm | Fit mm | Insert shrink mm | Socket overcut mm | Bottom clearance mm | Fixed-depth rule | Watertight | Open edges | Over-shared edges | File |",
            "|---|---|---|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|",
        ]
    )
    for part in report["parts"]:
        parent = part.get("assembly_parent_index")
        parent_text = "none" if parent is None else f"P{int(parent):02d}"
        children = part.get("assembly_child_indices", [])
        children_text = ",".join(f"P{int(child):02d}" for child in children) if children else "none"
        selected_caps = ",".join(part.get("selected_cap_modes", [])) or part.get("cap_mode", "")
        lines.append(
            f"| {part['part_id']} | {parent_text} | {children_text} | {part.get('assembly_depth', 0)} | {part.get('recursive_minimal_role', '')} | {part['processing_mode']} | {selected_caps} | {part.get('forced_flat_cap', False)} | {part['color_name']} `{part['color_hex']}` | "
            "{source_faces} | {output_faces} | {boundary_loops} | {child_socket_count} | "
            "{cap_faces_added} | {fixed_inward_depth_mm:.3f} | {extension_max_observed_mm:.3f} | {boundary_span_max_mm:.3f} | "
            "{fit_clearance_mm:.3f} | {insert_shrink_mm:.3f} | {socket_overcut_mm:.3f} | {bottom_clearance_mm:.3f} | "
            "{fixed_inward_depth_applied} | {reload_watertight} | {reload_open_edges} | {reload_over_shared_edges} | `{file}` |".format(**part)
        )
    skipped_socket_parts = [part for part in report["parts"] if part.get("child_socket_skipped_count", 0)]
    lines.extend(["", "## Skipped Child Sockets", ""])
    if not skipped_socket_parts:
        lines.append("- None")
    else:
        for part in skipped_socket_parts:
            for record in part.get("child_socket_skipped", []):
                lines.append(
                    f"- {part['part_id']}: skipped socket for P{int(record['matched_component_index']):02d}, "
                    f"reason=`{record.get('skip_reason', '')}`, "
                    f"parent_edges=`{record.get('shared_loop_parent_edges', 0)}`, "
                    f"child_edges=`{record.get('shared_loop_child_edges', 0)}`"
                )
    lines.extend(["", "## Recursive Minimal Layers", ""])
    layers = report.get("recursive_minimal_layers", [])
    if not layers:
        lines.append("- None")
    else:
        for layer in layers:
            children = ",".join(f"P{int(child):02d}" for child in layer.get("direct_child_indices", [])) or "none"
            leaves = ",".join(f"P{int(child):02d}" for child in layer.get("leaf_child_indices", [])) or "none"
            nested = ",".join(f"P{int(child):02d}" for child in layer.get("nested_child_indices", [])) or "none"
            lines.append(
                f"- depth `{int(layer.get('depth', 0))}` local body `P{int(layer['local_body_index']):02d}`: "
                f"direct=`{children}`, leaves=`{leaves}`, nested=`{nested}`"
            )
    lines.extend(["", "## Strict Recursive Layer Stage Outputs", ""])
    stage_parts = report.get("recursive_layer_stage_parts", [])
    if not stage_parts:
        lines.append("- None")
    else:
        lines.append(f"- Directory: `{report.get('recursive_layer_stage_output_dir')}`")
        lines.extend(
            [
                "",
                "| Layer | Local body | Stage part | Role | Contains | Direct children | Parent-contact loops | Ignored non-parent loops | Contact edges | Watertight | Open edges | Over-shared edges | File |",
                "|---:|---|---|---|---|---|---:|---:|---:|---|---:|---:|---|",
            ]
        )
        for part in stage_parts:
            contains = ",".join(f"P{int(index):02d}" for index in part.get("stage_contains", part.get("contains", []))) or "none"
            direct = ",".join(f"P{int(index):02d}" for index in part.get("stage_direct_child_indices", [])) or "none"
            local_body = f"P{int(part.get('stage_local_body_index', 0)):02d}"
            selected_loops = part.get("selected_parent_contact_loops", "")
            ignored_loops = part.get("ignored_non_parent_loops", "")
            contact_edges = part.get("parent_contact_edges", "")
            lines.append(
                f"| {int(part.get('stage_layer_order', 0))} | {local_body} | {part['part_id']} | {part.get('stage_role', '')} | "
                f"{contains} | {direct} | {selected_loops} | {ignored_loops} | {contact_edges} | "
                f"{part.get('reload_watertight')} | {part.get('reload_open_edges')} | {part.get('reload_over_shared_edges')} | `{part.get('file')}` |"
            )
    lines.extend(["", "## Assembly Tree", ""])
    for record in report.get("assembly_tree", []):
        parent = record.get("parent_index")
        parent_text = "none" if parent is None else f"P{int(parent):02d}"
        previous_parent = record.get("previous_parent_index")
        previous_text = "" if previous_parent is None else f", previous_parent=`P{int(previous_parent):02d}`"
        lines.append(
            f"- P{int(record['part_index']):02d} -> {parent_text}{previous_text}, "
            f"shared_edges=`{record.get('shared_edges_to_parent', 0)}`, reason=`{record.get('reason', '')}`"
        )
    lines.extend(["", "## Visual Semantic Parent Overrides", ""])
    semantic_parent_changes = report.get("visual_semantic_parent_overrides", {})
    applied_changes = semantic_parent_changes.get("applied", [])
    rejected_changes = semantic_parent_changes.get("rejected", [])
    if not applied_changes and not rejected_changes:
        lines.append("- None")
    else:
        for record in applied_changes:
            previous_parent = record.get("previous_parent_index")
            previous_text = "none" if previous_parent is None else f"P{int(previous_parent):02d}"
            lines.append(
                f"- {record.get('change_type', 'applied')} P{int(record['child_index']):02d}: {previous_text} -> P{int(record['new_parent_index']):02d}, "
                f"confidence=`{record.get('confidence', 'UNKNOWN')}`, relation=`{record.get('relation', '')}`, "
                f"new_shared_edges=`{record.get('topology_shared_edges_to_new_parent', 0)}`, "
                f"previous_shared_edges=`{record.get('topology_shared_edges_to_previous_parent', 0)}`, "
                f"parent_larger=`{record.get('parent_larger_by_topology_rule')}`, reason=`{record.get('reason', '')}`"
            )
        for record in rejected_changes:
            lines.append(
                f"- rejected P{int(record['child_index']):02d} -> P{int(record['parent_index']):02d}, "
                f"confidence=`{record.get('confidence', 'UNKNOWN')}`, reject_reason=`{record.get('reject_reason', '')}`"
            )
    lines.extend(["", "## Recursive Minimal Reparents", ""])
    recursive_reparents = report.get("recursive_minimal_reparents", [])
    if not recursive_reparents:
        lines.append("- None")
    else:
        for record in recursive_reparents:
            previous_parent = record.get("previous_parent_index")
            previous_text = "none" if previous_parent is None else f"P{int(previous_parent):02d}"
            lines.append(
                f"- P{int(record['part_index']):02d}: {previous_text} -> P{int(record['new_parent_index']):02d}, "
                f"loop=`{record.get('parent_loop_index')}`, child_edges=`{record.get('child_edges_on_parent_loop')}`, "
                f"excluded_edges=`{record.get('excluded_neighbor_edges_on_parent_loop')}`, "
                f"reason=`{record.get('reason', '')}`"
            )
    lines.extend(["", "## Mixed Boundary Reparents", ""])
    reparent_records = report.get("mixed_boundary_reparents", [])
    if not reparent_records:
        lines.append("- None")
    else:
        for record in reparent_records:
            lines.append(
                f"- P{int(record['part_index']):02d}: P{int(record['previous_parent_index']):02d} -> "
                f"P{int(record['new_parent_index']):02d}, parent_loop=`{record['parent_loop_index']}`, "
                f"child_edges=`{record['child_edges_on_parent_loop']}`, "
                f"grandparent_edges=`{record['grandparent_edges_on_parent_loop']}`"
            )
    lines.extend(["", "## Cycle Breaks", ""])
    cycle_breaks = report.get("cycle_breaks", [])
    if not cycle_breaks:
        lines.append("- None")
    else:
        for record in cycle_breaks:
            previous_parent = record.get("previous_parent_index")
            new_parent = record.get("new_parent_index")
            previous_text = "none" if previous_parent is None else f"P{int(previous_parent):02d}"
            new_text = "none" if new_parent is None else f"P{int(new_parent):02d}"
            cycle_nodes = ",".join(f"P{int(index):02d}" for index in record.get("cycle_nodes", []))
            lines.append(
                f"- P{int(record['part_index']):02d}: {previous_text} -> {new_text}, "
                f"cycle=`{cycle_nodes}`, shared_edges=`{record.get('shared_edges_to_new_parent', 0)}`, "
                f"reason=`{record.get('reason', '')}`"
            )
    lines.extend(["", "## Inward Direction Overrides", ""])
    overrides = report.get("inward_direction_overrides", [])
    if not overrides:
        lines.append("- None")
    else:
        for record in overrides:
            parent = record.get("parent_index")
            parent_text = "none" if parent is None else f"P{int(parent):02d}"
            status = "override" if record.get("overridden") else "preserve"
            lines.append(
                f"- P{int(record['part_index']):02d} -> {parent_text}: `{status}`, "
                f"alignment_dot=`{float(record.get('alignment_dot', 0.0)):.3f}`, threshold=`{record.get('threshold')}`"
            )
    lines.extend(["", "## Semantic-Preserved Tiny Components", ""])
    semantic_preserved = report.get("semantic_preserved_tiny_components", [])
    if not semantic_preserved:
        lines.append("- None")
    else:
        for item in semantic_preserved:
            lines.append(
                f"- Fragment `{int(item['fragment_id'])}` `{item['color_code']}`: "
                f"`{int(item['faces'])}` faces, score=`{float(item['semantic_keep_score']):.3f}`, "
                f"loops=`{int(item['boundary_loop_count'])}`, "
                f"projected span=`{float(item['projected_max_span_pixels']):.2f}px`, "
                f"decision=`{item['semantic_decision']}`"
            )
    lines.extend(["", "## Merged Tiny Components", ""])
    merged_tiny = report.get("merged_tiny_components", [])
    if not merged_tiny:
        lines.append("- None")
    else:
        by_part: dict[int, list[dict]] = collections.defaultdict(list)
        for item in merged_tiny:
            by_part[int(item["assigned_component_index"])].append(item)
        for part_index, items in sorted(by_part.items()):
            total_faces = sum(int(item["faces"]) for item in items)
            total_area = sum(float(item["area_mm2"]) for item in items)
            shared = sum(1 for item in items if item.get("assignment_method") == "shared_edge")
            nearest = sum(1 for item in items if item.get("assignment_method") == "nearest_bbox")
            lines.append(
                f"- P{int(part_index):02d}: `{len(items)}` fragments merged, `{total_faces}` faces, "
                f"`{total_area:.6f} mm^2`, shared_edge=`{shared}`, nearest_bbox=`{nearest}`"
            )
    lines.extend(["", "## Ignored Tiny Components", ""])
    by_color: dict[str, list[dict]] = collections.defaultdict(list)
    for item in report["ignored_tiny_components"]:
        by_color[item["color_code"]].append(item)
    if not by_color:
        lines.append("- None")
    else:
        for color, items in sorted(by_color.items(), key=lambda kv: COLOR_ORDER.get(kv[0], 99)):
            total_faces = sum(i["faces"] for i in items)
            total_area = sum(i["area_mm2"] for i in items)
            lines.append(
                f"- `{color}`: `{len(items)}` components, `{total_faces}` faces, `{total_area:.6f} mm^2` total area"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

