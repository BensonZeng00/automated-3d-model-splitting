from __future__ import annotations

import copy
import hashlib
import os
import tempfile

from .common import *
from .project import *
from .recognition import *
from .mesh import *
from .package_io import *
from .selection import *
from .assembly import *
from .validation import *
from .inward import *
from .local_connectors import (
    assembly_interface_policy,
)
from .connector_planning import DEFAULT_DEPTH_POLICY
from .domain import PlanarArcRetopologyContext, CapDecision
from .reporting import runtime_log
from .boolean_case_cache import write_boolean_case_cache
from .stage_cache import RecursiveStageCache
from .uniform_fit import exact_unscaled_cutter


def export_retopology_failure_diagnostics(
    payload: dict,
    output_dir: Path,
) -> dict[str, str]:
    """Persist compact, source-backed plots for one blocked surface band."""

    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "split3mf-matplotlib"),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    source = np.asarray(payload["source_points"], dtype=np.float64)
    result = np.asarray(payload["result_points"], dtype=np.float64)
    faces = np.asarray(payload["faces"], dtype=np.int64)
    boundary = np.asarray(payload["boundary_ids"], dtype=np.int64)
    targets = np.asarray(payload["target_boundary_points"], dtype=np.float64)
    affected = np.asarray(payload["affected_face_ids"], dtype=np.int64)
    degenerate = np.asarray(payload["degenerate_face_ids"], dtype=np.int64)
    reversed_ids = np.asarray(payload["reversed_face_ids"], dtype=np.int64)
    overstretched = np.asarray(payload["overstretched_face_ids"], dtype=np.int64)
    bad_ids = np.unique(np.concatenate((degenerate, reversed_ids, overstretched)))

    boundary_source = source[boundary]
    origin = boundary_source.mean(axis=0)
    _u, _singular_values, basis = np.linalg.svd(
        boundary_source - origin[None, :],
        full_matrices=False,
    )
    axis_u = basis[0]
    axis_v = basis[1]

    def project(points: np.ndarray) -> np.ndarray:
        centered = np.asarray(points, dtype=np.float64) - origin[None, :]
        return np.column_stack((centered @ axis_u, centered @ axis_v))

    source_2d = project(boundary_source)
    target_2d = project(targets)
    displacement = np.linalg.norm(targets - boundary_source, axis=1)
    maximum_index = int(np.argmax(displacement))
    maximum_offset = float(displacement[maximum_index])

    boundary_png = destination / "retopology_failure_boundary.png"
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 6.4))
    figure.suptitle("Blocked visible seam: source vs planar-arc target", fontsize=16)
    for axis in axes:
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, color="#D1D5DB", linewidth=0.7, alpha=0.7)
        axis.set_xlabel("stable-plane U (mm)")
        axis.set_ylabel("stable-plane V (mm)")
    closed_source = np.vstack((source_2d, source_2d[:1]))
    closed_target = np.vstack((target_2d, target_2d[:1]))
    axes[0].plot(closed_source[:, 0], closed_source[:, 1], "#2563EB", lw=1.1, label="source")
    axes[0].plot(closed_target[:, 0], closed_target[:, 1], "#EA580C", lw=1.5, label="target")
    axes[0].set_title(f"Full loop ({len(boundary):,} boundary vertices)")
    axes[0].legend(frameon=False)
    axes[1].plot(closed_source[:, 0], closed_source[:, 1], "#2563EB", lw=1.2)
    axes[1].plot(closed_target[:, 0], closed_target[:, 1], "#EA580C", lw=1.6)
    axes[1].plot(
        [source_2d[maximum_index, 0], target_2d[maximum_index, 0]],
        [source_2d[maximum_index, 1], target_2d[maximum_index, 1]],
        "-o",
        color="#111827",
        ms=3,
    )
    local_center = 0.5 * (source_2d[maximum_index] + target_2d[maximum_index])
    half_span = max(maximum_offset * 1.4, 0.8)
    axes[1].set_xlim(local_center[0] - half_span, local_center[0] + half_span)
    axes[1].set_ylim(local_center[1] - half_span, local_center[1] + half_span)
    axes[1].set_title(f"Maximum requested move: {maximum_offset:.3f} mm")
    figure.text(
        0.5,
        0.02,
        "Blue is the original painted seam; orange is the manufacturing target. No failed mesh was exported.",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0.0, 0.05, 1.0, 0.94))
    figure.savefig(boundary_png, dpi=180, bbox_inches="tight")
    plt.close(figure)

    quality_png = destination / "retopology_failure_quality_map.png"
    display_ids = affected
    if len(display_ids) > 15000:
        display_ids = display_ids[
            np.linspace(0, len(display_ids) - 1, 15000, dtype=np.int64)
        ]
    display_triangles = project(result[faces[display_ids]].reshape(-1, 3)).reshape(-1, 3, 2)
    display_segments = np.concatenate(
        (
            display_triangles[:, [0, 1]],
            display_triangles[:, [1, 2]],
            display_triangles[:, [2, 0]],
        ),
        axis=0,
    )
    figure, axis = plt.subplots(figsize=(10.5, 8.2))
    axis.add_collection(LineCollection(display_segments, colors="#CBD5E1", linewidths=0.25))
    categories = (
        (overstretched, "#A855F7", "edge stretch > 8×"),
        (reversed_ids, "#F97316", "locally reversed"),
        (degenerate, "#DC2626", "degenerate"),
    )
    for face_ids, color, label in categories:
        if not len(face_ids):
            continue
        selected = face_ids[:5000]
        polygons = project(result[faces[selected]].reshape(-1, 3)).reshape(-1, 3, 2)
        axis.add_collection(
            PolyCollection(
                polygons,
                facecolors=color,
                edgecolors=color,
                linewidths=0.35,
                alpha=0.72,
                label=f"{label} ({len(face_ids):,})",
            )
        )
    projected_result = project(result[faces[display_ids]].reshape(-1, 3))
    axis.update_datalim(projected_result)
    axis.autoscale_view()
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("stable-plane U (mm)")
    axis.set_ylabel("stable-plane V (mm)")
    axis.set_title("Actual blocked surface-band quality map")
    axis.grid(True, color="#E5E7EB", linewidth=0.6)
    if any(len(face_ids) for face_ids, _color, _label in categories):
        axis.legend(frameon=False, loc="best")
    figure.tight_layout()
    figure.savefig(quality_png, dpi=190, bbox_inches="tight")
    plt.close(figure)

    section_png = destination / "retopology_failure_45deg_section.png"
    seed_vertex = int(boundary[maximum_index])
    incident_ids = np.flatnonzero(np.any(faces == seed_vertex, axis=1))
    local_vertex_ids = np.unique(faces[incident_ids])
    local_ids = np.flatnonzero(np.any(np.isin(faces, local_vertex_ids), axis=1))
    local_ids = local_ids[:1500]
    incident_source = source[faces[incident_ids]]
    incident_normals = np.cross(
        incident_source[:, 1] - incident_source[:, 0],
        incident_source[:, 2] - incident_source[:, 0],
    )
    section_y = incident_normals.sum(axis=0)
    section_y /= max(float(np.linalg.norm(section_y)), 1e-15)
    section_x = targets[maximum_index] - boundary_source[maximum_index]
    section_x -= float(section_x @ section_y) * section_y
    if float(np.linalg.norm(section_x)) <= 1e-12:
        section_x = axis_u - float(axis_u @ section_y) * section_y
    section_x /= max(float(np.linalg.norm(section_x)), 1e-15)
    section_origin = boundary_source[maximum_index]

    def section_project(points: np.ndarray) -> np.ndarray:
        centered = np.asarray(points, dtype=np.float64) - section_origin[None, :]
        return np.column_stack((centered @ section_x, centered @ section_y))

    source_local = section_project(source[faces[local_ids]].reshape(-1, 3)).reshape(-1, 3, 2)
    result_local = section_project(result[faces[local_ids]].reshape(-1, 3)).reshape(-1, 3, 2)
    source_segments = np.concatenate(
        (source_local[:, [0, 1]], source_local[:, [1, 2]], source_local[:, [2, 0]]),
        axis=0,
    )
    result_segments = np.concatenate(
        (result_local[:, [0, 1]], result_local[:, [1, 2]], result_local[:, [2, 0]]),
        axis=0,
    )
    target_section = section_project(targets[maximum_index][None, :])[0]
    ray_length = max(3.0, maximum_offset)
    nominal_end = target_section + ray_length * np.asarray([2.0 ** -0.5, -2.0 ** -0.5])
    figure, axis = plt.subplots(figsize=(9.5, 7.0))
    axis.add_collection(LineCollection(source_segments, colors="#2563EB", linewidths=0.65, alpha=0.55, label="source local mesh"))
    axis.add_collection(LineCollection(result_segments, colors="#EA580C", linewidths=0.65, alpha=0.55, label="deformed local mesh"))
    axis.plot(0.0, 0.0, "o", color="#2563EB", label="source seam point")
    axis.plot(target_section[0], target_section[1], "o", color="#EA580C", label="target seam point")
    axis.plot(
        [target_section[0], nominal_end[0]],
        [target_section[1], nominal_end[1]],
        "--",
        color="#059669",
        lw=2.0,
        label="nominal 45° backing direction",
    )
    all_section_points = np.vstack((source_local.reshape(-1, 2), result_local.reshape(-1, 2), nominal_end))
    axis.update_datalim(all_section_points)
    axis.autoscale_view()
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("local seam travel (mm)")
    axis.set_ylabel("local surface-normal direction (mm)")
    axis.set_title("Local section projection at maximum requested seam move")
    axis.grid(True, color="#E5E7EB", linewidth=0.6)
    axis.legend(frameon=False, loc="best")
    figure.tight_layout()
    figure.savefig(section_png, dpi=190, bbox_inches="tight")
    plt.close(figure)

    data_path = destination / "retopology_failure_data.npz"
    np.savez_compressed(
        data_path,
        boundary_source=boundary_source,
        boundary_target=targets,
        bad_face_ids=bad_ids,
        bad_source_triangles=source[faces[bad_ids]],
        bad_result_triangles=result[faces[bad_ids]],
        degenerate_face_ids=degenerate,
        reversed_face_ids=reversed_ids,
        overstretched_face_ids=overstretched,
    )
    quality_path = destination / "retopology_failure_quality.json"
    quality_path.write_text(
        json.dumps(payload["quality"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    artifacts = {
        "boundary_png": str(boundary_png),
        "quality_map_png": str(quality_png),
        "section_png": str(section_png),
        "data_npz": str(data_path),
        "quality_json": str(quality_path),
    }
    runtime_log(
        "失败诊断",
        "retopology_failure_artifacts_written",
        "表面带失败图与可复用诊断数据已写出",
        **artifacts,
    )
    return artifacts


def shared_child_cap_decisions(
    child_refs: list[dict],
    child_index: int,
) -> dict[int, CapDecision]:
    """Select the prevalidated cap decisions owned by one direct child."""
    decisions: dict[int, CapDecision] = {}
    for ref in child_refs:
        if int(ref.get("component_index", -1)) != int(child_index):
            continue
        decision = ref.get("cap_decision")
        if decision is None:
            continue
        loop_index = int(ref["loop_index"])
        if loop_index in decisions:
            raise ValueError(
                f"duplicate shared cap decision for P{int(child_index):02d} "
                f"loop {loop_index}"
            )
        decisions[loop_index] = decision
    return decisions


def validate_shared_child_cap_decisions(
    child_index: int,
    decisions: dict[int, CapDecision],
    child_stats: dict,
    *,
    interface_geometry: str = "boundary-extrusion",
) -> list[dict]:
    """Block publication when a child did not consume its planned cap field."""
    repair = child_stats.get('actual_backing_validation') or {}
    if repair.get('repaired'):
        from .print_tolerance import current
        geometry = repair.get('construction', {})
        if not (current().repair_thin_backing and interface_geometry == 'local-connector'
                and repair.get('source_front_triangles_preserved')
                and repair.get('after_finalization', {}).get('valid')
                and repair.get('matching_socket_fit', {}).get('valid')
                and repair.get('outside_volume_mm3', float('inf')) <= 1e-8
                and geometry.get('minimum_parent_reserve_mm', 0) >= .05-1e-8
                and 0 < geometry.get('maximum_depth_mm', 0) <= 10):
            raise ValueError(f'P{child_index:02d}: incomplete source-following backing safety record')
        return [dict(loop_index=int(loop_index), cap_mode='source_following_local_normals',
                     validation='explicitly_rebuilt_complete_child_with_fresh_thickness_and_containment',
                     original_plan_superseded=True, exact_complete_child_cutter_required=True)
                for loop_index in sorted(decisions)]
    if not decisions:
        return []
    extensions = {
        int(record["loop_index"]): record
        for record in child_stats.get("loop_extensions", [])
    }
    validations: list[dict] = []
    for loop_index, decision in sorted(decisions.items()):
        extension = extensions.get(int(loop_index))
        if extension is None:
            raise ValueError(
                f"P{int(child_index):02d} did not build planned cap loop "
                f"{int(loop_index)}"
            )
        actual_mode = str(extension.get("cap_mode"))
        expected_mode = (
            "local-connector"
            if interface_geometry == "local-connector"
            else str(decision.mode)
        )
        if actual_mode != expected_mode:
            raise ValueError(
                f"P{int(child_index):02d} cap mode changed after planning on "
                f"loop {int(loop_index)}: planned={decision.mode}, "
                f"actual={actual_mode}"
            )
        planned_distances = np.asarray(decision.distances, dtype=np.float64)
        planned_minimum = float(planned_distances.min())
        planned_maximum = float(planned_distances.max())
        actual_minimum = float(extension["extension_min_mm"])
        actual_maximum = float(extension["extension_max_mm"])
        if interface_geometry == "local-connector":
            if (
                bool(extension.get("preview_not_printable", False))
                and os.environ.get("SPLIT3MF_FORCE_PREVIEW", "") == "1"
            ):
                validations.append(
                    {
                        "loop_index": int(loop_index),
                        "cap_mode": actual_mode,
                        "preview_not_printable": True,
                        "planned_safe_minimum_mm": planned_minimum,
                        "planned_safe_maximum_mm": planned_maximum,
                        "validation": "forced_preview_connector_skip",
                    }
                )
                continue
            planned_connector_safety = local_connector_safe_depth_from_field(
                planned_distances
            )
            planned_connector_budget = float(
                planned_connector_safety["local_connector_safety_budget_mm"]
            )
            actual_connector_budget = float(
                extension.get(
                    "local_connector_safety_budget_mm",
                    planned_connector_budget,
                )
            )
            footprint_probe_applied = bool(
                extension.get("local_connector_footprint_probe_applied", False)
            )
            connector_budget_valid = (
                0.0
                < actual_connector_budget
                <= float(DEFAULT_DEPTH_POLICY.maximum_total_depth_mm) + 1e-9
                if footprint_probe_applied
                else np.isclose(
                    actual_connector_budget,
                    planned_connector_budget,
                    atol=1e-9,
                    rtol=0.0,
                )
            )
            backing_depth = float(
                extension.get("full_boundary_backing_depth_mm", 0.0)
            )
            elastic_shrink_applied = bool(
                extension.get("elastic_shrink_applied", False)
            )
            minimum_backing_depth = float(
                extension.get("minimum_elastic_backing_depth_mm", 3.0)
                if elastic_shrink_applied
                else 3.0
            )
            socket_depth = float(
                extension.get("socket_depth_mm", actual_maximum)
            )
            generated_total_depth = backing_depth + socket_depth
            if not (
                np.isclose(actual_minimum, actual_maximum, atol=1e-9, rtol=0.0)
                and backing_depth >= minimum_backing_depth - 1e-9
                and connector_budget_valid
                and generated_total_depth <= actual_connector_budget + 1e-9
                and generated_total_depth
                <= float(DEFAULT_DEPTH_POLICY.maximum_total_depth_mm) + 1e-9
            ):
                raise ValueError(
                    f"P{int(child_index):02d} local connector depth changed after "
                    f"planning on loop {int(loop_index)}"
                )
            validations.append(
                {
                    "loop_index": int(loop_index),
                    "cap_mode": actual_mode,
                    "minimum_distance_mm": actual_minimum,
                    "maximum_distance_mm": actual_maximum,
                    "planned_safe_minimum_mm": planned_minimum,
                    "planned_local_connector_budget_mm": planned_connector_budget,
                    "actual_local_connector_budget_mm": actual_connector_budget,
                    "footprint_probe_applied": footprint_probe_applied,
                    "printable_backing_depth_mm": backing_depth,
                    "elastic_shrink_applied": elastic_shrink_applied,
                    "elastic_backing_scale": float(
                        extension.get("elastic_backing_scale", 1.0)
                    ),
                    "elastic_lateral_scale": float(
                        extension.get("elastic_lateral_scale", 1.0)
                    ),
                    "generated_total_depth_mm": generated_total_depth,
                    "status": "local_connector_consumed_shared_safety_budget",
                }
            )
            continue
        if not (
            np.isclose(actual_minimum, planned_minimum, atol=1e-9, rtol=0.0)
            and np.isclose(actual_maximum, planned_maximum, atol=1e-9, rtol=0.0)
        ):
            raise ValueError(
                f"P{int(child_index):02d} cap distance field changed after "
                f"planning on loop {int(loop_index)}"
            )
        validations.append(
            {
                "loop_index": int(loop_index),
                "cap_mode": actual_mode,
                "minimum_distance_mm": actual_minimum,
                "maximum_distance_mm": actual_maximum,
                "status": "shared_cap_decision_reused",
            }
        )
    return validations


def export_recursive_layer_stage_outputs(
    output_dir: Path,
    report_path_mode: str,
    input_stem: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: list[str],
    components: list[Component],
    assembly_parents: dict[int, int | None],
    assembly_children: dict[int, list[int]],
    recursive_minimal_layers: list[dict],
    boundary_neighbor_lookup: dict[tuple[int, int], set[int]],
    component_centers: dict[int, np.ndarray],
    inward_overrides: dict[int, np.ndarray],
    model_center: np.ndarray,
    max_extension_mm: float,
    interface_retopology: PlanarArcRetopologyContext,
    flat_clearance_mm: float,
    fit_clearance_mm: float,
    insert_shrink_mm: float,
    lead_in_mm: float,
    top_edge_clearance_mm: float,
    clearance_mode: str,
    sibling_clearance_mm: float,
    socket_overcut_mm: float,
    bottom_clearance_mm: float,
    planar_extra_limit_mm: float,
    effective_cap_mode,
    layer_child_context,
    source_application: str | None = None,
    layer_order_offset: int = 0,
) -> tuple[Path, list[dict]]:
    layers_dir = output_dir / "recursive_layers"
    layers_dir.mkdir(parents=True, exist_ok=True)
    stage_records: list[dict] = []

    for inward_layer_order, layer in enumerate(recursive_minimal_layers):
        layer_order = int(layer_order_offset + inward_layer_order)
        local_body_index = int(layer["local_body_index"])
        direct_children = [int(child) for child in layer.get("direct_child_indices", [])]
        layer_dir = layers_dir / f"layer_{layer_order:02d}_INWARD_P{local_body_index:02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        layer_3mf_parts: list[dict] = []
        child_refs, union_by_child, subtree_by_child = layer_child_context(local_body_index)
        local_component = components[local_body_index - 1]
        color_info = COLOR_INFO.get(local_component.color_code, {"name": local_component.color_code, "hex": "", "rgba": [200, 200, 200, 255]})
        local_role = "root_body" if assembly_parents.get(local_body_index) is None else "local_body"
        local_part_id = f"L{layer_order:02d}_P{local_body_index:02d}_{sanitize_name(color_info['name'])}_{local_role.upper()}_CUT"
        part_cap_mode = effective_cap_mode(local_body_index)
        if assembly_parents.get(local_body_index) is None:
            mesh, stats = make_body_cut_mesh(
                vertices=vertices,
                faces=faces,
                body_component=local_component,
                part_id=local_part_id,
                cut_refs=child_refs,
                max_extension_mm=max_extension_mm,
                interface_retopology=interface_retopology,
                flat_clearance_mm=flat_clearance_mm,
                fit_clearance_mm=fit_clearance_mm,
                socket_overcut_mm=socket_overcut_mm,
                bottom_clearance_mm=bottom_clearance_mm,
                clearance_mode=clearance_mode,
                model_center=model_center,
                cap_mode=part_cap_mode,
                planar_extra_limit_mm=planar_extra_limit_mm,
                lead_in_mm=lead_in_mm,
            )
        else:
            mesh, stats = make_part_mesh(
                vertices=vertices,
                faces=faces,
                component=local_component,
                component_index=local_body_index,
                assembly_parent_index=assembly_parents.get(local_body_index),
                part_id=local_part_id,
                max_extension_mm=max_extension_mm,
                interface_retopology=interface_retopology,
                flat_clearance_mm=flat_clearance_mm,
                fit_clearance_mm=fit_clearance_mm,
                insert_shrink_mm=insert_shrink_mm,
                lead_in_mm=lead_in_mm,
                top_edge_clearance_mm=top_edge_clearance_mm,
                clearance_mode=clearance_mode,
                child_cut_refs=child_refs,
                boundary_neighbor_lookup=boundary_neighbor_lookup,
                component_centers=component_centers,
                sibling_clearance_mm=sibling_clearance_mm,
                socket_overcut_mm=socket_overcut_mm,
                bottom_clearance_mm=bottom_clearance_mm,
                inward_override=inward_overrides.get(local_body_index),
                model_center=model_center,
                cap_mode=part_cap_mode,
                planar_extra_limit_mm=planar_extra_limit_mm,
                parent_contact_only=True,
            )
        local_path = layer_dir / f"{local_part_id}.stl"
        mesh.export(local_path)
        stats["file"] = serialize_report_path(local_path, output_dir, report_path_mode)
        stats.update(validate_exported_mesh(local_path))
        stats["stage_layer_order"] = layer_order
        stats["stage_depth"] = int(layer.get("depth", 0))
        stats["stage_local_body_index"] = local_body_index
        stats["stage_role"] = local_role
        stats["stage_direct_child_indices"] = direct_children
        stats["stage_contains"] = [local_body_index]
        stage_records.append(stats)
        layer_3mf_parts.append(
            {
                "part_id": local_part_id,
                "mesh": mesh,
                "color_code": local_component.color_code,
                "color_name": color_info.get("name", local_component.color_code),
                "color_hex": color_info.get("hex", "#C8C8C8"),
                "annotation": {
                    "debug_recursive_layer": layer_order,
                    "stage_role": local_role,
                    "source_part_index": local_body_index,
                    "unit": "millimeter",
                },
            }
        )

        for child_index in direct_children:
            subtree = subtree_by_child.get(child_index) or subtree_component_indices(child_index, assembly_children)
            union_component = union_by_child.get(child_index) or build_subassembly_component(
                vertices,
                faces,
                components,
                subtree,
                components[child_index - 1].color_code,
            )
            child_color = COLOR_INFO.get(union_component.color_code, {"name": union_component.color_code})
            child_part_id = f"L{layer_order:02d}_P{child_index:02d}_{sanitize_name(child_color['name'])}_SUBASSEMBLY"
            child_cap_mode = effective_cap_mode(child_index)
            child_cap_decisions = shared_child_cap_decisions(
                child_refs,
                child_index,
            )
            child_mesh, child_stats = make_layer_child_subassembly_mesh(
                vertices=vertices,
                faces=faces,
                source_colors=colors,
                components=components,
                component=union_component,
                root_child_index=child_index,
                subtree_indices=subtree,
                parent_index=local_body_index,
                part_id=child_part_id,
                max_extension_mm=max_extension_mm,
                interface_retopology=interface_retopology,
                flat_clearance_mm=flat_clearance_mm,
                fit_clearance_mm=fit_clearance_mm,
                insert_shrink_mm=insert_shrink_mm,
                lead_in_mm=lead_in_mm,
                top_edge_clearance_mm=top_edge_clearance_mm,
                boundary_neighbor_lookup=boundary_neighbor_lookup,
                inward_override=inward_overrides.get(child_index),
                model_center=model_center,
                cap_mode=child_cap_mode,
                planar_extra_limit_mm=planar_extra_limit_mm,
                cap_decisions_by_loop=child_cap_decisions,
            )
            child_stats["shared_cap_decision_validation"] = (
                validate_shared_child_cap_decisions(
                    child_index,
                    child_cap_decisions,
                    child_stats,
                )
            )
            child_path = layer_dir / f"{child_part_id}.stl"
            child_mesh.export(child_path)
            child_stats["file"] = serialize_report_path(child_path, output_dir, report_path_mode)
            child_stats.update(validate_exported_mesh(child_path))
            child_stats["stage_layer_order"] = layer_order
            child_stats["stage_depth"] = int(layer.get("depth", 0))
            child_stats["stage_local_body_index"] = local_body_index
            child_stats["stage_role"] = "direct_child_leaf" if not assembly_children.get(child_index) else "direct_child_subassembly"
            child_stats["stage_direct_child_indices"] = []
            child_stats["stage_contains"] = [int(index) for index in subtree]
            stage_records.append(child_stats)
            layer_3mf_parts.append(
                {
                    "part_id": child_part_id,
                    "mesh": child_mesh,
                    "color_code": union_component.color_code,
                    "color_name": child_color.get("name", union_component.color_code),
                    "color_hex": child_color.get("hex", "#C8C8C8"),
                    "annotation": {
                        "debug_recursive_layer": layer_order,
                        "stage_role": child_stats["stage_role"],
                        "source_part_index": child_index,
                        "contains": child_stats["stage_contains"],
                        "unit": "millimeter",
                    },
                }
            )

        layer_package_path = layer_dir / f"layer_{layer_order:02d}_INWARD_P{local_body_index:02d}_mm.3mf"
        export_colored_parts_3mf(
            layer_package_path,
            layer_3mf_parts,
            title=f"{input_stem} recursive layer {layer_order:02d} (millimeter)",
            source_application=source_application,
        )
        for layer_part_stats in stage_records[-len(layer_3mf_parts):]:
            layer_part_stats["layer_3mf"] = serialize_report_path(layer_package_path, output_dir, report_path_mode)

    return layers_dir, stage_records


def cumulative_snapshot_parts(
    active_parts: dict[int, dict],
    step_order: int,
    transition: dict,
) -> list[dict]:
    snapshot_parts = []
    for _active_index, entry in sorted(active_parts.items()):
        snapshot_part = {
            key: value
            for key, value in entry.items()
            if key
            not in {
                "stats",
                "source_part_index",
                "contains",
                "state_role",
                "origin_step",
            }
        }
        snapshot_part["annotation"] = {
            **entry.get("annotation", {}),
            "strict_recursive_snapshot_step": int(step_order),
            "snapshot_active_indices": transition["after_active_indices"],
            "snapshot_transition": transition,
        }
        snapshot_parts.append(snapshot_part)
    return snapshot_parts


def recursive_face_color_payload(
    mesh,
    default_color_code: str,
    default_color_hex: str,
    default_filament_slot_index: int | None,
) -> tuple[list[str], list[int | None]]:
    """Preserve the material meaning of every triangle in a recursive 3MF."""
    face_count = int(len(mesh.faces))
    face_codes = list(mesh.metadata.get("face_color_codes", []))
    face_slots = list(mesh.metadata.get("face_filament_slot_indices", []))
    if len(face_codes) == face_count:
        face_hexes = [
            COLOR_INFO.get(code, {}).get("hex", default_color_hex)
            for code in face_codes
        ]
        if len(face_slots) != face_count:
            face_slots = [
                COLOR_INFO.get(code, {}).get("filament_slot")
                for code in face_codes
            ]
        return face_hexes, [
            None if slot is None else int(slot) for slot in face_slots
        ]
    return (
        [default_color_hex for _ in range(face_count)],
        [default_filament_slot_index for _ in range(face_count)],
    )


def recursive_face_paint_token_payload(
    mesh,
    default_color_code: str,
) -> list[str]:
    """Return Bambu paint tokens for every recursive triangle."""
    face_count = int(len(mesh.faces))
    face_codes = list(mesh.metadata.get("face_color_codes", []))
    if len(face_codes) == face_count:
        return [str(code) for code in face_codes]
    return [str(default_color_code) for _ in range(face_count)]


def recursive_part_package_payload(entry: dict) -> dict:
    return {
        key: value
        for key, value in entry.items()
        if key
        not in {
            "stats",
            "source_part_index",
            "contains",
            "state_role",
            "origin_step",
            "source_3mf_path",
            "source_3mf_origin_step",
            "source_3mf_sha256",
        }
    }


def recursive_artifact_sha256(path: Path) -> str:
    """Return the identity of one on-disk parent-emitted recursive package."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_loaded_recursive_part(
    loaded: list[dict],
    expected_entry: dict,
    path: Path,
) -> dict:
    """Validate one already parsed parent-emitted package object."""
    if len(loaded) != 1:
        raise ValueError(
            f"recursive input {path} must contain exactly one mesh object; "
            f"found {len(loaded)}"
        )
    result = loaded[0]
    annotation = result.get("annotation", {})
    checks = {
        "part_id": str(expected_entry["part_id"]),
        "source_part_index": int(expected_entry["source_part_index"]),
        "state_role": str(expected_entry["state_role"]),
        "contains": [int(index) for index in expected_entry["contains"]],
        "origin_step": int(expected_entry["origin_step"]),
    }
    actual = {
        "part_id": str(result.get("part_id", "")),
        "source_part_index": int(annotation.get("source_part_index", -1)),
        "state_role": str(annotation.get("state_role", "")),
        "contains": [int(index) for index in annotation.get("contains", [])],
        "origin_step": int(annotation.get("origin_step", -1)),
    }
    if actual != checks:
        raise ValueError(
            f"recursive input provenance mismatch for {path}: "
            f"expected {checks}, found {actual}"
        )
    expected_face_colors = [
        normalize_3mf_color(value)
        for value in expected_entry.get("face_color_hexes", [])
    ]
    actual_face_colors = [
        normalize_3mf_color(value)
        for value in result.get("face_color_hexes", [])
    ]
    if actual_face_colors != expected_face_colors:
        raise ValueError(f"recursive input face colors changed in {path}")
    expected_slots = [
        None if value is None else int(value)
        for value in expected_entry.get("face_filament_slot_indices", [])
    ]
    actual_slots = [
        None if value is None else int(value)
        for value in result.get("face_filament_slot_indices", [])
    ]
    if actual_slots != expected_slots:
        raise ValueError(f"recursive input filament-slot meaning changed in {path}")
    expected_paint_tokens = [
        str(value)
        for value in expected_entry.get("face_paint_color_tokens", [])
    ]
    actual_paint_tokens = [
        str(value)
        for value in result.get("face_paint_color_tokens", [])
    ]
    if expected_paint_tokens and actual_paint_tokens != expected_paint_tokens:
        raise ValueError(f"recursive input Bambu paint tokens changed in {path}")
    return result


def reload_recursive_part_input(
    path: Path,
    expected_entry: dict,
    *,
    require_artifact_identity: bool = False,
) -> dict:
    """Reload exactly one parent-emitted colored 3MF and verify its provenance.

    Strict recursive consumption requires the exact standalone file emitted by
    the parent step. The checksum check runs before parsing so a replaced,
    edited, or redirected package cannot silently become the next input.
    """
    path = Path(path)
    if require_artifact_identity:
        validate_recursive_artifact_identity(path, expected_entry)
    loaded = load_colored_mesh_objects_3mf(path)
    return validate_loaded_recursive_part(loaded, expected_entry, path)


def validate_recursive_artifact_identity(path: Path, expected_entry: dict) -> str:
    path = Path(path)
    if "CUMULATIVE" in path.name.upper():
        raise ValueError(
            f"recursive input must be a parent-emitted standalone 3MF, not {path.name}"
        )
    expected_path_text = str(expected_entry.get("source_3mf_path", ""))
    expected_sha256 = str(expected_entry.get("source_3mf_sha256", ""))
    if not expected_path_text or not expected_sha256:
        raise ValueError(
            f"recursive input identity is missing for {expected_entry.get('part_id')}"
        )
    expected_path = Path(expected_path_text)
    if path.resolve(strict=True) != expected_path.resolve(strict=True):
        raise ValueError(
            f"recursive input path changed for {expected_entry.get('part_id')}: "
            f"expected {expected_path}, found {path}"
        )
    actual_sha256 = recursive_artifact_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"recursive input SHA-256 changed for {expected_entry.get('part_id')}: "
            f"expected {expected_sha256}, found {actual_sha256}"
        )
    return actual_sha256


class VerifiedRecursiveArtifactCache:
    """Reuse an exact 3MF parse only after path, hash, and provenance checks."""

    def __init__(self, metrics: dict | None = None) -> None:
        self._loaded: dict[tuple[str, str], dict] = {}
        self.metrics = metrics if metrics is not None else {}

    def remember(self, path: Path, sha256: str, loaded: dict) -> None:
        key = (str(Path(path).resolve(strict=True)), str(sha256))
        self._loaded[key] = loaded

    def load(self, path: Path, expected_entry: dict, loader=None) -> dict:
        path = Path(path)
        sha256 = validate_recursive_artifact_identity(path, expected_entry)
        key = (str(path.resolve(strict=True)), sha256)
        if key in self._loaded:
            self.metrics["verified_recursive_parse_hits"] = int(
                self.metrics.get("verified_recursive_parse_hits", 0)
            ) + 1
            return validate_loaded_recursive_part(
                [self._loaded[key]], expected_entry, path
            )
        self.metrics["verified_recursive_parse_misses"] = int(
            self.metrics.get("verified_recursive_parse_misses", 0)
        ) + 1
        if loader is None:
            loaded = load_colored_mesh_objects_3mf(path)
            result = validate_loaded_recursive_part(
                loaded, expected_entry, path
            )
        else:
            result = loader()
        self._loaded[key] = result
        return result


def recursive_component_material_key(component: Component) -> tuple[str, object]:
    """Return the serialized material identity expected for one component."""
    info = COLOR_INFO.get(str(component.color_code), {})
    slot = info.get("filament_slot")
    if slot is not None:
        return ("filament_slot", int(slot))
    return ("color_hex", normalize_3mf_color(info.get("hex", "")))


def recursive_input_geometry_context(
    reloaded_target: dict,
    expected_indices: list[int],
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    source_components: list[Component],
    interface_retopology: PlanarArcRetopologyContext,
) -> dict:
    """Rebuild the next recursive step entirely from its parent-emitted 3MF.

    Component ids remain the ids inferred from the original assembly tree, but
    their face ids, boundary graph, normals, centers, and model center are all
    rebound to the mesh that was serialized and reloaded by the parent step.
    """
    mesh = reloaded_target["mesh"]
    local_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    local_faces = np.asarray(mesh.faces, dtype=np.int64)
    face_hexes = [
        normalize_3mf_color(value)
        for value in reloaded_target.get("face_color_hexes", [])
    ]
    face_slots = [
        None if value is None else int(value)
        for value in reloaded_target.get("face_filament_slot_indices", [])
    ]
    if len(face_hexes) != len(local_faces) or len(face_slots) != len(local_faces):
        raise ValueError(
            "recursive input geometry cannot be rebound because its face-material "
            "payload does not match its face count"
        )

    expected_indices = sorted({int(index) for index in expected_indices})
    expected_keys = {
        recursive_component_material_key(source_components[index - 1])
        for index in expected_indices
    }
    face_material_keys: list[tuple[str, object]] = []
    for face_hex, face_slot in zip(face_hexes, face_slots):
        slot_key = (
            None
            if face_slot is None
            else ("filament_slot", int(face_slot))
        )
        hex_key = ("color_hex", face_hex)
        if slot_key in expected_keys:
            face_material_keys.append(slot_key)
        elif hex_key in expected_keys:
            face_material_keys.append(hex_key)
        else:
            raise ValueError(
                "recursive input contains a face material that is absent from "
                f"its expected descendants: slot={face_slot}, color={face_hex}"
            )

    material_labels = [
        f"{key[0]}:{key[1]}" for key in face_material_keys
    ]
    connected_regions = connected_components_by_color(local_faces, material_labels)
    local_areas = triangle_areas(local_vertices, local_faces)
    candidates_by_key: dict[tuple[str, object], list[Component]] = (
        collections.defaultdict(list)
    )
    for region in connected_regions:
        material_key = face_material_keys[int(region[0])]
        candidates_by_key[material_key].append(
            make_component_from_global_faces(
                local_vertices,
                local_faces,
                local_areas,
                region,
                str(material_key),
            )
        )

    source_extent = np.asarray(source_vertices, dtype=np.float64)
    source_scale = max(
        float(
            np.linalg.norm(
                source_extent.max(axis=0) - source_extent.min(axis=0)
            )
        ),
        1e-9,
    )

    def match_cost(source_component: Component, candidate: Component) -> float:
        center_cost = float(
            np.linalg.norm(candidate.center - source_component.center)
        ) / source_scale
        bbox_cost = float(
            np.linalg.norm(candidate.bbox_min - source_component.bbox_min)
            + np.linalg.norm(candidate.bbox_max - source_component.bbox_max)
        ) / (2.0 * source_scale)
        face_cost = abs(
            math.log(
                (float(candidate.face_count) + 1.0)
                / (float(source_component.face_count) + 1.0)
            )
        )
        return center_cost + 0.25 * bbox_cost + 0.02 * face_cost

    assigned_regions: dict[int, list[np.ndarray]] = collections.defaultdict(list)
    mapping_records: list[dict] = []
    for material_key in sorted(expected_keys, key=lambda value: (value[0], str(value[1]))):
        material_expected = [
            index
            for index in expected_indices
            if recursive_component_material_key(source_components[index - 1])
            == material_key
        ]
        material_candidates = candidates_by_key.get(material_key, [])
        if len(material_candidates) < len(material_expected):
            raise ValueError(
                "recursive input lost a connected material component for "
                f"{material_key}: expected {len(material_expected)}, "
                f"found {len(material_candidates)}"
            )
        costs = np.asarray(
            [
                [
                    match_cost(source_components[index - 1], candidate)
                    for candidate in material_candidates
                ]
                for index in material_expected
            ],
            dtype=np.float64,
        )
        from scipy.optimize import linear_sum_assignment

        expected_rows, candidate_columns = linear_sum_assignment(costs)
        claimed_candidates: set[int] = set()
        for row, column in zip(expected_rows, candidate_columns):
            component_index = int(material_expected[int(row)])
            candidate_index = int(column)
            claimed_candidates.add(candidate_index)
            candidate = material_candidates[candidate_index]
            assigned_regions[component_index].append(candidate.global_faces)
            mapping_records.append(
                {
                    "source_part_index": component_index,
                    "material_key": [material_key[0], material_key[1]],
                    "local_faces": int(candidate.face_count),
                    "match_cost": float(costs[int(row), candidate_index]),
                    "primary_region": True,
                }
            )
        for candidate_index, candidate in enumerate(material_candidates):
            if candidate_index in claimed_candidates:
                continue
            nearest_row = int(np.argmin(costs[:, candidate_index]))
            component_index = int(material_expected[nearest_row])
            assigned_regions[component_index].append(candidate.global_faces)
            mapping_records.append(
                {
                    "source_part_index": component_index,
                    "material_key": [material_key[0], material_key[1]],
                    "local_faces": int(candidate.face_count),
                    "match_cost": float(costs[nearest_row, candidate_index]),
                    "primary_region": False,
                }
            )

    zero = np.zeros(3, dtype=np.float64)
    local_components = [
        Component(
            color_code=component.color_code,
            global_faces=np.empty(0, dtype=np.int64),
            face_count=0,
            area=0.0,
            bbox_min=zero.copy(),
            bbox_max=zero.copy(),
            center=zero.copy(),
        )
        for component in source_components
    ]
    local_colors = ["" for _ in range(len(local_faces))]
    assigned_face_count = 0
    for component_index in expected_indices:
        regions = assigned_regions.get(component_index, [])
        if not regions:
            raise ValueError(
                f"recursive input did not materialize expected P{component_index:02d}"
            )
        global_faces = np.unique(np.concatenate(regions)).astype(np.int64)
        source_component = source_components[component_index - 1]
        local_component = make_component_from_global_faces(
            local_vertices,
            local_faces,
            local_areas,
            global_faces,
            source_component.color_code,
        )
        local_components[component_index - 1] = local_component
        for face_index in global_faces:
            if local_colors[int(face_index)]:
                raise ValueError(
                    f"recursive input face {int(face_index)} mapped more than once"
                )
            local_colors[int(face_index)] = source_component.color_code
            assigned_face_count += 1
    if assigned_face_count != len(local_faces) or any(
        not color for color in local_colors
    ):
        raise ValueError(
            "recursive input geometry rebind did not assign every serialized face"
        )

    local_boundary_neighbor_lookup = component_boundary_neighbor_lookup(
        local_faces,
        local_components,
    )
    return {
        "vertices": local_vertices,
        "faces": local_faces,
        "colors": local_colors,
        "components": local_components,
        "boundary_neighbor_lookup": local_boundary_neighbor_lookup,
        "component_centers": {
            index: local_components[index - 1].center
            for index in expected_indices
        },
        "model_center": local_vertices.mean(axis=0),
        "interface_retopology": PlanarArcRetopologyContext(
            config=interface_retopology.config,
            curve_review_sink=interface_retopology.curve_review_sink,
        ),
        "mapping_records": mapping_records,
        "source_vertex_count": int(len(source_vertices)),
        "source_face_count": int(len(source_faces)),
        "reloaded_vertex_count": int(len(local_vertices)),
        "reloaded_face_count": int(len(local_faces)),
    }


def execute_strict_recursive_split(
    output_dir: Path | None,
    report_path_mode: str,
    input_stem: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: list[str],
    components: list[Component],
    assembly_parents: dict[int, int | None],
    assembly_children: dict[int, list[int]],
    recursive_steps: list[dict],
    boundary_neighbor_lookup: dict[tuple[int, int], set[int]],
    component_centers: dict[int, np.ndarray],
    inward_overrides: dict[int, np.ndarray],
    model_center: np.ndarray,
    max_extension_mm: float,
    interface_retopology: PlanarArcRetopologyContext,
    flat_clearance_mm: float,
    fit_clearance_by_part: dict[int, float],
    boundary_reconciliation_tolerance_mm: float,
    lead_in_mm: float,
    clearance_mode: str,
    sibling_clearance_mm: float,
    bottom_clearance_mm: float,
    effective_cap_mode,
    effective_planar_extra_limit,
    layer_child_context,
    validation_profile: str,
    max_topology_defect_ratio: float,
    source_application: str | None = None,
    source_filament_colors: list[str] | None = None,
    source_project_settings: dict | None = None,
    interface_geometry: str = "boundary-extrusion",
    allow_partial: bool = False,
    stage_cache: RecursiveStageCache | None = None,
    run_fingerprint: str = "",
    stage_implementation_fingerprint: str = "",
    source_artifact_sha256: str = "",
    optimization_metrics: dict | None = None,
    boundary_review=None,
) -> tuple[Path | None, list[dict], dict[int, dict], list[dict]]:
    temporary_artifacts = (
        tempfile.TemporaryDirectory(prefix="strict-recursive-3mf-")
        if output_dir is None
        else None
    )
    artifact_root = (
        Path(temporary_artifacts.name)
        if temporary_artifacts is not None
        else output_dir
    )
    artifact_layers_dir = artifact_root / "recursive_layers"
    artifact_layers_dir.mkdir(parents=True, exist_ok=True)
    layers_dir = artifact_layers_dir if output_dir is not None else None

    active_parts: dict[int, dict] = {}
    stage_records: list[dict] = []
    snapshot_records: list[dict] = []
    optimization_metrics = (
        optimization_metrics if optimization_metrics is not None else {}
    )
    verified_artifacts = VerifiedRecursiveArtifactCache(optimization_metrics)
    expected_internal_indices = {
        int(index) for index, child_indices in assembly_children.items() if child_indices
    }
    executed_internal_indices: set[int] = set()
    runtime_log(
        "递归拆件",
        "recursive_executor_start",
        "严格串行递归执行器已启动",
        recursive_step_count=int(len(recursive_steps)),
        source_component_count=int(len(components)),
        artifact_root=str(artifact_root),
        debug_output_enabled=bool(output_dir is not None),
    )

    def part_color(component_index: int) -> dict:
        component = components[int(component_index) - 1]
        return COLOR_INFO.get(
            component.color_code,
            {
                "name": component.color_code,
                "hex": "#C8C8C8",
                "rgba": [200, 200, 200, 255],
                "filament_slot": None,
            },
        )

    def active_part(
        component_index: int,
        part_id: str,
        mesh,
        stats: dict,
        state_role: str,
        contains: list[int],
        origin_step: int,
    ) -> dict:
        component = components[int(component_index) - 1]
        color_info = part_color(component_index)
        face_color_hexes, face_filament_slot_indices = recursive_face_color_payload(
            mesh,
            component.color_code,
            color_info.get("hex", "#C8C8C8"),
            color_info.get("filament_slot"),
        )
        face_paint_color_tokens = recursive_face_paint_token_payload(
            mesh,
            component.color_code,
        )
        return {
            "part_id": part_id,
            "mesh": mesh,
            "color_code": component.color_code,
            "color_name": color_info.get("name", component.color_code),
            "color_hex": color_info.get("hex", "#C8C8C8"),
            "filament_slot_index": color_info.get("filament_slot"),
            "face_color_hexes": face_color_hexes,
            "face_filament_slot_indices": face_filament_slot_indices,
            "face_paint_color_tokens": face_paint_color_tokens,
            "source_part_index": int(component_index),
            "contains": [int(index) for index in contains],
            "state_role": state_role,
            "origin_step": int(origin_step),
            "stats": stats,
            "annotation": {
                "strict_serial_recursive": True,
                "state_role": state_role,
                "source_part_index": int(component_index),
                "contains": [int(index) for index in contains],
                "origin_step": int(origin_step),
                "unit": "millimeter",
            },
        }

    def validate_changed_part(
        layer_dir: Path | None,
        part_id: str,
        mesh,
        stats: dict,
        step_order: int,
        step: dict,
        state_role: str,
        contains: list[int],
    ) -> dict:
        validation = validate_mesh_in_memory(mesh)
        stats.update(
            {
                "reload_watertight": bool(validation["watertight"]),
                "reload_open_edges": int(validation["open_edges"]),
                "reload_over_shared_edges": int(validation["over_shared_edges"]),
                "reload_winding_consistent": bool(validation["winding_consistent"]),
                "reload_inconsistent_shared_edges": int(
                    validation["inconsistent_shared_edges"]
                ),
                "reload_topology_defect_ratio": float(
                    validation["topology_defect_ratio"]
                ),
            }
        )
        stats["stage_layer_order"] = int(step_order)
        stats["stage_depth"] = int(step.get("depth", 0))
        stats["stage_local_body_index"] = int(step["local_body_index"])
        stats["stage_role"] = state_role
        stats["stage_direct_child_indices"] = [
            int(index) for index in step.get("direct_child_indices", [])
        ]
        stats["stage_contains"] = [int(index) for index in contains]
        return stats

    def materialize_changed_part(
        entry: dict,
        layer_dir: Path,
        step_order: int,
    ) -> dict:
        package_path = layer_dir / f"{entry['part_id']}_mm.3mf"
        package_part = recursive_part_package_payload(entry)
        runtime_log(
            "递归落盘",
            "standalone_write_start",
            "正在写入本阶段独立零件 3MF",
            step_order=int(step_order),
            part_id=str(entry["part_id"]),
            state_role=str(entry["state_role"]),
            face_count=int(len(entry["mesh"].faces)),
            output_3mf=str(package_path),
        )
        export_colored_parts_3mf(
            package_path,
            [package_part],
            title=f"{input_stem} recursive part {entry['part_id']}",
            source_application=source_application,
            source_filament_colors=source_filament_colors,
            source_project_settings=source_project_settings,
            output_layout="separate-items",
        )
        package_validation = validate_colored_parts_3mf(
            package_path,
            [package_part],
            source_filament_colors=source_filament_colors,
            source_application=source_application,
            source_project_settings=source_project_settings,
            output_layout="separate-items",
            include_loaded_objects=True,
        )
        loaded_objects = package_validation.pop("_loaded_objects", [])
        package_errors = [
            error
            for error in package_validation.get("errors", [])
            if "reloaded mesh is not watertight" not in error
            and not (
                "reloaded mesh has " in error
                and ("open edges" in error or "over-shared edges" in error)
            )
        ]
        if package_errors:
            raise ValueError(
                f"recursive part package validation failed for {entry['part_id']}: "
                + "; ".join(package_errors)
            )
        entry["source_3mf_path"] = str(package_path)
        entry["source_3mf_origin_step"] = int(step_order)
        entry["source_3mf_sha256"] = recursive_artifact_sha256(package_path)
        reloaded = validate_loaded_recursive_part(
            loaded_objects,
            entry,
            package_path,
        )
        verified_artifacts.remember(
            package_path,
            entry["source_3mf_sha256"],
            reloaded,
        )
        entry["mesh"] = reloaded["mesh"]
        entry["face_color_hexes"] = reloaded["face_color_hexes"]
        entry["face_filament_slot_indices"] = reloaded[
            "face_filament_slot_indices"
        ]
        entry["face_paint_color_tokens"] = reloaded[
            "face_paint_color_tokens"
        ]
        entry["stats"]["debug_format"] = "3mf"
        report_package_path = (
            serialize_report_path(package_path, output_dir, report_path_mode)
            if output_dir is not None
            else None
        )
        entry["stats"]["file"] = report_package_path
        entry["stats"]["recursive_output_3mf"] = report_package_path
        entry["stats"]["recursive_output_3mf_sha256"] = entry[
            "source_3mf_sha256"
        ]
        entry["stats"]["recursive_color_palette"] = reloaded["palette"]
        entry["stats"]["recursive_face_color_indices"] = sorted(
            set(int(index) for index in reloaded["face_color_indices"])
        )
        runtime_log(
            "递归落盘",
            "standalone_write_done",
            "本阶段独立零件 3MF 已写入、校验并重载",
            step_order=int(step_order),
            part_id=str(entry["part_id"]),
            state_role=str(entry["state_role"]),
            face_count=int(len(entry["mesh"].faces)),
            output_3mf=str(package_path),
            sha256=str(entry["source_3mf_sha256"]),
        )
        return entry

    def restore_cached_stage(cache_hit: dict) -> tuple[list[dict], list[dict]]:
        stage_dir = Path(cache_hit["stage_dir"])
        manifest = cache_hit["manifest"]
        restored_entries: list[dict] = []
        for cached_record in manifest.get("entries", []):
            entry = dict(cached_record["entry"])
            artifact_path = stage_dir / str(cached_record["artifact"])
            entry["source_3mf_path"] = str(artifact_path)
            entry["source_3mf_sha256"] = str(cached_record["sha256"])
            reloaded = verified_artifacts.load(artifact_path, entry)
            entry["mesh"] = reloaded["mesh"]
            entry["face_color_hexes"] = reloaded["face_color_hexes"]
            entry["face_filament_slot_indices"] = reloaded[
                "face_filament_slot_indices"
            ]
            entry["face_paint_color_tokens"] = reloaded[
                "face_paint_color_tokens"
            ]
            validation = validate_mesh_in_memory(entry["mesh"])
            if (
                not validation["winding_consistent"]
                or validation["inconsistent_shared_edges"]
                or float(validation["topology_defect_ratio"])
                > float(max_topology_defect_ratio)
            ):
                raise ValueError(
                    f"cached recursive part failed topology validation: {entry['part_id']}"
                )
            restored_entries.append(entry)
        restored_records = [
            {**dict(record), "stage_cache_hit": True}
            for record in manifest.get("stage_records", [])
        ]
        return restored_entries, restored_records

    for step_order, step in enumerate(recursive_steps):
        local_body_index = int(step["local_body_index"])
        direct_children = [
            int(child) for child in step.get("direct_child_indices", [])
        ]
        runtime_log(
            "递归拆件",
            "recursive_step_start",
            "开始递归拆件步骤",
            step_order=int(step_order),
            step_number=int(step_order + 1),
            total_steps=int(len(recursive_steps)),
            local_body_index=int(local_body_index),
            depth=int(step.get("depth", 0)),
            direct_child_indices=direct_children,
            active_part_indices=sorted(int(index) for index in active_parts),
            input_source=(
                "root_source_3mf"
                if not active_parts
                else "parent_emitted_standalone_3mf"
            ),
        )
        if int(step.get("step_order", step_order)) != step_order:
            raise ValueError(
                f"strict recursive step order mismatch at P{local_body_index:02d}"
            )

        cache_key = None
        cache_input_sha256 = str(source_artifact_sha256)
        if active_parts:
            cache_input_entry = active_parts.get(local_body_index)
            if cache_input_entry is None:
                raise ValueError(
                    f"strict recursive step {step_order} cannot consume P{local_body_index:02d}; "
                    "the previous output does not contain that active subassembly"
                )
            cache_input_sha256 = str(
                cache_input_entry.get("source_3mf_sha256", "")
            )
            if stage_cache is not None:
                validate_recursive_artifact_identity(
                    Path(str(cache_input_entry.get("source_3mf_path", ""))),
                    cache_input_entry,
                )
        if stage_cache is not None:
            cache_key = stage_cache.stage_key(
                run_fingerprint=run_fingerprint,
                implementation=stage_implementation_fingerprint,
                step=step,
                input_artifact_sha256=cache_input_sha256,
            )
            cache_hit = stage_cache.lookup(cache_key)
            if cache_hit is not None:
                restored_entries, restored_records = restore_cached_stage(
                    cache_hit
                )
                local_candidates = [
                    entry
                    for entry in restored_entries
                    if int(entry["source_part_index"]) == local_body_index
                ]
                if len(local_candidates) != 1:
                    raise ValueError(
                        f"cached recursive stage {step_order} must contain exactly one "
                        f"local body P{local_body_index:02d}"
                    )
                local_cached_part = local_candidates[0]
                direct_cached_parts = {
                    int(entry["source_part_index"]): entry
                    for entry in restored_entries
                    if int(entry["source_part_index"]) != local_body_index
                }
                if sorted(direct_cached_parts) != sorted(direct_children):
                    raise ValueError(
                        f"cached recursive stage {step_order} child coverage mismatch: "
                        f"expected {sorted(direct_children)}, found {sorted(direct_cached_parts)}"
                    )
                active_parts, transition = advance_strict_recursive_state(
                    active_parts,
                    local_body_index,
                    local_cached_part,
                    direct_cached_parts,
                )
                stage_records.extend(restored_records)
                executed_internal_indices.add(local_body_index)
                optimization_metrics["stage_cache_hits"] = int(
                    optimization_metrics.get("stage_cache_hits", 0)
                ) + 1
                snapshot_records.append(
                    {
                        **transition,
                        "step_order": int(step_order),
                        "depth": int(step.get("depth", 0)),
                        "execution_order": "strict_depth_first_preorder",
                        "stage_cache_hit": True,
                        "stage_cache_key": cache_key,
                        "mesh_count": int(len(active_parts)),
                        "part_ids": [
                            active_parts[index]["part_id"]
                            for index in sorted(active_parts)
                        ],
                        "package_validation": {
                            "valid": True,
                            "source": "validated_recursive_stage_cache",
                        },
                        "reload_validation": [],
                        "output_3mf": None,
                    }
                )
                runtime_log(
                    "递归缓存",
                    "recursive_stage_cache_hit",
                    "已恢复通过验证的递归阶段，跳过重复几何与布尔",
                    step_order=int(step_order),
                    local_body_index=int(local_body_index),
                    cache_key=str(cache_key),
                    restored_part_indices=sorted(
                        int(entry["source_part_index"])
                        for entry in restored_entries
                    ),
                )
                continue
            optimization_metrics["stage_cache_misses"] = int(
                optimization_metrics.get("stage_cache_misses", 0)
            ) + 1

        recursive_input_path = None
        recursive_input_origin_step = None
        step_vertices = vertices
        step_faces = faces
        step_colors = colors
        step_components = components
        step_boundary_neighbor_lookup = boundary_neighbor_lookup
        step_component_centers = component_centers
        step_model_center = model_center
        step_interface_retopology = interface_retopology
        recursive_geometry_mapping: list[dict] = []
        if active_parts:
            target = active_parts.get(local_body_index)
            if target is None:
                raise ValueError(
                    f"strict recursive step {step_order} cannot consume P{local_body_index:02d}; "
                    "the previous output does not contain that active subassembly"
                )
            expected_contains = subtree_component_indices(
                local_body_index, assembly_children
            )
            if sorted(int(index) for index in target.get("contains", [])) != expected_contains:
                raise ValueError(
                    f"strict recursive step {step_order} provenance mismatch for "
                    f"P{local_body_index:02d}: expected {expected_contains}, "
                    f"found {target.get('contains', [])}"
                )
            if target.get("state_role") != "pending_subassembly":
                raise ValueError(
                    f"strict recursive step {step_order} expected pending P{local_body_index:02d}, "
                    f"found role {target.get('state_role')}"
                )
            recursive_input_path = Path(str(target.get("source_3mf_path", "")))
            if not recursive_input_path.is_file():
                raise ValueError(
                    f"strict recursive step {step_order} has no parent-emitted "
                    f"3MF input for P{local_body_index:02d}"
                )
            recursive_input_origin_step = int(
                target.get("source_3mf_origin_step", -1)
            )
            runtime_log(
                "递归输入",
                "recursive_input_reload_start",
                "正在从上阶段输出的独立 3MF 重载本阶段输入",
                step_order=int(step_order),
                local_body_index=int(local_body_index),
                parent_output_step=int(recursive_input_origin_step),
                input_3mf=str(recursive_input_path),
                expected_sha256=str(target.get("source_3mf_sha256", "")),
                cumulative_3mf_is_input=False,
            )
            reloaded_target = verified_artifacts.load(
                recursive_input_path,
                target,
                loader=lambda: reload_recursive_part_input(
                    recursive_input_path,
                    target,
                    require_artifact_identity=True,
                ),
            )
            target["mesh"] = reloaded_target["mesh"]
            target["face_color_hexes"] = reloaded_target["face_color_hexes"]
            target["face_filament_slot_indices"] = reloaded_target[
                "face_filament_slot_indices"
            ]
            recursive_context = recursive_input_geometry_context(
                reloaded_target=reloaded_target,
                expected_indices=expected_contains,
                source_vertices=vertices,
                source_faces=faces,
                source_components=components,
                interface_retopology=interface_retopology,
            )
            step_vertices = recursive_context["vertices"]
            step_faces = recursive_context["faces"]
            step_colors = recursive_context["colors"]
            step_components = recursive_context["components"]
            step_boundary_neighbor_lookup = recursive_context[
                "boundary_neighbor_lookup"
            ]
            step_component_centers = recursive_context["component_centers"]
            step_model_center = recursive_context["model_center"]
            step_interface_retopology = recursive_context["interface_retopology"]
            recursive_geometry_mapping = recursive_context["mapping_records"]
            runtime_log(
                "递归输入",
                "recursive_input_reload_done",
                "上阶段独立 3MF 已完成身份校验并成为本阶段几何输入",
                step_order=int(step_order),
                local_body_index=int(local_body_index),
                parent_output_step=int(recursive_input_origin_step),
                input_3mf=str(recursive_input_path),
                verified_sha256=str(target.get("source_3mf_sha256", "")),
                reloaded_vertex_count=int(len(step_vertices)),
                reloaded_face_count=int(len(step_faces)),
                component_mapping_count=int(len(recursive_geometry_mapping)),
                cumulative_3mf_is_input=False,
            )
        elif assembly_parents.get(local_body_index) is not None:
            raise ValueError("the first strict recursive step must consume the root model")

        if boundary_review is not None:
            from .boundary_review import owners_from_components, apply_component_ownership
            stage_owners = owners_from_components(len(step_faces), step_components)
            checked_owners, boundary_record = boundary_review.prepare(
                step_vertices, step_faces, stage_owners,
                context=f"step_{step_order:02d}_P{local_body_index:02d}",
            )
            if not np.array_equal(stage_owners, checked_owners):
                step_components = apply_component_ownership(
                    step_vertices, step_faces, step_components, checked_owners)
                step_boundary_neighbor_lookup = component_boundary_neighbor_lookup(
                    step_faces, step_components)
                step_component_centers = {
                    index: step_components[index - 1].center for index in step_component_centers
                }
                # Root helper closures use the original component list: do not reuse
                # their already-computed child references after a local approval.
                if recursive_input_path is None:
                    raise ValueError("Root ownership changed after planning; rerun with input-level boundary approval")

        layer_dir = artifact_layers_dir / (
            f"layer_{step_order:02d}_INWARD_P{local_body_index:02d}"
        )
        layer_dir.mkdir(parents=True, exist_ok=True)

        if recursive_input_path is None:
            child_refs, union_by_child, subtree_by_child = layer_child_context(
                local_body_index
            )
        else:
            child_refs, union_by_child, subtree_by_child = (
                build_layer_child_cut_references(
                    vertices=step_vertices,
                    faces=step_faces,
                    components=step_components,
                    parent_index=local_body_index,
                    direct_child_indices=direct_children,
                    assembly_children=assembly_children,
                    model_center=step_model_center,
                    boundary_neighbor_lookup=step_boundary_neighbor_lookup,
                    inward_overrides=inward_overrides,
                    effective_cap_mode=effective_cap_mode,
                    effective_planar_extra_limit=effective_planar_extra_limit,
                    fit_clearance_by_part=fit_clearance_by_part,
                    boundary_reconciliation_tolerance_mm=(
                        boundary_reconciliation_tolerance_mm
                    ),
                    clearance_mode=clearance_mode,
                    interface_retopology=step_interface_retopology,
                    max_extension_mm=max_extension_mm,
                    flat_clearance_mm=flat_clearance_mm,
                    bottom_clearance_mm=bottom_clearance_mm,
                    lead_in_mm=lead_in_mm,
                    interface_geometry=interface_geometry,
                )
            )
        local_component = step_components[local_body_index - 1]
        local_color = part_color(local_body_index)
        local_role = (
            "root_body"
            if assembly_parents.get(local_body_index) is None
            else "local_body"
        )
        local_part_id = (
            f"S{step_order:02d}_P{local_body_index:02d}_"
            f"{sanitize_name(local_color['name'])}_{local_role.upper()}_CUT"
        )
        local_cap_mode = effective_cap_mode(local_body_index)
        local_fit_clearance_mm = float(
            fit_clearance_by_part[local_body_index]
        )
        local_insert_shrink_mm, local_socket_overcut_mm = clearance_offsets(
            clearance_mode, local_fit_clearance_mm
        )
        local_top_edge_clearance_mm = visible_top_edge_clearance(
            local_insert_shrink_mm
        )
        local_planar_extra_limit_mm = float(
            effective_planar_extra_limit(local_body_index)
        )
        interface_policy = assembly_interface_policy(interface_geometry)

        # Child-solid construction may annotate the shared CapDecision objects.
        # The parent closure must use the original, prevalidated interface state;
        # otherwise building children first can change which closure faces the
        # parent emits and leave it open before Boolean subtraction.
        parent_cut_refs = copy.deepcopy(child_refs)

        # Close the parent while the source arrays and interface-planning state
        # are still pristine. Child construction intentionally reuses and may
        # annotate those shared structures, so delaying parent closure until
        # after child construction makes the result order-dependent.
        closed_parent_mesh, closed_parent_stats = make_body_cut_mesh(
            vertices=step_vertices,
            faces=step_faces,
            body_component=local_component,
            part_id=local_part_id,
            cut_refs=parent_cut_refs,
            max_extension_mm=max_extension_mm,
            interface_retopology=step_interface_retopology,
            flat_clearance_mm=flat_clearance_mm,
            fit_clearance_mm=local_fit_clearance_mm,
            socket_overcut_mm=local_socket_overcut_mm,
            bottom_clearance_mm=bottom_clearance_mm,
            clearance_mode=clearance_mode,
            model_center=step_model_center,
            cap_mode=local_cap_mode,
            interface_geometry=interface_geometry,
            planar_extra_limit_mm=local_planar_extra_limit_mm,
            lead_in_mm=lead_in_mm,
            preserve_unmatched_source_geometry=bool(
                recursive_input_path is not None
            ),
            defer_local_connector_boolean=bool(
                interface_policy.complete_child_boolean
            ),
        )
        closed_parent_stats["assembly_interface_policy"] = (
            interface_policy.as_record()
        )
        if str(closed_parent_stats.get("interface_geometry")) != str(
            interface_geometry
        ):
            raise ValueError(
                "parent closure did not preserve the requested interface geometry: "
                f"requested={interface_geometry!r}, "
                f"actual={closed_parent_stats.get('interface_geometry')!r}"
            )
        runtime_log(
            "assembly-boolean",
            "closed_parent_ready_before_child_prebuild",
            "Closed parent solid is ready before child construction",
            step_order=int(step_order),
            parent_body_index=int(local_body_index),
            parent_watertight=bool(closed_parent_mesh.is_watertight),
            parent_face_count=int(len(closed_parent_mesh.faces)),
        )

        # A compact local connector is authoritative only when its cutter comes
        # from the complete emitted child.  Boundary extrusions deliberately do
        # not enter this path: their visible rim is coplanar with the parent and
        # both sides are instead generated from one shared source-patch field.
        prebuilt_child_geometry: dict[
            int,
            tuple[trimesh.Trimesh, dict, list[trimesh.Trimesh]],
        ] = {}
        authoritative_child_indices = (
            direct_children if interface_policy.complete_child_boolean else []
        )
        runtime_log(
            "assembly-boolean",
            "complete_child_prebuild_start",
            "Building complete direct-child solids before parent subtraction",
            step_order=int(step_order),
            parent_body_index=int(local_body_index),
            child_indices=[int(index) for index in authoritative_child_indices],
            interface_geometry=str(interface_geometry),
        )
        for child_index in authoritative_child_indices:
            subtree = (
                subtree_by_child.get(child_index)
                or subtree_component_indices(child_index, assembly_children)
            )
            union_component = (
                union_by_child.get(child_index)
                or build_subassembly_component(
                    step_vertices,
                    step_faces,
                    step_components,
                    subtree,
                    step_components[child_index - 1].color_code,
                )
            )
            child_color = part_color(child_index)
            child_has_descendants = bool(assembly_children.get(child_index))
            child_part_id = (
                f"S{step_order:02d}_P{child_index:02d}_"
                f"{sanitize_name(child_color['name'])}_"
                f"{'PENDING_SUBASSEMBLY' if child_has_descendants else 'LEAF_INSERT'}"
            )
            child_fit_clearance_mm = float(
                fit_clearance_by_part[child_index]
            )
            child_insert_shrink_mm, _child_socket_overcut_mm = (
                clearance_offsets(clearance_mode, child_fit_clearance_mm)
            )
            child_mesh, child_stats = make_layer_child_subassembly_mesh(
                vertices=step_vertices,
                faces=step_faces,
                source_colors=step_colors,
                components=step_components,
                component=union_component,
                root_child_index=child_index,
                subtree_indices=subtree,
                parent_index=local_body_index,
                part_id=child_part_id,
                max_extension_mm=max_extension_mm,
                interface_retopology=step_interface_retopology,
                flat_clearance_mm=flat_clearance_mm,
                fit_clearance_mm=child_fit_clearance_mm,
                insert_shrink_mm=child_insert_shrink_mm,
                lead_in_mm=lead_in_mm,
                top_edge_clearance_mm=visible_top_edge_clearance(
                    child_insert_shrink_mm
                ),
                boundary_neighbor_lookup=step_boundary_neighbor_lookup,
                inward_override=inward_overrides.get(child_index),
                model_center=step_model_center,
                cap_mode=effective_cap_mode(child_index),
                planar_extra_limit_mm=float(
                    effective_planar_extra_limit(child_index)
                ),
                cap_decisions_by_loop=shared_child_cap_decisions(
                    child_refs,
                    child_index,
                ),
                bottom_clearance_mm=bottom_clearance_mm,
                interface_geometry=interface_geometry,
            )
            local_connector_cutters = list(
                child_stats.pop(
                    "_local_connector_boolean_cutters",
                    child_stats.pop("_boolean_socket_cutters", []),
                )
            )
            prebuilt_child_geometry[int(child_index)] = (
                child_mesh,
                child_stats,
                local_connector_cutters,
            )
        runtime_log(
            "assembly-boolean",
            "complete_child_prebuild_done",
            "Complete direct-child solids are ready for parent subtraction",
            step_order=int(step_order),
            parent_body_index=int(local_body_index),
            child_count=int(len(prebuilt_child_geometry)),
            interface_geometry=str(interface_geometry),
        )

        runtime_log(
            "递归几何",
            "local_body_cut_start",
            "正在切分当前局部主体",
            step_order=int(step_order),
            local_body_index=int(local_body_index),
            direct_child_count=int(len(direct_children)),
            input_source=(
                "root_source_3mf"
                if recursive_input_path is None
                else "parent_emitted_standalone_3mf"
            ),
            cap_mode=str(local_cap_mode),
            preserve_unmatched_source_geometry=bool(
                recursive_input_path is not None
            ),
            interface_geometry=interface_geometry,
        )
        local_mesh, local_stats = closed_parent_mesh, closed_parent_stats
        if prebuilt_child_geometry:
            inherited_boolean_audit = boolean_collapsed_face_audit(
                closed_parent_mesh
            )
            boolean_finalization_policy = BooleanFinalizationPolicy(
                inherited_collapsed_face_budget=int(
                    inherited_boolean_audit["collapsed_face_count"]
                ),
                inherited_source="closed_parent_before_complete_child_difference",
            )
            clearance_cutters = []
            clearance_cutter_labels: list[str] = []
            clearance_cutter_records = []
            for child_index in direct_children:
                runtime_log(
                    "assembly-boolean",
                    "complete_child_proxy_start",
                    "Building one exact full-size child Boolean cutter",
                    step_order=int(step_order),
                    child_index=int(child_index),
                    source_faces=int(
                        prebuilt_child_geometry[child_index][0].metadata.get(
                            "protected_source_face_count",
                            0,
                        )
                    ),
                )
                cutter, cutter_record = exact_unscaled_cutter(
                    prebuilt_child_geometry[child_index][0]
                )
                clearance_cutters.append(cutter)
                clearance_cutter_labels.append(f"P{int(child_index):02d}_exact_unscaled_child")
                clearance_cutter_records.append(
                    {
                        "child_index": int(child_index),
                        **cutter_record,
                        "local_connector_cutter_count": 0,
                        "backing_clearance_cutter_count": 0,
                        "compact_socket_cutter_count": 0,
                    }
                )
                runtime_log(
                    "assembly-boolean",
                    "complete_child_proxy_done",
                    "Exact full-size child cutter is ready; no additional clearance cutters",
                    step_order=int(step_order),
                    child_index=int(child_index),
                    proxy_faces=int(len(cutter.faces)),
                    local_connector_cutter_count=0,
                )
            boolean_case_cache_path = os.environ.get(
                "SPLIT3MF_BOOLEAN_CASE_CACHE",
                "",
            ).strip()
            if not boolean_case_cache_path:
                from .print_tolerance import current
                recovery_dir = current().recovery_dir
                if recovery_dir is not None:
                    boolean_case_cache_path = str(
                        recovery_dir / 'boolean' / f'step_{step_order:02d}_parent_{local_body_index:02d}.npz'
                    )
            if boolean_case_cache_path:
                cached_path = write_boolean_case_cache(
                    Path(boolean_case_cache_path),
                    local_mesh,
                    clearance_cutters,
                    labels=clearance_cutter_labels,
                    metadata={
                        "step_order": int(step_order),
                        "parent_body_index": int(local_body_index),
                        "interface_geometry": str(interface_geometry),
                    },
                )
                runtime_log(
                    "assembly-boolean",
                    "boolean_case_cache_written",
                    "Reusable recursive Boolean harness case is ready",
                    cache_path=str(cached_path),
                    cutter_count=int(len(clearance_cutters)),
                )
            try:
                local_mesh, complete_child_boolean_record = subtract_socket_cutters(
                    local_mesh,
                    clearance_cutters,
                    allow_empty_intersection=True,
                    inherited_collapsed_face_budget=int(
                        boolean_finalization_policy.inherited_collapsed_face_budget
                    ),
                    cleanup_volume_envelope_cap_mm3=(
                        5e-5
                        if step_interface_retopology.config.surface_band_validation
                        == "advisory"
                        else 1e-5
                    ),
                    topology_healthy_export_volume_envelope_cap_mm3=(
                        5e-3
                        if step_interface_retopology.config.surface_band_validation
                        == "advisory"
                        else 0.0
                    ),
                )
                ratio_accepted_collapsed_faces = int(
                    complete_child_boolean_record.get(
                        "ratio_accepted_new_collapsed_faces",
                        0,
                    )
                )
                micro_accepted_collapsed_faces = int(
                    complete_child_boolean_record.get(
                        "micro_accepted_new_collapsed_faces",
                        0,
                    )
                )
                local_mesh = finalize_boolean_difference_mesh(
                    local_mesh,
                    policy=BooleanFinalizationPolicy(
                        inherited_collapsed_face_budget=int(
                            boolean_finalization_policy.inherited_collapsed_face_budget
                        ),
                        inherited_source=str(
                            boolean_finalization_policy.inherited_source
                        ),
                        ratio_accepted_collapsed_face_budget=(
                            ratio_accepted_collapsed_faces
                        ),
                        maximum_collapsed_face_ratio=0.005,
                        micro_accepted_collapsed_face_budget=(
                            micro_accepted_collapsed_faces
                        ),
                        maximum_micro_collapsed_face_ratio=0.005,
                        maximum_micro_collapsed_face_edge_mm=0.25,
                    ),
                )
                local_mesh.metadata["name"] = local_part_id
            except ValueError as error:
                if os.environ.get("SPLIT3MF_FORCE_PREVIEW", "") != "1":
                    raise
                complete_child_boolean_record = {
                    "status": "skipped_for_forced_preview",
                    "error": str(error),
                    "preview_not_printable": True,
                }
                local_mesh.metadata["preview_not_printable"] = True
            local_stats["complete_child_boolean_record"] = {
                **complete_child_boolean_record,
                "boolean_scope": "complete_emitted_exact_unscaled_child_solids",
                "interface_geometry": str(interface_geometry),
                "authoritative_clearance_mm": 0.0,
                "clearance_strategy": "post_split_xyz_uniform_scale_only",
                "inherited_boolean_collapsed_face_audit": (
                    inherited_boolean_audit
                ),
                "child_indices": [int(index) for index in direct_children],
                "derived_clearance_cutters": clearance_cutter_records,
            }
            local_stats["authoritative_female_boolean_source"] = (
                "complete_emitted_exact_unscaled_child_solids_only"
            )
            local_stats["output_faces"] = int(len(local_mesh.faces))
            local_stats["output_vertices"] = int(len(local_mesh.vertices))
            local_bbox = local_mesh.bounds
            local_stats["bbox_min"] = local_bbox[0].round(6).tolist()
            local_stats["bbox_max"] = local_bbox[1].round(6).tolist()
            local_stats["bbox_size_mm"] = (
                local_bbox[1] - local_bbox[0]
            ).round(6).tolist()
            local_stats.update(mesh_runtime_stats(local_mesh))
        runtime_log(
            "递归几何",
            "local_body_cut_done",
            "当前局部主体切分完成",
            step_order=int(step_order),
            local_body_index=int(local_body_index),
            output_vertex_count=int(len(local_mesh.vertices)),
            output_face_count=int(len(local_mesh.faces)),
        )
        if recursive_input_path is not None:
            local_stats["source_faces"] = int(
                components[local_body_index - 1].face_count
            )
        local_stats["recursive_geometry_input"] = (
            "root_source_mesh"
            if recursive_input_path is None
            else "parent_emitted_standalone_3mf"
        )
        local_stats["recursive_geometry_input_3mf"] = (
            None if recursive_input_path is None else str(recursive_input_path)
        )
        local_stats["recursive_geometry_component_mapping"] = (
            recursive_geometry_mapping
        )
        local_stats = validate_changed_part(
            layer_dir,
            local_part_id,
            local_mesh,
            local_stats,
            step_order,
            step,
            local_role,
            [local_body_index],
        )
        stage_records.append(local_stats)
        local_active_part = active_part(
            local_body_index,
            local_part_id,
            local_mesh,
            local_stats,
            local_role,
            [local_body_index],
            step_order,
        )
        local_active_part = materialize_changed_part(
            local_active_part,
            layer_dir,
            step_order,
        )

        direct_child_parts: dict[int, dict] = {}
        for child_index in direct_children:
            subtree = (
                subtree_by_child.get(child_index)
                or subtree_component_indices(child_index, assembly_children)
            )
            union_component = (
                union_by_child.get(child_index)
                or build_subassembly_component(
                    step_vertices,
                    step_faces,
                    step_components,
                    subtree,
                    step_components[child_index - 1].color_code,
                )
            )
            child_color = part_color(child_index)
            child_has_descendants = bool(assembly_children.get(child_index))
            child_state_role = (
                "pending_subassembly" if child_has_descendants else "leaf_insert"
            )
            child_part_id = (
                f"S{step_order:02d}_P{child_index:02d}_"
                f"{sanitize_name(child_color['name'])}_"
                f"{'PENDING_SUBASSEMBLY' if child_has_descendants else 'LEAF_INSERT'}"
            )
            child_fit_clearance_mm = float(fit_clearance_by_part[child_index])
            child_insert_shrink_mm, _child_socket_overcut_mm = clearance_offsets(
                clearance_mode, child_fit_clearance_mm
            )
            child_top_edge_clearance_mm = visible_top_edge_clearance(
                child_insert_shrink_mm
            )
            child_cap_decisions = shared_child_cap_decisions(
                child_refs,
                child_index,
            )
            runtime_log(
                "递归几何",
                "child_subassembly_build_start",
                "正在构建直属子件或待递归子装配",
                step_order=int(step_order),
                parent_body_index=int(local_body_index),
                child_index=int(child_index),
                state_role=str(child_state_role),
                subtree_indices=[int(index) for index in subtree],
            )
            if child_index in prebuilt_child_geometry:
                child_mesh, child_stats, _compact_socket_cutters = (
                    prebuilt_child_geometry[child_index]
                )
            else:
                child_mesh, child_stats = make_layer_child_subassembly_mesh(
                    vertices=step_vertices,
                    faces=step_faces,
                    source_colors=step_colors,
                    components=step_components,
                    component=union_component,
                    root_child_index=child_index,
                    subtree_indices=subtree,
                    parent_index=local_body_index,
                    part_id=child_part_id,
                    max_extension_mm=max_extension_mm,
                    interface_retopology=step_interface_retopology,
                    flat_clearance_mm=flat_clearance_mm,
                    fit_clearance_mm=child_fit_clearance_mm,
                    insert_shrink_mm=child_insert_shrink_mm,
                    lead_in_mm=lead_in_mm,
                    top_edge_clearance_mm=child_top_edge_clearance_mm,
                    boundary_neighbor_lookup=step_boundary_neighbor_lookup,
                    inward_override=inward_overrides.get(child_index),
                    model_center=step_model_center,
                    cap_mode=effective_cap_mode(child_index),
                    planar_extra_limit_mm=float(
                        effective_planar_extra_limit(child_index)
                    ),
                    cap_decisions_by_loop=child_cap_decisions,
                    bottom_clearance_mm=bottom_clearance_mm,
                    interface_geometry=interface_geometry,
                )
            runtime_log(
                "递归几何",
                "child_subassembly_build_done",
                "直属子件或待递归子装配构建完成",
                step_order=int(step_order),
                parent_body_index=int(local_body_index),
                child_index=int(child_index),
                state_role=str(child_state_role),
                output_vertex_count=int(len(child_mesh.vertices)),
                output_face_count=int(len(child_mesh.faces)),
            )
            child_stats["recursive_geometry_input"] = (
                "root_source_mesh"
                if recursive_input_path is None
                else "parent_emitted_standalone_3mf"
            )
            child_stats["recursive_geometry_input_3mf"] = (
                None if recursive_input_path is None else str(recursive_input_path)
            )
            child_stats["shared_cap_decision_validation"] = (
                validate_shared_child_cap_decisions(
                    child_index,
                    child_cap_decisions,
                    child_stats,
                    interface_geometry=interface_geometry,
                )
            )
            child_stats = validate_changed_part(
                layer_dir,
                child_part_id,
                child_mesh,
                child_stats,
                step_order,
                step,
                child_state_role,
                subtree,
            )
            stage_records.append(child_stats)
            child_active_part = active_part(
                child_index,
                child_part_id,
                child_mesh,
                child_stats,
                child_state_role,
                subtree,
                step_order,
            )
            direct_child_parts[child_index] = materialize_changed_part(
                child_active_part,
                layer_dir,
                step_order,
            )

        active_parts, transition = advance_strict_recursive_state(
            active_parts,
            local_body_index,
            local_active_part,
            direct_child_parts,
        )
        transition.update(
            {
                "step_order": int(step_order),
                "depth": int(step.get("depth", 0)),
                "recursion_path": [
                    int(index) for index in step.get("recursion_path", [])
                ],
                "execution_order": "strict_depth_first_preorder",
                "recursive_input_source": (
                    "root_source_3mf"
                    if recursive_input_path is None
                    else "parent_emitted_standalone_3mf"
                ),
                "recursive_input_origin_step": recursive_input_origin_step,
                "recursive_input_sha256": (
                    None
                    if recursive_input_path is None
                    else target.get("source_3mf_sha256")
                ),
                "recursive_input_3mf": (
                    None
                    if recursive_input_path is None or output_dir is None
                    else serialize_report_path(
                        recursive_input_path,
                        output_dir,
                        report_path_mode,
                    )
                ),
                "cumulative_3mf_is_recursive_input": False,
            }
        )
        executed_internal_indices.add(local_body_index)

        if stage_cache is not None and cache_key is not None:
            changed_entries = [
                local_active_part,
                *[
                    direct_child_parts[index]
                    for index in sorted(direct_child_parts)
                ],
            ]
            current_stage_records = stage_records[-len(changed_entries) :]
            try:
                stage_cache.commit(
                    key=cache_key,
                    run_fingerprint=run_fingerprint,
                    implementation=stage_implementation_fingerprint,
                    step=step,
                    input_artifact_sha256=cache_input_sha256,
                    changed_entries=changed_entries,
                    stage_records=current_stage_records,
                )
                optimization_metrics["stage_cache_commits"] = int(
                    optimization_metrics.get("stage_cache_commits", 0)
                ) + 1
                runtime_log(
                    "递归缓存",
                    "recursive_stage_cache_commit",
                    "本递归阶段已通过验证并写入断点缓存",
                    step_order=int(step_order),
                    local_body_index=int(local_body_index),
                    cache_key=str(cache_key),
                    changed_part_count=int(len(changed_entries)),
                )
            except (OSError, ValueError) as exc:
                if stage_cache.mode == "strict":
                    raise
                runtime_log(
                    "递归缓存",
                    "recursive_stage_cache_commit_skipped",
                    "断点缓存写入失败，本次几何结果继续使用但不缓存",
                    step_order=int(step_order),
                    local_body_index=int(local_body_index),
                    cache_key=str(cache_key),
                    error=str(exc),
                )

        snapshot_parts = cumulative_snapshot_parts(
            active_parts,
            step_order,
            transition,
        )

        snapshot_record = {
            **transition,
            "mesh_count": len(snapshot_parts),
            "part_ids": [part["part_id"] for part in snapshot_parts],
            "package_validation": None,
            "reload_validation": [],
            "output_3mf": None,
        }

        if output_dir is not None:
            candidate_path = layer_dir / (
                f"layer_{step_order:02d}_INWARD_P{local_body_index:02d}_"
                "CUMULATIVE_candidate.3mf"
            )
            layer_package_path = layer_dir / (
                f"layer_{step_order:02d}_INWARD_P{local_body_index:02d}_"
                "CUMULATIVE_mm.3mf"
            )
            export_colored_parts_3mf(
                candidate_path,
                snapshot_parts,
                title=(
                    f"{input_stem} strict recursive cumulative step "
                    f"{step_order:02d} after P{local_body_index:02d}"
                ),
                source_application=source_application,
                source_filament_colors=source_filament_colors,
                source_project_settings=source_project_settings,
            )
            package_validation = validate_colored_parts_3mf(
                candidate_path,
                snapshot_parts,
                source_filament_colors=source_filament_colors,
                source_application=source_application,
                source_project_settings=source_project_settings,
                output_layout="assembly",
            )
            known_topology_errors = [
                error
                for error in package_validation.get("errors", [])
                if "reloaded mesh is not watertight" in error
                or (
                    "reloaded mesh has " in error
                    and ("open edges" in error or "over-shared edges" in error)
                )
            ]
            package_errors = [
                error
                for error in package_validation.get("errors", [])
                if error not in known_topology_errors
            ]
            if package_errors:
                failed_path = candidate_path.with_name(
                    candidate_path.stem.replace("_candidate", "_FAILED")
                    + candidate_path.suffix
                )
                candidate_path.replace(failed_path)
                snapshot_record["package_validation"] = package_validation
                snapshot_record["output_3mf"] = serialize_report_path(
                    failed_path, output_dir, report_path_mode
                )
                snapshot_records.append(snapshot_record)
                raise ValueError(
                    f"strict recursive snapshot {step_order} package validation failed: "
                    + "; ".join(package_errors)
                )

            candidate_path.replace(layer_package_path)
            reloaded_by_name = {
                record["part_id"]: record
                for record in load_colored_mesh_objects_3mf(layer_package_path)
            }
            reload_validation = []
            blocking_reload_parts = []
            for active_index, entry in sorted(active_parts.items()):
                reloaded_entry = reloaded_by_name.get(entry["part_id"])
                if reloaded_entry is None:
                    blocking_reload_parts.append(entry["part_id"])
                    reload_validation.append(
                        {
                            "part_id": entry["part_id"],
                            "valid": False,
                            "error": "missing after cumulative 3MF reload",
                        }
                    )
                    continue
                reloaded_mesh = reloaded_entry["mesh"]
                expected_face_colors = [
                    normalize_3mf_color(value)
                    for value in entry.get("face_color_hexes", [])
                ]
                if reloaded_entry["face_color_hexes"] != expected_face_colors:
                    blocking_reload_parts.append(entry["part_id"])
                    reload_validation.append(
                        {
                            "part_id": entry["part_id"],
                            "valid": False,
                            "error": "per-triangle colors changed in cumulative audit",
                        }
                    )
                    continue
                validation = validate_mesh_in_memory(reloaded_mesh)
                ratio_accepted = bool(
                    validation["winding_consistent"]
                    and not validation["inconsistent_shared_edges"]
                    and float(validation["topology_defect_ratio"])
                    <= float(max_topology_defect_ratio)
                )
                valid = bool(
                    validation_profile == "ratio"
                    and ratio_accepted
                    or validation_profile == "strict"
                    and validation["watertight"]
                    and validation["winding_consistent"]
                    and not validation["open_edges"]
                    and not validation["over_shared_edges"]
                    and not validation["inconsistent_shared_edges"]
                )
                reload_validation.append(
                    {
                        "part_id": entry["part_id"],
                        "valid": valid,
                        "watertight": bool(validation["watertight"]),
                        "open_edges": int(validation["open_edges"]),
                        "over_shared_edges": int(validation["over_shared_edges"]),
                        "inconsistent_shared_edges": int(
                            validation["inconsistent_shared_edges"]
                        ),
                        "topology_defect_ratio": float(
                            validation["topology_defect_ratio"]
                        ),
                    }
                )
                if not valid:
                    blocking_reload_parts.append(entry["part_id"])
            snapshot_record["package_validation"] = {
                **package_validation,
                "known_topology_errors": known_topology_errors,
                "errors": package_errors,
                "valid": not package_errors,
            }
            snapshot_record["reload_validation"] = reload_validation
            snapshot_record["output_3mf"] = serialize_report_path(
                layer_package_path, output_dir, report_path_mode
            )
            if blocking_reload_parts:
                snapshot_record["blocking_parts"] = blocking_reload_parts
                if allow_partial:
                    runtime_log(
                        "recursive-debug",
                        "partial_snapshot_topology_warning",
                        "Partial recursive debug snapshot retained topology warnings for manual review",
                        step_order=int(step_order),
                        blocking_parts=blocking_reload_parts,
                        cumulative_3mf=str(layer_package_path),
                    )
                else:
                    snapshot_records.append(snapshot_record)
                    raise ValueError(
                        f"strict recursive snapshot {step_order} reload validation failed: "
                        + ", ".join(blocking_reload_parts)
                    )
            for record in stage_records[-(1 + len(direct_children)):]:
                record["layer_3mf"] = serialize_report_path(
                    layer_package_path, output_dir, report_path_mode
                )

        snapshot_records.append(snapshot_record)
        runtime_log(
            "递归拆件",
            "recursive_step_done",
            "递归拆件步骤完成，状态已提交给下一步骤",
            step_order=int(step_order),
            step_number=int(step_order + 1),
            total_steps=int(len(recursive_steps)),
            local_body_index=int(local_body_index),
            active_part_indices=sorted(int(index) for index in active_parts),
            emitted_child_indices=sorted(
                int(index) for index in direct_child_parts
            ),
            next_step_must_reload_parent_output=True,
            cumulative_3mf_is_input=False,
        )

    if not allow_partial and executed_internal_indices != expected_internal_indices:
        missing = sorted(expected_internal_indices - executed_internal_indices)
        extra = sorted(executed_internal_indices - expected_internal_indices)
        raise ValueError(
            f"strict recursive execution coverage mismatch: missing={missing}, extra={extra}"
        )
    final_indices = sorted(active_parts)
    expected_final_indices = list(range(1, len(components) + 1))
    if not allow_partial and final_indices != expected_final_indices:
        raise ValueError(
            f"strict recursive execution did not finish with every part active: "
            f"expected {expected_final_indices}, found {final_indices}"
        )
    if temporary_artifacts is not None:
        temporary_artifacts.cleanup()
        for entry in active_parts.values():
            entry.pop("source_3mf_path", None)
    runtime_log(
        "递归拆件",
        "recursive_executor_done",
        "严格串行递归执行完成",
        recursive_step_count=int(len(recursive_steps)),
        final_part_indices=final_indices,
        stage_record_count=int(len(stage_records)),
        snapshot_record_count=int(len(snapshot_records)),
        partial_debug_run=bool(allow_partial),
    )
    return layers_dir, stage_records, active_parts, snapshot_records

