from __future__ import annotations

import re

import numpy as np

from .interface_surface import (
    MAX_INTERFACE_EXTENSION_MM,
    build_pairwise_interface_surfaces as build_interface_surface_pair,
)
from .mesh import build_local_mesh
from .part_interface_composition import InterfaceAttachment, compose_part_with_interfaces
from .project import color_resolution_status
from .spatial_intersections import (
    clean_tiny_boundary_contacts_3d,
    remove_true_self_intersections_3d,
)
from .interface_relations import interface_role_index
from .interface_direction import inward_interface_axis
from .local_ray_probe import LocalRayProbe
from .stage_cache import fingerprint_payload
import trimesh


def _part_index(part_id: str) -> int:
    match = re.fullmatch(r"P(\d+)", str(part_id))
    if match is None:
        raise ValueError(f"invalid Stage 04 part id: {part_id!r}")
    return int(match.group(1))


def build_pairwise_interface_surfaces(
    interface_plan: dict,
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    components: list,
    scale_ratio: float,
    clearance_mm: float,
) -> tuple[dict[str, np.ndarray], dict]:
    """Build paired annular surfaces using only the Stage 04 interface contract."""
    if interface_plan.get("schema") != "contact-interface-plan/v1":
        raise ValueError("interface builder requires contact-interface-plan/v1")
    unmatched = interface_plan.get("unmatched_contact_segments", [])
    if unmatched:
        pairs = ["/".join(item["parts"]) for item in unmatched]
        raise ValueError(f"Stage 04 has unresolved contact segments: {pairs}")
    interfaces = interface_plan.get("interfaces")
    if not isinstance(interfaces, list):
        raise ValueError("Stage 04 interface plan has no interfaces list")
    role_index = interface_role_index(interface_plan)
    arrays: dict[str, np.ndarray] = {}
    records: list[dict] = []
    mortise_source_triangles = {}
    source_probe = LocalRayProbe(trimesh.Trimesh(vertices, faces, process=False))
    source_ray_limit = max(float(np.linalg.norm(np.ptp(vertices, axis=0))) * 2, 1.0)
    for relation in interfaces:
        interface_id = str(relation.get("interface_id", ""))
        if not interface_id:
            raise ValueError("Stage 04 interface entry is missing interface_id")
        roles = role_index[interface_id]
        mortise_index = _part_index(roles["mortise_part"])
        if not 1 <= mortise_index <= len(components):
            raise ValueError(f"{interface_id}: mortise references an unknown part")
        if mortise_index not in mortise_source_triangles:
            mortise_component = components[mortise_index - 1]
            mortise_source_triangles[mortise_index] = (
                vertices[faces[mortise_component.global_faces]]
                if mortise_component is not None
                and getattr(mortise_component, "global_faces", None) is not None
                else None
            )
        axis = np.asarray(relation.get("mating_axis_toward_mortise"), dtype=np.float64)
        if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) <= 1e-10:
            raise ValueError(f"{interface_id}: Stage 04 mating axis is missing or degenerate")
        axis /= np.linalg.norm(axis)
        side_directions = {
            "tenon": relation.get("insertion_direction"),
            "mortise": relation.get("socket_inward_direction"),
        }
        if any(value is None for value in side_directions.values()):
            raise ValueError(f"{interface_id}: Stage 04 plan is missing a side direction")
        insertion_axis = np.asarray(side_directions["tenon"], dtype=np.float64)
        if (insertion_axis.shape != (3,) or not np.isfinite(insertion_axis).all()
                or np.linalg.norm(insertion_axis) <= 1e-10):
            raise ValueError(f"{interface_id}: Stage 04 insertion direction is invalid")
        insertion_axis /= np.linalg.norm(insertion_axis)
        boundary_loops = relation.get("contact", {}).get("shared_boundary_loops", [])
        if not boundary_loops:
            raise ValueError(f"{interface_id}: Stage 04 plan has no ordered shared boundary")

        for loop_position, loop in enumerate(boundary_loops):
            points = np.asarray(loop.get("boundary_points_mm"), dtype=np.float64)
            if len(points) < 3:
                raise ValueError(f"{interface_id}: shared boundary loop has fewer than three points")
            if points.ndim != 2 or points.shape[1:] != (3,):
                raise ValueError(f"{interface_id}: shared boundary points must be an N×3 array")
            if not np.isfinite(points).all():
                raise ValueError(f"{interface_id}: shared boundary contains non-finite coordinates")
            cleaned_points, _ephemeral_ids, boundary_cleanup = remove_true_self_intersections_3d(
                points
            )
            if not np.array_equal(cleaned_points, points):
                _candidate, _ids, tolerance = clean_tiny_boundary_contacts_3d(
                    points, np.arange(len(points), dtype=np.int64)
                )
                if (tolerance["remaining_intersection_count"]
                        or not tolerance["removed_crossing_loop_count"]):
                    raise ValueError(
                        f"{interface_id}: frozen Stage 03 boundary has an unsafe "
                        "self-intersection"
                    )
                boundary_cleanup["tolerated_point_contacts"] = tolerance
            if len(points) < 3:
                raise ValueError(f"{interface_id}: 3D crossing cleanup left fewer than three boundary points")
            simplified_boundary_point_count = len(points)
            tenon_index = _part_index(roles["tenon_part"])
            if not 1 <= tenon_index <= len(components):
                raise ValueError(f"{interface_id}: tenon references an unknown part")
            tenon_vertices, tenon_faces, _, _ = build_local_mesh(
                vertices, faces, components[tenon_index - 1]
            )
            contact_axis, direction_audit = inward_interface_axis(
                points, tenon_vertices[tenon_faces], source_probe=source_probe,
                mortise_triangles=mortise_source_triangles[mortise_index],
                toward_mortise=axis, maximum_mm=source_ray_limit,
            )
            tenon_loop_index = int(loop.get("tenon_loop_index", -1))
            mortise_loop_index = int(loop.get("mortise_loop_index", -1))
            imprinted_host = (
                relation.get("contact", {}).get("counterpart_mode")
                == "imprinted_host_surface"
            )
            if tenon_loop_index < 0 or (mortise_loop_index < 0 and not imprinted_host):
                raise ValueError(f"{interface_id}: shared boundary is missing Stage 04 loop indices")
            stem = f"{interface_id.lower()}_loop_{loop_position:03d}"
            if not re.fullmatch(r"[a-z0-9_]+", stem):
                raise ValueError(f"invalid interface array key: {stem!r}")
            pair = build_interface_surface_pair(
                boundary_points_mm=points,
                insertion_direction=contact_axis,
                socket_inward_direction=contact_axis,
                scale_ratio=scale_ratio,
                clearance_mm=clearance_mm,
                mortise_shell_probe=source_probe,
                interface_id=interface_id,
                extension_depth_limit_mm=MAX_INTERFACE_EXTENSION_MM,
            )
            paired_topology_matches = bool(
                np.array_equal(pair.tenon.faces, pair.mortise.faces)
            )
            boundary_count = len(points)
            socket_direction = np.asarray(
                pair.tenon.record["extension_direction"], dtype=np.float64
            )
            socket_direction /= np.linalg.norm(socket_direction)
            if float(np.dot(socket_direction, contact_axis)) < 1.0 - 1e-7:
                raise ValueError(
                    f"{interface_id}: construction axis does not point toward the mortise"
                )
            tip_start = pair.tenon.record["tip_vertex_start"]
            tip_count = pair.tenon.record["tip_vertex_count"]
            tenon_tip = pair.tenon.vertices[tip_start : tip_start + tip_count]
            mortise_floor = pair.mortise.vertices[tip_start : tip_start + tip_count]
            tip_difference = mortise_floor - tenon_tip
            axial_gaps = tip_difference @ socket_direction
            radial_gaps = tip_difference - axial_gaps[:, None] * socket_direction
            measured_gaps = np.linalg.norm(radial_gaps, axis=1)
            measured_gap = float(np.mean(measured_gaps))
            measured_depth_clearance = float(np.mean(axial_gaps))
            clearance_matches = bool(
                np.allclose(measured_gaps, clearance_mm, atol=1e-7, rtol=1e-7)
                and np.isclose(
                    measured_depth_clearance, pair.tenon.record.get("bottom_clearance_mm", clearance_mm), atol=1e-7, rtol=1e-7
                )
            )
            for side, surface in (("tenon", pair.tenon), ("mortise", pair.mortise)):
                arrays[f"{stem}_{side}_vertices"] = surface.vertices
                arrays[f"{stem}_{side}_faces"] = surface.faces
            records.append({
                "interface_id": interface_id,
                "loop_position": loop_position,
                "tenon_part": roles["tenon_part"],
                "mortise_part": roles["mortise_part"],
                "tenon_loop_index": tenon_loop_index,
                "mortise_loop_index": mortise_loop_index,
                "boundary_cleanup_3d": boundary_cleanup,
                "simplified_boundary_point_count": int(simplified_boundary_point_count),
                "densified_boundary_point_count": int(len(points)),
                "boundary_densification_subdivisions": 0,
                "selected_geometry_strategy": pair.tenon.record["selected_strategy"],
                "insertion_direction": relation.get("insertion_direction"),
                "socket_inward_direction": relation.get("socket_inward_direction"),
                "mating_axis_toward_mortise": axis.tolist(),
                "stage04_insertion_direction": insertion_axis.tolist(),
                "construction_axis_toward_mortise": socket_direction.tolist(),
                "direction_audit": direction_audit,
                "mean_inner_ring_clearance_mm": measured_gap,
                "measured_clearance_mm": measured_gap,
                "mean_mortise_bottom_clearance_mm": measured_depth_clearance,
                "clearance_matches_configuration": clearance_matches,
                "paired_topology_matches": paired_topology_matches,
                "quality_gates_blocking": False,
                "tenon_surface": pair.tenon.record,
                "mortise_surface": pair.mortise.record,
            })

    result = {
        "schema": "pairwise-interface-surfaces/v1",
        "interface_role_index": role_index,
        "interface_role_index_sha256": fingerprint_payload(role_index),
        "source_interface_schema": interface_plan["schema"],
        "recognized_boundary_fingerprint": interface_plan.get(
            "recognized_boundary_fingerprint"
        ),
        "scale_ratio": float(scale_ratio),
        "total_clearance_mm": float(clearance_mm),
        "extension_depth_limit_mm": MAX_INTERFACE_EXTENSION_MM,
        "extension_depth_minimum_mm": 0.2,
        "extension_depth_policy": "local_cap_root_single_source_query",
        "collision_acceptance_policy": "any_source_exit_blocks",
        "clearance_allocation": "mortise_only_full_side_and_bottom_clearance",
        "interface_surface_count": len(records),
        "interfaces": records,
    }
    return arrays, result


def _color_info_for_component(color_code: str) -> dict:
    from .common import COLOR_INFO

    return COLOR_INFO.get(color_code, {})


def build_pairwise_part_meshes(
    interface_plan: dict,
    interface_arrays: dict[str, np.ndarray],
    interface_summary: dict,
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    components: list,
    recognized_components: list | None = None,
    face_color_tokens: np.ndarray | None = None,
    remove_detached_micro_shells_part_ids: frozenset[str] = frozenset(),
) -> tuple[list[dict], list[dict]]:
    """Attach every interface to each part's preserved exterior in one pass."""
    if recognized_components is not None and len(recognized_components) != len(components):
        raise ValueError("recognized and completed component counts differ")
    records = {
        (item["interface_id"], item["loop_position"]): item
        for item in interface_summary["interfaces"]
    }
    role_index = interface_role_index(interface_plan)
    role_index_sha256 = fingerprint_payload(role_index)
    if interface_summary.get("interface_role_index_sha256") != role_index_sha256:
        raise ValueError("interface summary was built from a different interface role table")
    summary_roles = {}
    for item in interface_summary["interfaces"]:
        interface_id = str(item["interface_id"])
        roles = role_index.get(interface_id)
        if roles is None or (
            item.get("tenon_part"), item.get("mortise_part")
        ) != (roles["tenon_part"], roles["mortise_part"]):
            raise ValueError(
                f"{interface_id}: built interface roles do not match the hashed role table"
            )
        summary_roles[interface_id] = roles
    if set(summary_roles) != set(role_index):
        raise ValueError("built interface surfaces do not match the hashed role table")
    attachments_by_part: dict[str, list[InterfaceAttachment]] = {
        f"P{index:02d}": [] for index in range(1, len(components) + 1)
    }
    imprinted_pockets = []
    for relation in interface_plan["interfaces"]:
        interface_id = str(relation["interface_id"])
        roles = role_index[interface_id]
        for loop_position, _loop in enumerate(
            relation["contact"]["shared_boundary_loops"]
        ):
            record = records[(interface_id, loop_position)]
            stem = f"{interface_id.lower()}_loop_{loop_position:03d}"
            for side in ("tenon", "mortise"):
                part_id = roles[f"{side}_part"]
                if (side == "mortise" and relation["contact"].get("counterpart_mode")
                        == "imprinted_host_surface"):
                    imprinted_pockets.append((
                        part_id, roles["tenon_part"], interface_id,
                        interface_arrays[f"{stem}_{side}_vertices"],
                        interface_arrays[f"{stem}_{side}_faces"],
                        record["densified_boundary_point_count"],
                        record[f"{side}_surface"]["outer_boundary_cap_face_count"],
                        vertices[np.asarray(_loop["boundary_vertex_ids"], dtype=np.int64)],
                        np.asarray(relation["contact"].get("contact_segments_mm", []), dtype=np.float64).reshape((-1, 2, 3)),
                        record["mortise_surface"]["plane_alignment_travel_max_mm"],
                    ))
                    continue
                attachments_by_part[part_id].append(InterfaceAttachment(
                    interface_id=interface_id,
                    side=side,
                    vertices=interface_arrays[f"{stem}_{side}_vertices"],
                    faces=interface_arrays[f"{stem}_{side}_faces"],
                    outer_vertex_count=record["densified_boundary_point_count"],
                    outer_cap_face_count=record[f"{side}_surface"]["outer_boundary_cap_face_count"],
                    original_boundary_points=vertices[np.asarray(_loop["boundary_vertex_ids"], dtype=np.int64)],
                ))
    parts = []
    part_records = []
    for index, component in enumerate(components, start=1):
        part_id = f"P{index:02d}"
        source_vertices, source_faces, _lookup, _source_ids = build_local_mesh(
            vertices, faces, component
        )
        try:
            mesh, composition = compose_part_with_interfaces(
                source_vertices, source_faces, attachments_by_part[part_id]
            )
        except ValueError as exc:
            raise ValueError(f"{part_id}: {exc}") from exc
        source_face_origin_indices = np.asarray(
            composition.pop("source_face_origin_indices"), dtype=np.int64
        )
        source_mesh = trimesh.Trimesh(
            vertices=source_vertices, faces=source_faces, process=False
        )
        source_degenerate = int(np.count_nonzero(source_mesh.area_faces <= 1e-10))
        degenerate = int(np.count_nonzero(mesh.area_faces <= 1e-10))
        if not mesh.is_winding_consistent or degenerate > source_degenerate:
            edge_counts = np.bincount(
                mesh.edges_unique_inverse, minlength=len(mesh.edges_unique)
            )
            raise ValueError(
                f"{part_id}: joined exterior and interfaces are not a valid closed mesh "
                f"(watertight={mesh.is_watertight}, winding={mesh.is_winding_consistent}, "
                f"degenerate_faces={degenerate}, source_degenerate_faces={source_degenerate}, "
                f"open_edges={int(np.count_nonzero(edge_counts == 1))}, "
                f"over_shared_edges={int(np.count_nonzero(edge_counts > 2))})"
            )
        color_code = str(component.color_code)
        color_info = _color_info_for_component(color_code)
        part = {
            "part_id": part_id,
            "color_code": color_code,
            "color_name": color_info.get("name", color_code),
            "color_hex": color_info.get("hex", "#C8C8C8"),
            "filament_slot_index": color_info.get("filament_slot"),
            "color_resolution_status": color_resolution_status(color_info),
            "mesh": mesh,
        }
        if face_color_tokens is not None:
            completed_face_ids = np.asarray(component.global_faces)
            source_tokens = np.asarray(face_color_tokens)[completed_face_ids].astype(object)
            if len(source_tokens) != len(source_faces):
                raise ValueError(f"{part_id}: source face colors do not align with the mesh")
            recolored_faces = 0
            if recognized_components is not None:
                from .source_face_ownership import color_assigned_faces_as_part

                source_tokens, recolored_faces = color_assigned_faces_as_part(
                    source_tokens, completed_face_ids,
                    recognized_components[index - 1].global_faces, color_code,
                )
            composition["assigned_source_faces_recolored_to_part_color"] = recolored_faces
            source_tokens = source_tokens[source_face_origin_indices]
            generated_count = len(mesh.faces) - len(source_tokens)
            if generated_count < 0:
                raise ValueError(f"{part_id}: composition lost source triangles")
            if np.any(source_tokens != color_code):
                tokens = source_tokens.tolist() + [color_code] * generated_count
                infos = [_color_info_for_component(token) for token in tokens]
                part["face_color_hexes"] = [
                    info.get("hex", color_info.get("hex", "#C8C8C8")) for info in infos
                ]
                part["face_filament_slot_indices"] = [
                    info.get("filament_slot") for info in infos
                ]
                part["face_paint_color_tokens"] = tokens
        parts.append(part)
        part_records.append({
            "part_id": part_id,
            "face_count": int(len(mesh.faces)),
            "watertight": bool(mesh.is_watertight),
            "winding_consistent": bool(mesh.is_winding_consistent),
            "degenerate_face_count": degenerate,
            "source_degenerate_face_count": source_degenerate,
            "interface_role_index_sha256": role_index_sha256,
            "interface_roles": {
                item["interface_id"]: item["side"]
                for item in composition.get("interfaces", [])
            },
            "composition": composition,
        })
    if imprinted_pockets:
        from .imprinted_host_attachment import carve_imprinted_mortise

        part_positions = {part["part_id"]: index for index, part in enumerate(parts)}
        for (part_id, tenon_part_id, interface_id, pocket_vertices,
             pocket_faces, outer_count, cap_count, original_points, contact_segments,
             plane_alignment_mm) in imprinted_pockets:
            index = part_positions[part_id]
            tenon_index = _part_index(tenon_part_id)
            tenon_vertices, tenon_faces, _, _ = build_local_mesh(
                vertices, faces, components[tenon_index - 1]
            )
            # Join the open socket surface to its source shell before carving.
            cutter, _ = compose_part_with_interfaces(
                tenon_vertices, tenon_faces,
                [InterfaceAttachment(
                    interface_id=interface_id, side="mortise",
                    vertices=pocket_vertices, faces=pocket_faces,
                    outer_vertex_count=outer_count,
                    outer_cap_face_count=cap_count,
                    original_boundary_points=original_points,
                )],
            )
            carve_imprinted_mortise(
                parts[index], part_records[index], cutter.vertices, cutter.faces,
                interface_id, contact_segments,
                MAX_INTERFACE_EXTENSION_MM + plane_alignment_mm + 2.0,
            )
            part_records[part_positions[tenon_part_id]]["interface_roles"][interface_id] = "tenon"
            part_records[index]["interface_roles"][interface_id] = "mortise"
    if remove_detached_micro_shells_part_ids:
        from .detached_micro_shells import remove_detached_micro_shells

        known_ids = {part["part_id"] for part in parts}
        unknown_ids = set(remove_detached_micro_shells_part_ids) - known_ids
        if unknown_ids:
            raise ValueError(f"Unknown part IDs for micro-shell cleanup: {sorted(unknown_ids)}")
        for part, record in zip(parts, part_records):
            if part["part_id"] not in remove_detached_micro_shells_part_ids:
                continue
            record["user_requested_detached_micro_shell_cleanup"] = (
                remove_detached_micro_shells(part)
            )
            record["face_count"] = len(part["mesh"].faces)
            record["watertight"] = bool(part["mesh"].is_watertight)
            record["winding_consistent"] = bool(part["mesh"].is_winding_consistent)
    return parts, part_records
