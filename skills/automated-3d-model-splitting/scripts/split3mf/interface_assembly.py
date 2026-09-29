from __future__ import annotations

import re

import numpy as np

from .interface_surface import (
    InterfaceSurface,
    PairedInterfaceSurfaces,
    MAX_INTERFACE_EXTENSION_MM,
    MORTISE_COLLISION_DISTANCE_THRESHOLD_MM,
    MORTISE_COLLISION_RATIO_THRESHOLD,
    MORTISE_FAR_COLLISION_DISTANCE_MM,
    MORTISE_FAR_COLLISION_RATIO_THRESHOLD,
    build_mortise_shell_probe,
    build_pairwise_interface_surfaces as build_interface_surface_pair,
    _curved_cap_triangulation,
    _mortise_collision_statistics,
)
from .mesh import build_local_mesh, orthonormal_basis, radial_offset_points
from .part_interface_composition import InterfaceAttachment, compose_part_with_interfaces
from .project import color_resolution_status
from .spatial_intersections import (
    first_nonincident_triangle_intersection_3d,
    remove_true_self_intersections_3d,
)
import trimesh


def _part_index(part_id: str) -> int:
    match = re.fullmatch(r"P(\d+)", str(part_id))
    if match is None:
        raise ValueError(f"invalid Stage 04 part id: {part_id!r}")
    return int(match.group(1))


SIMPLIFIED_BOUNDARY_FALLBACK_DEPTH_MM = 1.0


def _build_simplified_boundary_fallback_pair(
    *,
    simplified_boundary: np.ndarray,
    insertion_axis: np.ndarray,
    clearance_mm: float,
    mortise_shell_probe: LocalRayProbe,
    interface_id: str,
) -> PairedInterfaceSurfaces:
    """Extrude the complete Stage 03 simplified contour after the 50% candidate fails."""
    outer = np.asarray(simplified_boundary, dtype=np.float64)
    axis = np.asarray(insertion_axis, dtype=np.float64)
    if outer.ndim != 2 or outer.shape[1:] != (3,) or len(outer) < 3:
        raise ValueError(f"{interface_id}: simplified fallback contour must be an N×3 loop")
    if not np.isfinite(outer).all() or not np.isfinite(axis).all():
        raise ValueError(f"{interface_id}: simplified fallback contour or axis is non-finite")
    axis_length = float(np.linalg.norm(axis))
    if axis_length <= 1e-10:
        raise ValueError(f"{interface_id}: simplified fallback axis is degenerate")
    axis /= axis_length
    count = len(outer)
    origin = outer.mean(axis=0)

    def build_side(side: str) -> InterfaceSurface:
        is_mortise = side == "mortise"
        clearance = float(clearance_mm) if is_mortise else 0.0
        depth = SIMPLIFIED_BOUNDARY_FALLBACK_DEPTH_MM + clearance
        if is_mortise:
            u, v = orthonormal_basis(axis)
            inner = radial_offset_points(outer, origin, u, v, axis, clearance)
        else:
            inner = outer.copy()
        inner = inner + axis * depth
        vertices = np.vstack((outer, inner))
        strip_faces = []
        for index in range(count):
            following = (index + 1) % count
            strip_faces.extend((
                (index, following, count + index),
                (following, count + following, count + index),
            ))
        cap_triangles, cap_record, cap_center = _curved_cap_triangulation(inner, axis)
        if cap_center is not None:
            raise ValueError(
                f"{interface_id} {side}: simplified 100% contour cap cannot be triangulated "
                "without a centre fan"
            )
        cap_faces = [
            tuple(int(vertex) + count for vertex in triangle)
            for triangle in cap_triangles
        ]
        outer_cap_faces, outer_cap_record, outer_cap_center = _curved_cap_triangulation(
            outer, axis
        )
        if outer_cap_center is not None:
            raise ValueError(
                f"{interface_id} {side}: simplified boundary cannot be triangulated "
                "without a centre fan"
            )
        faces = np.asarray(strip_faces + cap_faces + outer_cap_faces, dtype=np.int64)
        outer_cap_face_count = len(outer_cap_faces)
        construction_faces = faces[:-outer_cap_face_count]
        xyz_audit = first_nonincident_triangle_intersection_3d(
            construction_faces, vertices
        )
        if not xyz_audit["valid"]:
            raise ValueError(
                f"{interface_id} {side}: simplified-boundary 1 mm fallback has a 3D intersection: {xyz_audit}"
            )
        if is_mortise:
            sample = inner
            hit_distances, _ = mortise_shell_probe.exits(
                sample, np.broadcast_to(axis, sample.shape).copy(),
                depth + MORTISE_RAY_NUMERICAL_EPSILON_MM,
                epsilon=MORTISE_RAY_NUMERICAL_EPSILON_MM,
            )
            thickness = _mortise_collision_statistics(hit_distances)
            if thickness["collision"]:
                raise ValueError(
                    f"{interface_id}: simplified-boundary 1 mm fallback exceeds P01 thickness limits: {thickness}"
                )
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        trimesh.repair.fix_winding(mesh)
        trimesh.repair.fix_inversion(mesh, multibody=True)
        record = {
            "interface_id": interface_id,
            "side": side,
            "geometry_role": "additive_tenon_surface" if side == "tenon" else "subtractive_mortise_surface",
            "boundary_vertex_count": count,
            "scale_ratio": 1.0,
            "selected_strategy": "translated_simplified_boundary_1mm",
            "fallback_from_scale_ratio": 0.5,
            "inner_planar_contour": {"strategy": "full_simplified_boundary_translation"},
            "ellipse_plane_normal": axis.round(8).tolist(),
            "ellipse_extrusion_axis": axis.round(8).tolist(),
            "inner_contour_attempts": [{"scale_ratio": 0.5, "accepted": False},
                                       {"scale_ratio": 1.0, "accepted": True}],
            "clearance_mm": float(clearance_mm),
            "inner_ring_offset_mm": clearance,
            "outer_boundary_shift_mm": 0.0,
            "extension_direction": axis.round(8).tolist(),
            "stage04_mating_direction": axis.round(8).tolist(),
            "extension_depth_mm": depth,
            "tenon_extension_depth_mm": SIMPLIFIED_BOUNDARY_FALLBACK_DEPTH_MM,
            "mortise_recess_depth_mm": SIMPLIFIED_BOUNDARY_FALLBACK_DEPTH_MM + clearance_mm,
            "inner_cap_face_count": int(len(cap_faces)),
            "inner_cap_geometry": "translated_simplified_boundary_triangulation",
            "inner_cap_triangulation": cap_record,
            "outer_boundary_cap_face_count": int(outer_cap_face_count),
            "outer_boundary_cap_triangulation": outer_cap_record,
            "outer_boundary_closed_from_frozen_simplified_loop": True,
            "outer_to_extended_inner_face_count": int(len(strip_faces)),
            "quality_diagnostics": {
                "fallback_selected_after_50_percent_candidate_failure": True,
                "simplified_boundary_vertex_count": int(count),
                "three_dimensional_self_intersection": xyz_audit,
                "quality_gates_blocking": False,
            },
        }
        return InterfaceSurface(vertices=vertices, faces=np.asarray(mesh.faces), record=record)

    return PairedInterfaceSurfaces(
        tenon=build_side("tenon"),
        mortise=build_side("mortise"),
    )

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
    interfaces = interface_plan.get("interfaces")
    if not isinstance(interfaces, list):
        raise ValueError("Stage 04 interface plan has no interfaces list")

    arrays: dict[str, np.ndarray] = {}
    records: list[dict] = []
    mortise_probes = {}
    for relation in interfaces:
        interface_id = str(relation.get("interface_id", ""))
        if not interface_id:
            raise ValueError("Stage 04 interface entry is missing interface_id")
        if relation.get("tenon_part") not in relation.get("parts", []):
            raise ValueError(f"{interface_id}: tenon is not one of the declared parts")
        if relation.get("mortise_part") not in relation.get("parts", []):
            raise ValueError(f"{interface_id}: mortise is not one of the declared parts")
        mortise_index = _part_index(relation["mortise_part"])
        if not 1 <= mortise_index <= len(components):
            raise ValueError(f"{interface_id}: mortise references an unknown part")
        if mortise_index not in mortise_probes:
            shell_vertices, shell_faces, _lookup, _global_ids = build_local_mesh(
                vertices, faces, components[mortise_index - 1]
            )
            mortise_probes[mortise_index] = build_mortise_shell_probe(
                shell_vertices[shell_faces]
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
            points, _ephemeral_ids, boundary_cleanup = remove_true_self_intersections_3d(
                points
            )
            if len(points) < 3:
                raise ValueError(f"{interface_id}: 3D crossing cleanup left fewer than three boundary points")
            simplified_boundary_point_count = len(points)
            centered_points = points - points.mean(axis=0)
            contact_axis = np.cross(
                centered_points, np.roll(centered_points, -1, axis=0)
            ).sum(axis=0)
            contact_axis_length = float(np.linalg.norm(contact_axis))
            if contact_axis_length <= 1e-10:
                raise ValueError(f"{interface_id}: contact boundary has no stable local axis")
            contact_axis /= contact_axis_length
            if float(contact_axis @ axis) < 0.0:
                contact_axis = -contact_axis
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
            used_simplified_fallback = False
            try:
                pair = build_interface_surface_pair(
                    boundary_points_mm=points,
                    insertion_direction=contact_axis,
                    socket_inward_direction=np.asarray(
                        side_directions["mortise"], dtype=np.float64
                    ),
                    scale_ratio=scale_ratio,
                    clearance_mm=clearance_mm,
                    mortise_shell_probe=mortise_probes[mortise_index],
                    interface_id=interface_id,
                    extension_depth_limit_mm=MAX_INTERFACE_EXTENSION_MM,
                )
            except ValueError as primary_error:
                used_simplified_fallback = True
                pair = _build_simplified_boundary_fallback_pair(
                    simplified_boundary=simplified_points,
                    insertion_axis=contact_axis,
                    clearance_mm=clearance_mm,
                    mortise_shell_probe=mortise_probes[mortise_index],
                    interface_id=interface_id,
                )
                points = pair.tenon.vertices[:pair.tenon.record["boundary_vertex_count"]]
                simplified_boundary_point_count = len(points)
                pair.tenon.record["fallback_reason"] = str(primary_error)
                pair.mortise.record["fallback_reason"] = str(primary_error)
            paired_topology_matches = bool(
                np.array_equal(pair.tenon.faces, pair.mortise.faces)
            )
            boundary_count = len(points)
            socket_direction = np.asarray(
                pair.tenon.record["extension_direction"], dtype=np.float64
            )
            socket_direction /= np.linalg.norm(socket_direction)
            if float(np.dot(socket_direction, axis)) <= 1e-8:
                raise ValueError(
                    f"{interface_id}: construction axis does not point toward the mortise"
                )
            tenon_tip = pair.tenon.vertices[boundary_count : 2 * boundary_count]
            mortise_floor = pair.mortise.vertices[boundary_count : 2 * boundary_count]
            tip_difference = mortise_floor - tenon_tip
            axial_gaps = tip_difference @ socket_direction
            radial_gaps = tip_difference - axial_gaps[:, None] * socket_direction
            measured_gaps = np.linalg.norm(radial_gaps, axis=1)
            measured_gap = float(np.mean(measured_gaps))
            measured_depth_clearance = float(np.mean(axial_gaps))
            clearance_matches = bool(
                np.allclose(measured_gaps, clearance_mm, atol=1e-7, rtol=1e-7)
                and np.isclose(
                    measured_depth_clearance, clearance_mm, atol=1e-7, rtol=1e-7
                )
            )
            for side, surface in (("tenon", pair.tenon), ("mortise", pair.mortise)):
                arrays[f"{stem}_{side}_vertices"] = surface.vertices
                arrays[f"{stem}_{side}_faces"] = surface.faces
            records.append({
                "interface_id": interface_id,
                "loop_position": loop_position,
                "tenon_part": relation["tenon_part"],
                "mortise_part": relation["mortise_part"],
                "tenon_loop_index": tenon_loop_index,
                "mortise_loop_index": mortise_loop_index,
                "boundary_cleanup_3d": boundary_cleanup,
                "simplified_boundary_point_count": int(simplified_boundary_point_count),
                "densified_boundary_point_count": int(len(points)),
                "boundary_densification_subdivisions": 0,
                "selected_geometry_strategy": (
                    "translated_simplified_boundary_1mm"
                    if used_simplified_fallback else "scaled_inner_contour"
                ),
                "insertion_direction": relation.get("insertion_direction"),
                "socket_inward_direction": relation.get("socket_inward_direction"),
                "mating_axis_toward_mortise": axis.tolist(),
                "stage04_insertion_direction": insertion_axis.tolist(),
                "construction_axis_toward_mortise": socket_direction.tolist(),
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
        "source_interface_schema": interface_plan["schema"],
        "recognized_boundary_fingerprint": interface_plan.get(
            "recognized_boundary_fingerprint"
        ),
        "scale_ratio": float(scale_ratio),
        "total_clearance_mm": float(clearance_mm),
        "extension_depth_limit_mm": MAX_INTERFACE_EXTENSION_MM,
        "extension_depth_minimum_mm": 0.2,
        "extension_depth_policy": "halve_on_mortise_shell_hit_until_safe_or_minimum",
        "mortise_collision_distance_threshold_mm": MORTISE_COLLISION_DISTANCE_THRESHOLD_MM,
        "mortise_collision_ratio_threshold": MORTISE_COLLISION_RATIO_THRESHOLD,
        "mortise_far_collision_distance_mm": MORTISE_FAR_COLLISION_DISTANCE_MM,
        "mortise_far_collision_ratio_threshold": MORTISE_FAR_COLLISION_RATIO_THRESHOLD,
        "collision_distance_bands": "short: (0.1, 1.0] mm; far: (1.0, infinity) mm; OR",
        "collision_ratio_denominator": "all_probe_rays",
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
    attachments_by_part: dict[str, list[InterfaceAttachment]] = {
        f"P{index:02d}": [] for index in range(1, len(components) + 1)
    }
    imprinted_pockets = []
    for relation in interface_plan["interfaces"]:
        interface_id = str(relation["interface_id"])
        for loop_position, _loop in enumerate(
            relation["contact"]["shared_boundary_loops"]
        ):
            record = records[(interface_id, loop_position)]
            stem = f"{interface_id.lower()}_loop_{loop_position:03d}"
            for side, part_field in (("tenon", "tenon_part"), ("mortise", "mortise_part")):
                part_id = str(relation[part_field])
                if (side == "mortise" and relation["contact"].get("counterpart_mode")
                        == "imprinted_host_surface"):
                    imprinted_pockets.append((
                        part_id, interface_id,
                        interface_arrays[f"{stem}_{side}_vertices"],
                        interface_arrays[f"{stem}_{side}_faces"],
                    ))
                    continue
                attachments_by_part[part_id].append(InterfaceAttachment(
                    interface_id=interface_id,
                    side=side,
                    vertices=interface_arrays[f"{stem}_{side}_vertices"],
                    faces=interface_arrays[f"{stem}_{side}_faces"],
                    outer_vertex_count=record["densified_boundary_point_count"],
                    outer_cap_face_count=record[f"{side}_surface"]["outer_boundary_cap_face_count"],
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
            source_tokens = np.asarray(face_color_tokens)[component.global_faces].astype(object)
            if len(source_tokens) != len(source_faces):
                raise ValueError(f"{part_id}: source face colors do not align with the mesh")
            recolored_faces = 0
            if recognized_components is not None:
                from .source_face_ownership import color_assigned_faces_as_part

                source_tokens, recolored_faces = color_assigned_faces_as_part(
                    source_tokens, component.global_faces,
                    recognized_components[index - 1].global_faces, color_code,
                )
            composition["assigned_source_faces_recolored_to_part_color"] = recolored_faces
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
            "composition": composition,
        })
    if imprinted_pockets:
        from .imprinted_host_attachment import carve_imprinted_mortise

        part_positions = {part["part_id"]: index for index, part in enumerate(parts)}
        for part_id, interface_id, pocket_vertices, pocket_faces in imprinted_pockets:
            index = part_positions[part_id]
            carve_imprinted_mortise(
                parts[index], part_records[index], pocket_vertices, pocket_faces,
                interface_id,
            )
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
