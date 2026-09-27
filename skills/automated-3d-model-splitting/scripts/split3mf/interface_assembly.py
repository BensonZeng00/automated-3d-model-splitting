from __future__ import annotations

import re

import numpy as np

from .interface_surface import (
    MAX_INTERFACE_EXTENSION_MM,
    build_mortise_shell_probe,
    build_pairwise_interface_surfaces as build_interface_surface_pair,
    densify_closed_contour,
)
from .mesh import build_local_mesh
from .part_interface_composition import InterfaceAttachment, compose_part_with_interfaces
from .spatial_intersections import remove_true_self_intersections_3d
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
            points = densify_closed_contour(points)
            tenon_loop_index = int(loop.get("tenon_loop_index", -1))
            mortise_loop_index = int(loop.get("mortise_loop_index", -1))
            if tenon_loop_index < 0 or mortise_loop_index < 0:
                raise ValueError(f"{interface_id}: shared boundary is missing Stage 04 loop indices")
            stem = f"{interface_id.lower()}_loop_{loop_position:03d}"
            if not re.fullmatch(r"[a-z0-9_]+", stem):
                raise ValueError(f"invalid interface array key: {stem!r}")
            pair = build_interface_surface_pair(
                boundary_points_mm=points,
                insertion_direction=axis,
                socket_inward_direction=axis,
                scale_ratio=scale_ratio,
                clearance_mm=clearance_mm,
                mortise_shell_probe=mortise_probes[mortise_index],
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
                "boundary_densification_subdivisions": 3,
                "insertion_direction": relation.get("insertion_direction"),
                "socket_inward_direction": relation.get("socket_inward_direction"),
                "mating_axis_toward_mortise": axis.tolist(),
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
) -> tuple[list[dict], list[dict]]:
    """Attach every interface to each part's preserved exterior in one pass."""
    records = {
        (item["interface_id"], item["loop_position"]): item
        for item in interface_summary["interfaces"]
    }
    attachments_by_part: dict[str, list[InterfaceAttachment]] = {
        f"P{index:02d}": [] for index in range(1, len(components) + 1)
    }
    for relation in interface_plan["interfaces"]:
        interface_id = str(relation["interface_id"])
        for loop_position, _loop in enumerate(
            relation["contact"]["shared_boundary_loops"]
        ):
            record = records[(interface_id, loop_position)]
            stem = f"{interface_id.lower()}_loop_{loop_position:03d}"
            for side, part_field in (("tenon", "tenon_part"), ("mortise", "mortise_part")):
                part_id = str(relation[part_field])
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
        mesh, composition = compose_part_with_interfaces(
            source_vertices, source_faces, attachments_by_part[part_id]
        )
        source_mesh = trimesh.Trimesh(
            vertices=source_vertices, faces=source_faces, process=False
        )
        source_degenerate = int(np.count_nonzero(source_mesh.area_faces <= 1e-10))
        degenerate = int(np.count_nonzero(mesh.area_faces <= 1e-10))
        if not mesh.is_watertight or not mesh.is_winding_consistent or degenerate > source_degenerate:
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
        parts.append({
            "part_id": part_id,
            "color_code": color_code,
            "color_name": color_info.get("name", color_code),
            "color_hex": color_info.get("hex", "#C8C8C8"),
            "mesh": mesh,
        })
        part_records.append({
            "part_id": part_id,
            "face_count": int(len(mesh.faces)),
            "watertight": bool(mesh.is_watertight),
            "winding_consistent": bool(mesh.is_winding_consistent),
            "degenerate_face_count": degenerate,
            "source_degenerate_face_count": source_degenerate,
            "composition": composition,
        })
    return parts, part_records
