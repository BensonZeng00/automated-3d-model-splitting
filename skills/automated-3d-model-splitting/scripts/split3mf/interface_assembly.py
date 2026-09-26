from __future__ import annotations

import re

import numpy as np

from .interface_surface import (
    build_mortise_shell_probe,
    build_pairwise_interface_surfaces as build_interface_surface_pair,
)
from .mesh import build_local_mesh
from .spatial_intersections import remove_true_self_intersections_3d


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
            vertex_ids = np.asarray(loop.get("boundary_vertex_ids"), dtype=np.int64)
            if len(points) != len(vertex_ids):
                raise ValueError(f"{interface_id}: shared boundary ids and points do not correspond")
            if len(points) < 3:
                raise ValueError(f"{interface_id}: shared boundary loop has fewer than three points")
            if points.ndim != 2 or points.shape[1:] != (3,):
                raise ValueError(f"{interface_id}: shared boundary points must be an N×3 array")
            if len(np.unique(vertex_ids)) != len(vertex_ids):
                raise ValueError(f"{interface_id}: shared boundary contains duplicate source vertex ids")
            if not np.isfinite(points).all():
                raise ValueError(f"{interface_id}: shared boundary contains non-finite coordinates")
            points, vertex_ids, boundary_cleanup = remove_true_self_intersections_3d(
                points, vertex_ids
            )
            if len(points) < 3:
                raise ValueError(f"{interface_id}: 3D crossing cleanup left fewer than three boundary points")
            tenon_loop_index = int(loop.get("tenon_loop_index", -1))
            mortise_loop_index = int(loop.get("mortise_loop_index", -1))
            if tenon_loop_index < 0 or mortise_loop_index < 0:
                raise ValueError(f"{interface_id}: shared boundary is missing Stage 04 loop indices")
            stem = f"{interface_id.lower()}_loop_{loop_position:03d}"
            if not re.fullmatch(r"[a-z0-9_]+", stem):
                raise ValueError(f"invalid interface array key: {stem!r}")
            pair = build_interface_surface_pair(
                boundary_points_mm=points,
                insertion_direction=np.asarray(side_directions["tenon"], dtype=np.float64),
                socket_inward_direction=np.asarray(side_directions["mortise"], dtype=np.float64),
                scale_ratio=scale_ratio,
                clearance_mm=clearance_mm,
                mortise_shell_probe=mortise_probes[mortise_index],
                interface_id=interface_id,
            )
            paired_topology_matches = bool(
                np.array_equal(pair.tenon.faces, pair.mortise.faces)
            )
            boundary_count = len(points)
            tenon_inner = pair.tenon.vertices[boundary_count : 2 * boundary_count]
            mortise_inner = pair.mortise.vertices[boundary_count : 2 * boundary_count]
            measured_gaps = np.linalg.norm(mortise_inner - tenon_inner, axis=1)
            measured_gap = float(np.mean(measured_gaps))
            socket_direction = np.asarray(
                relation["socket_inward_direction"], dtype=np.float64
            )
            socket_direction /= max(float(np.linalg.norm(socket_direction)), 1e-12)
            tenon_tip = pair.tenon.vertices[2 * boundary_count : 3 * boundary_count]
            mortise_floor = pair.mortise.vertices[2 * boundary_count : 3 * boundary_count]
            measured_depth_clearance = float(
                np.mean(
                    (
                        (mortise_floor - tenon_tip)
                        - (mortise_inner - tenon_inner)
                    )
                    @ socket_direction
                )
            )
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
                "tenon_part": relation["tenon_part"],
                "mortise_part": relation["mortise_part"],
                "tenon_loop_index": tenon_loop_index,
                "mortise_loop_index": mortise_loop_index,
                "boundary_vertex_ids": vertex_ids.tolist(),
                "boundary_cleanup_3d": boundary_cleanup,
                "insertion_direction": relation.get("insertion_direction"),
                "socket_inward_direction": relation.get("socket_inward_direction"),
                "mating_axis_toward_mortise": axis.tolist(),
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
        "extension_depth_limit_mm": 10.0,
        "extension_depth_minimum_mm": 0.2,
        "extension_depth_policy": "halve_on_mortise_shell_hit_until_safe_or_minimum",
        "clearance_allocation": "mortise_only_full_side_and_bottom_clearance",
        "interface_surface_count": len(records),
        "interfaces": records,
    }
    return arrays, result


def build_pairwise_part_meshes(**_kwargs):
    """Part-mesh adaptation must consume Stage 04 geometry directly.

    The legacy source-loop remapping and arc-length expansion path has been
    removed. The replacement part-mesh integration is intentionally left for
    the next implementation pass.
    """
    raise NotImplementedError(
        "Stage 05 part-mesh adaptation is pending direct frozen-boundary integration"
    )
