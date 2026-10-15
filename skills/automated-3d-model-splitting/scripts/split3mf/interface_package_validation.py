"""Validate exported Stage 05 parts after the 3MF has been read back."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .topology import audit_mesh_topology, topology_defects_within_tolerance
from .package_io import load_colored_mesh_objects_3mf


def validate_reloaded_interface_parts(
    path: Path,
    expected_parts: list[dict],
    part_records: list[dict],
) -> list[dict]:
    loaded = load_colored_mesh_objects_3mf(path)
    expected = {part["part_id"]: part for part in expected_parts}
    records = {record["part_id"]: record for record in part_records}
    if len(loaded) != len(expected) or {part["part_id"] for part in loaded} != set(expected):
        raise ValueError("exported 3MF does not contain exactly the planned parts")
    audits = []
    for part in loaded:
        part_id = part["part_id"]
        mesh = part["mesh"]
        original = expected[part_id]["mesh"]
        source_degenerate = int(records[part_id]["source_degenerate_face_count"])
        boolean_allowance = int(records[part_id].get("boolean_microface_allowance", 0))
        degenerates = int(np.count_nonzero(mesh.area_faces <= 1e-10))
        nearest = cKDTree(mesh.vertices).query(original.vertices)[0]
        maximum_shift = float(nearest.max())
        component_count = len(mesh.split())
        expected_component_count = len(original.split())
        topology = audit_mesh_topology(mesh)
        audit = {
            "part_id": part_id,
            "face_count": int(len(mesh.faces)),
            "watertight": topology["watertight"],
            "topology_defect_edges": topology["defect_edges"],
            "topology_unique_edges": topology["unique_edges"],
            "topology_defect_ratio": topology["topology_defect_ratio"],
            "topology_tolerance_accepted": topology_defects_within_tolerance(
                topology
            ),
            "winding_consistent": bool(mesh.is_winding_consistent),
            "single_component": component_count == 1,
            "component_count": component_count,
            "expected_component_count": expected_component_count,
            "degenerate_face_count": degenerates,
            "source_degenerate_face_count": source_degenerate,
            "boolean_microface_allowance": boolean_allowance,
            "maximum_export_vertex_shift_mm": maximum_shift,
        }
        if (
            not audit["winding_consistent"]
            or component_count != expected_component_count
            or degenerates > source_degenerate + boolean_allowance
            or len(mesh.faces) != len(original.faces)
            or maximum_shift > 1e-4
        ):
            raise ValueError(f"{part_id}: exported 3MF geometry failed readback: {audit}")
        audits.append(audit)
    return audits
