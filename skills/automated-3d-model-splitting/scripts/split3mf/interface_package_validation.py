"""Validate exported Stage 05 parts after the 3MF has been read back."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

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
        degenerates = int(np.count_nonzero(mesh.area_faces <= 1e-10))
        nearest = cKDTree(mesh.vertices).query(original.vertices)[0]
        maximum_shift = float(nearest.max())
        connected = len(mesh.split()) == 1
        audit = {
            "part_id": part_id,
            "face_count": int(len(mesh.faces)),
            "watertight": bool(mesh.is_watertight),
            "winding_consistent": bool(mesh.is_winding_consistent),
            "single_component": connected,
            "degenerate_face_count": degenerates,
            "source_degenerate_face_count": source_degenerate,
            "maximum_export_vertex_shift_mm": maximum_shift,
        }
        if (
            not audit["watertight"]
            or not audit["winding_consistent"]
            or not connected
            or degenerates > source_degenerate
            or len(mesh.faces) != len(original.faces)
            or maximum_shift > 1e-4
        ):
            raise ValueError(f"{part_id}: exported 3MF geometry failed readback: {audit}")
        audits.append(audit)
    return audits
