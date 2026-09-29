"""Attach an ordinary mortise where the host source boundary is fragmented."""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .common import COLOR_INFO
from .validation import boolean_collapsed_face_audit


def _transfer_paint_tokens(old_part: dict, carved: trimesh.Trimesh) -> list[str]:
    """Preserve exterior paint and assign newly exposed socket walls to the host."""
    old_mesh = old_part["mesh"]
    base = str(old_part["color_code"])
    original_tokens = np.asarray(
        old_part.get("face_paint_color_tokens", [base] * len(old_mesh.faces))
    ).astype(str)
    old_centers = old_mesh.triangles_center
    tree = cKDTree(old_centers)
    tokens = []
    for start in range(0, len(carved.faces), 4096):
        centers = carved.triangles_center[start:start + 4096]
        _, candidate_ids = tree.query(centers, k=min(8, len(old_centers)))
        candidate_ids = np.asarray(candidate_ids).reshape((len(centers), -1))
        repeated_points = np.repeat(centers, candidate_ids.shape[1], axis=0)
        candidate_triangles = old_mesh.triangles[candidate_ids.reshape(-1)]
        closest = trimesh.triangles.closest_point(candidate_triangles, repeated_points)
        distances = np.linalg.norm(closest - repeated_points, axis=1).reshape(candidate_ids.shape)
        nearest = np.argmin(distances, axis=1)
        for row, position in enumerate(nearest):
            face_id = int(candidate_ids[row, position])
            tokens.append(
                str(original_tokens[face_id])
                if distances[row, position] <= 1e-4 else base
            )
    return tokens


def carve_imprinted_mortise(
    host_part: dict,
    host_record: dict,
    mortise_vertices: np.ndarray,
    mortise_faces: np.ndarray,
    interface_id: str,
) -> dict:
    """Use the paired interface's mortise solid as a local subtractive cutter."""
    cutter = trimesh.Trimesh(
        vertices=np.asarray(mortise_vertices, dtype=np.float64),
        faces=np.asarray(mortise_faces, dtype=np.int64),
        process=False,
    )
    if not cutter.is_watertight or not cutter.is_winding_consistent:
        raise ValueError(f"{interface_id}: mortise cutter is not a closed solid")
    original = host_part["mesh"]
    carved = trimesh.boolean.difference([original, cutter], engine="manifold")
    if not carved.is_watertight or not carved.is_winding_consistent:
        raise ValueError(f"{interface_id}: imprinted host is not a closed solid")
    degenerates = int(np.count_nonzero(carved.area_faces <= 1e-10))
    collapse = boolean_collapsed_face_audit(carved)
    new_degenerates = max(
        0, degenerates - int(host_record["source_degenerate_face_count"])
    )
    micro_limit = int(len(carved.faces) * 0.005)
    if (new_degenerates > micro_limit
            or int(collapse["collapsed_face_count"]) > micro_limit):
        raise ValueError(
            f"{interface_id}: imprinted host gained nonlocal or excessive microfaces "
            f"({degenerates} after, {host_record['degenerate_face_count']} before, "
            f"{host_record['source_degenerate_face_count']} source, "
            f"collapsed={collapse['collapsed_face_count']}, "
            f"maximum_allowed={micro_limit})"
        )
    removed_volume = float(original.volume - carved.volume)
    if removed_volume <= 1e-4:
        raise ValueError(f"{interface_id}: mortise cutter missed the host solid")
    tokens = _transfer_paint_tokens(host_part, carved)
    infos = [COLOR_INFO.get(token, {}) for token in tokens]
    host_part["mesh"] = carved
    host_part["face_paint_color_tokens"] = tokens
    host_part["face_color_hexes"] = [
        info.get("hex", host_part["color_hex"]) for info in infos
    ]
    host_part["face_filament_slot_indices"] = [
        info.get("filament_slot") for info in infos
    ]
    host_record["face_count"] = int(len(carved.faces))
    host_record["watertight"] = True
    host_record["winding_consistent"] = True
    host_record["degenerate_face_count"] = degenerates
    host_record["boolean_microface_allowance"] = new_degenerates
    host_record["boolean_collapsed_face_audit"] = collapse
    pocket = {
        "interface_id": interface_id,
        "strategy": "paired_mortise_solid_subtracted_from_fragmented_host",
        "removed_volume_mm3": removed_volume,
        "face_count_after": int(len(carved.faces)),
    }
    host_record["composition"].setdefault("imprinted_mortise_pockets", []).append(pocket)
    return pocket
