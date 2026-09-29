"""Remove explicitly selected, negligible detached shells from a finished part."""

import numpy as np
import trimesh


FACE_COLOR_FIELDS = (
    "face_color_hexes", "face_filament_slot_indices", "face_paint_color_tokens",
)


def remove_detached_micro_shells(
    part: dict,
    *,
    max_shell_area_mm2: float = 0.2,
    max_shell_volume_mm3: float = 0.001,
    max_shell_diameter_mm: float = 2.0,
    max_total_area_fraction: float = 0.001,
) -> dict:
    """Keep the largest shell and remove tiny companions only when all guards pass.

    This is an explicit cleanup operation, not an automatic part classification.
    A substantial secondary shell causes an error rather than silent data loss.
    """
    mesh = part["mesh"]
    groups = trimesh.graph.connected_components(
        mesh.face_adjacency, nodes=np.arange(len(mesh.faces))
    )
    if len(groups) == 1:
        return {"removed_shell_count": 0, "removed_face_count": 0,
                "removed_area_mm2": 0.0, "removed_volume_mm3": 0.0}
    face_areas = mesh.area_faces
    main = max(groups, key=lambda group: float(face_areas[group].sum()))
    small = [group for group in groups if group is not main]
    stats = []
    for group in small:
        shell = mesh.submesh([group], append=True, repair=False)
        diameter = float(np.linalg.norm(shell.extents))
        item = (float(shell.area), abs(float(shell.volume)), diameter, len(group))
        if (item[0] > max_shell_area_mm2 or item[1] > max_shell_volume_mm3
                or item[2] > max_shell_diameter_mm):
            raise ValueError(
                f"{part['part_id']}: detached shell is too large for micro-shell cleanup "
                f"(area={item[0]:.6g} mm², volume={item[1]:.6g} mm³, "
                f"diameter={item[2]:.6g} mm)"
            )
        stats.append(item)
    removed_area = sum(item[0] for item in stats)
    if removed_area > float(mesh.area) * max_total_area_fraction:
        raise ValueError(
            f"{part['part_id']}: detached shells total {removed_area:.6g} mm² "
            "exceeds cleanup area budget"
        )
    keep = np.zeros(len(mesh.faces), dtype=bool)
    keep[main] = True
    for field in FACE_COLOR_FIELDS:
        if field in part and len(part[field]) != len(keep):
            raise ValueError(f"{part['part_id']}: {field} does not align with faces")
    cleaned = mesh.copy()
    cleaned.update_faces(keep)
    cleaned.remove_unreferenced_vertices()
    if not cleaned.is_watertight or not cleaned.is_winding_consistent:
        raise ValueError(f"{part['part_id']}: cleanup did not preserve a closed main shell")
    part["mesh"] = cleaned
    for field in FACE_COLOR_FIELDS:
        if field in part:
            part[field] = np.asarray(part[field])[keep].tolist()
    return {
        "removed_shell_count": len(small),
        "removed_face_count": int(np.count_nonzero(~keep)),
        "removed_area_mm2": removed_area,
        "removed_volume_mm3": sum(item[1] for item in stats),
    }
