from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .common import Component


@dataclass(frozen=True)
class ExplicitBodyMergeResult:
    components: list[Component]
    body_index: int
    original_to_effective: dict[int, int]
    record: dict


def _component_from_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    global_faces: np.ndarray,
    color_code: str,
) -> Component:
    selected_faces = faces[global_faces]
    triangles = vertices[selected_faces]
    points = triangles.reshape(-1, 3)
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    return Component(
        color_code=str(color_code),
        global_faces=np.asarray(global_faces, dtype=np.int64),
        face_count=int(len(global_faces)),
        area=float((np.linalg.norm(normals, axis=1) * 0.5).sum()),
        bbox_min=points.min(axis=0),
        bbox_max=points.max(axis=0),
        center=points.mean(axis=0),
    )


def parse_part_group(value: str) -> tuple[int, ...]:
    """Parse a stable recognition-space group such as ``P14+P04``."""
    raw_tokens = str(value).replace(",", "+").split("+")
    indices: list[int] = []
    for raw_token in raw_tokens:
        token = raw_token.strip().upper()
        if token.startswith("P"):
            token = token[1:]
        if not token or not token.isdigit() or int(token) <= 0:
            raise ValueError(
                "--merge-body-parts expects positive recognized part ids, "
                "for example P14+P04"
            )
        index = int(token)
        if index in indices:
            raise ValueError("--merge-body-parts must not repeat a part id")
        indices.append(index)
    if len(indices) < 2:
        raise ValueError("--merge-body-parts requires at least two part ids")
    return tuple(indices)


def merge_body_components(
    vertices: np.ndarray,
    faces: np.ndarray,
    components: list[Component],
    requested_indices: tuple[int, ...],
) -> ExplicitBodyMergeResult:
    """Merge recognized parts into one multi-material body.

    ``requested_indices`` are evaluated before the merge. The first part is the
    body/color anchor. Per-face materials live outside ``Component`` and are
    intentionally not modified here.
    """
    if len(requested_indices) < 2:
        raise ValueError("an explicit body merge requires at least two parts")
    invalid = [index for index in requested_indices if index > len(components)]
    if invalid:
        raise ValueError(
            "--merge-body-parts references unknown recognized parts: "
            + ", ".join(f"P{index:02d}" for index in invalid)
        )

    anchor_index = int(requested_indices[0])
    merged_index_set = {int(index) for index in requested_indices}
    anchor = components[anchor_index - 1]
    merged_faces = np.unique(
        np.concatenate(
            [
                np.asarray(components[index - 1].global_faces, dtype=np.int64)
                for index in requested_indices
            ]
        )
    )
    expected_face_count = sum(
        int(components[index - 1].face_count) for index in requested_indices
    )
    if len(merged_faces) != expected_face_count:
        raise ValueError("explicit body merge contains overlapping source faces")

    merged_component = _component_from_faces(
        vertices,
        faces,
        merged_faces,
        anchor.color_code,
    )
    effective_components: list[Component] = []
    original_to_effective: dict[int, int] = {}
    body_index = 0
    for original_index, component in enumerate(components, start=1):
        if original_index in merged_index_set and original_index != anchor_index:
            continue
        effective_index = len(effective_components) + 1
        effective_components.append(
            merged_component if original_index == anchor_index else component
        )
        if original_index == anchor_index:
            body_index = effective_index
        original_to_effective[original_index] = effective_index
    for original_index in merged_index_set:
        original_to_effective[original_index] = body_index

    members = []
    for index in requested_indices:
        component = components[index - 1]
        members.append(
            {
                "original_part_index": int(index),
                "color_code": str(component.color_code),
                "face_count": int(component.face_count),
            }
        )
    return ExplicitBodyMergeResult(
        components=effective_components,
        body_index=body_index,
        original_to_effective=original_to_effective,
        record={
            "mode": "explicit_multi_material_body_merge",
            "requested_original_part_indices": [
                int(index) for index in requested_indices
            ],
            "anchor_original_part_index": anchor_index,
            "effective_body_index": int(body_index),
            "members": members,
            "merged_face_count": int(len(merged_faces)),
            "output_component_count": int(len(effective_components)),
            "per_face_materials_preserved": True,
        },
    )
