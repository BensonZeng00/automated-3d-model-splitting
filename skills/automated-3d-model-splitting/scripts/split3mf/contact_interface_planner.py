from __future__ import annotations

from collections import defaultdict

import numpy as np

from .direction_field import component_inward_direction


def plan_contact_interfaces(
    vertices: np.ndarray,
    faces: np.ndarray,
    components: list,
    recognized_boundaries,
    model_center: np.ndarray,
    recognition_records: list[dict] | None = None,
) -> dict:
    """Plan one independent tenon/mortise relation for each recognized contact.

    Side selection is based on the alignment of each part's geometric inward
    direction with the line between the two part centers. If both sides are
    equally plausible, the smaller recognized region becomes the tenon side;
    part index breaks an exact size tie. No root body or parent tree is used.
    """
    sample_owners: dict[int, dict[tuple[int, int], np.ndarray]] = defaultdict(dict)
    ordered_loop_vertices: dict[tuple[int, int], list[int]] = {}
    simplified_loop_points: dict[tuple[int, int], np.ndarray] = {}
    for component_index in range(1, len(components) + 1):
        loops = recognized_boundaries.loops_for_component(component_index)
        point_loops = recognized_boundaries.component_loop_points[component_index - 1]
        if len(loops) != len(point_loops):
            raise ValueError(f"boundary snapshot loop mismatch for P{component_index:02d}")
        for loop_index, (loop, points) in enumerate(zip(loops, point_loops)):
            if len(loop) != len(points):
                raise ValueError(f"boundary sample mismatch for P{component_index:02d}")
            ordered_loop_vertices[(component_index, loop_index)] = [
                int(vertex_id) for vertex_id in loop
            ]
            simplified_loop_points[(component_index, loop_index)] = np.asarray(
                points, dtype=np.float64
            )
            for vertex_id, point in zip(loop, points):
                sample_owners[int(vertex_id)][(component_index, loop_index)] = np.asarray(
                    point, dtype=np.float64
                )

    contact_samples: dict[tuple[int, int], dict[int, np.ndarray]] = defaultdict(dict)
    loop_contacts: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
    for vertex_id, owners_by_loop in sample_owners.items():
        owner_components = sorted({owner[0] for owner in owners_by_loop})
        for left_pos, left in enumerate(owner_components):
            for right in owner_components[left_pos + 1:]:
                pair = (left, right)
                left_owners = [
                    (owner, point) for owner, point in owners_by_loop.items()
                    if owner[0] == left
                ]
                right_owners = [
                    (owner, point) for owner, point in owners_by_loop.items()
                    if owner[0] == right
                ]
                points = [point for _owner, point in left_owners + right_owners]
                contact_samples[pair][vertex_id] = np.mean(points, axis=0)
                for left_owner, _left_point in left_owners:
                    for right_owner, _right_point in right_owners:
                        loop_contacts[pair].add((left_owner[1], right_owner[1]))

    # One isolated shared sample can occur where contours merely touch at a
    # vertex. Require two distinct shared samples to count a physical interface.
    contact_pairs = {
        pair: samples for pair, samples in contact_samples.items()
        if len(samples) >= 2
    }

    part_centers = {
        index: np.asarray(component.center, dtype=np.float64)
        for index, component in enumerate(components, start=1)
    }
    recognition_by_index = {
        int(record["part_index"]): record
        for record in (recognition_records or [])
    }

    def part_semantics(index: int) -> dict:
        record = recognition_by_index.get(index, {})
        visual_label = next((
            str(record[key]).strip()
            for key in ("region_review_label", "visual_semantic_label", "label", "semantic_label")
            if record.get(key) and str(record[key]).strip()
        ), None)
        return {
            "part": f"P{index:02d}",
            "visual_semantic_label": visual_label,
            "visual_semantic_status": "provided" if visual_label else "not_provided",
            "visual_semantic_description": record.get("visual_semantic_description"),
            "visual_semantic_confidence": record.get("visual_semantic_confidence"),
            "visual_semantic_confidence_score": record.get("visual_semantic_confidence_score"),
            "visual_semantic_evidence": record.get("visual_semantic_evidence"),
            "recognition_role": record.get("role"),
            "source_region_classification": record.get("source_region_classification"),
            "color_name": record.get("color_name"),
            "color_hex": record.get("color_hex"),
            "color_code": record.get("color_code"),
            "faces": int(record.get("faces", components[index - 1].face_count)),
            "area_mm2": round(float(record.get("area_mm2", components[index - 1].area)), 6),
            "center_mm": record.get("center", part_centers[index].round(6).tolist()),
        }

    part_records = [part_semantics(index) for index in range(1, len(components) + 1)]
    part_labels = {
        part["part"]: part["visual_semantic_label"]
        for part in part_records
    }

    def table_part_name(part_id: str) -> str:
        label = part_labels.get(part_id)
        return f"{part_id} {label}" if label else part_id

    inward = {
        index: component_inward_direction(
            vertices, faces, component, np.asarray(model_center, dtype=np.float64)
        )
        for index, component in enumerate(components, start=1)
    }
    relations = []
    for interface_number, (left, right) in enumerate(sorted(contact_pairs), start=1):
        samples = contact_pairs[(left, right)]
        left_center, right_center = part_centers[left], part_centers[right]
        axis = right_center - left_center
        axis_length = float(np.linalg.norm(axis))
        axis = axis / axis_length if axis_length > 1e-12 else np.zeros(3, dtype=np.float64)
        left_alignment = float(np.dot(inward[left], axis))
        right_alignment = float(np.dot(inward[right], -axis))
        score_gap = abs(left_alignment - right_alignment)
        if score_gap > 0.05:
            tenon, mortise = (left, right) if left_alignment > right_alignment else (right, left)
            decision = "inward_alignment"
        else:
            left_size = (float(components[left - 1].area), int(components[left - 1].face_count), left)
            right_size = (float(components[right - 1].area), int(components[right - 1].face_count), right)
            tenon, mortise = (left, right) if left_size <= right_size else (right, left)
            decision = "smaller_region_tiebreak"
        contact_points = np.asarray(list(samples.values()), dtype=np.float64)
        contact_center = (
            contact_points.mean(axis=0)
            if len(contact_points)
            else (left_center + right_center) * 0.5
        )
        relations.append({
            "interface_id": f"I{interface_number:03d}",
            "parts": [f"P{left:02d}", f"P{right:02d}"],
            "part_recognition": [part_semantics(left), part_semantics(right)],
            "contact": {
                "recognized_boundary_loops": [
                    {"part": f"P{a:02d}", "loop_index": int(b)}
                    for a, b in sorted({
                        (left, item[0]) for item in loop_contacts[(left, right)]
                    } | {
                        (right, item[1]) for item in loop_contacts[(left, right)]
                    })
                ],
                "shared_boundary_sample_count": len(samples),
                "contact_center_mm": contact_center.round(6).tolist(),
                "shared_boundary_loops": [
                    {
                        "tenon_loop_index": int(
                            left_loop if tenon == left else right_loop
                        ),
                        "mortise_loop_index": int(
                            right_loop if mortise == right else left_loop
                        ),
                        "boundary_vertex_ids": [
                            int(vertex_id) for vertex_id in ordered_loop_vertices[
                                (tenon, left_loop if tenon == left else right_loop)
                            ]
                        ],
                        "boundary_points_mm": [
                            point.round(6).tolist()
                            for point in simplified_loop_points[
                                (tenon, left_loop if tenon == left else right_loop)
                            ]
                        ],
                    }
                    for left_loop, right_loop in sorted(
                        loop_contacts[(left, right)]
                    )
                    if (
                        len(
                            set(ordered_loop_vertices[(left, left_loop)])
                            & set(ordered_loop_vertices[(right, right_loop)])
                        )
                        >= 2
                    )
                ],
            },
            "tenon_part": f"P{tenon:02d}",
            "mortise_part": f"P{mortise:02d}",
            "mating_axis_toward_mortise": (
                (part_centers[mortise] - part_centers[tenon])
                / max(float(np.linalg.norm(part_centers[mortise] - part_centers[tenon])), 1e-12)
            ).round(6).tolist(),
            "insertion_direction": inward[tenon].round(6).tolist(),
            "socket_inward_direction": inward[mortise].round(6).tolist(),
            "direction_evidence": {
                "method": decision,
                "P_left_inward_alignment": round(left_alignment, 6),
                "P_right_inward_alignment": round(right_alignment, 6),
                "alignment_margin": round(score_gap, 6),
                "part_areas_mm2": {
                    f"P{left:02d}": round(float(components[left - 1].area), 6),
                    f"P{right:02d}": round(float(components[right - 1].area), 6),
                },
            },
        })
    return {
        "schema": "contact-interface-plan/v1",
        "boundary_source": "recognized_simplified_smoothed_boundaries",
        "recognized_boundary_fingerprint": recognized_boundaries.fingerprint,
        "part_recognition_source": "stage03_component_recognition",
        "part_semantic_label_status": (
            "complete" if all(part["visual_semantic_label"] for part in part_records)
            else "incomplete_visual_labels"
        ),
        "parts": part_records,
        "interface_count": len(relations),
        "relation_table": {
            "columns": ["接口", "榫方", "卯方"],
            "rows": [
                [
                    relation["interface_id"],
                    table_part_name(relation["tenon_part"]),
                    table_part_name(relation["mortise_part"]),
                ]
                for relation in relations
            ],
        },
        "interfaces": relations,
    }
