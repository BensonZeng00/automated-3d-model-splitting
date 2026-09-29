from __future__ import annotations

from collections import defaultdict

import numpy as np

from .boundary_matching import (
    MINIMUM_BOUNDARY_COVERAGE,
    compare_boundary_loops,
    compare_loop_to_boundary_segments,
)
from .mesh import average_outward_normal, build_local_mesh, boundary_loops
from .source_face_ownership import complete_component_face_ownership
from .application.boundary_snapshot_builder import MAX_SMOOTHING_DISPLACEMENT_MM


def _component_inward_direction(
    vertices: np.ndarray, faces: np.ndarray, component, model_center: np.ndarray
) -> np.ndarray:
    local_vertices, local_faces, _, _ = build_local_mesh(vertices, faces, component)
    component_center = local_vertices[local_faces.reshape(-1)].mean(axis=0)
    return -average_outward_normal(
        local_vertices, local_faces, component_center, model_center
    )


def _contact_axis_toward_host(
    loops: list[np.ndarray], fallback: np.ndarray, toward_host: np.ndarray
) -> tuple[np.ndarray, str]:
    """Use the joined seam's local area normal for the insertion direction."""
    area_vector = np.zeros(3, dtype=np.float64)
    for points in loops:
        centered = points - points.mean(axis=0)
        area_vector += np.cross(centered, np.roll(centered, -1, axis=0)).sum(axis=0)
    length = float(np.linalg.norm(area_vector))
    if length <= 1e-10:
        direction = np.asarray(fallback, dtype=np.float64).copy()
        source = "component_inward_fallback"
    else:
        direction = area_vector / length
        source = "contact_boundary_area_normal"
    if float(np.dot(direction, toward_host)) < 0.0:
        direction = -direction
    return direction, source


def _fragmented_host_contacts(
    vertices: np.ndarray,
    faces: np.ndarray,
    components: list,
    simplified_loop_points: dict[tuple[int, int], np.ndarray],
    recognition_records: list[dict],
    already_contacted: set[int],
) -> list[tuple[int, int, int, dict]]:
    """Find a clear patch seam supported by several host boundary fragments.

    A painted patch can have one continuous visible contour while the host
    triangulation divides its other side into many tiny open loops. Verify the
    whole contour against their union, with a unique nearby host, rather than
    matching it to an unrelated complete loop.
    """
    confirmed = {
        int(record["part_index"])
        for record in recognition_records
        if record.get("source_region_classification") == "part"
    }
    candidates = sorted(confirmed - already_contacted)
    if not candidates:
        return []
    completed, _ = complete_component_face_ownership(
        vertices, faces, components,
        protected_part_ids={f"P{index:02d}" for index in confirmed},
    )
    host_boundaries: dict[int, np.ndarray] = {}
    results = []
    for candidate in candidates:
        loops = [
            (index, points)
            for (part, index), points in simplified_loop_points.items()
            if part == candidate and len(points) >= 3
        ]
        if not loops:
            continue
        # Merged semantic regions can contain several independently attached
        # contours. Evaluate each contour against the host boundary union;
        # choosing only the largest silently drops the remaining interfaces.
        for loop_index, points in loops:
            frozen_edges = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
            median_spacing = float(np.median(frozen_edges))
            smoothing = MAX_SMOOTHING_DISPLACEMENT_MM
            tolerance = min(0.60, max(0.08, 0.30 * median_spacing, 0.06 * float(np.ptp(points, axis=0).max())))
            scored = []
            for host in range(1, len(components) + 1):
                if host == candidate:
                    continue
                host_component = components[host - 1]
                if (
                    np.any(points.min(axis=0) > host_component.bbox_max + 2.0)
                    or np.any(host_component.bbox_min > points.max(axis=0) + 2.0)
                ):
                    continue
                if host not in host_boundaries:
                    local_vertices, local_faces, *_ = build_local_mesh(
                        vertices, faces, completed[host - 1]
                    )
                    host_loops = boundary_loops(local_faces, local_vertices)
                    host_boundaries[host] = (
                        np.concatenate([
                            np.stack((local_vertices[loop], np.roll(local_vertices[loop], -1, axis=0)), axis=1)
                            for loop in host_loops if len(loop) >= 2
                        ], axis=0)
                        if host_loops else np.empty((0, 2, 3), dtype=np.float64)
                    )
                boundary_segments = host_boundaries[host]
                if not len(boundary_segments):
                    continue
                comparison = compare_loop_to_boundary_segments(points, boundary_segments, tolerance)
                if comparison is None or not comparison["matched"]:
                    continue
                loop_to_host = comparison["loop_to_host"]
                host_to_loop = comparison["host_to_loop"]
                p90 = max(loop_to_host["p90_distance_mm"], host_to_loop["p90_distance_mm"])
                coverage = min(loop_to_host["coverage"], host_to_loop["coverage"])
                scored.append((p90, -coverage, host, {
                    "method": "fragmented_host_boundary_union",
                    "median_frozen_edge_mm": median_spacing,
                    "smoothing_allowance_mm": smoothing,
                    **comparison,
                    "required_boundary_coverage": MINIMUM_BOUNDARY_COVERAGE,
                }))
            scored.sort()
            if scored and (len(scored) == 1 or scored[1][0] > scored[0][0] * 2.0):
                results.append((scored[0][2], candidate, loop_index, scored[0][3]))
    return results


from .common import DEFAULT_AREA_PRIORITY_RATIO


def plan_contact_interfaces(
    vertices: np.ndarray,
    faces: np.ndarray,
    components: list,
    recognized_boundaries,
    model_center: np.ndarray,
    recognition_records: list[dict] | None = None,
    direction_policy_overrides: dict[str, str] | None = None,
    area_priority_ratio: float = DEFAULT_AREA_PRIORITY_RATIO,
) -> dict:
    """Plan one independent tenon/mortise relation for each recognized contact.

    A large area ratio prefers the smaller region before inward alignment.
    Otherwise side selection is based on each part's geometric inward
    direction with the line between the two part centers. If both sides are
    equally plausible, the smaller recognized region becomes the tenon side;
    part index breaks an exact size tie. No root body or parent tree is used.
    """
    if not np.isfinite(area_priority_ratio) or area_priority_ratio <= 1.0:
        raise ValueError("area_priority_ratio must be finite and greater than 1")
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
    contact_samples: dict[tuple[int, int], dict[int, np.ndarray]] = defaultdict(dict)
    loop_contacts: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
    for (left, left_loop), left_points in simplified_loop_points.items():
        for (right, right_loop), right_points in simplified_loop_points.items():
            if left >= right:
                continue
            comparison = compare_boundary_loops(left_points, right_points)
            if comparison is None or not comparison["matched"]:
                continue
            pair = (left, right)
            loop_contacts[pair].add((left_loop, right_loop))
            sample_offset = len(contact_samples[pair])
            for offset, point in enumerate(left_points):
                contact_samples[pair][sample_offset + offset] = point
    contacted_parts = {part for pair in loop_contacts for part in pair}
    imprinted_pairs: dict[tuple[int, int], tuple[int, dict]] = {}
    for host, candidate, loop_index, evidence in _fragmented_host_contacts(
        vertices, faces, components, simplified_loop_points,
        recognition_records or [], contacted_parts,
    ):
        left, right = sorted((host, candidate))
        pair = (left, right)
        left_loop = loop_index if left == candidate else -1
        right_loop = loop_index if right == candidate else -1
        loop_contacts[pair].add((left_loop, right_loop))
        points = simplified_loop_points[(candidate, loop_index)]
        offset = len(contact_samples[pair])
        for index, point in enumerate(points):
            contact_samples[pair][offset + index] = point
        imprinted_pairs.setdefault(pair, (candidate, evidence))
    matched_loops: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
    for (left, right), contacts in loop_contacts.items():
        for left_loop, right_loop in contacts:
            if left_loop >= 0:
                matched_loops[(left, left_loop)].add((right, right_loop))
            if right_loop >= 0:
                matched_loops[(right, right_loop)].add((left, left_loop))
    ambiguous = [key for key, matches in matched_loops.items() if len(matches) > 1]
    if ambiguous:
        raise ValueError(f"simplified boundary has ambiguous contact matches: {ambiguous}")
    contact_pairs = dict(contact_samples)

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
        index: _component_inward_direction(
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
        interface_id = f"I{interface_number:03d}"
        direction_policy = (direction_policy_overrides or {}).get(interface_id)
        left_size = (float(components[left - 1].area), int(components[left - 1].face_count), left)
        right_size = (float(components[right - 1].area), int(components[right - 1].face_count), right)
        smaller, larger = sorted((left_size[0], right_size[0]))
        area_ratio = larger / smaller if smaller > 0 and np.isfinite(larger) else None
        area_order = (left, right) if left_size <= right_size else (right, left)
        if direction_policy == "smaller_area_first":
            tenon, mortise = area_order
            decision = "user_requested_smaller_area_first"
        elif (left, right) in imprinted_pairs:
            tenon = imprinted_pairs[(left, right)][0]
            mortise = right if tenon == left else left
            decision = "fragmented_host_boundary_union"
        elif area_ratio is not None and (
            area_ratio >= area_priority_ratio
            or np.isclose(area_ratio, area_priority_ratio, rtol=1e-12, atol=0.0)
        ):
            tenon, mortise = area_order
            decision = "large_area_difference"
        elif score_gap > 0.05:
            tenon, mortise = (left, right) if left_alignment > right_alignment else (right, left)
            decision = "inward_alignment"
        else:
            tenon, mortise = area_order
            decision = "smaller_region_tiebreak"
        contact_points = np.asarray(list(samples.values()), dtype=np.float64)
        contact_center = (
            contact_points.mean(axis=0)
            if len(contact_points)
            else (left_center + right_center) * 0.5
        )
        toward_host = part_centers[mortise] - part_centers[tenon]
        tenon_contact_loops = [
            simplified_loop_points[(tenon, left_loop if tenon == left else right_loop)]
            for left_loop, right_loop in sorted(loop_contacts[(left, right)])
        ]
        insertion_axis, insertion_axis_source = _contact_axis_toward_host(
            tenon_contact_loops, inward[tenon], toward_host
        )
        relations.append({
            "interface_id": interface_id,
            "parts": [f"P{left:02d}", f"P{right:02d}"],
            "part_recognition": [part_semantics(left), part_semantics(right)],
            "contact": {
                "recognized_boundary_loops": [
                    {"part": f"P{a:02d}", "loop_index": int(b)}
                    for a, b in sorted({
                        (left, item[0]) for item in loop_contacts[(left, right)] if item[0] >= 0
                    } | {
                        (right, item[1]) for item in loop_contacts[(left, right)] if item[1] >= 0
                    })
                ],
                "counterpart_mode": (
                    "imprinted_host_surface" if (left, right) in imprinted_pairs
                    else "matched_source_loops"
                ),
                "counterpart_evidence": (
                    imprinted_pairs[(left, right)][1]
                    if (left, right) in imprinted_pairs else {}
                ),
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
                ],
            },
            "tenon_part": f"P{tenon:02d}",
            "mortise_part": f"P{mortise:02d}",
            "mating_axis_toward_mortise": (
                (part_centers[mortise] - part_centers[tenon])
                / max(float(np.linalg.norm(part_centers[mortise] - part_centers[tenon])), 1e-12)
            ).round(6).tolist(),
            "insertion_direction": insertion_axis.round(6).tolist(),
            "socket_inward_direction": inward[mortise].round(6).tolist(),
            "direction_evidence": {
                "method": decision,
                "insertion_axis_source": insertion_axis_source,
                "insertion_axis_center_alignment": round(float(
                    insertion_axis @ (
                        toward_host / max(float(np.linalg.norm(toward_host)), 1e-12)
                    )
                ), 6),
                "area_ratio": area_ratio,
                "area_priority_ratio": float(area_priority_ratio),
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
