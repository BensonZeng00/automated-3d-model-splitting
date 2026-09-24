from __future__ import annotations

from .common import *
from .project import *
from .recognition import *
from .mesh import *
from .selection import *

def component_boundary_neighbor_lookup(faces: np.ndarray, components: list[Component]) -> dict[tuple[int, int], set[int]]:
    face_to_component: dict[int, int] = {}
    for index, component in enumerate(components, start=1):
        for face_index in component.global_faces:
            face_to_component[int(face_index)] = index

    edge_to_components: dict[tuple[int, int], set[int]] = collections.defaultdict(set)
    for face_index, tri in enumerate(faces):
        component_index = face_to_component.get(face_index)
        if component_index is None:
            continue
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge_to_components[tuple(sorted((int(a), int(b))))].add(component_index)
    return edge_to_components


def component_boundary_loop_neighbors(
    vertices: np.ndarray,
    faces: np.ndarray,
    component: Component,
    component_index: int,
    boundary_neighbor_lookup: dict[tuple[int, int], set[int]],
) -> list[dict]:
    _local_vertices, local_faces, _global_to_local, global_vertex_ids = build_local_mesh(vertices, faces, component)
    loops = boundary_loops(local_faces)
    records = []
    for loop_index, loop in enumerate(loops):
        neighbor_counts: collections.Counter[int] = collections.Counter()
        for position, local_a in enumerate(loop):
            local_b = loop[(position + 1) % len(loop)]
            global_edge = tuple(sorted((int(global_vertex_ids[local_a]), int(global_vertex_ids[local_b]))))
            for neighbor_index in boundary_neighbor_lookup.get(global_edge, set()):
                if int(neighbor_index) != component_index:
                    neighbor_counts[int(neighbor_index)] += 1
        records.append(
            {
                "component_index": component_index,
                "loop_index": loop_index,
                "edge_count": len(loop),
                "neighbor_counts": {str(key): int(value) for key, value in sorted(neighbor_counts.items())},
            }
        )
    return records


def rebuild_assembly_children(parents: dict[int, int | None]) -> dict[int, list[int]]:
    children: dict[int, list[int]] = collections.defaultdict(list)
    for child_index, parent_index in parents.items():
        if parent_index is not None:
            children[int(parent_index)].append(int(child_index))
    return {parent: sorted(values) for parent, values in children.items()}


def adjacency_record_between(adjacency: dict[tuple[int, int], dict], left: int | None, right: int | None) -> dict | None:
    if left is None or right is None:
        return None
    return adjacency.get(tuple(sorted((int(left), int(right)))))


def build_loop_cache(loop_records: list[dict]) -> dict[int, list[dict]]:
    loop_cache: dict[int, list[dict]] = collections.defaultdict(list)
    for record in loop_records:
        loop_cache[int(record["component_index"])].append(record)
    return loop_cache


def best_loop_contact(
    loop_cache: dict[int, list[dict]],
    parent_index: int,
    child_index: int,
    excluded_neighbor_index: int | None = None,
    require_separate: bool = False,
    min_edges: int = 3,
) -> dict | None:
    best: dict | None = None
    for loop_record in loop_cache.get(int(parent_index), []):
        counts = loop_record.get("neighbor_counts", {})
        child_edges = int(counts.get(str(int(child_index)), 0))
        if child_edges < int(min_edges):
            continue
        excluded_edges = int(counts.get(str(int(excluded_neighbor_index)), 0)) if excluded_neighbor_index is not None else 0
        if require_separate and excluded_edges >= int(min_edges):
            continue
        candidate = {
            "parent_loop_index": int(loop_record["loop_index"]),
            "parent_loop_edges": int(loop_record["edge_count"]),
            "child_edges_on_parent_loop": child_edges,
            "excluded_neighbor_index": excluded_neighbor_index,
            "excluded_neighbor_edges_on_parent_loop": excluded_edges,
            "child_loop_dominance": float(child_edges) / max(float(loop_record["edge_count"]), 1.0),
        }
        if best is None:
            best = candidate
            continue
        best_key = (
            candidate["child_edges_on_parent_loop"],
            candidate["child_loop_dominance"],
            -candidate["excluded_neighbor_edges_on_parent_loop"],
            -candidate["parent_loop_edges"],
        )
        current_key = (
            best["child_edges_on_parent_loop"],
            best["child_loop_dominance"],
            -best["excluded_neighbor_edges_on_parent_loop"],
            -best["parent_loop_edges"],
        )
        if best_key > current_key:
            best = candidate
    return best


def component_is_larger(candidate: Component, component: Component) -> bool:
    return candidate.area > component.area or candidate.face_count > component.face_count


def component_size_sort_key(component: Component) -> tuple[float, int]:
    return (float(component.area), int(component.face_count))


def parse_part_index_tokens(raw: str | None) -> set[int]:
    indices: set[int] = set()
    if not raw:
        return indices
    for token in re.split(r"[,\s]+", raw):
        token = token.strip()
        if not token:
            continue
        match = re.search(r"p?0*([0-9]+)", token, re.IGNORECASE)
        if match:
            indices.add(int(match.group(1)))
    return indices


def parse_part_mode_overrides(raw: str | None) -> dict[int, str]:
    """Parse legacy-compatible P10=inward processing overrides."""
    overrides: dict[int, str] = {}
    if not raw:
        return overrides
    for token in re.split(r"[,\s]+", raw):
        token = token.strip()
        if not token:
            continue
        match = re.fullmatch(r"p?0*([0-9]+)[:=](inward)", token, re.IGNORECASE)
        if not match:
            raise ValueError(
                f"Invalid part mode override {token!r}; only inward geometry is supported, for example P10=inward"
            )
        overrides[int(match.group(1))] = match.group(2).lower()
    return overrides


CONFIDENCE_SCORES = {
    "VERYLOW": 0.1,
    "LOW": 0.35,
    "MED": 0.65,
    "MEDIUM": 0.65,
    "HIGH": 0.9,
    "UNKNOWN": 0.0,
}


def confidence_score(value, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    token = str(value).strip().upper()
    return CONFIDENCE_SCORES.get(token, default)


def parse_part_index(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return int(value)
    text = str(value).strip()
    match = re.search(r"p?0*([0-9]+)", text, re.IGNORECASE)
    return int(match.group(1)) if match else None


def load_visual_semantics(path: Path | None) -> dict:
    if path is None:
        return {
            "source": None,
            "parts": {},
            "parent_relations": [],
            "interface_retreats": [],
        }
    raw = json.loads(path.read_text(encoding="utf-8"))
    parts: dict[int, dict] = {}
    raw_parts = raw.get("parts", {})
    if isinstance(raw_parts, dict):
        iterable = raw_parts.items()
    elif isinstance(raw_parts, list):
        iterable = [(item.get("part") or item.get("part_id") or item.get("part_index"), item) for item in raw_parts if isinstance(item, dict)]
    else:
        iterable = []

    for key, value in iterable:
        index = parse_part_index(key)
        if index is None or not isinstance(value, dict):
            continue
        label = value.get("label") or value.get("name") or value.get("type") or value.get("semantic_label") or ""
        description = value.get("description") or value.get("role") or value.get("semantic_description") or ""
        confidence = value.get("confidence", "UNKNOWN")
        parts[int(index)] = {
            "part_index": int(index),
            "label": str(label),
            "description": str(description),
            "confidence": confidence,
            "confidence_score": confidence_score(confidence),
            "visual_evidence": value.get("visual_evidence") or value.get("evidence") or "",
            "source_views": value.get("source_views", []),
            "force_inward_vector": value.get("force_inward_vector"),
            "force_parent_direction": bool(value.get("force_parent_direction", False)),
            "guided_internal_cut": value.get("guided_internal_cut"),
        }

    raw_relations = raw.get("parent_relations") or raw.get("relations") or []
    parent_relations = []
    for item in raw_relations:
        if not isinstance(item, dict):
            continue
        child_index = parse_part_index(item.get("child") or item.get("child_part") or item.get("part"))
        parent_index = parse_part_index(item.get("parent") or item.get("parent_part"))
        if child_index is None or parent_index is None:
            continue
        confidence = item.get("confidence", "UNKNOWN")
        parent_relations.append(
            {
                "child_index": int(child_index),
                "parent_index": int(parent_index),
                "relation": str(item.get("relation") or item.get("type") or "visual_parent"),
                "confidence": confidence,
                "confidence_score": confidence_score(confidence),
                "reason": str(item.get("reason") or item.get("visual_evidence") or item.get("evidence") or ""),
                "source_views": item.get("source_views", []),
                "apply": bool(item.get("apply", item.get("apply_parent_override", True))),
            }
        )

    interface_retreats = []
    for item in raw.get("interface_retreats", []):
        if not isinstance(item, dict):
            continue
        child_index = parse_part_index(
            item.get("child") or item.get("child_part")
        )
        parent_index = parse_part_index(
            item.get("parent") or item.get("parent_part")
        )
        if child_index is None or parent_index is None:
            continue
        confidence = item.get("confidence", "UNKNOWN")
        interface_retreats.append(
            {
                "child_index": int(child_index),
                "parent_index": int(parent_index),
                "seed_point_mm": item.get("seed_point_mm"),
                "seed_radius_mm": item.get("seed_radius_mm", 2.0),
                "retreat_distance_mm": item.get("retreat_distance_mm"),
                "maximum_parent_face_fraction": item.get(
                    "maximum_parent_face_fraction",
                    0.25,
                ),
                "confidence": confidence,
                "confidence_score": confidence_score(confidence),
                "reason": str(
                    item.get("reason")
                    or item.get("visual_evidence")
                    or item.get("evidence")
                    or ""
                ),
                "source_views": item.get("source_views", []),
                "apply": bool(item.get("apply", True)),
            }
        )

    return {
        "source": str(path),
        "parts": parts,
        "parent_relations": parent_relations,
        "interface_retreats": interface_retreats,
        "raw": raw,
    }


def annotate_recognition_with_visual_semantics(records: list[dict], part_semantics: dict[int, dict]) -> list[dict]:
    annotated = []
    for record in records:
        updated = dict(record)
        semantic = part_semantics.get(int(record["part_index"]))
        if semantic:
            updated["visual_semantic_label"] = semantic.get("label", "")
            updated["visual_semantic_description"] = semantic.get("description", "")
            updated["visual_semantic_confidence"] = semantic.get("confidence", "UNKNOWN")
            updated["visual_semantic_confidence_score"] = semantic.get("confidence_score", 0.0)
            updated["visual_semantic_evidence"] = semantic.get("visual_evidence", "")
        annotated.append(updated)
    return annotated


def apply_visual_semantic_parent_relations(
    components: list[Component],
    adjacency: dict[tuple[int, int], dict],
    parents: dict[int, int | None],
    records: list[dict],
    relations: list[dict],
    min_confidence: float,
) -> tuple[dict[int, int | None], dict[int, list[int]], list[dict], dict]:
    refined_parents = dict(parents)
    record_by_part = {int(record["part_index"]): dict(record) for record in records}
    applied = []
    rejected = []
    valid_indices = set(range(1, len(components) + 1))

    for relation in relations:
        child_index = int(relation["child_index"])
        parent_index = int(relation["parent_index"])
        confidence = float(relation.get("confidence_score", 0.0))
        rejection = {
            "child_index": child_index,
            "parent_index": parent_index,
            "relation": relation.get("relation", ""),
            "confidence": relation.get("confidence", "UNKNOWN"),
            "confidence_score": confidence,
            "reason": relation.get("reason", ""),
        }
        if not relation.get("apply", True):
            rejection["reject_reason"] = "semantic_relation_disabled"
            rejected.append(rejection)
            continue
        if confidence < min_confidence:
            rejection["reject_reason"] = "semantic_confidence_below_threshold"
            rejected.append(rejection)
            continue
        if child_index not in valid_indices or parent_index not in valid_indices:
            rejection["reject_reason"] = "unknown_part_index"
            rejected.append(rejection)
            continue
        if child_index == parent_index:
            rejection["reject_reason"] = "self_parent"
            rejected.append(rejection)
            continue
        if refined_parents.get(child_index) is None:
            rejection["reject_reason"] = "body_cannot_be_reparented"
            rejected.append(rejection)
            continue
        if parent_chain_contains(refined_parents, parent_index, child_index):
            rejection["reject_reason"] = "would_create_parent_cycle"
            rejected.append(rejection)
            continue

        previous_parent_index = refined_parents.get(child_index)
        change_type = "override" if previous_parent_index != parent_index else "confirmed"
        edge_record = adjacency_record_between(adjacency, child_index, parent_index) or {}
        shared_edges = int(edge_record.get("shared_edges", 0))
        shared_vertices = int(edge_record.get("shared_vertex_count", 0))
        child = components[child_index - 1]
        parent = components[parent_index - 1]
        current_edge_record = adjacency_record_between(adjacency, child_index, previous_parent_index) or {}
        change = {
            "change_type": change_type,
            "child_index": child_index,
            "previous_parent_index": previous_parent_index,
            "new_parent_index": parent_index,
            "relation": relation.get("relation", ""),
            "confidence": relation.get("confidence", "UNKNOWN"),
            "confidence_score": confidence,
            "reason": relation.get("reason", ""),
            "topology_shared_edges_to_new_parent": shared_edges,
            "topology_shared_vertices_to_new_parent": shared_vertices,
            "topology_shared_edges_to_previous_parent": int(current_edge_record.get("shared_edges", 0)),
            "parent_larger_by_topology_rule": bool(component_is_larger(parent, child)),
            "child_area_mm2": float(child.area),
            "parent_area_mm2": float(parent.area),
            "child_faces": int(child.face_count),
            "parent_faces": int(parent.face_count),
            "source_views": relation.get("source_views", []),
        }

        refined_parents[child_index] = parent_index
        record = record_by_part.get(child_index, {"part_index": child_index})
        record["previous_parent_index"] = previous_parent_index
        record["parent_index"] = parent_index
        record["shared_edges_to_parent"] = shared_edges
        record["shared_vertices_to_parent"] = shared_vertices
        record["reason"] = "visual_semantic_parent_override" if change_type == "override" else "visual_semantic_parent_confirmed"
        if change_type == "override":
            record.pop("recursive_minimal_loop", None)
            record.pop("mixed_boundary_loop", None)
        record["visual_semantic_parent_relation"] = change
        record_by_part[child_index] = record
        applied.append(change)

    refined_children = rebuild_assembly_children(refined_parents)
    refined_records = [record_by_part[index] for index in sorted(record_by_part)]
    return refined_parents, refined_children, refined_records, {"applied": applied, "rejected": rejected}


def parent_chain_contains(parents: dict[int, int | None], start_index: int | None, target_index: int) -> bool:
    seen: set[int] = set()
    current = start_index
    while current is not None:
        current = int(current)
        if current == int(target_index):
            return True
        if current in seen:
            return True
        seen.add(current)
        current = parents.get(current)
    return False


def find_parent_cycles(parents: dict[int, int | None]) -> list[list[int]]:
    cycles: list[list[int]] = []
    seen_cycles: set[tuple[int, ...]] = set()
    for start_index in sorted(parents):
        path: list[int] = []
        positions: dict[int, int] = {}
        current = start_index
        while current is not None:
            current = int(current)
            if current in positions:
                cycle = path[positions[current] :]
                if cycle:
                    min_pos = min(range(len(cycle)), key=lambda pos: cycle[pos])
                    normalized = tuple(cycle[min_pos:] + cycle[:min_pos])
                    if normalized not in seen_cycles:
                        seen_cycles.add(normalized)
                        cycles.append(list(normalized))
                break
            if current not in parents:
                break
            positions[current] = len(path)
            path.append(current)
            current = parents.get(current)
    return cycles


def break_assembly_parent_cycles(
    components: list[Component],
    body_index: int | None,
    adjacency: dict[tuple[int, int], dict],
    parents: dict[int, int | None],
    records: list[dict],
    min_shared_edges: int,
) -> tuple[dict[int, int | None], dict[int, list[int]], list[dict], list[dict]]:
    refined_parents = dict(parents)
    record_by_part = {int(record["part_index"]): dict(record) for record in records}
    changes: list[dict] = []

    for _ in range(len(refined_parents) + 1):
        cycles = find_parent_cycles(refined_parents)
        if not cycles:
            break
        for cycle in cycles:
            cycle_set = set(cycle)
            replacement_options = []
            for child_index in cycle:
                component = components[child_index - 1]
                for (left, right), edge_record in adjacency.items():
                    if child_index not in (left, right):
                        continue
                    candidate_index = right if left == child_index else left
                    if candidate_index in cycle_set:
                        continue
                    shared_edges = int(edge_record.get("shared_edges", 0))
                    if shared_edges < min_shared_edges:
                        continue
                    if parent_chain_contains(refined_parents, candidate_index, child_index):
                        continue
                    candidate = components[candidate_index - 1]
                    candidate_is_body = candidate_index == body_index
                    candidate_is_larger = candidate.area > component.area or candidate.face_count > component.face_count
                    replacement_options.append(
                        (
                            int(candidate_is_body),
                            int(candidate_is_larger),
                            shared_edges,
                            float(candidate.area),
                            int(candidate.face_count),
                            child_index,
                            candidate_index,
                            edge_record,
                        )
                    )

            previous_parent_index: int | None
            if replacement_options:
                replacement_options.sort(reverse=True)
                *_score, child_index, new_parent_index, edge_record = replacement_options[0]
                previous_parent_index = refined_parents.get(child_index)
                refined_parents[child_index] = int(new_parent_index)
                shared_edges = int(edge_record.get("shared_edges", 0))
                shared_vertices = int(edge_record.get("shared_vertex_count", 0))
                reason = "cycle_break_to_external_parent"
            else:
                if body_index is None:
                    continue
                child_index = max(cycle, key=lambda index: (components[index - 1].face_count, components[index - 1].area))
                if child_index == body_index:
                    continue
                previous_parent_index = refined_parents.get(child_index)
                refined_parents[child_index] = int(body_index)
                shared_edges = 0
                shared_vertices = 0
                reason = "cycle_break_to_body_fallback"

            record = record_by_part.get(int(child_index), {"part_index": int(child_index)})
            record["previous_parent_index"] = previous_parent_index
            record["parent_index"] = refined_parents[child_index]
            record["shared_edges_to_parent"] = shared_edges
            record["shared_vertices_to_parent"] = shared_vertices
            record["reason"] = reason
            record["cycle_nodes"] = [int(index) for index in cycle]
            record_by_part[int(child_index)] = record
            changes.append(
                {
                    "part_index": int(child_index),
                    "previous_parent_index": previous_parent_index,
                    "new_parent_index": refined_parents[child_index],
                    "cycle_nodes": [int(index) for index in cycle],
                    "shared_edges_to_new_parent": shared_edges,
                    "reason": reason,
                }
            )

    refined_children = rebuild_assembly_children(refined_parents)
    refined_records = [record_by_part[index] for index in sorted(record_by_part)]
    return refined_parents, refined_children, refined_records, changes


def reparent_shared_parent_child_loops(
    parents: dict[int, int | None],
    records: list[dict],
    loop_records: list[dict],
    adjacency: dict[tuple[int, int], dict],
) -> tuple[dict[int, int | None], dict[int, list[int]], list[dict], list[dict]]:
    refined_parents = dict(parents)
    record_by_part = {int(record["part_index"]): dict(record) for record in records}
    loop_cache: dict[int, list[dict]] = collections.defaultdict(list)
    for record in loop_records:
        loop_cache[int(record["component_index"])].append(record)

    changes: list[dict] = []
    changed = True
    while changed:
        changed = False
        children = rebuild_assembly_children(refined_parents)
        for parent_index in sorted(children):
            grandparent_index = refined_parents.get(parent_index)
            if grandparent_index is None:
                continue
            for loop_record in loop_cache.get(parent_index, []):
                counts = loop_record["neighbor_counts"]
                grandparent_edges = int(counts.get(str(grandparent_index), 0))
                if grandparent_edges < 3:
                    continue
                for child_index in list(children.get(parent_index, [])):
                    child_edges = int(counts.get(str(child_index), 0))
                    if child_edges < 3:
                        continue
                    previous_parent_index = int(parent_index)
                    refined_parents[child_index] = int(grandparent_index)
                    edge_record = adjacency.get(tuple(sorted((int(child_index), int(grandparent_index)))))
                    shared_edges = int(edge_record.get("shared_edges", 0)) if edge_record else 0
                    shared_vertices = int(edge_record.get("shared_vertex_count", 0)) if edge_record else 0
                    record = record_by_part.get(int(child_index), {"part_index": int(child_index)})
                    record["previous_parent_index"] = previous_parent_index
                    record["parent_index"] = int(grandparent_index)
                    record["shared_edges_to_parent"] = shared_edges
                    record["shared_vertices_to_parent"] = shared_vertices
                    record["reason"] = "reparented_shared_parent_child_loop"
                    record["mixed_boundary_loop"] = {
                        "parent_loop_index": int(loop_record["loop_index"]),
                        "parent_loop_edges": int(loop_record["edge_count"]),
                        "child_edges_on_parent_loop": child_edges,
                        "grandparent_edges_on_parent_loop": grandparent_edges,
                    }
                    record_by_part[int(child_index)] = record
                    changes.append(
                        {
                            "part_index": int(child_index),
                            "previous_parent_index": previous_parent_index,
                            "new_parent_index": int(grandparent_index),
                            "parent_loop_index": int(loop_record["loop_index"]),
                            "parent_loop_edges": int(loop_record["edge_count"]),
                            "child_edges_on_parent_loop": child_edges,
                            "grandparent_edges_on_parent_loop": grandparent_edges,
                            "shared_edges_to_new_parent": shared_edges,
                            "reason": "reparented_shared_parent_child_loop",
                        }
                    )
                    changed = True
                if changed:
                    break
            if changed:
                break

    refined_children = rebuild_assembly_children(refined_parents)
    refined_records = [record_by_part[index] for index in sorted(record_by_part)]
    return refined_parents, refined_children, refined_records, changes


def refine_mixed_boundary_parents(
    vertices: np.ndarray,
    faces: np.ndarray,
    components: list[Component],
    boundary_neighbor_lookup: dict[tuple[int, int], set[int]],
    adjacency: dict[tuple[int, int], dict],
    parents: dict[int, int | None],
    records: list[dict],
    min_shared_edges: int,
) -> tuple[dict[int, int | None], dict[int, list[int]], list[dict], list[dict], list[dict]]:
    loop_records = []
    loop_cache: dict[int, list[dict]] = {}
    for index, component in enumerate(components, start=1):
        component_records = component_boundary_loop_neighbors(
            vertices,
            faces,
            component,
            index,
            boundary_neighbor_lookup,
        )
        loop_cache[index] = component_records
        loop_records.extend(component_records)

    refined_parents = dict(parents)
    record_by_part = {int(record["part_index"]): dict(record) for record in records}
    changes: list[dict] = []

    for child_index in sorted(parents):
        parent_index = refined_parents.get(child_index)
        if parent_index is None:
            continue
        grandparent_index = refined_parents.get(parent_index)
        if grandparent_index is None:
            continue
        child_grandparent_key = tuple(sorted((int(child_index), int(grandparent_index))))
        child_grandparent_adjacency = adjacency.get(child_grandparent_key)
        if child_grandparent_adjacency is None:
            continue
        if int(child_grandparent_adjacency.get("shared_edges", 0)) < min_shared_edges:
            continue

        conflict_loop = None
        for loop_record in loop_cache.get(parent_index, []):
            counts = loop_record["neighbor_counts"]
            child_edges = int(counts.get(str(child_index), 0))
            grandparent_edges = int(counts.get(str(grandparent_index), 0))
            if child_edges >= 3 and grandparent_edges >= 3:
                conflict_loop = {
                    "parent_loop_index": int(loop_record["loop_index"]),
                    "parent_loop_edges": int(loop_record["edge_count"]),
                    "child_edges_on_parent_loop": child_edges,
                    "grandparent_edges_on_parent_loop": grandparent_edges,
                }
                break
        if conflict_loop is None:
            continue

        previous_parent_index = int(parent_index)
        refined_parents[child_index] = int(grandparent_index)
        record = record_by_part[child_index]
        record["previous_parent_index"] = previous_parent_index
        record["parent_index"] = int(grandparent_index)
        record["shared_edges_to_parent"] = int(child_grandparent_adjacency.get("shared_edges", 0))
        record["shared_vertices_to_parent"] = int(child_grandparent_adjacency.get("shared_vertex_count", 0))
        record["reason"] = "reparented_mixed_parent_child_boundary_loop"
        record["mixed_boundary_loop"] = conflict_loop
        record_by_part[child_index] = record
        changes.append(
            {
                "part_index": int(child_index),
                "previous_parent_index": previous_parent_index,
                "new_parent_index": int(grandparent_index),
                **conflict_loop,
            }
        )

    refined_children = rebuild_assembly_children(refined_parents)
    refined_records = [record_by_part[index] for index in sorted(record_by_part)]
    return refined_parents, refined_children, refined_records, changes, loop_records


def group_mixed_parent_loop_children(
    components: list[Component],
    adjacency: dict[tuple[int, int], dict],
    parents: dict[int, int | None],
    records: list[dict],
    loop_records: list[dict],
    min_shared_edges: int = 3,
) -> tuple[dict[int, int | None], dict[int, list[int]], list[dict], list[dict]]:
    """Group sibling color patches whose union owns one parent boundary loop.

    At a three-color junction a closed loop on the parent can be composed of
    arcs belonging to two or more direct children.  Treating every child as an
    independent parent socket pairs different closed source loops (for
    example, parent A+B versus child A+C).  Build one recursive subassembly for
    the sibling group so the parent cut follows the group's exact outer loop;
    the internal color seams are split at the next recursive layer.
    """
    refined_parents = dict(parents)
    record_by_part = {int(record["part_index"]): dict(record) for record in records}
    changes: list[dict] = []

    for loop_record in sorted(
        loop_records,
        key=lambda record: (
            int(record["component_index"]),
            int(record["loop_index"]),
        ),
    ):
        parent_index = int(loop_record["component_index"])
        def direct_branch_root(component_index: int) -> int | None:
            current = int(component_index)
            seen: set[int] = set()
            while current not in seen:
                seen.add(current)
                owner = refined_parents.get(current)
                if owner == parent_index:
                    return current
                if owner is None:
                    return None
                current = int(owner)
            return None

        loop_neighbors_by_branch: dict[int, set[int]] = collections.defaultdict(set)
        for neighbor_index, edge_count in loop_record.get("neighbor_counts", {}).items():
            if int(edge_count) < int(min_shared_edges):
                continue
            neighbor_index = int(neighbor_index)
            branch_root = direct_branch_root(neighbor_index)
            if branch_root is not None:
                loop_neighbors_by_branch[int(branch_root)].add(neighbor_index)
        direct_branches = sorted(loop_neighbors_by_branch)
        if len(direct_branches) < 2:
            continue

        branch_members: dict[int, set[int]] = collections.defaultdict(set)
        for component_index in range(1, len(components) + 1):
            branch_root = direct_branch_root(component_index)
            if branch_root in loop_neighbors_by_branch:
                branch_members[int(branch_root)].add(int(component_index))

        # Form connected sibling groups using their source shared-boundary
        # graph.  Components merely appearing on the same parent loop but not
        # touching each other must remain independent.
        remaining = set(direct_branches)
        sibling_groups: list[set[int]] = []
        while remaining:
            seed = min(remaining)
            group = {seed}
            frontier = [seed]
            remaining.remove(seed)
            while frontier:
                left = frontier.pop()
                for right in sorted(list(remaining)):
                    connected = any(
                        int(
                            (
                                adjacency_record_between(adjacency, left_member, right_member)
                                or {}
                            ).get("shared_edges", 0)
                        )
                        >= int(min_shared_edges)
                        for left_member in branch_members[int(left)]
                        for right_member in branch_members[int(right)]
                    )
                    if not connected:
                        continue
                    remaining.remove(right)
                    group.add(right)
                    frontier.append(right)
            sibling_groups.append(group)

        for group in sibling_groups:
            if len(group) < 2:
                continue
            anchor = max(
                group,
                key=lambda index: (
                    float(components[index - 1].area),
                    int(components[index - 1].face_count),
                    -int(index),
                ),
            )
            attached = {int(anchor)}
            pending = set(int(index) for index in group if int(index) != int(anchor))
            while pending:
                candidates = []
                for child_branch_root in sorted(pending):
                    # Reparent the root of the pending branch to the exact
                    # component it touches inside an already attached branch.
                    # This preserves the whole pending subtree and records a
                    # real source interface (P02->P14 in the lulumo case).
                    for attached_branch_root in sorted(attached):
                        for candidate_parent in sorted(
                            branch_members[int(attached_branch_root)]
                        ):
                            edge_record = adjacency_record_between(
                                adjacency,
                                child_branch_root,
                                candidate_parent,
                            )
                            shared_edges = int(edge_record.get("shared_edges", 0)) if edge_record else 0
                            if shared_edges < int(min_shared_edges):
                                continue
                            candidates.append(
                                (
                                    shared_edges,
                                    float(components[candidate_parent - 1].area),
                                    int(components[candidate_parent - 1].face_count),
                                    -int(child_branch_root),
                                    -int(candidate_parent),
                                    int(child_branch_root),
                                    int(candidate_parent),
                                    int(attached_branch_root),
                                    edge_record,
                                )
                            )
                if not candidates:
                    break
                (
                    *_rank,
                    child_index,
                    new_parent_index,
                    attached_branch_root,
                    edge_record,
                ) = max(candidates)
                previous_parent_index = refined_parents.get(child_index)
                refined_parents[child_index] = int(new_parent_index)
                shared_edges = int(edge_record.get("shared_edges", 0))
                shared_vertices = int(edge_record.get("shared_vertex_count", 0))
                evidence = {
                    "parent_loop_owner_index": int(parent_index),
                    "parent_loop_index": int(loop_record["loop_index"]),
                    "parent_loop_edges": int(loop_record["edge_count"]),
                    "sibling_group_indices": sorted(int(index) for index in group),
                    "subassembly_anchor_index": int(anchor),
                    "attached_branch_root_index": int(attached_branch_root),
                }
                record = record_by_part.get(child_index, {"part_index": child_index})
                record.update(
                    {
                        "previous_parent_index": previous_parent_index,
                        "parent_index": int(new_parent_index),
                        "shared_edges_to_parent": shared_edges,
                        "shared_vertices_to_parent": shared_vertices,
                        "reason": "grouped_mixed_parent_boundary_subassembly",
                        "mixed_parent_loop_group": evidence,
                    }
                )
                record_by_part[child_index] = record
                changes.append(
                    {
                        "part_index": int(child_index),
                        "previous_parent_index": previous_parent_index,
                        "new_parent_index": int(new_parent_index),
                        "shared_edges_to_new_parent": shared_edges,
                        "reason": "grouped_mixed_parent_boundary_subassembly",
                        **evidence,
                    }
                )
                pending.remove(child_index)
                attached.add(child_index)

    refined_children = rebuild_assembly_children(refined_parents)
    refined_records = [record_by_part[index] for index in sorted(record_by_part)]
    return refined_parents, refined_children, refined_records, changes


def build_strongest_path_assembly_tree(
    components: list[Component],
    body_component: Component | None,
    adjacency: dict[tuple[int, int], dict],
    min_shared_edges: int,
) -> tuple[dict[int, int | None], dict[int, list[int]], list[dict]]:
    body_index = component_identity_index(components, body_component)
    neighbors: dict[int, list[tuple[int, dict]]] = collections.defaultdict(list)
    for (left, right), record in adjacency.items():
        if int(record["shared_edges"]) < min_shared_edges:
            continue
        neighbors[left].append((right, record))
        neighbors[right].append((left, record))

    parents: dict[int, int | None] = {}
    children: dict[int, list[int]] = collections.defaultdict(list)
    record_by_part: dict[int, dict] = {}

    if body_index is not None:
        parents[body_index] = None
        record_by_part[body_index] = {
            "part_index": body_index,
            "parent_index": None,
            "shared_edges_to_parent": 0,
            "shared_vertices_to_parent": 0,
            "reason": "body_root",
        }

        visited = {body_index}
        while True:
            candidates = []
            for parent_index in sorted(visited):
                for child_index, edge_record in neighbors.get(parent_index, []):
                    if child_index in visited:
                        continue
                    child = components[child_index - 1]
                    candidates.append(
                        (
                            int(edge_record["shared_edges"]),
                            int(edge_record.get("shared_vertex_count", 0)),
                            float(child.area),
                            int(child.face_count),
                            -child_index,
                            parent_index,
                            child_index,
                            edge_record,
                        )
                    )
            if not candidates:
                break

            (
                shared_edges,
                shared_vertices,
                _area,
                _faces,
                _tie_break_child,
                parent_index,
                child_index,
                edge_record,
            ) = max(candidates)
            visited.add(child_index)
            parents[child_index] = parent_index
            children[parent_index].append(child_index)
            record_by_part[child_index] = {
                "part_index": child_index,
                "parent_index": parent_index,
                "shared_edges_to_parent": shared_edges,
                "shared_vertices_to_parent": shared_vertices,
                "reason": "body_rooted_strongest_path",
            }

    for child_index, component in enumerate(components, start=1):
        if child_index in parents:
            continue
        if child_index == body_index:
            parents[child_index] = None
            record_by_part[child_index] = {
                "part_index": child_index,
                "parent_index": None,
                "shared_edges_to_parent": 0,
                "shared_vertices_to_parent": 0,
                "reason": "body_root",
            }
            continue

        candidates = []
        fallback_candidates = []
        for neighbor_index, edge_record in neighbors.get(child_index, []):
            neighbor = components[neighbor_index - 1]
            neighbor_is_body = neighbor_index == body_index
            neighbor_is_larger = neighbor.area > component.area or neighbor.face_count > component.face_count
            fallback_candidates.append(
                (
                    int(edge_record["shared_edges"]),
                    float(neighbor.area),
                    int(neighbor.face_count),
                    neighbor_index,
                    edge_record,
                )
            )
            if not (neighbor_is_body or neighbor_is_larger):
                continue
            candidates.append(
                (
                    int(edge_record["shared_edges"]),
                    float(neighbor.area),
                    int(neighbor.face_count),
                    neighbor_index,
                    edge_record,
                )
            )

        if candidates:
            candidates.sort(reverse=True)
            shared_edges, _area, _faces, parent_index, edge_record = candidates[0]
            parents[child_index] = parent_index
            children[parent_index].append(child_index)
            record_by_part[child_index] = {
                "part_index": child_index,
                "parent_index": parent_index,
                "shared_edges_to_parent": shared_edges,
                "shared_vertices_to_parent": int(edge_record.get("shared_vertex_count", 0)),
                "reason": "largest_shared_boundary_to_larger_or_body",
            }
            continue

        if fallback_candidates:
            fallback_candidates.sort(reverse=True)
            shared_edges, _area, _faces, parent_index, edge_record = fallback_candidates[0]
            parents[child_index] = parent_index
            children[parent_index].append(child_index)
            record_by_part[child_index] = {
                "part_index": child_index,
                "parent_index": parent_index,
                "shared_edges_to_parent": shared_edges,
                "shared_vertices_to_parent": int(edge_record.get("shared_vertex_count", 0)),
                "reason": "strongest_shared_boundary_fallback",
            }
            continue

        parents[child_index] = body_index
        if body_index is not None:
            children[body_index].append(child_index)
        record_by_part[child_index] = {
            "part_index": child_index,
            "parent_index": body_index,
            "shared_edges_to_parent": 0,
            "shared_vertices_to_parent": 0,
            "reason": "fallback_to_body",
        }

    for value in children.values():
        value.sort()
    records = [record_by_part[index] for index in sorted(record_by_part)]
    return parents, dict(children), records


def build_recursive_minimal_assembly_tree(
    components: list[Component],
    body_component: Component | None,
    adjacency: dict[tuple[int, int], dict],
    min_shared_edges: int,
    loop_records: list[dict],
) -> tuple[dict[int, int | None], dict[int, list[int]], list[dict]]:
    body_index = component_identity_index(components, body_component)
    neighbors: dict[int, list[tuple[int, dict]]] = collections.defaultdict(list)
    for (left, right), record in adjacency.items():
        if int(record["shared_edges"]) < min_shared_edges:
            continue
        neighbors[left].append((right, record))
        neighbors[right].append((left, record))

    loop_cache = build_loop_cache(loop_records)
    parents: dict[int, int | None] = {}
    children: dict[int, list[int]] = collections.defaultdict(list)
    record_by_part: dict[int, dict] = {}

    if body_index is not None:
        parents[body_index] = None
        record_by_part[body_index] = {
            "part_index": body_index,
            "parent_index": None,
            "shared_edges_to_parent": 0,
            "shared_vertices_to_parent": 0,
            "reason": "body_root",
        }

    for child_index, component in enumerate(components, start=1):
        if child_index == body_index:
            continue

        direct_body_option: dict | None = None
        larger_options: list[dict] = []
        fallback_options: list[dict] = []
        for neighbor_index, edge_record in neighbors.get(child_index, []):
            neighbor = components[neighbor_index - 1]
            option = {
                "parent_index": int(neighbor_index),
                "shared_edges": int(edge_record["shared_edges"]),
                "shared_vertices": int(edge_record.get("shared_vertex_count", 0)),
                "edge_record": edge_record,
            }
            fallback_options.append(option)
            if neighbor_index == body_index:
                direct_body_option = option
                continue
            if not component_is_larger(neighbor, component):
                continue
            separate_from_body = best_loop_contact(
                loop_cache,
                neighbor_index,
                child_index,
                body_index,
                require_separate=body_index is not None,
                min_edges=3,
            )
            loop_contact = separate_from_body or best_loop_contact(
                loop_cache,
                neighbor_index,
                child_index,
                None,
                require_separate=False,
                min_edges=3,
            )
            option["loop_contact"] = loop_contact
            option["separate_from_body_loop"] = separate_from_body is not None
            option["loop_dominance"] = float(loop_contact.get("child_loop_dominance", 0.0)) if loop_contact else 0.0
            larger_options.append(option)

        selected: dict | None = None
        reason = ""
        if direct_body_option is not None:
            body_shared_edges = int(direct_body_option["shared_edges"])
            nested_options = [
                option
                for option in larger_options
                if option.get("separate_from_body_loop")
                and (
                    float(option.get("loop_dominance", 0.0)) >= 0.50
                    or int(option["shared_edges"]) >= body_shared_edges
                )
            ]
            if nested_options:
                selected = min(
                    nested_options,
                    key=lambda option: (
                        *component_size_sort_key(components[int(option["parent_index"]) - 1]),
                        -int(option["shared_edges"]),
                        int(option["parent_index"]),
                    ),
                )
                reason = "recursive_minimal_nearest_larger_separate_body_loop"
            else:
                selected = direct_body_option
                reason = "recursive_minimal_direct_body_child"
        elif larger_options:
            selected = min(
                larger_options,
                key=lambda option: (
                    *component_size_sort_key(components[int(option["parent_index"]) - 1]),
                    -int(option["shared_edges"]),
                    int(option["parent_index"]),
                ),
            )
            reason = "recursive_minimal_nearest_larger_neighbor"
        elif fallback_options:
            selected = max(
                fallback_options,
                key=lambda option: (
                    int(option["shared_edges"]),
                    int(option["shared_vertices"]),
                    float(components[int(option["parent_index"]) - 1].area),
                    int(components[int(option["parent_index"]) - 1].face_count),
                    -int(option["parent_index"]),
                ),
            )
            reason = "strongest_shared_boundary_fallback"

        if selected is None:
            parent_index = body_index
            shared_edges = 0
            shared_vertices = 0
            record = {
                "part_index": child_index,
                "parent_index": parent_index,
                "shared_edges_to_parent": shared_edges,
                "shared_vertices_to_parent": shared_vertices,
                "reason": "fallback_to_body" if body_index is not None else "no_parent_available",
            }
        else:
            parent_index = int(selected["parent_index"])
            shared_edges = int(selected["shared_edges"])
            shared_vertices = int(selected["shared_vertices"])
            record = {
                "part_index": child_index,
                "parent_index": parent_index,
                "shared_edges_to_parent": shared_edges,
                "shared_vertices_to_parent": shared_vertices,
                "reason": reason,
            }
            if selected.get("loop_contact"):
                record["recursive_minimal_loop"] = selected["loop_contact"]

        parents[child_index] = parent_index
        if parent_index is not None:
            children[int(parent_index)].append(child_index)
        record_by_part[child_index] = record

    for value in children.values():
        value.sort()
    records = [record_by_part[index] for index in sorted(record_by_part)]
    return parents, dict(children), records


def refine_recursive_minimal_parents(
    components: list[Component],
    adjacency: dict[tuple[int, int], dict],
    parents: dict[int, int | None],
    records: list[dict],
    loop_records: list[dict],
    min_shared_edges: int,
) -> tuple[dict[int, int | None], dict[int, list[int]], list[dict], list[dict]]:
    refined_parents = dict(parents)
    record_by_part = {int(record["part_index"]): dict(record) for record in records}
    loop_cache = build_loop_cache(loop_records)
    changes: list[dict] = []

    for _ in range(len(refined_parents) + 1):
        changed = False
        for child_index in sorted(refined_parents):
            current_parent_index = refined_parents.get(child_index)
            if current_parent_index is None:
                continue
            child = components[child_index - 1]
            current_edge_record = adjacency_record_between(adjacency, child_index, current_parent_index)
            current_shared_edges = int(current_edge_record.get("shared_edges", 0)) if current_edge_record else 0

            options = []
            for (left, right), edge_record in adjacency.items():
                if child_index not in (left, right):
                    continue
                candidate_parent_index = right if left == child_index else left
                if candidate_parent_index == current_parent_index:
                    continue
                candidate_parent_parent = refined_parents.get(candidate_parent_index)
                if candidate_parent_parent is None:
                    continue
                if parent_chain_contains(refined_parents, candidate_parent_index, child_index):
                    continue
                if not parent_chain_contains(refined_parents, candidate_parent_index, current_parent_index):
                    continue
                candidate_parent = components[candidate_parent_index - 1]
                if not component_is_larger(candidate_parent, child):
                    continue
                shared_edges = int(edge_record.get("shared_edges", 0))
                if shared_edges < min_shared_edges:
                    continue
                loop_contact = best_loop_contact(
                    loop_cache,
                    candidate_parent_index,
                    child_index,
                    candidate_parent_parent,
                    require_separate=True,
                    min_edges=3,
                )
                if loop_contact is None:
                    continue
                loop_dominance = float(loop_contact.get("child_loop_dominance", 0.0))
                if loop_dominance < 0.50 and shared_edges < current_shared_edges:
                    continue
                options.append(
                    (
                        *component_size_sort_key(candidate_parent),
                        -shared_edges,
                        int(candidate_parent_index),
                        edge_record,
                        loop_contact,
                    )
                )

            if not options:
                continue
            options.sort()
            _area, _faces, _neg_shared_edges, new_parent_index, edge_record, loop_contact = options[0]
            previous_parent_index = refined_parents.get(child_index)
            refined_parents[child_index] = int(new_parent_index)
            shared_edges = int(edge_record.get("shared_edges", 0))
            shared_vertices = int(edge_record.get("shared_vertex_count", 0))
            record = record_by_part.get(int(child_index), {"part_index": int(child_index)})
            record["previous_parent_index"] = previous_parent_index
            record["parent_index"] = int(new_parent_index)
            record["shared_edges_to_parent"] = shared_edges
            record["shared_vertices_to_parent"] = shared_vertices
            record["reason"] = "recursive_minimal_reparent_to_nearest_subassembly_body"
            record["recursive_minimal_loop"] = loop_contact
            record_by_part[int(child_index)] = record
            changes.append(
                {
                    "part_index": int(child_index),
                    "previous_parent_index": previous_parent_index,
                    "new_parent_index": int(new_parent_index),
                    "shared_edges_to_new_parent": shared_edges,
                    "reason": "recursive_minimal_reparent_to_nearest_subassembly_body",
                    **loop_contact,
                }
            )
            changed = True
        if not changed:
            break

    refined_children = rebuild_assembly_children(refined_parents)
    refined_records = [record_by_part[index] for index in sorted(record_by_part)]
    return refined_parents, refined_children, refined_records, changes


def assembly_depths(parents: dict[int, int | None]) -> dict[int, int]:
    depths: dict[int, int] = {}
    for index in sorted(parents):
        depth = 0
        seen: set[int] = set()
        current = parents.get(index)
        while current is not None and current not in seen:
            seen.add(int(current))
            depth += 1
            current = parents.get(int(current))
        depths[int(index)] = depth
    return depths


def build_recursive_minimal_layers(parents: dict[int, int | None], children: dict[int, list[int]]) -> list[dict]:
    depths = assembly_depths(parents)
    layers: list[dict] = []
    visited: set[int] = set()

    def visit(local_body_index: int, path: tuple[int, ...]) -> None:
        local_body_index = int(local_body_index)
        if local_body_index in visited or not children.get(local_body_index):
            return
        visited.add(local_body_index)
        direct_children = sorted(int(child) for child in children.get(local_body_index, []))
        step_order = len(layers)
        layers.append(
            {
                "step_order": int(step_order),
                "depth": int(depths.get(local_body_index, 0)),
                "local_body_index": local_body_index,
                "direct_child_indices": direct_children,
                "leaf_child_indices": [child for child in direct_children if not children.get(child)],
                "nested_child_indices": [child for child in direct_children if children.get(child)],
                "recursion_path": [int(index) for index in (*path, local_body_index)],
                "execution_order": "strict_depth_first_preorder",
            }
        )
        for child_index in direct_children:
            if children.get(child_index):
                visit(child_index, (*path, local_body_index))

    root_indices = sorted(
        int(index)
        for index, parent_index in parents.items()
        if parent_index is None and children.get(int(index))
    )
    for root_index in root_indices:
        visit(root_index, ())

    for remaining_index in sorted(int(index) for index in children if children.get(int(index))):
        visit(remaining_index, ())
    return layers


def advance_strict_recursive_state(
    active_parts: dict[int, dict],
    local_body_index: int,
    local_body_part: dict,
    direct_child_parts: dict[int, dict],
) -> tuple[dict[int, dict], dict]:
    local_body_index = int(local_body_index)
    before_indices = sorted(int(index) for index in active_parts)
    if before_indices and local_body_index not in active_parts:
        raise ValueError(
            f"strict recursive step cannot expand P{local_body_index:02d}; "
            f"active parts are {before_indices}"
        )
    if not before_indices and local_body_part.get("state_role") != "root_body":
        raise ValueError("the first strict recursive step must expand the root body")

    next_parts = dict(active_parts)
    replaced_part = next_parts.pop(local_body_index, None)
    if local_body_index in direct_child_parts:
        raise ValueError(
            f"strict recursive step for P{local_body_index:02d} cannot add itself as a child"
        )
    next_parts[local_body_index] = dict(local_body_part)
    for child_index, child_part in sorted(direct_child_parts.items()):
        child_index = int(child_index)
        if child_index in next_parts:
            raise ValueError(
                f"strict recursive step for P{local_body_index:02d} would overwrite active "
                f"P{child_index:02d}"
            )
        next_parts[child_index] = dict(child_part)
    next_parts = {index: next_parts[index] for index in sorted(next_parts)}

    after_indices = sorted(next_parts)
    added_indices = sorted(
        index for index in after_indices if index not in before_indices or index == local_body_index
    )
    retained_indices = sorted(
        index for index in before_indices if index != local_body_index and index in next_parts
    )
    transition = {
        "local_body_index": local_body_index,
        "before_active_indices": before_indices,
        "replaced_index": local_body_index if replaced_part is not None else None,
        "added_indices": added_indices,
        "retained_indices": retained_indices,
        "after_active_indices": after_indices,
    }
    return next_parts, transition




def build_assembly_tree(
    components: list[Component],
    body_component: Component | None,
    adjacency: dict[tuple[int, int], dict],
    min_shared_edges: int,
    tree_strategy: str,
    loop_records: list[dict],
) -> tuple[dict[int, int | None], dict[int, list[int]], list[dict]]:
    if tree_strategy == "strongest-path":
        return build_strongest_path_assembly_tree(components, body_component, adjacency, min_shared_edges)
    if tree_strategy == "recursive-minimal":
        return build_recursive_minimal_assembly_tree(components, body_component, adjacency, min_shared_edges, loop_records)
    raise ValueError(f"Unknown assembly tree strategy: {tree_strategy}")


def print_recognition(records: list[dict]) -> None:
    print("recognized_parts:", flush=True)
    for record in records:
        sx, sy, sz = record["bbox_size_mm"]
        cx, cy, cz = record["center"]
        slot_index = record.get("filament_slot_index")
        slot_text = "unmapped" if slot_index is None else f"{int(slot_index) + 1}(index={int(slot_index)})"
        line = (
            "  P{part_index:02d} role={role} color_name={color_name} color_hex={color_hex} "
            "raw_token={raw_color_token} filament_slot={slot_text} "
            "mapping_source={color_mapping_source} resolution={color_resolution_status} "
            "faces={faces} size_mm=({sx:.2f},{sy:.2f},{sz:.2f}) center=({cx:.2f},{cy:.2f},{cz:.2f}) "
            "loops={boundary_loops} processing={processing}".format(
                sx=sx,
                sy=sy,
                sz=sz,
                cx=cx,
                cy=cy,
                cz=cz,
                slot_text=slot_text,
                processing=record.get("selected_processing_mode", "unclassified"),
                **record,
            )
        )
        semantic_label = record.get("visual_semantic_label")
        if semantic_label:
            line += f" semantic={semantic_label}({record.get('visual_semantic_confidence', 'UNKNOWN')})"
        print(line, flush=True)
        print("recognized_part_json=" + json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)




class AssemblyPlanner:
    """Build and refine a recursive-minimal parent tree."""

    build_tree = staticmethod(build_assembly_tree)
    depths = staticmethod(assembly_depths)
    layers = staticmethod(build_recursive_minimal_layers)
