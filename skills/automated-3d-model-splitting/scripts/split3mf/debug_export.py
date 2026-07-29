from __future__ import annotations

import hashlib
import tempfile

from .common import *
from .project import *
from .recognition import *
from .mesh import *
from .package_io import *
from .selection import *
from .assembly import *
from .validation import *
from .inward import *
from .domain import BoundaryFairingContext, CapDecision
from .reporting import runtime_log


def shared_child_cap_decisions(
    child_refs: list[dict],
    child_index: int,
) -> dict[int, CapDecision]:
    """Select the prevalidated cap decisions owned by one direct child."""
    decisions: dict[int, CapDecision] = {}
    for ref in child_refs:
        if int(ref.get("component_index", -1)) != int(child_index):
            continue
        decision = ref.get("cap_decision")
        if decision is None:
            continue
        loop_index = int(ref["loop_index"])
        if loop_index in decisions:
            raise ValueError(
                f"duplicate shared cap decision for P{int(child_index):02d} "
                f"loop {loop_index}"
            )
        decisions[loop_index] = decision
    return decisions


def validate_shared_child_cap_decisions(
    child_index: int,
    decisions: dict[int, CapDecision],
    child_stats: dict,
) -> list[dict]:
    """Block publication when a child did not consume its planned cap field."""
    if not decisions:
        return []
    extensions = {
        int(record["loop_index"]): record
        for record in child_stats.get("loop_extensions", [])
    }
    validations: list[dict] = []
    for loop_index, decision in sorted(decisions.items()):
        extension = extensions.get(int(loop_index))
        if extension is None:
            raise ValueError(
                f"P{int(child_index):02d} did not build planned cap loop "
                f"{int(loop_index)}"
            )
        actual_mode = str(extension.get("cap_mode"))
        if actual_mode != str(decision.mode):
            raise ValueError(
                f"P{int(child_index):02d} cap mode changed after planning on "
                f"loop {int(loop_index)}: planned={decision.mode}, "
                f"actual={actual_mode}"
            )
        planned_distances = np.asarray(decision.distances, dtype=np.float64)
        planned_minimum = float(planned_distances.min())
        planned_maximum = float(planned_distances.max())
        actual_minimum = float(extension["extension_min_mm"])
        actual_maximum = float(extension["extension_max_mm"])
        if not (
            np.isclose(actual_minimum, planned_minimum, atol=1e-9, rtol=0.0)
            and np.isclose(actual_maximum, planned_maximum, atol=1e-9, rtol=0.0)
        ):
            raise ValueError(
                f"P{int(child_index):02d} cap distance field changed after "
                f"planning on loop {int(loop_index)}"
            )
        validations.append(
            {
                "loop_index": int(loop_index),
                "cap_mode": actual_mode,
                "minimum_distance_mm": actual_minimum,
                "maximum_distance_mm": actual_maximum,
                "status": "shared_cap_decision_reused",
            }
        )
    return validations


def export_recursive_layer_stage_outputs(
    output_dir: Path,
    report_path_mode: str,
    input_stem: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: list[str],
    components: list[Component],
    assembly_parents: dict[int, int | None],
    assembly_children: dict[int, list[int]],
    recursive_minimal_layers: list[dict],
    boundary_neighbor_lookup: dict[tuple[int, int], set[int]],
    component_centers: dict[int, np.ndarray],
    inward_overrides: dict[int, np.ndarray],
    model_center: np.ndarray,
    max_extension_mm: float,
    boundary_fairing: BoundaryFairingContext,
    flat_clearance_mm: float,
    fit_clearance_mm: float,
    insert_shrink_mm: float,
    lead_in_mm: float,
    top_edge_clearance_mm: float,
    clearance_mode: str,
    sibling_clearance_mm: float,
    socket_overcut_mm: float,
    bottom_clearance_mm: float,
    planar_extra_limit_mm: float,
    effective_cap_mode,
    layer_child_context,
    source_application: str | None = None,
    layer_order_offset: int = 0,
) -> tuple[Path, list[dict]]:
    layers_dir = output_dir / "recursive_layers"
    layers_dir.mkdir(parents=True, exist_ok=True)
    stage_records: list[dict] = []

    for inward_layer_order, layer in enumerate(recursive_minimal_layers):
        layer_order = int(layer_order_offset + inward_layer_order)
        local_body_index = int(layer["local_body_index"])
        direct_children = [int(child) for child in layer.get("direct_child_indices", [])]
        layer_dir = layers_dir / f"layer_{layer_order:02d}_INWARD_P{local_body_index:02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        layer_3mf_parts: list[dict] = []
        child_refs, union_by_child, subtree_by_child = layer_child_context(local_body_index)
        local_component = components[local_body_index - 1]
        color_info = COLOR_INFO.get(local_component.color_code, {"name": local_component.color_code, "hex": "", "rgba": [200, 200, 200, 255]})
        local_role = "root_body" if assembly_parents.get(local_body_index) is None else "local_body"
        local_part_id = f"L{layer_order:02d}_P{local_body_index:02d}_{sanitize_name(color_info['name'])}_{local_role.upper()}_CUT"
        part_cap_mode = effective_cap_mode(local_body_index)
        if assembly_parents.get(local_body_index) is None:
            mesh, stats = make_body_cut_mesh(
                vertices=vertices,
                faces=faces,
                body_component=local_component,
                part_id=local_part_id,
                cut_refs=child_refs,
                max_extension_mm=max_extension_mm,
                boundary_fairing=boundary_fairing,
                flat_clearance_mm=flat_clearance_mm,
                fit_clearance_mm=fit_clearance_mm,
                socket_overcut_mm=socket_overcut_mm,
                bottom_clearance_mm=bottom_clearance_mm,
                clearance_mode=clearance_mode,
                model_center=model_center,
                cap_mode=part_cap_mode,
                planar_extra_limit_mm=planar_extra_limit_mm,
            )
        else:
            mesh, stats = make_part_mesh(
                vertices=vertices,
                faces=faces,
                component=local_component,
                component_index=local_body_index,
                assembly_parent_index=assembly_parents.get(local_body_index),
                part_id=local_part_id,
                max_extension_mm=max_extension_mm,
                boundary_fairing=boundary_fairing,
                flat_clearance_mm=flat_clearance_mm,
                fit_clearance_mm=fit_clearance_mm,
                insert_shrink_mm=insert_shrink_mm,
                lead_in_mm=lead_in_mm,
                top_edge_clearance_mm=top_edge_clearance_mm,
                clearance_mode=clearance_mode,
                child_cut_refs=child_refs,
                boundary_neighbor_lookup=boundary_neighbor_lookup,
                component_centers=component_centers,
                sibling_clearance_mm=sibling_clearance_mm,
                socket_overcut_mm=socket_overcut_mm,
                bottom_clearance_mm=bottom_clearance_mm,
                inward_override=inward_overrides.get(local_body_index),
                model_center=model_center,
                cap_mode=part_cap_mode,
                planar_extra_limit_mm=planar_extra_limit_mm,
                parent_contact_only=True,
            )
        local_path = layer_dir / f"{local_part_id}.stl"
        mesh.export(local_path)
        stats["file"] = serialize_report_path(local_path, output_dir, report_path_mode)
        stats.update(validate_exported_mesh(local_path))
        stats["stage_layer_order"] = layer_order
        stats["stage_depth"] = int(layer.get("depth", 0))
        stats["stage_local_body_index"] = local_body_index
        stats["stage_role"] = local_role
        stats["stage_direct_child_indices"] = direct_children
        stats["stage_contains"] = [local_body_index]
        stage_records.append(stats)
        layer_3mf_parts.append(
            {
                "part_id": local_part_id,
                "mesh": mesh,
                "color_code": local_component.color_code,
                "color_name": color_info.get("name", local_component.color_code),
                "color_hex": color_info.get("hex", "#C8C8C8"),
                "annotation": {
                    "debug_recursive_layer": layer_order,
                    "stage_role": local_role,
                    "source_part_index": local_body_index,
                    "unit": "millimeter",
                },
            }
        )

        for child_index in direct_children:
            subtree = subtree_by_child.get(child_index) or subtree_component_indices(child_index, assembly_children)
            union_component = union_by_child.get(child_index) or build_subassembly_component(
                vertices,
                faces,
                components,
                subtree,
                components[child_index - 1].color_code,
            )
            child_color = COLOR_INFO.get(union_component.color_code, {"name": union_component.color_code})
            child_part_id = f"L{layer_order:02d}_P{child_index:02d}_{sanitize_name(child_color['name'])}_SUBASSEMBLY"
            child_cap_mode = effective_cap_mode(child_index)
            child_cap_decisions = shared_child_cap_decisions(
                child_refs,
                child_index,
            )
            child_mesh, child_stats = make_layer_child_subassembly_mesh(
                vertices=vertices,
                faces=faces,
                source_colors=colors,
                components=components,
                component=union_component,
                root_child_index=child_index,
                subtree_indices=subtree,
                parent_index=local_body_index,
                part_id=child_part_id,
                max_extension_mm=max_extension_mm,
                boundary_fairing=boundary_fairing,
                flat_clearance_mm=flat_clearance_mm,
                fit_clearance_mm=fit_clearance_mm,
                insert_shrink_mm=insert_shrink_mm,
                lead_in_mm=lead_in_mm,
                top_edge_clearance_mm=top_edge_clearance_mm,
                boundary_neighbor_lookup=boundary_neighbor_lookup,
                inward_override=inward_overrides.get(child_index),
                model_center=model_center,
                cap_mode=child_cap_mode,
                planar_extra_limit_mm=planar_extra_limit_mm,
                cap_decisions_by_loop=child_cap_decisions,
            )
            child_stats["shared_cap_decision_validation"] = (
                validate_shared_child_cap_decisions(
                    child_index,
                    child_cap_decisions,
                    child_stats,
                )
            )
            child_path = layer_dir / f"{child_part_id}.stl"
            child_mesh.export(child_path)
            child_stats["file"] = serialize_report_path(child_path, output_dir, report_path_mode)
            child_stats.update(validate_exported_mesh(child_path))
            child_stats["stage_layer_order"] = layer_order
            child_stats["stage_depth"] = int(layer.get("depth", 0))
            child_stats["stage_local_body_index"] = local_body_index
            child_stats["stage_role"] = "direct_child_leaf" if not assembly_children.get(child_index) else "direct_child_subassembly"
            child_stats["stage_direct_child_indices"] = []
            child_stats["stage_contains"] = [int(index) for index in subtree]
            stage_records.append(child_stats)
            layer_3mf_parts.append(
                {
                    "part_id": child_part_id,
                    "mesh": child_mesh,
                    "color_code": union_component.color_code,
                    "color_name": child_color.get("name", union_component.color_code),
                    "color_hex": child_color.get("hex", "#C8C8C8"),
                    "annotation": {
                        "debug_recursive_layer": layer_order,
                        "stage_role": child_stats["stage_role"],
                        "source_part_index": child_index,
                        "contains": child_stats["stage_contains"],
                        "unit": "millimeter",
                    },
                }
            )

        layer_package_path = layer_dir / f"layer_{layer_order:02d}_INWARD_P{local_body_index:02d}_mm.3mf"
        export_colored_parts_3mf(
            layer_package_path,
            layer_3mf_parts,
            title=f"{input_stem} recursive layer {layer_order:02d} (millimeter)",
            source_application=source_application,
        )
        for layer_part_stats in stage_records[-len(layer_3mf_parts):]:
            layer_part_stats["layer_3mf"] = serialize_report_path(layer_package_path, output_dir, report_path_mode)

    return layers_dir, stage_records


def cumulative_snapshot_parts(
    active_parts: dict[int, dict],
    step_order: int,
    transition: dict,
) -> list[dict]:
    snapshot_parts = []
    for _active_index, entry in sorted(active_parts.items()):
        snapshot_part = {
            key: value
            for key, value in entry.items()
            if key
            not in {
                "stats",
                "source_part_index",
                "contains",
                "state_role",
                "origin_step",
            }
        }
        snapshot_part["annotation"] = {
            **entry.get("annotation", {}),
            "strict_recursive_snapshot_step": int(step_order),
            "snapshot_active_indices": transition["after_active_indices"],
            "snapshot_transition": transition,
        }
        snapshot_parts.append(snapshot_part)
    return snapshot_parts


def recursive_face_color_payload(
    mesh,
    default_color_code: str,
    default_color_hex: str,
    default_filament_slot_index: int | None,
) -> tuple[list[str], list[int | None]]:
    """Preserve the material meaning of every triangle in a recursive 3MF."""
    face_count = int(len(mesh.faces))
    face_codes = list(mesh.metadata.get("face_color_codes", []))
    face_slots = list(mesh.metadata.get("face_filament_slot_indices", []))
    if len(face_codes) == face_count:
        face_hexes = [
            COLOR_INFO.get(code, {}).get("hex", default_color_hex)
            for code in face_codes
        ]
        if len(face_slots) != face_count:
            face_slots = [
                COLOR_INFO.get(code, {}).get("filament_slot")
                for code in face_codes
            ]
        return face_hexes, [
            None if slot is None else int(slot) for slot in face_slots
        ]
    return (
        [default_color_hex for _ in range(face_count)],
        [default_filament_slot_index for _ in range(face_count)],
    )


def recursive_part_package_payload(entry: dict) -> dict:
    return {
        key: value
        for key, value in entry.items()
        if key
        not in {
            "stats",
            "source_part_index",
            "contains",
            "state_role",
            "origin_step",
            "source_3mf_path",
            "source_3mf_origin_step",
            "source_3mf_sha256",
        }
    }


def recursive_artifact_sha256(path: Path) -> str:
    """Return the identity of one on-disk parent-emitted recursive package."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_loaded_recursive_part(
    loaded: list[dict],
    expected_entry: dict,
    path: Path,
) -> dict:
    """Validate one already parsed parent-emitted package object."""
    if len(loaded) != 1:
        raise ValueError(
            f"recursive input {path} must contain exactly one mesh object; "
            f"found {len(loaded)}"
        )
    result = loaded[0]
    annotation = result.get("annotation", {})
    checks = {
        "part_id": str(expected_entry["part_id"]),
        "source_part_index": int(expected_entry["source_part_index"]),
        "state_role": str(expected_entry["state_role"]),
        "contains": [int(index) for index in expected_entry["contains"]],
        "origin_step": int(expected_entry["origin_step"]),
    }
    actual = {
        "part_id": str(result.get("part_id", "")),
        "source_part_index": int(annotation.get("source_part_index", -1)),
        "state_role": str(annotation.get("state_role", "")),
        "contains": [int(index) for index in annotation.get("contains", [])],
        "origin_step": int(annotation.get("origin_step", -1)),
    }
    if actual != checks:
        raise ValueError(
            f"recursive input provenance mismatch for {path}: "
            f"expected {checks}, found {actual}"
        )
    expected_face_colors = [
        normalize_3mf_color(value)
        for value in expected_entry.get("face_color_hexes", [])
    ]
    actual_face_colors = [
        normalize_3mf_color(value)
        for value in result.get("face_color_hexes", [])
    ]
    if actual_face_colors != expected_face_colors:
        raise ValueError(f"recursive input face colors changed in {path}")
    expected_slots = [
        None if value is None else int(value)
        for value in expected_entry.get("face_filament_slot_indices", [])
    ]
    actual_slots = [
        None if value is None else int(value)
        for value in result.get("face_filament_slot_indices", [])
    ]
    if actual_slots != expected_slots:
        raise ValueError(f"recursive input filament-slot meaning changed in {path}")
    return result


def reload_recursive_part_input(
    path: Path,
    expected_entry: dict,
    *,
    require_artifact_identity: bool = False,
) -> dict:
    """Reload exactly one parent-emitted colored 3MF and verify its provenance.

    Strict recursive consumption requires the exact standalone file emitted by
    the parent step. The checksum check runs before parsing so a replaced,
    edited, or redirected package cannot silently become the next input.
    """
    path = Path(path)
    if require_artifact_identity:
        if "CUMULATIVE" in path.name.upper():
            raise ValueError(
                f"recursive input must be a parent-emitted standalone 3MF, not {path.name}"
            )
        expected_path_text = str(expected_entry.get("source_3mf_path", ""))
        expected_sha256 = str(expected_entry.get("source_3mf_sha256", ""))
        if not expected_path_text or not expected_sha256:
            raise ValueError(
                f"recursive input identity is missing for {expected_entry.get('part_id')}"
            )
        expected_path = Path(expected_path_text)
        if path.resolve(strict=True) != expected_path.resolve(strict=True):
            raise ValueError(
                f"recursive input path changed for {expected_entry.get('part_id')}: "
                f"expected {expected_path}, found {path}"
            )
        actual_sha256 = recursive_artifact_sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"recursive input SHA-256 changed for {expected_entry.get('part_id')}: "
                f"expected {expected_sha256}, found {actual_sha256}"
            )
    loaded = load_colored_mesh_objects_3mf(path)
    return validate_loaded_recursive_part(loaded, expected_entry, path)


def recursive_component_material_key(component: Component) -> tuple[str, object]:
    """Return the serialized material identity expected for one component."""
    info = COLOR_INFO.get(str(component.color_code), {})
    slot = info.get("filament_slot")
    if slot is not None:
        return ("filament_slot", int(slot))
    return ("color_hex", normalize_3mf_color(info.get("hex", "")))


def recursive_input_geometry_context(
    reloaded_target: dict,
    expected_indices: list[int],
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    source_components: list[Component],
    boundary_fairing: BoundaryFairingContext,
) -> dict:
    """Rebuild the next recursive step entirely from its parent-emitted 3MF.

    Component ids remain the ids inferred from the original assembly tree, but
    their face ids, boundary graph, normals, centers, and model center are all
    rebound to the mesh that was serialized and reloaded by the parent step.
    """
    mesh = reloaded_target["mesh"]
    local_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    local_faces = np.asarray(mesh.faces, dtype=np.int64)
    face_hexes = [
        normalize_3mf_color(value)
        for value in reloaded_target.get("face_color_hexes", [])
    ]
    face_slots = [
        None if value is None else int(value)
        for value in reloaded_target.get("face_filament_slot_indices", [])
    ]
    if len(face_hexes) != len(local_faces) or len(face_slots) != len(local_faces):
        raise ValueError(
            "recursive input geometry cannot be rebound because its face-material "
            "payload does not match its face count"
        )

    expected_indices = sorted({int(index) for index in expected_indices})
    expected_keys = {
        recursive_component_material_key(source_components[index - 1])
        for index in expected_indices
    }
    face_material_keys: list[tuple[str, object]] = []
    for face_hex, face_slot in zip(face_hexes, face_slots):
        slot_key = (
            None
            if face_slot is None
            else ("filament_slot", int(face_slot))
        )
        hex_key = ("color_hex", face_hex)
        if slot_key in expected_keys:
            face_material_keys.append(slot_key)
        elif hex_key in expected_keys:
            face_material_keys.append(hex_key)
        else:
            raise ValueError(
                "recursive input contains a face material that is absent from "
                f"its expected descendants: slot={face_slot}, color={face_hex}"
            )

    material_labels = [
        f"{key[0]}:{key[1]}" for key in face_material_keys
    ]
    connected_regions = connected_components_by_color(local_faces, material_labels)
    local_areas = triangle_areas(local_vertices, local_faces)
    candidates_by_key: dict[tuple[str, object], list[Component]] = (
        collections.defaultdict(list)
    )
    for region in connected_regions:
        material_key = face_material_keys[int(region[0])]
        candidates_by_key[material_key].append(
            make_component_from_global_faces(
                local_vertices,
                local_faces,
                local_areas,
                region,
                str(material_key),
            )
        )

    source_extent = np.asarray(source_vertices, dtype=np.float64)
    source_scale = max(
        float(
            np.linalg.norm(
                source_extent.max(axis=0) - source_extent.min(axis=0)
            )
        ),
        1e-9,
    )

    def match_cost(source_component: Component, candidate: Component) -> float:
        center_cost = float(
            np.linalg.norm(candidate.center - source_component.center)
        ) / source_scale
        bbox_cost = float(
            np.linalg.norm(candidate.bbox_min - source_component.bbox_min)
            + np.linalg.norm(candidate.bbox_max - source_component.bbox_max)
        ) / (2.0 * source_scale)
        face_cost = abs(
            math.log(
                (float(candidate.face_count) + 1.0)
                / (float(source_component.face_count) + 1.0)
            )
        )
        return center_cost + 0.25 * bbox_cost + 0.02 * face_cost

    assigned_regions: dict[int, list[np.ndarray]] = collections.defaultdict(list)
    mapping_records: list[dict] = []
    for material_key in sorted(expected_keys, key=lambda value: (value[0], str(value[1]))):
        material_expected = [
            index
            for index in expected_indices
            if recursive_component_material_key(source_components[index - 1])
            == material_key
        ]
        material_candidates = candidates_by_key.get(material_key, [])
        if len(material_candidates) < len(material_expected):
            raise ValueError(
                "recursive input lost a connected material component for "
                f"{material_key}: expected {len(material_expected)}, "
                f"found {len(material_candidates)}"
            )
        costs = np.asarray(
            [
                [
                    match_cost(source_components[index - 1], candidate)
                    for candidate in material_candidates
                ]
                for index in material_expected
            ],
            dtype=np.float64,
        )
        from scipy.optimize import linear_sum_assignment

        expected_rows, candidate_columns = linear_sum_assignment(costs)
        claimed_candidates: set[int] = set()
        for row, column in zip(expected_rows, candidate_columns):
            component_index = int(material_expected[int(row)])
            candidate_index = int(column)
            claimed_candidates.add(candidate_index)
            candidate = material_candidates[candidate_index]
            assigned_regions[component_index].append(candidate.global_faces)
            mapping_records.append(
                {
                    "source_part_index": component_index,
                    "material_key": [material_key[0], material_key[1]],
                    "local_faces": int(candidate.face_count),
                    "match_cost": float(costs[int(row), candidate_index]),
                    "primary_region": True,
                }
            )
        for candidate_index, candidate in enumerate(material_candidates):
            if candidate_index in claimed_candidates:
                continue
            nearest_row = int(np.argmin(costs[:, candidate_index]))
            component_index = int(material_expected[nearest_row])
            assigned_regions[component_index].append(candidate.global_faces)
            mapping_records.append(
                {
                    "source_part_index": component_index,
                    "material_key": [material_key[0], material_key[1]],
                    "local_faces": int(candidate.face_count),
                    "match_cost": float(costs[nearest_row, candidate_index]),
                    "primary_region": False,
                }
            )

    zero = np.zeros(3, dtype=np.float64)
    local_components = [
        Component(
            color_code=component.color_code,
            global_faces=np.empty(0, dtype=np.int64),
            face_count=0,
            area=0.0,
            bbox_min=zero.copy(),
            bbox_max=zero.copy(),
            center=zero.copy(),
        )
        for component in source_components
    ]
    local_colors = ["" for _ in range(len(local_faces))]
    assigned_face_count = 0
    for component_index in expected_indices:
        regions = assigned_regions.get(component_index, [])
        if not regions:
            raise ValueError(
                f"recursive input did not materialize expected P{component_index:02d}"
            )
        global_faces = np.unique(np.concatenate(regions)).astype(np.int64)
        source_component = source_components[component_index - 1]
        local_component = make_component_from_global_faces(
            local_vertices,
            local_faces,
            local_areas,
            global_faces,
            source_component.color_code,
        )
        local_components[component_index - 1] = local_component
        for face_index in global_faces:
            if local_colors[int(face_index)]:
                raise ValueError(
                    f"recursive input face {int(face_index)} mapped more than once"
                )
            local_colors[int(face_index)] = source_component.color_code
            assigned_face_count += 1
    if assigned_face_count != len(local_faces) or any(
        not color for color in local_colors
    ):
        raise ValueError(
            "recursive input geometry rebind did not assign every serialized face"
        )

    local_boundary_neighbor_lookup = component_boundary_neighbor_lookup(
        local_faces,
        local_components,
    )
    return {
        "vertices": local_vertices,
        "faces": local_faces,
        "colors": local_colors,
        "components": local_components,
        "boundary_neighbor_lookup": local_boundary_neighbor_lookup,
        "component_centers": {
            index: local_components[index - 1].center
            for index in expected_indices
        },
        "model_center": local_vertices.mean(axis=0),
        "boundary_fairing": BoundaryFairingContext(
            config=boundary_fairing.config,
            source_surface_normals=mesh_vertex_inward_normals(
                local_vertices,
                local_faces,
            ),
        ),
        "mapping_records": mapping_records,
        "source_vertex_count": int(len(source_vertices)),
        "source_face_count": int(len(source_faces)),
        "reloaded_vertex_count": int(len(local_vertices)),
        "reloaded_face_count": int(len(local_faces)),
    }


def execute_strict_recursive_split(
    output_dir: Path | None,
    report_path_mode: str,
    input_stem: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: list[str],
    components: list[Component],
    assembly_parents: dict[int, int | None],
    assembly_children: dict[int, list[int]],
    recursive_steps: list[dict],
    boundary_neighbor_lookup: dict[tuple[int, int], set[int]],
    component_centers: dict[int, np.ndarray],
    inward_overrides: dict[int, np.ndarray],
    model_center: np.ndarray,
    max_extension_mm: float,
    boundary_fairing: BoundaryFairingContext,
    flat_clearance_mm: float,
    fit_clearance_by_part: dict[int, float],
    lead_in_mm: float,
    clearance_mode: str,
    sibling_clearance_mm: float,
    bottom_clearance_mm: float,
    effective_cap_mode,
    effective_planar_extra_limit,
    layer_child_context,
    validation_profile: str,
    max_topology_defect_ratio: float,
    source_application: str | None = None,
    source_filament_colors: list[str] | None = None,
    source_project_settings: dict | None = None,
) -> tuple[Path | None, list[dict], dict[int, dict], list[dict]]:
    temporary_artifacts = (
        tempfile.TemporaryDirectory(prefix="strict-recursive-3mf-")
        if output_dir is None
        else None
    )
    artifact_root = (
        Path(temporary_artifacts.name)
        if temporary_artifacts is not None
        else output_dir
    )
    artifact_layers_dir = artifact_root / "recursive_layers"
    artifact_layers_dir.mkdir(parents=True, exist_ok=True)
    layers_dir = artifact_layers_dir if output_dir is not None else None

    active_parts: dict[int, dict] = {}
    stage_records: list[dict] = []
    snapshot_records: list[dict] = []
    expected_internal_indices = {
        int(index) for index, child_indices in assembly_children.items() if child_indices
    }
    executed_internal_indices: set[int] = set()
    runtime_log(
        "递归拆件",
        "recursive_executor_start",
        "严格串行递归执行器已启动",
        recursive_step_count=int(len(recursive_steps)),
        source_component_count=int(len(components)),
        artifact_root=str(artifact_root),
        debug_output_enabled=bool(output_dir is not None),
    )

    def part_color(component_index: int) -> dict:
        component = components[int(component_index) - 1]
        return COLOR_INFO.get(
            component.color_code,
            {
                "name": component.color_code,
                "hex": "#C8C8C8",
                "rgba": [200, 200, 200, 255],
                "filament_slot": None,
            },
        )

    def active_part(
        component_index: int,
        part_id: str,
        mesh,
        stats: dict,
        state_role: str,
        contains: list[int],
        origin_step: int,
    ) -> dict:
        component = components[int(component_index) - 1]
        color_info = part_color(component_index)
        face_color_hexes, face_filament_slot_indices = recursive_face_color_payload(
            mesh,
            component.color_code,
            color_info.get("hex", "#C8C8C8"),
            color_info.get("filament_slot"),
        )
        return {
            "part_id": part_id,
            "mesh": mesh,
            "color_code": component.color_code,
            "color_name": color_info.get("name", component.color_code),
            "color_hex": color_info.get("hex", "#C8C8C8"),
            "filament_slot_index": color_info.get("filament_slot"),
            "face_color_hexes": face_color_hexes,
            "face_filament_slot_indices": face_filament_slot_indices,
            "source_part_index": int(component_index),
            "contains": [int(index) for index in contains],
            "state_role": state_role,
            "origin_step": int(origin_step),
            "stats": stats,
            "annotation": {
                "strict_serial_recursive": True,
                "state_role": state_role,
                "source_part_index": int(component_index),
                "contains": [int(index) for index in contains],
                "origin_step": int(origin_step),
                "unit": "millimeter",
            },
        }

    def validate_changed_part(
        layer_dir: Path | None,
        part_id: str,
        mesh,
        stats: dict,
        step_order: int,
        step: dict,
        state_role: str,
        contains: list[int],
    ) -> dict:
        validation = validate_mesh_in_memory(mesh)
        stats.update(
            {
                "reload_watertight": bool(validation["watertight"]),
                "reload_open_edges": int(validation["open_edges"]),
                "reload_over_shared_edges": int(validation["over_shared_edges"]),
                "reload_winding_consistent": bool(validation["winding_consistent"]),
                "reload_inconsistent_shared_edges": int(
                    validation["inconsistent_shared_edges"]
                ),
                "reload_topology_defect_ratio": float(
                    validation["topology_defect_ratio"]
                ),
            }
        )
        stats["stage_layer_order"] = int(step_order)
        stats["stage_depth"] = int(step.get("depth", 0))
        stats["stage_local_body_index"] = int(step["local_body_index"])
        stats["stage_role"] = state_role
        stats["stage_direct_child_indices"] = [
            int(index) for index in step.get("direct_child_indices", [])
        ]
        stats["stage_contains"] = [int(index) for index in contains]
        return stats

    def materialize_changed_part(
        entry: dict,
        layer_dir: Path,
        step_order: int,
    ) -> dict:
        package_path = layer_dir / f"{entry['part_id']}_mm.3mf"
        package_part = recursive_part_package_payload(entry)
        runtime_log(
            "递归落盘",
            "standalone_write_start",
            "正在写入本阶段独立零件 3MF",
            step_order=int(step_order),
            part_id=str(entry["part_id"]),
            state_role=str(entry["state_role"]),
            face_count=int(len(entry["mesh"].faces)),
            output_3mf=str(package_path),
        )
        export_colored_parts_3mf(
            package_path,
            [package_part],
            title=f"{input_stem} recursive part {entry['part_id']}",
            source_application=source_application,
            source_filament_colors=source_filament_colors,
            source_project_settings=source_project_settings,
            output_layout="separate-items",
        )
        package_validation = validate_colored_parts_3mf(
            package_path,
            [package_part],
            source_filament_colors=source_filament_colors,
            source_application=source_application,
            source_project_settings=source_project_settings,
            output_layout="separate-items",
            include_loaded_objects=True,
        )
        loaded_objects = package_validation.pop("_loaded_objects", [])
        package_errors = [
            error
            for error in package_validation.get("errors", [])
            if "reloaded mesh is not watertight" not in error
            and not (
                "reloaded mesh has " in error
                and ("open edges" in error or "over-shared edges" in error)
            )
        ]
        if package_errors:
            raise ValueError(
                f"recursive part package validation failed for {entry['part_id']}: "
                + "; ".join(package_errors)
            )
        entry["source_3mf_path"] = str(package_path)
        entry["source_3mf_origin_step"] = int(step_order)
        entry["source_3mf_sha256"] = recursive_artifact_sha256(package_path)
        reloaded = validate_loaded_recursive_part(
            loaded_objects,
            entry,
            package_path,
        )
        entry["mesh"] = reloaded["mesh"]
        entry["face_color_hexes"] = reloaded["face_color_hexes"]
        entry["face_filament_slot_indices"] = reloaded[
            "face_filament_slot_indices"
        ]
        entry["stats"]["debug_format"] = "3mf"
        report_package_path = (
            serialize_report_path(package_path, output_dir, report_path_mode)
            if output_dir is not None
            else None
        )
        entry["stats"]["file"] = report_package_path
        entry["stats"]["recursive_output_3mf"] = report_package_path
        entry["stats"]["recursive_output_3mf_sha256"] = entry[
            "source_3mf_sha256"
        ]
        entry["stats"]["recursive_color_palette"] = reloaded["palette"]
        entry["stats"]["recursive_face_color_indices"] = sorted(
            set(int(index) for index in reloaded["face_color_indices"])
        )
        runtime_log(
            "递归落盘",
            "standalone_write_done",
            "本阶段独立零件 3MF 已写入、校验并重载",
            step_order=int(step_order),
            part_id=str(entry["part_id"]),
            state_role=str(entry["state_role"]),
            face_count=int(len(entry["mesh"].faces)),
            output_3mf=str(package_path),
            sha256=str(entry["source_3mf_sha256"]),
        )
        return entry

    for step_order, step in enumerate(recursive_steps):
        local_body_index = int(step["local_body_index"])
        direct_children = [
            int(child) for child in step.get("direct_child_indices", [])
        ]
        runtime_log(
            "递归拆件",
            "recursive_step_start",
            "开始递归拆件步骤",
            step_order=int(step_order),
            step_number=int(step_order + 1),
            total_steps=int(len(recursive_steps)),
            local_body_index=int(local_body_index),
            depth=int(step.get("depth", 0)),
            direct_child_indices=direct_children,
            active_part_indices=sorted(int(index) for index in active_parts),
            input_source=(
                "root_source_3mf"
                if not active_parts
                else "parent_emitted_standalone_3mf"
            ),
        )
        if int(step.get("step_order", step_order)) != step_order:
            raise ValueError(
                f"strict recursive step order mismatch at P{local_body_index:02d}"
            )

        recursive_input_path = None
        recursive_input_origin_step = None
        step_vertices = vertices
        step_faces = faces
        step_colors = colors
        step_components = components
        step_boundary_neighbor_lookup = boundary_neighbor_lookup
        step_component_centers = component_centers
        step_model_center = model_center
        step_boundary_fairing = boundary_fairing
        recursive_geometry_mapping: list[dict] = []
        if active_parts:
            target = active_parts.get(local_body_index)
            if target is None:
                raise ValueError(
                    f"strict recursive step {step_order} cannot consume P{local_body_index:02d}; "
                    "the previous output does not contain that active subassembly"
                )
            expected_contains = subtree_component_indices(
                local_body_index, assembly_children
            )
            if sorted(int(index) for index in target.get("contains", [])) != expected_contains:
                raise ValueError(
                    f"strict recursive step {step_order} provenance mismatch for "
                    f"P{local_body_index:02d}: expected {expected_contains}, "
                    f"found {target.get('contains', [])}"
                )
            if target.get("state_role") != "pending_subassembly":
                raise ValueError(
                    f"strict recursive step {step_order} expected pending P{local_body_index:02d}, "
                    f"found role {target.get('state_role')}"
                )
            recursive_input_path = Path(str(target.get("source_3mf_path", "")))
            if not recursive_input_path.is_file():
                raise ValueError(
                    f"strict recursive step {step_order} has no parent-emitted "
                    f"3MF input for P{local_body_index:02d}"
                )
            recursive_input_origin_step = int(
                target.get("source_3mf_origin_step", -1)
            )
            runtime_log(
                "递归输入",
                "recursive_input_reload_start",
                "正在从上阶段输出的独立 3MF 重载本阶段输入",
                step_order=int(step_order),
                local_body_index=int(local_body_index),
                parent_output_step=int(recursive_input_origin_step),
                input_3mf=str(recursive_input_path),
                expected_sha256=str(target.get("source_3mf_sha256", "")),
                cumulative_3mf_is_input=False,
            )
            reloaded_target = reload_recursive_part_input(
                recursive_input_path,
                target,
                require_artifact_identity=True,
            )
            target["mesh"] = reloaded_target["mesh"]
            target["face_color_hexes"] = reloaded_target["face_color_hexes"]
            target["face_filament_slot_indices"] = reloaded_target[
                "face_filament_slot_indices"
            ]
            recursive_context = recursive_input_geometry_context(
                reloaded_target=reloaded_target,
                expected_indices=expected_contains,
                source_vertices=vertices,
                source_faces=faces,
                source_components=components,
                boundary_fairing=boundary_fairing,
            )
            step_vertices = recursive_context["vertices"]
            step_faces = recursive_context["faces"]
            step_colors = recursive_context["colors"]
            step_components = recursive_context["components"]
            step_boundary_neighbor_lookup = recursive_context[
                "boundary_neighbor_lookup"
            ]
            step_component_centers = recursive_context["component_centers"]
            step_model_center = recursive_context["model_center"]
            step_boundary_fairing = recursive_context["boundary_fairing"]
            recursive_geometry_mapping = recursive_context["mapping_records"]
            runtime_log(
                "递归输入",
                "recursive_input_reload_done",
                "上阶段独立 3MF 已完成身份校验并成为本阶段几何输入",
                step_order=int(step_order),
                local_body_index=int(local_body_index),
                parent_output_step=int(recursive_input_origin_step),
                input_3mf=str(recursive_input_path),
                verified_sha256=str(target.get("source_3mf_sha256", "")),
                reloaded_vertex_count=int(len(step_vertices)),
                reloaded_face_count=int(len(step_faces)),
                component_mapping_count=int(len(recursive_geometry_mapping)),
                cumulative_3mf_is_input=False,
            )
        elif assembly_parents.get(local_body_index) is not None:
            raise ValueError("the first strict recursive step must consume the root model")

        layer_dir = artifact_layers_dir / (
            f"layer_{step_order:02d}_INWARD_P{local_body_index:02d}"
        )
        layer_dir.mkdir(parents=True, exist_ok=True)

        if recursive_input_path is None:
            child_refs, union_by_child, subtree_by_child = layer_child_context(
                local_body_index
            )
        else:
            child_refs, union_by_child, subtree_by_child = (
                build_layer_child_cut_references(
                    vertices=step_vertices,
                    faces=step_faces,
                    components=step_components,
                    parent_index=local_body_index,
                    direct_child_indices=direct_children,
                    assembly_children=assembly_children,
                    model_center=step_model_center,
                    boundary_neighbor_lookup=step_boundary_neighbor_lookup,
                    inward_overrides=inward_overrides,
                    effective_cap_mode=effective_cap_mode,
                    effective_planar_extra_limit=effective_planar_extra_limit,
                    fit_clearance_by_part=fit_clearance_by_part,
                    clearance_mode=clearance_mode,
                    boundary_fairing=step_boundary_fairing,
                    max_extension_mm=max_extension_mm,
                    flat_clearance_mm=flat_clearance_mm,
                    bottom_clearance_mm=bottom_clearance_mm,
                )
            )
        local_component = step_components[local_body_index - 1]
        local_color = part_color(local_body_index)
        local_role = (
            "root_body"
            if assembly_parents.get(local_body_index) is None
            else "local_body"
        )
        local_part_id = (
            f"S{step_order:02d}_P{local_body_index:02d}_"
            f"{sanitize_name(local_color['name'])}_{local_role.upper()}_CUT"
        )
        local_cap_mode = effective_cap_mode(local_body_index)
        local_fit_clearance_mm = float(
            fit_clearance_by_part[local_body_index]
        )
        local_insert_shrink_mm, local_socket_overcut_mm = clearance_offsets(
            clearance_mode, local_fit_clearance_mm
        )
        local_top_edge_clearance_mm = min(local_insert_shrink_mm, 0.05)
        local_planar_extra_limit_mm = float(
            effective_planar_extra_limit(local_body_index)
        )

        runtime_log(
            "递归几何",
            "local_body_cut_start",
            "正在切分当前局部主体",
            step_order=int(step_order),
            local_body_index=int(local_body_index),
            direct_child_count=int(len(direct_children)),
            input_source=(
                "root_source_3mf"
                if recursive_input_path is None
                else "parent_emitted_standalone_3mf"
            ),
            cap_mode=str(local_cap_mode),
            preserve_unmatched_source_geometry=bool(
                recursive_input_path is not None
            ),
        )
        local_mesh, local_stats = make_body_cut_mesh(
            vertices=step_vertices,
            faces=step_faces,
            body_component=local_component,
            part_id=local_part_id,
            cut_refs=child_refs,
            max_extension_mm=max_extension_mm,
            boundary_fairing=step_boundary_fairing,
            flat_clearance_mm=flat_clearance_mm,
            fit_clearance_mm=local_fit_clearance_mm,
            socket_overcut_mm=local_socket_overcut_mm,
            bottom_clearance_mm=bottom_clearance_mm,
            clearance_mode=clearance_mode,
            model_center=step_model_center,
            cap_mode=local_cap_mode,
            planar_extra_limit_mm=local_planar_extra_limit_mm,
            preserve_unmatched_source_geometry=bool(
                recursive_input_path is not None
            ),
        )
        runtime_log(
            "递归几何",
            "local_body_cut_done",
            "当前局部主体切分完成",
            step_order=int(step_order),
            local_body_index=int(local_body_index),
            output_vertex_count=int(len(local_mesh.vertices)),
            output_face_count=int(len(local_mesh.faces)),
        )
        if recursive_input_path is not None:
            local_stats["source_faces"] = int(
                components[local_body_index - 1].face_count
            )
        local_stats["recursive_geometry_input"] = (
            "root_source_mesh"
            if recursive_input_path is None
            else "parent_emitted_standalone_3mf"
        )
        local_stats["recursive_geometry_input_3mf"] = (
            None if recursive_input_path is None else str(recursive_input_path)
        )
        local_stats["recursive_geometry_component_mapping"] = (
            recursive_geometry_mapping
        )
        local_stats = validate_changed_part(
            layer_dir,
            local_part_id,
            local_mesh,
            local_stats,
            step_order,
            step,
            local_role,
            [local_body_index],
        )
        stage_records.append(local_stats)
        local_active_part = active_part(
            local_body_index,
            local_part_id,
            local_mesh,
            local_stats,
            local_role,
            [local_body_index],
            step_order,
        )
        local_active_part = materialize_changed_part(
            local_active_part,
            layer_dir,
            step_order,
        )

        direct_child_parts: dict[int, dict] = {}
        for child_index in direct_children:
            subtree = (
                subtree_by_child.get(child_index)
                or subtree_component_indices(child_index, assembly_children)
            )
            union_component = (
                union_by_child.get(child_index)
                or build_subassembly_component(
                    step_vertices,
                    step_faces,
                    step_components,
                    subtree,
                    step_components[child_index - 1].color_code,
                )
            )
            child_color = part_color(child_index)
            child_has_descendants = bool(assembly_children.get(child_index))
            child_state_role = (
                "pending_subassembly" if child_has_descendants else "leaf_insert"
            )
            child_part_id = (
                f"S{step_order:02d}_P{child_index:02d}_"
                f"{sanitize_name(child_color['name'])}_"
                f"{'PENDING_SUBASSEMBLY' if child_has_descendants else 'LEAF_INSERT'}"
            )
            child_fit_clearance_mm = float(fit_clearance_by_part[child_index])
            child_insert_shrink_mm, _child_socket_overcut_mm = clearance_offsets(
                clearance_mode, child_fit_clearance_mm
            )
            child_top_edge_clearance_mm = min(child_insert_shrink_mm, 0.05)
            child_cap_decisions = shared_child_cap_decisions(
                child_refs,
                child_index,
            )
            runtime_log(
                "递归几何",
                "child_subassembly_build_start",
                "正在构建直属子件或待递归子装配",
                step_order=int(step_order),
                parent_body_index=int(local_body_index),
                child_index=int(child_index),
                state_role=str(child_state_role),
                subtree_indices=[int(index) for index in subtree],
            )
            child_mesh, child_stats = make_layer_child_subassembly_mesh(
                vertices=step_vertices,
                faces=step_faces,
                source_colors=step_colors,
                components=step_components,
                component=union_component,
                root_child_index=child_index,
                subtree_indices=subtree,
                parent_index=local_body_index,
                part_id=child_part_id,
                max_extension_mm=max_extension_mm,
                boundary_fairing=step_boundary_fairing,
                flat_clearance_mm=flat_clearance_mm,
                fit_clearance_mm=child_fit_clearance_mm,
                insert_shrink_mm=child_insert_shrink_mm,
                lead_in_mm=lead_in_mm,
                top_edge_clearance_mm=child_top_edge_clearance_mm,
                boundary_neighbor_lookup=step_boundary_neighbor_lookup,
                inward_override=inward_overrides.get(child_index),
                model_center=step_model_center,
                cap_mode=effective_cap_mode(child_index),
                planar_extra_limit_mm=float(
                    effective_planar_extra_limit(child_index)
                ),
                cap_decisions_by_loop=child_cap_decisions,
            )
            runtime_log(
                "递归几何",
                "child_subassembly_build_done",
                "直属子件或待递归子装配构建完成",
                step_order=int(step_order),
                parent_body_index=int(local_body_index),
                child_index=int(child_index),
                state_role=str(child_state_role),
                output_vertex_count=int(len(child_mesh.vertices)),
                output_face_count=int(len(child_mesh.faces)),
            )
            child_stats["recursive_geometry_input"] = (
                "root_source_mesh"
                if recursive_input_path is None
                else "parent_emitted_standalone_3mf"
            )
            child_stats["recursive_geometry_input_3mf"] = (
                None if recursive_input_path is None else str(recursive_input_path)
            )
            child_stats["shared_cap_decision_validation"] = (
                validate_shared_child_cap_decisions(
                    child_index,
                    child_cap_decisions,
                    child_stats,
                )
            )
            child_stats = validate_changed_part(
                layer_dir,
                child_part_id,
                child_mesh,
                child_stats,
                step_order,
                step,
                child_state_role,
                subtree,
            )
            stage_records.append(child_stats)
            child_active_part = active_part(
                child_index,
                child_part_id,
                child_mesh,
                child_stats,
                child_state_role,
                subtree,
                step_order,
            )
            direct_child_parts[child_index] = materialize_changed_part(
                child_active_part,
                layer_dir,
                step_order,
            )

        active_parts, transition = advance_strict_recursive_state(
            active_parts,
            local_body_index,
            local_active_part,
            direct_child_parts,
        )
        transition.update(
            {
                "step_order": int(step_order),
                "depth": int(step.get("depth", 0)),
                "recursion_path": [
                    int(index) for index in step.get("recursion_path", [])
                ],
                "execution_order": "strict_depth_first_preorder",
                "recursive_input_source": (
                    "root_source_3mf"
                    if recursive_input_path is None
                    else "parent_emitted_standalone_3mf"
                ),
                "recursive_input_origin_step": recursive_input_origin_step,
                "recursive_input_sha256": (
                    None
                    if recursive_input_path is None
                    else target.get("source_3mf_sha256")
                ),
                "recursive_input_3mf": (
                    None
                    if recursive_input_path is None or output_dir is None
                    else serialize_report_path(
                        recursive_input_path,
                        output_dir,
                        report_path_mode,
                    )
                ),
                "cumulative_3mf_is_recursive_input": False,
            }
        )
        executed_internal_indices.add(local_body_index)

        snapshot_parts = cumulative_snapshot_parts(
            active_parts,
            step_order,
            transition,
        )

        snapshot_record = {
            **transition,
            "mesh_count": len(snapshot_parts),
            "part_ids": [part["part_id"] for part in snapshot_parts],
            "package_validation": None,
            "reload_validation": [],
            "output_3mf": None,
        }

        if output_dir is not None:
            candidate_path = layer_dir / (
                f"layer_{step_order:02d}_INWARD_P{local_body_index:02d}_"
                "CUMULATIVE_candidate.3mf"
            )
            layer_package_path = layer_dir / (
                f"layer_{step_order:02d}_INWARD_P{local_body_index:02d}_"
                "CUMULATIVE_mm.3mf"
            )
            export_colored_parts_3mf(
                candidate_path,
                snapshot_parts,
                title=(
                    f"{input_stem} strict recursive cumulative step "
                    f"{step_order:02d} after P{local_body_index:02d}"
                ),
                source_application=source_application,
                source_filament_colors=source_filament_colors,
                source_project_settings=source_project_settings,
            )
            package_validation = validate_colored_parts_3mf(
                candidate_path,
                snapshot_parts,
                source_filament_colors=source_filament_colors,
                source_application=source_application,
                source_project_settings=source_project_settings,
                output_layout="assembly",
            )
            known_topology_errors = [
                error
                for error in package_validation.get("errors", [])
                if "reloaded mesh is not watertight" in error
                or (
                    "reloaded mesh has " in error
                    and ("open edges" in error or "over-shared edges" in error)
                )
            ]
            package_errors = [
                error
                for error in package_validation.get("errors", [])
                if error not in known_topology_errors
            ]
            if package_errors:
                failed_path = candidate_path.with_name(
                    candidate_path.stem.replace("_candidate", "_FAILED")
                    + candidate_path.suffix
                )
                candidate_path.replace(failed_path)
                snapshot_record["package_validation"] = package_validation
                snapshot_record["output_3mf"] = serialize_report_path(
                    failed_path, output_dir, report_path_mode
                )
                snapshot_records.append(snapshot_record)
                raise ValueError(
                    f"strict recursive snapshot {step_order} package validation failed: "
                    + "; ".join(package_errors)
                )

            candidate_path.replace(layer_package_path)
            reloaded_by_name = {
                record["part_id"]: record
                for record in load_colored_mesh_objects_3mf(layer_package_path)
            }
            reload_validation = []
            blocking_reload_parts = []
            for active_index, entry in sorted(active_parts.items()):
                reloaded_entry = reloaded_by_name.get(entry["part_id"])
                if reloaded_entry is None:
                    blocking_reload_parts.append(entry["part_id"])
                    reload_validation.append(
                        {
                            "part_id": entry["part_id"],
                            "valid": False,
                            "error": "missing after cumulative 3MF reload",
                        }
                    )
                    continue
                reloaded_mesh = reloaded_entry["mesh"]
                expected_face_colors = [
                    normalize_3mf_color(value)
                    for value in entry.get("face_color_hexes", [])
                ]
                if reloaded_entry["face_color_hexes"] != expected_face_colors:
                    blocking_reload_parts.append(entry["part_id"])
                    reload_validation.append(
                        {
                            "part_id": entry["part_id"],
                            "valid": False,
                            "error": "per-triangle colors changed in cumulative audit",
                        }
                    )
                    continue
                validation = validate_mesh_in_memory(reloaded_mesh)
                ratio_accepted = bool(
                    validation["winding_consistent"]
                    and not validation["inconsistent_shared_edges"]
                    and float(validation["topology_defect_ratio"])
                    <= float(max_topology_defect_ratio)
                )
                valid = bool(
                    validation_profile == "ratio"
                    and ratio_accepted
                    or validation_profile == "strict"
                    and validation["watertight"]
                    and validation["winding_consistent"]
                    and not validation["open_edges"]
                    and not validation["over_shared_edges"]
                    and not validation["inconsistent_shared_edges"]
                )
                reload_validation.append(
                    {
                        "part_id": entry["part_id"],
                        "valid": valid,
                        "watertight": bool(validation["watertight"]),
                        "open_edges": int(validation["open_edges"]),
                        "over_shared_edges": int(validation["over_shared_edges"]),
                        "inconsistent_shared_edges": int(
                            validation["inconsistent_shared_edges"]
                        ),
                        "topology_defect_ratio": float(
                            validation["topology_defect_ratio"]
                        ),
                    }
                )
                if not valid:
                    blocking_reload_parts.append(entry["part_id"])
            snapshot_record["package_validation"] = {
                **package_validation,
                "known_topology_errors": known_topology_errors,
                "errors": package_errors,
                "valid": not package_errors,
            }
            snapshot_record["reload_validation"] = reload_validation
            snapshot_record["output_3mf"] = serialize_report_path(
                layer_package_path, output_dir, report_path_mode
            )
            if blocking_reload_parts:
                snapshot_record["blocking_parts"] = blocking_reload_parts
                snapshot_records.append(snapshot_record)
                raise ValueError(
                    f"strict recursive snapshot {step_order} reload validation failed: "
                    + ", ".join(blocking_reload_parts)
                )
            for record in stage_records[-(1 + len(direct_children)):]:
                record["layer_3mf"] = serialize_report_path(
                    layer_package_path, output_dir, report_path_mode
                )

        snapshot_records.append(snapshot_record)
        runtime_log(
            "递归拆件",
            "recursive_step_done",
            "递归拆件步骤完成，状态已提交给下一步骤",
            step_order=int(step_order),
            step_number=int(step_order + 1),
            total_steps=int(len(recursive_steps)),
            local_body_index=int(local_body_index),
            active_part_indices=sorted(int(index) for index in active_parts),
            emitted_child_indices=sorted(
                int(index) for index in direct_child_parts
            ),
            next_step_must_reload_parent_output=True,
            cumulative_3mf_is_input=False,
        )

    if executed_internal_indices != expected_internal_indices:
        missing = sorted(expected_internal_indices - executed_internal_indices)
        extra = sorted(executed_internal_indices - expected_internal_indices)
        raise ValueError(
            f"strict recursive execution coverage mismatch: missing={missing}, extra={extra}"
        )
    final_indices = sorted(active_parts)
    expected_final_indices = list(range(1, len(components) + 1))
    if final_indices != expected_final_indices:
        raise ValueError(
            f"strict recursive execution did not finish with every part active: "
            f"expected {expected_final_indices}, found {final_indices}"
        )
    if temporary_artifacts is not None:
        temporary_artifacts.cleanup()
        for entry in active_parts.values():
            entry.pop("source_3mf_path", None)
    runtime_log(
        "递归拆件",
        "recursive_executor_done",
        "严格串行递归执行完成",
        recursive_step_count=int(len(recursive_steps)),
        final_part_indices=final_indices,
        stage_record_count=int(len(stage_records)),
        snapshot_record_count=int(len(snapshot_records)),
    )
    return layers_dir, stage_records, active_parts, snapshot_records
