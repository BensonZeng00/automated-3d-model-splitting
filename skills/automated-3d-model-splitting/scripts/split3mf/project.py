from __future__ import annotations

import time

from .common import *
from .reporting import runtime_log


def expand_vendor_paint_mesh(
    vertices: np.ndarray,
    source_faces: np.ndarray,
    paint_tokens: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """Restore painted subtriangles and conform their source-edge topology.

    Source-edge identities and dyadic parameters are propagated while the
    selector tree is expanded.  Conformance consequently never has to recover
    topology by projecting every generated vertex back onto source geometry.
    """
    started_at = time.perf_counter()
    if len(source_faces) != len(paint_tokens):
        raise ValueError("Source face and vendor paint token counts do not match")

    mutable_vertices = [np.asarray(vertex, dtype=np.float64) for vertex in vertices]
    midpoint_cache: dict[tuple[int, int], int] = {}
    provisional_faces: list[tuple[int, int, int]] = []
    provisional_colors: list[str] = []
    provisional_sources: list[int] = []
    state_counts: collections.Counter[int] = collections.Counter()
    split_source_faces = 0
    maximum_leaf_count = 1

    # A vertex may be on two source edges at a source corner.  Parameters are
    # always expressed from the smaller vertex id (0) to the larger one (1).
    vertex_edge_parameters: dict[int, dict[tuple[int, int], float]] = collections.defaultdict(dict)
    original_edge_points: dict[tuple[int, int], set[int]] = collections.defaultdict(set)
    original_edge_sources: dict[tuple[int, int], set[int]] = collections.defaultdict(set)

    def source_edge(a: int, b: int) -> tuple[int, int]:
        return (a, b) if a < b else (b, a)

    def register_edge_parameter(vertex_id: int, edge: tuple[int, int], parameter: float) -> None:
        existing = vertex_edge_parameters[vertex_id].get(edge)
        if existing is not None and abs(existing - parameter) > 1e-12:
            raise ValueError(
                f"Vertex {vertex_id} has conflicting parameters on original edge {edge}"
            )
        vertex_edge_parameters[vertex_id][edge] = parameter
        original_edge_points[edge].add(vertex_id)

    for source_id, source_face in enumerate(source_faces):
        a, b, c = (int(vertex_id) for vertex_id in source_face)
        for start_id, end_id in ((a, b), (b, c), (c, a)):
            # A collapsed source edge has no usable one-dimensional parameter
            # domain.  It remains part of the emitted source face, but cannot
            # participate in cross-face T-junction conformance.
            if start_id == end_id:
                continue
            edge = source_edge(start_id, end_id)
            original_edge_sources[edge].add(source_id)
            register_edge_parameter(edge[0], edge, 0.0)
            register_edge_parameter(edge[1], edge, 1.0)

    decode_started_at = time.perf_counter()
    decoded_tokens = {
        token: decode_vendor_paint_tree(token)
        for token in set(paint_tokens)
    }
    decode_duration = time.perf_counter() - decode_started_at

    def midpoint(a: int, b: int) -> int:
        if a == b:
            return a
        key = source_edge(a, b)
        existing = midpoint_cache.get(key)
        if existing is None:
            vertex_id = len(mutable_vertices)
            mutable_vertices.append((mutable_vertices[a] + mutable_vertices[b]) * 0.5)
            midpoint_cache[key] = vertex_id
        else:
            vertex_id = existing

        shared_source_edges = (
            vertex_edge_parameters[a].keys() & vertex_edge_parameters[b].keys()
        )
        for edge in shared_source_edges:
            parameter = (
                vertex_edge_parameters[a][edge] + vertex_edge_parameters[b][edge]
            ) * 0.5
            register_edge_parameter(vertex_id, edge, parameter)
        return vertex_id

    def split_triangle(face: tuple[int, int, int], split_sides: int, special_side: int) -> list[tuple[int, int, int]]:
        if split_sides == 3:
            special_side = 0
        a, b, c = (face[(special_side + offset) % 3] for offset in range(3))
        if split_sides == 1:
            opposite_midpoint = midpoint(c, b)
            return [(a, b, opposite_midpoint), (opposite_midpoint, c, a)]
        if split_sides == 2:
            midpoint_ab = midpoint(a, b)
            midpoint_ac = midpoint(a, c)
            return [
                (a, midpoint_ab, midpoint_ac),
                (midpoint_ab, b, midpoint_ac),
                (b, c, midpoint_ac),
            ]
        if split_sides == 3:
            midpoint_ab = midpoint(a, b)
            midpoint_bc = midpoint(b, c)
            midpoint_ca = midpoint(c, a)
            return [
                (a, midpoint_ab, midpoint_ca),
                (midpoint_ab, b, midpoint_bc),
                (midpoint_bc, c, midpoint_ca),
                (midpoint_ab, midpoint_bc, midpoint_ca),
            ]
        raise ValueError(f"Invalid vendor paint split count: {split_sides}")

    def emit_leaves(
        node: VendorPaintNode,
        face: tuple[int, int, int],
        source_id: int,
    ) -> int:
        if node.split_sides == 0:
            provisional_faces.append(face)
            provisional_colors.append(vendor_paint_state_token(node.state))
            provisional_sources.append(source_id)
            state_counts[node.state] += 1
            return 1
        children = split_triangle(face, node.split_sides, node.special_side)
        if node.children is None or len(node.children) != len(children):
            raise ValueError("Vendor paint split tree does not match generated children")
        leaf_count = 0
        for child_node, child_face in zip(node.children, children):
            leaf_count += emit_leaves(child_node, child_face, source_id)
        return leaf_count

    for source_id, (source_face, token) in enumerate(zip(source_faces, paint_tokens)):
        root = decoded_tokens[token]
        leaf_count = emit_leaves(
            root,
            tuple(int(vertex_id) for vertex_id in source_face),
            source_id,
        )
        if root.split_sides:
            split_source_faces += 1
        maximum_leaf_count = max(maximum_leaf_count, leaf_count)

    sources_requiring_conformance = np.zeros(len(source_faces), dtype=bool)
    subdivided_edges = {
        edge for edge, point_ids in original_edge_points.items() if len(point_ids) > 2
    }
    for edge in subdivided_edges:
        for source_id in original_edge_sources[edge]:
            sources_requiring_conformance[source_id] = True

    conformed_faces: list[tuple[int, int, int]] = []
    conformed_colors: list[str] = []
    conformed_sources: list[int] = []
    t_joint_faces = 0
    for face, color, source_id in zip(provisional_faces, provisional_colors, provisional_sources):
        if not sources_requiring_conformance[source_id]:
            conformed_faces.append(face)
            conformed_colors.append(color)
            conformed_sources.append(source_id)
            continue

        boundary: list[int] = []
        for edge_start_id, edge_end_id in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            shared_source_edges = (
                vertex_edge_parameters[edge_start_id].keys()
                & vertex_edge_parameters[edge_end_id].keys()
            )
            if len(shared_source_edges) > 1:
                raise ValueError("Painted leaf edge maps to multiple original source edges")
            if not shared_source_edges:
                ordered = [(0.0, edge_start_id)]
            else:
                edge = next(iter(shared_source_edges))
                start_parameter = vertex_edge_parameters[edge_start_id][edge]
                end_parameter = vertex_edge_parameters[edge_end_id][edge]
                span = end_parameter - start_parameter
                if abs(span) <= 1e-15:
                    raise ValueError("Painted leaf edge has zero original-edge parameter span")
                ordered = []
                for candidate_id in original_edge_points[edge]:
                    local_parameter = (
                        vertex_edge_parameters[candidate_id][edge] - start_parameter
                    ) / span
                    if -1e-12 <= local_parameter < 1.0 - 1e-12:
                        ordered.append((local_parameter, candidate_id))
                ordered.sort(key=lambda item: (item[0], item[1]))
            for _parameter, candidate_id in ordered:
                if not boundary or boundary[-1] != candidate_id:
                    boundary.append(candidate_id)

        if len(boundary) <= 3:
            conformed_faces.append(face)
            conformed_colors.append(color)
            conformed_sources.append(source_id)
            continue

        # A centroid fan preserves all collinear boundary subdivisions without
        # producing the degenerate triangles created by a corner-based fan.
        center_id = len(mutable_vertices)
        mutable_vertices.append(
            (mutable_vertices[face[0]] + mutable_vertices[face[1]] + mutable_vertices[face[2]]) / 3.0
        )
        t_joint_faces += 1
        for boundary_index, boundary_start in enumerate(boundary):
            boundary_end = boundary[(boundary_index + 1) % len(boundary)]
            if boundary_start == boundary_end:
                continue
            conformed_faces.append((center_id, boundary_start, boundary_end))
            conformed_colors.append(color)
            conformed_sources.append(source_id)

    if len(conformed_sources) != len(conformed_faces) or any(
        source_id < 0 or source_id >= len(source_faces)
        for source_id in conformed_sources
    ):
        raise ValueError("Conformed face source ownership audit failed")

    emitted_edge_counts: collections.Counter[tuple[int, int]] = collections.Counter()
    for a, b, c in conformed_faces:
        for start_id, end_id in ((a, b), (b, c), (c, a)):
            emitted_edge_counts[source_edge(start_id, end_id)] += 1

    audited_segments = 0
    for edge in subdivided_edges:
        ordered_points = sorted(
            original_edge_points[edge],
            key=lambda vertex_id: (vertex_edge_parameters[vertex_id][edge], vertex_id),
        )
        expected_uses = len(original_edge_sources[edge])
        for start_id, end_id in zip(ordered_points, ordered_points[1:]):
            audited_segments += 1
            actual_uses = emitted_edge_counts[source_edge(start_id, end_id)]
            if actual_uses != expected_uses:
                raise ValueError(
                    "Subdivided original edge segment audit failed: "
                    f"edge={edge}, segment=({start_id}, {end_id}), "
                    f"expected={expected_uses}, actual={actual_uses}"
                )

    diagnostics = {
        "profile": "bambu_triangle_selector",
        "source_faces": int(len(source_faces)),
        "split_source_faces": int(split_source_faces),
        "decoded_leaf_faces": int(len(provisional_faces)),
        "conformed_faces": int(len(conformed_faces)),
        "t_joint_faces_retriangulated": int(t_joint_faces),
        "source_faces_checked_for_t_joints": int(np.count_nonzero(sources_requiring_conformance)),
        "maximum_leaf_faces_per_source_triangle": int(maximum_leaf_count),
        "unique_paint_tokens": int(len(decoded_tokens)),
        "paint_token_decode_success_ratio": 1.0,
        "paint_token_decode_duration_seconds": round(float(decode_duration), 6),
        "audited_subdivided_edges": int(len(subdivided_edges)),
        "audited_subdivided_segments": int(audited_segments),
        "shared_edge_segment_consistency_ratio": 1.0,
        "source_face_ownership_ratio": 1.0,
        "duration_seconds": round(float(time.perf_counter() - started_at), 6),
        "leaf_state_counts": {
            vendor_paint_state_token(state): int(count)
            for state, count in sorted(state_counts.items())
        },
    }
    return (
        np.asarray(mutable_vertices, dtype=np.float64),
        np.asarray(conformed_faces, dtype=np.int64),
        conformed_colors,
        diagnostics,
    )


def hex_to_rgba(value: str, alpha: int = 255) -> list[int]:
    value = value.strip()
    if not value:
        return [200, 200, 200, alpha]
    if not value.startswith("#"):
        value = "#" + value
    if not re.match(r"^#[0-9a-fA-F]{6}$", value):
        return [200, 200, 200, alpha]
    return [int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16), alpha]


def load_json_if_exists(archive: zipfile.ZipFile, name: str) -> dict:
    if name not in archive.namelist():
        return {}
    try:
        return json.loads(archive.read(name))
    except Exception:
        return {}


def model_entry_triangle_count(archive: zipfile.ZipFile, entry: str) -> int:
    try:
        root = ET.fromstring(archive.read(entry))
    except ET.ParseError:
        return 0
    count = 0
    for mesh in root.findall(f".//{CORE_NS}mesh"):
        triangles = mesh.find(CORE_NS + "triangles")
        if triangles is not None:
            count += len(list(triangles))
    return count


def discover_mesh_model_entries(archive: zipfile.ZipFile) -> list[tuple[str, int]]:
    entries = []
    for name in archive.namelist():
        if not name.lower().endswith(".model"):
            continue
        count = model_entry_triangle_count(archive, name)
        if count > 0:
            entries.append((name, count))
    entries.sort(key=lambda item: item[1], reverse=True)
    return entries


def package_application_name(archive: zipfile.ZipFile, selected_root: ET.Element) -> str | None:
    roots = [selected_root]
    for preferred_entry in ("3D/3dmodel.model", "3dmodel.model"):
        if preferred_entry not in archive.namelist():
            continue
        try:
            roots.insert(0, ET.fromstring(archive.read(preferred_entry)))
        except ET.ParseError:
            pass
    for root in roots:
        for element in root.findall(CORE_NS + "metadata"):
            if element.attrib.get("name") == "Application" and str(element.text or "").strip():
                return str(element.text).strip()
    return None


def normalize_package_entry(value: str) -> str:
    return str(value or "").replace("\\", "/").lstrip("/")


def parse_3mf_transform(value: str | None) -> np.ndarray:
    """Parse the 3MF 4x3 affine transform into a conventional 4x4 matrix."""
    if not value:
        return np.eye(4, dtype=np.float64)
    values = [float(token) for token in str(value).split()]
    if len(values) != 12:
        raise ValueError(f"Invalid 3MF transform with {len(values)} values; expected 12")
    return np.array(
        [
            [values[0], values[3], values[6], values[9]],
            [values[1], values[4], values[7], values[10]],
            [values[2], values[5], values[8], values[11]],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def resolve_component_entry(current_entry: str, path_value: str | None, available_entries: set[str]) -> str:
    if not path_value:
        return normalize_package_entry(current_entry)
    normalized = normalize_package_entry(path_value)
    if normalized in available_entries:
        return normalized
    relative = normalize_package_entry(
        posixpath.normpath(posixpath.join(posixpath.dirname(normalize_package_entry(current_entry)), path_value))
    )
    if relative in available_entries:
        return relative
    raise ValueError(f"3MF component path not found: {path_value!r} from {current_entry!r}")


def selected_mesh_instance_transform(
    archive: zipfile.ZipFile,
    selected_entry: str,
    selected_root: ET.Element,
    unit_scale_mm: float,
) -> dict:
    """Resolve the unique build/component transform that places the selected mesh in the project."""
    model_roots: dict[str, ET.Element] = {}
    object_maps: dict[str, dict[str, ET.Element]] = {}
    for name in archive.namelist():
        normalized = normalize_package_entry(name)
        if not normalized.lower().endswith(".model"):
            continue
        try:
            root = ET.fromstring(archive.read(name))
        except ET.ParseError:
            continue
        model_roots[normalized] = root
        object_maps[normalized] = {
            str(obj.attrib["id"]): obj
            for obj in root.findall(f"./{CORE_NS}resources/{CORE_NS}object")
            if obj.attrib.get("id")
        }

    selected = normalize_package_entry(selected_entry)
    selected_mesh_ids = {
        str(obj.attrib["id"])
        for obj in selected_root.findall(f"./{CORE_NS}resources/{CORE_NS}object")
        if obj.attrib.get("id") and obj.find(CORE_NS + "mesh") is not None
    }
    matches: list[dict] = []

    def visit(entry: str, object_id: str, world: np.ndarray, trail: list[str], stack: set[tuple[str, str]]) -> None:
        key = (entry, str(object_id))
        if key in stack:
            raise ValueError("Cyclic 3MF component graph: " + " -> ".join(trail + [f"{entry}#{object_id}"]))
        obj = object_maps.get(entry, {}).get(str(object_id))
        if obj is None:
            raise ValueError(f"3MF object {object_id!r} not found in {entry!r}")
        next_trail = trail + [f"{entry}#{object_id}"]
        if entry == selected and str(object_id) in selected_mesh_ids and obj.find(CORE_NS + "mesh") is not None:
            matches.append({"matrix": world.copy(), "path": next_trail})
        components = obj.find(CORE_NS + "components")
        if components is None:
            return
        next_stack = set(stack)
        next_stack.add(key)
        for component in components.findall(CORE_NS + "component"):
            child_id = component.attrib.get("objectid")
            if not child_id:
                continue
            path_value = component.attrib.get(PRODUCTION_NS + "path") or component.attrib.get("path")
            child_entry = resolve_component_entry(entry, path_value, set(model_roots))
            child_transform = parse_3mf_transform(component.attrib.get("transform"))
            visit(child_entry, str(child_id), world @ child_transform, next_trail, next_stack)

    for entry, root in model_roots.items():
        build = root.find(CORE_NS + "build")
        if build is None:
            continue
        for item in build.findall(CORE_NS + "item"):
            object_id = item.attrib.get("objectid")
            if not object_id:
                continue
            visit(entry, str(object_id), parse_3mf_transform(item.attrib.get("transform")), [f"build:{entry}"], set())

    if not matches:
        raw_matrix = np.eye(4, dtype=np.float64)
        status = "unreferenced_mesh_entry_identity"
        instance_count = 0
        paths = []
    else:
        unique_matrices: list[np.ndarray] = []
        for match in matches:
            if not any(np.allclose(match["matrix"], candidate, rtol=1e-9, atol=1e-9) for candidate in unique_matrices):
                unique_matrices.append(match["matrix"])
        if len(unique_matrices) != 1:
            raise ValueError(
                f"Selected mesh entry has {len(unique_matrices)} distinct project instances; "
                "splitting multiple transformed instances is not supported"
            )
        raw_matrix = unique_matrices[0]
        status = "resolved_unique_project_instance"
        instance_count = len(matches)
        paths = [match["path"] for match in matches]

    matrix_mm = raw_matrix.copy()
    matrix_mm[:3, 3] *= float(unit_scale_mm)
    linear = matrix_mm[:3, :3]
    scale_factors = np.linalg.svd(linear, compute_uv=False)
    return {
        "status": status,
        "instance_count": int(instance_count),
        "matrix_4x4": matrix_mm.round(12).tolist(),
        "scale_factors": scale_factors.round(12).tolist(),
        "uniform_scale": float(np.mean(scale_factors)),
        "translation_mm": matrix_mm[:3, 3].round(12).tolist(),
        "linear_determinant": float(np.linalg.det(linear)),
        "paths": paths,
        "_matrix": matrix_mm,
    }


def model_settings_extruders(archive: zipfile.ZipFile) -> dict[str, dict]:
    """Read Bambu object/part extruders as zero-based filament slots."""
    entry = "Metadata/model_settings.config"
    if entry not in archive.namelist():
        return {}
    try:
        root = ET.fromstring(archive.read(entry))
    except ET.ParseError:
        return {}

    records: dict[str, dict] = {}
    for object_element in root.findall("./object"):
        object_id = object_element.attrib.get("id")
        if not object_id:
            continue
        object_slot = None
        for metadata in object_element.findall("./metadata"):
            if metadata.attrib.get("key") == "extruder":
                try:
                    one_based = int(metadata.attrib.get("value", ""))
                except ValueError:
                    one_based = 0
                object_slot = one_based - 1 if one_based > 0 else None
                break
        part_slots: dict[str, int] = {}
        for part in object_element.findall("./part"):
            part_id = part.attrib.get("id")
            if not part_id:
                continue
            for metadata in part.findall("./metadata"):
                if metadata.attrib.get("key") != "extruder":
                    continue
                try:
                    one_based = int(metadata.attrib.get("value", ""))
                except ValueError:
                    one_based = 0
                if one_based > 0:
                    part_slots[str(part_id)] = one_based - 1
                break
        records[str(object_id)] = {"object_slot": object_slot, "part_slots": part_slots}
    return records


def component_parent_object_ids(archive: zipfile.ZipFile, selected_entry: str) -> list[str]:
    """Find package objects whose component path references the selected mesh entry."""
    selected = normalize_package_entry(selected_entry)
    parents: set[str] = set()
    for name in archive.namelist():
        if not name.lower().endswith(".model") or normalize_package_entry(name) == selected:
            continue
        try:
            root = ET.fromstring(archive.read(name))
        except ET.ParseError:
            continue
        for object_element in root.findall(f".//{CORE_NS}object"):
            object_id = object_element.attrib.get("id")
            components = object_element.find(CORE_NS + "components")
            if not object_id or components is None:
                continue
            for component in components.findall(CORE_NS + "component"):
                path_value = component.attrib.get(PRODUCTION_NS + "path") or component.attrib.get("path")
                if normalize_package_entry(path_value) == selected:
                    parents.add(str(object_id))
    return sorted(parents)


def infer_default_filament_slot(
    archive: zipfile.ZipFile,
    selected_entry: str,
    selected_root: ET.Element,
) -> dict:
    """Resolve unpainted triangles through Bambu object/part extruder metadata."""
    settings = model_settings_extruders(archive)
    candidates: list[tuple[int, str, str]] = []

    mesh_object_ids = [
        str(obj.attrib["id"])
        for obj in selected_root.findall(f".//{CORE_NS}object")
        if obj.attrib.get("id") and obj.find(CORE_NS + "mesh") is not None
    ]
    for object_id in mesh_object_ids:
        record = settings.get(object_id)
        if not record:
            continue
        part_slots = record.get("part_slots", {})
        if object_id in part_slots:
            candidates.append((int(part_slots[object_id]), "model_settings_part_extruder", object_id))
        elif record.get("object_slot") is not None:
            candidates.append((int(record["object_slot"]), "model_settings_object_extruder", object_id))

    for parent_id in component_parent_object_ids(archive, selected_entry):
        record = settings.get(parent_id)
        if not record:
            continue
        matched_part_slot = False
        for part_id in mesh_object_ids:
            if part_id in record.get("part_slots", {}):
                candidates.append(
                    (int(record["part_slots"][part_id]), "model_settings_component_part_extruder", f"{parent_id}:{part_id}")
                )
                matched_part_slot = True
        if not matched_part_slot and record.get("object_slot") is not None:
            candidates.append((int(record["object_slot"]), "model_settings_component_parent_extruder", parent_id))

    if not candidates and len(settings) == 1:
        object_id, record = next(iter(settings.items()))
        if record.get("object_slot") is not None:
            candidates.append((int(record["object_slot"]), "model_settings_single_object_extruder", object_id))

    unique_slots = sorted({slot for slot, _source, _object_id in candidates})
    if len(unique_slots) == 1:
        selected_candidates = [item for item in candidates if item[0] == unique_slots[0]]
        return {
            "slot": int(unique_slots[0]),
            "slot_number": int(unique_slots[0]) + 1,
            "mapping_source": selected_candidates[0][1],
            "object_ids": sorted({item[2] for item in selected_candidates}),
            "status": "resolved",
        }
    return {
        "slot": None,
        "slot_number": None,
        "mapping_source": "unresolved_default_extruder",
        "object_ids": sorted({item[2] for item in candidates}),
        "candidate_slots": [slot + 1 for slot in unique_slots],
        "status": "conflict" if unique_slots else "missing",
    }


def inspect_model_support(root: ET.Element, requested_profile: str) -> dict:
    triangles = root.findall(f".//{CORE_NS}triangle")
    objects = root.findall(f".//{CORE_NS}object")
    build_items = root.findall(f".//{CORE_NS}build/{CORE_NS}item")
    mesh_object_ids = {
        obj.attrib.get("id")
        for obj in objects
        if obj.find(CORE_NS + "mesh") is not None and obj.attrib.get("id")
    }
    build_object_ids = {item.attrib.get("objectid") for item in build_items if item.attrib.get("objectid")}
    has_vendor_paint = any("paint_color" in triangle.attrib for triangle in triangles)
    has_standard_properties = any(
        any(key in triangle.attrib for key in ("pid", "p1", "p2", "p3"))
        for triangle in triangles
    ) or any(any(key in obj.attrib for key in ("pid", "pindex")) for obj in objects)
    has_components = any(obj.find(CORE_NS + "components") is not None for obj in objects)
    transformed_build_items = [item.attrib.get("objectid") for item in build_items if item.attrib.get("transform")]

    if requested_profile == "auto":
        detected_profile = "vendor-paint" if has_vendor_paint else "standard-materials" if has_standard_properties else "unpainted"
    else:
        detected_profile = requested_profile

    unsupported = []
    if requested_profile == "vendor-paint" and not has_vendor_paint:
        unsupported.append("--format-profile vendor-paint requires at least one triangle with paint_color")
    if detected_profile != "vendor-paint":
        unsupported.append(
            "this release supports per-triangle paint_color data only; standard 3MF material/color properties are not yet supported"
        )
    if has_components:
        unsupported.append("3MF component objects are not supported")
    if build_object_ids and mesh_object_ids and build_object_ids != mesh_object_ids:
        unsupported.append("the build references a different mesh-object set than the model resources")
    if unsupported:
        raise ValueError("Unsupported 3MF features: " + "; ".join(unsupported))

    return {
        "requested_profile": requested_profile,
        "detected_profile": detected_profile,
        "has_vendor_paint": has_vendor_paint,
        "has_standard_properties": has_standard_properties,
        "mesh_object_count": len(mesh_object_ids),
        "build_item_count": len(build_items),
        "transformed_build_item_count": len(transformed_build_items),
    }


def parse_3mf_model(
    path: Path,
    model_entry: str | None = None,
    format_profile: str = "auto",
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    with zipfile.ZipFile(path) as archive:
        mesh_entries = discover_mesh_model_entries(archive)
        if not mesh_entries:
            raise ValueError(f"No mesh model entries found in {path}")
        selected_entry = model_entry or mesh_entries[0][0]
        if selected_entry not in archive.namelist():
            raise ValueError(f"Requested model entry not found in 3MF: {selected_entry}")
        model_xml = archive.read(selected_entry)
        root = ET.fromstring(model_xml)
        project_settings = load_json_if_exists(archive, "Metadata/project_settings.config")
        default_filament = infer_default_filament_slot(archive, selected_entry, root)
        source_unit = root.attrib.get("unit", "millimeter").lower()
        if source_unit not in UNIT_TO_MM:
            raise ValueError(f"Unsupported 3MF unit: {source_unit!r}")
        unit_scale_mm = UNIT_TO_MM[source_unit]
        instance_transform = selected_mesh_instance_transform(
            archive,
            selected_entry,
            root,
            unit_scale_mm,
        )
        source_application = package_application_name(archive, root)

    support = inspect_model_support(root, format_profile)
    vertices_blocks = []
    faces_blocks = []
    raw_colors = []
    vertex_offset = 0
    for mesh in root.findall(f".//{CORE_NS}mesh"):
        vertices_elem = mesh.find(CORE_NS + "vertices")
        triangles_elem = mesh.find(CORE_NS + "triangles")
        if vertices_elem is None or triangles_elem is None:
            continue
        mesh_vertices = np.array(
            [
                (float(v.attrib["x"]), float(v.attrib["y"]), float(v.attrib["z"]))
                for v in vertices_elem
            ],
            dtype=np.float64,
        ) * unit_scale_mm
        mesh_faces = []
        for triangle in triangles_elem:
            mesh_faces.append(
                (
                    int(triangle.attrib["v1"]) + vertex_offset,
                    int(triangle.attrib["v2"]) + vertex_offset,
                    int(triangle.attrib["v3"]) + vertex_offset,
                )
            )
            raw_colors.append(triangle.attrib.get("paint_color", "DEFAULT"))
        if len(mesh_vertices) and mesh_faces:
            vertices_blocks.append(mesh_vertices)
            faces_blocks.append(np.array(mesh_faces, dtype=np.int64))
            vertex_offset += len(mesh_vertices)

    if not vertices_blocks or not faces_blocks:
        raise ValueError(f"Selected model entry has no mesh triangles: {selected_entry}")

    project_settings["_selected_model_entry"] = selected_entry
    project_settings["_available_model_entries"] = [{"entry": e, "triangles": c} for e, c in mesh_entries]
    project_settings["_format_support"] = support
    project_settings["_source_unit"] = source_unit
    project_settings["_unit_scale_mm"] = unit_scale_mm
    project_settings["_source_application"] = source_application
    project_settings["_default_filament"] = default_filament
    project_settings["_default_filament_slot"] = default_filament.get("slot")
    source_vertices = np.vstack(vertices_blocks)
    source_faces = np.vstack(faces_blocks)
    raw_bbox = np.vstack((source_vertices.min(axis=0), source_vertices.max(axis=0)))
    matrix_mm = instance_transform.pop("_matrix")
    source_vertices = (
        np.column_stack((source_vertices, np.ones(len(source_vertices), dtype=np.float64))) @ matrix_mm.T
    )[:, :3]
    if float(instance_transform["linear_determinant"]) < 0.0:
        source_faces = source_faces[:, [0, 2, 1]]
    transformed_bbox = np.vstack((source_vertices.min(axis=0), source_vertices.max(axis=0)))
    instance_transform["raw_mesh_bbox_size_mm"] = (raw_bbox[1] - raw_bbox[0]).round(12).tolist()
    instance_transform["project_instance_bbox_min_mm"] = transformed_bbox[0].round(12).tolist()
    instance_transform["project_instance_bbox_max_mm"] = transformed_bbox[1].round(12).tolist()
    instance_transform["project_instance_bbox_size_mm"] = (transformed_bbox[1] - transformed_bbox[0]).round(12).tolist()
    project_settings["_source_instance_transform"] = instance_transform
    expanded_vertices, expanded_faces, colors, paint_diagnostics = expand_vendor_paint_mesh(
        source_vertices,
        source_faces,
        raw_colors,
    )
    project_settings["_vendor_paint_decode"] = paint_diagnostics
    runtime_log(
        "paint-expansion",
        "vendor_paint_mesh_conformed",
        "Paint selector expansion and source-edge conformance completed",
        **paint_diagnostics,
    )
    return expanded_vertices, expanded_faces, colors, project_settings


def decode_vendor_color_slot(code: str) -> int | None:
    """Decode Bambu-style paint_color tokens to a zero-based filament slot.

    The token is an opaque hexadecimal paint ID, not a decimal list index.
    Known vendor sequences are 4, 8, C, 1C, 2C .... DEFAULT is resolved
    separately from object/part extruder metadata and is never assumed slot zero.
    """
    token = str(code).strip().upper()
    if token == "DEFAULT":
        return None
    if not re.fullmatch(r"[0-9A-F]+", token):
        return None
    try:
        node = decode_vendor_paint_tree(token)
    except ValueError:
        return None
    if node.split_sides:
        return None
    return int(node.state) - 1 if node.state > 0 else None


def build_color_info_map(project_settings: dict, colors: list[str], override_json: Path | None = None) -> tuple[dict, dict]:
    color_info = dict(DEFAULT_COLOR_INFO)
    filament_colours = project_settings.get("filament_colour") or project_settings.get("filament_multi_colour") or []
    default_slot = project_settings.get("_default_filament_slot")
    default_filament = project_settings.get("_default_filament", {})
    if (
        isinstance(default_slot, int)
        and filament_colours
        and 0 <= default_slot < len(filament_colours)
    ):
        color_info["DEFAULT"] = {
            "name": f"default_filament_{default_slot + 1}",
            "hex": filament_colours[default_slot],
            "rgba": hex_to_rgba(filament_colours[default_slot]),
            "filament_slot": default_slot,
            "mapping_source": default_filament.get("mapping_source", "model_settings_object_extruder"),
        }
    unique_codes = sorted(set(colors), key=lambda c: (0 if c == "DEFAULT" else 1, str(c)))
    for idx, code in enumerate(unique_codes):
        if code in color_info:
            continue
        palette_hex = None
        vendor_slot = decode_vendor_color_slot(code)
        slot = vendor_slot
        if slot is None and str(code).isdigit():
            slot = int(code)
        mapping_source = "fallback_palette"
        if slot is not None and filament_colours and 0 <= slot < len(filament_colours):
            palette_hex = filament_colours[slot]
            mapping_source = (
                "project_filament_metadata_vendor_token"
                if vendor_slot is not None
                else "project_filament_metadata_numeric_index"
            )
        if palette_hex is None:
            palette_hex = FALLBACK_PALETTE[idx % len(FALLBACK_PALETTE)]
        color_info[code] = {
            "name": f"color_{sanitize_name(str(code)) or idx}",
            "hex": palette_hex,
            "rgba": hex_to_rgba(palette_hex),
            "filament_slot": slot,
            "mapping_source": mapping_source,
        }

    if override_json:
        overrides = json.loads(override_json.read_text(encoding="utf-8"))
        for code, value in overrides.items():
            if isinstance(value, str):
                color_info[code] = {
                    "name": color_info.get(code, {}).get("name", f"color_{sanitize_name(code)}"),
                    "hex": value,
                    "rgba": hex_to_rgba(value),
                    "filament_slot": color_info.get(code, {}).get("filament_slot"),
                    "mapping_source": "user_override",
                }
            elif isinstance(value, dict):
                hex_value = value.get("hex", color_info.get(code, {}).get("hex", "#C8C8C8"))
                color_info[code] = {
                    "name": value.get("name", f"color_{sanitize_name(code)}"),
                    "hex": hex_value,
                    "rgba": value.get("rgba", hex_to_rgba(hex_value)),
                    "filament_slot": color_info.get(code, {}).get("filament_slot"),
                    "mapping_source": "user_override",
                }

    color_order = {code: index for index, code in enumerate(unique_codes)}
    for code, index in COLOR_ORDER.items():
        if code in color_order:
            color_order[code] = index
    return color_info, color_order


def color_resolution_status(color_info: dict) -> str:
    mapping_source = str(color_info.get("mapping_source", "unmapped"))
    if mapping_source == "user_override":
        return "explicit_override"
    if mapping_source.startswith("project_filament_metadata") or mapping_source.startswith("model_settings_"):
        return "source_metadata"
    return "fallback_estimate"


def material_connectivity_labels(colors: list[str]) -> tuple[list[str], dict]:
    """Coalesce raw paint tokens only when source metadata resolves them to one material."""
    token_to_label: dict[str, str] = {}
    material_groups: dict[str, set[str]] = collections.defaultdict(set)
    for token in sorted(set(str(color) for color in colors)):
        info = COLOR_INFO.get(token, {})
        status = color_resolution_status(info)
        slot = info.get("filament_slot")
        if status in {"source_metadata", "explicit_override"} and slot is not None:
            label = f"material:filament-slot:{int(slot)}"
        elif status == "explicit_override" and info.get("hex"):
            label = f"material:rgba:{normalize_3mf_color(str(info['hex']))}"
        else:
            label = f"raw-token:{token}"
        token_to_label[token] = label
        material_groups[label].add(token)
    merged_groups = [
        {"material_label": label, "raw_color_tokens": sorted(tokens)}
        for label, tokens in sorted(material_groups.items())
        if len(tokens) > 1
    ]
    return [token_to_label[str(color)] for color in colors], {
        "basis": "resolved_filament_slot_then_explicit_rgba_else_raw_token",
        "token_to_material_label": token_to_label,
        "merged_material_groups": merged_groups,
    }




class VendorPaintDecoder:
    """Decode vendor TriangleSelector paint streams without owning pipeline state."""

    decode = staticmethod(decode_vendor_paint_tree)
    expand = staticmethod(expand_vendor_paint_mesh)


class ThreeMFReader:
    """Read one selected 3MF project instance in millimeter space."""

    def read(self, input_path: Path, *, model_entry: str | None, format_profile: str):
        return parse_3mf_model(input_path, model_entry=model_entry, format_profile=format_profile)
