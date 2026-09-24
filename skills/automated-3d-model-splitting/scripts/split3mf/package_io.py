from __future__ import annotations

from .common import *
from .project import *
from .recognition import *
from .mesh import *

def open_edge_count(mesh: trimesh.Trimesh) -> int:
    check = mesh.copy()
    check.merge_vertices(digits_vertex=6)
    if len(check.edges_unique_inverse) == 0:
        return 0
    counts = np.bincount(check.edges_unique_inverse)
    return int(np.sum(counts == 1))


def sanitize_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_")


def serialize_report_path(path: Path, output_dir: Path, mode: str) -> str:
    resolved = path.expanduser().resolve()
    if mode == "absolute":
        return str(resolved)
    try:
        return resolved.relative_to(output_dir.expanduser().resolve()).as_posix()
    except ValueError:
        return resolved.name


def normalize_3mf_color(value: str) -> str:
    token = str(value or "").strip().upper()
    if not token.startswith("#"):
        token = "#" + token
    if re.fullmatch(r"#[0-9A-F]{6}", token):
        return token + "FF"
    if re.fullmatch(r"#[0-9A-F]{8}", token):
        return token
    return "#C8C8C8FF"


def output_color_layout(parts: list[dict], source_filament_colors: list[str] | None = None) -> tuple[list[str], list[int]]:
    """Preserve source filament-slot order and resolve each object's color index."""
    source_palette = [normalize_3mf_color(value) for value in (source_filament_colors or [])]
    ordered_colors = list(source_palette)
    first_color_index: dict[str, int] = {}
    for index, rgba in enumerate(ordered_colors):
        first_color_index.setdefault(rgba, index)

    part_color_indices: list[int] = []
    for part in parts:
        rgba = normalize_3mf_color(part["color_hex"])
        slot = part.get("filament_slot_index")
        resolution = str(part.get("color_resolution_status", ""))
        source_mapped = resolution == "source_metadata" and slot is not None
        if source_mapped:
            slot_index = int(slot)
            if not 0 <= slot_index < len(source_palette):
                raise ValueError(f"{part['part_id']}: source filament slot {slot_index} is outside the preserved palette")
            if source_palette[slot_index] != rgba:
                raise ValueError(
                    f"{part['part_id']}: resolved color {rgba} does not match source filament slot "
                    f"{slot_index} color {source_palette[slot_index]}"
                )
            part_color_indices.append(slot_index)
            continue
        if rgba not in first_color_index:
            first_color_index[rgba] = len(ordered_colors)
            ordered_colors.append(rgba)
        part_color_indices.append(first_color_index[rgba])
    return ordered_colors, part_color_indices


def output_face_color_layout(
    parts: list[dict],
    ordered_colors: list[str],
) -> list[list[int] | None]:
    """Resolve optional per-triangle colors without losing source filament slots."""
    first_color_index: dict[str, int] = {}
    for index, rgba in enumerate(ordered_colors):
        first_color_index.setdefault(rgba, index)

    face_color_indices: list[list[int] | None] = []
    for part in parts:
        raw_face_colors = part.get("face_color_hexes")
        if raw_face_colors is None:
            face_color_indices.append(None)
            continue
        face_colors = [normalize_3mf_color(value) for value in raw_face_colors]
        face_count = int(len(part["mesh"].faces))
        if len(face_colors) != face_count:
            raise ValueError(
                f"{part['part_id']}: expected {face_count} face colors, "
                f"found {len(face_colors)}"
            )
        raw_slots = part.get("face_filament_slot_indices")
        if raw_slots is None:
            slots: list[int | None] = [None] * face_count
        else:
            slots = [
                None if value is None else int(value)
                for value in raw_slots
            ]
            if len(slots) != face_count:
                raise ValueError(
                    f"{part['part_id']}: expected {face_count} face filament slots, "
                    f"found {len(slots)}"
                )

        resolved_indices: list[int] = []
        for face_index, (rgba, slot) in enumerate(zip(face_colors, slots)):
            if slot is not None:
                if not 0 <= slot < len(ordered_colors):
                    raise ValueError(
                        f"{part['part_id']}: face {face_index} filament slot {slot} "
                        "is outside the preserved palette"
                    )
                if ordered_colors[slot] != rgba:
                    raise ValueError(
                        f"{part['part_id']}: face {face_index} color {rgba} does not "
                        f"match source filament slot {slot} color {ordered_colors[slot]}"
                    )
                resolved_indices.append(slot)
                continue
            if rgba not in first_color_index:
                first_color_index[rgba] = len(ordered_colors)
                ordered_colors.append(rgba)
            resolved_indices.append(first_color_index[rgba])
        face_color_indices.append(resolved_indices)
    return face_color_indices


def rgba_bytes_from_3mf_color(value: str) -> np.ndarray:
    rgba = normalize_3mf_color(value)
    return np.asarray(
        [int(rgba[index : index + 2], 16) for index in (1, 3, 5, 7)],
        dtype=np.uint8,
    )


def load_colored_mesh_objects_3mf(path: Path) -> list[dict]:
    """Load mesh objects, annotations, and per-triangle 3MF color properties."""
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("3D/3dmodel.model"))
    resources = root.find(CORE_NS + "resources")
    if resources is None:
        raise ValueError(f"{path}: model is missing resources")
    color_groups = resources.findall(MATERIAL_NS + "colorgroup")
    if len(color_groups) != 1:
        raise ValueError(f"{path}: expected one color group, found {len(color_groups)}")
    color_group = color_groups[0]
    color_group_id = color_group.attrib.get("id")
    palette = [
        normalize_3mf_color(element.attrib.get("color", ""))
        for element in color_group.findall(MATERIAL_NS + "color")
    ]

    loaded: list[dict] = []
    for object_element in resources.findall(CORE_NS + "object"):
        mesh_element = object_element.find(CORE_NS + "mesh")
        if mesh_element is None:
            continue
        vertices_parent = mesh_element.find(CORE_NS + "vertices")
        triangles_parent = mesh_element.find(CORE_NS + "triangles")
        vertices = np.asarray(
            [
                [float(vertex.attrib[axis]) for axis in ("x", "y", "z")]
                for vertex in (
                    list(vertices_parent) if vertices_parent is not None else []
                )
            ],
            dtype=np.float64,
        )
        triangle_elements = (
            list(triangles_parent) if triangles_parent is not None else []
        )
        faces = np.asarray(
            [
                [int(triangle.attrib[key]) for key in ("v1", "v2", "v3")]
                for triangle in triangle_elements
            ],
            dtype=np.int64,
        )
        try:
            default_color_index = int(object_element.attrib.get("pindex", "0"))
        except ValueError as exc:
            raise ValueError(
                f"{path}: object {object_element.attrib.get('name')} has invalid pindex"
            ) from exc
        if not 0 <= default_color_index < len(palette):
            raise ValueError(f"{path}: object default color is outside the palette")
        face_color_indices: list[int] = []
        face_paint_color_tokens: list[str | None] = []
        for triangle in triangle_elements:
            triangle_pid = triangle.attrib.get(
                "pid", object_element.attrib.get("pid")
            )
            if triangle_pid is not None and triangle_pid != color_group_id:
                raise ValueError(
                    f"{path}: object {object_element.attrib.get('name')} triangle "
                    f"references unsupported color group {triangle_pid}"
                )
            property_values = [
                triangle.attrib.get("p1"),
                triangle.attrib.get("p2"),
                triangle.attrib.get("p3"),
            ]
            present = [value for value in property_values if value is not None]
            if present and len(set(present)) != 1:
                raise ValueError(
                    f"{path}: object {object_element.attrib.get('name')} uses "
                    "per-vertex color interpolation; recursive inputs require one "
                    "material meaning per triangle"
                )
            color_index = int(present[0]) if present else default_color_index
            if not 0 <= color_index < len(palette):
                raise ValueError(
                    f"{path}: object {object_element.attrib.get('name')} triangle "
                    f"color index {color_index} is outside the palette"
                )
            face_color_indices.append(color_index)
            face_paint_color_tokens.append(triangle.attrib.get("paint_color"))

        metadata_element = object_element.find(
            CORE_NS
            + "metadata[@name='automated-3d-model-splitting:annotation']"
        )
        annotation = {}
        if metadata_element is not None and metadata_element.text:
            annotation = json.loads(metadata_element.text)
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            process=False,
            metadata={"name": object_element.attrib.get("name", "")},
        )
        if face_color_indices:
            palette_rgba = np.asarray(
                [rgba_bytes_from_3mf_color(color) for color in palette],
                dtype=np.uint8,
            )
            mesh.visual.face_colors = palette_rgba[
                np.asarray(face_color_indices, dtype=np.int64)
            ]
        loaded.append(
            {
                "part_id": object_element.attrib.get("name", ""),
                "mesh": mesh,
                "annotation": annotation,
                "palette": palette,
                "object_color_index": default_color_index,
                "color_hex": palette[default_color_index],
                "filament_slot_index": default_color_index,
                "color_resolution_status": "source_metadata",
                "color_code": annotation.get("color_code", ""),
                "color_name": annotation.get("color_name", ""),
                "face_color_indices": face_color_indices,
                "face_color_hexes": [
                    palette[index] for index in face_color_indices
                ],
                "face_filament_slot_indices": face_color_indices,
                "face_paint_color_tokens": face_paint_color_tokens,
            }
        )
    return loaded


def format_3mf_float(value: float) -> str:
    number = float(value)
    if abs(number) < 5e-12:
        number = 0.0
    return format(number, ".10g")


def mesh_ready_for_3mf_serialization(
    mesh: trimesh.Trimesh,
) -> tuple[trimesh.Trimesh, dict]:
    """Orient closed shells on the exact coordinate grid written to 3MF.

    A Boolean can leave disconnected cavity or micro-shell components whose
    containment sample lies within a few ulps of a neighbouring surface.
    Auditing full-precision vertices and then rounding each coordinate during
    XML serialization may change that containment classification.  Quantize
    first, repair only triangle winding, and make the writer and reload audit
    operate on precisely the same geometry.
    """

    serialized_vertices = np.asarray(
        [
            [float(format_3mf_float(value)) for value in vertex]
            for vertex in np.asarray(mesh.vertices, dtype=np.float64)
        ],
        dtype=np.float64,
    )
    result = trimesh.Trimesh(
        vertices=serialized_vertices,
        faces=np.asarray(mesh.faces, dtype=np.int64).copy(),
        process=False,
        metadata=mesh.metadata.copy(),
    )
    orientation_record = orient_mesh_faces_consistently(result)
    orientation_audit = watertight_component_orientation_audit(result)
    if bool(result.is_watertight) and (
        not bool(result.is_winding_consistent)
        or orientation_audit.get("inward_closed_component_count")
    ):
        raise ValueError(
            "3MF serialization-grid orientation repair did not converge"
        )
    return result, {
        "coordinate_format": ".10g",
        "orientation_repair": orientation_record,
        "orientation_audit": orientation_audit,
    }


def xml_document_bytes(root: ET.Element) -> bytes:
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def write_deterministic_zip_member(archive: zipfile.ZipFile, name: str, data: str | bytes) -> None:
    payload = data.encode("utf-8") if isinstance(data, str) else data
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, payload)


def bambu_project_settings_payload(
    source_project_settings: dict | None,
    source_filament_colors: list[str] | None,
) -> bytes | None:
    """Serialize source Bambu settings without parser-only internal fields."""
    if not isinstance(source_project_settings, dict):
        return None
    settings = {
        str(key): value
        for key, value in source_project_settings.items()
        if not str(key).startswith("_")
    }
    if not settings:
        return None
    if source_filament_colors:
        settings["filament_colour"] = [normalize_3mf_color(value)[:7] for value in source_filament_colors]
    return (json.dumps(settings, ensure_ascii=False, indent=4, sort_keys=True) + "\n").encode("utf-8")


def bambu_model_settings_bytes(
    build_records: list[dict],
    output_layout: str = "assembly",
    assembly_id: int | None = None,
    assembly_name: str = "Split Painted 3MF Assembly",
) -> bytes:
    """Create Bambu object, part, plate, and one-based extruder configuration."""
    config = ET.Element("config")
    if output_layout == "assembly":
        if assembly_id is None:
            raise ValueError("assembly layout requires an assembly id")
        object_element = ET.SubElement(config, "object", {"id": str(int(assembly_id))})
        ET.SubElement(object_element, "metadata", {"key": "name", "value": str(assembly_name)})
        ET.SubElement(object_element, "metadata", {"key": "extruder", "value": "1"})
        ET.SubElement(object_element, "metadata", {"key": "flush_into_infill", "value": "0"})
        ET.SubElement(object_element, "metadata", {"key": "flush_into_support", "value": "0"})
        ET.SubElement(
            object_element,
            "metadata",
            {"face_count": str(sum(int(record["triangles"]) for record in build_records))},
        )
        object_records = [(object_element, record) for record in build_records]
    else:
        object_records = []
        for record in build_records:
            object_id = int(record["object_id"])
            name = str(record["part_id"])
            object_element = ET.SubElement(config, "object", {"id": str(object_id)})
            ET.SubElement(object_element, "metadata", {"key": "name", "value": name})
            slot = record.get("filament_slot_index")
            extruder = int(slot) + 1 if slot is not None else int(record["color_index"]) + 1
            ET.SubElement(object_element, "metadata", {"key": "extruder", "value": str(extruder)})
            ET.SubElement(object_element, "metadata", {"key": "flush_into_infill", "value": "0"})
            ET.SubElement(object_element, "metadata", {"key": "flush_into_support", "value": "0"})
            ET.SubElement(object_element, "metadata", {"face_count": str(int(record["triangles"]))})
            object_records.append((object_element, record))

    for object_element, record in object_records:
        object_id = int(record["object_id"])
        name = str(record["part_id"])
        slot = record.get("filament_slot_index")
        extruder = int(slot) + 1 if slot is not None else int(record["color_index"]) + 1
        part_element = ET.SubElement(
            object_element,
            "part",
            {"id": str(object_id), "subtype": "normal_part"},
        )
        ET.SubElement(part_element, "metadata", {"key": "name", "value": name})
        ET.SubElement(part_element, "metadata", {"key": "extruder", "value": str(extruder)})
        ET.SubElement(
            part_element,
            "metadata",
            {"key": "matrix", "value": "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"},
        )
        ET.SubElement(
            part_element,
            "mesh_stat",
            {
                "face_count": str(int(record["triangles"])),
                "edges_fixed": "0",
                "degenerate_facets": "0",
                "facets_removed": "0",
                "facets_reversed": "0",
                "backwards_edges": "0",
            },
        )

    plate = ET.SubElement(config, "plate")
    ET.SubElement(plate, "metadata", {"key": "plater_id", "value": "1"})
    ET.SubElement(plate, "metadata", {"key": "plater_name", "value": ""})
    ET.SubElement(plate, "metadata", {"key": "locked", "value": "false"})
    plate_records = (
        [{"object_id": int(assembly_id)}]
        if output_layout == "assembly"
        else build_records
    )
    for instance_id, record in enumerate(plate_records):
        model_instance = ET.SubElement(plate, "model_instance")
        ET.SubElement(model_instance, "metadata", {"key": "object_id", "value": str(record["object_id"])})
        ET.SubElement(model_instance, "metadata", {"key": "instance_id", "value": "0"})
        ET.SubElement(model_instance, "metadata", {"key": "identify_id", "value": str(instance_id + 1)})

    assemble = ET.SubElement(config, "assemble")
    for record in plate_records:
        ET.SubElement(
            assemble,
            "assemble_item",
            {
                "object_id": str(record["object_id"]),
                "instance_id": "0",
                "transform": "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1",
                "offset": "0 0 0",
            },
        )
    return xml_document_bytes(config)


def bambu_slice_info_bytes(source_application: str) -> bytes:
    version = str(source_application).removeprefix("BambuStudio-")
    config = ET.Element("config")
    header = ET.SubElement(config, "header")
    ET.SubElement(header, "header_item", {"key": "X-BBL-Client-Type", "value": "slicer"})
    ET.SubElement(header, "header_item", {"key": "X-BBL-Client-Version", "value": version})
    return xml_document_bytes(config)


def export_colored_parts_3mf(
    path: Path, parts: list[dict], title: str,
    source_application: str | None = None,
    source_filament_colors: list[str] | None = None,
    source_project_settings: dict | None = None,
    output_layout: str = "assembly",
) -> dict:
    from .package_stream import MeshPayloadStore
    with MeshPayloadStore() as payloads:
        return _export_colored_parts_3mf(
            path, parts, title, source_application, source_filament_colors,
            source_project_settings, output_layout, _payloads=payloads)


def _export_colored_parts_3mf(
    path: Path,
    parts: list[dict],
    title: str,
    source_application: str | None = None,
    source_filament_colors: list[str] | None = None,
    source_project_settings: dict | None = None,
    output_layout: str = "assembly",
    *, _payloads,
) -> dict:
    """Write printable meshes as one assembly or legacy parallel build items."""
    if output_layout not in {"assembly", "separate-items"}:
        raise ValueError(f"unknown output layout: {output_layout}")
    ET.register_namespace("", CORE_URI)
    ET.register_namespace("m", MATERIAL_URI)
    ordered_colors, part_color_indices = output_color_layout(parts, source_filament_colors)
    face_color_indices = output_face_color_layout(parts, ordered_colors)

    model = ET.Element(
        CORE_NS + "model",
        {
            "unit": "millimeter",
            "{http://www.w3.org/XML/1998/namespace}lang": "en-US",
            "requiredextensions": "m",
        },
    )
    title_metadata = ET.SubElement(model, CORE_NS + "metadata", {"name": "Title"})
    title_metadata.text = title
    application_metadata = ET.SubElement(model, CORE_NS + "metadata", {"name": "Application"})
    project_settings_payload = bambu_project_settings_payload(source_project_settings, ordered_colors)
    bambu_project_compatible = bool(
        source_application
        and str(source_application).startswith("BambuStudio-")
        and project_settings_payload
    )
    application_metadata.text = str(source_application) if bambu_project_compatible else f"automated-3d-model-splitting {VERSION}"
    generator_metadata = ET.SubElement(model, CORE_NS + "metadata", {"name": "automated-3d-model-splitting:Generator"})
    generator_metadata.text = f"automated-3d-model-splitting {VERSION}"
    if bambu_project_compatible:
        version_metadata = ET.SubElement(model, CORE_NS + "metadata", {"name": "BambuStudio:3mfVersion"})
        version_metadata.text = "1"
    resources = ET.SubElement(model, CORE_NS + "resources")

    color_group_id = 1
    color_group = ET.SubElement(resources, MATERIAL_NS + "colorgroup", {"id": str(color_group_id)})
    for rgba in ordered_colors:
        ET.SubElement(color_group, MATERIAL_NS + "color", {"color": rgba})

    build_records = []
    for part_offset, part in enumerate(parts):
        offset = part_offset + 2
        mesh, serialization_record = mesh_ready_for_3mf_serialization(
            part["mesh"]
        )
        part.setdefault("annotation", {})[
            "serialization_grid_orientation"
        ] = serialization_record
        rgba = normalize_3mf_color(part["color_hex"])
        part_color_index = int(part_color_indices[part_offset])
        object_element = ET.SubElement(
            resources,
            CORE_NS + "object",
            {
                "id": str(offset),
                "type": "model",
                "name": str(part["part_id"]),
                "pid": str(color_group_id),
                "pindex": str(part_color_index),
            },
        )
        annotation = dict(part.get("annotation", {}))
        annotation.update(
            {
                "part_id": str(part["part_id"]),
                "color_code": str(part.get("color_code", "")),
                "color_name": str(part.get("color_name", "")),
                "color_hex": rgba,
                "filament_slot_index": part.get("filament_slot_index"),
            }
        )
        object_metadata = ET.SubElement(
            object_element,
            CORE_NS + "metadata",
            {"name": "automated-3d-model-splitting:annotation"},
        )
        object_metadata.text = json.dumps(annotation, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        per_face_indices = face_color_indices[part_offset]
        raw_paint_tokens = part.get("face_paint_color_tokens")
        if raw_paint_tokens is None:
            raw_paint_tokens = part.get("face_color_codes")
        paint_tokens = None
        if raw_paint_tokens is not None:
            raw_paint_tokens = list(raw_paint_tokens)
            if len(raw_paint_tokens) != len(mesh.faces):
                raise ValueError(
                    f"{part['part_id']} has {len(mesh.faces)} faces but "
                    f"{len(raw_paint_tokens)} Bambu paint tokens"
                )
            present_paint_tokens = [
                value for value in raw_paint_tokens if value not in (None, "")
            ]
            if present_paint_tokens and len(present_paint_tokens) != len(
                raw_paint_tokens
            ):
                raise ValueError(
                    f"{part['part_id']} has a partial Bambu paint-token payload"
                )
            if present_paint_tokens:
                paint_tokens = [str(value) for value in raw_paint_tokens]
        payload_index = _payloads.add_mesh(
            mesh, per_face_indices, color_group_id, paint_tokens, format_3mf_float)
        ET.SubElement(object_element, "_split3mf_payload", {"index": str(payload_index)})
        build_records.append(
            {
                "object_id": offset,
                "part_id": str(part["part_id"]),
                "color_hex": rgba,
                "color_index": part_color_index,
                "filament_slot_index": part.get("filament_slot_index"),
                "vertices": int(len(mesh.vertices)),
                "triangles": int(len(mesh.faces)),
                "annotation": annotation,
                "face_paint_color_tokens": paint_tokens,
            }
        )

    assembly_id = None
    if output_layout == "assembly":
        assembly_id = max(int(record["object_id"]) for record in build_records) + 1
        assembly_object = ET.SubElement(
            resources,
            CORE_NS + "object",
            {
                "id": str(assembly_id),
                "type": "model",
                "name": str(title),
            },
        )
        assembly_metadata = ET.SubElement(
            assembly_object,
            CORE_NS + "metadata",
            {"name": "automated-3d-model-splitting:assembly"},
        )
        assembly_metadata.text = json.dumps(
            {
                "component_count": len(build_records),
                "layout": "assembly",
                "generator_version": VERSION,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        components_element = ET.SubElement(assembly_object, CORE_NS + "components")
        for record in build_records:
            ET.SubElement(
                components_element,
                CORE_NS + "component",
                {"objectid": str(record["object_id"])},
            )

    build = ET.SubElement(model, CORE_NS + "build")
    if output_layout == "assembly":
        ET.SubElement(build, CORE_NS + "item", {"objectid": str(assembly_id)})
    else:
        for record in build_records:
            ET.SubElement(build, CORE_NS + "item", {"objectid": str(record["object_id"])})

    content_types = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>
"""
    relationships = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        write_deterministic_zip_member(archive, "[Content_Types].xml", content_types)
        write_deterministic_zip_member(archive, "_rels/.rels", relationships)
        _payloads.write_model(archive, xml_document_bytes(model))
        if bambu_project_compatible and project_settings_payload is not None:
            write_deterministic_zip_member(archive, "Metadata/project_settings.config", project_settings_payload)
            write_deterministic_zip_member(
                archive,
                "Metadata/model_settings.config",
                bambu_model_settings_bytes(
                    build_records,
                    output_layout=output_layout,
                    assembly_id=assembly_id,
                    assembly_name=title,
                ),
            )
            write_deterministic_zip_member(
                archive,
                "Metadata/slice_info.config",
                bambu_slice_info_bytes(str(source_application)),
            )
    return {
        "part_count": len(parts),
        "colors": ordered_colors,
        "objects": build_records,
        "application": application_metadata.text,
        "bambu_project_compatible": bambu_project_compatible,
        "bambu_project_metadata_entries": (
            [
                "Metadata/project_settings.config",
                "Metadata/model_settings.config",
                "Metadata/slice_info.config",
            ]
            if bambu_project_compatible
            else []
        ),
        "source_filament_colors": [normalize_3mf_color(value) for value in (source_filament_colors or [])],
        "source_filament_palette_preserved": bool(source_filament_colors),
        "output_layout": output_layout,
        "assembly_id": assembly_id,
        "build_item_count": 1 if output_layout == "assembly" else len(build_records),
    }


def validate_colored_parts_3mf(
    path: Path,
    expected_parts: list[dict],
    source_filament_colors: list[str] | None = None,
    source_application: str | None = None,
    source_project_settings: dict | None = None,
    output_layout: str = "assembly",
    include_loaded_objects: bool = False,
) -> dict:
    errors = []
    object_checks = []
    loaded_objects = []
    expect_bambu_project = bool(
        source_application
        and str(source_application).startswith("BambuStudio-")
        and bambu_project_settings_payload(source_project_settings, source_filament_colors)
    )
    required_entries = {"[Content_Types].xml", "_rels/.rels", "3D/3dmodel.model"}
    if expect_bambu_project:
        required_entries.update(
            {
                "Metadata/project_settings.config",
                "Metadata/model_settings.config",
                "Metadata/slice_info.config",
            }
        )
    project_config = None
    model_settings_root = None
    slice_info_root = None
    try:
        with zipfile.ZipFile(path) as archive:
            missing = sorted(required_entries - set(archive.namelist()))
            if missing:
                errors.append("missing package entries: " + ", ".join(missing))
            root = ET.fromstring(archive.read("3D/3dmodel.model"))
            if expect_bambu_project and not missing:
                project_config = json.loads(archive.read("Metadata/project_settings.config"))
                model_settings_root = ET.fromstring(archive.read("Metadata/model_settings.config"))
                slice_info_root = ET.fromstring(archive.read("Metadata/slice_info.config"))
    except (OSError, zipfile.BadZipFile, KeyError, ET.ParseError, json.JSONDecodeError) as exc:
        return {"valid": False, "errors": [str(exc)], "objects": [], "colors": []}

    resources = root.find(CORE_NS + "resources")
    build = root.find(CORE_NS + "build")
    if resources is None or build is None:
        return {"valid": False, "errors": errors + ["model is missing resources or build"], "objects": [], "colors": []}

    color_groups = resources.findall(MATERIAL_NS + "colorgroup")
    if len(color_groups) != 1:
        errors.append(f"expected one color group, found {len(color_groups)}")
        colors = []
        color_group_id = None
    else:
        color_group_id = color_groups[0].attrib.get("id")
        colors = [normalize_3mf_color(element.attrib.get("color", "")) for element in color_groups[0].findall(MATERIAL_NS + "color")]

    try:
        expected_colors, expected_color_indices = output_color_layout(expected_parts, source_filament_colors)
        expected_face_color_indices = output_face_color_layout(
            expected_parts, expected_colors
        )
    except ValueError as exc:
        errors.append(str(exc))
        expected_colors, expected_color_indices, expected_face_color_indices = [], [], []
    if colors != expected_colors:
        errors.append(f"color palette mismatch: expected {expected_colors}, found {colors}")

    metadata_values = {
        element.attrib.get("name"): element.text or ""
        for element in root.findall(CORE_NS + "metadata")
    }
    if expect_bambu_project:
        if metadata_values.get("Application") != str(source_application):
            errors.append("Bambu project application marker mismatch")
        if metadata_values.get("BambuStudio:3mfVersion") != "1":
            errors.append("missing BambuStudio:3mfVersion metadata")
        project_colors = project_config.get("filament_colour", []) if isinstance(project_config, dict) else []
        if [normalize_3mf_color(value) for value in project_colors] != expected_colors:
            errors.append("Bambu project filament_colour does not match the preserved source palette")
        if not project_colors:
            errors.append("Bambu project settings contain no filament_colour")
        if model_settings_root is None:
            errors.append("missing parsed Bambu model settings")
        if slice_info_root is None:
            errors.append("missing parsed Bambu slice info")

    objects = resources.findall(CORE_NS + "object")
    items = build.findall(CORE_NS + "item")
    mesh_objects = [element for element in objects if element.find(CORE_NS + "mesh") is not None]
    assembly_objects = [
        element for element in objects if element.find(CORE_NS + "components") is not None
    ]
    expected_object_count = len(expected_parts) + (1 if output_layout == "assembly" else 0)
    expected_build_count = 1 if output_layout == "assembly" else len(expected_parts)
    if len(objects) != expected_object_count:
        errors.append(f"expected {expected_object_count} objects, found {len(objects)}")
    if len(mesh_objects) != len(expected_parts):
        errors.append(f"expected {len(expected_parts)} mesh objects, found {len(mesh_objects)}")
    if len(items) != expected_build_count:
        errors.append(f"expected {expected_build_count} build items, found {len(items)}")
    built_ids = [item.attrib.get("objectid") for item in items]
    component_ids: list[str] = []
    assembly_id = None
    if output_layout == "assembly":
        if len(assembly_objects) != 1:
            errors.append(f"expected one assembly object, found {len(assembly_objects)}")
        else:
            assembly_id = assembly_objects[0].attrib.get("id")
            component_ids = [
                element.attrib.get("objectid", "")
                for element in assembly_objects[0].findall(
                    CORE_NS + "components/" + CORE_NS + "component"
                )
            ]
            expected_component_ids = [element.attrib.get("id", "") for element in mesh_objects]
            if component_ids != expected_component_ids:
                errors.append(
                    f"assembly component mismatch: expected {expected_component_ids}, found {component_ids}"
                )
            if built_ids != [assembly_id]:
                errors.append(f"build does not reference only the assembly object: {built_ids}")
    elif assembly_objects:
        errors.append("separate-items layout unexpectedly contains an assembly object")
    model_config_objects = (
        {element.attrib.get("id"): element for element in model_settings_root.findall("object")}
        if model_settings_root is not None
        else {}
    )

    for index, (object_element, expected) in enumerate(zip(mesh_objects, expected_parts)):
        object_id = object_element.attrib.get("id")
        part_errors = []
        metadata_element = object_element.find(
            CORE_NS
            + "metadata[@name='automated-3d-model-splitting:annotation']"
        )
        annotation = {}
        if metadata_element is not None and metadata_element.text:
            try:
                annotation = json.loads(metadata_element.text)
            except json.JSONDecodeError:
                part_errors.append("invalid object annotation JSON")
        referenced_ids = component_ids if output_layout == "assembly" else built_ids
        if object_id not in referenced_ids:
            part_errors.append("object is not referenced by the selected output layout")
        if object_element.attrib.get("name") != str(expected["part_id"]):
            part_errors.append("part name mismatch")
        if color_group_id is None or object_element.attrib.get("pid") != color_group_id:
            part_errors.append("color group reference mismatch")
        try:
            part_color_index = int(object_element.attrib.get("pindex", "-1"))
            if not 0 <= part_color_index < len(colors):
                raise IndexError(part_color_index)
            part_color = colors[part_color_index]
        except (ValueError, IndexError):
            part_color_index = None
            part_color = None
            part_errors.append("invalid color index")
        expected_color_index = expected_color_indices[index] if index < len(expected_color_indices) else None
        if part_color_index != expected_color_index:
            part_errors.append(
                f"filament color index mismatch: expected {expected_color_index}, found {part_color_index}"
            )
        expected_color = normalize_3mf_color(expected["color_hex"])
        if part_color != expected_color:
            part_errors.append(f"color mismatch: expected {expected_color}, found {part_color}")
        if expect_bambu_project:
            config_object = (
                model_config_objects.get(assembly_id)
                if output_layout == "assembly"
                else model_config_objects.get(object_id)
            )
            if config_object is None:
                part_errors.append("missing object in Bambu model settings")
            else:
                if output_layout == "assembly":
                    config_part = config_object.find(f"part[@id='{object_id}']")
                    if config_part is None:
                        part_errors.append("missing part in grouped Bambu model settings")
                        config_metadata = {}
                    else:
                        config_metadata = {
                            element.attrib.get("key"): element.attrib.get("value")
                            for element in config_part.findall("metadata")
                            if "key" in element.attrib
                        }
                else:
                    config_metadata = {
                        element.attrib.get("key"): element.attrib.get("value")
                        for element in config_object.findall("metadata")
                        if "key" in element.attrib
                    }
                expected_slot = expected.get("filament_slot_index")
                expected_extruder = int(expected_slot) + 1 if expected_slot is not None else int(expected_color_index) + 1
                if config_metadata.get("extruder") != str(expected_extruder):
                    part_errors.append(
                        f"Bambu extruder mismatch: expected {expected_extruder}, found {config_metadata.get('extruder')}"
                    )

        mesh_element = object_element.find(CORE_NS + "mesh")
        if mesh_element is None:
            part_errors.append("missing mesh")
            vertices = np.empty((0, 3), dtype=np.float64)
            faces = np.empty((0, 3), dtype=np.int64)
            loaded_face_indices = []
        else:
            vertices_parent = mesh_element.find(CORE_NS + "vertices")
            triangles_parent = mesh_element.find(CORE_NS + "triangles")
            triangle_elements = (
                list(triangles_parent) if triangles_parent is not None else []
            )
            vertices = np.array(
                [
                    [float(vertex.attrib[axis]) for axis in ("x", "y", "z")]
                    for vertex in (list(vertices_parent) if vertices_parent is not None else [])
                ],
                dtype=np.float64,
            )
            faces = np.array(
                [
                    [int(triangle.attrib[key]) for key in ("v1", "v2", "v3")]
                    for triangle in triangle_elements
                ],
                dtype=np.int64,
            )
            loaded_face_indices = []
            for triangle in triangle_elements:
                triangle_pid = triangle.attrib.get(
                    "pid",
                    object_element.attrib.get("pid"),
                )
                property_values = [
                    triangle.attrib.get("p1"),
                    triangle.attrib.get("p2"),
                    triangle.attrib.get("p3"),
                ]
                present = [value for value in property_values if value is not None]
                if (
                    triangle_pid is not None
                    and triangle_pid != color_group_id
                    or present
                    and len(set(present)) != 1
                ):
                    loaded_face_indices.append(None)
                    continue
                try:
                    loaded_face_indices.append(
                        int(present[0]) if present else int(part_color_index)
                    )
                except (TypeError, ValueError):
                    loaded_face_indices.append(None)
            expected_face_indices = (
                expected_face_color_indices[index]
                if index < len(expected_face_color_indices)
                else None
            )
            if expected_face_indices is not None:
                actual_face_indices = []
                for triangle in triangle_elements:
                    properties = [
                        triangle.attrib.get("p1"),
                        triangle.attrib.get("p2"),
                        triangle.attrib.get("p3"),
                    ]
                    present = [value for value in properties if value is not None]
                    if (
                        triangle.attrib.get("pid") != color_group_id
                        or len(present) != 3
                        or len(set(present)) != 1
                    ):
                        actual_face_indices.append(None)
                    else:
                        actual_face_indices.append(int(present[0]))
                if actual_face_indices != expected_face_indices:
                    part_errors.append("per-triangle color meaning mismatch")
            expected_paint_tokens = expected.get("face_paint_color_tokens")
            if expected_paint_tokens is None:
                expected_paint_tokens = expected.get("face_color_codes")
            if expected_paint_tokens is not None:
                expected_paint_tokens = [
                    str(value) for value in expected_paint_tokens
                ]
                actual_paint_tokens = [
                    triangle.attrib.get("paint_color")
                    for triangle in triangle_elements
                ]
                if actual_paint_tokens != expected_paint_tokens:
                    part_errors.append("Bambu per-triangle paint_color mismatch")
        if len(vertices) != len(expected["mesh"].vertices) or len(faces) != len(expected["mesh"].faces):
            part_errors.append("serialized mesh count mismatch")
        if len(vertices) and len(faces):
            reloaded_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            valid_loaded_face_indices = bool(
                len(loaded_face_indices) == len(faces)
                and all(
                    color_index is not None
                    and 0 <= int(color_index) < len(colors)
                    for color_index in loaded_face_indices
                )
            )
            if valid_loaded_face_indices:
                palette_rgba = np.asarray(
                    [rgba_bytes_from_3mf_color(color) for color in colors],
                    dtype=np.uint8,
                )
                reloaded_mesh.visual.face_colors = palette_rgba[
                    np.asarray(loaded_face_indices, dtype=np.int64)
                ]
            reloaded_open_edges = open_edge_count(reloaded_mesh)
            if not reloaded_mesh.is_watertight:
                part_errors.append("reloaded mesh is not watertight")
            if not reloaded_mesh.is_winding_consistent:
                part_errors.append("reloaded mesh winding is inconsistent")
            component_orientation = watertight_component_orientation_audit(
                reloaded_mesh
            )
            if component_orientation.get("inward_closed_component_count"):
                part_errors.append(
                    "reloaded mesh has inward-oriented closed components: "
                    f"{component_orientation['inward_closed_component_count']}"
                )
            if reloaded_open_edges:
                part_errors.append(f"reloaded mesh has {reloaded_open_edges} open edges")
            if include_loaded_objects:
                loaded_objects.append(
                    {
                        "part_id": object_element.attrib.get("name", ""),
                        "mesh": reloaded_mesh,
                        "annotation": annotation,
                        "palette": colors,
                        "object_color_index": part_color_index,
                        "face_color_indices": (
                            [int(value) for value in loaded_face_indices]
                            if valid_loaded_face_indices
                            else []
                        ),
                        "face_color_hexes": (
                            [colors[int(value)] for value in loaded_face_indices]
                            if valid_loaded_face_indices
                            else []
                        ),
                        "face_filament_slot_indices": (
                            [int(value) for value in loaded_face_indices]
                            if valid_loaded_face_indices
                            else []
                        ),
                        "face_paint_color_tokens": [
                            triangle.attrib.get("paint_color")
                            for triangle in triangle_elements
                        ],
                    }
                )
        object_checks.append(
            {
                "index": index,
                "part_id": str(expected["part_id"]),
                "object_id": object_id,
                "color_hex": part_color,
                "color_index": part_color_index,
                "filament_slot_index": expected.get("filament_slot_index"),
                "vertices": int(len(vertices)),
                "triangles": int(len(faces)),
                "valid": not part_errors,
                "errors": part_errors,
            }
        )
        errors.extend(f"{expected['part_id']}: {message}" for message in part_errors)

    result = {
        "valid": not errors,
        "errors": errors,
        "objects": object_checks,
        "colors": colors,
        "bambu_project_metadata_valid": bool(expect_bambu_project and not errors),
        "output_layout": output_layout,
        "assembly_id": assembly_id,
        "build_item_count": len(items),
    }
    if include_loaded_objects:
        result["_loaded_objects"] = loaded_objects
    return result


def default_output_3mf(input_path: Path, explicit_output: str | None) -> Path:
    output = (
        Path(explicit_output).expanduser()
        if explicit_output
        else input_path.expanduser().resolve().with_name(f"{sanitize_name(input_path.stem) or 'model'}_split_parts.3mf")
    )
    if output.suffix.lower() != ".3mf":
        raise ValueError("--output must end with .3mf")
    return output


def prepare_output_3mf(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise ValueError(f"Output already exists: {path}. Pass --overwrite explicitly to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)


def prepare_debug_directory(path: Path, overwrite: bool = False) -> Path:
    """Atomically reserve a fresh run; final-output overwrite never erases logs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for sequence in range(1000000):
        candidate = path if sequence == 0 else path.with_name(f"{path.name}_run{sequence:04d}")
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise OSError(f"Cannot reserve a unique debug directory beside {path}")




class ThreeMFWriter:
    """Write the deterministic multi-object 3MF package."""

    write = staticmethod(export_colored_parts_3mf)
