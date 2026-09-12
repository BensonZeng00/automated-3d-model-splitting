from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np


CORE_NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CONTENT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"


def build_mesh(grid_size: int = 61) -> tuple[np.ndarray, np.ndarray, list[str]]:
    coordinates = np.linspace(-15.0, 15.0, grid_size)
    layer_vertex_count = grid_size * grid_size

    vertices: list[list[float]] = []
    for z in (0.0, 10.0):
        for y in coordinates:
            for x in coordinates:
                vertices.append([float(x), float(y), z])

    def bottom(i: int, j: int) -> int:
        return j * grid_size + i

    def top(i: int, j: int) -> int:
        return layer_vertex_count + bottom(i, j)

    faces: list[list[int]] = []
    paint: list[str] = []

    # Bottom faces point down.
    for j in range(grid_size - 1):
        for i in range(grid_size - 1):
            a, b = bottom(i, j), bottom(i + 1, j)
            c, d = bottom(i + 1, j + 1), bottom(i, j + 1)
            faces.extend(([a, c, b], [a, d, c]))
            paint.extend(("4", "4"))

    # Top faces point up. The centered 12 x 12 mm patch is orange.
    for j in range(grid_size - 1):
        for i in range(grid_size - 1):
            a, b = top(i, j), top(i + 1, j)
            c, d = top(i + 1, j + 1), top(i, j + 1)
            faces.extend(([a, b, c], [a, c, d]))
            center_x = 0.5 * (coordinates[i] + coordinates[i + 1])
            center_y = 0.5 * (coordinates[j] + coordinates[j + 1])
            token = "8" if abs(center_x) < 6.0 and abs(center_y) < 6.0 else "4"
            paint.extend((token, token))

    # Four side walls, sharing the exact top and bottom perimeter vertices.
    last = grid_size - 1
    for i in range(last):
        faces.extend(
            (
                [bottom(i, 0), bottom(i + 1, 0), top(i + 1, 0)],
                [bottom(i, 0), top(i + 1, 0), top(i, 0)],
                [bottom(i + 1, last), bottom(i, last), top(i, last)],
                [bottom(i + 1, last), top(i, last), top(i + 1, last)],
            )
        )
        paint.extend(("4", "4", "4", "4"))
    for j in range(last):
        faces.extend(
            (
                [bottom(0, j + 1), bottom(0, j), top(0, j)],
                [bottom(0, j + 1), top(0, j), top(0, j + 1)],
                [bottom(last, j), bottom(last, j + 1), top(last, j + 1)],
                [bottom(last, j), top(last, j + 1), top(last, j)],
            )
        )
        paint.extend(("4", "4", "4", "4"))

    return (
        np.asarray(vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int64),
        paint,
    )


def model_xml(vertices: np.ndarray, faces: np.ndarray, paint: list[str]) -> bytes:
    ET.register_namespace("", CORE_NS)
    model = ET.Element(f"{{{CORE_NS}}}model", {"unit": "millimeter"})
    resources = ET.SubElement(model, f"{{{CORE_NS}}}resources")
    obj = ET.SubElement(
        resources,
        f"{{{CORE_NS}}}object",
        {"id": "1", "type": "model", "name": "painted_boolean_test"},
    )
    mesh = ET.SubElement(obj, f"{{{CORE_NS}}}mesh")
    vertices_xml = ET.SubElement(mesh, f"{{{CORE_NS}}}vertices")
    for x, y, z in vertices:
        ET.SubElement(
            vertices_xml,
            f"{{{CORE_NS}}}vertex",
            {"x": f"{x:.6f}", "y": f"{y:.6f}", "z": f"{z:.6f}"},
        )
    triangles_xml = ET.SubElement(mesh, f"{{{CORE_NS}}}triangles")
    for face, token in zip(faces, paint, strict=True):
        ET.SubElement(
            triangles_xml,
            f"{{{CORE_NS}}}triangle",
            {
                "v1": str(int(face[0])),
                "v2": str(int(face[1])),
                "v3": str(int(face[2])),
                "paint_color": token,
            },
        )
    build = ET.SubElement(model, f"{{{CORE_NS}}}build")
    ET.SubElement(build, f"{{{CORE_NS}}}item", {"objectid": "1"})
    return ET.tostring(model, encoding="utf-8", xml_declaration=True)


def content_types_xml() -> bytes:
    ET.register_namespace("", CONTENT_NS)
    types = ET.Element(f"{{{CONTENT_NS}}}Types")
    ET.SubElement(
        types,
        f"{{{CONTENT_NS}}}Default",
        {"Extension": "rels", "ContentType": "application/vnd.openxmlformats-package.relationships+xml"},
    )
    ET.SubElement(
        types,
        f"{{{CONTENT_NS}}}Default",
        {"Extension": "model", "ContentType": "application/vnd.ms-package.3dmanufacturing-3dmodel+xml"},
    )
    return ET.tostring(types, encoding="utf-8", xml_declaration=True)


def relationships_xml() -> bytes:
    ET.register_namespace("", REL_NS)
    relationships = ET.Element(f"{{{REL_NS}}}Relationships")
    ET.SubElement(
        relationships,
        f"{{{REL_NS}}}Relationship",
        {
            "Target": "/3D/3dmodel.model",
            "Id": "rel-1",
            "Type": "http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel",
        },
    )
    return ET.tostring(relationships, encoding="utf-8", xml_declaration=True)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: build_painted_boolean_test_source.py OUTPUT.3mf")
    output = Path(sys.argv[1]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    vertices, faces, paint = build_mesh()
    project_settings = {
        "filament_colour": ["#EDC148", "#E86825"],
        "filament_type": ["PLA", "PLA"],
        "filament_vendor": ["Generic", "Generic"],
        "filament_settings_id": ["Generic PLA", "Generic PLA"],
    }
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types_xml())
        archive.writestr("_rels/.rels", relationships_xml())
        archive.writestr("3D/3dmodel.model", model_xml(vertices, faces, paint))
        archive.writestr(
            "Metadata/project_settings.config",
            json.dumps(project_settings, ensure_ascii=False, separators=(",", ":")),
        )
    print(
        json.dumps(
            {
                "output": str(output),
                "vertices": int(len(vertices)),
                "faces": int(len(faces)),
                "yellow_faces": int(paint.count("4")),
                "orange_faces": int(paint.count("8")),
                "bbox_mm": [30.0, 30.0, 10.0],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
