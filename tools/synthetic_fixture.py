"""Create a deterministic, license-safe painted 3MF fixture for release tests."""

from __future__ import annotations

import json
from pathlib import Path
import zipfile


CORE_NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"


def _cube_mesh(subdivisions: int = 8) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int, str]]]:
    """Return a closed, outward-oriented cube with a white top and black sides/bottom."""

    vertices: list[tuple[float, float, float]] = []
    vertex_ids: dict[tuple[float, float, float], int] = {}
    triangles: list[tuple[int, int, int, str]] = []
    size = 10.0

    def vertex(point: tuple[float, float, float]) -> int:
        key = tuple(round(value, 9) for value in point)
        if key not in vertex_ids:
            vertex_ids[key] = len(vertices)
            vertices.append(key)
        return vertex_ids[key]

    def add_face(
        origin: tuple[float, float, float],
        axis_u: tuple[float, float, float],
        axis_v: tuple[float, float, float],
        paint_token: str,
    ) -> None:
        for row in range(subdivisions):
            for column in range(subdivisions):
                u0, u1 = column / subdivisions, (column + 1) / subdivisions
                v0, v1 = row / subdivisions, (row + 1) / subdivisions

                def point(u: float, v: float) -> tuple[float, float, float]:
                    return tuple(
                        origin[index]
                        + size * (u * axis_u[index] + v * axis_v[index])
                        for index in range(3)
                    )

                a = vertex(point(u0, v0))
                b = vertex(point(u1, v0))
                c = vertex(point(u1, v1))
                d = vertex(point(u0, v1))
                triangles.append((a, b, c, paint_token))
                triangles.append((a, c, d, paint_token))

    # axis_u x axis_v points outward for every face.
    add_face((0.0, 10.0, 0.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0), "8")
    add_face((0.0, 0.0, 10.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), "4")
    add_face((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), "8")
    add_face((10.0, 10.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), "8")
    add_face((10.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), "8")
    add_face((0.0, 10.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0), "8")
    return vertices, triangles


def _model_xml() -> bytes:
    vertices, triangles = _cube_mesh()
    vertex_xml = "\n".join(
        f'          <vertex x="{x:g}" y="{y:g}" z="{z:g}" />'
        for x, y, z in vertices
    )
    triangle_xml = "\n".join(
        f'          <triangle v1="{v1}" v2="{v2}" v3="{v3}" paint_color="{paint}" />'
        for v1, v2, v3, paint in triangles
    )
    document = f'''<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xml:lang="en-US" xmlns="{CORE_NS}">
  <metadata name="Application">Synthetic painted 3MF release fixture</metadata>
  <resources>
    <object id="1" type="model" name="synthetic-two-colour-cube">
      <mesh>
        <vertices>
{vertex_xml}
        </vertices>
        <triangles>
{triangle_xml}
        </triangles>
      </mesh>
    </object>
  </resources>
  <build>
    <item objectid="1" />
  </build>
</model>
'''
    return document.encode("utf-8")


def _write_member(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, payload)


def write_fixture(path: Path) -> Path:
    """Write a deterministic two-colour vendor-painted 3MF and return its path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    content_types = b'''<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml" />
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml" />
  <Default Extension="config" ContentType="application/json" />
</Types>
'''
    relationships = b'''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel" />
</Relationships>
'''
    project_settings = json.dumps(
        {"filament_colour": ["#FFFFFFFF", "#000000FF"]},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    with zipfile.ZipFile(path, "w") as archive:
        _write_member(archive, "[Content_Types].xml", content_types)
        _write_member(archive, "_rels/.rels", relationships)
        _write_member(archive, "3D/3dmodel.model", _model_xml())
        _write_member(archive, "Metadata/project_settings.config", project_settings)
    return path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    print(write_fixture(arguments.output))
