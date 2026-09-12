#!/usr/bin/env python3
"""Regression harness for vendor sliver preservation during mesh finalization."""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from split3mf import common

common.load_core_dependencies()

from split3mf.common import trimesh
from split3mf.mesh_finalization import finalize_source_preserving_mesh
from split3mf.validation import finalize_mesh, validate_mesh_in_memory


def face_key(vertices: np.ndarray, face: np.ndarray) -> tuple:
    points = np.round(vertices[np.asarray(face, dtype=np.int64)], decimals=10)
    return tuple(sorted(tuple(float(value) for value in point) for point in points))


def source_inventory(mesh: trimesh.Trimesh) -> collections.Counter:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    return collections.Counter(
        face_key(vertices, face)
        for face in np.asarray(mesh.faces, dtype=np.int64)
    )


def missing_inventory(
    expected: collections.Counter,
    mesh: trimesh.Trimesh,
) -> int:
    actual = source_inventory(mesh)
    return int(
        sum(max(int(count) - int(actual.get(key, 0)), 0) for key, count in expected.items())
    )


def vendor_sliver_fixture(count: int = 7) -> trimesh.Trimesh:
    all_vertices: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    for index in range(int(count)):
        base = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, (index + 1) * 1e-11, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        base[:, 0] += float(index) * 2.0
        vertex_offset = len(all_vertices) * 4
        all_vertices.append(base)
        all_faces.append(
            np.asarray(
                [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
                dtype=np.int64,
            )
            + vertex_offset
        )
    return trimesh.Trimesh(
        vertices=np.vstack(all_vertices),
        faces=np.vstack(all_faces),
        process=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slivers", type=int, default=7)
    args = parser.parse_args()
    fixture = vendor_sliver_fixture(max(int(args.slivers), 1))
    expected = source_inventory(fixture)

    legacy = finalize_mesh(fixture.copy())
    protected = finalize_source_preserving_mesh(
        fixture.copy(),
        protected_source_face_count=len(fixture.faces),
    )
    record = {
        "case": "vendor_sliver_source_face_prefix",
        "source_faces": int(len(fixture.faces)),
        "legacy_source_faces_missing": missing_inventory(expected, legacy),
        "protected_source_faces_missing": missing_inventory(expected, protected),
        "legacy_topology": validate_mesh_in_memory(legacy),
        "protected_topology": validate_mesh_in_memory(protected),
        "protected_strategy": protected.metadata[
            "source_preserving_finalization"
        ]["strategy"],
    }
    print(json.dumps(record, ensure_ascii=False, sort_keys=True))
    if record["protected_source_faces_missing"] != 0:
        return 1
    if record["protected_topology"]["open_edges"] != 0:
        return 1
    if record["protected_topology"]["over_shared_edges"] != 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
