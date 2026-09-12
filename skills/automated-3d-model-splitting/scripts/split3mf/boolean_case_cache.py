"""Reusable on-disk cases for expensive recursive Boolean debugging.

The production pipeline can optionally snapshot the already-built parent and
ordered cutter solids immediately before subtraction.  The cache is never a
deliverable and is enabled only through ``SPLIT3MF_BOOLEAN_CASE_CACHE``; it
lets the Boolean policy and proxy geometry be exercised in seconds without
repeating several minutes of dense backing triangulation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh


def write_boolean_case_cache(
    path: Path,
    parent: trimesh.Trimesh,
    cutters: list[trimesh.Trimesh],
    *,
    labels: list[str] | None = None,
    metadata: dict | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    cutter_labels = list(labels or [f"cutter_{index}" for index in range(len(cutters))])
    if len(cutter_labels) != len(cutters):
        raise ValueError("Boolean cache labels must match the cutter count")
    arrays: dict[str, np.ndarray] = {
        "parent_vertices": np.asarray(parent.vertices, dtype=np.float64),
        "parent_faces": np.asarray(parent.faces, dtype=np.int64),
        "metadata_json": np.asarray(
            json.dumps(
                {
                    **(metadata or {}),
                    "labels": cutter_labels,
                    "cutter_count": len(cutters),
                },
                ensure_ascii=False,
            )
        ),
    }
    for index, cutter in enumerate(cutters):
        arrays[f"cutter_{index:03d}_vertices"] = np.asarray(
            cutter.vertices,
            dtype=np.float64,
        )
        arrays[f"cutter_{index:03d}_faces"] = np.asarray(
            cutter.faces,
            dtype=np.int64,
        )
    np.savez_compressed(target, **arrays)
    return target


def load_boolean_case_cache(
    path: Path,
) -> tuple[trimesh.Trimesh, list[trimesh.Trimesh], dict]:
    with np.load(Path(path), allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        parent = trimesh.Trimesh(
            vertices=np.asarray(payload["parent_vertices"], dtype=np.float64),
            faces=np.asarray(payload["parent_faces"], dtype=np.int64),
            process=False,
        )
        cutters = [
            trimesh.Trimesh(
                vertices=np.asarray(
                    payload[f"cutter_{index:03d}_vertices"],
                    dtype=np.float64,
                ),
                faces=np.asarray(
                    payload[f"cutter_{index:03d}_faces"],
                    dtype=np.int64,
                ),
                process=False,
            )
            for index in range(int(metadata["cutter_count"]))
        ]
    return parent, cutters, metadata
