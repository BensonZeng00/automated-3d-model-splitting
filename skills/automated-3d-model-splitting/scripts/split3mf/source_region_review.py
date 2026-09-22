from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

from .common import *
from .recognition import classify_review_groups_semantically


REVIEW_SCHEMA_VERSION = 2
REVIEW_VIEW_DIRECTIONS = (
    (1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, -1.0),
)


def default_region_review_dir(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_region_review")


def _view_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direction = np.asarray(direction, dtype=np.float64)
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    reference = np.asarray(
        [0.0, 0.0, 1.0] if abs(float(direction[2])) < 0.9 else [0.0, 1.0, 0.0],
        dtype=np.float64,
    )
    u = np.cross(reference, direction)
    u /= max(float(np.linalg.norm(u)), 1e-12)
    v = np.cross(direction, u)
    v /= max(float(np.linalg.norm(v)), 1e-12)
    return u, v


def _write_rgb_png(path: Path, image: np.ndarray) -> None:
    image = np.asarray(image, dtype=np.uint8)
    height, width, channels = image.shape
    if channels != 3:
        raise ValueError("PNG review image must be RGB")
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    payload = b"\x89PNG\r\n\x1a\n"
    payload += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += chunk(b"IDAT", zlib.compress(raw, level=6))
    payload += chunk(b"IEND", b"")
    path.write_bytes(payload)


def _splat(image: np.ndarray, x: np.ndarray, y: np.ndarray, color: tuple[int, int, int], radius: int) -> None:
    height, width, _ = image.shape
    for dy in range(-radius, radius + 1):
        yy = y + dy
        valid_y = (yy >= 0) & (yy < height)
        for dx in range(-radius, radius + 1):
            xx = x + dx
            valid = valid_y & (xx >= 0) & (xx < width)
            image[yy[valid], xx[valid]] = color


def _draw_face_wireframe(
    image: np.ndarray,
    faces: np.ndarray,
    face_indices: np.ndarray,
    vertex_x: np.ndarray,
    vertex_y: np.ndarray,
    color: tuple[int, int, int],
    radius: int,
    maximum_faces: int | None = None,
) -> None:
    face_indices = np.asarray(face_indices, dtype=np.int64)
    if maximum_faces is not None and len(face_indices) > maximum_faces:
        slots = np.linspace(0, len(face_indices) - 1, maximum_faces, dtype=np.int64)
        face_indices = face_indices[slots]
    if not len(face_indices):
        return
    triangles = faces[face_indices]
    starts = np.concatenate((triangles[:, 0], triangles[:, 1], triangles[:, 2]))
    ends = np.concatenate((triangles[:, 1], triangles[:, 2], triangles[:, 0]))
    start_x = vertex_x[starts].astype(np.float64)
    start_y = vertex_y[starts].astype(np.float64)
    end_x = vertex_x[ends].astype(np.float64)
    end_y = vertex_y[ends].astype(np.float64)
    maximum_length = np.maximum(np.abs(end_x - start_x), np.abs(end_y - start_y))
    sample_count = int(min(32, max(4, np.ceil(maximum_length.max(initial=0.0) / 3.0) + 1)))
    for amount in np.linspace(0.0, 1.0, sample_count):
        x = np.rint(start_x + (end_x - start_x) * amount).astype(np.int64)
        y = np.rint(start_y + (end_y - start_y) * amount).astype(np.int64)
        _splat(image, x, y, color, radius)


def _render_review_tile(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces: np.ndarray,
    direction: tuple[float, float, float],
    resolution: int,
    zoom: bool,
) -> np.ndarray:
    direction_array = np.asarray(direction, dtype=np.float64)
    u, v = _view_basis(direction_array)
    centroids = vertices[faces].mean(axis=1)
    projected_u = centroids @ u
    projected_v = centroids @ v
    depths = centroids @ direction_array
    target_mask = np.zeros(len(faces), dtype=bool)
    target_mask[np.asarray(target_faces, dtype=np.int64)] = True

    model_vertices_u = vertices @ u
    model_vertices_v = vertices @ v
    if zoom:
        target_vertex_ids = np.unique(faces[np.asarray(target_faces, dtype=np.int64)].reshape(-1))
        target_u = model_vertices_u[target_vertex_ids]
        target_v = model_vertices_v[target_vertex_ids]
        center_u = float((target_u.min() + target_u.max()) * 0.5)
        center_v = float((target_v.min() + target_v.max()) * 0.5)
        span = max(float(np.ptp(target_u)), float(np.ptp(target_v)), 1e-6) * 2.5
        minimum_u, maximum_u = center_u - span * 0.5, center_u + span * 0.5
        minimum_v, maximum_v = center_v - span * 0.5, center_v + span * 0.5
    else:
        minimum_u, maximum_u = float(model_vertices_u.min()), float(model_vertices_u.max())
        minimum_v, maximum_v = float(model_vertices_v.min()), float(model_vertices_v.max())
        center_u = (minimum_u + maximum_u) * 0.5
        center_v = (minimum_v + maximum_v) * 0.5
        span = max(maximum_u - minimum_u, maximum_v - minimum_v, 1e-6) * 1.08
        minimum_u, maximum_u = center_u - span * 0.5, center_u + span * 0.5
        minimum_v, maximum_v = center_v - span * 0.5, center_v + span * 0.5

    scale_u = (resolution - 9) / max(maximum_u - minimum_u, 1e-9)
    scale_v = (resolution - 9) / max(maximum_v - minimum_v, 1e-9)
    x = np.rint(4 + (projected_u - minimum_u) * scale_u).astype(np.int64)
    y = np.rint(4 + (maximum_v - projected_v) * scale_v).astype(np.int64)
    vertex_x = np.rint(4 + (model_vertices_u - minimum_u) * scale_u).astype(np.int64)
    vertex_y = np.rint(4 + (maximum_v - model_vertices_v) * scale_v).astype(np.int64)
    inside = (x >= 0) & (x < resolution) & (y >= 0) & (y < resolution)
    flat = y[inside] * resolution + x[inside]
    maximum_depth = np.full(resolution * resolution, -np.inf, dtype=np.float64)
    np.maximum.at(maximum_depth, flat, depths[inside])
    visible = inside & (depths >= maximum_depth[np.clip(y, 0, resolution - 1) * resolution + np.clip(x, 0, resolution - 1)] - 1e-9)

    image = np.full((resolution, resolution, 3), 248, dtype=np.uint8)
    all_face_indices = np.arange(len(faces), dtype=np.int64)
    background_face_indices = all_face_indices[~target_mask]
    _draw_face_wireframe(
        image,
        faces,
        background_face_indices,
        vertex_x,
        vertex_y,
        (188, 194, 202),
        0,
        maximum_faces=8000,
    )
    _draw_face_wireframe(
        image,
        faces,
        np.asarray(target_faces, dtype=np.int64),
        vertex_x,
        vertex_y,
        (239, 40, 96),
        1,
    )
    background = visible & ~target_mask
    target = visible & target_mask
    _splat(image, x[background], y[background], (155, 163, 175), 1 if zoom else 0)
    _splat(image, x[target], y[target], (239, 40, 96), 3 if zoom else 2)

    target_x = x[target_mask & inside]
    target_y = y[target_mask & inside]
    if len(target_x):
        left = max(1, int(target_x.min()) - 4)
        right = min(resolution - 2, int(target_x.max()) + 4)
        top = max(1, int(target_y.min()) - 4)
        bottom = min(resolution - 2, int(target_y.max()) + 4)
        image[top : top + 2, left : right + 1] = (0, 150, 136)
        image[bottom - 1 : bottom + 1, left : right + 1] = (0, 150, 136)
        image[top : bottom + 1, left : left + 2] = (0, 150, 136)
        image[top : bottom + 1, right - 1 : right + 1] = (0, 150, 136)
    return image


def render_region_review_image(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces: np.ndarray,
    path: Path,
    tile_resolution: int = 320,
) -> None:
    tiles = []
    for zoom in (False, True):
        row = [
            _render_review_tile(
                vertices,
                faces,
                target_faces,
                direction,
                resolution=tile_resolution,
                zoom=zoom,
            )
            for direction in REVIEW_VIEW_DIRECTIONS
        ]
        tiles.append(np.concatenate(row, axis=1))
    sheet = np.concatenate(tiles, axis=0)
    _write_rgb_png(path, sheet)


def build_source_region_review(
    *,
    input_path: Path,
    review_dir: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    connectivity_colors: list[str],
    display_colors: list[str],
    groups: list[np.ndarray],
    noise_max_faces: int,
    small_region_max_faces: int,
    visible_faces: np.ndarray | None,
    view_count: int,
    depth_map_resolution: int,
    image_resolution: int,
) -> dict:
    normalized_groups = [np.asarray(group, dtype=np.int64) for group in groups]
    from .region_review import select_region_review_groups
    _, review_candidates = select_region_review_groups(
        vertices, faces, normalized_groups, noise_max_faces,
        small_region_max_faces, visible_faces)
    review_groups = [group for group, _ in review_candidates]
    review_categories = [category for _, category in review_candidates]
    records = classify_review_groups_semantically(
        vertices=vertices,
        faces=faces,
        connectivity_colors=connectivity_colors,
        display_colors=display_colors,
        all_groups=normalized_groups,
        review_groups=review_groups,
        min_faces=small_region_max_faces + 1,
        visible_faces=visible_faces,
        view_count=view_count,
        depth_map_resolution=depth_map_resolution,
    )
    review_dir.mkdir(parents=True, exist_ok=True)
    manifest_items = []
    decision_items = []
    for group, category, record in zip(review_groups, review_categories, records):
        fragment_id = int(record["fragment_id"])
        image_path = review_dir / f"F{fragment_id:03d}_context_and_zoom.png"
        render_region_review_image(
            vertices,
            faces,
            group,
            image_path,
            tile_resolution=image_resolution,
        )
        item = dict(record)
        item["geometry_evidence_score"] = item.pop("semantic_keep_score", None)
        item["geometry_evidence_threshold"] = item.pop("semantic_keep_threshold", None)
        item.pop("semantic_decision", None)
        item.pop("semantic_decision_confidence", None)
        item["review_status"] = "awaiting_user_classification"
        item["review_category"] = category
        item["review_image"] = str(image_path)
        item["review_layout"] = {
            "top_row": "whole-model context with candidate highlighted magenta",
            "bottom_row": "candidate-centered zoom with local context",
            "columns": ["+X", "-X", "+Y", "-Y", "+Z", "-Z"],
        }
        manifest_items.append(item)
        decision_items.append(
            {
                "fragment_id": fragment_id,
                "source_min_face_index": int(record["source_min_face_index"]),
                "semantic_label": "",
                "visual_confidence": "UNKNOWN",
                "classification": "",
            }
        )
    manifest = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "source": str(input_path.resolve()),
        "source_face_count": int(len(faces)),
        "noise_max_faces": int(noise_max_faces),
        "small_region_max_faces": int(small_region_max_faces),
        "review_required": bool(manifest_items),
        "instruction": (
            "Regions through the small-region threshold and long-thin regions require "
            "semantic classification. Set classification to noise, part, or uncertain. "
            "Classification never merges, deletes, recolors, or repairs source geometry."
        ),
        "items": manifest_items,
    }
    manifest_path = review_dir / "manifest.json"
    decision_path = review_dir / "user_decisions.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    decision_template = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "source": str(input_path.resolve()),
        "source_face_count": int(len(faces)),
        "noise_max_faces": int(noise_max_faces),
        "small_region_max_faces": int(small_region_max_faces),
        "user_confirmed": False,
        "items": decision_items,
    }
    decision_path.write_text(
        json.dumps(decision_template, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(manifest_path)
    manifest["decision_path"] = str(decision_path)
    return manifest


def load_confirmed_region_decisions(
    path: Path,
    *,
    input_path: Path,
    source_face_count: int,
    noise_max_faces: int,
    small_region_max_faces: int,
    expected_records: list[dict],
) -> dict[int, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("region review JSON must contain an object")
    if int(payload.get("schema_version", -1)) != REVIEW_SCHEMA_VERSION:
        raise ValueError("region review JSON has an unsupported schema_version")
    if Path(str(payload.get("source", ""))).resolve() != input_path.resolve():
        raise ValueError("region review JSON belongs to a different source file")
    if int(payload.get("source_face_count", -1)) != int(source_face_count):
        raise ValueError("region review JSON source_face_count does not match the source")
    if int(payload.get("noise_max_faces", -1)) != int(noise_max_faces):
        raise ValueError("region review JSON noise_max_faces does not match --noise-review-max-faces")
    if int(payload.get("small_region_max_faces", -1)) != int(small_region_max_faces):
        raise ValueError("region review JSON small_region_max_faces does not match --small-region-review-max-faces")
    if payload.get("user_confirmed") is not True:
        raise ValueError("region review JSON requires user_confirmed=true")

    expected = {
        int(record["fragment_id"]): int(record["source_min_face_index"])
        for record in expected_records
    }
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("region review JSON items must be an array")
    decisions: dict[int, dict] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("each region review item must be an object")
        fragment_id = int(item.get("fragment_id", -1))
        if fragment_id in decisions:
            raise ValueError(f"duplicate region review fragment_id F{fragment_id:03d}")
        if fragment_id not in expected:
            raise ValueError(f"unknown region review fragment_id F{fragment_id:03d}")
        source_min_face_index = int(item.get("source_min_face_index", -1))
        if source_min_face_index != expected[fragment_id]:
            raise ValueError(f"F{fragment_id:03d} source_min_face_index does not match current recognition")
        classification = str(item.get("classification", "")).strip().lower()
        if classification not in {"noise", "part", "uncertain"}:
            raise ValueError(
                f"F{fragment_id:03d} classification must be noise, part, or uncertain"
            )
        label = str(item.get("semantic_label", "")).strip()
        if not label:
            raise ValueError(f"F{fragment_id:03d} requires a semantic_label from image review")
        decisions[fragment_id] = {
            "fragment_id": fragment_id,
            "source_min_face_index": source_min_face_index,
            "semantic_label": label,
            "visual_confidence": str(item.get("visual_confidence", "UNKNOWN")),
            "classification": classification,
        }
    if set(decisions) != set(expected):
        missing = sorted(set(expected) - set(decisions))
        raise ValueError(
            "region review JSON must include every candidate; missing "
            + ", ".join(f"F{fragment_id:03d}" for fragment_id in missing)
        )
    return decisions
