#!/usr/bin/env python3
"""Split exterior paint-connected regions from a vendor-painted 3MF.

Part recognition uses only paint visible from outside the model. Paint on
occluded deep surfaces is reassigned to the recognition base color before
component, body, boundary, and assembly inference. Through-like structure is
evidence for choosing the root body only. Every non-body component uses the
same recursive inward-extrusion workflow.
"""

from __future__ import annotations

import argparse
import collections
import heapq
import importlib.util
import json
import math
import posixpath
import re
import shutil
import shlex
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

np = None
trimesh = None
cKDTree = None


CORE_URI = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
MATERIAL_URI = "http://schemas.microsoft.com/3dmanufacturing/material/2015/02"
PRODUCTION_URI = "http://schemas.microsoft.com/3dmanufacturing/production/2015/06"
CORE_NS = "{" + CORE_URI + "}"
MATERIAL_NS = "{" + MATERIAL_URI + "}"
PRODUCTION_NS = "{" + PRODUCTION_URI + "}"
VERSION = "2.1.0"
DEFAULT_EFFECTIVE_MINIMUM_INWARD_DEPTH_MM = 3.0
MAXIMUM_SAFE_INWARD_DEPTH_MM = 10.0
DEFAULT_LEAD_IN_SLOPE_DEGREES = 45.0
PARENT_THICKNESS_CLEARANCE_MM = 0.05
# When the measured parent is thinner than the preferred backing depth, a
# coherent floor may legitimately have less travel at the high side of a
# curved source rim.  Keep enough depth beyond the 0.60 mm lead-in for a
# continuous printable skin instead of forcing a folded local-offset cap.
MINIMUM_COHERENT_PLANAR_FLOOR_DEPTH_MM = 0.08
MAXIMUM_INWARD_CAP_PLANARITY_ERROR_MM = 0.05
MINIMUM_INWARD_CAP_NORMAL_COSINE = 0.90
# Compatibility alias for older call sites and reports. Geometry may go below
# this value only when the measured parent thickness requires it.
MINIMUM_INWARD_DEPTH_MM = DEFAULT_EFFECTIVE_MINIMUM_INWARD_DEPTH_MM
CORE_DEPENDENCIES = ("numpy", "scipy", "trimesh", "networkx", "manifold3d", "matplotlib", "PIL")
UNIT_TO_MM = {
    "micron": 0.001,
    "millimeter": 1.0,
    "centimeter": 10.0,
    "inch": 25.4,
    "foot": 304.8,
    "meter": 1000.0,
}

DEFAULT_COLOR_INFO = {
    "DEFAULT": {
        "name": "default",
        "hex": "#C8C8C8",
        "rgba": [200, 200, 200, 255],
        "filament_slot": None,
        "mapping_source": "default_fallback",
    },
}

FALLBACK_PALETTE = [
    "#F4EE2A",
    "#EC008C",
    "#424379",
    "#A3D8E1",
    "#FCFB79",
    "#C12E1F",
    "#FFFFFF",
    "#018001",
    "#FF7F00",
    "#00A0FF",
]

class ColorCatalog:
    """Mutable color registry shared by pure geometry modules through stable dict identities."""

    def __init__(self) -> None:
        self.info = dict(DEFAULT_COLOR_INFO)
        self.order = {"DEFAULT": 0}

    def replace(self, info: dict, order: dict) -> None:
        self.info.clear()
        self.info.update(info)
        self.order.clear()
        self.order.update(order)


COLOR_CATALOG = ColorCatalog()
COLOR_INFO = COLOR_CATALOG.info
COLOR_ORDER = COLOR_CATALOG.order


def sanitize_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_")


def normalize_3mf_color(value: str) -> str:
    token = str(value or "").strip().upper()
    if not token.startswith("#"):
        token = "#" + token
    if re.fullmatch(r"#[0-9A-F]{6}", token):
        return token + "FF"
    if re.fullmatch(r"#[0-9A-F]{8}", token):
        return token
    return "#C8C8C8FF"


def segments_intersect_2d_strict(
    first_left: np.ndarray,
    first_right: np.ndarray,
    second_left: np.ndarray,
    second_right: np.ndarray,
    epsilon: float,
) -> bool:
    """Return whether two 2D segments cross away from their endpoints."""

    def orient(left: np.ndarray, right: np.ndarray, sample: np.ndarray) -> float:
        return float(
            (right[0] - left[0]) * (sample[1] - left[1])
            - (right[1] - left[1]) * (sample[0] - left[0])
        )

    first_a = orient(first_left, first_right, second_left)
    first_b = orient(first_left, first_right, second_right)
    second_a = orient(second_left, second_right, first_left)
    second_b = orient(second_left, second_right, first_right)
    return bool(
        first_a * first_b < -(epsilon * epsilon)
        and second_a * second_b < -(epsilon * epsilon)
    )


def fit_plane_normal(points: np.ndarray, fallback_normal: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0)
    try:
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
        normal = np.asarray(vh[-1], dtype=np.float64)
    except np.linalg.LinAlgError:
        normal = np.asarray(fallback_normal, dtype=np.float64)
    if float(np.linalg.norm(normal)) <= 1e-12:
        normal = np.asarray(fallback_normal, dtype=np.float64)
    normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
    if float(np.dot(normal, fallback_normal)) < 0.0:
        normal = -normal
    return normal


def progress(stage: str, message: str, **details) -> None:
    """Emit a readable Chinese progress annotation plus optional structured data."""
    suffix = ""
    if details:
        suffix = " " + json.dumps(details, ensure_ascii=False, sort_keys=True)
    print(f"[{stage}] {message}{suffix}", flush=True)


def dependency_install_command() -> str:
    """Suggest an interpreter-scoped install, usable in PowerShell or POSIX shells."""
    requirements = Path(__file__).resolve().parents[2] / "requirements.txt"
    arguments = [sys.executable, "-m", "pip", "install", "-r", str(requirements)]
    if sys.platform == "win32":
        return "& " + " ".join("'" + argument.replace("'", "''") + "'" for argument in arguments)
    return shlex.join(arguments)


def load_core_dependencies() -> None:
    """Import heavy geometry dependencies only after preflight succeeds."""
    global np, trimesh, cKDTree
    try:
        import numpy as numpy_module
        import trimesh as trimesh_module
        from scipy.spatial import cKDTree as scipy_ckdtree
    except ImportError as exc:
        print(f"缺少 Python 依赖：{exc.name}", file=sys.stderr)
        print("建议安装命令：" + dependency_install_command(), file=sys.stderr)
        raise SystemExit(2) from exc
    np = numpy_module
    trimesh = trimesh_module
    cKDTree = scipy_ckdtree


class DSU:
    def __init__(self, count: int) -> None:
        self.parent = list(range(count))
        self.size = [1] * count

    def find(self, item: int) -> int:
        parent = self.parent
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        if self.size[root_a] < self.size[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        self.size[root_a] += self.size[root_b]


@dataclass
class Component:
    color_code: str
    global_faces: np.ndarray
    face_count: int
    area: float
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    center: np.ndarray


@dataclass
class VendorPaintNode:
    """One node in Bambu's serialized per-source-triangle paint tree."""

    split_sides: int
    special_side: int = 0
    state: int = 0
    children: list["VendorPaintNode"] | None = None


def vendor_paint_state_token(state: int) -> str:
    """Return the canonical Bambu leaf token for a decoded paint state."""
    state = int(state)
    if state == 0:
        return "DEFAULT"
    if state < 3:
        return format(state << 2, "X")
    remainder = state - 3
    stream = [0xC]
    while remainder >= 15:
        stream.append(0xF)
        remainder -= 15
    stream.append(remainder)
    return "".join(format(value, "X") for value in reversed(stream))


def decode_vendor_paint_tree(token: str) -> VendorPaintNode:
    """Decode Bambu/Prusa's reversed hexadecimal TriangleSelector stream.

    Each nibble is either a leaf state or a split instruction.  The 3MF string
    stores the first stream nibble at the right-hand end, and split children are
    serialized in reverse child order.
    """
    raw = str(token).strip().upper()
    if raw == "DEFAULT" or not raw:
        return VendorPaintNode(split_sides=0, state=0)
    if not re.fullmatch(r"[0-9A-F]+", raw):
        raise ValueError(f"Invalid vendor paint token: {token!r}")
    codes = [int(char, 16) for char in reversed(raw)]
    offset = 0

    def decode_node() -> VendorPaintNode:
        nonlocal offset
        if offset >= len(codes):
            raise ValueError(f"Truncated vendor paint token: {token!r}")
        code = codes[offset]
        offset += 1
        split_sides = code & 0b11
        if split_sides:
            special_side = code >> 2
            serialized_children = [decode_node() for _ in range(split_sides + 1)]
            return VendorPaintNode(
                split_sides=split_sides,
                special_side=special_side,
                children=list(reversed(serialized_children)),
            )

        if (code & 0b1100) == 0b1100:
            extension = 0
            while True:
                if offset >= len(codes):
                    raise ValueError(f"Truncated extended vendor paint state: {token!r}")
                next_code = codes[offset]
                offset += 1
                if next_code == 0xF:
                    extension += 15
                    continue
                state = next_code + extension + 3
                break
        else:
            state = code >> 2
        return VendorPaintNode(split_sides=0, state=state)

    root = decode_node()
    if offset != len(codes):
        raise ValueError(
            f"Vendor paint token has trailing data: {token!r} ({offset}/{len(codes)} nibbles consumed)"
        )
    return root




def preflight(input_path: Path) -> dict:
    checks = {
        "python": sys.executable,
        "input_exists": input_path.exists(),
        "input_suffix": input_path.suffix.lower(),
        "dependencies": {},
        "script_line_endings": "unknown",
    }
    for module_name in CORE_DEPENDENCIES:
        checks["dependencies"][module_name] = importlib.util.find_spec(module_name) is not None
    missing = [name for name, ok in checks["dependencies"].items() if not ok]
    checks["dependency_ready"] = not missing
    checks["missing_dependencies"] = missing
    checks["install_command"] = dependency_install_command() if missing else None

    script_path = Path(__file__)
    if script_path.exists():
        data = script_path.read_bytes()
        checks["script_line_endings"] = "crlf" if b"\r\n" in data else "lf"

    return checks


def preflight_failures(checks: dict) -> list[str]:
    failures = []
    if not checks["input_exists"]:
        failures.append("input file does not exist")
    if checks["input_suffix"] != ".3mf":
        failures.append("input file is not a .3mf")
    missing = [name for name, ok in checks["dependencies"].items() if not ok]
    if missing:
        failures.append("missing Python dependencies: " + ", ".join(missing))
    return failures
