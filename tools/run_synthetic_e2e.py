"""Run the public synthetic painted-3MF end-to-end release test."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile

from synthetic_fixture import write_fixture


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = REPOSITORY_ROOT / "skills" / "automated-3d-model-splitting"
ENTRY_SCRIPT = SKILL_ROOT / "scripts" / "split_painted_3mf.py"
CORE_NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"


def inspect_output(path: Path) -> dict[str, object]:
    with zipfile.ZipFile(path, "r") as archive:
        entries = set(archive.namelist())
        required = {"[Content_Types].xml", "_rels/.rels", "3D/3dmodel.model"}
        missing = sorted(required - entries)
        if missing:
            raise AssertionError(f"output package is missing entries: {missing}")
        root = ET.fromstring(archive.read("3D/3dmodel.model"))

    namespace = {"m": CORE_NS}
    resources = root.find("m:resources", namespace)
    build = root.find("m:build", namespace)
    if resources is None or build is None:
        raise AssertionError("output has no 3MF resources/build section")

    objects = resources.findall("m:object", namespace)
    mesh_objects = [obj for obj in objects if obj.find("m:mesh", namespace) is not None]
    assembly_objects = [obj for obj in objects if obj.find("m:components", namespace) is not None]
    build_items = build.findall("m:item", namespace)
    if len(mesh_objects) != 2:
        raise AssertionError(f"expected 2 mesh parts, found {len(mesh_objects)}")
    if len(assembly_objects) != 1:
        raise AssertionError(f"expected 1 assembly object, found {len(assembly_objects)}")
    components = assembly_objects[0].findall("m:components/m:component", namespace)
    if len(components) != 2:
        raise AssertionError(f"expected 2 assembly components, found {len(components)}")
    if len(build_items) != 1:
        raise AssertionError(f"expected one build item, found {len(build_items)}")
    if build_items[0].get("objectid") != assembly_objects[0].get("id"):
        raise AssertionError("build item does not reference the assembly object")

    metadata = {
        item.get("name"): item.text or ""
        for item in root.findall("m:metadata", namespace)
    }
    application = metadata.get("Application", "")
    if "automated-3d-model-splitting 1.3.5" not in application:
        raise AssertionError(f"unexpected generator metadata: {application!r}")

    return {
        "mesh_parts": len(mesh_objects),
        "assembly_objects": len(assembly_objects),
        "assembly_components": len(components),
        "build_items": len(build_items),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="split3mf-release-") as temporary:
        directory = Path(temporary)
        source = write_fixture(directory / "synthetic-painted-cube.3mf")
        output = directory / "synthetic-painted-cube-assembly.3mf"
        command = [
            sys.executable,
            "-X",
            "utf8",
            "-B",
            str(ENTRY_SCRIPT),
            "--input",
            str(source),
            "--output",
            str(output),
            "--min-faces",
            "1",
            "--exterior-view-count",
            "8",
            "--exterior-depth-map-resolution",
            "64",
            "--visual-validation-view-count",
            "8",
            "--visual-validation-resolution",
            "64",
            "--visual-max-intrusion-ratio",
            "0.05",
            "--boundary-fairing-mode",
            "off",
            "--lead-in-mm",
            "0",
            "--fit-clearance-mm",
            "0.10",
            "--overwrite",
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0:
            print(completed.stdout)
            print(completed.stderr, file=sys.stderr)
            raise SystemExit(f"split command failed with exit code {completed.returncode}")
        result = inspect_output(output)
        result["source_faces"] = 768
        result["status"] = "passed"
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
