from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


STAGE_CACHE_SCHEMA_VERSION = 1


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, set):
        return [_json_value(item) for item in sorted(value, key=str)]
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def fingerprint_payload(payload: Any) -> str:
    encoded = json.dumps(
        _json_value(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def implementation_fingerprint(module_dir: Path) -> str:
    relevant_modules = (
        "common.py",
        "micro_interfaces.py", "winding.py", "micro_mesh_repair.py", "print_tolerance.py", "surface_preservation.py", "edge_index.py",
        "quad_regularization.py", "finalization_case.py",
        "backing_repair.py", "backing_thickness.py", "curved_backing.py", "local_ray_probe.py",
        "assembly_overlap.py", "post_fit_difference.py", "post_fit_audit.py",
        "boundary_clarity.py",
        "curve_clarity.py",
        "curve_preview.py",
        "confirmed_seam.py",
        "domain/__init__.py",
        "domain/split_config.py",
        "domain/seam_smoothing_policy.py",
        "domain/planar_arc_retopology_config.py",
        "domain/planar_arc_retopology_context.py",
        "domain/cap_decision.py",
        "boundary_review.py",
        "pipeline.py",
        "application/__init__.py",
        "application/stage_artifacts.py",
        "application/recognized_boundaries.py",
        "application/boundary_snapshot_builder.py",
        "uniform_fit.py",
        "assembly.py",
        "layer_seam_planning.py",
        "boolean_cutters.py",
        "interface_retopology.py",
        "planar_arc.py",
        "cap_template.py",
        "connector_geometry.py",
        "connector_planning.py",
        "connector_surface.py",
        "connector_topology.py",
        "debug_export.py",
        "part_geometry.py", "direction_field.py", "assembly_references.py", "interface_thickness.py", "cap_planning.py", "surface_construction.py", "connector_building.py", "part_mesh_building.py", "parent_thickness_probe.py", "direction_planner.py", "adaptive_cap_planner.py", "boundary_triangulator.py", "part_mesh_builder.py",
        "interface_retreat.py",
        "local_connectors.py",
        "explicit_merge.py",
        "region_review.py",
        "boundary_correspondence.py",
        "subdivision_proof.py",
        "cap_backoff.py",
        "guided_internal_cut.py",
        "mesh.py",
        "mesh_finalization.py",
        "package_io.py",
        "validation.py",
    )
    digest = hashlib.sha256()
    for name in relevant_modules:
        path = Path(module_dir) / name
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def normalized_run_arguments(namespace: Any) -> dict[str, Any]:
    ignored = {
        "cache_dir",
        "debug_recursive_3mf",
        "debug_recursive_steps",
        "diagnostic_preview",
        "output",
        "overwrite",
        "preflight_only",
        "recognize_only",
        "resume",
    }
    result: dict[str, Any] = {}
    for name, value in sorted(vars(namespace).items()):
        if name in ignored:
            continue
        if name == "full_tree_preflight":
            # Preflight controls when validation runs, not the generated
            # geometry.  Canonicalize to the historical enabled value so a
            # diagnostic resume can skip the repeated check while reusing
            # artifacts produced before this normalization was introduced.
            result[name] = True
            continue
        if name.endswith("_json") and value:
            path = Path(str(value)).expanduser()
            result[name] = {
                "path": str(path.resolve()),
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        else:
            result[name] = _json_value(value)
    return result


class RecursiveStageCache:
    """Validated content-addressed storage for strict recursive stages."""

    def __init__(self, root: Path, *, mode: str = "auto") -> None:
        self.root = Path(root).expanduser().resolve()
        self.mode = str(mode)
        self.stages_dir = self.root / "stages"
        self.stages_dir.mkdir(parents=True, exist_ok=True)

    def stage_key(
        self,
        *,
        run_fingerprint: str,
        implementation: str,
        step: dict,
        input_artifact_sha256: str,
    ) -> str:
        return fingerprint_payload(
            {
                "schema_version": STAGE_CACHE_SCHEMA_VERSION,
                "run_fingerprint": run_fingerprint,
                "implementation_fingerprint": implementation,
                "step": step,
                "input_artifact_sha256": input_artifact_sha256,
            }
        )

    def lookup(self, key: str) -> dict | None:
        stage_dir = self.stages_dir / key
        manifest_path = stage_dir / "manifest.json"
        if not manifest_path.is_file():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self._validate_manifest(stage_dir, key, manifest)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            if self.mode == "strict":
                raise ValueError(f"recursive stage cache is invalid for {key}: {exc}") from exc
            return None
        return {"stage_dir": stage_dir, "manifest": manifest}

    def commit(
        self,
        *,
        key: str,
        run_fingerprint: str,
        implementation: str,
        step: dict,
        input_artifact_sha256: str,
        changed_entries: list[dict],
        stage_records: list[dict],
    ) -> dict:
        final_dir = self.stages_dir / key
        existing = self.lookup(key)
        if existing is not None:
            return existing

        temporary_dir = Path(
            tempfile.mkdtemp(prefix=f".{key[:12]}-", dir=str(self.stages_dir))
        )
        try:
            entries = []
            for position, entry in enumerate(changed_entries):
                source_path = Path(str(entry["source_3mf_path"]))
                artifact_name = f"{position:02d}_{source_path.name}"
                artifact_path = temporary_dir / artifact_name
                shutil.copy2(source_path, artifact_path)
                entries.append(
                    {
                        "artifact": artifact_name,
                        "sha256": sha256_file(artifact_path),
                        "entry": self._serializable_entry(entry),
                    }
                )
            manifest = {
                "schema_version": STAGE_CACHE_SCHEMA_VERSION,
                "key": key,
                "run_fingerprint": run_fingerprint,
                "implementation_fingerprint": implementation,
                "input_artifact_sha256": input_artifact_sha256,
                "step": _json_value(step),
                "entries": entries,
                "stage_records": _json_value(stage_records),
            }
            (temporary_dir / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2),
                encoding="utf-8",
            )
            try:
                os.replace(temporary_dir, final_dir)
            except FileExistsError:
                shutil.rmtree(temporary_dir)
            return self.lookup(key) or {
                "stage_dir": final_dir,
                "manifest": manifest,
            }
        except Exception:
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir)
            raise

    @staticmethod
    def _serializable_entry(entry: dict) -> dict:
        included = {
            "annotation",
            "color_code",
            "color_hex",
            "color_name",
            "contains",
            "face_color_hexes",
            "face_filament_slot_indices",
            "face_paint_color_tokens",
            "filament_slot_index",
            "origin_step",
            "part_id",
            "source_3mf_origin_step",
            "source_part_index",
            "state_role",
            "stats",
        }
        return _json_value(
            {name: value for name, value in entry.items() if name in included}
        )

    @staticmethod
    def _validate_manifest(stage_dir: Path, key: str, manifest: dict) -> None:
        if int(manifest.get("schema_version", -1)) != STAGE_CACHE_SCHEMA_VERSION:
            raise ValueError("schema version mismatch")
        if str(manifest.get("key")) != key:
            raise ValueError("stage key mismatch")
        entries = manifest.get("entries")
        if not isinstance(entries, list) or not entries:
            raise ValueError("cache contains no changed parts")
        for record in entries:
            artifact = stage_dir / str(record.get("artifact", ""))
            if not artifact.is_file():
                raise ValueError(f"missing artifact {artifact.name}")
            if sha256_file(artifact) != str(record.get("sha256", "")):
                raise ValueError(f"artifact hash mismatch for {artifact.name}")

