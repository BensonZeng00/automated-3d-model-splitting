from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


class StageArtifactStore:
    """Write one human-readable manifest per application stage atomically."""

    def __init__(self, root: Path, *, run_id: str | None = None) -> None:
        self.root = Path(root).expanduser().resolve()
        self.run_id = run_id or uuid.uuid4().hex
        if any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in self.run_id):
            raise ValueError("invalid stage artifact run id")
        self.run_dir = self.root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, stage: str, result: Any, *, inputs: Any = None) -> Path:
        stage = self._stage_name(stage)
        target = self.run_dir / f"{stage}.json"
        record = {
            "schema_version": 1,
            "stage": stage,
            "status": "completed",
            "inputs": self._json_value(inputs),
            "result": self._json_value(result),
        }
        self._atomic_write(target, json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True))
        return target

    def write_arrays(self, stage: str, **arrays: Any) -> Path:
        """Persist numerical stage data in NPZ with a JSON index and digest."""
        import numpy as np

        stage = self._stage_name(stage)
        target = self.run_dir / f"{stage}.npz"
        fd, temporary = tempfile.mkstemp(prefix=f".{stage}-", suffix=".npz", dir=self.run_dir)
        os.close(fd)
        try:
            np.savez_compressed(temporary, **arrays)
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        self.write_json(
            f"{stage}_manifest",
            {"artifact": target.name, "sha256": digest,
             "arrays": {name: {"shape": list(np.asarray(value).shape),
                                "dtype": str(np.asarray(value).dtype)}
                        for name, value in arrays.items()}},
        )
        return target

    @staticmethod
    def _stage_name(value: str) -> str:
        name = str(value).strip().lower().replace("-", "_")
        if not name or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_" for char in name):
            raise ValueError(f"invalid stage artifact name: {value!r}")
        return name

    @classmethod
    def _json_value(cls, value: Any) -> Any:
        if is_dataclass(value):
            return cls._json_value(asdict(value))
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): cls._json_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [cls._json_value(item) for item in value]
        if hasattr(value, "tolist"):
            return cls._json_value(value.tolist())
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)

    @staticmethod
    def _atomic_write(target: Path, content: str) -> None:
        fd, temporary = tempfile.mkstemp(prefix=f".{target.stem}-", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
                stream.write("\n")
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
