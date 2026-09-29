import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from split3mf.detached_micro_shells import remove_detached_micro_shells


def _part(second_radius: float) -> dict:
    main = trimesh.creation.box(extents=(20, 20, 20))
    tiny = trimesh.creation.icosphere(subdivisions=1, radius=second_radius)
    tiny.apply_translation((30, 0, 0))
    mesh = trimesh.util.concatenate((main, tiny))
    return {"part_id": "P03", "mesh": mesh,
            "face_paint_color_tokens": ["white"] * len(mesh.faces)}


def test_removes_only_small_detached_shell_and_aligns_face_colors():
    part = _part(0.03)
    result = remove_detached_micro_shells(part)
    assert result["removed_shell_count"] == 1
    assert len(part["mesh"].faces) == 12
    assert len(part["face_paint_color_tokens"]) == 12
    assert part["mesh"].is_watertight


def test_rejects_substantial_detached_shell_without_mutating_part():
    part = _part(1.0)
    original_faces = len(part["mesh"].faces)
    with pytest.raises(ValueError, match="too large"):
        remove_detached_micro_shells(part)
    assert len(part["mesh"].faces) == original_faces
