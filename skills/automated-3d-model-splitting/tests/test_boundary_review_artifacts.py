from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from split3mf.application.boundary_review_artifacts import (
    _component_paint_color,
    _boundary_display_groups,
    _seam_color,
)
from split3mf.recognition_metadata import annotate_recognition_with_visual_semantics


def test_shared_simplified_seam_is_drawn_once_with_shared_stroke_color() -> None:
    angles_a = np.linspace(0, 2 * np.pi, 30, endpoint=False)
    angles_b = np.linspace(0.03, 2 * np.pi + 0.03, 26, endpoint=False)
    first = np.column_stack((np.cos(angles_a), np.sin(angles_a), np.zeros(30)))
    second = np.column_stack((np.cos(angles_b), np.sin(angles_b), np.zeros(26)))
    distinct = second + np.array([5.0, 0.0, 0.0])

    visible, colors = _boundary_display_groups(((first,), (second, distinct)))

    assert list(visible) == [1, 2]
    assert len(visible[1]) == 1
    assert len(visible[2]) == 1
    assert np.array_equal(visible[2][0][0], distinct)
    assert colors[1][0] == colors[2][0]
    assert colors[2][1] != colors[2][0]
    records = {1: {"color_code": "0C", "color_hex": "#312121"}}
    assert _component_paint_color(1, records, {"0C": "#FFFFFF"}) == "#312121"


def test_unlabeled_region_is_reviewable_without_invented_part_identity() -> None:
    records = annotate_recognition_with_visual_semantics([{"part_index": 1}], {})

    assert records[0]["visual_semantic_label"] == "待确认区域 P01"
    assert records[0]["visual_semantic_confidence_score"] == 0.0


def test_boundary_colors_do_not_repeat_after_base_palette() -> None:
    assert len({_seam_color(index) for index in range(30)}) == 30
