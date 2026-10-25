"""CLI gates, bounded logs and strict failures, independent of vendor models."""
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from split3mf.admission import AdmissionError, ComplexityLimits, check, limits_scope
from split3mf.execution_log import emit, run_logged
from split3mf.model_inventory import inspect_models, select_model
from split3mf.cli import main


def package(tmp_path, count=2):
    path = tmp_path / "sample.3mf"
    objects = "".join(f'<object id="{i}" name="part{i}"><mesh><vertices><vertex x="0" y="0" z="0"/><vertex x="1" y="0" z="0"/><vertex x="0" y="1" z="0"/></vertices><triangles><triangle v1="0" v2="1" v3="2" paint_color="8"/></triangles></mesh></object>' for i in range(1, count+1))
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("3D/3dmodel.model", '<model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"><resources>'+objects+'</resources></model>')
    return path


def test_multi_mesh_inventory_requires_explicit_choice(tmp_path):
    items = inspect_models(package(tmp_path))
    assert len(items) == 2
    assert select_model(items) is None
    assert select_model(items, items[1]["object_id"])["name"] == "part2"
    with pytest.raises(ValueError):
        select_model(items, "missing")


def test_cli_selection_does_not_enter_pipeline(tmp_path, monkeypatch, capsys):
    import split3mf.cli as cli
    monkeypatch.setattr(cli, "_run", lambda *a: pytest.fail("pipeline started without selection"))
    with pytest.raises(SystemExit) as caught:
        main(["--input", str(package(tmp_path)), "--log-file", str(tmp_path / "run.jsonl")])
    assert caught.value.code == 5
    assert 'selection_required' in capsys.readouterr().out


def test_limits_reject_before_pipeline_and_validate_configuration(tmp_path, monkeypatch):
    import split3mf.cli as cli
    monkeypatch.setattr(cli, "_run", lambda *a: pytest.fail("over-budget pipeline started"))
    path = package(tmp_path, 1)
    cfg = tmp_path / "limits.json"
    cfg.write_text(json.dumps({"model_xml_bytes": 1}))
    with pytest.raises(SystemExit) as caught:
        main(["--input", str(path), "--complexity-config", str(cfg), "--log-file", str(tmp_path / "run.jsonl")])
    assert caught.value.code == 6
    cfg.write_text('{"paint_token_chars":0}')
    with pytest.raises(ValueError):
        ComplexityLimits.load(cfg)


def test_budget_boundary_and_context_reset():
    with limits_scope(ComplexityLimits(paint_token_chars=10)):
        assert check("load", paint_token_chars=10)["status"] == "supported"
        with pytest.raises(AdmissionError):
            check("load", paint_token_chars=11)
    check("load", paint_token_chars=11)


def test_logs_keep_details_but_console_is_bounded(tmp_path, capsys):
    path = tmp_path / "run.jsonl"
    def execute():
        emit("recognition", "start", "start", samples=list(range(1000)))
        print("full detail " * 10000)
        raise ValueError("I001: depth failed; diagnostic=" + "X" * 10000)
    assert run_logged(execute, path) == 3
    console = capsys.readouterr().out
    assert len(console) < 2000
    records = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]
    assert {r["level"] for r in records} == {"INFO", "DEBUG", "ERROR"}
    assert any(r["level"] == "DEBUG" and len(r["message"]) > 10000 for r in records)
    assert all(len(r["message"]) <= 400 for r in records if r["level"] == "ERROR")
    assert json.loads(path.with_suffix('.summary.json').read_text(encoding='utf8'))["exit_code"] == 3


def test_boundary_self_contact_is_rejected_without_mutation():
    from split3mf.admission import audit_frozen_boundary
    points = np.array([[0.,0.,0.], [2.,2.,0.], [0.,2.,0.], [2.,0.,0.]])
    original = points.copy()
    with pytest.raises(AdmissionError):
        audit_frozen_boundary(points)
    np.testing.assert_array_equal(points, original)


def test_selected_mesh_does_not_include_sibling(tmp_path):
    from split3mf.common import load_core_dependencies
    load_core_dependencies()
    from split3mf.project import parse_3mf_model
    path = package(tmp_path)
    vertices, faces, _, settings = parse_3mf_model(path, model_entry="3D/3dmodel.model#2")
    assert len(vertices) == 3 and len(faces) == 1
    assert settings["_selected_mesh_object_id"] == "2"
    with pytest.raises(ValueError, match="selected mesh object is missing"):
        parse_3mf_model(path, model_entry="3D/3dmodel.model#99")


def test_overview_keeps_numbered_six_view_pages(tmp_path):
    from PIL import Image
    from split3mf.review_overview import build_review_overviews
    image_path = tmp_path / "full.png"
    Image.new("RGB", (1920, 640), "gray").save(image_path)
    items = [{"fragment_id": i, "review_image": str(image_path)} for i in range(1, 7)]
    pages = build_review_overviews(items, tmp_path)
    assert len(pages) == 2
    with Image.open(pages[0]) as first, Image.open(pages[1]) as second:
        assert first.size == (960, 950)
        assert second.size == (960, 190)
    assert image_path.exists()


def test_source_face_count_is_informational(tmp_path, monkeypatch):
    import split3mf.cli as cli
    path = package(tmp_path, 1)
    items = inspect_models(path)
    items[0]["source_faces"] = 10_000_000
    monkeypatch.setattr("split3mf.model_inventory.inspect_models", lambda *a, **kw: items)
    reached = []
    monkeypatch.setattr(cli, "_run", lambda *a: reached.append(True))
    with pytest.raises(SystemExit) as caught:
        main(["--input", str(path), "--log-file", str(tmp_path / "run.jsonl")])
    assert caught.value.code == 0
    assert reached == [True]
    assert "source_faces" not in ComplexityLimits.__dataclass_fields__


def test_expanded_faces_do_not_block_region_assessment():
    from split3mf.admission import assess_mesh
    # Avoid allocating a huge fixture: the old face gate reads this length;
    # region analysis operates on the actual one-triangle ndarray.
    class ReportedLargeMesh(np.ndarray):
        def __len__(self):
            return 500_001
    faces = np.array([[0, 1, 2]]).view(ReportedLargeMesh)
    vertices = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]])
    assess_mesh(vertices, faces, ["8"])
    assert "expanded_faces" not in ComplexityLimits.__dataclass_fields__


def test_many_regions_are_metrics_not_rejections():
    from split3mf.admission import assess_mesh
    count = 5001
    faces = np.arange(count * 3).reshape(-1, 3)
    vertices = np.tile([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]], (count, 1))
    report = assess_mesh(vertices, faces, ["8"] * count)
    assert report["metrics"]["raw_regions"] == count
    assert report["metrics"]["effective_regions"] == count
    assert report["status"] == "undetermined"


def test_golden_artifact_reader_requires_successful_json():
    from golden_cli import artifact_directory
    assert artifact_directory('{"exit_code":0,"stage_artifacts":"run"}', "stage_artifacts") == Path("run")
    with pytest.raises(AssertionError):
        artifact_directory('{"exit_code":6,"stage_artifacts":"run"}', "stage_artifacts")
    with pytest.raises(AssertionError):
        artifact_directory('stage_artifacts=run', "stage_artifacts")


def test_budget_is_not_structural_failure():
    with limits_scope(ComplexityLimits(paint_token_chars=1)):
        with pytest.raises(AdmissionError) as caught:
            check("load", paint_token_chars=2)
    assert caught.value.record["status"] == "budget_exceeded"
