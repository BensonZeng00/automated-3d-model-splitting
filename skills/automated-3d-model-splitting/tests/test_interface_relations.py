from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from split3mf.interface_relations import interface_role_index


def _plan():
    relation = {
        "interface_id": "I014",
        "parts": ["P05", "P10"],
        "tenon_part": "P05",
        "mortise_part": "P10",
        "contact": {"shared_boundary_loops": [{"tenon_loop_index": 0, "mortise_loop_index": 1}]},
    }
    return {
        "relation_table": {"rows": [["I014", "P05 eye", "P10 mask"]]},
        "interfaces": [relation],
    }


def test_interface_role_index_uses_and_checks_published_table():
    assert interface_role_index(_plan()) == {
        "I014": {"tenon_part": "P05", "mortise_part": "P10"}
    }


def test_interface_role_index_rejects_table_plan_disagreement():
    plan = _plan()
    plan["relation_table"]["rows"][0][1:] = ["P10 mask", "P05 eye"]
    with pytest.raises(ValueError, match="roles do not match"):
        interface_role_index(plan)


def test_interface_role_index_rejects_duplicate_ids():
    plan = _plan()
    plan["relation_table"]["rows"].append(["I014", "P05 eye", "P10 mask"])
    with pytest.raises(ValueError, match="duplicate"):
        interface_role_index(plan)


def test_interface_role_index_keeps_roles_scoped_to_interface_id():
    plan = _plan()
    plan["relation_table"]["rows"].append(["I005", "P10 eye", "P01 body"])
    plan["interfaces"].append({
        "interface_id": "I005",
        "parts": ["P10", "P01"],
        "tenon_part": "P10",
        "mortise_part": "P01",
        "contact": {"shared_boundary_loops": [{
            "tenon_loop_index": 1, "mortise_loop_index": 2,
        }]},
    })

    roles = interface_role_index(plan)

    assert roles["I005"]["tenon_part"] == "P10"
    assert roles["I014"]["mortise_part"] == "P10"
