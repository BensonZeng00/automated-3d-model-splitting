from pathlib import Path
import sys

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from split3mf.common import load_core_dependencies

load_core_dependencies()

from split3mf.topology import (
    TOPOLOGY_DEFECT_RATIO_LIMIT,
    audit_mesh_topology,
    topology_defects_within_tolerance,
)


def test_one_local_over_shared_patch_is_within_one_percent_edge_tolerance():
    mesh = trimesh.creation.icosphere(subdivisions=2)
    mesh.faces = np.vstack((mesh.faces, mesh.faces[:1]))

    audit = audit_mesh_topology(mesh)

    assert audit["watertight"] is False
    assert audit["over_shared_edges"] == 3
    assert audit["topology_defect_ratio"] < TOPOLOGY_DEFECT_RATIO_LIMIT
    assert topology_defects_within_tolerance(audit) is True


def test_topology_tolerance_is_strictly_less_than_one_percent():
    assert topology_defects_within_tolerance({
        "unique_edges": 1000,
        "topology_defect_ratio": 0.009,
    }) is True
    assert topology_defects_within_tolerance({
        "unique_edges": 1000,
        "topology_defect_ratio": 0.01,
    }) is False
