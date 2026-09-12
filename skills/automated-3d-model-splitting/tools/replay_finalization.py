"""Replay a saved finalization input without source decoding or assembly planning."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from split3mf.common import load_core_dependencies
load_core_dependencies()
import numpy as np
import trimesh
from split3mf.mesh_finalization import SourcePreservingMeshFinalizer, SourcePreservingFinalizationPolicy
from split3mf.print_tolerance import PrintTolerance, tolerance_scope

parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--case',type=Path,required=True)
args = parser.parse_args()
settings = json.loads((args.case/'policy.json').read_text(encoding='utf-8'))
data = np.load(args.case/'input.npz',allow_pickle=False)
metadata = json.loads((args.case/'input_metadata.json').read_text(encoding='utf-8'))
mesh = trimesh.Trimesh(vertices=data['vertices'],faces=data['faces'],metadata=metadata,process=False)
data.close()
with tolerance_scope(PrintTolerance(settings['micro_area_mm2'],settings['surface_distance_mm'],args.case.parent.parent)):
    result = SourcePreservingMeshFinalizer.finalize(mesh,SourcePreservingFinalizationPolicy(**settings['policy']))
(args.case/'replay_audit.json').write_text(json.dumps(result.audit,default=str,indent=2),encoding='utf-8')
print(json.dumps(result.audit,default=str))
