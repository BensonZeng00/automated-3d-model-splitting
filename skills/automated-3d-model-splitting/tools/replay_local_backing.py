#!/usr/bin/env python3
"""Rebuild a captured local backing with the production policy and builder."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from split3mf.common import load_core_dependencies
load_core_dependencies()
from split3mf.part_geometry import add_local_male_connector_and_backing
from split3mf.local_connectors import LocalConnectorSpec
from split3mf.print_tolerance import PrintTolerance, tolerance_scope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--refine', action='store_true')
    parser.add_argument('--finalize', action='store_true')
    args = parser.parse_args()
    with np.load(args.case, allow_pickle=False) as data:
        vertices = data['output_vertices'].tolist()
        faces = data['output_faces'].tolist()
        kwargs = {key: data[key] for key in ('boundary_points', 'inward', 'inward_directions')
                  if key in data}
        with tolerance_scope(PrintTolerance(preserve_audited_hidden_surfaces=not args.refine)):
            record = add_local_male_connector_and_backing(
                output_vertices=vertices, output_faces=faces,
                boundary_ids=data['boundary_ids'].tolist(), **kwargs,
                spec=LocalConnectorSpec(**json.loads(args.spec.read_text(encoding='utf-8'))))
        np.testing.assert_array_equal(vertices[:len(data['output_vertices'])], data['output_vertices'])
        np.testing.assert_array_equal(faces[:len(data['output_faces'])], data['output_faces'])
        if args.finalize:
            import trimesh
            from split3mf.mesh_finalization import finalize_source_preserving_mesh
            with tolerance_scope(PrintTolerance(preserve_audited_hidden_surfaces=not args.refine)):
                mesh = finalize_source_preserving_mesh(
                    trimesh.Trimesh(vertices, faces, process=False), len(data['output_faces']))
            vertices, faces = mesh.vertices, mesh.faces
            record['finalized_topology'] = {'watertight': bool(mesh.is_watertight),
                                           'winding_consistent': bool(mesh.is_winding_consistent)}
    np.savez_compressed(args.output, vertices=vertices, faces=faces)
    args.output.with_suffix('.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'local_backing_rebuilt', 'source_unchanged': True,
                      'output': str(args.output.resolve()),
                      'projection': record['backing_strip_projection_audits'],
                      'shape': record['backing_wedge_internal_dihedral']}))


if __name__ == '__main__':
    main()
