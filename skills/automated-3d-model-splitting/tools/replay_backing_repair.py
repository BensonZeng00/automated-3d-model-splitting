"""Replay a captured failed backing without repeating source recognition or cutting."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from split3mf.common import load_core_dependencies
load_core_dependencies()
from split3mf.backing_repair import ensure_backing
from split3mf.print_tolerance import PrintTolerance, tolerance_scope


def mesh(path):
    with np.load(path, allow_pickle=False) as data:
        return trimesh.Trimesh(data['vertices'], data['faces'], process=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    options = json.loads((args.case / 'replay_options.json').read_text(encoding='utf-8'))
    policy = PrintTolerance(micro_area_mm2=options['area_budget_mm2'], recovery_dir=args.output,
                            repair_thin_backing=True, backing_validation_scale=options['scale_factor'])
    with tolerance_scope(policy):
        result, audit, colors = ensure_backing(mesh(args.case / 'source_patch_unvalidated.npz'),
            mesh(args.case / 'input_unvalidated.npz'), mesh(args.case / 'complete_parent.npz'),
            options['source_codes'], options['inward'], part_id=options['part_id'])
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / 'locally_validated_backing.npz',
                        vertices=result.vertices, faces=result.faces)
    (args.output / 'result.json').write_text(json.dumps(dict(audit=audit, colors=colors),
                                           ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(status='backing_validated_locally', final_assembly_validated=False)))


if __name__ == '__main__':
    main()
