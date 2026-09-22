"""Recover a saved current-parent NPZ without claiming a finished split body."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from split3mf.common import load_core_dependencies
load_core_dependencies()
import trimesh
from split3mf.complete_parent_repair import repair_complete_parent
from split3mf.print_tolerance import PrintTolerance, tolerance_scope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(input=str(args.input.resolve()),
                  input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  complete_assembly=False)
    try:
        with np.load(args.input, allow_pickle=False) as data:
            mesh = trimesh.Trimesh(data['vertices'], data['faces'], process=False)
        with tolerance_scope(PrintTolerance(recovery_dir=args.output_dir / 'recovery')):
            candidate, audit = repair_complete_parent(mesh)
        report.update(audit)
        np.savez_compressed(args.output_dir / 'intact_parent_validated.npz',
                            vertices=candidate.vertices, faces=candidate.faces)
    except (ValueError, AssertionError) as error:
        report.update(valid=False, error=str(error))
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'valid': report['valid'], 'report': str(args.output_dir / 'report.json')}))
    return 0 if report['valid'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
