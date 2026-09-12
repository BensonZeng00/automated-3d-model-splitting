"""Reproduce a captured numerical boundary and export the actual edited path."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from split3mf.planar_arc import fit_stable_plane
from split3mf.curve_clarity import propose_clear_curve, crossings
from split3mf.curve_preview import export_curve_review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    with np.load(args.capture, allow_pickle=False) as data:
        source, target = data['source'], data['target']
        basis = fit_stable_plane(source)
        started = time.perf_counter()
        proposal = propose_clear_curve(target, *basis)
        record = dict(compute_seconds=time.perf_counter()-started)
        if 'guide' in data:
            record['dense_guide_projected_crossings'] = len(crossings(
                (data['guide']-basis[0]) @ np.column_stack(basis[1:3])))
        print(json.dumps(export_curve_review(source, target, proposal, basis,
                                             args.output, record=record), indent=2))


if __name__ == '__main__':
    main()
