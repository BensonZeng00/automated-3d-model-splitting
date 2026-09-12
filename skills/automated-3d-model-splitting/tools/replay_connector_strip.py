#!/usr/bin/env python3
"""Replay a captured ring strip without loading or cutting its vendor model."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from split3mf.common import load_core_dependencies
load_core_dependencies()
from split3mf.connector_topology import triangulate_bounded_ring_strip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    args = parser.parse_args()
    keys = ('outer_ids', 'outer_points', 'inner_ids', 'inner_points',
            'outer_projected', 'inner_projected')
    with np.load(args.case, allow_pickle=False) as data:
        result = triangulate_bounded_ring_strip(
            *(data[key] for key in keys),
            maximum_fanout=int(data['maximum_fanout']),
            allow_projection_bridge_passthrough=False)
    print(json.dumps(asdict(result.audit), ensure_ascii=False))
    return 0 if result.audit.valid else 3


if __name__ == '__main__':
    raise SystemExit(main())
