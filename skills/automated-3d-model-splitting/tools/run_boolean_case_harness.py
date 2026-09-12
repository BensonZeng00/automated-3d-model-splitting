#!/usr/bin/env python3
"""Replay one cached recursive Boolean case without rebuilding connectors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from split3mf import common  # noqa: E402

common.load_core_dependencies()

from split3mf.boolean_case_cache import load_boolean_case_cache  # noqa: E402
from split3mf.local_connectors import subtract_socket_cutters  # noqa: E402
from split3mf.validation import boolean_collapsed_face_audit  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--through-cutter", type=int, default=None)
    args = parser.parse_args()
    parent, cutters, metadata = load_boolean_case_cache(args.input)
    if args.through_cutter is not None:
        cutters = cutters[: max(int(args.through_cutter) + 1, 0)]
    result, record = subtract_socket_cutters(
        parent,
        cutters,
        allow_empty_intersection=True,
        inherited_collapsed_face_budget=0,
        maximum_new_collapsed_face_ratio=0.005,
    )
    print(
        json.dumps(
            {
                "passed": True,
                "input": str(args.input),
                "metadata": metadata,
                "applied_cutter_count": record.get("applied_socket_cutter_count", 0),
                "faces": len(result.faces),
                "audit": boolean_collapsed_face_audit(result),
                "boolean_record": record,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
