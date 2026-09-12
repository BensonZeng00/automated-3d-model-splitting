"""Replay only final fit using complete scaled inputs and unchanged production gates."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from split3mf.common import load_core_dependencies
load_core_dependencies()
from split3mf.assembly_case import load_assembly_case
from split3mf.assembly_visibility import validate_and_seat_assembly


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--strict', action='store_true')
    parser.add_argument('--assembly-ignore-overlap-ratio', type=float, default=None,
                        help='Override the saved overlap/cutting-volume ratio; 0 disables.')
    args = parser.parse_args()
    parts, source, options = load_assembly_case(args.case)
    options['allow_manual_adjustment'] = not args.strict
    # Old snapshots must not silently restore the retired fixed 1 mm³ policy.
    options.pop('ignore_overlap_below_mm3', None)
    if args.assembly_ignore_overlap_ratio is not None:
        options['ignore_overlap_ratio'] = args.assembly_ignore_overlap_ratio
    result = validate_and_seat_assembly(parts, *source, recovery_dir=args.output, **options)
    print(json.dumps(dict(status=result.get('status', 'validated'), valid=result['valid'],
                         affected_parts=result.get('affected_parts', []))))


if __name__ == '__main__':
    main()
