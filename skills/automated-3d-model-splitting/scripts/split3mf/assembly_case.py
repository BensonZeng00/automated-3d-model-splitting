"""Self-contained scaled assembly inputs for local replay with provenance."""
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Unsupported assembly metadata: {type(value).__name__}')


def save_assembly_case(directory, parts, vertices, faces, labels, options):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = dict(source_vertices=vertices, source_faces=faces, source_labels=labels)
    records = []
    for index, part in enumerate(parts):
        arrays[f'part_{index}_vertices'] = np.asarray(part['mesh'].vertices)
        arrays[f'part_{index}_faces'] = np.asarray(part['mesh'].faces)
        records.append({key: value for key, value in part.items() if key != 'mesh'})
    mesh_path = directory / 'scaled_inputs.npz'
    np.savez_compressed(mesh_path, **arrays)
    record = dict(schema_version=1, units='millimeter', already_scaled=True,
                  scale_must_not_be_reapplied=True, parts=records, options=options,
                  meshes_sha256=hashlib.sha256(mesh_path.read_bytes()).hexdigest())
    (directory / 'case.json').write_text(json.dumps(record, default=_json_default,
                                       ensure_ascii=False, indent=2), encoding='utf-8')
    return directory


def load_assembly_case(directory):
    directory = Path(directory)
    record = json.loads((directory / 'case.json').read_text(encoding='utf-8'))
    path = directory / 'scaled_inputs.npz'
    if hashlib.sha256(path.read_bytes()).hexdigest() != record['meshes_sha256']:
        raise ValueError('Assembly replay input checksum mismatch')
    if record['schema_version'] != 1 or not record['already_scaled']:
        raise ValueError('Unsupported assembly replay stage')
    with np.load(path, allow_pickle=False) as data:
        parts = [{**part, 'mesh': trimesh.Trimesh(data[f'part_{i}_vertices'],
                                                data[f'part_{i}_faces'], process=False)}
                 for i, part in enumerate(record['parts'])]
        source = [data[name].copy() for name in ('source_vertices', 'source_faces', 'source_labels')]
    return parts, source, record['options']
