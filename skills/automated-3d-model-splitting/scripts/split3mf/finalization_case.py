"""Non-pickle stage snapshots for direct replay of source mesh finalization."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import os
import numpy as np
from .print_tolerance import current


class FinalizationCase:
    def __init__(self, mesh, policy):
        self.directory = None
        if current().recovery_dir is None:
            return
        digest = hashlib.sha256()
        for values in (mesh.vertices,mesh.faces):
            digest.update(np.ascontiguousarray(values).tobytes())
        digest.update(json.dumps(asdict(policy),sort_keys=True).encode())
        self.geometry_key = digest.hexdigest()
        implementation = hashlib.sha256()
        for path in sorted(Path(__file__).parent.glob('*.py')):
            implementation.update(path.read_bytes())
        self.implementation_key = implementation.hexdigest()
        self.directory = Path(current().recovery_dir)/'finalization'/digest.hexdigest()[:24]
        self.directory.mkdir(parents=True,exist_ok=True)
        self.save('input',mesh)
        settings = dict(policy=asdict(policy), micro_area_mm2=current().micro_area_mm2,
                        surface_distance_mm=current().surface_distance_mm)
        (self.directory/'policy.json').write_text(json.dumps(settings,indent=2),encoding='utf-8')
        print('finalization_case='+str(self.directory),flush=True)

    def load(self, source_mesh):
        if self.directory is None or not (self.directory/'accepted.json').exists():
            return None
        try:
            receipt = json.loads((self.directory/'accepted.json').read_text(encoding='utf-8'))
            path = self.directory/'accepted.npz'
            if (receipt['implementation'] != self.implementation_key
                or receipt['micro_area_mm2'] != current().micro_area_mm2
                or receipt['surface_distance_mm'] != current().surface_distance_mm
                or receipt['sha256'] != hashlib.sha256(path.read_bytes()).hexdigest()):
                return None
            data = np.load(path,allow_pickle=False)
            import trimesh
            mesh = trimesh.Trimesh(vertices=data['vertices'],faces=data['faces'],process=False,
                                   metadata=source_mesh.metadata.copy())
            data.close()
            from .validation import validate_mesh_in_memory
            validation = validate_mesh_in_memory(mesh)
            if any(validation[k] for k in ('open_edges','over_shared_edges','inconsistent_shared_edges')):
                return None
            mesh.metadata.update(source_preserving_finalization=receipt['audit'],
                                 orientation_repair=receipt['audit']['orientation_repair'])
            print('finalization_cache_hit='+str(self.directory),flush=True)
            return mesh, receipt['audit']
        except (OSError,ValueError,KeyError):
            return None

    def accept(self, mesh, audit):
        if self.directory is None:
            return
        self.save('accepted',mesh)
        receipt = dict(implementation=self.implementation_key, audit=audit,
                       micro_area_mm2=current().micro_area_mm2,
                       surface_distance_mm=current().surface_distance_mm,
                       sha256=hashlib.sha256((self.directory/'accepted.npz').read_bytes()).hexdigest())
        (self.directory/'accepted.json').write_text(json.dumps(receipt,default=str),encoding='utf-8')

    def save(self, name, mesh):
        if self.directory is None:
            return
        path = self.directory/(name+'.npz')
        if name == 'input' and path.exists():
            return
        temp = self.directory/(name+f'.{os.getpid()}.tmp')
        with temp.open('wb') as stream:
            np.savez_compressed(stream,vertices=np.asarray(mesh.vertices),faces=np.asarray(mesh.faces))
        temp.replace(path)
        (self.directory/(name+'_metadata.json')).write_text(
            json.dumps(mesh.metadata,default=lambda value:value.tolist() if hasattr(value,'tolist') else str(value)),encoding='utf-8')
