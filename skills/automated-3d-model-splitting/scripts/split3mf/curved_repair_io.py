"""Use the ordinary 3MF package contract for a normalized two-part repair."""
from pathlib import Path
import json
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial import cKDTree
from .package_io import load_colored_mesh_objects_3mf


def read_stage(path):
    with zipfile.ZipFile(path) as archive:
        root=ET.fromstring(archive.read('3D/3dmodel.model'))
        if root.get('unit','millimeter')!='millimeter':
            raise ValueError('Repair requires an already normalized millimeter stage')
        for node in root.iter():
            if 'transform' in node.attrib and not np.allclose(
                np.fromstring(node.get('transform'),sep=' '),[1,0,0,0,1,0,0,0,1,0,0,0]):
                raise ValueError('Repair requires baked stage transforms')
        settings=json.loads(archive.read('Metadata/project_settings.config')) if 'Metadata/project_settings.config' in archive.namelist() else {}
        application=next((e.text for e in root if e.get('name')=='Application'),None)
    objects=load_colored_mesh_objects_3mf(Path(path))
    if len(objects)!=1:
        raise ValueError('Repair requires a single standalone source/part mesh')
    return objects[0],settings,application


def match_source_faces(mesh,source,tolerance=1e-6):
    distances,ids=cKDTree(source.triangles_center).query(mesh.triangles_center)
    matched=(distances<tolerance)&(np.sum(mesh.face_normals*source.face_normals[ids],axis=1)>.999999)
    candidates=np.flatnonzero(matched)
    delta=mesh.triangles[candidates,:,None,:]-source.triangles[ids[candidates],None,:,:]
    matched[candidates] &= np.linalg.norm(delta,axis=3).min(axis=2).max(axis=1)<tolerance
    return matched,ids


def part_record(template,mesh,role,index,parent=None):
    slots=set(template['face_filament_slot_indices'])
    if len(slots)!=1:
        raise ValueError('Curved repair currently requires single-material part templates')
    slot=int(next(iter(slots)))
    return {'part_id':f'P{index:02d}_{role}', 'mesh':mesh,
            'color_hex':template['palette'][slot], 'color_name':f'filament_{slot+1}',
            'color_code':str(slot+1), 'filament_slot_index':slot,
            'color_resolution_status':'source_metadata',
            'annotation':{'selected_processing_mode':'body' if role=='body' else 'inward',
                          'parent_part':parent,'unit':'millimeter','repair':'source_following_backing'}}
