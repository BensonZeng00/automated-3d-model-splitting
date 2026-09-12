"""Rebuild a curved leaf backing from normalized, source-matched recovery stages."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import trimesh
from split3mf.common import load_core_dependencies
load_core_dependencies()
from split3mf.curved_repair_io import read_stage,match_source_faces,part_record
from split3mf.curved_backing import build
from split3mf.backing_thickness import audit_backing
from split3mf.local_connectors import _manifold64
from split3mf.uniform_fit import scale_finished_insert
from split3mf.assembly_seating import seat_insert_outward,SeatingError
from split3mf.insert_visibility import assess_insert_visibility
from split3mf.package_io import export_colored_parts_3mf,validate_colored_parts_3mf,load_colored_mesh_objects_3mf
from split3mf.validation import validate_multiview_visual_consistency,validate_mesh_in_memory


def run(args,report):
    source_entry,settings,application=read_stage(args.source)
    insert_entry,_,_=read_stage(args.insert)
    parent_entry,_,_=read_stage(args.parent)
    source,old=source_entry['mesh'],insert_entry['mesh']
    mask,ids=match_source_faces(old,source)
    if not mask.any():raise ValueError('Insert has no matching oriented source faces')
    patch=source.submesh([ids[mask]],append=True,repair=False)
    axis=-(patch.face_normals*patch.area_faces[:,None]).sum(axis=0)
    axis/=np.linalg.norm(axis)
    report['before_thickness']=audit_backing(patch,old,axis)
    print('Rebuilding source-following backing',flush=True)
    child,report['backing']=build(patch,source,axis,preferred_depth=args.max_depth,taper_slope=args.taper_slope)
    report['after_thickness']=audit_backing(patch,child,axis,minimum_mm=.45/args.scale)
    if report['after_thickness']['missing_eligible_faces']:
        raise ValueError('Repaired backing has unmeasured eligible source faces')
    if report['after_thickness']['thin_interior_area_mm2']>args.micro_area:
        raise ValueError('Repaired backing still has a broad thin interior')
    full,male=_manifold64(source),_manifold64(child)
    outside=male-full
    outside_mesh=outside.to_mesh64()
    outside_area=float(outside.surface_area())
    report['outside_source']={'volume_mm3':abs(float(outside.volume())),'area_mm2':outside_area}
    if outside_area>args.micro_area:
        raise ValueError('Backing protrudes outside the source beyond the local area budget')
    if len(outside_mesh.tri_verts):
        exterior=trimesh.Trimesh(np.asarray(outside_mesh.vert_properties)[:,:3],
                               np.asarray(outside_mesh.tri_verts),process=False)
        spans=[float(np.linalg.norm(component.extents)) for component in exterior.split(only_watertight=False)]
        report['outside_source']['component_spans_mm']=spans
        if spans and max(spans)>5.:raise ValueError('Outside-source patch exceeds local span budget')
        samples=np.asarray(outside_mesh.vert_properties)[:,:3]
        _,distance,_=trimesh.proximity.closest_point_naive(source,samples)
        report['outside_source']['maximum_sample_distance_mm']=float(distance.max())
        if distance.max()>.05:raise ValueError('Backing protrusion exceeds surface tolerance')
    female=full-male
    raw=female.to_mesh64()
    parent=trimesh.Trimesh(np.asarray(raw.vert_properties)[:,:3],np.asarray(raw.tri_verts),process=False)
    child,report['scaling']=scale_finished_insert(child,args.scale)
    child,report['seating']=seat_insert_outward(child,parent,axis)
    recovery=args.output.with_name(args.output.stem+'_candidate.npz')
    np.savez_compressed(recovery,child_vertices=child.vertices,child_faces=child.faces,
                        parent_vertices=parent.vertices,parent_faces=parent.faces,
                        source_face_ids=ids[mask])
    report['recovery']={'path':str(recovery.resolve()),'validated_deliverable':False}
    report['visibility']=assess_insert_visibility(patch,child,parent,reference=source)
    if report['visibility'].get('covered_fraction',0)>.01 and report['visibility'].get('covered_area_estimate_mm2',0)>args.micro_area:
        raise ValueError('The seated insert remains visibly covered')
    parts=[part_record(insert_entry,child,'insert',1,'P02'),part_record(parent_entry,parent,'body',2)]
    parts[0]['annotation'].update(post_split_uniform_scaling=report['scaling'],post_fit_seating=report['seating'])
    parts[0]['source_surface_face_count']=len(patch.faces)
    parts[1]['source_surface_face_count']=0  # conservatively audit the complete recut body
    report['topology']=[validate_mesh_in_memory(part['mesh']) for part in parts]
    if any(not r['watertight'] or not r['winding_consistent'] for r in report['topology']):
        raise ValueError('Repaired parts failed topology validation')
    labels=np.full(len(source.faces),2);labels[ids[mask]]=1
    report['multiview']=validate_multiview_visual_consistency(source.vertices,source.faces,labels,parts)
    if not report['multiview']['valid']:raise ValueError('Repaired assembly failed multiview validation')
    palette=source_entry['palette']
    common={'source_filament_colors':palette,'source_project_settings':settings,'source_application':application}
    temporary=args.output.with_suffix('.3mf.tmp')
    export_colored_parts_3mf(temporary,parts,args.output.stem,**common)
    report['package']=validate_colored_parts_3mf(temporary,parts,**common)
    if not report['package']['valid']:raise ValueError('Repaired package failed reload validation')
    reloaded=load_colored_mesh_objects_3mf(temporary)
    overlap=abs(float((_manifold64(reloaded[0]['mesh'])^_manifold64(reloaded[1]['mesh'])).volume()))
    report['reloaded_intersection_mm3']=overlap
    if overlap>1e-8:raise ValueError('Serialization reintroduced assembly interference')
    temporary.replace(args.output)
    report['output']=str(args.output.resolve())
    report['standalone_parts']=[]
    for part in parts:
        output=args.output.with_name(args.output.stem+'_'+part['part_id']+'.3mf')
        tmp=output.with_suffix('.3mf.tmp')
        export_colored_parts_3mf(tmp,[part],part['part_id'],**common)
        check=validate_colored_parts_3mf(tmp,[part],**common)
        if not check['valid']:raise ValueError('Standalone repaired part failed reload validation')
        tmp.replace(output);report['standalone_parts'].append(str(output.resolve()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','insert','parent','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--max-depth',type=float,default=3.)
    parser.add_argument('--scale',type=float,default=.99)
    parser.add_argument('--micro-area',type=float,default=1.)
    parser.add_argument('--taper-slope',type=float,default=1.)
    args=parser.parse_args()
    if args.output.exists():parser.error('Output already exists; use a new repair namespace')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    report={'source_sha256':hashlib.sha256(args.source.read_bytes()).hexdigest(),'source_vertices_moved':0}
    start=time.perf_counter()
    try:
        run(args,report);report['status']='validated'
    except Exception as exc:
        report.update(status='failed',error=str(exc))
        if isinstance(exc,SeatingError):report['seating_failure']=exc.record
        raise
    finally:
        report['seconds']=time.perf_counter()-start
        args.output.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({key:report.get(key) for key in ('status','output','seconds','error')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
