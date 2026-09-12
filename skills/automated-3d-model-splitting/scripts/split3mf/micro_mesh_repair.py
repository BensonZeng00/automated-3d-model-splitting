"""Close print-scale defects using coherent directed boundary cycles."""
import numpy as np
import trimesh
from .print_tolerance import current
from .winding import fix_winding_indexed


def edge_topology(faces):
    faces=np.asarray(faces,dtype=np.int64)
    edges=np.vstack((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]))
    owners=np.tile(np.arange(len(faces)),3)
    unique,inverse,counts=np.unique(np.sort(edges,axis=1),axis=0,return_inverse=True,return_counts=True)
    balance=np.bincount(inverse,weights=np.where(edges[:,0]<edges[:,1],1,-1))
    order=np.argsort(inverse,kind='stable')
    starts=np.r_[0,np.cumsum(counts)]
    return edges,owners,unique,inverse,counts,balance,order,starts


def directed_cycles(edges):
    adjacency={}
    for a,b in edges:
        adjacency.setdefault(int(a),[]).append(int(b))
    cycles=[]
    for start in sorted(adjacency):
        if not adjacency[start]:
            continue
        path=[start];positions={start:0}
        while adjacency.get(path[-1]):
            nxt=adjacency[path[-1]].pop()
            if nxt in positions:
                offset=positions[nxt]
                cycle=path[offset:]
                if len(cycle)>=3:
                    cycles.append(cycle)
                for value in path[offset+1:]:
                    del positions[value]
                path=path[:offset+1]
            else:
                positions[nxt]=len(path);path.append(nxt)
    return cycles


def repair_micro_mesh(mesh):
    limit=current().micro_area_mm2
    if limit<=0:
        return mesh,dict(applied=False)
    result=mesh.copy();removed_area=0.;removed_count=0
    for _ in range(3):
        fix_winding_indexed(result)
        e,owner,u,inv,count,balance,order,starts=edge_topology(result.faces)
        conflicts=np.flatnonzero((count==2)&(np.abs(balance)==2))
        if not len(conflicts):
            break
        area=result.area_faces;remove=set()
        for edge in conflicts:
            candidates=owner[order[starts[edge]:starts[edge+1]]]
            face=int(candidates[np.argmin(area[candidates])])
            if face not in remove and removed_area+float(area[face])<=limit:
                remove.add(face);removed_area+=float(area[face])
        if not remove:
            break
        mask=np.ones(len(result.faces),dtype=bool);mask[list(remove)]=False
        removed_count+=len(remove);result.update_faces(mask)
    e,owner,u,inv,count,balance,order,starts=edge_topology(result.faces)
    positions=np.flatnonzero(count[inv]==1)
    lookup={tuple(map(int,e[pos])):int(pos) for pos in positions}
    loops=directed_cycles(e[positions])
    vertices=result.vertices.tolist();faces=result.faces.tolist()
    filled=[];skipped=0
    for loop in loops:
        keys=list(zip(loop,np.roll(loop,-1).astype(int).tolist()))
        if not all(key in lookup for key in keys):
            skipped+=1;continue
        positions=[lookup[key] for key in keys]
        rim=np.asarray(result.vertices)[loop]
        normal=result.face_normals[owner[positions]].sum(axis=0)
        length=np.linalg.norm(normal)
        if length<=1e-12:
            normal=np.array([0.,0.,1.]);length=1.
        offset=min(0.005,current().surface_distance_mm)
        center=rim.mean(axis=0)-normal/length*offset
        new=[[int(e[pos,1]),int(e[pos,0]),len(vertices)] for pos in positions]
        triangles=np.asarray([[result.vertices[a],result.vertices[b],center] for a,b,_ in new])
        areas=trimesh.triangles.area(triangles)
        span=float(np.linalg.norm(np.ptp(rim,axis=0)))
        if areas.sum()>limit or span>5.0 or areas.min(initial=1.)<=1e-15:
            skipped+=1;continue
        vertices.append(center.tolist());faces.extend(new)
        filled.append(dict(area_mm2=float(areas.sum()),span_mm=span,center_offset_mm=offset))
    output=trimesh.Trimesh(vertices=np.asarray(vertices),faces=np.asarray(faces),
                           process=False,metadata=mesh.metadata.copy())
    fix_winding_indexed(output)
    record=dict(applied=bool(removed_count or filled),removed_faces=removed_count,
                removed_area_mm2=removed_area,closed_micro_loops=len(filled),
                skipped_loops=skipped,closures=filled,
                method='directed_boundary_micro_closure')
    output.metadata['print_micro_mesh_repair']=record
    return output,record
