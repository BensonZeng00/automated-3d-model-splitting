"""Rebuild hidden backing from an oriented source patch, retaining its rim."""
import numpy as np
import trimesh
from .rim_chord_repair import _split_face

def first_exit(triangles, points, axis, epsilon=1e-7):
    a=triangles[:,0]; e=triangles[:,1]-a; g=triangles[:,2]-a
    h=np.cross(axis,g); det=np.einsum('ij,ij->i',e,h)
    valid=np.abs(det)>1e-12
    # Only outward exits of the intact parent solid can certify material depth.
    valid &= np.einsum('ij,j->i',np.cross(e,g),axis)>1e-12
    a,e,g,h,det=a[valid],e[valid],g[valid],h[valid],det[valid]
    inv=1/det;result=np.full(len(points),np.nan)
    for i,p in enumerate(points):
        delta=p-a;u=np.einsum('ij,ij->i',delta,h)*inv;q=np.cross(delta,e)
        v=(q@axis)*inv;d=np.einsum('ij,ij->i',g,q)*inv
        inside=(u>=-1e-9)&(v>=-1e-9)&(u+v<=1+1e-9)&(d>epsilon)
        if inside.any():result[i]=d[inside].min()
    return result

def segment_distance(points, segments):
    a=segments[:,0]; d=segments[:,1]-a; den=np.sum(d*d,axis=1)
    result=[]
    for p in points:
        t=np.clip(np.sum((p-a)*d,axis=1)/np.maximum(den,1e-24),0,1)
        result.append(np.linalg.norm(p-a-t[:,None]*d,axis=1).min())
    return np.array(result)

def build(patch, parent, axis, preferred_depth=3., taper_slope=1., *, direction_mode='axis'):
    if not np.isfinite(preferred_depth) or not 0 < preferred_depth <= 10:
        raise ValueError('Backing depth must be finite and within (0, 10] mm')
    if not np.isfinite(taper_slope) or not .577 <= taper_slope <= 3.732:
        raise ValueError('Backing taper slope must stay within the 30-75 degree design range')
    if np.shape(axis)!=(3,) or not np.isfinite(axis).all() or np.linalg.norm(axis)<1e-12:
        raise ValueError('Backing axis must be a finite nonzero vector')
    if not len(patch.faces) or not patch.is_winding_consistent:
        raise ValueError('A consistently oriented source patch is required')
    axis=np.asarray(axis);axis=axis/np.linalg.norm(axis)
    vertices=patch.vertices.copy().tolist(); front=patch.faces.tolist()
    counts=np.bincount(patch.edges_unique_inverse)
    rim_edges=patch.edges_unique[counts==1]
    if not len(rim_edges) or np.any(counts>2):
        raise ValueError('Source patch must have a manifold open rim')
    rim_ids=set(rim_edges.ravel().tolist())
    midpoint={}
    for edge,count in zip(patch.edges_unique,counts):
        if count==2 and all(int(i) in rim_ids for i in edge):
            midpoint[tuple(edge)]=len(vertices)
            vertices.append(patch.vertices[edge].mean(axis=0).tolist())
    back=[]
    back_owners=[]
    for owner,face in enumerate(front):
        split=_split_face(face,midpoint)
        for f in split:
            if all(i in rim_ids for i in f):
                center=len(vertices);vertices.append(np.array(vertices)[f].mean(axis=0).tolist())
                back.extend([[f[0],f[1],center],[f[1],f[2],center],[f[2],f[0],center]])
                back_owners.extend([owner]*3)
            else:
                back.append(f)
                back_owners.append(owner)
    points=np.asarray(vertices)
    seed=np.eye(3)[np.argmin(np.abs(axis))];u=np.cross(axis,seed);u/=np.linalg.norm(u)
    basis=np.column_stack((u,np.cross(axis,u)))
    if direction_mode not in ('axis', 'local-normal'):
        raise ValueError('Unknown source-following backing direction mode')
    if direction_mode == 'local-normal':
        from .local_ray_probe import LocalRayProbe
        back_surface=trimesh.Trimesh(points,back,process=False)
        directions=-np.asarray(back_surface.vertex_normals)
        distance=segment_distance(points,patch.vertices[rim_edges])
    else:
        directions=np.tile(axis,(len(points),1))
        distance=segment_distance(points@basis,patch.vertices[rim_edges]@basis)
    active=np.array([i for i in range(len(points)) if i not in rim_ids])
    if direction_mode == 'local-normal':
        exits,_=LocalRayProbe(parent).exits(points[active],directions[active],
                                           float(np.linalg.norm(parent.extents)+1))
        safety=exits-.05
    else:
        safety=first_exit(parent.triangles,points[active],axis)-.05
    if not np.isfinite(safety).all() or (safety<=0).any():
        raise ValueError(f'Invalid parent safety: {np.nanmin(safety)}; {np.sum(~np.isfinite(safety))} missing')
    depths=np.minimum(np.minimum(distance[active]*taper_slope,preferred_depth),safety)
    if (depths<=1e-8).any():
        raise ValueError(f'Backing has unresolved zero-width interior: {depths.min()}')
    mapping={int(i):int(i) for i in rim_ids}
    for i,depth in zip(active,depths):
        mapping[int(i)]=len(vertices);vertices.append((points[i]+directions[i]*depth).tolist())
    faces=front+[[mapping[i] for i in reversed(f)] for f in back]
    mesh=trimesh.Trimesh(vertices,faces,process=False)
    mesh.remove_unreferenced_vertices()
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume<=0:
        raise ValueError('Source-following backing failed solid topology validation')
    if np.any(mesh.area_faces[len(front):]<=1e-12):
        raise ValueError('Source-following backing generated degenerate faces')
    return mesh, {'source_faces':len(front),'back_faces':len(back),'boundary_vertices':len(rim_ids),
                  'minimum_interior_depth_mm':float(depths.min()),'maximum_depth_mm':float(depths.max()),
                  'minimum_parent_reserve_mm':float(np.min(safety+.05-depths)),
                  'taper_slope':taper_slope,'preferred_depth_mm':preferred_depth,
                  'direction_mode':direction_mode,
                  'back_face_source_indices':back_owners,
                  'coowned_source_chords_split':len(midpoint)}
