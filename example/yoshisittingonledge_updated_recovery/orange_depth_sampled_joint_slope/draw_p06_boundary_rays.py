from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "skills" / "automated-3d-model-splitting" / "scripts"))
from split3mf.common import load_core_dependencies, COLOR_CATALOG
load_core_dependencies()
import trimesh
from split3mf.project import parse_3mf_model, build_color_info_map, material_connectivity_labels
from split3mf.recognition import (
    exterior_visible_face_mask, recognition_colors_from_exterior,
    connected_components_by_color, material_identity,
    _merge_partitioned_groups_into_components,
)
from split3mf.hidden_interface import boundary_screening_indices
from split3mf.inward import ParentThicknessProbe

source = Path(r"D:\3Dmodel\yoshisittingonledge.3mf")
out_dir = Path(__file__).resolve().parent
image_path = out_dir / "P06_boundary_sampled_rays.png"
json_path = out_dir / "P06_boundary_sampled_rays.json"
vertices, faces, colors, settings = parse_3mf_model(source)
info, order = build_color_info_map(settings, colors)
COLOR_CATALOG.replace(info, order)
visible, visibility = exterior_visible_face_mask(
    vertices, faces, view_count=32, depth_map_resolution=768, depth_tolerance_mm=0.08
)
recognition_tokens, _ = recognition_colors_from_exterior(colors, visible, faces=faces)
connectivity, _ = material_connectivity_labels(recognition_tokens)
groups = connected_components_by_color(faces, connectivity)
if len(groups) > 1000:
    largest_by_material = {}
    for group in groups:
        identity = material_identity(str(recognition_tokens[int(group[0])]))
        if identity not in largest_by_material or len(group) > len(largest_by_material[identity]):
            largest_by_material[identity] = group
    anchors = [group for group in groups if len(group) > 999 or any(group is g for g in largest_by_material.values())]
    fragments = [group for group in groups if not any(group is anchor for anchor in anchors)]
    components, _, _ = _merge_partitioned_groups_into_components(
        vertices, faces, connectivity, anchors, fragments, display_colors=recognition_tokens
    )
else:
    from split3mf.recognition import make_components
    components, _ = make_components(vertices, faces, connectivity, groups, min_faces=1, display_colors=recognition_tokens)

# Match the P06 recognition record for the supplied model version.
p06 = min(components, key=lambda c: abs(c.face_count - 90734) + np.linalg.norm(c.bbox_min - [155.146045,177.384399,0.000002]) * 1000)
if p06.face_count < 50000:
    raise RuntimeError(f"P06 selection did not match expected component: {p06.face_count} faces")
owner = np.zeros(len(faces), dtype=np.int8)
owner[np.asarray(p06.global_faces, dtype=np.int64)] = 6
p01_candidates = [c for c in components if c is not p06 and c.face_count > 200000]
if not p01_candidates:
    raise RuntimeError("Could not identify root parent P01")
p01 = max(p01_candidates, key=lambda c: c.face_count)
owner[np.asarray(p01.global_faces, dtype=np.int64)] = 1
mesh_all = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
adjacency = np.asarray(mesh_all.face_adjacency, dtype=np.int64)
pair_mask = ((owner[adjacency[:, 0]] == 6) & (owner[adjacency[:, 1]] == 1)) | ((owner[adjacency[:, 0]] == 1) & (owner[adjacency[:, 1]] == 6))
pairs = adjacency[pair_mask]
if not len(pairs):
    raise RuntimeError("P06/P01 contact was not found in source-face adjacency")
tri_a = faces[pairs[:, 0]]
tri_b = faces[pairs[:, 1]]
common = np.empty((len(pairs), 2), dtype=np.int64)
for i, (a, b) in enumerate(zip(tri_a, tri_b)):
    overlap = np.intersect1d(a, b, assume_unique=False)
    if len(overlap) != 2:
        raise RuntimeError("Unexpected contact edge cardinality")
    common[i] = overlap
edges = np.sort(common, axis=1)
edges = np.unique(edges, axis=0)
# Connected edge islands are retained and sampled together.
parent = {}
def find(x):
    parent.setdefault(int(x), int(x))
    if parent[int(x)] != int(x): parent[int(x)] = find(parent[int(x)])
    return parent[int(x)]
def union(a,b):
    ra,rb=find(a),find(b)
    if ra!=rb: parent[rb]=ra
for a,b in edges: union(a,b)
components_by_root={}
for edge in edges:
    root=find(int(edge[0])); components_by_root.setdefault(root, []).append(edge)
all_contact_edges=np.asarray(edges,dtype=np.int64)
edge_lengths=np.linalg.norm(vertices[all_contact_edges[:,1]]-vertices[all_contact_edges[:,0]],axis=1)
sample_count=min(768,max(1,len(all_contact_edges)))
targets=(np.arange(sample_count,dtype=np.float64)+0.5)*(float(edge_lengths.sum())/sample_count)
cumulative=np.cumsum(edge_lengths)
sample_edge_slots=np.searchsorted(cumulative,targets,side="left").clip(0,len(all_contact_edges)-1)
sample_ratios=np.divide(
    targets-(cumulative[sample_edge_slots]-edge_lengths[sample_edge_slots]),
    edge_lengths[sample_edge_slots],
    out=np.zeros(sample_count,dtype=np.float64),
    where=edge_lengths[sample_edge_slots]>1e-12,
)
sample_ratios=np.clip(sample_ratios,0.0,1.0)
sample_edges=all_contact_edges[sample_edge_slots]
samples=(vertices[sample_edges[:,0]]*(1.0-sample_ratios[:,None])+
         vertices[sample_edges[:,1]]*sample_ratios[:,None])

parent_face_ids=np.asarray(p01.global_faces,dtype=np.int64)
parent_faces=faces[parent_face_ids]
parent_mesh=trimesh.Trimesh(vertices=vertices, faces=parent_faces, process=False)
probe=ParentThicknessProbe(vertices, parent_faces, triangle_source_face_indices=parent_face_ids)
# Contact-face outward normals provide the direction from the P06 surface into its parent.
p06_face_ids=pairs[:,0].copy()
swap=(owner[p06_face_ids]!=6)
p06_face_ids[swap]=pairs[swap,1]
contact_normals=mesh_all.face_normals[p06_face_ids]
# Sample per-ray direction from the nearest contact-edge midpoint normal.
ray_count=9
ray_slots=np.linspace(0,len(samples)-1,num=min(ray_count,len(samples)),dtype=np.int64)
origins=samples[ray_slots]
all_edge_mids=vertices[all_contact_edges].mean(axis=1)
edge_face_normals=mesh_all.face_normals[np.where(owner[pairs[:,0]]==6,pairs[:,0],pairs[:,1])]
# Fit the interface plane and find its projected center for centerward rays.
interface_center=vertices[all_contact_edges].reshape((-1,3)).mean(axis=0)
centered=vertices[all_contact_edges].reshape((-1,3))-interface_center
_,_,plane_axes=np.linalg.svd(centered,full_matrices=False)
interface_plane_normal=plane_axes[-1]
# Each sample gets a 30-75 degree fan, measured from its local inward normal.
fan_angles=np.linspace(30.0,75.0,10)
fan_directions=[]
for origin in origins:
    nearest=int(np.argmin(np.linalg.norm(all_edge_mids-origin[None,:],axis=1)))
    inward= edge_face_normals[nearest].astype(np.float64)
    if np.dot(inward,p01.center-origin)<0:
        inward=-inward
    inward/=max(np.linalg.norm(inward),1e-12)
    toward_center=interface_center-origin
    toward_center-=float(np.dot(toward_center,interface_plane_normal))*interface_plane_normal
    toward_center-=float(np.dot(toward_center,inward))*inward
    if np.linalg.norm(toward_center)<=1e-10:
        toward_center=interface_center-origin
        toward_center-=float(np.dot(toward_center,inward))*inward
    toward_center/=max(np.linalg.norm(toward_center),1e-12)
    radians=np.deg2rad(fan_angles)
    fan=np.cos(radians)[:,None]*inward[None,:]+np.sin(radians)[:,None]*toward_center[None,:]
    fan_directions.append(fan)
fan_directions=np.asarray(fan_directions,dtype=np.float64)
search_limit=10.0
ray_start=np.repeat(origins[:,None,:],len(fan_angles),axis=1)+fan_directions*0.03
hits=probe.first_hit_distances(ray_start.reshape((-1,3)),fan_directions.reshape((-1,3)),search_limit).reshape((len(origins),len(fan_angles)))
hit_mask=hits<search_limit+0.05-1e-9
safe_depths=np.where(hit_mask,np.maximum(0.0,hits+0.03-0.05),search_limit)
best_indices=np.argmax(safe_depths,axis=1)
directions=fan_directions[np.arange(len(origins)),best_indices]
best_hits=hits[np.arange(len(origins)),best_indices]
best_is_hit=hit_mask[np.arange(len(origins)),best_indices]
records=[]
for i,(origin,direction,hit,was_hit) in enumerate(zip(origins,directions,best_hits,best_is_hit)):
    miss=not bool(was_hit)
    depth=search_limit if miss else max(0.0,float(hit)+0.03-0.05)
    endpoint=origin+direction*(search_limit if miss else float(hit)+0.03)
    records.append({
        "sample_slot":int(ray_slots[i]),
        "origin_xyz_mm":[float(x) for x in origin],
        "direction_unit": [float(x) for x in direction],
        "angle_to_inward_normal_degrees":float(fan_angles[best_indices[i]]),
        "hit":not miss,
        "safe_depth_mm":depth,
        "surface_hit_depth_mm":None if miss else float(hit+0.03),
        "endpoint_xyz_mm":[float(x) for x in endpoint],
        "search_limit_mm":search_limit,
        "unhit_means_safe_to_limit":bool(miss),
    })
# Context triangles: decimated face subsets, rendering only for orientation.
rng=np.random.default_rng(20260925)
def face_context(ids, maximum):
    ids=np.asarray(ids,dtype=np.int64)
    if len(ids)>maximum: ids=np.sort(rng.choice(ids,size=maximum,replace=False))
    return vertices[faces[ids]]
p06_tris=face_context(p06.global_faces, 14000)
p01_tris=face_context(p01.global_faces, 14000)
fig=plt.figure(figsize=(17,11),facecolor="white")
ax=fig.add_subplot(221,projection="3d")
for tris,color,alpha in ((p01_tris,"#93a8ba",0.08),(p06_tris,"#2786d5",0.15)):
    ax.add_collection3d(Poly3DCollection(tris,facecolor=color,edgecolor="none",alpha=alpha))
line_segments=vertices[all_contact_edges]
ax.add_collection3d(Line3DCollection(line_segments,colors="#e52432",linewidths=0.7,alpha=0.9))
ax.scatter(*samples.T,s=5,c="#ffb000",depthshade=False,label=f"equal-arc samples ({len(samples)})")
# Show the tested 30-75 degree direction fan at the first selected sample.
fan_origin=origins[0]
for angle,direction in zip(fan_angles,fan_directions[0]):
    ax.plot(*np.vstack((fan_origin,fan_origin+direction*3.0)).T,color="#707780",linestyle=":",linewidth=0.7,alpha=0.65)
for record in records:
    origin=np.asarray(record["origin_xyz_mm"]); end=np.asarray(record["endpoint_xyz_mm"])
    style="-" if record["hit"] else "--"
    ax.plot(*np.vstack((origin,end)).T,color="#ff6b00",linestyle=style,linewidth=2.0)
    ax.scatter(*end,s=18,c=("#16a34a" if record["hit"] else "#dc2626"),depthshade=False)
center=(p06.bbox_min+p06.bbox_max)/2
span=float(np.max(p06.bbox_max-p06.bbox_min))*0.65
ax.set_xlim(center[0]-span,center[0]+span); ax.set_ylim(center[1]-span,center[1]+span); ax.set_zlim(center[2]-span,center[2]+span)
ax.set_box_aspect((1,1,1)); ax.set_title("P06 / P01 interface + actual sampled depth rays")
ax.set_xlabel("X mm"); ax.set_ylabel("Y mm"); ax.set_zlabel("Z mm"); ax.legend(loc="upper left")

views=[("XY",0,1,2),("XZ",0,2,1),("YZ",1,2,0)]
for subplot,(title,a,b,hidden) in enumerate(views,start=2):
    ax2=fig.add_subplot(2,2,subplot)
    # faint outline of projected P06 sample cloud bounds and full interface.
    ax2.scatter(vertices[p06.global_faces[:0]].T[0] if False else samples[:,a], samples[:,b],s=2,c="#2786d5",alpha=0.20)
    projected_segments=vertices[all_contact_edges][:,:,[a,b]]
    ax2.add_collection(LineCollection(projected_segments,colors="#e52432",linewidths=0.7))
    ax2.scatter(samples[:,a],samples[:,b],s=6,c="#ffb000",label="equal-arc samples")
    for idx,record in enumerate(records):
        o=np.asarray(record["origin_xyz_mm"]); e=np.asarray(record["endpoint_xyz_mm"])
        ax2.plot([o[a],e[a]],[o[b],e[b]],color="#ff6b00",ls=("-" if record["hit"] else "--"),lw=1.2)
        ax2.text(o[a],o[b],str(idx+1),fontsize=8,color="#a04400")
    ax2.set_aspect("equal",adjustable="datalim"); ax2.grid(True,alpha=.2)
    ax2.set_title(f"{title} projection"); ax2.set_xlabel(f"{'XYZ'[a]} (mm)"); ax2.set_ylabel(f"{'XYZ'[b]} (mm)")
fig.suptitle(
    f"P06 interface boundary | P06 faces={p06.face_count:,}, P01 faces={p01.face_count:,} | "
    f"contact edges={len(all_contact_edges):,}, contact length={edge_lengths.sum():.1f} mm, equal-arc samples={len(samples)}\n"
    "Rays combine centerward direction + inward normal (30-75 deg from normal); no hit by 10 mm means 10 mm safe",
    fontsize=14,
)
fig.tight_layout(rect=(0,0,1,.92))
fig.savefig(image_path,dpi=180,bbox_inches="tight")
plt.close(fig)
result={
    "source_3mf":str(source),"recognition_profile":"exterior-visible/32-view/768-depth-map",
    "visibility_faces":int(visibility["visible_faces"]),
    "P06":{"face_count":int(p06.face_count),"bbox_min":p06.bbox_min.tolist(),"bbox_max":p06.bbox_max.tolist()},
    "P01":{"face_count":int(p01.face_count),"bbox_min":p01.bbox_min.tolist(),"bbox_max":p01.bbox_max.tolist()},
    "interface":{"P06_P01_face_adjacency_pairs":int(len(pairs)),"contact_edges":int(len(all_contact_edges)),"contact_edge_components":int(len(components_by_root)),"contact_edge_length_sum_mm":float(edge_lengths.sum()),"equal_arc_sample_count":int(len(samples)),"sampling_policy":"equal arc-length samples along all P06/P01 contact edges; maximum 768"},
    "ray_direction_source":"slerp-like blend of projected planar-center direction and local P06-to-P01 inward normal",
    "angle_to_inward_normal_range_degrees":[30.0,75.0],"ray_start_offset_mm":0.03,
    "parent_bbox_diagonal_search_limit_mm":float(np.linalg.norm(p01.bbox_max-p01.bbox_min)),
    "search_limit_mm":search_limit,"unhit_policy":"safe_to_search_limit","rays":records,
    "note":"Rendered all P06/P01 contact edges and sparse surface context; equal-arc samples are measured along the combined physical edge length.",
    "image":str(image_path),
}
json_path.write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps({"image":str(image_path),"json":str(json_path),"P06_faces":int(p06.face_count),"P01_faces":int(p01.face_count),"contact_edges":int(len(all_contact_edges)),"contact_edge_components":int(len(components_by_root)),"samples":int(len(samples)),"rays":records},indent=2))
