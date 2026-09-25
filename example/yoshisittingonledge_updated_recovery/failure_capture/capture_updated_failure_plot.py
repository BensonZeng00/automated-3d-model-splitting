import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
sys.path.insert(0, str(Path(r'E:\automated-3d-model-splitting\skills\automated-3d-model-splitting\scripts')))
from split3mf.layer_surface_plan import LayerSurfacePlan
from split3mf.planar_arc import PlanarArcError

original_apply = LayerSurfacePlan.apply
out_dir = Path(r'E:\automated-3d-model-splitting\example\yoshisittingonledge_updated_recovery\failure_capture')
out_dir.mkdir(parents=True, exist_ok=True)

def diagnostic_apply(self, vertices, faces, global_ids, loops):
    try:
        return original_apply(self, vertices, faces, global_ids, loops)
    except PlanarArcError as exc:
        if 'Unplanned boundary loop in shared layer surface' not in str(exc):
            raise
        ids = np.asarray(global_ids, dtype=np.int64)
        loops_arr = [np.asarray(loop, dtype=np.int64) for loop in loops]
        missed = []
        record_keys = set(self.records)
        for li, loop in enumerate(loops_arr):
            key = __import__('split3mf.shared_seam_topology', fromlist=['canonical_loop']).canonical_loop(ids[loop])
            if key not in record_keys:
                missed.append((li, ids[loop]))
        if not missed:
            raise
        li, bad_ids = missed[0]
        bad = self.vertices[bad_ids]
        center = bad.mean(axis=0)
        planned = []
        for key in self.records:
            p = self.vertices[np.asarray(key, dtype=np.int64)]
            planned.append((float(np.linalg.norm(p.mean(axis=0)-center)), p, key))
        planned.sort(key=lambda item:item[0])
        nearest = planned[:4]
        allpts = np.vstack([bad] + [item[1] for item in nearest])
        origin = allpts.mean(axis=0)
        _, _, basis = np.linalg.svd(allpts-origin, full_matrices=False)
        proj = (allpts-origin) @ basis[:2].T
        bad2 = (bad-origin) @ basis[:2].T
        fig = plt.figure(figsize=(13, 6.5))
        ax = fig.add_subplot(121, projection='3d')
        # Local source surface context around the failed loop.
        f = np.asarray(self.faces, dtype=np.int64)
        tri = np.asarray(self.vertices[f])
        span = max(float(np.ptp(bad, axis=0).max()), 1.0)
        pad = max(1.5, span * 0.20)
        lo, hi = bad.min(axis=0)-pad, bad.max(axis=0)+pad
        cent = tri.mean(axis=1)
        keep = np.flatnonzero(np.all((cent >= lo) & (cent <= hi), axis=1))
        if len(keep) > 12000:
            keep = keep[np.linspace(0, len(keep)-1, 12000, dtype=np.int64)]
        if len(keep):
            ax.add_collection3d(Poly3DCollection(tri[keep], facecolors='#cbd1d8', edgecolors='none', alpha=.6))
        ax.plot(*np.vstack([bad,bad[:1]]).T, color='#e51f58', lw=2.2, label='UNPLANNED loop')
        for j, (_, p, _) in enumerate(nearest):
            ax.plot(*np.vstack([p,p[:1]]).T, color=['#009e73','#0072b2','#e69f00','#cc79a7'][j], lw=1.1, alpha=.9, label=f'planned ring {j+1}')
        lo3, hi3 = allpts.min(axis=0), allpts.max(axis=0)
        mid=(lo3+hi3)/2; rad=max(float(np.ptp(allpts,axis=0).max())*.58,.5)
        ax.set(xlim=(mid[0]-rad,mid[0]+rad),ylim=(mid[1]-rad,mid[1]+rad),zlim=(mid[2]-rad,mid[2]+rad))
        ax.set_box_aspect((1,1,1)); ax.view_init(elev=24,azim=-58); ax.set_title(f'3D local context / loop {li}')
        ax.legend(loc='upper left',fontsize=8)
        ax2=fig.add_subplot(122)
        for j, (_, p, _) in enumerate(nearest):
            q=(p-origin) @ basis[:2].T
            ax2.plot(*np.vstack([q,q[:1]]).T,color=['#009e73','#0072b2','#e69f00','#cc79a7'][j],lw=1.2,label=f'planned ring {j+1}')
        ax2.plot(*np.vstack([bad2,bad2[:1]]).T,color='#e51f58',lw=2.2,label='UNPLANNED loop')
        ax2.scatter(*bad2.T,s=4,color='#e51f58')
        ax2.set_aspect('equal'); ax2.grid(alpha=.25); ax2.set_xlabel('local projection 1 (mm)'); ax2.set_ylabel('local projection 2 (mm)')
        ax2.set_title('Loop comparison (PCA projection)'); ax2.legend(fontsize=8)
        fig.suptitle('Preflight failure: shared layer boundary loop has no matching planned record\nRed = unmatched boundary; colored = nearest planned loops')
        fig.tight_layout()
        path=out_dir/'failure_unplanned_boundary_loop.png'
        fig.savefig(path,dpi=170); plt.close(fig)
        np.savez_compressed(out_dir/'failure_unplanned_boundary_loop.npz', vertices=self.vertices, faces=self.faces, failed_loop_ids=bad_ids, failed_loop_points=bad, nearest_loop_points=np.array([x[1] for x in nearest],dtype=object))
        print(f'FAILURE_DIAGNOSTIC_IMAGE={path}', flush=True)
        print(f'FAILURE_LOOP_INDEX={li}; VERTICES={len(bad_ids)}; PLANNED_RECORDS={len(self.records)}', flush=True)
        raise
LayerSurfacePlan.apply = diagnostic_apply
sys.path.insert(0, str(Path(r'E:\automated-3d-model-splitting\skills\automated-3d-model-splitting\scripts')))
from split3mf.cli import main
sys.argv = ['split_painted_3mf.py','--input',r'D:\3Dmodel\yoshisittingonledge.3mf','--output',str(out_dir/'yoshisittingonledge_split_parts.3mf'),'--region-review-json',str(out_dir.parent.parent/'yoshisittingonledge_region_review_updated'/'user_decisions.json'),'--recovery-dir',str(out_dir/'recovery'),'--connector-slope-validation','advisory','--surface-band-validation','advisory','--connector-surface-validation','advisory','--diagnostic-preview']
main()
