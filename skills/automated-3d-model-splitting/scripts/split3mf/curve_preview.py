"""Persist numerical topology-change proposals without mutating a source mesh."""
from __future__ import annotations
import json
import os
from pathlib import Path
import tempfile
import numpy as np


def export_curve_review(source, target, proposal, basis, directory, *, record=None):
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'split3mf-matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    origin, u, v, normal = basis
    details = dict(proposal.record, **(record or {}))
    weights = proposal.source_weights
    mapping = {} if weights is None else dict(
        weights_data=weights.data, weights_indices=weights.indices,
        weights_indptr=weights.indptr, weights_shape=weights.shape)
    np.savez_compressed(directory/'clear_curve.npz', source=source, old_target=target,
                        candidate=proposal.points, origin=origin, u=u, v=v,
                        normal=normal, **mapping)
    details['mapping_scope'] = 'candidate vertices -> old target curve vertices; NOT source mesh vertex IDs'
    (directory/'clear_curve.json').write_text(json.dumps(details, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
    plane = np.column_stack((u, v))
    for ax, points, title in zip(axes.flat, (target,proposal.points,target,proposal.points),
            ('Before: actual ordered target', 'After: single main contour proposal',
             'Before: upper boundary detail', 'After: upper boundary detail')):
        for values, color, label, lw in ((source,'#2563eb','Original seam',0.9),
                                        (points,'#ea580c',f'Actual target ({len(points)} vertices)',2)):
            xy = (values-origin) @ plane
            xy = np.vstack((xy,xy[:1]))
            ax.plot(*xy.T, color=color, linewidth=lw, label=label)
        ax.set(title=title, xlabel='Stable-plane U (mm)', ylabel='Stable-plane V (mm)')
        ax.set_aspect('equal'); ax.grid(alpha=.25); ax.legend(fontsize=8)
    xy = (target-origin) @ plane
    for ax in axes[1]:
        ax.set_xlim(xy[:,0].min()-.2, xy[:,0].max()+.2)
        ax.set_ylim(float(np.median(xy[:,1])),xy[:,1].max()+.2)
    fig.suptitle('Orange boundary: remove crossing lobes, not just smooth them\n'
                 'Proposal only - source mesh unchanged; surface remeshing not yet performed')
    fig.savefig(directory/'clear_curve_comparison.png',dpi=170)
    plt.close(fig)
    return details
