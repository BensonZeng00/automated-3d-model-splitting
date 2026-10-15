"""Bounded offline review drawings; full-mesh acceptance stays elsewhere."""
from pathlib import Path
import os
import numpy as np

MAX_PREVIEW_FACES = 50_000
MAX_PREVIEW_SEGMENTS = 50_000


def _even_sample(ids, limit):
    ids = np.asarray(ids, dtype=np.int64)
    if len(ids) <= limit:
        return ids
    positions = np.linspace(0, len(ids) - 1, num=limit, dtype=np.int64)
    return ids[positions]


def render_boundary_review(graph, labels, assessment, candidates, directory, display_colors=None):
    os.environ.setdefault('MPLCONFIGDIR', str(Path(directory) / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

    vertices, faces = graph.vertices, graph.faces
    uncertain_faces = np.asarray(assessment.uncertain_faces, dtype=np.int64)
    changed_faces = np.unique(np.concatenate([
        np.asarray(candidate['changed_faces'], dtype=np.int64)
        for candidate in candidates
    ])) if candidates else np.empty(0, dtype=np.int64)
    priority_faces = np.unique(np.r_[uncertain_faces, changed_faces])
    priority_faces = _even_sample(priority_faces, MAX_PREVIEW_FACES * 3 // 4)
    remaining = MAX_PREVIEW_FACES - len(priority_faces)
    context_faces = _even_sample(np.arange(len(faces)), max(remaining, 0))
    preview_face_ids = np.unique(np.r_[priority_faces, context_faces])
    preview_faces = faces[preview_face_ids]
    triangles = vertices[preview_faces]
    codes = np.unique(labels)
    palette = plt.get_cmap('tab10')
    colors = {label: palette(i % 10) for i, label in enumerate(codes)}
    if display_colors:
        colors.update({str(k): v for k, v in display_colors.items()})
    local_face_ids = _even_sample(uncertain_faces, MAX_PREVIEW_FACES)
    local_faces = faces[local_face_ids]
    uncertain_vertices = np.unique(local_faces)
    configurations = [('Current', labels)] + [(c['id'], c['owners']) for c in candidates]
    if len(configurations) == 1:
        configurations.append(('No admissible local candidate', labels))
    fig = plt.figure(figsize=(14, 4.5*len(configurations)))
    for row, (title, owners) in enumerate(configurations):
        seam = owners[graph.adjacency[:, 0]] != owners[graph.adjacency[:, 1]]
        seam_ids = _even_sample(np.flatnonzero(seam), MAX_PREVIEW_SEGMENTS)
        segments = vertices[graph.edges[seam_ids]]
        for column, (azim, zoom) in enumerate(((-90, False), (90, False), (30, True))):
            ax = fig.add_subplot(len(configurations), 3, row*3+column+1, projection='3d')
            face_colors = [colors.get(str(label), '#999999') for label in owners[preview_face_ids]]
            ax.add_collection3d(Poly3DCollection(triangles, facecolors=face_colors,
                linewidths=0, edgecolors='none', alpha=1.0))
            ax.add_collection3d(Line3DCollection(segments, colors='#00b878', linewidths=1.2))
            if row == 0:
                points = vertices[uncertain_vertices]
                ax.scatter(*points.T, s=2, c='#ff00b7', depthshade=False)
            points = vertices[uncertain_vertices] if zoom else vertices
            low, high = points.min(axis=0), points.max(axis=0)
            center = (low+high)/2
            radius = max(float(np.max(high-low))/2*1.12, 0.1)
            ax.set(xlim=(center[0]-radius, center[0]+radius),
                   ylim=(center[1]-radius, center[1]+radius), zlim=(center[2]-radius, center[2]+radius))
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=20, azim=azim)
            ax.set_title(f'{title} / '+('local' if zoom else ('front' if azim < 0 else 'back')))
            ax.set_axis_off()
    fig.suptitle('Green: physical partition boundary | Magenta: uncertain band\nBounded review visualization only; exact topology audits use the full mesh')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = Path(directory)/'boundary_candidates.png'
    fig.savefig(path, dpi=130)
    plt.close(fig)
    # Orthographic local projections expose the seam even when generated caps
    # obscure it in the assembly view. These are X-ray diagrams, not surface renders.
    from matplotlib.collections import LineCollection, PolyCollection
    local_vertices = vertices[np.unique(local_faces)]
    center = local_vertices.mean(axis=0)
    _, _, basis = np.linalg.svd(local_vertices-center, full_matrices=False)
    projected = (vertices-center) @ basis.T
    fig, axes = plt.subplots(len(configurations), 2, figsize=(12, 4*len(configurations)), squeeze=False)
    for row, (title, owners) in enumerate(configurations):
        seam = owners[graph.adjacency[:, 0]] != owners[graph.adjacency[:, 1]]
        local = np.any(np.isin(graph.adjacency, local_face_ids), axis=1)
        for column, dimensions in enumerate(((0, 1), (0, 2))):
            ax = axes[row, column]
            points = projected[:, dimensions]
            ax.add_collection(PolyCollection(points[local_faces], facecolors='#eeeeee',
                                            edgecolors='#cccccc', linewidths=.3))
            local_seam_ids = _even_sample(np.flatnonzero(seam & local), MAX_PREVIEW_SEGMENTS)
            ax.add_collection(LineCollection(points[graph.edges[local_seam_ids]],
                                            colors='#007c55', linewidths=1.4))
            changed = np.flatnonzero(owners != labels)
            if len(changed):
                changed = _even_sample(changed, MAX_PREVIEW_FACES)
                ax.add_collection(PolyCollection(points[faces[changed]], facecolors='#ffac33',
                                                edgecolors='#c56b00', alpha=.6, linewidths=.4))
            ax.autoscale()
            ax.set_aspect('equal')
            ax.set_title(f'{title} / local X-ray projection {column+1}')
            ax.set_xlabel('mm')
            ax.set_ylabel('mm')
    fig.suptitle('Local review X-ray: green = partition seam; orange = changed face ownership\nBounded display samples are not a self-intersection diagnosis')
    fig.tight_layout(rect=(0, 0, 1, .92))
    detail_path = Path(directory)/'boundary_local_detail.png'
    fig.savefig(detail_path, dpi=140)
    plt.close(fig)
    return [path, detail_path]
