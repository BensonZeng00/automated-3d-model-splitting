"""Offline exact-triangle review drawings, not an acceptance/raster audit."""
from pathlib import Path
import os
import numpy as np


def render_boundary_review(graph, labels, assessment, candidates, directory, display_colors=None):
    os.environ.setdefault('MPLCONFIGDIR', str(Path(directory) / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

    vertices, faces = graph.vertices, graph.faces
    triangles = vertices[faces]
    codes = np.unique(labels)
    palette = plt.get_cmap('tab10')
    colors = {label: palette(i % 10) for i, label in enumerate(codes)}
    if display_colors:
        colors.update({str(k): v for k, v in display_colors.items()})
    uncertain_vertices = np.unique(faces[assessment.uncertain_faces])
    configurations = [('Current', labels)] + [(c['id'], c['owners']) for c in candidates]
    if len(configurations) == 1:
        configurations.append(('No admissible local candidate', labels))
    fig = plt.figure(figsize=(14, 4.5*len(configurations)))
    for row, (title, owners) in enumerate(configurations):
        seam = owners[graph.adjacency[:, 0]] != owners[graph.adjacency[:, 1]]
        segments = vertices[graph.edges[seam]]
        for column, (azim, zoom) in enumerate(((-90, False), (90, False), (30, True))):
            ax = fig.add_subplot(len(configurations), 3, row*3+column+1, projection='3d')
            face_colors = [colors.get(str(label), '#999999') for label in owners]
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
    fig.suptitle('Green: physical partition boundary | Magenta: uncertain band\nColors below show proposed ownership, NOT recoloring the source model')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = Path(directory)/'boundary_candidates.png'
    fig.savefig(path, dpi=130)
    plt.close(fig)
    # Orthographic local projections expose the seam even when generated caps
    # obscure it in the assembly view. These are X-ray diagrams, not surface renders.
    from matplotlib.collections import LineCollection, PolyCollection
    local_faces = faces[assessment.uncertain_faces]
    local_vertices = vertices[np.unique(local_faces)]
    center = local_vertices.mean(axis=0)
    _, _, basis = np.linalg.svd(local_vertices-center, full_matrices=False)
    projected = (vertices-center) @ basis.T
    fig, axes = plt.subplots(len(configurations), 2, figsize=(12, 4*len(configurations)), squeeze=False)
    for row, (title, owners) in enumerate(configurations):
        seam = owners[graph.adjacency[:, 0]] != owners[graph.adjacency[:, 1]]
        local = np.any(np.isin(graph.adjacency, assessment.uncertain_faces), axis=1)
        for column, dimensions in enumerate(((0, 1), (0, 2))):
            ax = axes[row, column]
            points = projected[:, dimensions]
            ax.add_collection(PolyCollection(points[local_faces], facecolors='#eeeeee',
                                            edgecolors='#cccccc', linewidths=.3))
            ax.add_collection(LineCollection(points[graph.edges[seam & local]],
                                            colors='#007c55', linewidths=1.4))
            changed = np.flatnonzero(owners != labels)
            if len(changed):
                ax.add_collection(PolyCollection(points[faces[changed]], facecolors='#ffac33',
                                                edgecolors='#c56b00', alpha=.6, linewidths=.4))
            ax.autoscale()
            ax.set_aspect('equal')
            ax.set_title(f'{title} / local X-ray projection {column+1}')
            ax.set_xlabel('mm')
            ax.set_ylabel('mm')
    fig.suptitle('Local exact-geometry X-ray: green = partition seam; orange = changed face ownership\nOverlaps in this projection are not a self-intersection diagnosis')
    fig.tight_layout(rect=(0, 0, 1, .92))
    detail_path = Path(directory)/'boundary_local_detail.png'
    fig.savefig(detail_path, dpi=140)
    plt.close(fig)
    return [path, detail_path]
