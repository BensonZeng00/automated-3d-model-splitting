from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection


ROOT = Path(__file__).resolve().parent
data = np.load(ROOT / "failure_unplanned_boundary_loop.npz")
vertices = np.asarray(data["vertices"], dtype=np.float64)
faces = np.asarray(data["faces"], dtype=np.int64)
triangles = vertices[faces]
centers = triangles.mean(axis=1)
rng = np.random.default_rng(621)
sample = np.sort(rng.choice(len(faces), min(90000, len(faces)), replace=False))
overview = triangles[sample]
bbox_min, bbox_max = vertices.min(axis=0), vertices.max(axis=0)

critical = np.array([166.885199184, 199.05606494, 15.167173842])
shallow = np.array([159.838544766, 193.734144924, 2.995692714])
shallow2 = np.array([159.742942878, 193.732831277, 2.966391632])
points = [
    (critical, "0.323 mm wall", "#e31a4c"),
    (shallow, "0.786 mm wall", "#1769aa"),
    (shallow2, "0.814 mm wall", "#e58b16"),
]

fig, axes = plt.subplots(2, 2, figsize=(15, 9), facecolor="white")
fig.suptitle("P06 thickness audit — stopped before depth selection", fontsize=17, fontweight="bold")

for ax, xi, yi, title, xl, yl in (
    (axes[0, 0], 0, 1, "WHOLE MODEL — top view (XY)", "X (mm)", "Y (mm)"),
    (axes[0, 1], 0, 2, "WHOLE MODEL — side view (XZ)", "X (mm)", "Z (mm)"),
):
    ax.add_collection(PolyCollection(overview[:, :, [xi, yi]], facecolors="#aeb8c4",
                                     edgecolors="none", alpha=0.18, rasterized=True))
    for point, label, color in points:
        ax.scatter(point[xi], point[yi], s=110, color=color, edgecolor="white",
                    linewidth=0.9, zorder=5)
    p, label, color = points[0]
    ax.annotate(f"{label}\nX={p[0]:.2f}, Y={p[1]:.2f}, Z={p[2]:.2f}",
                (p[xi], p[yi]), xytext=(12, 14), textcoords="offset points",
                color=color, fontsize=10, fontweight="bold",
                arrowprops={"arrowstyle": "->", "color": color, "lw": 1.5})
    ax.set_xlim(bbox_min[xi]-1, bbox_max[xi]+1)
    ax.set_ylim(bbox_min[yi]-1, bbox_max[yi]+1)
    ax.set_aspect("equal")
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(title, fontweight="bold")
    ax.grid(alpha=0.18)

for ax, xi, yi, title, xl, yl in (
    (axes[1, 0], 0, 1, "P06 LOWER REGION — XY close-up", "X (mm)", "Y (mm)"),
    (axes[1, 1], 0, 2, "P06 THIN REGION — XZ close-up", "X (mm)", "Z (mm)"),
):
    focus = critical if ax is axes[1, 1] else shallow
    pad = 3.0
    mask = np.all((centers >= focus-pad) & (centers <= focus+pad), axis=1)
    local = triangles[mask]
    if len(local) > 22000:
        local = local[np.sort(rng.choice(len(local), 22000, replace=False))]
    ax.add_collection(PolyCollection(local[:, :, [xi, yi]], facecolors="#bac3ce",
                                     edgecolors="none", alpha=0.34, rasterized=True))
    for point, label, color in points:
        if np.all(np.abs(point-focus) < pad):
            ax.scatter(point[xi], point[yi], s=130, color=color, edgecolor="white",
                       linewidth=0.9, zorder=5, label=label)
            ax.annotate(label, (point[xi], point[yi]), xytext=(9, 8),
                        textcoords="offset points", color=color, fontsize=9,
                        fontweight="bold")
    ax.set_xlim(focus[xi]-pad, focus[xi]+pad)
    ax.set_ylim(focus[yi]-pad, focus[yi]+pad)
    ax.set_aspect("equal")
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(title, fontweight="bold")
    ax.grid(alpha=0.2)

fig.text(0.5, 0.025,
         "Stopped after ~6.5 min in P06 full-boundary audit: 21,761 rim points; ~14–33 million triangle candidates per direction. No split export.",
         ha="center", fontsize=10.5)
fig.tight_layout(rect=(0, 0.055, 1, 0.94))
output = ROOT / "thickness_search_stopped.png"
fig.savefig(output, dpi=170)
print(output)
