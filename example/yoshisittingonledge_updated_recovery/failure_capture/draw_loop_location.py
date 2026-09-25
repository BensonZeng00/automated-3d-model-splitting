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
loop = np.asarray(data["failed_loop_points"], dtype=np.float64)
center = loop.mean(axis=0)
all_triangles = vertices[faces]
triangle_centers = all_triangles.mean(axis=1)
rng = np.random.default_rng(218)
sample_count = min(90000, len(faces))
sample_ids = np.sort(rng.choice(len(faces), sample_count, replace=False))
overview_triangles = all_triangles[sample_ids]

fig, axes = plt.subplots(2, 2, figsize=(15, 9), facecolor="white")
fig.suptitle(
    "218-point unplanned boundary loop — location on updated model",
    fontsize=17,
    fontweight="bold",
)

bounds_min = vertices.min(axis=0)
bounds_max = vertices.max(axis=0)
pad = 2.5
for ax, x_index, y_index, title, x_label, y_label in (
    (axes[0, 0], 0, 1, "WHOLE MODEL — top view (XY)", "X (mm)", "Y (mm)"),
    (axes[0, 1], 0, 2, "WHOLE MODEL — side view (XZ)", "X (mm)", "Z (mm)"),
):
    ax.add_collection(
        PolyCollection(
            overview_triangles[:, :, [x_index, y_index]],
            facecolors="#b6bec8",
            edgecolors="none",
            alpha=0.10,
            rasterized=True,
        )
    )
    ax.scatter(center[x_index], center[y_index], s=180, color="#ed174c",
               edgecolor="white", linewidth=1.3, zorder=5, label="218-point loop")
    ax.scatter(center[x_index], center[y_index], s=520, facecolors="none",
               edgecolors="#ed174c", linewidth=2.3, zorder=4)
    ax.annotate(
        f"218-point loop\n({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}) mm",
        (center[x_index], center[y_index]),
        xytext=(14, 14), textcoords="offset points", color="#b80f3a",
        fontsize=10, fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": "#ed174c", "lw": 1.6},
    )
    ax.set_xlim(bounds_min[x_index]-1, bounds_max[x_index]+1)
    ax.set_ylim(bounds_min[y_index]-1, bounds_max[y_index]+1)
    ax.set_aspect("equal")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontweight="bold")
    ax.grid(alpha=0.18)

local_mask = np.all(
    (triangle_centers >= center - pad) & (triangle_centers <= center + pad), axis=1
)
local_triangles = all_triangles[local_mask]
if len(local_triangles) > 24000:
    local_triangles = local_triangles[
        np.sort(rng.choice(len(local_triangles), 24000, replace=False))
    ]
closed = np.vstack([loop, loop[0]])

for ax, x_index, y_index, title, x_label, y_label in (
    (axes[1, 0], 0, 1, "CLOSE-UP — same area (XY)", "X (mm)", "Y (mm)"),
    (axes[1, 1], 0, 2, "CLOSE-UP — height (XZ)", "X (mm)", "Z (mm)"),
):
    ax.add_collection(
        PolyCollection(
            local_triangles[:, :, [x_index, y_index]],
            facecolors="#bbc3cd",
            edgecolors="none",
            alpha=0.42,
            rasterized=True,
        )
    )
    ax.plot(closed[:, x_index], closed[:, y_index], color="#ed174c", linewidth=2.7)
    ax.scatter(loop[:, x_index], loop[:, y_index], s=5, color="#ed174c", zorder=4)
    ax.scatter(center[x_index], center[y_index], marker="+", s=125, color="#a80d33",
               linewidth=2.0, zorder=5)
    ax.set_xlim(center[x_index]-pad, center[x_index]+pad)
    ax.set_ylim(center[y_index]-pad, center[y_index]+pad)
    ax.set_aspect("equal")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontweight="bold")
    ax.grid(alpha=0.20)

fig.text(
    0.5,
    0.015,
    "Red marks the exact boundary. Location: X 159.03, Y 194.53, Z 3.28 mm; Z=0 is the model base.",
    ha="center",
    fontsize=11,
)
fig.tight_layout(rect=(0, 0.04, 1, 0.94))
output = ROOT / "failure_loop_location.png"
fig.savefig(output, dpi=170)
print(output)
