"""Exact local rays with a conservative finite-segment spatial broad phase."""
import numpy as np
from scipy.spatial import cKDTree


class LocalRayProbe:
    def __init__(self, mesh):
        self.triangles = np.asarray(mesh.triangles)
        centers = self.triangles.mean(axis=1)
        radii = np.linalg.norm(self.triangles-centers[:, None], axis=2).max(axis=1)
        # One large cap triangle must not expand every tiny source-face query.
        # Each bucket retains its own exact enclosing radius; no hit is culled.
        levels = np.floor(np.log2(np.maximum(radii, 1e-9))).astype(int)
        self.buckets = []
        for level in np.unique(levels):
            ids = np.flatnonzero(levels == level)
            self.buckets.append((cKDTree(centers[ids]), ids, float(radii[ids].max())))

    def exits(self, points, directions, maximum_mm, epsilon=1e-7):
        """Return first outward exits within the bound; NaN means no bounded hit."""
        return self._hits(points, directions, maximum_mm, epsilon, outward_only=True)

    def first_hits(self, points, directions, maximum_mm, epsilon=1e-7):
        """Return the first surface hit, including entries, for side classification."""
        return self._hits(points, directions, maximum_mm, epsilon, outward_only=False)

    def exits_allowing_short_gaps(self, points, directions, maximum_mm,
                                  gap_tolerance_mm=1.0, epsilon=1e-7):
        """Return the material exit after sub-tolerance surface recesses.

        A painted inset may briefly leave and re-enter the original surface.
        Such gaps are recorded for review instead of being called a thin host.
        """
        return self._hits(points, directions, maximum_mm, epsilon,
                          outward_only=False, short_gap_mm=gap_tolerance_mm)

    def _hits(self, points, directions, maximum_mm, epsilon, *, outward_only,
              short_gap_mm=None):
        points = np.asarray(points, dtype=float).reshape((-1, 3))
        directions = np.broadcast_to(np.asarray(directions, dtype=float), points.shape).copy()
        lengths = np.linalg.norm(directions, axis=1)
        if not np.isfinite(maximum_mm) or maximum_mm <= 0 or np.any(lengths <= 1e-12):
            raise ValueError('Finite positive ray bounds and nonzero directions required')
        if not np.isfinite(points).all() or not np.isfinite(directions).all():
            raise ValueError('Ray coordinates must be finite')
        directions /= lengths[:, None]
        distances = np.full(len(points), np.nan)
        faces = np.full(len(points), -1, dtype=np.int64)
        gap_records = []
        for i, (point, axis) in enumerate(zip(points, directions)):
            candidates = []
            for tree, bucket_ids, radius in self.buckets:
                # Cover the entire finite ray with short enclosing spheres.
                # Every intersected triangle's centroid lies within its own
                # enclosing radius of one segment; no intersection is culled.
                count = max(1, int(np.ceil(maximum_mm / max(2., 2.*radius))))
                length = maximum_mm / count
                centers = point + ((np.arange(count) + .5) * length)[:, None] * axis
                groups = tree.query_ball_point(centers, length/2 + radius + 1e-9)
                populated = [group for group in groups if len(group)]
                if populated:
                    local_ids = np.unique(np.concatenate(populated)).astype(np.int64)
                    candidates.append(bucket_ids[local_ids])
            if not candidates:
                continue
            ids = np.concatenate(candidates)
            triangles = self.triangles[ids]
            a = triangles[:, 0]
            e, g = triangles[:, 1]-a, triangles[:, 2]-a
            h = np.cross(axis, g)
            det = np.einsum('ij,ij->i', e, h)
            facing = np.einsum('ij,j->i', np.cross(e, g), axis)
            valid = facing > 1e-12 if outward_only else np.abs(facing) > 1e-12
            ids, a, e, g, h, det = ids[valid], a[valid], e[valid], g[valid], h[valid], det[valid]
            delta = point-a
            u = np.einsum('ij,ij->i', delta, h)/det
            q = np.cross(delta, e)
            v = (q@axis)/det
            travel = np.einsum('ij,ij->i', g, q)/det
            hit = (u >= -1e-9) & (v >= -1e-9) & (u+v <= 1+1e-9)
            hit &= (travel > epsilon) & (travel <= maximum_mm)
            if hit.any():
                if short_gap_mm is None:
                    best = np.flatnonzero(hit)[np.argmin(travel[hit])]
                    distances[i], faces[i] = travel[best], ids[best]
                else:
                    ordered = np.flatnonzero(hit)[np.argsort(travel[hit])]
                    # Multiple triangles on a shared edge describe one event.
                    events = []
                    for index in ordered:
                        if events and travel[index] - events[-1][0] <= 1e-6:
                            continue
                        events.append((float(travel[index]), int(ids[index]),
                                       bool(facing[valid][index] > 0)))
                    for position, (distance, face, outward) in enumerate(events):
                        if not outward:
                            continue
                        following_entry = next(
                            (event for event in events[position + 1:] if not event[2]),
                            None,
                        )
                        if (following_entry is not None
                                and following_entry[0] - distance <= short_gap_mm):
                            gap_records.append({"probe_index": i,
                                                "exit_mm": distance,
                                                "entry_mm": following_entry[0],
                                                "gap_mm": following_entry[0] - distance,
                                                "exit_face": face})
                            continue
                        distances[i], faces[i] = distance, face
                        break
        if short_gap_mm is not None:
            return distances, faces, gap_records
        return distances, faces
