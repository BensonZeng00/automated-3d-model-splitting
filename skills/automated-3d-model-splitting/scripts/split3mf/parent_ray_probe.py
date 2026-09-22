"""Bounded exit queries on a closed parent, with actionable early failures."""
import numpy as np
from .local_connectors import _manifold64
from .local_ray_probe import LocalRayProbe


class ParentSafetyError(ValueError):
    def __init__(self, record):
        self.record = record
        super().__init__(
            f"Invalid parent safety at sample {record['sample_index']}: "
            f"exit={record['exit_mm']}, required reserve={record['minimum_reserve_mm']}, "
            f"point={record['point_mm']}, direction={record['direction']}")


def parent_exit_distances(parent, points, directions, maximum_mm, epsilon=1e-7,
                          minimum_reserve_mm=None):
    points = np.asarray(points, dtype=float).reshape((-1, 3))
    axes = np.broadcast_to(np.asarray(directions, dtype=float), points.shape).copy()
    lengths = np.linalg.norm(axes, axis=1)
    if (not np.isfinite(points).all() or not np.isfinite(axes).all()
            or np.any(lengths <= 1e-12) or not np.isfinite(maximum_mm) or maximum_mm <= 0
            or not np.isfinite(epsilon) or epsilon < 0):
        raise ValueError('Finite coordinates, positive bounds and nonzero directions required')
    if minimum_reserve_mm is not None and (
            not np.isfinite(minimum_reserve_mm) or minimum_reserve_mm < 0):
        raise ValueError('Parent reserve must be finite and nonnegative')
    solid = _manifold64(parent)
    if str(solid.status()) != 'Error.NoError' or solid.is_empty():
        raise ValueError('Parent exit queries require a valid closed kernel solid')
    axes /= lengths[:, None]
    result = np.full(len(points), np.nan)
    ray_cast = getattr(solid, 'ray_cast', None)
    fallback = None if callable(ray_cast) else LocalRayProbe(parent)
    for index, (point, axis) in enumerate(zip(points, axes)):
        if fallback is not None:
            result[index] = fallback.exits([point], [axis], maximum_mm, epsilon)[0][0]
        else:
            for hit in ray_cast(point, point + axis * maximum_mm):
                distance = float(hit.distance) * maximum_mm
                if distance > epsilon and distance <= maximum_mm and np.dot(hit.normal, axis) > 1e-12:
                    result[index] = distance
                    break
        if minimum_reserve_mm is not None and (
                not np.isfinite(result[index]) or result[index] <= minimum_reserve_mm):
            # An early rejection is a partial audit, never an accepted depth map.
            raise ParentSafetyError(dict(
                sample_index=index, total_samples=len(points), tested_samples=index+1,
                complete=False, valid=False,
                exit_mm=float(result[index]) if np.isfinite(result[index]) else None,
                minimum_reserve_mm=float(minimum_reserve_mm),
                point_mm=point.tolist(), direction=axis.tolist(),
                method='native_closed_parent' if fallback is None else 'exact_segmented_probe'))
    return result
