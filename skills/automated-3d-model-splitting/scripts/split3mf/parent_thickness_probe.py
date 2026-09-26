from __future__ import annotations

from scipy.ndimage import convolve1d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from .cap_template import fit_affine_cap_inside_parent, progressive_boundary_deformation, refined_harmonic_heightfield_cap
import copy
import os
import time
from .common import *
from .print_tolerance import current as current_print_tolerance
from .project import *
from .recognition import *
from .mesh import *
from .package_io import *
from .selection import *
from .assembly import *
from .validation import *
from .interface_retopology import InterfaceRetopologyService, generated_geometry_boundary_vertices
from .domain import PlanarArcRetopologyContext, CapDecision
from .hidden_interface import HIDDEN_INTERFACE_MINIMUM_LOAD_BEARING_DEPTH_MM, MAX_BOUNDARY_THICKNESS_PROBES, HiddenInterfacePlanner, boundary_screening_indices
from .guided_internal_cut import GuidedInternalCutPlanner, GuidedInternalCutSpec, adaptive_guided_entry_ring
from .local_connectors import LocalConnectorSpec, build_socket_cutter_from_plan, plan_local_connector, subtract_socket_cutters
from .connector_planning import local_connector_safe_depth_at_boundary, local_connector_safe_depth_from_field, local_connector_spec_for_interface
from .connector_geometry import backing_taper_angle_audit as _backing_taper_angle_audit, line_preserving_inset_displacements as _line_preserving_inset_displacements, printable_backing_profile as _printable_backing_profile, printable_backing_rings as _printable_backing_rings, project_connector_points as _project_connector_points, user_reviewed_shallow_minimal_needle_advisory_is_eligible
from .connector_topology import orient_face_patch_consistently, triangulate_bounded_ring_strip, triangulate_connector_annulus
from .connector_surface import refine_connector_annulus_heightfield
from .mesh_finalization import finalize_source_preserving_mesh
from .reporting import runtime_log


class ParentThicknessProbe:
    """Measure the first opposite-surface hit along inward boundary rays."""

    # A merely positive dot accepts an almost tangent side wall as an opposing
    # shell.  Prefer a materially opposing face whenever one exists later on
    # the same finite ray, while retaining weak-facing hits as a compatibility
    # fallback for genuinely oblique shells.
    minimum_preferred_opposing_face_dot = 0.15

    @dataclass(frozen=True)
    class BroadPhaseCandidateCache:
        """Direction-independent conservative candidates for repeated rays.

        Candidate groups contain global triangle ids.  A triangle intersecting
        any finite ray of ``search_limit_mm`` from an origin has its centroid
        within that limit plus its bounding radius, irrespective of direction.
        The normal per-direction capsule filter and exact ray test still run
        for every measurement.
        """

        probe_identity: int
        points: np.ndarray
        search_limit_mm: float
        candidate_groups: tuple[tuple[np.ndarray, ...], ...]
        candidate_count: int

    def prepare_safety_limit_candidates(
        self,
        points: np.ndarray,
        global_ceiling_mm: float,
    ) -> "ParentThicknessProbe.BroadPhaseCandidateCache":
        """Build a reusable superset for repeated ``safety_limit`` calls."""
        origins = np.asarray(points, dtype=np.float64)
        global_ceiling = min(
            max(float(global_ceiling_mm), 0.0),
            MAXIMUM_SAFE_INWARD_DEPTH_MM,
        )
        search_limit = global_ceiling + PARENT_THICKNESS_CLEARANCE_MM
        bucket_groups: list[tuple[np.ndarray, ...]] = []
        candidate_count = 0
        for triangle_ids, tree, maximum_radius in self.radius_buckets:
            local_groups = tree.query_ball_point(
                origins,
                search_limit + float(maximum_radius) + 1e-8,
                workers=-1 if len(origins) >= 64 else 1,
            )
            global_groups = tuple(
                triangle_ids[np.asarray(group, dtype=np.int64)]
                for group in local_groups
            )
            candidate_count += sum(len(group) for group in global_groups)
            bucket_groups.append(global_groups)
        return self.BroadPhaseCandidateCache(
            probe_identity=id(self),
            points=origins.copy(),
            search_limit_mm=search_limit,
            candidate_groups=tuple(bucket_groups),
            candidate_count=int(candidate_count),
        )

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        triangle_source_face_indices: np.ndarray | None = None,
        triangle_owner_indices: np.ndarray | None = None,
    ) -> None:
        self.mesh_vertices = np.asarray(vertices, dtype=np.float64)
        self.mesh_faces = np.asarray(faces, dtype=np.int64)
        self.triangles = self.mesh_vertices[self.mesh_faces]
        self._surface_topology_cache: dict = {}
        triangle_count = int(len(self.triangles))
        self.triangle_source_face_indices = (
            np.arange(triangle_count, dtype=np.int64)
            if triangle_source_face_indices is None
            else np.asarray(
                triangle_source_face_indices,
                dtype=np.int64,
            ).copy()
        )
        self.triangle_owner_indices = (
            np.zeros(triangle_count, dtype=np.int32)
            if triangle_owner_indices is None
            else np.asarray(
                triangle_owner_indices,
                dtype=np.int32,
            ).copy()
        )
        if self.triangle_source_face_indices.shape != (triangle_count,):
            raise ValueError(
                "triangle_source_face_indices must match the probe triangle count"
            )
        if self.triangle_owner_indices.shape != (triangle_count,):
            raise ValueError(
                "triangle_owner_indices must match the probe triangle count"
            )
        self.active_triangle_mask = np.ones(triangle_count, dtype=bool)
        self.active_triangle_count = triangle_count
        active_points = self.triangles.reshape(-1, 3)
        self.maximum_probe_distance_mm = (
            float(np.linalg.norm(np.ptp(active_points, axis=0)))
            if len(active_points)
            else 0.0
        )
        self.filter_context: dict = {
            "excluded_triangle_count": 0,
            "included_triangle_count": triangle_count,
        }
        self.centroids = self.triangles.mean(axis=1)
        self.radii = np.linalg.norm(
            self.triangles - self.centroids[:, None, :],
            axis=2,
        ).max(axis=1)
        self.normals = np.cross(
            self.triangles[:, 1] - self.triangles[:, 0],
            self.triangles[:, 2] - self.triangles[:, 0],
        )
        self.normals /= np.maximum(
            np.linalg.norm(self.normals, axis=1)[:, None],
            1e-12,
        )
        self.maximum_radius = float(self.radii.max()) if len(self.radii) else 0.0
        self.radius_buckets: list[tuple[np.ndarray, cKDTree, float]] = []
        if len(self.centroids):
            radius_scale_keys = np.ceil(
                np.log2(np.maximum(self.radii, 1e-9))
            ).astype(np.int16)
            for key in np.unique(radius_scale_keys):
                triangle_ids = np.flatnonzero(
                    radius_scale_keys == key
                ).astype(np.int64)
                bucket_centroids = self.centroids[triangle_ids]
                bucket_maximum_radius = float(
                    self.radii[triangle_ids].max()
                )
                self.radius_buckets.append(
                    (
                        triangle_ids,
                        cKDTree(bucket_centroids),
                        bucket_maximum_radius,
                    )
                )

    def excluding_triangles(
        self,
        triangle_indices: np.ndarray,
        filter_context: dict | None = None,
    ) -> "ParentThicknessProbe":
        """Return a lightweight probe view that shares acceleration data."""
        excluded = np.unique(
            np.asarray(triangle_indices, dtype=np.int64)
        )
        if len(excluded) and (
            int(excluded.min()) < 0
            or int(excluded.max()) >= len(self.triangles)
        ):
            raise ValueError("excluded triangle index is outside the probe")
        active_mask = np.asarray(
            self.active_triangle_mask,
            dtype=bool,
        ).copy()
        active_mask[excluded] = False
        view = object.__new__(type(self))
        for attribute in (
            "mesh_vertices",
            "mesh_faces",
            "_surface_topology_cache",
            "triangles",
            "triangle_source_face_indices",
            "triangle_owner_indices",
            "centroids",
            "radii",
            "normals",
            "maximum_radius",
            "maximum_probe_distance_mm",
        ):
            setattr(view, attribute, getattr(self, attribute))
        view.active_triangle_mask = active_mask
        view.active_triangle_count = int(np.count_nonzero(active_mask))
        active_points = view.triangles[active_mask].reshape(-1, 3)
        view.maximum_probe_distance_mm = (
            float(np.linalg.norm(np.ptp(active_points, axis=0)))
            if len(active_points)
            else 0.0
        )
        # Rebuild only the inexpensive centroid trees for the active parent
        # shell.  Sharing the original bucket trees made every later query
        # return child-subtree triangles which were immediately discarded.
        # Recursive cap planning probes the same filtered parent many times, so
        # paying this construction cost once removes that noise from every ray.
        view.radius_buckets = []
        for bucket_triangle_ids, _bucket_tree, _bucket_maximum_radius in self.radius_buckets:
            active_triangle_ids = bucket_triangle_ids[
                active_mask[bucket_triangle_ids]
            ]
            if not len(active_triangle_ids):
                continue
            active_centroids = self.centroids[active_triangle_ids]
            active_maximum_radius = float(
                self.radii[active_triangle_ids].max()
            )
            view.radius_buckets.append(
                (
                    active_triangle_ids,
                    cKDTree(active_centroids),
                    active_maximum_radius,
                )
            )
        view.maximum_radius = max(
            (float(bucket[2]) for bucket in view.radius_buckets),
            default=0.0,
        )
        excluded_owner_indices = sorted(
            int(value)
            for value in np.unique(
                self.triangle_owner_indices[excluded]
            )
            if int(value) > 0
        )
        view.filter_context = {
            **dict(filter_context or {}),
            "excluded_triangle_count": int(
                len(active_mask) - view.active_triangle_count
            ),
            "included_triangle_count": int(view.active_triangle_count),
            "excluded_owner_indices": excluded_owner_indices,
        }
        return view

    def _same_local_surface_hits(
        self,
        points: np.ndarray,
        hit_triangle_ids: np.ndarray,
        hit_distances: np.ndarray,
    ) -> np.ndarray:
        """Recognize short ray hits reachable along the sampled shell.

        Euclidean distance alone cannot distinguish a nearby fold of the
        source surface from an opposing wall.  A fold is also nearby along the
        mesh graph, while an actual inner/opposite shell requires a much longer
        route around the closed solid.  Only the small near-origin candidate
        band is audited, keeping this exact topological test inexpensive.
        """
        result = np.zeros(len(points), dtype=bool)
        # Audit the whole preferred printable-depth range.  Stopping at a
        # small Euclidean band merely moved the minimum to the next triangle
        # on a continuous curved surface (0.13 -> 0.26 mm on the Yoshi seam).
        # Topology, rather than this horizon, decides whether a hit is local;
        # the horizon only avoids work on hits that already provide the normal
        # requested insertion depth.
        horizon = DEFAULT_EFFECTIVE_MINIMUM_INWARD_DEPTH_MM
        candidates = np.flatnonzero(
            (hit_triangle_ids >= 0)
            & np.isfinite(hit_distances)
            & (hit_distances <= horizon + 1e-9)
        )
        if not len(candidates) or not len(self.mesh_vertices):
            return result
        cache = self._surface_topology_cache
        if "vertex_tree" not in cache:
            cache["vertex_tree"] = cKDTree(self.mesh_vertices)
        if "vertex_graph" not in cache:
            edges = np.vstack(
                (
                    self.mesh_faces[:, [0, 1]],
                    self.mesh_faces[:, [1, 2]],
                    self.mesh_faces[:, [2, 0]],
                )
            )
            lengths = np.linalg.norm(
                self.mesh_vertices[edges[:, 0]]
                - self.mesh_vertices[edges[:, 1]],
                axis=1,
            )
            rows = np.concatenate((edges[:, 0], edges[:, 1]))
            cols = np.concatenate((edges[:, 1], edges[:, 0]))
            weights = np.concatenate((lengths, lengths))
            cache["vertex_graph"] = coo_matrix(
                (weights, (rows, cols)),
                shape=(len(self.mesh_vertices), len(self.mesh_vertices)),
            ).tocsr()
        _nearest_distances, origin_vertices = cache["vertex_tree"].query(
            np.asarray(points)[candidates], k=1
        )
        for candidate, origin_vertex in zip(candidates, origin_vertices):
            triangle_id = int(hit_triangle_ids[candidate])
            if triangle_id >= len(self.mesh_faces):
                continue
            chord = float(hit_distances[candidate])
            # Allow curved/folded paths to be longer than their chord.  The
            # search remains local and cannot walk around a hollow shell to an
            # actual opposite wall.
            geodesic_limit = max(4.0 * chord, horizon)
            distances = dijkstra(
                cache["vertex_graph"],
                directed=False,
                indices=int(origin_vertex),
                limit=geodesic_limit,
            )
            if np.any(np.isfinite(distances[self.mesh_faces[triangle_id]])):
                result[candidate] = True
        return result

    def _owner_counts(self, triangle_ids: np.ndarray) -> dict[str, int]:
        triangle_ids = np.asarray(triangle_ids, dtype=np.int64)
        owner_indices = np.asarray(
            getattr(
                self,
                "triangle_owner_indices",
                np.empty(0, dtype=np.int32),
            ),
            dtype=np.int32,
        )
        triangle_ids = triangle_ids[
            (triangle_ids >= 0)
            & (triangle_ids < len(owner_indices))
        ]
        if not len(triangle_ids):
            return {}
        owners, counts = np.unique(
            owner_indices[triangle_ids],
            return_counts=True,
        )
        return {
            str(int(owner)): int(count)
            for owner, count in zip(owners, counts)
            if int(owner) > 0
        }

    def first_hit_distances(
        self,
        points: np.ndarray,
        directions: np.ndarray,
        search_limit_mm: float,
        broad_phase_cache: "ParentThicknessProbe.BroadPhaseCandidateCache | None" = None,
    ) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        directions = np.asarray(directions, dtype=np.float64)
        directions /= np.maximum(
            np.linalg.norm(directions, axis=1)[:, None],
            1e-12,
        )
        search_limit = max(float(search_limit_mm), 0.0)
        misses = search_limit + PARENT_THICKNESS_CLEARANCE_MM
        hits = np.full(len(points), misses, dtype=np.float64)
        self.last_hit_had_preceding_entry = np.zeros(
            len(points),
            dtype=bool,
        )
        if not self.radius_buckets or search_limit <= 0.0:
            return hits
        if broad_phase_cache is not None:
            if broad_phase_cache.probe_identity != id(self):
                raise ValueError("parent-thickness candidate cache belongs to another probe")
            if (
                broad_phase_cache.points.shape != points.shape
                or not np.array_equal(broad_phase_cache.points, points)
                or search_limit > broad_phase_cache.search_limit_mm + 1e-9
            ):
                raise ValueError("parent-thickness candidate cache does not cover this query")
            if len(broad_phase_cache.candidate_groups) != len(self.radius_buckets):
                raise ValueError("parent-thickness candidate cache bucket mismatch")

        query_started_at = time.perf_counter()
        runtime_log(
            "厚度测量",
            "thickness_candidates_start",
            "开始查询父体厚度射线候选三角形",
            probe_vertex_count=int(len(points)),
            source_triangle_count=int(len(self.triangles)),
            search_limit_mm=round(float(search_limit), 6),
            maximum_triangle_radius_mm=round(float(self.maximum_radius), 6),
            radius_bucket_count=int(len(self.radius_buckets)),
        )
        midpoints = points + directions * (search_limit * 0.5)
        candidate_count = 0
        query_elapsed_seconds = 0.0
        intersection_elapsed_seconds = 0.0
        broad_phase_segment_counts: collections.Counter = collections.Counter()
        exit_hits: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = [
            [] for _ in range(len(points))
        ]
        entry_hits: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = [
            [] for _ in range(len(points))
        ]
        selected_exit_distances = np.full(len(points), np.nan, dtype=np.float64)
        selected_entry_distances = np.full(len(points), np.nan, dtype=np.float64)
        selected_exit_triangle_ids = np.full(len(points), -1, dtype=np.int64)
        selected_entry_triangle_ids = np.full(len(points), -1, dtype=np.int64)
        for bucket_index, (
            bucket_triangle_ids,
            bucket_tree,
            bucket_maximum_radius,
        ) in enumerate(self.radius_buckets):
            bucket_query_started_at = time.perf_counter()
            # A single sphere around the complete segment has radius L/2+r and
            # is extremely loose for dense small triangles.  Cover the ray by
            # up to eight conservative capsule cells instead.  If a triangle
            # intersects a cell, its centroid is within triangle_radius plus
            # half the cell length of that cell's midpoint, so this cannot drop
            # a true hit while dramatically shrinking the broad phase.
            target_cell_length = max(
                2.0 * float(bucket_maximum_radius),
                search_limit / 8.0,
                1e-6,
            )
            segment_count = int(
                np.clip(
                    math.ceil(search_limit / target_cell_length),
                    1,
                    8,
                )
            )
            broad_phase_segment_counts[int(segment_count)] += 1
            if broad_phase_cache is not None:
                # The cached groups are a direction-independent superset.
                # Keep them as global ids; the segment-sphere rejection below
                # restores the tight per-direction capsule before exact tests.
                candidate_groups = broad_phase_cache.candidate_groups[bucket_index]
            elif segment_count == 1:
                candidate_groups = bucket_tree.query_ball_point(
                    midpoints,
                    search_limit * 0.5 + bucket_maximum_radius + 1e-8,
                    workers=-1 if len(points) >= 64 else 1,
                )
            else:
                cell_length = search_limit / float(segment_count)
                fractions = (
                    np.arange(segment_count, dtype=np.float64) + 0.5
                ) / float(segment_count)
                sample_points = (
                    points[:, None, :]
                    + directions[:, None, :]
                    * (search_limit * fractions[None, :, None])
                )
                sampled_groups = bucket_tree.query_ball_point(
                    sample_points.reshape(-1, 3),
                    bucket_maximum_radius + cell_length * 0.5 + 1e-8,
                    workers=-1 if len(points) >= 64 else 1,
                )
                candidate_groups = []
                for point_index in range(len(points)):
                    start = int(point_index * segment_count)
                    stop = int(start + segment_count)
                    groups = sampled_groups[start:stop]
                    nonempty = [
                        np.asarray(group, dtype=np.int64)
                        for group in groups
                        if len(group)
                    ]
                    if not nonempty:
                        candidate_groups.append([])
                    elif len(nonempty) == 1:
                        candidate_groups.append(nonempty[0].tolist())
                    else:
                        candidate_groups.append(
                            np.unique(np.concatenate(nonempty)).tolist()
                        )
            query_elapsed_seconds += float(
                time.perf_counter() - bucket_query_started_at
            )
            candidate_count += int(
                sum(len(group) for group in candidate_groups)
            )

            bucket_intersection_started_at = time.perf_counter()
            for index, (origin, direction, candidates) in enumerate(
                zip(points, directions, candidate_groups)
            ):
                if not len(candidates):
                    continue
                candidate_ids = (
                    np.asarray(candidates, dtype=np.int64)
                    if broad_phase_cache is not None
                    else bucket_triangle_ids[np.asarray(candidates, dtype=np.int64)]
                )
                candidate_ids = candidate_ids[
                    self.active_triangle_mask[candidate_ids]
                ]
                if not len(candidate_ids):
                    continue

                centroid_delta = self.centroids[candidate_ids] - origin
                ray_projection = centroid_delta @ direction
                clamped_projection = np.clip(
                    ray_projection,
                    0.0,
                    search_limit,
                )
                closest = origin + clamped_projection[:, None] * direction
                sphere_distance = np.linalg.norm(
                    self.centroids[candidate_ids] - closest,
                    axis=1,
                )
                candidate_ids = candidate_ids[
                    sphere_distance <= self.radii[candidate_ids] + 1e-7
                ]
                if not len(candidate_ids):
                    continue

                triangles = self.triangles[candidate_ids]
                edge_1 = triangles[:, 1] - triangles[:, 0]
                edge_2 = triangles[:, 2] - triangles[:, 0]
                h = np.cross(
                    np.broadcast_to(direction, edge_2.shape),
                    edge_2,
                )
                determinant = np.einsum("ij,ij->i", edge_1, h)
                active = np.abs(determinant) > 1e-12
                inverse = np.zeros_like(determinant)
                inverse[active] = 1.0 / determinant[active]
                s = origin - triangles[:, 0]
                u = inverse * np.einsum("ij,ij->i", s, h)
                q = np.cross(s, edge_1)
                v = inverse * (q @ direction)
                distance = inverse * np.einsum("ij,ij->i", edge_2, q)
                geometric_hit = (
                    active
                    & (u >= -1e-9)
                    & (v >= -1e-9)
                    & (u + v <= 1.0 + 1e-9)
                    & (distance > 1e-4)
                    & (distance <= search_limit + 1e-9)
                )
                if not np.any(geometric_hit):
                    continue
                hit_distances = distance[geometric_hit]
                hit_triangle_ids = candidate_ids[geometric_hit]
                hit_facing = (
                    self.normals[candidate_ids][geometric_hit] @ direction
                )
                exiting_mask = hit_facing > 1e-6
                entering_mask = hit_facing < -1e-6
                exiting = hit_distances[exiting_mask]
                entering = hit_distances[entering_mask]
                if len(exiting):
                    exit_hits[index].append(
                        (
                            exiting,
                            hit_triangle_ids[exiting_mask],
                            hit_facing[exiting_mask],
                        )
                    )
                if len(entering):
                    entry_hits[index].append(
                        (
                            entering,
                            hit_triangle_ids[entering_mask],
                            hit_facing[entering_mask],
                        )
                    )
            intersection_elapsed_seconds += float(
                time.perf_counter() - bucket_intersection_started_at
            )

        runtime_log(
            "厚度测量",
            "thickness_candidates_ready",
            "父体厚度射线候选三角形查询完成",
            probe_vertex_count=int(len(points)),
            source_triangle_count=int(len(self.triangles)),
            candidate_count=candidate_count,
            average_candidates_per_vertex=round(
                float(candidate_count / max(len(points), 1)),
                3,
            ),
            maximum_triangle_radius_mm=round(float(self.maximum_radius), 6),
            radius_bucket_count=int(len(self.radius_buckets)),
            broad_phase_segment_counts={
                str(int(key)): int(value)
                for key, value in sorted(broad_phase_segment_counts.items())
            },
            query_elapsed_seconds=round(float(query_elapsed_seconds), 3),
        )
        weak_exit_fallback_vertices = 0
        rejected_weak_exit_hits = 0
        preferred_dot = float(self.minimum_preferred_opposing_face_dot)
        for index, exiting_groups in enumerate(exit_hits):
            if not exiting_groups:
                continue
            has_preferred_exit = any(
                bool(np.any(group_facing >= preferred_dot))
                for _group_distances, _group_triangle_ids, group_facing in exiting_groups
            )
            if not has_preferred_exit:
                weak_exit_fallback_vertices += 1
            exit_distance = math.inf
            exit_triangle_id = -1
            for group_distances, group_triangle_ids, group_facing in exiting_groups:
                eligible_indices = np.flatnonzero(
                    group_facing >= preferred_dot
                ) if has_preferred_exit else np.arange(len(group_distances))
                if has_preferred_exit:
                    rejected_weak_exit_hits += int(
                        np.count_nonzero(group_facing < preferred_dot)
                    )
                if not len(eligible_indices):
                    continue
                group_index = int(
                    eligible_indices[
                        np.argmin(group_distances[eligible_indices])
                    ]
                )
                candidate_distance = float(group_distances[int(group_index)])
                if candidate_distance < exit_distance:
                    exit_distance = candidate_distance
                    exit_triangle_id = int(group_triangle_ids[int(group_index)])
            selected_exit_distances[index] = exit_distance
            selected_exit_triangle_ids[index] = exit_triangle_id
            entry_distance = 0.0
            entry_triangle_id = -1
            entering_groups = entry_hits[index]
            has_preferred_entry = any(
                bool(np.any(group_facing <= -preferred_dot))
                for _group_distances, _group_triangle_ids, group_facing in entering_groups
            )
            for group_distances, group_triangle_ids, group_facing in entering_groups:
                eligible_indices = np.flatnonzero(
                    group_distances < exit_distance - 1e-5
                )
                if has_preferred_entry:
                    eligible_indices = eligible_indices[
                        group_facing[eligible_indices] <= -preferred_dot
                    ]
                if len(eligible_indices):
                    group_index = int(
                        eligible_indices[
                            np.argmax(group_distances[eligible_indices])
                        ]
                    )
                    candidate_distance = float(
                        group_distances[group_index]
                    )
                    if candidate_distance > entry_distance:
                        entry_distance = candidate_distance
                        entry_triangle_id = int(
                            group_triangle_ids[group_index]
                        )
            if entry_distance > 0.0:
                selected_entry_distances[index] = entry_distance
                selected_entry_triangle_ids[index] = entry_triangle_id
            hits[index] = exit_distance - entry_distance
        near_hit_mask = hits <= PARENT_THICKNESS_CLEARANCE_MM + 1e-9
        paired_hit_mask = np.isfinite(selected_entry_distances)
        self.last_hit_had_preceding_entry = paired_hit_mask.copy()
        self.last_selected_exit_triangle_ids = (
            selected_exit_triangle_ids.copy()
        )
        self.last_selected_entry_triangle_ids = (
            selected_entry_triangle_ids.copy()
        )
        # Preserve the actual ray distances as well as the interval thickness.
        # When a ray starts in free space and later crosses a very thin opposing
        # shell, the usable insertion depth is limited by the shell entry, not by
        # the tiny entry-to-exit thickness of that remote shell.
        self.last_selected_exit_distances = selected_exit_distances.copy()
        self.last_selected_entry_distances = selected_entry_distances.copy()
        self.last_hit_diagnostics = {
            "near_hit_vertices": int(np.count_nonzero(near_hit_mask)),
            "near_hit_with_preceding_entry_vertices": int(
                np.count_nonzero(near_hit_mask & paired_hit_mask)
            ),
            "near_hit_without_preceding_entry_vertices": int(
                np.count_nonzero(near_hit_mask & ~paired_hit_mask)
            ),
            "selected_exit_distance_min_mm": (
                float(np.nanmin(selected_exit_distances))
                if np.any(np.isfinite(selected_exit_distances))
                else None
            ),
            "selected_preceding_entry_distance_max_mm": (
                float(np.nanmax(selected_entry_distances))
                if np.any(np.isfinite(selected_entry_distances))
                else None
            ),
            "selected_exit_owner_counts": self._owner_counts(
                selected_exit_triangle_ids
            ),
            "selected_preceding_entry_owner_counts": self._owner_counts(
                selected_entry_triangle_ids
            ),
            "preferred_opposing_face_dot_threshold": preferred_dot,
            "weak_exit_fallback_vertices": int(weak_exit_fallback_vertices),
            "rejected_weak_exit_hits": int(rejected_weak_exit_hits),
            "probe_filter": dict(
                getattr(self, "filter_context", {})
            ),
        }
        runtime_log(
            "厚度测量",
            "thickness_intersections_done",
            "父体厚度射线相交计算完成",
            probe_vertex_count=int(len(points)),
            candidate_count=candidate_count,
            measured_hit_count=int(np.count_nonzero(hits < misses)),
            intersection_elapsed_seconds=round(
                float(intersection_elapsed_seconds),
                3,
            ),
            total_elapsed_seconds=round(
                float(time.perf_counter() - query_started_at),
                3,
            ),
            **self.last_hit_diagnostics,
        )
        return hits

    def safety_limit(
        self,
        points: np.ndarray,
        directions: np.ndarray,
        global_ceiling_mm: float,
        broad_phase_cache: "ParentThicknessProbe.BroadPhaseCandidateCache | None" = None,
    ) -> tuple[float, dict]:
        points = np.asarray(points, dtype=np.float64)
        directions = np.asarray(directions, dtype=np.float64)
        full_boundary_probe_count = int(len(points))
        # Preserve pre-sampled loops passed by multi-candidate searches. This
        # keeps query hits, directions, and source-vertex diagnostics aligned.
        if len(points) <= MAX_BOUNDARY_THICKNESS_PROBES:
            boundary_sample_indices = np.arange(len(points), dtype=np.int64)
        else:
            boundary_sample_indices = boundary_screening_indices(
                points,
                MAX_BOUNDARY_THICKNESS_PROBES,
            )
            points = points[boundary_sample_indices]
            directions = directions[boundary_sample_indices]
        global_ceiling = min(
            max(float(global_ceiling_mm), 0.0),
            MAXIMUM_SAFE_INWARD_DEPTH_MM,
            float(getattr(self, "maximum_probe_distance_mm", global_ceiling_mm)),
        )
        search_limit = global_ceiling + PARENT_THICKNESS_CLEARANCE_MM
        if broad_phase_cache is None:
            # Preserve the overridable three-argument hook used by lightweight
            # policy probes and downstream integrations.
            thicknesses = self.first_hit_distances(
                points,
                directions,
                search_limit,
            )
        else:
            thicknesses = self.first_hit_distances(
                points,
                directions,
                search_limit,
                broad_phase_cache=broad_phase_cache,
            )
        original_thicknesses = thicknesses.copy()
        preceding_entry_distances = getattr(
            self,
            "last_selected_entry_distances",
            None,
        )
        selected_entry_triangle_ids = getattr(
            self,
            "last_selected_entry_triangle_ids",
            np.full(len(thicknesses), -1, dtype=np.int64),
        )
        parent_part_index = int(
            getattr(self, "filter_context", {}).get("parent_part_index", 0)
        )
        triangle_owner_indices = np.asarray(
            getattr(self, "triangle_owner_indices", np.empty(0, dtype=np.int32))
        )
        entry_owner_indices = np.full(len(thicknesses), 0, dtype=np.int32)
        valid_entry_triangle_ids = (
            (selected_entry_triangle_ids >= 0)
            & (selected_entry_triangle_ids < len(triangle_owner_indices))
        )
        entry_owner_indices[valid_entry_triangle_ids] = triangle_owner_indices[
            selected_entry_triangle_ids[valid_entry_triangle_ids]
        ]
        # Re-entering the shell currently being inset is a local surface fold,
        # even when curvature puts the re-entry farther from the ray origin.
        # It is not a remote obstacle.  Owner identity supplies the topological
        # distinction that a distance threshold cannot: genuinely separate
        # shells retain a different owner and continue to limit travel at
        # their entry point.
        paired_parent_reentry = (
            parent_part_index > 0
        ) & (entry_owner_indices == parent_part_index)
        if (
            hasattr(self, "mesh_vertices")
            and isinstance(preceding_entry_distances, np.ndarray)
            and preceding_entry_distances.shape == thicknesses.shape
        ):
            # Paint ownership partitions one physical shell into many parts;
            # it is not a shell-connectivity label.  A differently painted
            # entry that is reachable locally along the same mesh is still a
            # folded/re-entered source surface, not a remote obstacle.
            paired_parent_reentry |= self._same_local_surface_hits(
                points,
                selected_entry_triangle_ids,
                preceding_entry_distances,
            )
        paired_remote_shell_repair = np.zeros_like(thicknesses, dtype=bool)
        if (
            isinstance(preceding_entry_distances, np.ndarray)
            and preceding_entry_distances.shape == thicknesses.shape
        ):
            # ``first_hit_distances`` returns the material interval between a
            # preceding entry and its exit.  That interval is a valid local
            # wall thickness only when the ray starts on (within numerical
            # reserve of) that wall.  If the entry is remote, the generated
            # cap travels through free space first and is limited by the entry
            # location, regardless of whether the remote shell itself is
            # hair-thin or substantial.  Restricting this repair to intervals
            # below the clearance made a bounded screening ray report a safe
            # target while the longer authoritative ray incorrectly treated a
            # 0.218 mm remote shell as the available depth.
            paired_remote_shell_repair = (
                np.isfinite(preceding_entry_distances)
                & ~paired_parent_reentry
                & (
                    preceding_entry_distances
                    > PARENT_THICKNESS_CLEARANCE_MM + 1e-9
                )
            )
            thicknesses[paired_remote_shell_repair] = (
                preceding_entry_distances[paired_remote_shell_repair]
            )
        raw_minimum = float(thicknesses.min()) if len(thicknesses) else search_limit
        selected_exit_distances = getattr(
            self,
            "last_selected_exit_distances",
            None,
        )
        has_exit_classification = bool(
            isinstance(selected_exit_distances, np.ndarray)
            and selected_exit_distances.shape == thicknesses.shape
        )
        selected_exit_triangle_ids = getattr(
            self,
            "last_selected_exit_triangle_ids",
            np.full(len(thicknesses), -1, dtype=np.int64),
        )
        topologically_local_surface = np.zeros_like(thicknesses, dtype=bool)
        if (
            parent_part_index > 0
            and hasattr(self, "mesh_vertices")
            and selected_exit_triangle_ids.shape == thicknesses.shape
        ):
            exit_owner_indices = np.zeros(len(thicknesses), dtype=np.int32)
            valid_exit_ids = (
                (selected_exit_triangle_ids >= 0)
                & (selected_exit_triangle_ids < len(triangle_owner_indices))
            )
            exit_owner_indices[valid_exit_ids] = triangle_owner_indices[
                selected_exit_triangle_ids[valid_exit_ids]
            ]
            topologically_local_surface = self._same_local_surface_hits(
                points,
                selected_exit_triangle_ids,
                np.where(
                    exit_owner_indices == parent_part_index,
                    selected_exit_distances,
                    np.inf,
                ),
            )
        # A hit belongs to the sampled surface neighbourhood when either its
        # complete exit or its still-local material interval is within the
        # origin tolerance.  A paired entry inside the manufacturing clearance
        # is also unconditionally local: it means that the ray started in the
        # seam uncertainty band, briefly crossed out of the tessellated parent,
        # then entered it again.  The following exit can be arbitrarily far
        # from that entry along a folded surface, so classifying the pair by
        # interval length creates a brittle threshold chase.  Remote paired
        # shells were replaced by their entry distance above and are excluded
        # from this local-pair classification.
        local_paired_surface = (
            np.isfinite(preceding_entry_distances)
            & (~paired_remote_shell_repair | paired_parent_reentry)
            if isinstance(preceding_entry_distances, np.ndarray)
            and preceding_entry_distances.shape == thicknesses.shape
            else np.zeros_like(thicknesses, dtype=bool)
        )
        surface_near = (
            (thicknesses <= PARENT_SURFACE_HIT_TOLERANCE_MM + 1e-9)
            | local_paired_surface
            | topologically_local_surface
            | (
                (
                    selected_exit_distances
                    if has_exit_classification
                    else np.full_like(thicknesses, np.inf)
                )
                <= PARENT_SURFACE_HIT_TOLERANCE_MM + 1e-9
            )
        )
        coincident = thicknesses <= PARENT_THICKNESS_CLEARANCE_MM + 1e-9
        coincident_count = int(np.count_nonzero(coincident))
        coincident_ratio = float(coincident_count / max(len(thicknesses), 1))
        preceding_entry_mask = getattr(
            self,
            "last_hit_had_preceding_entry",
            None,
        )
        has_entry_classification = bool(
            isinstance(preceding_entry_mask, np.ndarray)
            and preceding_entry_mask.shape == thicknesses.shape
        )
        near_origin_surface_hits = (
            surface_near
            if has_entry_classification
            else np.zeros_like(coincident)
        )
        near_origin_surface_hit_count = int(
            np.count_nonzero(near_origin_surface_hits)
        )
        usable_mask = ~near_origin_surface_hits
        after_unpaired = thicknesses[usable_mask]
        after_unpaired_coincident = (
            after_unpaired
            <= PARENT_THICKNESS_CLEARANCE_MM + 1e-9
        )
        after_unpaired_coincident_count = int(
            np.count_nonzero(after_unpaired_coincident)
        )
        after_unpaired_coincident_ratio = float(
            after_unpaired_coincident_count
            / max(len(thicknesses), 1)
        )
        discard_isolated_coincident = bool(
            after_unpaired_coincident_count
            and after_unpaired_coincident_ratio < 0.01
            and np.any(~after_unpaired_coincident)
        )
        usable_thicknesses = (
            after_unpaired[~after_unpaired_coincident]
            if discard_isolated_coincident
            else after_unpaired
        )
        if discard_isolated_coincident:
            usable_mask &= ~coincident
        measured_minimum = (
            float(usable_thicknesses.min())
            if len(usable_thicknesses)
            else search_limit
        )
        hit_exit_ids = np.asarray(
            getattr(
                self,
                "last_selected_exit_triangle_ids",
                np.full(len(thicknesses), -1, dtype=np.int64),
            ),
            dtype=np.int64,
        )
        measured_hits = (
            hit_exit_ids >= 0
        ) | (selected_entry_triangle_ids >= 0)
        has_opposing_surface_hit = bool(
            hit_exit_ids.shape == thicknesses.shape
            and np.any(measured_hits & usable_mask)
        )
        thickness_quantiles = (
            np.quantile(usable_thicknesses, [0.0, 0.01, 0.05, 0.50, 0.95, 1.0])
            if len(usable_thicknesses)
            else np.full(6, search_limit, dtype=np.float64)
        )
        safe_maximum = (
            min(
                global_ceiling,
                max(0.0, measured_minimum - PARENT_THICKNESS_CLEARANCE_MM),
            )
            if has_opposing_surface_hit
            else global_ceiling
        )
        limiting_vertex_indices = np.flatnonzero(
            usable_mask
            & np.isclose(
                thicknesses,
                measured_minimum,
                rtol=0.0,
                atol=1e-9,
            )
        )
        selected_entry_triangle_ids = getattr(
            self,
            "last_selected_entry_triangle_ids",
            np.full(len(thicknesses), -1, dtype=np.int64),
        )
        limiting_exit_triangle_ids = selected_exit_triangle_ids[
            limiting_vertex_indices
        ]
        limiting_entry_triangle_ids = selected_entry_triangle_ids[
            limiting_vertex_indices
        ]
        triangle_source_face_indices = np.asarray(
            getattr(
                self,
                "triangle_source_face_indices",
                np.empty(0, dtype=np.int64),
            ),
            dtype=np.int64,
        )
        valid_limiting_exit_triangle_ids = limiting_exit_triangle_ids[
            (limiting_exit_triangle_ids >= 0)
            & (
                limiting_exit_triangle_ids
                < len(triangle_source_face_indices)
            )
        ]
        valid_limiting_entry_triangle_ids = limiting_entry_triangle_ids[
            (limiting_entry_triangle_ids >= 0)
            & (
                limiting_entry_triangle_ids
                < len(triangle_source_face_indices)
            )
        ]
        limiting_exit_source_faces_all = sorted(
            {
            int(value)
            for value in triangle_source_face_indices[
                valid_limiting_exit_triangle_ids
            ]
            }
        )
        limiting_entry_source_faces_all = sorted(
            {
            int(value)
            for value in triangle_source_face_indices[
                valid_limiting_entry_triangle_ids
            ]
            }
        )
        limiting_vertex_sample = [
            int(value) for value in limiting_vertex_indices[:16]
        ]
        limiting_probe_points = np.asarray(points, dtype=np.float64)[
            limiting_vertex_indices[:16]
        ].round(9).tolist()
        limiting_probe_directions = np.asarray(directions, dtype=np.float64)[
            limiting_vertex_indices[:16]
        ].round(9).tolist()
        limiting_exit_source_faces = limiting_exit_source_faces_all[:16]
        limiting_entry_source_faces = limiting_entry_source_faces_all[:16]
        result = {
            "parent_thickness_min_mm": measured_minimum,
            "parent_thickness_raw_min_mm": raw_minimum,
            "parent_thickness_interval_raw_min_mm": (
                float(original_thicknesses.min())
                if len(original_thicknesses)
                else search_limit
            ),
            "parent_thickness_remote_shell_interval_hits_repaired": int(
                np.count_nonzero(paired_remote_shell_repair)
            ),
            "parent_thickness_parent_reentry_hits_discarded": int(
                np.count_nonzero(paired_parent_reentry)
            ),
            "parent_thickness_topologically_local_hits_discarded": int(
                np.count_nonzero(topologically_local_surface)
            ),
            "parent_thickness_coincident_hit_vertices": coincident_count,
            "parent_thickness_coincident_hit_ratio": coincident_ratio,
            "parent_thickness_unpaired_surface_hit_vertices_discarded": (
                near_origin_surface_hit_count
            ),
            "parent_thickness_near_origin_surface_hit_vertices_discarded": (
                near_origin_surface_hit_count
            ),
            "parent_thickness_remaining_coincident_hit_vertices": (
                after_unpaired_coincident_count
            ),
            "parent_thickness_remaining_coincident_hit_ratio": (
                after_unpaired_coincident_ratio
            ),
            "parent_thickness_isolated_coincident_hits_discarded": discard_isolated_coincident,
            "parent_thickness_hit_vertices": int(np.count_nonzero(measured_hits)),
            "parent_thickness_probe_vertices": int(len(points)),
            "parent_thickness_boundary_sampling_policy": (
                "equal_arc_boundary_vertices_capped"
            ),
            "parent_thickness_boundary_input_vertices": full_boundary_probe_count,
            "parent_thickness_boundary_sampled_vertices": int(len(points)),
            "parent_thickness_boundary_sampled_indices": [
                int(value) for value in boundary_sample_indices[:32]
            ],
            "parent_thickness_full_boundary_audit_applied": False,
            "parent_thickness_unhit_policy": "safe_through_finite_search_limit",
            "parent_thickness_no_opposing_hit_safe_limit_applied": bool(
                not has_opposing_surface_hit
            ),
            "parent_thickness_is_lower_bound": bool(not np.all(measured_hits)),
            "parent_thickness_quantiles_mm": {
                key: float(value)
                for key, value in zip(
                    ("minimum", "p01", "p05", "median", "p95", "maximum"),
                    thickness_quantiles,
                )
            },
            "parent_thickness_clearance_mm": PARENT_THICKNESS_CLEARANCE_MM,
            "parent_surface_hit_tolerance_mm": (
                PARENT_SURFACE_HIT_TOLERANCE_MM
            ),
            "safe_maximum_inward_depth_mm": safe_maximum,
            "parent_thickness_limiting_probe_vertex_indices": [
                int(value) for value in limiting_vertex_sample
            ],
            "parent_thickness_limiting_probe_vertex_count": int(
                len(limiting_vertex_indices)
            ),
            "parent_thickness_limiting_probe_points": limiting_probe_points,
            "parent_thickness_limiting_probe_directions": limiting_probe_directions,
            "parent_thickness_limiting_exit_face_indices": (
                limiting_exit_source_faces
            ),
            "parent_thickness_limiting_exit_face_count": int(
                len(limiting_exit_source_faces_all)
            ),
            "parent_thickness_limiting_entry_face_indices": (
                limiting_entry_source_faces
            ),
            "parent_thickness_limiting_entry_face_count": int(
                len(limiting_entry_source_faces_all)
            ),
            "parent_thickness_limiting_exit_owner_counts": (
                self._owner_counts(limiting_exit_triangle_ids)
            ),
            "parent_thickness_limiting_entry_owner_counts": (
                self._owner_counts(limiting_entry_triangle_ids)
            ),
            "parent_thickness_probe_filter": dict(
                getattr(self, "filter_context", {})
            ),
            "parent_thickness_hit_diagnostics": dict(
                getattr(self, "last_hit_diagnostics", {})
            ),
        }
        runtime_log(
            "厚度测量",
            "parent_thickness_limit_selected",
            "父体厚度安全上限及限制命中面已确定",
            measured_parent_thickness_mm=float(measured_minimum),
            safe_maximum_inward_depth_mm=float(safe_maximum),
            limiting_probe_vertex_indices=[
                int(value) for value in limiting_vertex_sample
            ],
            limiting_probe_vertex_count=int(len(limiting_vertex_indices)),
            limiting_probe_points=limiting_probe_points,
            limiting_probe_directions=limiting_probe_directions,
            limiting_exit_face_indices=limiting_exit_source_faces,
            limiting_exit_face_count=int(
                len(limiting_exit_source_faces_all)
            ),
            limiting_entry_face_indices=limiting_entry_source_faces,
            limiting_entry_face_count=int(
                len(limiting_entry_source_faces_all)
            ),
            limiting_exit_owner_counts=self._owner_counts(
                limiting_exit_triangle_ids
            ),
            limiting_entry_owner_counts=self._owner_counts(
                limiting_entry_triangle_ids
            ),
            probe_filter=dict(
                getattr(self, "filter_context", {})
            ),
        )
        return safe_maximum, result
