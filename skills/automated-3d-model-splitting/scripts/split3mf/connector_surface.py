"""Surface refinement for topology-safe local-connector annuli.

The topology module decides which edges are legal in a concave annulus.  This
module only improves the 3-D sampling of that already-valid surface.  Keeping
the responsibilities separate prevents visual smoothing from silently
changing connector topology or the source/model shared boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

from .common import *
from .connector_geometry import project_connector_points


# Dense hidden backing annuli have already passed the connector topology,
# bridge-length, and fanout audits before reaching this service.  Recursively
# subdividing thousands of accepted faces only improves hidden flat-shading
# density and can expand into millions of triangles.  Keep the audited surface
# unchanged above this limit; the visible source rim remains immutable either
# way.
DENSE_AUDITED_PASSTHROUGH_FACE_LIMIT = 2048


@dataclass(frozen=True)
class AnnulusSurfaceRefinement:
    passes: int
    source_faces: int
    refined_faces: int
    inserted_vertices: int
    quality_backoff_vertices: int
    maximum_internal_edge_before_mm: float
    maximum_internal_edge_after_mm: float
    target_maximum_internal_edge_mm: float
    effective_maximum_internal_edge_mm: float
    boundary_resolution_floor_mm: float
    topology_edge_flips: int = 0
    skipped_reason: str | None = None


def _ring_distance_and_residual(
    query_2d: np.ndarray,
    boundary_2d: np.ndarray,
    residual: np.ndarray,
    *,
    block_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Return closest-boundary distance and interpolated axial residual."""

    query = np.asarray(query_2d, dtype=np.float64)
    boundary = np.asarray(boundary_2d, dtype=np.float64)
    values = np.asarray(residual, dtype=np.float64)
    following = np.roll(boundary, -1, axis=0)
    edge = following - boundary
    length_squared = np.einsum("ij,ij->i", edge, edge)
    distances = np.empty(len(query), dtype=np.float64)
    carried = np.empty(len(query), dtype=np.float64)
    for start in range(0, len(query), max(1, int(block_size))):
        block = query[start : start + max(1, int(block_size))]
        relative = block[:, None, :] - boundary[None, :, :]
        parameter = np.divide(
            np.einsum("bij,ij->bi", relative, edge),
            length_squared[None, :],
            out=np.zeros((len(block), len(boundary)), dtype=np.float64),
            where=length_squared[None, :] > 1e-24,
        )
        parameter = np.clip(parameter, 0.0, 1.0)
        closest = boundary[None, :, :] + parameter[:, :, None] * edge[None, :, :]
        squared = np.einsum(
            "bij,bij->bi",
            block[:, None, :] - closest,
            block[:, None, :] - closest,
        )
        segment = np.argmin(squared, axis=1)
        local = parameter[np.arange(len(block)), segment]
        distances[start : start + len(block)] = np.sqrt(
            squared[np.arange(len(block)), segment]
        )
        carried[start : start + len(block)] = (
            values[segment] * (1.0 - local)
            + values[(segment + 1) % len(values)] * local
        )
    return distances, carried


def _maximum_non_boundary_edge(
    vertices: list[np.ndarray],
    faces: list[list[int]],
    boundary_edges: set[tuple[int, int]],
) -> float:
    edges: set[tuple[int, int]] = set()
    for face in faces:
        edges.update(
            tuple(sorted((int(left), int(right))))
            for left, right in (
                (face[0], face[1]),
                (face[1], face[2]),
                (face[2], face[0]),
            )
            if tuple(sorted((int(left), int(right)))) not in boundary_edges
        )
    return max(
        (
            float(
                np.linalg.norm(
                    np.asarray(vertices[right], dtype=np.float64)
                    - np.asarray(vertices[left], dtype=np.float64)
                )
            )
            for left, right in edges
        ),
        default=0.0,
    )


def _triangle_quality_2d(triangle: np.ndarray) -> float:
    """Return a scale-free 0..1 triangle quality score."""

    edges = np.roll(triangle, -1, axis=0) - triangle
    squared = float(np.einsum("ij,ij->", edges, edges))
    if squared <= 1e-24:
        return 0.0
    first = triangle[1] - triangle[0]
    second = triangle[2] - triangle[0]
    area2 = abs(float(first[0] * second[1] - first[1] * second[0]))
    return float(2.0 * np.sqrt(3.0) * area2 / squared)


from .quad_regularization import regularize as _regularize_unequal_ring_topology


def refine_connector_annulus_heightfield(
    *,
    output_vertices: list[np.ndarray],
    output_faces: list[list[int]],
    face_start: int,
    outer_ids: list[int],
    inner_ids: list[int],
    boundary_points: np.ndarray,
    plan: dict,
    taper_depth_mm: float,
    passes: int = 12,
    maximum_internal_edge_mm: float = 0.85,
    allow_audited_resolution_passthrough: bool = False,
    preserve_audited_surface: bool = False,
) -> AnnulusSurfaceRefinement:
    """Refine internal strip edges and project them to a continuous 45° field.

    Ring edges are immutable because they are shared with the source patch or
    backing floor.  Every other edge is split conformingly, and each new point
    receives axial depth equal to its lateral distance from the outer ring,
    capped at the selected taper depth.  Curved-rim height is faded by the same
    distance so refinement does not flatten the visible source seam.
    """

    selected = [list(map(int, face)) for face in output_faces[face_start:]]
    if not selected or int(passes) <= 0:
        return AnnulusSurfaceRefinement(
            0,
            len(selected),
            len(selected),
            0,
            0,
            0.0,
            0.0,
            float(maximum_internal_edge_mm),
            float(maximum_internal_edge_mm),
            0.0,
        )

    def ring_edges(ids: list[int]) -> set[tuple[int, int]]:
        return {
            tuple(sorted((int(ids[index]), int(ids[(index + 1) % len(ids)]))))
            for index in range(len(ids))
        }

    boundary_edges = ring_edges(outer_ids) | ring_edges(inner_ids)
    before = _maximum_non_boundary_edge(output_vertices, selected, boundary_edges)
    source_count = len(selected)
    inserted = 0
    executed_passes = 0

    axis = np.asarray(plan["inward"], dtype=np.float64)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    center = np.asarray(plan["center"], dtype=np.float64)
    boundary = np.asarray(boundary_points, dtype=np.float64)
    boundary_2d = project_connector_points(boundary, plan)
    source_axial = (boundary - center[None, :]) @ axis
    source_mean = float(np.mean(source_axial))
    source_residual = source_axial - source_mean
    depth_limit = max(float(taper_depth_mm), 1e-12)

    requested_target_edge = max(float(maximum_internal_edge_mm), 0.05)
    boundary_edge_lengths = [
        float(
            np.linalg.norm(
                np.asarray(output_vertices[right], dtype=np.float64)
                - np.asarray(output_vertices[left], dtype=np.float64)
            )
        )
        for left, right in boundary_edges
    ]
    boundary_resolution_floor = max(boundary_edge_lengths, default=0.0)
    # An immutable shared boundary segment sets a hard lower bound on the
    # incident triangles.  Preserve the requested visual target on normally
    # sampled rims without imposing an impossible demand on coarse test rings.
    target_edge = float(
        np.hypot(requested_target_edge, boundary_resolution_floor)
    )

    # The caller reaches this service only after the annulus has passed its
    # topology, bridge, fanout, winding, and degeneracy audits. Subdivision is
    # therefore a hidden flat-shading refinement, not a structural repair.
    if bool(allow_audited_resolution_passthrough) or preserve_audited_surface:
        return AnnulusSurfaceRefinement(
            0,
            int(source_count),
            int(source_count),
            0,
            0,
            float(before),
            float(before),
            float(requested_target_edge),
            float(max(target_edge, before)),
            float(boundary_resolution_floor),
            0,
            ("print_preserved_audited_hidden_surface" if preserve_audited_surface
             else "user_reviewed_hidden_surface_resolution_advisory"),
        )

    if source_count >= DENSE_AUDITED_PASSTHROUGH_FACE_LIMIT:
        return AnnulusSurfaceRefinement(
            0,
            int(source_count),
            int(source_count),
            0,
            0,
            float(before),
            float(before),
            float(requested_target_edge),
            float(max(target_edge, before)),
            float(boundary_resolution_floor),
            0,
            "dense_hidden_annulus_already_audited",
        )

    # The common production case is an index-aligned pair of rings.  Insert
    # complete intermediate rings instead of recursively bisecting triangles:
    # this makes a regular quad strip, keeps the shared rims byte-for-byte
    # fixed, and avoids feeding thousands of tiny facets into the socket
    # Boolean.  The generic conforming refinement below remains available for
    # genuinely unequal or visibility-constrained annuli.
    expected_faces: list[list[int]] = []
    if len(outer_ids) == len(inner_ids):
        for index in range(len(outer_ids)):
            following = (index + 1) % len(outer_ids)
            expected_faces.extend(
                (
                    [outer_ids[index], outer_ids[following], inner_ids[index]],
                    [outer_ids[following], inner_ids[following], inner_ids[index]],
                )
            )
    aligned = bool(
        expected_faces
        and len(selected) == len(expected_faces)
        and {
            tuple(sorted(int(value) for value in face)) for face in selected
        }
        == {
            tuple(sorted(int(value) for value in face)) for face in expected_faces
        }
    )
    if aligned and before > target_edge + 1e-12:
        outer = np.asarray(
            [output_vertices[int(value)] for value in outer_ids], dtype=np.float64
        )
        inner = np.asarray(
            [output_vertices[int(value)] for value in inner_ids], dtype=np.float64
        )
        radial_maximum = float(np.linalg.norm(inner - outer, axis=1).max())
        layer_count = max(
            2,
            int(np.ceil(radial_maximum / requested_target_edge)),
        )
        ring_ids: list[list[int]] = [list(map(int, outer_ids))]
        for layer_index in range(1, layer_count):
            ratio = float(layer_index) / float(layer_count)
            points = outer * (1.0 - ratio) + inner * ratio
            ids: list[int] = []
            for point in points:
                ids.append(len(output_vertices))
                output_vertices.append(np.asarray(point, dtype=np.float64))
            ring_ids.append(ids)
        ring_ids.append(list(map(int, inner_ids)))
        refined: list[list[int]] = []
        for first, second in zip(ring_ids[:-1], ring_ids[1:]):
            for index in range(len(first)):
                following = (index + 1) % len(first)
                refined.extend(
                    (
                        [first[index], first[following], second[index]],
                        [first[following], second[following], second[index]],
                    )
                )
        source_normal = np.cross(
            np.asarray(output_vertices[selected[0][1]])
            - np.asarray(output_vertices[selected[0][0]]),
            np.asarray(output_vertices[selected[0][2]])
            - np.asarray(output_vertices[selected[0][0]]),
        )
        refined_normal = np.cross(
            np.asarray(output_vertices[refined[0][1]])
            - np.asarray(output_vertices[refined[0][0]]),
            np.asarray(output_vertices[refined[0][2]])
            - np.asarray(output_vertices[refined[0][0]]),
        )
        if float(np.dot(source_normal, refined_normal)) < 0.0:
            refined = [[face[0], face[2], face[1]] for face in refined]
        output_faces[face_start:] = refined
        after = _maximum_non_boundary_edge(
            output_vertices, refined, boundary_edges
        )
        if after > target_edge + 1e-9:
            raise ValueError(
                "aligned connector ring refinement missed its resolution target: "
                f"maximum={after:.6f} mm, target={target_edge:.6f} mm"
            )
        return AnnulusSurfaceRefinement(
            int(layer_count - 1),
            int(source_count),
            int(len(refined)),
            int((layer_count - 1) * len(outer_ids)),
            0,
            float(before),
            float(after),
            float(requested_target_edge),
            float(target_edge),
            float(boundary_resolution_floor),
        )
    for _ in range(int(passes)):
        internal_edges = {
            tuple(sorted((int(left), int(right))))
            for face in selected
            for left, right in (
                (face[0], face[1]),
                (face[1], face[2]),
                (face[2], face[0]),
            )
            if tuple(sorted((int(left), int(right)))) not in boundary_edges
        }
        long_edges = {
            edge
            for edge in internal_edges
            if float(
                np.linalg.norm(
                    np.asarray(output_vertices[edge[1]], dtype=np.float64)
                    - np.asarray(output_vertices[edge[0]], dtype=np.float64)
                )
            )
            > target_edge + 1e-12
        }
        if not long_edges:
            break
        # Refine a complete local triangle stencil around every long edge.
        # Splitting only the current longest edge can create a different long
        # diagonal in the replacement triangles and cycle indefinitely on
        # coarse four-corner rings.  Expanding to the other non-boundary edges
        # of those faces gives a conforming red/green refinement while keeping
        # both shared boundary rings immutable.
        active_faces = [
            face
            for face in selected
            if any(
                tuple(sorted((int(left), int(right)))) in long_edges
                for left, right in (
                    (face[0], face[1]),
                    (face[1], face[2]),
                    (face[2], face[0]),
                )
            )
        ]
        split_edges = {
            edge
            for face in active_faces
            for edge in (
                tuple(sorted((int(face[0]), int(face[1])))),
                tuple(sorted((int(face[1]), int(face[2])))),
                tuple(sorted((int(face[2]), int(face[0])))),
            )
            if edge not in boundary_edges
        }
        executed_passes += 1
        midpoint_ids: dict[tuple[int, int], int] = {}
        midpoint_xy: list[np.ndarray] = []
        for edge in sorted(split_edges):
            left, right = edge
            midpoint_ids[edge] = len(output_vertices) + len(midpoint_xy)
            midpoint = 0.5 * (
                np.asarray(output_vertices[left], dtype=np.float64)
                + np.asarray(output_vertices[right], dtype=np.float64)
            )
            midpoint_xy.append(project_connector_points(midpoint[None, :], plan)[0])

        if midpoint_xy:
            query = np.asarray(midpoint_xy, dtype=np.float64)
            lateral, residual = _ring_distance_and_residual(
                query,
                boundary_2d,
                source_residual,
            )
            u = np.asarray(plan["u"], dtype=np.float64)
            v = np.asarray(plan["v"], dtype=np.float64)
            for xy, distance, carried in zip(query, lateral, residual):
                depth = min(max(float(distance), 0.0), depth_limit)
                fade = max(0.0, 1.0 - depth / depth_limit)
                axial = source_mean + depth + fade * float(carried)
                target = (
                    center
                    + float(xy[0]) * u
                    + float(xy[1]) * v
                    + axial * axis
                )
                output_vertices.append(target)
            inserted += len(midpoint_xy)

        refined: list[list[int]] = []
        for a, b, c in selected:
            mab = midpoint_ids.get(tuple(sorted((a, b))))
            mbc = midpoint_ids.get(tuple(sorted((b, c))))
            mca = midpoint_ids.get(tuple(sorted((c, a))))
            split_count = sum(value is not None for value in (mab, mbc, mca))
            if split_count == 0:
                refined.append([a, b, c])
            elif split_count == 1:
                if mab is not None:
                    refined.extend(([a, mab, c], [mab, b, c]))
                elif mbc is not None:
                    refined.extend(([b, mbc, a], [mbc, c, a]))
                else:
                    refined.extend(([c, mca, b], [mca, a, b]))
            elif split_count == 2:
                if mab is not None and mbc is not None:
                    refined.extend(([b, mbc, mab], [a, mab, c], [mab, mbc, c]))
                elif mbc is not None and mca is not None:
                    refined.extend(([c, mca, mbc], [b, mbc, a], [mbc, mca, a]))
                else:
                    refined.extend(([a, mab, mca], [c, mca, b], [mca, mab, b]))
            else:
                refined.extend(
                    (
                        [a, mab, mca],
                        [mab, b, mbc],
                        [mca, mbc, c],
                        [mab, mbc, mca],
                    )
                )
        selected = refined

    selected, topology_edge_flips = _regularize_unequal_ring_topology(
        vertices=output_vertices,
        faces=selected,
        boundary_edges=boundary_edges,
        plan=plan,
        maximum_internal_edge_mm=target_edge,
    )
    output_faces[face_start:] = selected
    after = _maximum_non_boundary_edge(output_vertices, selected, boundary_edges)
    if after > target_edge + 1e-9:
        raise ValueError(
            "connector annulus refinement left an overlong internal edge: "
            f"maximum={after:.6f} mm, target={target_edge:.6f} mm, "
            f"passes={executed_passes}/{int(passes)}"
        )
    return AnnulusSurfaceRefinement(
        int(executed_passes),
        int(source_count),
        int(len(selected)),
        int(inserted),
        0,
        float(before),
        float(after),
        float(requested_target_edge),
        float(target_edge),
        float(boundary_resolution_floor),
        int(topology_edge_flips),
    )
