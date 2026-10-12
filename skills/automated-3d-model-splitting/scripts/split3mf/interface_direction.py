"""Orient a shared interface from the confirmed tenon's local source surface."""

import numpy as np
from scipy.spatial import cKDTree

SOURCE_INTERIOR_PROBE_OFFSET_MM = 0.2
REGION_SIDE_SAMPLE_LIMIT = 128
REGION_SIDE_PROBE_SPAN_FRACTIONS = (0.08, 0.16, 0.24)
REGION_SIDE_MIN_PROBE_OFFSET_MM = 0.25
REGION_SIDE_MAX_PROBE_OFFSET_MM = 4.0
REGION_SIDE_DISTANCE_MARGIN_MM = 0.05
REGION_SIDE_MIN_DECISIVE_COVERAGE = 0.25
REGION_SIDE_MIN_DIRECTION_CONFIDENCE = 0.65
SOURCE_NORMAL_FALLBACK_MIN_CONSENSUS = 0.85


def _source_boundary_orientation(points, triangles):
    """Match seam tangents to outward-wound source half edges.

    The cap closing an exterior patch has the opposite boundary orientation.
    This establishes the material side even when the host wraps around the
    insert and proximity to its exterior suggests the opposite direction.
    """
    vertices, ids = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    faces = ids.reshape(-1, 3)
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    _, inverse, counts = np.unique(
        np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    boundary = edges[counts[inverse] == 1]
    if not len(boundary):
        return None, {"method": "source_boundary_half_edges", "matched": False}
    starts = vertices[boundary[:, 0]]
    vectors = vertices[boundary[:, 1]] - starts
    squared = np.einsum("ij,ij->i", vectors, vectors)
    valid = squared > 1e-20
    starts, vectors, squared = starts[valid], vectors[valid], squared[valid]
    if not len(starts):
        return None, {"method": "source_boundary_half_edges", "matched": False}
    tangents = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(tangents, axis=1)
    samples = (points[:, None] + tangents[:, None] *
               np.array([.2, .5, .8])[None, :, None]).reshape(-1, 3)
    tangent_samples = np.repeat(tangents, 3, axis=0)
    weights = np.repeat(lengths / 3, 3)
    distances = np.empty(len(samples))
    nearest = np.empty(len(samples), dtype=np.int64)
    batch_size = max(1, 200_000 // len(starts))
    for offset in range(0, len(samples), batch_size):
        batch = samples[offset:offset + batch_size]
        delta = batch[:, None] - starts
        fractions = np.clip(np.einsum("ijk,jk->ij", delta, vectors) / squared, 0, 1)
        residual = delta - fractions[:, :, None] * vectors
        costs = np.einsum("ijk,ijk->ij", residual, residual)
        selected = np.argmin(costs, axis=1)
        nearest[offset:offset + len(batch)] = selected
        distances[offset:offset + len(batch)] = np.sqrt(costs[np.arange(len(batch)), selected])
    tolerance = min(.60, max(.08, .06 * float(np.ptp(points, axis=0).max())))
    aligned = np.einsum("ij,ij->i", tangent_samples, vectors[nearest]) / np.maximum(
        np.repeat(lengths, 3) * np.sqrt(squared[nearest]), 1e-20
    )
    matched = (distances <= tolerance) & (np.abs(aligned) >= .25)
    coverage = float(weights[matched].sum() / max(weights.sum(), 1e-20))
    signed = float(np.sum(weights[matched] * aligned[matched]))
    magnitude = float(np.sum(weights[matched] * np.abs(aligned[matched])))
    consensus = abs(signed) / max(magnitude, 1e-20)
    accepted = coverage >= .80 and consensus >= .85
    evidence = {
        "method": "source_boundary_half_edges", "matched": accepted,
        "matched_arc_fraction": coverage, "orientation_consensus": consensus,
        "distance_tolerance_mm": tolerance,
        "p90_boundary_distance_mm": float(np.quantile(distances, .90)),
        "source_boundary_edge_count": len(starts),
    }
    return (-1 if signed > 0 else 1) if accepted else None, evidence


def _region_side_direction(points, axis, tenon_triangles, mortise_triangles):
    """Vote for the direction whose samples move from tenon surface to mortise.

    Painted regions are adjacent patches of one source shell, so whole-model
    inside/outside rays can report the same result on both sides of a seam.
    Local face-centroid proximity retains the component ownership that those
    rays lose. Several span-relative offsets avoid making the decision depend
    on one arbitrary probe depth.
    """
    triangles = []
    for name, raw in (("tenon", tenon_triangles), ("mortise", mortise_triangles)):
        value = np.asarray(raw, dtype=np.float64)
        if value.ndim != 3 or value.shape[1:] != (3, 3) or not len(value):
            raise ValueError(f"{name} component has no usable source triangles")
        centers = value.mean(axis=1)
        centers = centers[np.isfinite(centers).all(axis=1)]
        if not len(centers):
            raise ValueError(f"{name} component has no finite source triangles")
        triangles.append(cKDTree(centers))

    if len(points) > REGION_SIDE_SAMPLE_LIMIT:
        sample_ids = np.linspace(
            0, len(points) - 1, REGION_SIDE_SAMPLE_LIMIT, dtype=np.int64
        )
        samples = points[sample_ids]
    else:
        samples = points
    span = float(np.ptp(points, axis=0).max())
    offsets = sorted({
        min(
            REGION_SIDE_MAX_PROBE_OFFSET_MM,
            max(REGION_SIDE_MIN_PROBE_OFFSET_MM, fraction * span),
        )
        for fraction in REGION_SIDE_PROBE_SPAN_FRACTIONS
    })
    votes = {"positive_axis": 0, "negative_axis": 0}
    decisive = 0
    for distance in offsets:
        for sign in (1, -1):
            probes = samples + sign * axis * distance
            tenon_distance = triangles[0].query(probes)[0]
            mortise_distance = triangles[1].query(probes)[0]
            mortise_closer = mortise_distance + REGION_SIDE_DISTANCE_MARGIN_MM < tenon_distance
            tenon_closer = tenon_distance + REGION_SIDE_DISTANCE_MARGIN_MM < mortise_distance
            along_sign = "positive_axis" if sign > 0 else "negative_axis"
            against_sign = "negative_axis" if sign > 0 else "positive_axis"
            votes[along_sign] += int(np.count_nonzero(mortise_closer))
            votes[against_sign] += int(np.count_nonzero(tenon_closer))
            decisive += int(np.count_nonzero(mortise_closer | tenon_closer))

    total = len(samples) * len(offsets) * 2
    decisive_coverage = decisive / max(total, 1)
    vote_total = votes["positive_axis"] + votes["negative_axis"]
    confidence = max(votes.values()) / max(vote_total, 1)
    selected_sign = (
        1 if votes["positive_axis"] > votes["negative_axis"] else
        -1 if votes["negative_axis"] > votes["positive_axis"] else 0
    )
    evidence = {
        "method": "paired_region_surface_side_probe",
        "probe_offsets_mm": [float(value) for value in offsets],
        "sample_count": int(len(samples)),
        "positive_axis_votes": int(votes["positive_axis"]),
        "negative_axis_votes": int(votes["negative_axis"]),
        "decisive_probe_coverage": float(decisive_coverage),
        "direction_confidence": float(confidence),
        "distance_margin_mm": REGION_SIDE_DISTANCE_MARGIN_MM,
    }
    if (
        selected_sign == 0
        or decisive_coverage < REGION_SIDE_MIN_DECISIVE_COVERAGE
        or confidence < REGION_SIDE_MIN_DIRECTION_CONFIDENCE
    ):
        return None, evidence
    return selected_sign, evidence


def inward_interface_axis(points, triangles, *, source_probe=None,
                          mortise_triangles=None, toward_mortise=None,
                          maximum_mm=None):
    """Keep the seam axis, orienting its sign from source boundary topology.

    Matched outward source half edges determine the opposite closing-cap
    orientation. Rays and region proximity are legacy fallbacks only when the
    seam cannot be matched reliably. Source winding is never modified here.
    """
    points = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.float64)
    centered = points - points.mean(axis=0)
    axis = np.cross(centered, np.roll(centered, -1, axis=0)).sum(axis=0)
    length = np.linalg.norm(axis)
    if not np.isfinite(length) or length <= 1e-10:
        raise ValueError("interface boundary has no stable area normal")
    axis /= length
    normals = np.cross(triangles[:, 1] - triangles[:, 0],
                       triangles[:, 2] - triangles[:, 0])
    valid = np.isfinite(normals).all(axis=1) & (np.linalg.norm(normals, axis=1) > 1e-14)
    if not valid.any():
        raise ValueError("confirmed tenon has no usable source normals")
    normals = normals[valid]
    centers = triangles[valid].mean(axis=1)
    _, nearest = cKDTree(centers).query(points, k=min(8, len(centers)))
    selected = normals[np.unique(nearest)]
    projections = selected @ axis
    magnitude = float(np.abs(projections).sum())
    signed = float(projections.sum())
    total_area = float(np.linalg.norm(selected, axis=1).sum())
    consensus = abs(signed) / max(magnitude, 1e-30)
    evidence = {
        "method": "confirmed_tenon_local_source_inward",
        "source_face_count": int(len(selected)),
        "projected_normal_consensus": consensus,
    }
    boundary_sign, boundary_evidence = _source_boundary_orientation(points, triangles)
    evidence["source_boundary_evidence"] = boundary_evidence
    if boundary_sign is not None:
        axis *= boundary_sign
        evidence["method"] = "confirmed_tenon_source_boundary_orientation"
        evidence["selection"] = "opposite_outward_source_boundary"
        evidence["outward_normal_dot_axis"] = float(selected.sum(axis=0) @ axis) / total_area
        return axis, evidence
    if source_probe is None:
        if magnitude <= total_area * 1e-3 or consensus < 0.2:
            raise ValueError("confirmed tenon local source normals do not establish an inward interface direction")
        if signed > 0:
            axis = -axis
    else:
        # Probe just past the shared contour using the complete source mesh.
        # A curved rim may span many millimeters along its area normal; using
        # that global span as an origin offset can move the ray origin far
        # away from the interface and outside the model on both sides.
        probes = np.array([[.73, .41, .55], [-.37, .81, .45], [.21, -.57, .79]])
        probes /= np.linalg.norm(probes, axis=1)[:, None]
        center = points.mean(axis=0)
        votes = []
        for candidate in (axis, -axis):
            start = center + candidate * SOURCE_INTERIOR_PROBE_OFFSET_MM
            _, hit_ids = source_probe.first_hits(
                np.broadcast_to(start, (3, 3)), probes, maximum_mm
            )
            inside_votes = 0
            for ray, face_id in zip(probes, hit_ids):
                if face_id >= 0:
                    tri = source_probe.triangles[face_id]
                    normal = np.cross(tri[1]-tri[0], tri[2]-tri[0])
                    inside_votes += int(normal @ ray > 0)
            votes.append(inside_votes)
        supported = [count >= 2 for count in votes]
        evidence["source_interior_votes"] = votes
        evidence["method"] = "confirmed_roles_source_interior"
        evidence["source_probe_offset_mm"] = SOURCE_INTERIOR_PROBE_OFFSET_MM
        region_sign = None
        if mortise_triangles is not None:
            region_sign, region_evidence = _region_side_direction(
                points, axis, triangles, mortise_triangles
            )
            evidence["region_side_evidence"] = region_evidence
        if region_sign is not None:
            if region_sign < 0:
                axis = -axis
            evidence["selection"] = "paired_region_ownership_toward_confirmed_mortise"
        elif not any(supported):
            region_evidence = evidence.get("region_side_evidence", {})
            positive_votes = int(region_evidence.get("positive_axis_votes", 0))
            negative_votes = int(region_evidence.get("negative_axis_votes", 0))
            vote_sign = (
                1 if positive_votes > negative_votes else
                -1 if negative_votes > positive_votes else 0
            )
            decisive_coverage = float(
                region_evidence.get("decisive_probe_coverage", 0.0)
            )
            if (
                mortise_triangles is not None
                and vote_sign != 0
                and decisive_coverage >= REGION_SIDE_MIN_DECISIVE_COVERAGE
                and consensus >= SOURCE_NORMAL_FALLBACK_MIN_CONSENSUS
            ):
                normal_sign = -1 if signed > 0 else 1
                if normal_sign < 0:
                    axis = -axis
                evidence["selection"] = "strong_tenon_normals_pending_review"
                evidence["normal_fallback_consensus"] = consensus
                evidence["paired_region_vote_alignment"] = (
                    "agree" if vote_sign == normal_sign else "conflict"
                )
            else:
                raise ValueError(
                    "neither source-interior rays nor paired-region evidence "
                    "establish a direction with sufficiently consistent tenon normals"
                )
        elif all(supported):
            if mortise_triangles is not None:
                region_evidence = evidence.get("region_side_evidence", {})
                positive_votes = int(region_evidence.get("positive_axis_votes", 0))
                negative_votes = int(region_evidence.get("negative_axis_votes", 0))
                vote_sign = (
                    1 if positive_votes > negative_votes else
                    -1 if negative_votes > positive_votes else 0
                )
                decisive_coverage = float(
                    region_evidence.get("decisive_probe_coverage", 0.0)
                )
                normal_sign = -1 if signed > 0 else 1
                if (
                    vote_sign != 0
                    and decisive_coverage >= REGION_SIDE_MIN_DECISIVE_COVERAGE
                    and consensus >= SOURCE_NORMAL_FALLBACK_MIN_CONSENSUS
                ):
                    if normal_sign < 0:
                        axis = -axis
                    evidence["selection"] = "strong_tenon_normals_pending_review"
                    evidence["normal_fallback_consensus"] = consensus
                    evidence["paired_region_vote_alignment"] = (
                        "agree" if vote_sign == normal_sign else "conflict"
                    )
                else:
                    raise ValueError(
                        "source-interior rays are symmetric and paired-region evidence "
                        "does not corroborate a high-consensus tenon-normal direction"
                    )
            else:
                toward = np.asarray(toward_mortise, dtype=float)
                projection = float(axis @ toward)
                if abs(projection) <= max(np.linalg.norm(toward), 1.0) * 1e-6:
                    raise ValueError("both interface sides are interior but the confirmed mortise side is ambiguous")
                if projection < 0:
                    axis = -axis
                evidence["selection"] = "both_interior_toward_confirmed_mortise"
        else:
            if supported[1]:
                axis = -axis
            evidence["selection"] = "only_source_interior_side"
    evidence["outward_normal_dot_axis"] = float(selected.sum(axis=0) @ axis) / total_area
    return axis, evidence
