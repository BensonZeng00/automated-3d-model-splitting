"""Resolve and audit connector directions against an oriented source rim."""
import numpy as np


COHERENT_RIM_THRESHOLD = 0.5
OUTWARD_DOT_TOLERANCE = 1e-6


def _source_rim_measurement(vertices, faces, boundary_ids, direction):
    """Measure how a direction relates to the faces incident to a closed rim."""
    boundary = [int(value) for value in boundary_ids]
    edges = {
        tuple(sorted((a, b)))
        for a, b in zip(boundary, boundary[1:] + boundary[:1])
    }
    selected = []
    for face in faces:
        face_ids = [int(value) for value in face]
        face_edges = zip(face_ids, face_ids[1:] + face_ids[:1])
        if any(tuple(sorted(edge)) in edges for edge in face_edges):
            selected.append(face_ids)
    if not selected:
        return None, {
            "status": "not_evaluated",
            "source_rim_faces": 0,
        }

    triangles = np.asarray(vertices, dtype=float)[np.asarray(selected, dtype=int)]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    average = normals.sum(axis=0)
    length = float(np.linalg.norm(average))
    area = float(np.linalg.norm(normals, axis=1).sum())
    coherence = length / max(area, 1e-30)
    axis = np.asarray(direction, dtype=float)
    axis_length = float(np.linalg.norm(axis))
    cosine = float(average @ axis / max(length * axis_length, 1e-30))
    outward = average / max(length, 1e-30)
    return outward, {
        "status": "measured",
        "source_rim_faces": len(selected),
        "normal_coherence": coherence,
        "outward_dot_inward": cosine,
    }


def resolve_inward_axis(vertices, faces, boundary_ids, proposed_inward):
    """Flip a component-level axis when a coherent local rim proves it is outward.

    Components assembled from many paint islands can span differently oriented
    surfaces.  The component-wide direction remains a useful insertion axis, but
    the oriented faces at each individual rim are authoritative about which of
    its two signs points into material.
    """
    proposed = np.asarray(proposed_inward, dtype=float)
    proposed /= max(float(np.linalg.norm(proposed)), 1e-12)
    _outward, measurement = _source_rim_measurement(
        vertices, faces, boundary_ids, proposed
    )
    coherent = (
        measurement.get("status") == "measured"
        and float(measurement["normal_coherence"]) > COHERENT_RIM_THRESHOLD
    )
    flipped = bool(
        coherent
        and float(measurement["outward_dot_inward"]) > OUTWARD_DOT_TOLERANCE
    )
    resolved = -proposed if flipped else proposed
    resolved_dot = (
        -float(measurement["outward_dot_inward"])
        if flipped
        else measurement.get("outward_dot_inward")
    )
    return resolved, {
        **measurement,
        "resolution": "flipped_by_coherent_source_rim" if flipped else (
            "kept_by_coherent_source_rim" if coherent else "kept_component_fallback"
        ),
        "axis_flipped": flipped,
        "proposed_outward_dot_inward": measurement.get("outward_dot_inward"),
        "resolved_outward_dot_inward": resolved_dot,
    }


def audit_inward_axis(vertices, faces, boundary_ids, inward):
    """Reject a connector axis that still points out of a coherent source rim."""
    _outward, record = _source_rim_measurement(
        vertices, faces, boundary_ids, inward
    )
    if (
        record.get("status") == "measured"
        and float(record["normal_coherence"]) > COHERENT_RIM_THRESHOLD
        and float(record["outward_dot_inward"]) > OUTWARD_DOT_TOLERANCE
    ):
        raise ValueError(
            "connector inward axis points outside the oriented source surface: "
            f"outward_dot_inward={record['outward_dot_inward']:.6f}, "
            f"source_rim_faces={record['source_rim_faces']}"
        )
    return record
