"""Deterministic local peg-and-socket geometry.

This module deliberately separates two jobs which the legacy inward workflow
combined: the full paint boundary closes with a shallow backing surface, while
only a compact interior footprint receives the deep assembly connector.

The synthetic builder is intentionally boolean-free.  Every surface is sewn
from shared rings so tests can reject open edges before this topology is wired
into vendor meshes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
import trimesh

from .boolean_cutters import (
    build_protected_exterior_overshoot_proxy,
)
from .mesh import remove_new_redundant_boolean_micro_shells
from .reporting import runtime_log


DEFAULT_BOOLEAN_COLLAPSED_FACE_RATIO = 0.005
MAXIMUM_PRINT_NEUTRAL_COLLAPSED_FACE_RATIO = 0.005
MAXIMUM_PRINT_NEUTRAL_COLLAPSED_FACE_EDGE_MM = 0.25


def _mesh64_arrays(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact dense array contract required by ``Mesh64``.

    Trimesh may expose an empty/singleton tracked face view as a one-dimensional
    array.  Pybind then reports only an unhelpful overload mismatch.  Normalise
    both dimensions here so every Manifold entry point gets the same contract.
    """
    vertices = np.require(
        np.asarray(mesh.vertices, dtype=np.float64).reshape((-1, 3)),
        dtype=np.float64,
        requirements=("C", "A", "O", "W"),
    )
    faces = np.require(
        np.asarray(mesh.faces, dtype=np.uint64).reshape((-1, 3)),
        dtype=np.uint64,
        requirements=("C", "A", "O", "W"),
    )
    return vertices, faces


def _manifold64(mesh: trimesh.Trimesh, *, tolerance_mm: float = 0.0):
    import manifold3d

    vertices, faces = _mesh64_arrays(mesh)
    return manifold3d.Manifold(
        manifold3d.Mesh64(
            vert_properties=vertices,
            tri_verts=faces,
            tolerance=max(float(tolerance_mm), 0.0),
        )
    )


def _trimesh_from_manifold64(manifold) -> trimesh.Trimesh:
    exported = manifold.to_mesh64()
    return trimesh.Trimesh(
        vertices=np.asarray(exported.vert_properties[:, :3], dtype=np.float64),
        faces=np.asarray(exported.tri_verts, dtype=np.int64),
        process=False,
    )


def _reference_volume_after_redundant_shell_cleanup(
    reference_volume_mm3: float,
    cleanup_record: dict[str, object],
) -> float:
    """Return the expected retained volume after an audited shell removal."""
    removed_volume = max(float(cleanup_record.get("removed_volume_mm3", 0.0)), 0.0)
    return max(float(reference_volume_mm3) - removed_volume, 0.0)


def _audit_boolean_candidate(
    candidate: trimesh.Trimesh,
    *,
    reference_volume: float,
    movement_tolerance_mm: float,
    inherited_collapsed_face_budget: int = 0,
    maximum_new_collapsed_face_ratio: float = DEFAULT_BOOLEAN_COLLAPSED_FACE_RATIO,
    collapsed_seam_volume_envelope_cap_mm3: float = 1e-5,
    topology_healthy_export_volume_envelope_cap_mm3: float = 0.0,
) -> tuple[bool, dict[str, object]]:
    candidate.remove_unreferenced_vertices()
    trimesh.repair.fix_winding(candidate)
    trimesh.repair.fix_normals(candidate)
    faces = np.asarray(candidate.faces, dtype=np.int64).reshape((-1, 3))
    triangles = np.asarray(candidate.vertices, dtype=np.float64)[faces]
    double_areas = np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )
    triangle_edge_lengths = np.stack(
        (
            np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
            np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
            np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
        ),
        axis=1,
    )
    scale = max(float(np.linalg.norm(np.ptp(candidate.vertices, axis=0))), 1.0)
    minimum_double_area = scale * scale * 1e-14
    edge_counts = np.bincount(candidate.edges_unique_inverse)
    collapsed_mask = double_areas <= minimum_double_area
    degenerate_faces = int(np.count_nonzero(collapsed_mask))
    collapsed_maximum_edges = (
        np.max(triangle_edge_lengths[collapsed_mask], axis=1)
        if degenerate_faces
        else np.zeros(0, dtype=np.float64)
    )
    collapsed_maximum_edge_mm = (
        float(np.max(collapsed_maximum_edges)) if degenerate_faces else 0.0
    )
    collapsed_p95_maximum_edge_mm = (
        float(np.percentile(collapsed_maximum_edges, 95.0))
        if degenerate_faces
        else 0.0
    )
    inherited_budget = max(int(inherited_collapsed_face_budget), 0)
    new_collapsed_faces = max(degenerate_faces - inherited_budget, 0)
    new_collapsed_face_ratio = float(new_collapsed_faces / max(len(faces), 1))
    maximum_ratio = min(
        max(float(maximum_new_collapsed_face_ratio), 0.0),
        DEFAULT_BOOLEAN_COLLAPSED_FACE_RATIO,
    )
    microface_ratio_limit = MAXIMUM_PRINT_NEUTRAL_COLLAPSED_FACE_RATIO
    microface_edge_limit_mm = MAXIMUM_PRINT_NEUTRAL_COLLAPSED_FACE_EDGE_MM
    standard_ratio_accepted = bool(
        new_collapsed_faces > 0
        and new_collapsed_face_ratio <= maximum_ratio + 1e-15
    )
    microface_accepted = bool(
        new_collapsed_faces > 0
        and new_collapsed_face_ratio <= microface_ratio_limit + 1e-15
        and collapsed_maximum_edge_mm <= microface_edge_limit_mm + 1e-12
    )
    volume_delta = abs(float(abs(candidate.volume)) - float(reference_volume))
    # Simplifying a few accepted collinear seam faces can change the numerical
    # volume by more than the fixed 1e-6 mm^3 floor even though the entire
    # change is confined to their microscopic edge envelope.  Bound that
    # allowance by twice the aggregate edge cube and cap it at the requested
    # microscopic volume envelope; ordinary strict calls retain 1e-5 mm^3.
    # ordinary faces and larger geometric movement receive no extra budget.
    volume_envelope_cap = max(
        float(collapsed_seam_volume_envelope_cap_mm3),
        0.0,
    )
    collapsed_seam_volume_envelope_mm3 = (
        min(
            volume_envelope_cap,
            2.0
            * float(new_collapsed_faces)
            * float(collapsed_maximum_edge_mm) ** 3,
        )
        if standard_ratio_accepted or microface_accepted
        else 0.0
    )
    topology_healthy_export_volume_envelope_mm3 = (
        max(float(topology_healthy_export_volume_envelope_cap_mm3), 0.0)
        if (
            bool(candidate.is_watertight)
            and bool(candidate.is_winding_consistent)
            and int(np.count_nonzero(edge_counts == 1)) == 0
            and int(np.count_nonzero(edge_counts > 2)) == 0
            and degenerate_faces == 0
        )
        else 0.0
    )
    volume_tolerance = max(
        1e-6,
        float(reference_volume) * 1e-9,
        float(candidate.area) * float(movement_tolerance_mm) * 2.0,
        collapsed_seam_volume_envelope_mm3,
        topology_healthy_export_volume_envelope_mm3,
    )
    record = {
        "faces": int(len(candidate.faces)),
        "watertight": bool(candidate.is_watertight),
        "winding_consistent": bool(candidate.is_winding_consistent),
        "boundary_edges": int(np.count_nonzero(edge_counts == 1)),
        "over_shared_edges": int(np.count_nonzero(edge_counts > 2)),
        "degenerate_faces": degenerate_faces,
        "inherited_collapsed_face_budget": inherited_budget,
        "new_collapsed_faces": new_collapsed_faces,
        "new_collapsed_face_ratio": new_collapsed_face_ratio,
        "maximum_new_collapsed_face_ratio": maximum_ratio,
        "ratio_accepted_collapsed_faces": standard_ratio_accepted,
        "micro_collapsed_faces_accepted": microface_accepted,
        "collapsed_face_acceptance_mode": (
            "ratio_accepted"
            if standard_ratio_accepted
            else "print_neutral_microfaces"
            if microface_accepted
            else "rejected"
        ),
        "collapsed_maximum_edge_mm": collapsed_maximum_edge_mm,
        "collapsed_p95_maximum_edge_mm": collapsed_p95_maximum_edge_mm,
        "maximum_print_neutral_collapsed_face_ratio": microface_ratio_limit,
        "maximum_print_neutral_collapsed_face_edge_mm": microface_edge_limit_mm,
        "minimum_double_area_mm2": float(double_areas.min()),
        "volume_delta_mm3": float(volume_delta),
        "volume_tolerance_mm3": float(volume_tolerance),
        "collapsed_seam_volume_envelope_mm3": float(
            collapsed_seam_volume_envelope_mm3
        ),
        "collapsed_seam_volume_envelope_cap_mm3": volume_envelope_cap,
        "topology_healthy_export_volume_envelope_mm3": float(
            topology_healthy_export_volume_envelope_mm3
        ),
        "topology_healthy_export_volume_envelope_cap_mm3": max(
            float(topology_healthy_export_volume_envelope_cap_mm3),
            0.0,
        ),
    }
    valid = (
        bool(record["watertight"])
        and bool(record["winding_consistent"])
        and record["boundary_edges"] == 0
        and record["over_shared_edges"] == 0
        and (
            degenerate_faces <= inherited_budget
            or standard_ratio_accepted
            or microface_accepted
        )
        and volume_delta <= volume_tolerance
    )
    return bool(valid), record


def _validated_boolean_manifold64_with_kernel(
    manifold,
    *,
    inherited_collapsed_face_budget: int = 0,
    maximum_new_collapsed_face_ratio: float = DEFAULT_BOOLEAN_COLLAPSED_FACE_RATIO,
    topology_healthy_export_volume_envelope_cap_mm3: float = 0.0,
) -> tuple[trimesh.Trimesh, dict[str, object], object]:
    """Export a Boolean while retaining Manifold's closed topology.

    Dense vendor meshes can leave numerically collinear triangles at an exact
    Boolean seam.  Deleting those triangles after export opens or over-shares
    edges.  Manifold's own simplifier instead collapses the seam *inside* its
    solid topology, so try the smallest sub-micron tolerance that produces a
    clean indexed mesh.  No hole filling or child-geometry mutation occurs.
    """
    reference_volume = float(abs(manifold.volume()))
    # Boolean provenance deliberately prevents simplification across faces
    # originating in different operands.  Resetting the result as one new
    # original solid removes only those ancestry barriers and retriangulates
    # the same boundary; it does not move the surface.
    clean_manifold = manifold.as_original()
    attempts: list[dict[str, object]] = []
    for tolerance_mm in (0.0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5):
        candidate_manifold = (
            clean_manifold
            if tolerance_mm == 0.0
            else clean_manifold.simplify(tolerance_mm)
        )
        candidate = _trimesh_from_manifold64(candidate_manifold)
        valid, audit = _audit_boolean_candidate(
            candidate,
            reference_volume=reference_volume,
            movement_tolerance_mm=float(tolerance_mm),
            inherited_collapsed_face_budget=inherited_collapsed_face_budget,
            maximum_new_collapsed_face_ratio=maximum_new_collapsed_face_ratio,
            topology_healthy_export_volume_envelope_cap_mm3=(
                topology_healthy_export_volume_envelope_cap_mm3
            ),
        )
        attempt = {
            "tolerance_mm": float(tolerance_mm),
            **audit,
        }
        attempts.append(attempt)
        if valid:
            return candidate, {
                "applied": bool(tolerance_mm > 0.0),
                "policy": "manifold_topology_simplify_without_hole_filling",
                "boolean_provenance_reset": True,
                "selected_tolerance_mm": float(tolerance_mm),
                "maximum_tolerance_mm": 1e-5,
                "volume_delta_mm3": float(audit["volume_delta_mm3"]),
                "attempts": attempts,
                "accepted_audit": dict(audit),
            }, candidate_manifold
        # A Manifold export may be topologically closed yet contain a handful
        # of zero-area seam faces.  When the source had no inherited collapsed
        # faces, try the dedicated weld/removal repair before increasing the
        # geometric simplification tolerance.  The repair never fills holes
        # and is accepted only after a second closed-topology and volume audit.
        repair_face_limit = min(
            32,
            max(
                8,
                int(math.ceil(float(maximum_new_collapsed_face_ratio) * len(candidate.faces))),
            ),
        )
        if (
            int(inherited_collapsed_face_budget) == 0
            and bool(audit["watertight"])
            and bool(audit["winding_consistent"])
            and int(audit["boundary_edges"]) == 0
            and int(audit["over_shared_edges"]) == 0
            # Local repair is appropriate only for a sparse numerical seam.
            # Hundreds of collapsed faces indicate a bad cutter intersection;
            # walking every vertex star is both slow and incapable of fixing
            # that upstream geometry error.
            and 0 < int(audit["degenerate_faces"]) <= repair_face_limit
            and float(audit["volume_delta_mm3"])
            <= float(audit["volume_tolerance_mm3"])
        ):
            scale = max(
                float(np.linalg.norm(np.ptp(candidate.vertices, axis=0))),
                1.0,
            )
            minimum_double_area = scale * scale * 1e-14
            try:
                repaired, repair_record = _repair_collapsed_boolean_faces(
                    candidate,
                    minimum_double_area=float(minimum_double_area),
                    expected_degenerate_faces=int(audit["degenerate_faces"]),
                )
                repaired_valid, repaired_audit = _audit_boolean_candidate(
                    repaired,
                    reference_volume=reference_volume,
                    movement_tolerance_mm=float(tolerance_mm),
                    inherited_collapsed_face_budget=0,
                    maximum_new_collapsed_face_ratio=maximum_new_collapsed_face_ratio,
                    topology_healthy_export_volume_envelope_cap_mm3=(
                        topology_healthy_export_volume_envelope_cap_mm3
                    ),
                )
                attempt["collapsed_face_repair"] = {
                    **repair_record,
                    "post_repair_audit": repaired_audit,
                }
                if repaired_valid:
                    # Keep the already valid kernel solid resident for later
                    # sequential differences.  The collapsed triangle exists
                    # only in this particular surface export; re-importing the
                    # cleaned triangles is unnecessary and can be far slower
                    # than continuing with the authoritative Manifold object.
                    return repaired, {
                        "applied": True,
                        "policy": (
                            "manifold_topology_simplify_then_"
                            "collapsed_seam_face_removal_without_hole_filling"
                        ),
                        "boolean_provenance_reset": True,
                        "selected_tolerance_mm": float(tolerance_mm),
                        "maximum_tolerance_mm": 1e-5,
                        "volume_delta_mm3": float(
                            repaired_audit["volume_delta_mm3"]
                        ),
                        "collapsed_face_repair": repair_record,
                        "kernel_continuation": (
                            "authoritative_pre_export_manifold"
                        ),
                        "attempts": attempts,
                        "accepted_audit": dict(repaired_audit),
                    }, candidate_manifold
            except ValueError as error:
                attempt["collapsed_face_repair"] = {
                    "applied": False,
                    "error": str(error),
                }

    # The remaining faces are numerically collinear but still topologically
    # closed.  Snap only the already simplified Boolean output to the smallest
    # sub-micron decimal grid, then re-import it into Manifold so the kernel
    # collapses the short edges while preserving the solid topology.
    snap_source_tolerance_mm = 1e-8
    snap_source = _trimesh_from_manifold64(
        clean_manifold.simplify(snap_source_tolerance_mm)
    )
    snap_attempts: list[dict[str, object]] = []
    for grid_mm in (1e-9, 1e-8, 1e-7, 1e-6):
        snapped = snap_source.copy()
        original_vertices = np.asarray(snapped.vertices, dtype=np.float64)
        snapped_vertices = np.rint(original_vertices / grid_mm) * grid_mm
        maximum_coordinate_displacement = float(
            np.max(np.abs(snapped_vertices - original_vertices))
        )
        snapped.vertices = snapped_vertices
        snapped_manifold = _manifold64(snapped, tolerance_mm=grid_mm)
        if snapped_manifold.is_empty():
            snap_attempts.append(
                {
                    "grid_mm": float(grid_mm),
                    "status": str(snapped_manifold.status()),
                    "manifold_import_failed": True,
                }
            )
            continue
        candidate = _trimesh_from_manifold64(
            snapped_manifold.as_original().simplify(grid_mm)
        )
        valid, audit = _audit_boolean_candidate(
            candidate,
            reference_volume=reference_volume,
            movement_tolerance_mm=float(grid_mm),
            inherited_collapsed_face_budget=inherited_collapsed_face_budget,
            maximum_new_collapsed_face_ratio=maximum_new_collapsed_face_ratio,
            topology_healthy_export_volume_envelope_cap_mm3=(
                topology_healthy_export_volume_envelope_cap_mm3
            ),
        )
        snap_attempt = {
            "grid_mm": float(grid_mm),
            "maximum_coordinate_displacement_mm": maximum_coordinate_displacement,
            **audit,
        }
        snap_attempts.append(snap_attempt)
        if valid:
            accepted_manifold = snapped_manifold.as_original().simplify(grid_mm)
            return candidate, {
                "applied": True,
                "policy": "manifold_submicron_grid_snap_without_hole_filling",
                "boolean_provenance_reset": True,
                "pre_snap_simplify_tolerance_mm": snap_source_tolerance_mm,
                "selected_grid_mm": float(grid_mm),
                "maximum_grid_mm": 1e-6,
                "maximum_coordinate_displacement_mm": maximum_coordinate_displacement,
                "volume_delta_mm3": float(audit["volume_delta_mm3"]),
                "simplify_attempts": attempts,
                "snap_attempts": snap_attempts,
                "accepted_audit": dict(audit),
            }, accepted_manifold
    raise ValueError(
        "socket boolean subtraction remains invalid after Manifold topology "
        "simplification and sub-micron grid snap: "
        f"simplify_attempts={attempts}, snap_attempts={snap_attempts}"
    )


def _validated_boolean_manifold64(manifold) -> tuple[trimesh.Trimesh, dict[str, object]]:
    """Compatibility wrapper for callers which only need the audited mesh."""

    candidate, record, _accepted_manifold = _validated_boolean_manifold64_with_kernel(
        manifold
    )
    return candidate, record


def build_clearance_cutter_from_final_part(
    part_mesh: trimesh.Trimesh,
    *,
    radial_clearance_mm: float = 0.05,
) -> tuple[trimesh.Trimesh, dict[str, object]]:
    """Dilate one final emitted part into its authoritative female cutter.

    The final child is the sole geometry source.  Minkowski dilation adds a
    uniform manufacturing clearance without using per-vertex normals, so a
    smooth curved surface cannot turn into the old normal-driven hill field.
    """
    clearance = max(float(radial_clearance_mm), 0.0)
    if clearance <= 1e-9:
        return part_mesh.copy(), {
            "cutter_source": "complete_emitted_direct_child_solid",
            "dilation_method": "none",
            "radial_clearance_mm": 0.0,
        }
    source_mesh = part_mesh.copy()
    if source_mesh.is_watertight and not source_mesh.is_winding_consistent:
        trimesh.repair.fix_winding(source_mesh)
        trimesh.repair.fix_normals(source_mesh)
    if not source_mesh.is_watertight or not source_mesh.is_winding_consistent:
        edge_counts = np.bincount(
            np.asarray(source_mesh.edges_unique_inverse, dtype=np.int64),
            minlength=len(source_mesh.edges_unique),
        )
        raise ValueError(
            "complete child must be closed before clearance dilation: "
            f"watertight={bool(source_mesh.is_watertight)},"
            f"winding_consistent={bool(source_mesh.is_winding_consistent)},"
            f"open_edges={int(np.count_nonzero(edge_counts == 1))},"
            f"nonmanifold_edges={int(np.count_nonzero(edge_counts > 2))}"
        )
    import manifold3d

    source = _manifold64(source_mesh)
    if source.is_empty():
        raise ValueError(f"complete child cannot enter Manifold: status={source.status()}")
    # One refined octahedral kernel is convex and deterministic.  It gives
    # uniform clearance while keeping the Minkowski product bounded.
    kernel = manifold3d.Manifold.sphere(clearance, 4)
    dilated = source.minkowski_sum(kernel)
    if dilated.is_empty():
        raise ValueError(f"clearance Minkowski dilation failed: status={dilated.status()}")
    exported = dilated.to_mesh64()
    cutter = trimesh.Trimesh(
        vertices=np.asarray(exported.vert_properties[:, :3], dtype=np.float64),
        faces=np.asarray(exported.tri_verts, dtype=np.int64),
        process=False,
    )
    cutter.remove_unreferenced_vertices()
    trimesh.repair.fix_winding(cutter)
    trimesh.repair.fix_normals(cutter)
    if not cutter.is_watertight or not cutter.is_winding_consistent:
        raise ValueError("dilated complete-child cutter is not a closed solid")
    return cutter, {
        "cutter_source": "complete_emitted_direct_child_solid",
        "dilation_method": "manifold_minkowski_sphere",
        "kernel_circular_segments": 4,
        "radial_clearance_mm": float(clearance),
        "total_added_fit_clearance_mm": float(2.0 * clearance),
        "source_faces": int(len(source_mesh.faces)),
        "cutter_faces": int(len(cutter.faces)),
        "source_volume_mm3": float(abs(source_mesh.volume)),
        "cutter_volume_mm3": float(abs(cutter.volume)),
    }


def build_boolean_cutter_proxy_from_final_part(
    part_mesh: trimesh.Trimesh,
    *,
    radial_clearance_mm: float = 0.0,
    exterior_overshoot_mm: float = 0.01,
) -> tuple[trimesh.Trimesh, dict[str, object]]:
    """Derive a Boolean-only cutter without changing the emitted child.

    The source face prefix and its shared rim vertices are moved outside the
    restored parent.  Manufacturing clearance is applied only to that already
    non-coincident closed proxy.
    """

    source_face_count = int(
        part_mesh.metadata.get("protected_source_face_count", 0)
    )
    proxy, proxy_record = build_protected_exterior_overshoot_proxy(
        part_mesh,
        source_face_count=source_face_count,
        overshoot_mm=exterior_overshoot_mm,
    )
    cutter, clearance_record = build_clearance_cutter_from_final_part(
        proxy,
        radial_clearance_mm=radial_clearance_mm,
    )
    record = {
        "cutter_source": "complete_emitted_child_boolean_proxy",
        "printable_child_authoritative": True,
        "printable_child_mutated": False,
        "source_surface_contact_policy": "exterior_overshoot_no_coplanar_cap",
        **proxy_record,
        "clearance": clearance_record,
        "radial_clearance_mm": float(radial_clearance_mm),
        "total_added_fit_clearance_mm": float(2.0 * radial_clearance_mm),
        "cutter_faces": int(len(cutter.faces)),
        "cutter_vertices": int(len(cutter.vertices)),
        "cutter_volume_mm3": float(abs(cutter.volume)),
    }
    cutter.metadata["boolean_cutter_proxy"] = dict(record)
    return cutter, record


@dataclass(frozen=True)
class AssemblyInterfacePolicy:
    """Choose one coherent construction strategy for an assembly interface."""

    interface_geometry: str
    geometry_source: str
    complete_child_boolean: bool
    authoritative_clearance_mm: float
    exterior_overshoot_mm: float

    def as_record(self) -> dict[str, object]:
        return {
            "interface_geometry": self.interface_geometry,
            "geometry_source": self.geometry_source,
            "complete_child_boolean": self.complete_child_boolean,
            "authoritative_clearance_mm": self.authoritative_clearance_mm,
            "exterior_overshoot_mm": self.exterior_overshoot_mm,
        }


def assembly_interface_policy(interface_geometry: str) -> AssemblyInterfacePolicy:
    """Return the topology strategy instead of scattering mode conditionals.

    A compact local connector is a separate closed solid, so subtracting its
    final emitted geometry is authoritative.  A boundary extrusion shares its
    entire visible rim with the parent; its male and female surfaces must be
    generated from the same source-patch template.  Re-subtracting the complete
    visible child creates thousands of coplanar contacts and is intentionally
    forbidden for that mode.
    """
    mode = str(interface_geometry)
    if mode == "local-connector":
        return AssemblyInterfacePolicy(
            interface_geometry=mode,
            geometry_source="complete_emitted_child_boolean_proxy",
            complete_child_boolean=True,
            authoritative_clearance_mm=0.0,
            exterior_overshoot_mm=0.01,
        )
    if mode == "boundary-extrusion":
        return AssemblyInterfacePolicy(
            interface_geometry=mode,
            geometry_source="shared_source_patch_complementary_surfaces",
            complete_child_boolean=False,
            authoritative_clearance_mm=0.0,
            exterior_overshoot_mm=0.0,
        )
    raise ValueError(f"unknown interface geometry: {interface_geometry!r}")


def authoritative_complete_child_clearance_mm(interface_geometry: str) -> float:
    """Compatibility accessor for the policy-owned Boolean clearance."""
    return float(
        assembly_interface_policy(interface_geometry).authoritative_clearance_mm
    )


@dataclass(frozen=True)
class LocalConnectorSpec:
    """Manufacturing dimensions for one child-male / parent-female joint."""

    peg_width_mm: float = 4.0
    peg_length_mm: float = 4.0
    engagement_depth_mm: float = 5.0
    total_clearance_mm: float = 0.60
    socket_bottom_clearance_mm: float = 0.30
    socket_mouth_chamfer_mm: float = 0.80
    peg_tip_chamfer_mm: float = 0.0
    corner_radius_mm: float = 0.45
    full_boundary_backing_depth_mm: float = 0.80
    elastic_shrink_applied: bool = False
    backing_elastic_shrink_applied: bool = False
    engagement_elastic_shrink_applied: bool = False
    elastic_backing_scale: float = 1.0
    elastic_engagement_scale: float = 1.0
    elastic_lateral_scale: float = 1.0
    nominal_full_boundary_backing_depth_mm: float = 0.80
    nominal_engagement_depth_mm: float = 5.0
    minimum_elastic_backing_depth_mm: float = 0.45
    backing_safety_limit_mm: float = 0.80
    total_safety_limit_mm: float = 5.0
    compact_peg_enabled: bool = True
    slope_validation_mode: str = "strict"
    surface_validation_mode: str = "strict"

    def __post_init__(self) -> None:
        positive = {
            "peg_width_mm": self.peg_width_mm,
            "peg_length_mm": self.peg_length_mm,
            "socket_mouth_chamfer_mm": self.socket_mouth_chamfer_mm,
        }
        for name, value in positive.items():
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be positive")
        nonnegative = {
            "total_clearance_mm": self.total_clearance_mm,
            "engagement_depth_mm": self.engagement_depth_mm,
            "socket_bottom_clearance_mm": self.socket_bottom_clearance_mm,
            "peg_tip_chamfer_mm": self.peg_tip_chamfer_mm,
            "corner_radius_mm": self.corner_radius_mm,
            "full_boundary_backing_depth_mm": self.full_boundary_backing_depth_mm,
            "elastic_backing_scale": self.elastic_backing_scale,
            "elastic_engagement_scale": self.elastic_engagement_scale,
            "elastic_lateral_scale": self.elastic_lateral_scale,
            "nominal_full_boundary_backing_depth_mm": (
                self.nominal_full_boundary_backing_depth_mm
            ),
            "minimum_elastic_backing_depth_mm": (
                self.minimum_elastic_backing_depth_mm
            ),
            "nominal_engagement_depth_mm": self.nominal_engagement_depth_mm,
            "backing_safety_limit_mm": self.backing_safety_limit_mm,
            "total_safety_limit_mm": self.total_safety_limit_mm,
        }
        for name, value in nonnegative.items():
            if float(value) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if self.corner_radius_mm >= min(self.peg_width_mm, self.peg_length_mm) / 2.0:
            raise ValueError("corner_radius_mm must be smaller than half the peg width and length")
        if self.peg_tip_chamfer_mm >= min(self.peg_width_mm, self.peg_length_mm) / 2.0:
            raise ValueError("peg_tip_chamfer_mm consumes the peg footprint")
        if self.engagement_depth_mm <= 1e-9 and self.peg_tip_chamfer_mm > 1e-9:
            raise ValueError("zero engagement cannot have a peg tip chamfer")
        if (
            self.engagement_depth_mm > 1e-9
            and self.peg_tip_chamfer_mm >= self.engagement_depth_mm
        ):
            raise ValueError("peg_tip_chamfer_mm must be shorter than engagement_depth_mm")
        if self.elastic_backing_scale > 1.0 + 1e-9:
            raise ValueError("elastic_backing_scale cannot enlarge the connector")
        if self.elastic_lateral_scale > 1.0 + 1e-9:
            raise ValueError("elastic_lateral_scale cannot enlarge the connector")
        if self.elastic_engagement_scale > 1.0 + 1e-9:
            raise ValueError("elastic_engagement_scale cannot enlarge the connector")
        if self.compact_peg_enabled != bool(self.engagement_depth_mm > 1e-9):
            raise ValueError("compact_peg_enabled must match positive engagement depth")
        if self.slope_validation_mode not in {"strict", "advisory"}:
            raise ValueError("slope_validation_mode must be strict or advisory")
        if self.surface_validation_mode not in {"strict", "advisory"}:
            raise ValueError("surface_validation_mode must be strict or advisory")


def connector_profile(
    spec: LocalConnectorSpec,
    *,
    contact_width_mm: float,
    contact_length_mm: float,
) -> dict[str, float]:
    """Return auditable dimensions; the mouth is a true one-to-one 45° ramp."""

    contact_area = float(contact_width_mm) * float(contact_length_mm)
    peg_area = float(spec.peg_width_mm) * float(spec.peg_length_mm)
    if contact_area <= 0.0:
        raise ValueError("contact dimensions must be positive")
    throat_width = float(spec.peg_width_mm) + float(spec.total_clearance_mm)
    throat_length = float(spec.peg_length_mm) + float(spec.total_clearance_mm)
    chamfer = float(spec.socket_mouth_chamfer_mm)
    return {
        "peg_width_mm": float(spec.peg_width_mm),
        "peg_length_mm": float(spec.peg_length_mm),
        "peg_area_mm2": peg_area,
        "peg_area_ratio": peg_area / contact_area,
        "engagement_depth_mm": float(spec.engagement_depth_mm),
        "total_clearance_mm": float(spec.total_clearance_mm),
        "socket_throat_width_mm": throat_width,
        "socket_throat_length_mm": throat_length,
        "socket_mouth_width_mm": throat_width + 2.0 * chamfer,
        "socket_mouth_length_mm": throat_length + 2.0 * chamfer,
        "mouth_per_side_expansion_mm": chamfer,
        "socket_mouth_chamfer_mm": chamfer,
        "mouth_slope_degrees": math.degrees(math.atan2(chamfer, chamfer)),
        "socket_depth_mm": float(spec.engagement_depth_mm + spec.socket_bottom_clearance_mm),
        "full_boundary_backing_depth_mm": float(spec.full_boundary_backing_depth_mm),
    }


def _stable_basis(inward: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.asarray(inward, dtype=np.float64)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    candidates = np.eye(3, dtype=np.float64)
    reference = candidates[int(np.argmin(np.abs(candidates @ axis)))]
    u = np.cross(axis, reference)
    u /= max(float(np.linalg.norm(u)), 1e-12)
    v = np.cross(axis, u)
    v /= max(float(np.linalg.norm(v)), 1e-12)
    return axis, u, v


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    x, y = float(point[0]), float(point[1])
    inside = False
    for index in range(len(polygon)):
        left = polygon[index]
        right = polygon[(index + 1) % len(polygon)]
        if (float(left[1]) > y) == (float(right[1]) > y):
            continue
        denominator = float(right[1]) - float(left[1])
        crossing_x = float(left[0]) + (y - float(left[1])) * (
            float(right[0]) - float(left[0])
        ) / denominator
        if x < crossing_x:
            inside = not inside
    return inside


def _points_in_polygon_batch(
    points: np.ndarray,
    polygon: np.ndarray,
    *,
    chunk_size: int = 256,
) -> np.ndarray:
    """Vectorized equivalent of ``_point_in_polygon`` in bounded memory."""

    query = np.asarray(points, dtype=np.float64).reshape((-1, 2))
    left = np.asarray(polygon, dtype=np.float64)
    right = np.roll(left, -1, axis=0)
    denominator = right[:, 1] - left[:, 1]
    safe_denominator = np.where(
        np.abs(denominator) <= 1e-15,
        1.0,
        denominator,
    )
    result = np.zeros(len(query), dtype=bool)
    for start in range(0, len(query), max(int(chunk_size), 1)):
        block = query[start : start + max(int(chunk_size), 1)]
        x = block[:, 0, None]
        y = block[:, 1, None]
        crosses = (left[:, 1][None, :] > y) != (right[:, 1][None, :] > y)
        crossing_x = left[:, 0][None, :] + (
            (y - left[:, 1][None, :])
            * (right[:, 0] - left[:, 0])[None, :]
            / safe_denominator[None, :]
        )
        result[start : start + len(block)] = np.logical_xor.reduce(
            crosses & (x < crossing_x),
            axis=1,
        )
    return result


def _point_segment_distance(point: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    edge = right - left
    length_sq = float(np.dot(edge, edge))
    if length_sq <= 1e-18:
        return float(np.linalg.norm(point - left))
    ratio = float(np.dot(point - left, edge)) / length_sq
    ratio = min(max(ratio, 0.0), 1.0)
    return float(np.linalg.norm(point - (left + ratio * edge)))


def _minimum_point_to_polygon_segment_distances(
    points: np.ndarray,
    polygon: np.ndarray,
    *,
    chunk_size: int = 128,
) -> np.ndarray:
    """Return exact point-to-segment minima without Python per-edge loops."""

    query = np.asarray(points, dtype=np.float64).reshape((-1, 2))
    left = np.asarray(polygon, dtype=np.float64)
    edge = np.roll(left, -1, axis=0) - left
    length_squared = np.einsum("ij,ij->i", edge, edge)
    result = np.full(len(query), np.inf, dtype=np.float64)
    for start in range(0, len(query), max(int(chunk_size), 1)):
        block = query[start : start + max(int(chunk_size), 1)]
        relative = block[:, None, :] - left[None, :, :]
        parameter = np.divide(
            np.einsum("bij,ij->bi", relative, edge),
            length_squared[None, :],
            out=np.zeros((len(block), len(edge)), dtype=np.float64),
            where=length_squared[None, :] > 1e-18,
        )
        parameter = np.clip(parameter, 0.0, 1.0)
        closest = left[None, :, :] + parameter[:, :, None] * edge[None, :, :]
        delta = block[:, None, :] - closest
        result[start : start + len(block)] = np.sqrt(
            np.einsum("bij,bij->bi", delta, delta).min(axis=1)
        )
    return result


def _polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) * 0.5


def _interior_clearance_center(points: np.ndarray) -> tuple[np.ndarray, float]:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    candidates = [points.mean(axis=0)]
    grid_size = 25
    for x in np.linspace(float(minimum[0]), float(maximum[0]), grid_size):
        for y in np.linspace(float(minimum[1]), float(maximum[1]), grid_size):
            sample = np.asarray([x, y], dtype=np.float64)
            if _point_in_polygon(sample, points):
                candidates.append(sample)
    candidate_array = np.asarray(candidates, dtype=np.float64)
    inside = _points_in_polygon_batch(candidate_array, points)
    valid_candidates = candidate_array[inside]
    if not len(valid_candidates):
        raise ValueError("boundary has no safe interior connector location")
    clearances = _minimum_point_to_polygon_segment_distances(
        valid_candidates,
        points,
    )
    best_index = int(np.argmax(clearances))
    best_clearance = float(clearances[best_index])
    if best_clearance <= 1e-6:
        raise ValueError("boundary has no safe interior connector location")
    return np.asarray(valid_candidates[best_index], dtype=np.float64), best_clearance


def plan_local_connector(
    boundary_points: np.ndarray,
    inward: np.ndarray,
    spec: LocalConnectorSpec | None = None,
    *,
    samples: int = 32,
) -> dict[str, object]:
    """Fit one compact rounded-square connector safely inside an interface loop."""

    spec = spec or LocalConnectorSpec(full_boundary_backing_depth_mm=0.0)
    points = np.asarray(boundary_points, dtype=np.float64)
    # Three vertices are enough to define a non-degenerate closed footprint.
    # Tiny auxiliary loops created by a validated Boolean reload can therefore
    # still receive the no-peg axial backing path instead of aborting an
    # otherwise printable semantic detail.  The projected-area and interior-
    # clearance audits below remain authoritative and reject collinear rings.
    if len(points) < 3:
        raise ValueError("a local connector needs at least three boundary points")
    axis, u, v = _stable_basis(inward)
    base_origin = points.mean(axis=0)
    polygon = np.column_stack(((points - base_origin) @ u, (points - base_origin) @ v))
    center_2d, edge_clearance = _interior_clearance_center(polygon)
    area = _polygon_area(polygon)
    if area <= 1e-6:
        raise ValueError("interface projection is degenerate")

    compact_peg_enabled = bool(spec.compact_peg_enabled)
    requested_area = float(spec.peg_width_mm * spec.peg_length_mm)
    area_scale = math.sqrt(max(0.0, 0.12 * area) / max(requested_area, 1e-12))
    corner_distance = math.hypot(spec.peg_width_mm / 2.0, spec.peg_length_mm / 2.0)
    clearance_scale = 0.78 * edge_clearance / max(corner_distance, 1e-12)
    scale = min(1.0, area_scale, clearance_scale)
    if compact_peg_enabled and scale < 0.25:
        raise ValueError(
            f"interface is too small for a protected local connector (scale={scale:.3f})"
        )
    center = (
        base_origin
        + u * float(center_2d[0])
        + v * float(center_2d[1])
        + axis * float(spec.full_boundary_backing_depth_mm)
    )
    per_side_clearance = float(spec.total_clearance_mm) / 2.0
    requested_mouth_chamfer = (
        min(
            float(spec.socket_mouth_chamfer_mm),
            max(float(spec.engagement_depth_mm) * 0.25, 0.0),
        )
        if compact_peg_enabled
        else 0.0
    )
    minimum_mouth_chamfer = min(0.05, requested_mouth_chamfer)

    def footprint_fits(width: float, length: float, radius: float) -> bool:
        samples_2d = _rounded_rectangle_xy(width, length, radius, samples)
        translated = samples_2d + center_2d[None, :]
        return bool(np.all(_points_in_polygon_batch(translated, polygon)))

    backing_reserve = max(float(spec.full_boundary_backing_depth_mm), 0.0)

    def footprint_keeps_backing_reserve(
        width: float,
        length: float,
        radius: float,
    ) -> bool:
        """Reserve room for the complete 45-degree male backing wedge.

        A three millimetre wedge needs three millimetres of lateral room at its
        floor.  The old planner maximized the peg against the source outline,
        then the backing builder had no choice but to stop its taper early and
        grow a vertical skirt.  Fit the peg to the eroded source footprint
        first; socket clearance and its short mouth chamfer remain absolute.
        """
        if backing_reserve <= 1e-9:
            return True
        samples_2d = _rounded_rectangle_xy(width, length, radius, samples)
        translated = samples_2d + center_2d[None, :]
        # Keep a finite annular land at the backing floor.  Merely touching the
        # inset outline is mathematically contained but leaves zero-width
        # triangles around a dense concave ring.
        required_clearance = backing_reserve + max(0.15, 0.05 * backing_reserve)
        if not bool(np.all(_points_in_polygon_batch(translated, polygon))):
            return False
        clearances = _minimum_point_to_polygon_segment_distances(
            translated,
            polygon,
        )
        return bool(np.all(clearances >= required_clearance - 1e-6))

    minimum_scale = max(
        0.25,
        1.0 / max(float(spec.peg_width_mm), 1e-12),
        1.0 / max(float(spec.peg_length_mm), 1e-12),
    )

    def select_dimensions(*, require_backing_reserve: bool):
        if scale + 1e-9 < minimum_scale:
            return None
        for candidate_scale in np.linspace(scale, minimum_scale, 41):
            peg_width = float(spec.peg_width_mm * candidate_scale)
            peg_length = float(spec.peg_length_mm * candidate_scale)
            corner_radius = min(
                float(spec.corner_radius_mm * candidate_scale),
                peg_width * 0.24,
                peg_length * 0.24,
            )
            if require_backing_reserve and not footprint_keeps_backing_reserve(
                peg_width,
                peg_length,
                corner_radius,
            ):
                continue
            throat_width = peg_width + float(spec.total_clearance_mm)
            throat_length = peg_length + float(spec.total_clearance_mm)
            throat_radius = corner_radius + per_side_clearance
            if not footprint_fits(throat_width, throat_length, throat_radius):
                continue
            low = 0.0
            high = requested_mouth_chamfer
            for _ in range(32):
                trial = (low + high) * 0.5
                if footprint_fits(
                    throat_width + 2.0 * trial,
                    throat_length + 2.0 * trial,
                    throat_radius + trial,
                ):
                    low = trial
                else:
                    high = trial
            if low + 1e-9 < minimum_mouth_chamfer:
                continue
            return (
                float(candidate_scale),
                peg_width,
                peg_length,
                corner_radius,
                throat_width,
                throat_length,
                min(requested_mouth_chamfer, low),
            )
        return None

    if compact_peg_enabled:
        selected_dimensions = select_dimensions(require_backing_reserve=True)
        full_backing_taper_reserved = selected_dimensions is not None
        if selected_dimensions is None:
            if backing_reserve > 1e-9:
                # Load-bearing backing has priority over an optional compact
                # peg.  On a narrow interface, removing the peg preserves the
                # full source boundary and avoids forcing a fixed 45-degree
                # lateral reserve that the measured footprint cannot supply.
                compact_peg_enabled = False
                selected_dimensions = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            else:
                selected_dimensions = select_dimensions(require_backing_reserve=False)
        if selected_dimensions is None:
            raise ValueError("interface is too small for a protected local connector mouth")
    else:
        # A zero-depth compact peg is absent, not a coincident pair of rings.
        # Retain center/basis data for the backing inset while assigning no
        # protected footprint or socket-mouth area.
        selected_dimensions = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        full_backing_taper_reserved = True
    (
        scale,
        peg_width,
        peg_length,
        corner_radius,
        throat_width,
        throat_length,
        mouth_chamfer,
    ) = selected_dimensions
    effective_engagement_depth = (
        float(spec.engagement_depth_mm) if compact_peg_enabled else 0.0
    )
    socket_depth = (
        float(effective_engagement_depth + spec.socket_bottom_clearance_mm)
        if compact_peg_enabled else 0.0
    )

    def ring(width: float, length: float, radius: float, depth: float) -> np.ndarray:
        xy = _rounded_rectangle_xy(width, length, radius, samples)
        return (
            center[None, :]
            + xy[:, 0, None] * u[None, :]
            + xy[:, 1, None] * v[None, :]
            + float(depth) * axis[None, :]
        )

    tip_inset = min(
        float(spec.peg_tip_chamfer_mm),
        peg_width * 0.20,
        peg_length * 0.20,
    )
    plan = {
        "center": center,
        "inward": axis,
        "u": u,
        "v": v,
        "samples": int(samples),
        "peg_width_mm": peg_width,
        "peg_length_mm": peg_length,
        "corner_radius_mm": corner_radius,
        "engagement_depth_mm": effective_engagement_depth,
        "total_clearance_mm": float(spec.total_clearance_mm),
        "per_side_clearance_mm": per_side_clearance,
        "socket_depth_mm": socket_depth,
        "socket_mouth_chamfer_mm": mouth_chamfer,
        "requested_socket_mouth_chamfer_mm": requested_mouth_chamfer,
        "adaptive_socket_mouth_chamfer_applied": bool(
            mouth_chamfer < requested_mouth_chamfer - 1e-6
        ),
        "mouth_slope_degrees": 45.0,
        "footprint_area_ratio": (peg_width * peg_length) / area,
        "interface_area_mm2": area,
        "edge_clearance_mm": edge_clearance,
        "fit_scale": scale,
        "full_boundary_backing_depth_mm": float(
            spec.full_boundary_backing_depth_mm
        ),
        "elastic_shrink_applied": bool(spec.elastic_shrink_applied),
        "backing_elastic_shrink_applied": bool(
            spec.backing_elastic_shrink_applied
        ),
        "engagement_elastic_shrink_applied": bool(
            spec.engagement_elastic_shrink_applied
        ),
        "elastic_backing_scale": float(spec.elastic_backing_scale),
        "elastic_engagement_scale": float(spec.elastic_engagement_scale),
        "elastic_lateral_scale": float(spec.elastic_lateral_scale),
        "nominal_full_boundary_backing_depth_mm": float(
            spec.nominal_full_boundary_backing_depth_mm
        ),
        "minimum_elastic_backing_depth_mm": float(
            spec.minimum_elastic_backing_depth_mm
        ),
        "nominal_engagement_depth_mm": float(
            spec.nominal_engagement_depth_mm
        ),
        "backing_safety_limit_mm": float(spec.backing_safety_limit_mm),
        "total_safety_limit_mm": float(spec.total_safety_limit_mm),
        "compact_peg_enabled": bool(compact_peg_enabled),
        "backing_slope_validation_mode": str(spec.slope_validation_mode),
        "backing_surface_validation_mode": str(spec.surface_validation_mode),
        "full_backing_taper_reserved": bool(full_backing_taper_reserved),
        "peg_top": ring(peg_width, peg_length, corner_radius, 0.0),
        "peg_straight": ring(
            peg_width,
            peg_length,
            corner_radius,
            max(float(spec.engagement_depth_mm) - tip_inset, 0.0),
        ),
        "peg_tip": ring(
            peg_width - 2.0 * tip_inset,
            peg_length - 2.0 * tip_inset,
            max(0.05, corner_radius - tip_inset),
            float(spec.engagement_depth_mm),
        ),
        "socket_mouth": ring(
            throat_width + 2.0 * mouth_chamfer,
            throat_length + 2.0 * mouth_chamfer,
            corner_radius + per_side_clearance + mouth_chamfer,
            0.0,
        ),
        "socket_throat": ring(
            throat_width,
            throat_length,
            corner_radius + per_side_clearance,
            mouth_chamfer,
        ),
        "socket_bottom": ring(
            throat_width,
            throat_length,
            corner_radius + per_side_clearance,
            socket_depth,
        ),
    }
    for name in (("peg_top", "socket_mouth") if compact_peg_enabled else ()):
        projected = np.column_stack(
            (
                (np.asarray(plan[name]) - base_origin) @ u,
                (np.asarray(plan[name]) - base_origin) @ v,
            )
        )
        if not all(_point_in_polygon(sample, polygon) for sample in projected):
            raise ValueError(f"planned {name} leaves the source interface")
    if compact_peg_enabled and backing_reserve > 1e-9:
        # Clearance-to-source is only a fast conservative fit heuristic on a
        # concave outline.  The printable backing uses the exact trimmed
        # Clipper contour, which can lose a narrow bay that the heuristic still
        # counts as usable land.  Validate the load-bearing peg footprint
        # against that same physical floor now.  The socket mouth has its own
        # female-side clearance taper and is audited when that side is built.
        # The upstream safety planner treats this
        # deterministic failure as an unsupported compact peg and retries the
        # interface as a backing-only joint.
        from .connector_geometry import topology_safe_planar_inset_ring

        exact_floor = topology_safe_planar_inset_ring(
            points,
            plan,
            backing_reserve,
        )
        if exact_floor is None:
            raise ValueError(
                "interface has no exact full-depth backing contour for compact peg"
            )
        exact_floor_projected = np.column_stack(
            (
                (np.asarray(exact_floor) - base_origin) @ u,
                (np.asarray(exact_floor) - base_origin) @ v,
            )
        )
        peg_top_projected = np.column_stack(
            (
                (np.asarray(plan["peg_top"]) - base_origin) @ u,
                (np.asarray(plan["peg_top"]) - base_origin) @ v,
            )
        )
        if not bool(
            np.all(
                    _points_in_polygon_batch(
                        peg_top_projected,
                        exact_floor_projected,
                )
            )
        ):
            raise ValueError(
                "compact peg leaves the exact full-depth backing contour"
            )
        plan["exact_full_backing_taper_reserve_audited"] = True
    return plan


def build_socket_cutter_from_plan(
    plan: dict[str, object],
    *,
    outside_extension_mm: float = 2.0,
) -> trimesh.Trimesh:
    """Build the watertight negative tool used to cut the matching female socket."""
    mouth = np.asarray(plan["socket_mouth"], dtype=np.float64)
    throat = np.asarray(plan["socket_throat"], dtype=np.float64)
    bottom = np.asarray(plan["socket_bottom"], dtype=np.float64)
    inward = np.asarray(plan["inward"], dtype=np.float64)
    inward /= max(float(np.linalg.norm(inward)), 1e-12)
    outside = mouth - inward[None, :] * max(float(outside_extension_mm), 0.10)
    count = len(mouth)
    if count < 16 or len(throat) != count or len(bottom) != count:
        raise ValueError("socket cutter rings must have one shared sample count")

    vertices = np.concatenate((outside, mouth, throat, bottom), axis=0).tolist()
    faces: list[list[int]] = []

    def connect(first_start: int, second_start: int) -> None:
        for index in range(count):
            nxt = (index + 1) % count
            a = first_start + index
            aj = first_start + nxt
            b = second_start + index
            bj = second_start + nxt
            faces.extend(([a, aj, bj], [a, bj, b]))

    connect(0, count)
    connect(count, 2 * count)
    connect(2 * count, 3 * count)
    outside_center = len(vertices)
    vertices.append(np.asarray(outside, dtype=np.float64).mean(axis=0).tolist())
    bottom_center = len(vertices)
    vertices.append(np.asarray(bottom, dtype=np.float64).mean(axis=0).tolist())
    for index in range(count):
        nxt = (index + 1) % count
        faces.append([index, outside_center, nxt])
        faces.append([3 * count + index, 3 * count + nxt, bottom_center])

    cutter = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=True,
    )
    trimesh.repair.fix_winding(cutter)
    trimesh.repair.fix_normals(cutter)
    if not cutter.is_watertight or not cutter.is_winding_consistent:
        raise ValueError("generated socket boolean cutter is not watertight")
    return cutter


def subtract_socket_cutters(
    parent_mesh: trimesh.Trimesh,
    cutters: list[trimesh.Trimesh],
    *,
    allow_empty_intersection: bool = False,
    inherited_collapsed_face_budget: int = 0,
    maximum_new_collapsed_face_ratio: float = DEFAULT_BOOLEAN_COLLAPSED_FACE_RATIO,
    cleanup_volume_envelope_cap_mm3: float = 1e-5,
    topology_healthy_export_volume_envelope_cap_mm3: float = 0.0,
) -> tuple[trimesh.Trimesh, dict[str, object]]:
    """Subtract socket tools sequentially without unioning overlapping cutters.

    Unioning a full-boundary cavity tool with its compact clearance socket can
    leave coincident vertices along their tangent interface.  Manifold still
    reports the result as closed, but the later parent difference then contains
    zero-area bridge triangles which slicers display as cracks and shards.
    Sequential differences are set-theoretically equivalent and avoid that
    intermediate coplanar union seam.
    """
    if not cutters:
        return parent_mesh.copy(), {
            "boolean_engine": "manifold",
            "socket_cutter_count": 0,
            "intersection_volume_mm3": 0.0,
        }
    parent = parent_mesh.copy()
    if not parent.is_watertight:
        edge_counts = np.bincount(parent.edges_unique_inverse)
        raise ValueError(
            "parent must be watertight before socket boolean subtraction: "
            f"winding_consistent={bool(parent.is_winding_consistent)}, "
            f"boundary_edges={int(np.count_nonzero(edge_counts == 1))}, "
            f"nonmanifold_edges={int(np.count_nonzero(edge_counts > 2))}"
        )
    parent_volume_before = float(abs(parent.volume))
    result = parent
    result_manifold = _manifold64(parent)
    if result_manifold.is_empty():
        raise ValueError(f"parent cannot enter Manifold: status={result_manifold.status()}")
    step_records: list[dict[str, object]] = []
    total_overlap_volume = 0.0
    applied_count = 0
    for cutter_index, cutter in enumerate(cutters):
        step_started = time.perf_counter()
        runtime_log(
            "assembly-boolean",
            "socket_difference_step_start",
            "Applying one sequential child cutter",
            cutter_index=int(cutter_index),
            cutter_count=int(len(cutters)),
            cutter_faces=int(len(cutter.faces)),
            parent_faces_before=int(len(result.faces)),
        )
        if not cutter.is_watertight or not cutter.is_winding_consistent:
            raise ValueError(
                f"socket cutter {cutter_index} is not a valid closed solid"
            )
        cutter_manifold = _manifold64(cutter)
        parent_step_volume = float(abs(result_manifold.volume()))
        difference_manifold = result_manifold - cutter_manifold
        difference_volume = (
            0.0
            if difference_manifold.is_empty()
            else float(abs(difference_manifold.volume()))
        )
        overlap_volume = max(parent_step_volume - difference_volume, 0.0)
        if overlap_volume <= 1e-6:
            if allow_empty_intersection:
                step_records.append(
                    {
                        "cutter_index": int(cutter_index),
                        "intersection_volume_mm3": overlap_volume,
                        "skipped": True,
                        "skip_reason": "already_disjoint_after_previous_difference",
                    }
                )
                runtime_log(
                    "assembly-boolean",
                    "socket_difference_step_skipped",
                    "Sequential child cutter is already disjoint",
                    cutter_index=int(cutter_index),
                    cutter_count=int(len(cutters)),
                    step_elapsed_seconds=round(
                        time.perf_counter() - step_started,
                        3,
                    ),
                )
                continue
            raise ValueError(f"socket cutter {cutter_index} does not intersect the parent solid")
        if difference_manifold.is_empty():
            raise ValueError(
                f"socket boolean subtraction {cutter_index} returned no parent mesh"
            )
        candidate, cleanup_record, accepted_manifold = (
            _validated_boolean_manifold64_with_kernel(
                difference_manifold,
                inherited_collapsed_face_budget=(
                    inherited_collapsed_face_budget
                ),
                maximum_new_collapsed_face_ratio=(
                    maximum_new_collapsed_face_ratio
                ),
                topology_healthy_export_volume_envelope_cap_mm3=(
                    topology_healthy_export_volume_envelope_cap_mm3
                ),
            )
        )
        validated_before_cleanup = candidate.copy()
        candidate, redundant_shell_record = (
            remove_new_redundant_boolean_micro_shells(
                candidate,
                result,
            )
        )
        cleanup_record["redundant_micro_shell_cleanup"] = (
            redundant_shell_record
        )
        if bool(redundant_shell_record.get("applied", False)):
            runtime_log(
                "assembly-boolean",
                "redundant_micro_shells_removed",
                "Removed new near-coplanar Boolean micro-shells already covered by the body",
                cutter_index=int(cutter_index),
                removed_components=int(
                    redundant_shell_record["removed_component_count"]
                ),
                removed_faces=int(
                    redundant_shell_record["removed_face_count"]
                ),
                removed_volume_mm3=float(
                    redundant_shell_record["removed_volume_mm3"]
                ),
            )
            cleaned_valid, cleaned_audit = _audit_boolean_candidate(
                candidate,
                reference_volume=_reference_volume_after_redundant_shell_cleanup(
                    float(difference_volume),
                    redundant_shell_record,
                ),
                movement_tolerance_mm=float(
                    cleanup_record.get("selected_tolerance_mm", 0.0)
                ),
                inherited_collapsed_face_budget=(
                    inherited_collapsed_face_budget
                ),
                maximum_new_collapsed_face_ratio=(
                    maximum_new_collapsed_face_ratio
                ),
                collapsed_seam_volume_envelope_cap_mm3=(
                    cleanup_volume_envelope_cap_mm3
                ),
                topology_healthy_export_volume_envelope_cap_mm3=(
                    topology_healthy_export_volume_envelope_cap_mm3
                ),
            )
            if not cleaned_valid:
                candidate = validated_before_cleanup
                redundant_shell_record.update(
                    applied=False, reverted=True,
                    skipped_reason="optional_cleanup_failed_audit_keep_validated_boolean",
                    rejected_cleanup_audit=dict(cleaned_audit),
                )
                runtime_log(
                    "assembly-boolean", "optional_cleanup_reverted",
                    "可选碎屑清理未通过，保留已验证的清理前布尔结果",
                    cutter_index=int(cutter_index),
                )
            else:
                accepted_manifold = _manifold64(candidate)
                if accepted_manifold.is_empty():
                    raise ValueError("cleaned Boolean mesh cannot re-enter Manifold")
                cleanup_record["accepted_audit"] = dict(cleaned_audit)
                cleanup_record["post_cleanup_reference_volume_mm3"] = (
                    _reference_volume_after_redundant_shell_cleanup(
                        float(difference_volume), redundant_shell_record,
                    )
                )
                cleanup_record["volume_delta_mm3"] = float(cleaned_audit["volume_delta_mm3"])
        triangles = np.asarray(candidate.vertices, dtype=np.float64)[
            np.asarray(candidate.faces, dtype=np.int64)
        ]
        double_areas = np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            ),
            axis=1,
        )
        step_records.append(
            {
                "cutter_index": int(cutter_index),
                "intersection_volume_mm3": overlap_volume,
                "faces_after": int(len(candidate.faces)),
                "vertices_after": int(len(candidate.vertices)),
                "minimum_double_area_mm2": float(double_areas.min()),
                "boolean_coordinate_precision": "float64_mesh64",
                "collapsed_face_cleanup": cleanup_record,
                "skipped": False,
            }
        )
        total_overlap_volume += overlap_volume
        applied_count += 1
        result = candidate
        result_manifold = accepted_manifold
        accepted_audit = cleanup_record.get("accepted_audit", {})
        runtime_log(
            "assembly-boolean",
            "socket_difference_step_done",
            "Sequential child cutter passed Boolean audit",
            cutter_index=int(cutter_index),
            cutter_count=int(len(cutters)),
            faces_after=int(len(candidate.faces)),
            new_collapsed_faces=int(
                accepted_audit.get("new_collapsed_faces", 0)
            ),
            ratio_accepted=bool(
                accepted_audit.get("ratio_accepted_collapsed_faces", False)
            ),
            microfaces_accepted=bool(
                accepted_audit.get("micro_collapsed_faces_accepted", False)
            ),
            collapsed_maximum_edge_mm=float(
                accepted_audit.get("collapsed_maximum_edge_mm", 0.0)
            ),
            step_elapsed_seconds=round(
                time.perf_counter() - step_started,
                3,
            ),
        )
    if not applied_count:
        if allow_empty_intersection:
            return result, {
                "boolean_engine": "manifold",
                "boolean_strategy": "sequential_difference_without_cutter_union",
                "socket_cutter_count": int(len(cutters)),
                "intersection_volume_mm3": 0.0,
                "parent_volume_before_mm3": parent_volume_before,
                "parent_volume_after_mm3": parent_volume_before,
                "boolean_skipped": True,
                "boolean_skip_reason": "already_disjoint_after_clearance_socket",
                "cutter_steps": step_records,
            }
        raise ValueError("no socket cutter intersected the parent solid")
    final_applied_step = next(
        (
            record
            for record in reversed(step_records)
            if not bool(record.get("skipped", False))
        ),
        {},
    )
    final_accepted_audit = (
        final_applied_step.get("collapsed_face_cleanup", {})
        .get("accepted_audit", {})
    )
    final_new_collapsed_faces = int(
        final_accepted_audit.get("new_collapsed_faces", 0)
    )
    return result, {
        "boolean_engine": "manifold",
        "boolean_strategy": "sequential_difference_without_cutter_union",
        "socket_cutter_count": int(len(cutters)),
        "applied_socket_cutter_count": int(applied_count),
        "intersection_volume_mm3": float(total_overlap_volume),
        "parent_volume_before_mm3": parent_volume_before,
        "parent_volume_after_mm3": float(abs(result.volume)),
        "cutter_steps": step_records,
        # Every step is audited against the same inherited source budget, so
        # the final applied step describes the collapsed faces that actually
        # remain in the exported mesh.  Taking the maximum across historical
        # steps can over-budget faces that a later Boolean removed.
        "ratio_accepted_new_collapsed_faces": int(
            final_new_collapsed_faces
            if bool(
                final_accepted_audit.get(
                    "ratio_accepted_collapsed_faces",
                    False,
                )
            )
            else 0
        ),
        "micro_accepted_new_collapsed_faces": int(
            final_new_collapsed_faces
            if bool(
                final_accepted_audit.get(
                    "micro_collapsed_faces_accepted",
                    False,
                )
            )
            else 0
        ),
        "maximum_new_collapsed_face_ratio": min(
            max(float(maximum_new_collapsed_face_ratio), 0.0),
            DEFAULT_BOOLEAN_COLLAPSED_FACE_RATIO,
        ),
    }


def _repair_collapsed_boolean_faces(
    mesh: trimesh.Trimesh,
    *,
    minimum_double_area: float,
    expected_degenerate_faces: int,
) -> tuple[trimesh.Trimesh, dict[str, object]]:
    """Remove only numerically collapsed Manifold seam faces.

    A Boolean can return a topologically closed indexed mesh with a handful of
    almost-zero-area triangles where the cutter exits on a densely sampled
    source edge.  First remove a collapsed seam vertex by retriangulating its
    complete existing one-ring boundary without a centre fan or any new
    vertex.  This is a local edge-collapse operation, not generic hole filling:
    every replacement triangle is confined to the old vertex star.  Fall back
    to sub-micron coincident-position welding for detached ghost triangles.
    Every candidate must remain closed, consistently wound and manifold, with
    negligible volume change.
    """
    source_volume = float(abs(mesh.volume))
    source_faces = int(len(mesh.faces))
    maximum_removed_faces = max(32, int(expected_degenerate_faces) * 8)
    attempts: list[dict[str, object]] = []

    star_candidate = mesh.copy()
    star_steps: list[dict[str, object]] = []
    maximum_star_steps = max(4, int(expected_degenerate_faces) * 3)
    for _step in range(maximum_star_steps):
        faces = np.asarray(star_candidate.faces, dtype=np.int64).reshape((-1, 3))
        points = np.asarray(star_candidate.vertices, dtype=np.float64)[faces]
        double_areas = np.linalg.norm(
            np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]),
            axis=1,
        )
        collapsed_ids = np.flatnonzero(double_areas <= float(minimum_double_area))
        if not len(collapsed_ids):
            volume_delta = abs(float(abs(star_candidate.volume)) - source_volume)
            volume_tolerance = max(1e-6, source_volume * 1e-9)
            if (
                star_candidate.is_watertight
                and star_candidate.is_winding_consistent
                and volume_delta <= volume_tolerance
            ):
                return star_candidate, {
                    "applied": True,
                    "policy": (
                        "local_collapsed_vertex_star_retriangulation_"
                        "without_center_fan"
                    ),
                    "collapsed_faces_before": int(expected_degenerate_faces),
                    "star_retriangulation_steps": star_steps,
                    "removed_faces": int(source_faces - len(star_candidate.faces)),
                    "volume_delta_mm3": float(volume_delta),
                    "attempts": attempts,
                }
            break

        collapsed_face_id = int(collapsed_ids[0])
        collapsed_face = faces[collapsed_face_id]
        triangle = np.asarray(star_candidate.vertices, dtype=np.float64)[
            collapsed_face
        ]

        # Prefer the vertex lying between the other two points.  Removing that
        # vertex is the exact inverse of the seam subdivision which normally
        # creates a collinear Boolean triangle.
        ranked_vertices: list[tuple[float, int, int]] = []
        for local_index, vertex_id in enumerate(collapsed_face):
            other = [index for index in range(3) if index != local_index]
            segment = triangle[other[1]] - triangle[other[0]]
            denominator = float(np.dot(segment, segment))
            if denominator <= 1e-30:
                segment_distance = float(
                    np.linalg.norm(triangle[local_index] - triangle[other[0]])
                )
                outside_penalty = 0.0
            else:
                projection = float(
                    np.dot(triangle[local_index] - triangle[other[0]], segment)
                    / denominator
                )
                closest = triangle[other[0]] + np.clip(projection, 0.0, 1.0) * segment
                segment_distance = float(
                    np.linalg.norm(triangle[local_index] - closest)
                )
                outside_penalty = max(-projection, projection - 1.0, 0.0)
            incident_count = int(
                np.count_nonzero(np.any(faces == int(vertex_id), axis=1))
            )
            ranked_vertices.append(
                (float(outside_penalty * 1e6 + segment_distance), incident_count, int(vertex_id))
            )

        accepted_step = None
        for _score, incident_count, vertex_id in sorted(ranked_vertices):
            repaired_step, step_record = _retriangulate_boolean_vertex_star(
                star_candidate,
                vertex_id=int(vertex_id),
                minimum_double_area=float(minimum_double_area),
            )
            step_record.update(
                {
                    "collapsed_face_id": int(collapsed_face_id),
                    "candidate_vertex_incident_faces": int(incident_count),
                }
            )
            if repaired_step is None:
                star_steps.append(step_record)
                continue
            repaired_faces = np.asarray(repaired_step.faces, dtype=np.int64)
            repaired_points = np.asarray(repaired_step.vertices, dtype=np.float64)[
                repaired_faces
            ]
            repaired_areas = np.linalg.norm(
                np.cross(
                    repaired_points[:, 1] - repaired_points[:, 0],
                    repaired_points[:, 2] - repaired_points[:, 0],
                ),
                axis=1,
            )
            remaining_collapsed = int(
                np.count_nonzero(repaired_areas <= float(minimum_double_area))
            )
            step_record["remaining_collapsed_faces"] = remaining_collapsed
            if remaining_collapsed >= len(collapsed_ids):
                step_record["accepted"] = False
                step_record["reason"] = "collapsed_face_count_not_reduced"
                star_steps.append(step_record)
                continue
            step_volume_delta = abs(
                float(abs(repaired_step.volume)) - float(abs(star_candidate.volume))
            )
            step_record["step_volume_delta_mm3"] = float(step_volume_delta)
            step_volume_tolerance = max(1e-7, source_volume * 1e-10)
            if step_volume_delta > step_volume_tolerance:
                step_record["accepted"] = False
                step_record["reason"] = "local_volume_delta_exceeded"
                star_steps.append(step_record)
                continue
            step_record["accepted"] = True
            star_steps.append(step_record)
            accepted_step = repaired_step
            break
        if accepted_step is None:
            break
        star_candidate = accepted_step

    for digits in (9, 8, 7, 6):
        candidate = mesh.copy()
        candidate.merge_vertices(digits_vertex=int(digits))
        faces = np.asarray(candidate.faces, dtype=np.int64)
        distinct = np.asarray([len(set(map(int, face))) == 3 for face in faces])
        if np.any(distinct):
            kept_faces = faces[distinct]
            _keys, unique_indices = np.unique(
                np.sort(kept_faces, axis=1),
                axis=0,
                return_index=True,
            )
            kept_faces = kept_faces[np.sort(unique_indices)]
            points = np.asarray(candidate.vertices, dtype=np.float64)[kept_faces]
            areas = np.linalg.norm(
                np.cross(
                    points[:, 1] - points[:, 0],
                    points[:, 2] - points[:, 0],
                ),
                axis=1,
            )
            kept_faces = kept_faces[areas > float(minimum_double_area)]
        else:
            kept_faces = np.empty((0, 3), dtype=np.int64)
        candidate.faces = kept_faces
        candidate.remove_unreferenced_vertices()
        trimesh.repair.fix_winding(candidate)
        trimesh.repair.fix_normals(candidate)
        edge_counts = (
            np.bincount(candidate.edges_unique_inverse)
            if len(candidate.faces)
            else np.asarray([], dtype=np.int64)
        )
        remaining_triangles = np.asarray(candidate.vertices, dtype=np.float64)[
            np.asarray(candidate.faces, dtype=np.int64)
        ]
        remaining_areas = (
            np.linalg.norm(
                np.cross(
                    remaining_triangles[:, 1] - remaining_triangles[:, 0],
                    remaining_triangles[:, 2] - remaining_triangles[:, 0],
                ),
                axis=1,
            )
            if len(remaining_triangles)
            else np.asarray([], dtype=np.float64)
        )
        removed_faces = source_faces - int(len(candidate.faces))
        volume_delta = abs(float(abs(candidate.volume)) - source_volume)
        volume_tolerance = max(1e-6, source_volume * 1e-9)
        attempt = {
            "digits_vertex": int(digits),
            "removed_faces": int(removed_faces),
            "boundary_edges": int(np.count_nonzero(edge_counts == 1)),
            "over_shared_edges": int(np.count_nonzero(edge_counts > 2)),
            "remaining_collapsed_faces": int(
                np.count_nonzero(remaining_areas <= float(minimum_double_area))
            ),
            "volume_delta_mm3": float(volume_delta),
        }
        attempts.append(attempt)
        valid = (
            bool(candidate.is_watertight)
            and bool(candidate.is_winding_consistent)
            and attempt["over_shared_edges"] == 0
            and attempt["remaining_collapsed_faces"] == 0
            and 0 < removed_faces <= maximum_removed_faces
            and volume_delta <= volume_tolerance
        )
        if valid:
            return candidate, {
                "applied": True,
                "policy": "submicron_weld_without_hole_filling",
                "collapsed_faces_before": int(expected_degenerate_faces),
                "selected_digits_vertex": int(digits),
                "removed_faces": int(removed_faces),
                "volume_delta_mm3": float(volume_delta),
                "attempts": attempts,
            }
    raise ValueError(
        "socket boolean subtraction produced unrepairable collapsed faces: "
        f"degenerate_faces={int(expected_degenerate_faces)}, "
        f"star_steps={star_steps}, attempts={attempts}"
    )


def _retriangulate_boolean_vertex_star(
    mesh: trimesh.Trimesh,
    *,
    vertex_id: int,
    minimum_double_area: float,
) -> tuple[trimesh.Trimesh | None, dict[str, object]]:
    """Remove one seam vertex and retriangulate its existing one-ring cavity."""
    from .mesh import boundary_loops, triangulate_boundary_cap_without_center

    faces = np.asarray(mesh.faces, dtype=np.int64).reshape((-1, 3))
    star_mask = np.any(faces == int(vertex_id), axis=1)
    star_ids = np.flatnonzero(star_mask)
    record: dict[str, object] = {
        "vertex_id": int(vertex_id),
        "removed_star_faces": int(len(star_ids)),
        "accepted": False,
    }
    if len(star_ids) < 3:
        record["reason"] = "vertex_star_too_small"
        return None, record

    loops = boundary_loops(faces[star_ids])
    record["boundary_loop_count"] = int(len(loops))
    record["boundary_loop_sizes"] = [int(len(loop)) for loop in loops]
    if len(loops) != 1 or len(loops[0]) < 3 or int(vertex_id) in loops[0]:
        record["reason"] = "vertex_star_is_not_one_simple_disk"
        return None, record

    remaining_faces = faces[~star_mask]
    loop = [int(index) for index in loops[0]]
    loop_set = set(loop)
    occupied_edges: set[tuple[int, int]] = set()
    for face in remaining_faces:
        for left, right in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            if int(left) in loop_set and int(right) in loop_set:
                occupied_edges.add(tuple(sorted((int(left), int(right)))))

    star_points = np.asarray(mesh.vertices, dtype=np.float64)[faces[star_ids]]
    star_normals = np.cross(
        star_points[:, 1] - star_points[:, 0],
        star_points[:, 2] - star_points[:, 0],
    )
    normal_lengths = np.linalg.norm(star_normals, axis=1)
    valid_normals = star_normals[normal_lengths > float(minimum_double_area)]
    if len(valid_normals):
        fallback_normal = np.sum(valid_normals, axis=0)
    else:
        fallback_normal = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if float(np.linalg.norm(fallback_normal)) <= 1e-15:
        fallback_normal = valid_normals[0]

    vertices_list = [
        np.asarray(point, dtype=np.float64).copy()
        for point in np.asarray(mesh.vertices, dtype=np.float64)
    ]
    faces_list = [list(map(int, face)) for face in remaining_faces]
    added_faces, cap_record = triangulate_boundary_cap_without_center(
        vertices_list,
        faces_list,
        loop,
        np.asarray(fallback_normal, dtype=np.float64),
        occupied_edges=occupied_edges,
        require_planar_quality=False,
    )
    record["replacement_faces"] = int(added_faces)
    record["triangulation"] = cap_record
    if added_faces != len(loop) - 2:
        record["reason"] = "vertex_star_retriangulation_incomplete"
        return None, record

    candidate = trimesh.Trimesh(
        vertices=np.asarray(vertices_list, dtype=np.float64),
        faces=np.asarray(faces_list, dtype=np.int64),
        process=False,
    )
    candidate.remove_unreferenced_vertices()
    trimesh.repair.fix_winding(candidate)
    trimesh.repair.fix_normals(candidate)
    edge_counts = np.bincount(candidate.edges_unique_inverse)
    record.update(
        {
            "watertight": bool(candidate.is_watertight),
            "winding_consistent": bool(candidate.is_winding_consistent),
            "boundary_edges": int(np.count_nonzero(edge_counts == 1)),
            "over_shared_edges": int(np.count_nonzero(edge_counts > 2)),
        }
    )
    if (
        not candidate.is_watertight
        or not candidate.is_winding_consistent
        or int(record["boundary_edges"]) != 0
        or int(record["over_shared_edges"]) != 0
    ):
        record["reason"] = "vertex_star_retriangulation_not_closed_manifold"
        return None, record
    return candidate, record


def _rounded_rectangle_xy(
    width: float,
    length: float,
    radius: float,
    samples: int,
) -> np.ndarray:
    if samples < 16 or samples % 4:
        raise ValueError("samples must be a multiple of four and at least sixteen")
    half_x = float(width) / 2.0
    half_y = float(length) / 2.0
    radius = min(max(float(radius), 0.0), half_x, half_y)
    per_corner = samples // 4
    centers = (
        (half_x - radius, half_y - radius, 0.0),
        (-half_x + radius, half_y - radius, math.pi / 2.0),
        (-half_x + radius, -half_y + radius, math.pi),
        (half_x - radius, -half_y + radius, 3.0 * math.pi / 2.0),
    )
    points: list[tuple[float, float]] = []
    for center_x, center_y, start in centers:
        for offset in range(per_corner):
            angle = start + (math.pi / 2.0) * offset / per_corner
            points.append(
                (
                    center_x + radius * math.cos(angle),
                    center_y + radius * math.sin(angle),
                )
            )
    return np.asarray(points, dtype=np.float64)


class _MeshSew:
    def __init__(self) -> None:
        self.vertices: list[list[float]] = []
        self.faces: list[tuple[int, int, int]] = []

    def ring(self, xy: np.ndarray, z: float) -> np.ndarray:
        start = len(self.vertices)
        for x, y in np.asarray(xy, dtype=np.float64):
            self.vertices.append([float(x), float(y), float(z)])
        return np.arange(start, start + len(xy), dtype=np.int64)

    def point(self, xyz: tuple[float, float, float]) -> int:
        index = len(self.vertices)
        self.vertices.append([float(value) for value in xyz])
        return index

    def connect(self, first: np.ndarray, second: np.ndarray, *, reverse: bool = False) -> None:
        if len(first) != len(second):
            raise ValueError("connected rings must have equal sample counts")
        count = len(first)
        for index in range(count):
            nxt = (index + 1) % count
            a = int(first[index])
            aj = int(first[nxt])
            b = int(second[index])
            bj = int(second[nxt])
            triangles = ((a, aj, bj), (a, bj, b))
            if reverse:
                triangles = tuple((x, z, y) for x, y, z in triangles)
            self.faces.extend(triangles)

    def annulus(self, outer: np.ndarray, inner: np.ndarray, *, normal_up: bool) -> None:
        if len(outer) != len(inner):
            raise ValueError("annulus rings must have equal sample counts")
        count = len(outer)
        for index in range(count):
            nxt = (index + 1) % count
            oi = int(outer[index])
            oj = int(outer[nxt])
            ii = int(inner[index])
            ij = int(inner[nxt])
            triangles = ((oi, oj, ij), (oi, ij, ii))
            if not normal_up:
                triangles = tuple((x, z, y) for x, y, z in triangles)
            self.faces.extend(triangles)

    def cap(self, ring: np.ndarray, center: int, *, normal_up: bool) -> None:
        count = len(ring)
        for index in range(count):
            nxt = (index + 1) % count
            triangle = (int(ring[index]), int(ring[nxt]), int(center))
            if not normal_up:
                triangle = (triangle[0], triangle[2], triangle[1])
            self.faces.append(triangle)

    def mesh(self) -> trimesh.Trimesh:
        result = trimesh.Trimesh(
            vertices=np.asarray(self.vertices, dtype=np.float64),
            faces=np.asarray(self.faces, dtype=np.int64),
            process=False,
        )
        result.remove_unreferenced_vertices()
        trimesh.repair.fix_winding(result)
        trimesh.repair.fix_normals(result)
        return result


def _irregular_circle_xy(radius: float, samples: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    variation = 1.0 + 0.055 * np.sin(3.0 * angles + 0.45) + 0.022 * np.sin(7.0 * angles - 0.30)
    radial = float(radius) * variation
    return np.column_stack((radial * np.cos(angles), radial * np.sin(angles)))


def _build_child(spec: LocalConnectorSpec, samples: int) -> trimesh.Trimesh:
    sew = _MeshSew()
    base_radius = 7.0
    dome_height = 6.2
    ring_count = 7
    dome_rings: list[np.ndarray] = []
    for ring_offset in range(ring_count):
        theta = (math.pi / 2.0) * (ring_count - ring_offset) / ring_count
        z = dome_height * math.cos(theta)
        base_xy = _irregular_circle_xy(base_radius * math.sin(theta), samples)
        dome_rings.append(sew.ring(base_xy, z))
    top = sew.point((0.0, 0.0, dome_height * 1.015))
    for lower, upper in zip(dome_rings, dome_rings[1:]):
        sew.connect(lower, upper)
    sew.cap(dome_rings[-1], top, normal_up=True)

    peg_top_xy = _rounded_rectangle_xy(
        spec.peg_width_mm,
        spec.peg_length_mm,
        spec.corner_radius_mm,
        samples,
    )
    peg_top = sew.ring(peg_top_xy, 0.0)
    sew.annulus(dome_rings[0], peg_top, normal_up=False)
    if not spec.compact_peg_enabled:
        floor_center = sew.point((0.0, 0.0, 0.0))
        sew.cap(peg_top, floor_center, normal_up=False)
        return sew.mesh()

    straight_z = -(spec.engagement_depth_mm - spec.peg_tip_chamfer_mm)
    peg_straight = sew.ring(peg_top_xy, straight_z)
    sew.connect(peg_top, peg_straight, reverse=True)
    if spec.peg_tip_chamfer_mm > 1e-9:
        tip_width = spec.peg_width_mm - 2.0 * spec.peg_tip_chamfer_mm
        tip_length = spec.peg_length_mm - 2.0 * spec.peg_tip_chamfer_mm
        tip_radius = max(0.05, spec.corner_radius_mm - spec.peg_tip_chamfer_mm)
        peg_tip = sew.ring(
            _rounded_rectangle_xy(tip_width, tip_length, tip_radius, samples),
            -spec.engagement_depth_mm,
        )
        sew.connect(peg_straight, peg_tip, reverse=True)
    else:
        # A zero tip chamfer is a vertical peg with one flat terminal face.
        # Reusing the straight ring avoids a coincident zero-area side strip.
        peg_tip = peg_straight
    tip_center = sew.point((0.0, 0.0, -spec.engagement_depth_mm))
    sew.cap(peg_tip, tip_center, normal_up=False)
    return sew.mesh()


def _build_parent(spec: LocalConnectorSpec, samples: int) -> trimesh.Trimesh:
    sew = _MeshSew()
    parent_radius = 11.0
    parent_depth = max(7.0, spec.engagement_depth_mm + spec.socket_bottom_clearance_mm + 1.25)
    profile = connector_profile(spec, contact_width_mm=14.0, contact_length_mm=14.0)

    angles = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    outer_xy = np.column_stack((parent_radius * np.cos(angles), parent_radius * np.sin(angles)))
    outer_top = sew.ring(outer_xy, 0.0)
    outer_bottom = sew.ring(outer_xy, -parent_depth)

    if not spec.compact_peg_enabled:
        outer_top_center = sew.point((0.0, 0.0, 0.0))
        sew.cap(outer_top, outer_top_center, normal_up=True)
        sew.connect(outer_top, outer_bottom, reverse=True)
        outer_bottom_center = sew.point((0.0, 0.0, -parent_depth))
        sew.cap(outer_bottom, outer_bottom_center, normal_up=False)
        return sew.mesh()

    per_side_clearance = spec.total_clearance_mm / 2.0
    throat_radius = spec.corner_radius_mm + per_side_clearance
    mouth_radius = throat_radius + spec.socket_mouth_chamfer_mm
    mouth_xy = _rounded_rectangle_xy(
        profile["socket_mouth_width_mm"],
        profile["socket_mouth_length_mm"],
        mouth_radius,
        samples,
    )
    throat_xy = _rounded_rectangle_xy(
        profile["socket_throat_width_mm"],
        profile["socket_throat_length_mm"],
        throat_radius,
        samples,
    )
    mouth = sew.ring(mouth_xy, 0.0)
    throat = sew.ring(throat_xy, -spec.socket_mouth_chamfer_mm)
    socket_bottom = sew.ring(throat_xy, -profile["socket_depth_mm"])

    sew.annulus(outer_top, mouth, normal_up=True)
    sew.connect(mouth, throat)
    sew.connect(throat, socket_bottom)
    socket_center = sew.point((0.0, 0.0, -profile["socket_depth_mm"]))
    sew.cap(socket_bottom, socket_center, normal_up=True)

    sew.connect(outer_top, outer_bottom, reverse=True)
    outer_bottom_center = sew.point((0.0, 0.0, -parent_depth))
    sew.cap(outer_bottom, outer_bottom_center, normal_up=False)
    return sew.mesh()


def build_synthetic_local_connector_pair(
    spec: LocalConnectorSpec | None = None,
    *,
    samples: int = 64,
) -> dict[str, object]:
    """Build a rounded child and socketed parent for geometry-first review."""

    spec = spec or LocalConnectorSpec()
    child = _build_child(spec, samples)
    parent = _build_parent(spec, samples)
    return {
        "child": child,
        "parent": parent,
        "profile": connector_profile(spec, contact_width_mm=14.0, contact_length_mm=14.0),
        "spec": spec,
    }
