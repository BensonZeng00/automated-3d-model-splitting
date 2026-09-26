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

from .direction_field import safe_boundary_inward_directions

class InwardDirectionPlanner:
    """Build a smooth, locally inward-safe direction field for one boundary."""

    plan = staticmethod(safe_boundary_inward_directions)
