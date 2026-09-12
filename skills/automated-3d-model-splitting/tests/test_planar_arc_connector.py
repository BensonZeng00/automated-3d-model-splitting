from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from split3mf import common

common.load_core_dependencies()

from split3mf.connector_planning import local_connector_spec_for_interface


class PlanarArcConnectorTests(unittest.TestCase):
    def test_sub_printable_engagement_becomes_exactly_zero(self) -> None:
        spec = local_connector_spec_for_interface(
            fit_clearance_mm=0.50,
            bottom_clearance_mm=0.25,
            lead_in_mm=0.60,
            safe_engagement_depth_mm=3.25,
            safe_backing_depth_mm=3.25,
        )
        self.assertEqual(spec.engagement_depth_mm, 0.0)
        self.assertEqual(spec.socket_bottom_clearance_mm, 0.0)
        self.assertFalse(spec.compact_peg_enabled)
        self.assertEqual(spec.full_boundary_backing_depth_mm, 3.0)

    def test_printable_engagement_is_retained(self) -> None:
        spec = local_connector_spec_for_interface(
            fit_clearance_mm=0.50,
            bottom_clearance_mm=0.25,
            lead_in_mm=0.60,
            safe_engagement_depth_mm=4.0,
            safe_backing_depth_mm=3.0,
        )
        self.assertAlmostEqual(spec.engagement_depth_mm, 0.75)
        self.assertTrue(spec.compact_peg_enabled)



if __name__ == "__main__":
    unittest.main()
