import unittest
import numpy as np
from scripts.map_preview_routing import navigation_region


class RoutingTest(unittest.TestCase):
    def test_annotation_gaps_do_not_disconnect_routes(self):
        occupancy = np.ones((20, 30))
        occupancy[2:18, 2:28] = 0
        valid = np.zeros_like(occupancy)
        valid[5:10, 5:10] = 1
        valid[5:10, 20:25] = 1
        region = navigation_region(occupancy, valid, radius_pixels=1)
        self.assertTrue(region[7, 7] and region[7, 22])
        self.assertTrue(region[7, 15])  # unannotated connecting free space
        self.assertFalse(region[2, 2])  # footprint intersects the wall

    def test_exterior_is_excluded(self):
        occupancy = np.zeros((20, 30))
        occupancy[5:15, 5:25] = 1
        occupancy[7:13, 7:23] = 0
        valid = np.ones_like(occupancy)
        region = navigation_region(occupancy, valid, radius_pixels=0)
        self.assertTrue(region[10, 10])
        self.assertFalse(region[0, 0])

    def test_footprint_can_close_a_narrow_passage(self):
        occupancy = np.ones((20, 30))
        occupancy[3:17, 3:12] = 0
        occupancy[3:17, 18:27] = 0
        occupancy[10, 12:18] = 0
        valid = (occupancy == 0).astype(float)
        self.assertTrue(navigation_region(occupancy, valid, 0)[10, 20])
        self.assertFalse(navigation_region(occupancy, valid, 1)[10, 20])

    def test_explicit_domain_bounds_a_cropped_open_map(self):
        occupancy = np.zeros((20, 30))
        valid = np.zeros_like(occupancy)
        valid[7:12, 10:20] = 1
        with self.assertRaises(ValueError):
            navigation_region(occupancy, valid, 1)
        domain = np.zeros_like(occupancy, dtype=bool)
        domain[3:17, 3:27] = True
        region = navigation_region(occupancy, valid, 1, domain)
        self.assertTrue(region[10, 15])
        self.assertFalse(region[0, 0])
        self.assertFalse(region[3, 15])  # footprint respects virtual boundary


if __name__ == '__main__':
    unittest.main()
