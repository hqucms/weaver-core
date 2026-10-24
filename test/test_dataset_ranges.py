import math
import unittest

import numpy as np

from weaver.utils.dataset import _draw_load_ranges, _split_load_range


class SplitLoadRangeTest(unittest.TestCase):
    def test_campaign_restart_offsets_do_not_create_empty_chunks(self):
        cases = [
            ((0.7345173024982788, 0.7442323024982789), 0.009715),
            ((0.09771905011393996, 0.10568705011393996), 0.007968),
            ((0.14914521904107697, 0.15158921904107697), 0.002444),
        ]

        for load_range, fetch_step in cases:
            with self.subTest(load_range=load_range, fetch_step=fetch_step):
                self.assertEqual(_split_load_range(load_range, fetch_step), [load_range])

    def test_preserves_real_final_partial_chunk(self):
        load_range = (0.44973246208507933, 0.4622994620850793)
        ranges = _split_load_range(load_range, 0.01)

        self.assertEqual(len(ranges), 2)
        self.assertEqual(ranges[0][0], load_range[0])
        self.assertEqual(ranges[-1][1], load_range[1])
        self.assertTrue(all(start < end for start, end in ranges))
        self.assertTrue(math.isclose(ranges[0][1] - ranges[0][0], 0.01))

    def test_non_positive_step_loads_the_full_range(self):
        load_range = (0.2, 0.7)
        self.assertEqual(_split_load_range(load_range, 0), [load_range])
        self.assertEqual(_split_load_range(load_range, -1), [load_range])


class DrawLoadRangesTest(unittest.TestCase):
    def test_without_shuffle_takes_the_start_of_the_range(self):
        self.assertEqual(_draw_load_ranges((0.25, 0.75), 0.5, shuffle=False), [(0.25, 0.5)])

    def test_full_fraction_is_one_range(self):
        self.assertEqual(_draw_load_ranges((0, 1), 1, shuffle=True), [(0, 1)])

    def test_slices_stay_inside_the_range_and_keep_their_length(self):
        rng = np.random.default_rng(1)
        for _ in range(2000):
            ranges = _draw_load_ranges((0.2, 0.8), 0.1, shuffle=True, rng=rng)
            self.assertIn(len(ranges), (1, 2))
            self.assertTrue(all(0.2 <= a < b <= 0.8 for a, b in ranges))
            self.assertTrue(math.isclose(sum(b - a for a, b in ranges), 0.06, abs_tol=1e-12))
            if len(ranges) == 2:
                self.assertEqual((ranges[0][1], ranges[1][0]), (0.8, 0.2))

    def test_every_event_is_equally_likely(self):
        # the old start, uniform in [0, 1 - interval], read the first and last events ~10x less often
        rng = np.random.default_rng(2)
        n, hits = 1000, np.zeros(1000)
        for _ in range(20000):
            for a, b in _draw_load_ranges((0, 1), 0.1, shuffle=True, rng=rng):
                hits[math.trunc(a * n):math.trunc(b * n)] += 1
        coverage = hits / 20000
        self.assertTrue(np.all(np.abs(coverage - 0.1) < 0.01), (coverage.min(), coverage.max()))


if __name__ == "__main__":
    unittest.main()
