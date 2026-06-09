import math
import unittest

from weaver.utils.dataset import _split_load_range


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


if __name__ == "__main__":
    unittest.main()
