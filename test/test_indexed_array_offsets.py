"""Regression tests for `_get_content_and_offsets` and the padding helpers when
the input awkward array is wrapped by an IndexedArray (which is what
`table[mask]` produces, e.g. after `_apply_selection`).

Bug: the original implementation naively unwrapped IndexedArray to reach the
inner ListOffsetArray, returning the *underlying* offsets and silently dropping
the index. That caused row-count mismatches between selected loaded variables
(IndexedArray-wrapped) and computed-after-selection variables (plain
ListOffsetArray), which then broke `_fused_pad_and_stack` and the fallback
padding path.

Run: ``python -m pytest test/test_indexed_array_offsets.py`` or
``python -m unittest test.test_indexed_array_offsets``.
"""

import unittest

import awkward as ak
import numpy as np

from weaver.utils.data.tools import (
    _fused_pad_and_stack,
    _get_content_and_offsets,
    _pad,
    _repeat_pad,
)


def _make_jagged(rows, rng):
    """Build a 1-level jagged ak.Array (ListOffsetArray<NumpyArray<float32>>)."""
    flat = np.concatenate([rng.random(n).astype(np.float32) for n in rows])
    counts = np.asarray(rows, dtype=np.int64)
    return ak.unflatten(ak.Array(flat), counts)


def _select_from_record(arr, mask):
    """Return ``arr[mask]`` with the same IndexedArray wrapping that
    ``table[selected]['field']`` produces in the real pipeline."""
    table = ak.Array({"x": arr})
    return table[mask]["x"]


class TestGetContentAndOffsets(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        rows = [3, 5, 0, 7, 2, 4, 6, 1]
        self.arr = _make_jagged(rows, self.rng)
        self.rows = rows
        self.mask = np.array([True, False, True, True, False, True, False, True])
        self.expected_rows = [rows[i] for i, m in enumerate(self.mask) if m]

    def test_plain_list_offset_returns_true_offsets(self):
        content, offsets = _get_content_and_offsets(self.arr)
        self.assertEqual(len(offsets) - 1, len(self.rows))
        self.assertEqual(int(offsets[-1]), sum(self.rows))
        self.assertEqual(content.shape[0], sum(self.rows))

    def test_indexed_array_returns_filtered_offsets(self):
        # Selecting through a record array produces IndexedArray-wrapped fields
        # (same shape as `table[selected]['part_deta']` in the data pipeline).
        selected = _select_from_record(self.arr, self.mask)
        self.assertIsInstance(
            selected.layout,
            (ak.contents.IndexedArray, ak.contents.IndexedOptionArray),
            "record-array selection should produce an IndexedArray wrapper",
        )

        content, offsets = _get_content_and_offsets(selected)
        # Must reflect the filtered view, NOT the underlying array.
        self.assertEqual(len(offsets) - 1, len(self.expected_rows))
        self.assertEqual(int(offsets[-1]), sum(self.expected_rows))
        self.assertEqual(content.shape[0], sum(self.expected_rows))
        np.testing.assert_array_equal(np.diff(offsets), np.asarray(self.expected_rows))

    def test_pad_after_selection(self):
        selected = _select_from_record(self.arr, self.mask)
        out = _pad(selected, maxlen=8, value=-1.0)
        self.assertEqual(out.shape, (len(self.expected_rows), 8))
        for i, n in enumerate(self.expected_rows):
            if n < 8:
                np.testing.assert_array_equal(out[i, n:], -1.0)

    def test_repeat_pad_after_selection(self):
        selected = _select_from_record(self.arr, self.mask)
        out = _repeat_pad(selected, maxlen=8)
        self.assertEqual(out.shape, (len(self.expected_rows), 8))
        # wrap-mode: a non-empty row of length n must wrap with out[i, j] == row[j % n]
        for i, n in enumerate(self.expected_rows):
            if n == 0:
                continue
            row_src = np.asarray(selected[i])
            for j in range(8):
                self.assertAlmostEqual(float(out[i, j]), float(row_src[j % n]), places=5)


class TestFusedPadAndStackAfterSelection(unittest.TestCase):
    """Mirror the failure mode in the data pipeline: a group whose vars mix
    loaded-then-selected (IndexedArray) and computed-after-selection
    (ListOffsetArray) variables must still produce a regular padded stack."""

    def setUp(self):
        self.rng = np.random.default_rng(1)
        rows = [4, 0, 6, 3, 9, 2, 5]
        self.mask = np.array([True, True, False, True, True, False, True])
        self.expected_rows = [rows[i] for i, m in enumerate(self.mask) if m]

        # var_a/var_b emulate "loaded then selected" -> IndexedArray-wrapped
        # var_c emulates "computed after selection" -> plain ListOffsetArray
        full_table = ak.Array(
            {"a": _make_jagged(rows, self.rng), "b": _make_jagged(rows, self.rng)}
        )
        selected = full_table[self.mask]
        self.var_a = selected["a"]
        self.var_b = selected["b"]
        # var_c is built *after* selection -> plain ListOffsetArray
        self.var_c = _make_jagged(self.expected_rows, self.rng)
        # sanity-check the fixture really exhibits the bug-prone mix
        self.assertIsInstance(
            self.var_a.layout,
            (ak.contents.IndexedArray, ak.contents.IndexedOptionArray),
        )

        self.table = {"a": self.var_a, "b": self.var_b, "c": self.var_c}
        self.padlen = 10
        self.preprocess_params = {
            k: {
                "length": self.padlen,
                "pad_mode": "constant",
                "center": None,
                "scale": 1.0,
                "min": -5.0,
                "max": 5.0,
                "pad_value": 0.0,
            }
            for k in self.table
        }

    def test_fused_stack_handles_mixed_wrappers(self):
        result = _fused_pad_and_stack(
            self.table, ["a", "b", "c"], self.preprocess_params
        )
        self.assertIsNotNone(
            result,
            "fused path must succeed when offsets match after materializing IndexedArrays",
        )
        self.assertEqual(result.shape, (len(self.expected_rows), 3, self.padlen))
        # constant-pad: positions past row length must be the pad_value (0.0 here)
        for i, n in enumerate(self.expected_rows):
            if n < self.padlen:
                np.testing.assert_array_equal(result[i, :, n:], 0.0)


if __name__ == "__main__":
    unittest.main()
