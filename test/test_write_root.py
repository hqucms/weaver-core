"""Regression checks that _write_root writes every entry."""

import os
import tempfile
import unittest

import awkward as ak
import numpy as np
import uproot

from weaver.utils.data.fileio import _write_root


class WriteRootTest(unittest.TestCase):
    def _roundtrip(self, n, step):
        table = ak.Array({"x": np.arange(n, dtype="float32"), "y": np.arange(n, dtype="int64")})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.root")
            _write_root(path, table, step=step)
            with uproot.open(path) as f:
                return f["Events"].arrays(library="np")

    def test_all_entries_written(self):
        for n, step in ((1, 4), (3, 2), (4, 4), (5, 4), (9, 4)):
            with self.subTest(n=n, step=step):
                out = self._roundtrip(n, step)
                np.testing.assert_array_equal(out["x"], np.arange(n, dtype="float32"))
                np.testing.assert_array_equal(out["y"], np.arange(n))


if __name__ == "__main__":
    unittest.main()
