"""Regression check for _knn/_batch_knn with scipy >= 1.9 (n_jobs -> workers)."""

import unittest

import numpy as np

from weaver.utils.data.tools import _batch_knn, _knn


class KnnTest(unittest.TestCase):
    def test_knn(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]])
        np.testing.assert_array_equal(_knn(pts, pts, 2), [[0, 1], [1, 0], [2, 1]])

    def test_batch_knn_shape(self):
        s = np.random.rand(3, 6, 2)
        self.assertEqual(_batch_knn(s, s, 2, 4).shape, (3, 4, 2))


if __name__ == "__main__":
    unittest.main()
