"""
Copyright (c) 2014 Marshall Farrier
license http://opensource.org/licenses/MIT

@author: Marshall Farrier
@contact: marshalldfarrier@gmail.com
@since: 2014-11-02
@summary: Linear regression in Python
"""

import sys
import unittest

import numpy as np

sys.path.append('../experiments')
from linreg import get_wts

class TestLinReg(unittest.TestCase):

    def test_sanity(self):
        features = np.array([
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0]])
        labels = np.array([
            [0.0, 4.0],
            [1.0, 6.0],
            [2.0, 8.0],
            [3.0, 10.0]])
        wts = get_wts(features, labels)
        self.assertEqual(wts.shape, (2, 2))
        self.assertAlmostEqual(wts[0, 0], 0.0)
        self.assertAlmostEqual(wts[1, 0], 1.0)
        self.assertAlmostEqual(wts[0, 1], 4.0)
        self.assertAlmostEqual(wts[1, 1], 2.0)

    def test_noisy_linear(self):
        epsilon = 0.2
        features, labels = self._linear_data(epsilon)
        wts = get_wts(features, labels)
        self.assertEqual(wts.shape, (4, 1))
        self.assertAlmostEqual(wts[0, 0], 0.7)
        self.assertAlmostEqual(wts[1, 0], 1.2)
        self.assertAlmostEqual(wts[2, 0], 0.3)
        self.assertAlmostEqual(wts[3, 0], -0.4)

    def _linear_data(self, epsilon=0.0):
        # 64 evenly distributed data points in [0, 1)^3
        features = np.ones((64, 4))
        grid = np.mgrid[[slice(0, 1, 0.25)] * 3]
        features[:, 1:] = grid.flatten().reshape((3, 64)).transpose()
        w = np.array([0.7, 1.2, 0.3, -0.4]).reshape((4, 1))
        labels = features.dot(w)
        for i in range(64):
            if round(np.sum(features[i, 1:]) * 4.0) % 2 == 0:
                labels[i, 0] += epsilon
            else:
                labels[i, 0] -= epsilon
        return features, labels

if __name__ == '__main__':
    unittest.main()
