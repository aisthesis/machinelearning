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
import neural

class TestNeural(unittest.TestCase):

    def setUp(self):
        # octave:1> x = reshape(linspace(-10, 9, 20), 5, 4)'
        self.x = np.linspace(-10, 9, 20).reshape((4, 5))

    def test_sigmoid(self):
        """
        octave:3> sigmoid(x)
        ans =

            4.5398e-05   1.2339e-04   3.3535e-04   9.1105e-04   2.4726e-03
            6.6929e-03   1.7986e-02   4.7426e-02   1.1920e-01   2.6894e-01
            5.0000e-01   7.3106e-01   8.8080e-01   9.5257e-01   9.8201e-01
            9.9331e-01   9.9753e-01   9.9909e-01   9.9966e-01   9.9988e-01
        """
        sigx = neural.sigmoid(self.x)
        expected = np.array([
            [4.5398e-05, 1.2339e-04, 3.3535e-04, 9.1105e-04, 2.4726e-03],
            [6.6929e-03, 1.7986e-02, 4.7426e-02, 1.1920e-01, 2.6894e-01],
            [5.0000e-01, 7.3106e-01, 8.8080e-01, 9.5257e-01, 9.8201e-01],
            [9.9331e-01, 9.9753e-01, 9.9909e-01, 9.9966e-01, 9.9988e-01]])
        self.assertEqual(sigx.shape, (4, 5))
        for i in range(4):
            for j in range(5):
                if expected[i, j] < 0.01:
                    self.assertAlmostEqual(sigx[i, j], expected[i, j])
                elif expected[i, j] < 0.1:
                    self.assertAlmostEqual(sigx[i, j], expected[i, j], places=6)
                else:
                    self.assertAlmostEqual(sigx[i, j], expected[i, j], places=5)

    def test_sigmoid_grad(self):
        """
        octave:4> sigmoidGradient(x)
        ans =

            4.5396e-05   1.2338e-04   3.3524e-04   9.1022e-04   2.4665e-03
            6.6481e-03   1.7663e-02   4.5177e-02   1.0499e-01   1.9661e-01
            2.5000e-01   1.9661e-01   1.0499e-01   4.5177e-02   1.7663e-02
            6.6481e-03   2.4665e-03   9.1022e-04   3.3524e-04   1.2338e-04
        """
        expected = np.array([
            [4.5396e-05, 1.2338e-04, 3.3524e-04, 9.1022e-04, 2.4665e-03],
            [6.6481e-03, 1.7663e-02, 4.5177e-02, 1.0499e-01, 1.9661e-01],
            [2.5000e-01, 1.9661e-01, 1.0499e-01, 4.5177e-02, 1.7663e-02],
            [6.6481e-03, 2.4665e-03, 9.1022e-04, 3.3524e-04, 1.2338e-04]])
        siggradx = neural.sigmoid_grad(self.x)
        self.assertEqual(siggradx.shape, (4, 5))
        for i in range(4):
            for j in range(5):
                if expected[i, j] < 0.01:
                    self.assertAlmostEqual(siggradx[i, j], expected[i, j])
                elif expected[i, j] < 0.1:
                    self.assertAlmostEqual(siggradx[i, j], expected[i, j], places=6)
                else:
                    self.assertAlmostEqual(siggradx[i, j], expected[i, j], places=5)


if __name__ == '__main__':
    unittest.main()
