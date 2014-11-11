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
        # equivalent to octave> linspace(-10, 9, 20).reshape(4, 5)'
        self.x = np.linspace(-10, 9, 20).reshape((4, 5))

    def test_sigmoid(self):
        sigx = neural.sigmoid(self.x)

    def test_sigmoid_grad(self):
        siggradx = neural.sigmoid_grad(self.x)


if __name__ == '__main__':
    unittest.main()
