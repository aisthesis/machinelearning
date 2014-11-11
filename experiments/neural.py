"""
Copyright (c) 2014 Marshall Farrier
license http://opensource.org/licenses/MIT

@author: Marshall Farrier
@contact: marshalldfarrier@gmail.com
@since: 2014-11-10
@summary: Neural network
Resources:
Great performance ideas:
http://stackoverflow.com/questions/21106134/numpy-pure-functions-for-performance-caching
"""

import numpy as np
from scipy.special import expit

def sigmoid(x):
    return expit(x)

def sigmoid_grad(x):
    """
    Slightly faster than:
    def sigmoid_grad2(x):
        ex = np.exp(-x)
        return ex / (1 + ex)**2
    >>> s = '''
    ... import numpy as np
    ... from __main__ import neural
    ... x = np.random.random((30, 40)) * 10.0 - 5.0'''
    >>> import timeit
    >>> timeit.timeit('y = neural.sigmoid_grad(x)', setup=s, number=10000)
    0.21567797660827637
    >>> timeit.timeit('y = neural.sigmoid_grad2(x)', setup=s, number=10000)
    0.23735404014587402
    """
    sigx = sigmoid(x)
    return sigx * (1 - sigx)
