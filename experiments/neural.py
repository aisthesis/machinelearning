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

import copy

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

def least_sq_ntwk1(features, labels, initial_wts, **kwargs):
    """
    Return weight matrices for a 3-layer (1 hidden layer) neural
    network that minimizes the sum of squares of the difference
    between predictions and labels.

    For the sake of simplicity, this implementation has only 1
    hidden layer and only 1 label for each row, i.e., labels is
    a n x 1 column vector..

    Parameters
    ---
    features : ndarray of floats
        data set including constant in the first column
    labels : ndarray of floats (1 column)
        In contrast to traditional neural networks, the labels
        are not restricted to 2 values for yes and no but can
        take any real value.
    initial_wts : ndarray of floats
        randomly initialized weight matrix which determines
        the number of hidden units. If `features` is n x m,
        `initial_wts` must have m rows. The number of columns
        in `initial_wts` will determine the number of hidden
        units in the neural network. For example, if `initial_wts`
        is m x k, then the hidden layer will be of size k + 1 
        (including its constant feature).
    maxiter : int, optional
        Default: 64. Stop after this number of iterations
    eta : float, optional
        Default: 0.1. Learning rate. A high value of eta risks
        overshooting the minimum. A low value will converge very
        slowly.
    epsilon : float, optional (not implemented)
        Default: None. Stop when the improvement in error
        from one iteration to the next falls below this threshold

    Return
    ---
    A tuple of 2 weight matrices `wts0` and `wts1`. This model
    will predict in the following way:
    hidden = np.ones((k + 1, 1))
    """
    maxiter = kwargs.get('maxiter', 64)
    eta = kwargs.get('eta', 0.1)
    wts0 = copy.deepcopy(initial_wts)
    # get initial hidden layer
    hidden = np.empty((features.shape[0], wts0.shape[1] + 1))
    hidden[:, 0] = 1.0
    hidden[:, 1:] = sigmoid(features.dot(wts0))
    # get optimal wts1 given wts0
    wts1 = np.linalg.pinv(hidden).dot(labels)
    # backprop
    """
    As of 2014-11-11 something isn't right here. Algorithm isn't converging.
    Sample output on nonlinear function using random features:
    error after iteration 1: 0.389115314723
    error after iteration 2: 0.324511537066
    error after iteration 3: 0.728566874908
    error after iteration 4: 0.174713101869
    error after iteration 5: 0.47208655752
    error after iteration 6: 0.779610384956
    ...
    error after iteration 60: 1.26413065775
    error after iteration 61: 1.26394322385
    error after iteration 62: 1.26373950808
    error after iteration 63: 1.26351975815
    error after iteration 64: 1.26328420549
    """
    for i in range(maxiter):
        wts0 -= eta * sigmoid_grad(wts0) * wts1[1:]
        hidden[:, 1:] = sigmoid(features.dot(wts0))
        wts1 = np.linalg.pinv(hidden).dot(labels)
        predicted = least_sq_predict(features, wts0, wts1, hidden=hidden)
        print "error after iteration {0}: {1}".format(i + 1, ave_sq_error(predicted, labels))
    return wts0, wts1

def least_sq_predict(features, wts0, wts1, **kwargs):
    """
    Parameters
    ---
    hidden : ndarray of floats
        float array of the proper size for the hidden layer. Using
        this parameter will lead to improved performance because
        creating the hidden layer won't require reallocation of memory.
        Note that the values of the matrix passed in for this
        parameter will be overwritten when this function is called.
    """
    hidden = kwargs.get('hidden', np.ones((features.shape[0], wts0.shape[1] + 1)))
    hidden[:, 1:] = sigmoid(features.dot(wts0))
    return hidden.dot(wts1)

def ave_sq_error(predicted, actual):
    diff = predicted - actual
    return np.mean(diff * diff, axis=0)
