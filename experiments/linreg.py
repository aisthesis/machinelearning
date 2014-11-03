"""
Copyright (c) 2014 Marshall Farrier
license http://opensource.org/licenses/MIT

@author: Marshall Farrier
@contact: marshalldfarrier@gmail.com
@since: 2014-11-02
@summary: Linear regression in Python
"""

import numpy as np

def get_wts(features, labels):
    """
    @type features: numpy array
    @param features: all features including constant feature 1
    @type labels: numpy array
    @param labels: real-valued labels for each row of features,
    each column is a set of labels
    @return: numpy array w providing least squared errors when labels
    are taken as features.dot(w). If labels are a single column, w will
    be a column vector. In general, if there are m features and n columns
    of labels, w will be a m x n matrix
    """
    return np.linalg.pinv(features).dot(labels)
