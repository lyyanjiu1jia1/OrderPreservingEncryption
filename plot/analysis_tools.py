import numpy as np


def linear_regression(x, y):
    """

    :param x: n-by-m matrix, will be expanded to (m + 1)-columns
    :param y: n-by-1 matrix
    :return:
    """
    x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
    w = np.linalg.inv(x.transpose().dot(x)).dot(x.transpose()).dot(y)
    return w

