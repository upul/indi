import numpy as np


def entropy(x):
    x = x[x > 0]
    return -np.sum(x * np.log(x))


def information_gain(self, y, y_left, y_right):
    n = y.shape[0]
    n_l = y_left.shape[0]
    if n_l == 0:
        H_l = 0.0
    else:
        H_l = entropy(y_left.mean(axis=0))

    n_r = y_right.shape[0]
    if n_r == 0:
        H_r = 0
    else:
        H_r = entropy(y_right.mean(axis=0))

        H = self._entropy(y.mean(axis=0))
        return H - n_l / n * H_l - n_r / n * H_r
