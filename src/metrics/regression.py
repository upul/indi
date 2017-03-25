import numpy as np


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2, axis=0)


def root_mean_squated_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
