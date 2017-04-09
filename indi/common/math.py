import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-1.0*x))