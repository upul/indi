import numpy as np


def mean_squared(weight, X_train, y_train):
    diff = np.dot(X_train, weight) - y_train
    cost = np.dot(np.transpose(diff), diff) / 2.0
    grad_cost = np.dot(np.transpose(X_train), diff)
    return cost, grad_cost


def mean_squared_l2_loss(weight, X_train, y_train):
    pass


def mean_squared_l1_loss(weight, X_train, y_train):
    pass