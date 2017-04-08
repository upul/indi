import numpy as np


def mean_squared(weight, X_train, y_train):
    num_examples = X_train.shape[0]
    diff = y_train - np.dot(X_train, weight)
    cost = np.dot(np.transpose(diff), diff) / (2.0 * num_examples)
    grad_cost = -np.dot(np.transpose(X_train), diff) / num_examples
    return cost, grad_cost


def mean_squared_l2_loss(weight, X_train, y_train, regularization=0.01):
    num_examples = X_train.shape[0]
    diff = y_train - np.dot(X_train, weight)
    cost = (np.dot(np.transpose(diff), diff) + regularization * np.dot(weight, weight)) / (2.0 * num_examples)
    grad_cost = (-np.dot(np.transpose(X_train), diff) + regularization * weight) / num_examples
    return cost, grad_cost
