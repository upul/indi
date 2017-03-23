import numpy as np


def sgd(cost_func, X, y, learning_rate=0.01, max_iter=100, verbose=False):
    weights = np.zeros(X.shape[1])
    for i in range(max_iter):
        cost, grad_cost = cost_func(weights, X, y)
        print(cost)
        weights -= learning_rate * grad_cost
    return weights, cost
