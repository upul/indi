import numpy as np


def sgd(cost_func, X, y, learning_rate=0.01, max_iter=100, verbose=False):
    weights = np.ones(X.shape[1])
    for i in range(max_iter):
        cost, grad_cost = cost_func(weights, X, y)
        weights -= (learning_rate * grad_cost)
        print('cost: {}'.format(cost))
    return weights, cost
