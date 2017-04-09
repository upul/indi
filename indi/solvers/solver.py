import numpy as np

from indi.solvers.regularizationtype import RegularizationType
from indi.exceptions.modelbuilding import HyperParameterException


def sgd(cost_func, X, y,
        learning_rate=0.01,
        max_iter=100,
        regularization=1e-2,
        regularization_type=None,
        tolerance=1e-4,
        verbose=False):
    weights = np.zeros(X.shape[1])
    #cost = 0.0
    #grad_cost = 1e100
    iteration = 1
    while True:
        if regularization_type is None:
            cost, grad_cost = cost_func(weights, X, y)
        elif regularization_type == RegularizationType.L2:
            cost, grad_cost = cost_func(weights, X, y, regularization)
        else:
            raise HyperParameterException('regularization_type: {} '
                                          'is not applicable for Linear Regression'.format
                                          (regularization_type))
        weights -= (learning_rate * grad_cost)
        if verbose:
            if (iteration <= 10) or \
                    (iteration <= 100 and iteration % 10 == 0) or \
                    (iteration <= 1000 and iteration % 100 == 0) or \
                    (iteration <= 10000 and iteration % 1000 == 0) or \
                    (iteration <= 100000 and iteration % 10000 == 0):
                print('iteration: {:10d}\t\tcost: {:1.4e}'.format(iteration, cost))

        if iteration >= max_iter:
            break
        if np.linalg.norm(grad_cost) < tolerance:
            break
        iteration += 1

    if verbose:
        print('\nOptimization completed: number of iterations: {}, Norm of gradient of the cost: {}\n'.
              format(iteration, np.linalg.norm(grad_cost)))
    return weights, cost


def lasso_coordinate_descent(X, y, regularization=1e3, tolerance=1e-3):
    weights = np.zeros(X.shape[1])
    previous_weights = np.copy(weights)
    while True:
        for i in range(weights.shape[0]):
            weight_i = _lasso_coordinate_descent_step(i, X, y, weights, regularization)
            weights[i] = weight_i
        delta = np.sqrt((weights - previous_weights) ** 2)
        previous_weights = np.copy(weights)
        if (delta < tolerance).all():
            break
    return weights


def _lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty):
    weight_without_i = weights[np.arange(weights.shape[0]) != i]
    feature_matrix_weightout_i = feature_matrix[:, np.arange(feature_matrix.shape[1]) != i]
    prediction = _predict_output(feature_matrix_weightout_i, weight_without_i)
    ro_i = np.dot(feature_matrix[:, i], (output - prediction))

    if ro_i < -l1_penalty / 2.0:
        new_weight_i = ro_i + l1_penalty / 2.0
    elif ro_i > l1_penalty / 2.0:
        new_weight_i = ro_i - l1_penalty / 2.0
    else:
        new_weight_i = 0
    return new_weight_i


def _predict_output(feature_matrix, weights):
    predictions = np.dot(feature_matrix, weights)
    return predictions
