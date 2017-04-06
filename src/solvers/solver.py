import numpy as np


def sgd(cost_func, X, y, learning_rate=0.01, max_iter=100, verbose=False):
    weights = np.ones(X.shape[1])
    for i in range(max_iter):
        cost, grad_cost = cost_func(weights, X, y)
        weights -= (learning_rate * grad_cost)
        print('cost: {}'.format(cost))
    return weights, cost


def lasso_coordinate_descent(X, y, regularization = 0.0000001, tolerance=1):
    X, _ = _normalize_features(X)
    weights = np.zeros(X.shape[1])
    previous_weights = np.copy(weights)
    while True:
        for i in range(weights.shape[0]):
            weight_i = _lasso_coordinate_descent_step(i, X, y, weights, 1e3)
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

def _normalize_features(features):
    norms = np.linalg.norm(features, axis=0)
    normalized_features = features /  norms
    return (normalized_features, norms)

if __name__ == '__main__':
    import math

    print(_lasso_coordinate_descent_step(1, np.array([[3. / math.sqrt(13), 1. / math.sqrt(10)],
                                               [2. / math.sqrt(13), 3. / math.sqrt(10)]]), np.array([1., 1.]),
                                  np.array([1., 4.]), 0.1))
