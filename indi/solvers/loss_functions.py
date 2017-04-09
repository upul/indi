import numpy as np
from indi.common.math import sigmoid


def mean_squared(weight, X_train, y_train):
    num_examples = X_train.shape[0]
    diff = y_train - np.dot(X_train, weight)
    cost = np.dot(np.transpose(diff), diff) / (2.0 * num_examples)
    grad_cost = -np.dot(np.transpose(X_train), diff) / num_examples
    return cost, grad_cost


def mean_squared_l2_loss(weight, X_train, y_train, regularization=1e-3):
    num_examples = X_train.shape[0]
    diff = y_train - np.dot(X_train, weight)
    cost = (np.dot(np.transpose(diff), diff) + regularization * np.dot(weight, weight)) / (2.0 * num_examples)
    grad_cost = (-np.dot(np.transpose(X_train), diff) + regularization * weight[1:]) / num_examples
    return cost, grad_cost


def logistic_loss(weights, features, response, regularization=1e-3):
    num_of_samples = features.shape[0]
    cost = logistic_cost(weights, features, response) / num_of_samples
    diff = sigmoid(np.dot(features, weights)) - response
    grad_cost = np.dot(np.transpose(features), diff) / num_of_samples
    return cost, grad_cost


def logistic_cost(weights, features, response):
    cost = 0.0
    num_of_samples = features.shape[0]
    for i in range(num_of_samples):
        pred = sigmoid(np.dot(weights, features[i, :]))
        cost += -1*(response[i]*np.log(pred) + (1 - response[i])*np.log(1 - pred))
    return cost
