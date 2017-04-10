import numpy as np
from indi.exceptions.modelbuilding import HyperParameterException
from indi.solvers import solver
from indi.solvers.loss_functions import mean_squared, mean_squared_l2_loss
from indi.solvers.regularizationtype import RegularizationType


class LinearRegression:
    def __init__(self,
                 learning_rate=1e-3,
                 regularization_type=None,
                 regularization=1e-3,
                 max_iter=1e2,
                 tolerance=1e-5,
                 fit_intercept=True,
                 normalize=True,
                 verbose=False):
        self.learning_rate = learning_rate
        self.regularization_type = regularization_type
        self.regularization = regularization
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.verbose = verbose

        self.weight = None
        self.norm_vector = None

    def fit(self, X_train, y_train):
        """

        :param X_train:
        :param y_train:
        :return:
        """
        if self.regularization_type == RegularizationType.L1 and (not self.normalize):
            raise HyperParameterException('Set normalize is true for L1 regularization')

        if self.fit_intercept:
            X_train = self._fit_intercept(X_train)

        if self.normalize:
            X_train, self.norm_vector = self._normalize_features(X_train)

        if self.regularization_type is None or self.regularization_type == RegularizationType.L2:
            if self.regularization_type is None:
                loss_function = mean_squared
            else:
                loss_function = mean_squared_l2_loss
            self.weight, cost = solver.sgd(loss_function, X_train, y_train,
                                           self.learning_rate, self.max_iter,
                                           self.regularization,
                                           self.regularization_type,
                                           self.tolerance,
                                           self.verbose)
        elif self.regularization_type == RegularizationType.L1:
            self.weight = solver.lasso_coordinate_descent(X_train,
                                                          y_train,
                                                          self.regularization,
                                                          self.tolerance)
        else:
            raise HyperParameterException('regularization_type: {} '
                                          'is not applicable for Linear Regression'.format
                                          (self.regularization_type))

    def predict(self, X_test):
        """

        :param X_test:
        :return:
        """
        if self.fit_intercept:
            X_test = self._fit_intercept(X_test)

        if self.normalize and self.norm_vector is not None:
            X_test = X_test / self.norm_vector

        return np.dot(X_test, self.weight)

    @staticmethod
    def _fit_intercept(data):
        intercept = np.ones((data.shape[0], 1))
        return np.column_stack((intercept, data))

    @staticmethod
    def _normalize_features(features):
        norms = np.linalg.norm(features, axis=0)
        normalized_features = features / norms
        return normalized_features, norms
