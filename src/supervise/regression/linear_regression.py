from src.common.regularization import Regularization
from src.solvers import solver
from src.solvers.loss_functions import mean_squared, mean_squared_l1_loss, mean_squared_l2_loss
import numpy as np


class LinearRegression:
    """

    """

    def __init__(self, learning_rate=0.01, regularization=None,
                 max_iter=100, verbose=False):
        """

        :param learning_rate:
        :param regularization:
        :param max_iter:
        :param verbose:
        """
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.max_iter = max_iter
        self.verbose = verbose

        self.weight = None

    def fit(self, X_train, y_train):
        """

        :param X_train:
        :param y_train:
        :return:
        """
        loss_function = mean_squared
        if self.regularization == Regularization.L2:
            loss_function = mean_squared_l2_loss
        elif self.regularization == Regularization.L2:
            loss_function = mean_squared_l1_loss

        X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
        self.weight, cost = solver.sgd(loss_function, X_train, y_train,
                                       self.learning_rate, self.max_iter,
                                       self.verbose)

    def predict(self, X_test):
        """

        :param X_test:
        :return:
        """
        X_test = np.hstack((X_test, np.ones(X_test.shape[1])))
        return np.dot(X_test, self.weight)

if __name__ == '__main__':
    X = np.array([[1, 2, 3], [1,3, 4]])
    y = np.array([1, 2])
    lr = LinearRegression()
    lr.fit(X, y)
