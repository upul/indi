from src.common.regularization import Regularization
from src.solvers import solver
from src.solvers.loss_functions import mean_squared, mean_squared_l1_loss, mean_squared_l2_loss
import numpy as np
from src.metrics.regression import mean_squared_error, root_mean_squated_error


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
        X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))

        if self.regularization is None or self.regularization.L2:
            loss_function = mean_squared
            if self.regularization == Regularization.L2:
                loss_function = mean_squared_l2_loss
            self.weight, cost = solver.sgd(loss_function, X_train, y_train,
                                           self.learning_rate, self.max_iter,
                                           self.verbose)
        elif self.regularization == Regularization.L2:
            self.weight = solver.lasso_coordinate_descent(X_train, y_train, )


    def predict(self, X_test):
        """

        :param X_test:
        :return:
        """
        X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
        return np.dot(X_test, self.weight)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sbn;
    from sklearn import datasets

    # Load the diabetes dataset
    diabetes = datasets.load_diabetes()

    # Use only one feature
    diabetes_X = diabetes.data[:, np.newaxis, 2]

    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-200]
    diabetes_X_test = diabetes_X[-200:]

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes.target[:-200]
    diabetes_y_test = diabetes.target[-200:]

    lr = LinearRegression(learning_rate=0.1, max_iter=80000, regularization=Regularization.L2)
    lr.fit(diabetes_X_train, diabetes_y_train)
    print(lr.weight)

    y_predict = lr.predict(diabetes_X_test)
    print("Mean squared error: %.2f"
          % np.mean((y_predict - diabetes_y_test) ** 2))
    print(mean_squared_error(y_predict, diabetes_y_test))
    print(root_mean_squated_error(y_predict, diabetes_y_test))

    with plt.style.context('seaborn-deep'):
        plt.scatter(diabetes_X_test, diabetes_y_test, color='#dd1c77')
        plt.plot(diabetes_X_test, y_predict, color='#c994c7', linewidth=2)
        plt.show()


