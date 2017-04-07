from src.common.regularizationtype import RegularizationType
from src.solvers import solver
from src.solvers.loss_functions import mean_squared, mean_squared_l2_loss


class LinearRegression:
    """

    """

    def __init__(self,
                 learning_rate=1e-3,
                 regularization_type=None,
                 regularization=1e-3,
                 max_iter=1e2,
                 tolerance=1e5,
                 fit_intercept=True,
                 normalize=True,
                 verbose=False):
        """

        :param learning_rate:
        :param regularization_type:
        :param max_iter:
        :param verbose:
        """
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
                                           self.verbose)
        elif self.regularization_type == RegularizationType.L1:
            self.weight = solver.lasso_coordinate_descent(X_train,
                                                          y_train,
                                                          self.regularization,
                                                          self.tolerance)
        else:
            # exception
            pass

    def predict(self, X_test):
        """

        :param X_test:
        :return:
        """
        if self.fit_intercept:
            if self.norm_vector is None:
                pass
            X_test = X_test / self.norm_vector

        X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
        return np.dot(X_test, self.weight)

    def _fit_intercept(self, data):
        intercept = np.ones((data.shape[0], 1))
        return np.column_stack((intercept, data))

    def _normalize_features(self, features):
        norms = np.linalg.norm(features, axis=0)
        normalized_features = features / norms
        return normalized_features, norms


if __name__ == '__main__':
    import numpy as np
    # import seaborn as sbn;

    # # Load the diabetes dataset
    # diabetes = datasets.load_diabetes()
    #
    # # Use only one feature
    # diabetes_X = diabetes.data[:, np.newaxis, 2]
    #
    # # Split the data into training/testing sets
    # diabetes_X_train = diabetes_X[:-200]
    # diabetes_X_test = diabetes_X[-200:]
    #
    # # Split the targets into training/testing sets
    # diabetes_y_train = diabetes.target[:-200]
    # diabetes_y_test = diabetes.target[-200:]
    #
    # lr = LinearRegression(learning_rate=0.1, max_iter=80000, regularization=Regularization.L2)
    # lr.fit(diabetes_X_train, diabetes_y_train)
    # print(lr.weight)
    #
    # y_predict = lr.predict(diabetes_X_test)
    # print("Mean squared error: %.2f"
    #       % np.mean((y_predict - diabetes_y_test) ** 2))
    # print(mean_squared_error(y_predict, diabetes_y_test))
    # print(root_mean_squated_error(y_predict, diabetes_y_test))
    #
    # with plt.style.context('seaborn-deep'):
    #     plt.scatter(diabetes_X_test, diabetes_y_test, color='#dd1c77')
    #     plt.plot(diabetes_X_test, y_predict, color='#c994c7', linewidth=2)
    #     plt.show()

    from sklearn.datasets.samples_generator import make_regression

    x_lasso, y_lasso = make_regression(n_samples=200, n_features=10, random_state=0)
    # intercept = np.ones((x_lasso.shape[0], 1))
    # x_lasso = np.column_stack((intercept, x_lasso))

    # x_lasso = np.array([[0,0], [1, 1], [2, 2]])
    # print(x_lasso)
    # x_lasso, _ = normalize_features(x_lasso)
    # print(x_lasso)
    # y_lasso = np.array([0, 1, 2])

    lr = LinearRegression(learning_rate=0.05,
                          max_iter=100000,
                          regularization_type=RegularizationType.L2,
                          regularization=1e-8,
                          fit_intercept=True,
                          normalize=True,
                          verbose=False)
    lr.fit(x_lasso, y_lasso)
    print(np.count_nonzero(lr.weight))
    print(lr.weight)
    # print(np.count_nonzero(lr.weight))

    # lasso = Lasso()
    # lasso.fit(x_lasso, y_lasso)
    # print('done')
    # print(np.count_nonzero(lasso.coef_));
