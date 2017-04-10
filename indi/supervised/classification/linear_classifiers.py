import numpy as np

from indi.solvers.regularizationtype import RegularizationType
from indi.common.math import sigmoid
from indi.exceptions.modelbuilding import HyperParameterException
from indi.solvers import solver
from indi.solvers.loss_functions import logistic_loss


class LogisticRegression:
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
        if self.fit_intercept:
            X_train = self._fit_intercept(X_train)

        if self.normalize:
            X_train, self.norm_vector = self._normalize_features(X_train)

        if self.regularization_type is None:
            self.weight, cost = solver.sgd(logistic_loss,
                                           X_train, y_train,
                                           self.learning_rate, self.max_iter,
                                           self.regularization,
                                           self.regularization_type,
                                           self.tolerance,
                                           self.verbose
                                           )
        elif self.regularization_type == RegularizationType.L2:
            pass
        else:
            raise HyperParameterException('regularization_type: {} '
                                          'is not applicable for Logistic Regression'.format
                                          (self.regularization_type))

    def predict(self, X_test, cutoff=0.5):
        if self.fit_intercept:
            X_test = self._fit_intercept(X_test)

        if self.normalize and self.norm_vector is not None:
            X_test = X_test / self.norm_vector

        probabilities = sigmoid(np.dot(X_test, self.weight))
        return probabilities, probabilities > cutoff


    @staticmethod
    def _fit_intercept(data):
        intercept = np.ones((data.shape[0], 1))
        return np.column_stack((intercept, data))

    @staticmethod
    def _normalize_features(features):
        norms = np.linalg.norm(features, axis=0)
        normalized_features = features / norms
        return normalized_features, norms

if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import numpy.testing as npt
    import matplotlib.pylab as plt
    from matplotlib import colors

    import seaborn as sns

    #sns.set(color_codes=True)

    from sklearn.datasets.samples_generator import make_blobs

    #sns.set(color_codes=True)

    #from sklearn.datasets.samples_generator import make_blobs
    n_samples = 2000
    #from sklearn.datasets.samples_generator import make_blobs
    #import matplotlib.pyplot as plt

    X, y = make_blobs(n_samples=n_samples, centers=2, n_features=2, \
    cluster_std=0.65, random_state=1788)

    _, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14, 6))
    ax1.scatter(X[:, 0], X[:, 1], c=y, alpha=0.5, edgecolors='none', s=45, cmap=plt.cm.Spectral)

    logistic = LogisticRegression(learning_rate=0.3,
                          max_iter=1250,
                          regularization_type=None,
                          regularization=1e-3,
                          fit_intercept=True,
                          normalize=False,
                          tolerance=1e-4,
                          verbose=True)
    logistic.fit(X, y)


    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 100))
    Z = logistic.predict(np.c_[xx.ravel(), yy.ravel()])[1]
    Z = Z.reshape(xx.shape)
    ax2.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.Spectral)
    ax2.scatter(X[:, 0], X[:, 1], c=y, alpha=0.5, edgecolors='none', cmap=plt.cm.Spectral, s=45)

    plt.xlim(X[:, 0].min(), X[:, 0].max())
    plt.ylim(X[:, 1].min(), X[:, 1].max())
    plt.show()
