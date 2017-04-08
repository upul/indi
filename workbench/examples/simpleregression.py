import numpy as np
from workbench.common import RegularizationType
from workbench.supervise.regression.linear_regression import LinearRegression

from sklearn.datasets.samples_generator import make_regression

x_lasso, y_lasso = make_regression(n_samples=200, n_features=10, random_state=0)
lr = LinearRegression(learning_rate=0.1,
                          max_iter=100000,
                          regularization_type=RegularizationType.L2,
                          regularization=0,
                          fit_intercept=True,
                          normalize=True,
                          tolerance=1e-5,
                          verbose=True)
lr.fit(x_lasso, y_lasso)
print(np.count_nonzero(lr.weight))
print(lr.weight)
