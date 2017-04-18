import matplotlib.pylab as plt
import numpy as np
import seaborn as sbs;

sbs.set()
from sklearn.datasets.samples_generator import make_blobs
from indi.supervised.classification.linear_classifiers import LogisticRegression

n_samples = 5000
X, y = make_blobs(n_samples=n_samples, centers=2, n_features=2,
                  cluster_std=0.65, random_state=1788)
logistic = LogisticRegression(learning_rate=0.1,
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
_, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14, 6))
ax1.scatter(X[:, 0], X[:, 1], c=y, alpha=0.5, edgecolors='none', s=45, cmap=plt.cm.Spectral)
ax2.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.Spectral)
ax2.scatter(X[:, 0], X[:, 1], c=y, alpha=0.5, edgecolors='none', cmap=plt.cm.Spectral, s=45)

plt.xlim(X[:, 0].min(), X[:, 0].max())
plt.ylim(X[:, 1].min(), X[:, 1].max())
plt.show()