import numpy as np
from indi.ensemble import RandomForestClassifier

X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]])
y = np.array([0, 0, 0, 1, 1, 1])
T = np.array([[-1, -1], [2, 2], [3, 2]])
true_result = np.array([0, 1, 1])

def test_random_forest_classifier():
    #cls = RandomForestClassifier(max_depth=5, n_trees=120, n_trials=1)
    #cls.fit(X, y)
    #print(cls.predict(T))

    from sklearn.datasets.samples_generator import make_blobs
    import matplotlib.pylab as plt
    import seaborn as sbs;
    n_samples = 10000
    X, y = make_blobs(n_samples=n_samples, centers=2, n_features=2,
                      cluster_std=0.62, random_state=125)
    #X, y = make_blobs(n_samples=300, centers=4,
                      #random_state=0, cluster_std=1.0)

    cls = RandomForestClassifier(max_depth=125, n_trees=50, n_trials=1, n_min_leaf=1)
    cls.fit(X, y)
    #cls.visualize('./test.png')

    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 100))
    # Z = []
    data = np.c_[xx.ravel(), yy.ravel()]
    Z = cls.predict(data)
    # Z = np.array(Z)
    Z = Z.reshape(xx.shape)
    _, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14, 6))
    ax1.scatter(X[:, 0], X[:, 1], c=y, alpha=0.5, edgecolors='none', s=45, cmap=plt.cm.Spectral)
    ax2.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.Spectral)
    ax2.scatter(X[:, 0], X[:, 1], c=y, alpha=0.5, edgecolors='none', cmap=plt.cm.Spectral, s=45)

    plt.xlim(X[:, 0].min(), X[:, 0].max())
    plt.ylim(X[:, 1].min(), X[:, 1].max())
    plt.show()

    import sklearn.ensemble
    clf = sklearn.ensemble.RandomForestClassifier()
    clf.fit(X, y)
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 100))
    # Z = []
    data = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(data)
    # Z = np.array(Z)
    Z = Z.reshape(xx.shape)
    _, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14, 6))
    ax1.scatter(X[:, 0], X[:, 1], c=y, alpha=0.5, edgecolors='none', s=45, cmap=plt.cm.Spectral)
    ax2.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.Spectral)
    ax2.scatter(X[:, 0], X[:, 1], c=y, alpha=0.5, edgecolors='none', cmap=plt.cm.Spectral, s=45)

    plt.xlim(X[:, 0].min(), X[:, 0].max())
    plt.ylim(X[:, 1].min(), X[:, 1].max())
    plt.show()
