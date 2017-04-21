import numpy.testing as npt
import numpy as np

from indi.supervised.trees.nodes import Leaf
from indi.supervised.trees import ClassificationTree

y_unique_all_1 = np.array([1, 1, 1, 1, 1])
y_unique_all_2 = np.array([2, 2, 2, 2, 2])
y_mix = np.array([1, 2, 3, 1, 1])


def test_leaf_node_predict():
    leaf = Leaf(y_unique_all_2)
    output = leaf.predict()
    npt.assert_almost_equal(output, 2)

    leaf = Leaf(y_unique_all_1)
    output = leaf.predict()
    npt.assert_almost_equal(output, 1)

    leaf = Leaf(y_unique_all_2)
    output = leaf.predict()
    npt.assert_almost_equal(output, 2)

    leaf = Leaf(y_mix)
    output = leaf.predict()
    npt.assert_almost_equal(output, 1)

    leaf = Leaf(y_mix)
    output = leaf.predict_probability()
    npt.assert_array_equal(output, np.array([0.0, 0.6, 0.2, 0.2]))

    X = np.array([[1, 1], [2, 1], [4, 1], [5, 1], [3, 10]])
    y = np.array(([0, 0, 1, 1, 2]))
    t = ClassificationTree(max_depth=100, n_min_leaf=1, n_trials=100)
    t.fit(X, y)
    t.visualize('./img.png')
    print('O')

    from sklearn.datasets.samples_generator import make_blobs
    import matplotlib.pylab as plt
    import seaborn as sbs;
    n_samples = 1000
    X, y = make_blobs(n_samples=n_samples, centers=2, n_features=2,
                      cluster_std=0.65, random_state=1788)

    cls = ClassificationTree(max_depth=100, n_min_leaf=1, n_trials=100)
    root = cls._fit_training_data(X, y)

    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 100))
    #Z = []
    data = np.c_[xx.ravel(), yy.ravel()]
    Z = cls.predict(data)
    #Z = np.array(Z)
    Z = Z.reshape(xx.shape)
    _, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14, 6))
    ax1.scatter(X[:, 0], X[:, 1], c=y, alpha=0.5, edgecolors='none', s=45, cmap=plt.cm.Spectral)
    ax2.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.Spectral)
    ax2.scatter(X[:, 0], X[:, 1], c=y, alpha=0.5, edgecolors='none', cmap=plt.cm.Spectral, s=45)

    plt.xlim(X[:, 0].min(), X[:, 0].max())
    plt.ylim(X[:, 1].min(), X[:, 1].max())
    plt.show()
