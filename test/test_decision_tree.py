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








