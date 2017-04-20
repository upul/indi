import numpy as np
from .util import information_gain


class ClassificationTree(object):
    def __init__(self, max_depth, n_min_leaf, n_trials):
        self.max_depth = max_depth
        self.n_min_leaf = n_min_leaf
        self.n_trials = n_trials
        self._root = None

    def find_current_best_feature(self, X, response):
        unique_features = np.unique(X)
        best_info_gain = 1e-10
        best_category = None
        for feature in range(unique_features.shape[0]):
            less_than_or_eq_indices = np.where(X <= X[feature])[0]
            greater_than_indices = np.where(X > X[feature])[0]
            info_gain = information_gain(response,
                                         response[less_than_or_eq_indices],
                                         response[greater_than_indices])
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_category = X[feature]
        return best_category, best_info_gain

    def _find_split_parameters(self, X, Y, n_min_leaf, n_trials):
        best_gain = 1e-10
        best_dimension = -1
        best_threshold = None

        for dim in range(X.shape[1]):
            feature = X[:, dim]
            threshold, info_grain = self.find_current_best_feature(feature, Y)
            if info_grain >= best_gain:
                best_dimension = dim
                best_gain = info_grain
                best_threshold = threshold

        if best_threshold is None:
            return None
        else:
            return best_dimension, best_threshold

    def fit(self, X, y):
        pass
