import numpy as np
import uuid
import graphviz as gv
import copy

from indi.exceptions.modelbuilding import HyperParameterException
from indi.supervised.trees.nodes import Leaf, Internal
from indi.supervised.trees.util import information_gain
from indi.supervised.trees.util import predict_node_probability
from indi.supervised.trees.util import predict_node_label


class ClassificationTree(object):
    def __init__(self, max_depth=None, n_min_leaf=2, n_trials=None):
        self.max_depth = max_depth
        self.n_min_leaf = n_min_leaf
        self.n_trials = n_trials
        self._root = None
        self._n_classes = None

    def _find_current_best_feature(self, X, response):
        unique_features = np.unique(X)
        best_info_gain = float('-inf')
        best_category = None
        for feature in range(unique_features.shape[0]):
            less_than_or_eq_indices = np.where(X <= unique_features[feature])[0]
            greater_than_indices = np.where(X > unique_features[feature])[0]
            info_gain = information_gain(response,
                                         response[less_than_or_eq_indices],
                                         response[greater_than_indices])
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_category = unique_features[feature]
        return best_category, best_info_gain

    def _find_split_parameters(self, X, Y, n_min_leaf=None, n_trials=None):

        if n_min_leaf is not None and n_min_leaf >= Y.shape[0]:
            return None

        candidate_indices = None
        if n_trials is not None:
            if n_trials > X.shape[1]:
                raise HyperParameterException('n_trials should be less than number of features')
            else:
                candidate_indices = np.random.choice(X.shape[1], size=n_trials, replace=False)

        best_gain = float('-inf')
        best_dimension = None
        best_threshold = None
        if candidate_indices is None:
            candidate_indices = range(X.shape[1])

        for dim in candidate_indices:
            feature = X[:, dim]
            threshold, info_grain = self._find_current_best_feature(feature, Y)
            if info_grain >= best_gain:
                best_dimension = dim
                best_gain = info_grain
                best_threshold = threshold

        if best_threshold is None:
            return None
        else:
            return best_dimension, best_threshold

    def fit(self, X, y):
        self._n_classes = np.unique(y).shape[0]
        self._fit_training_data(X, y,
                                max_depth=self.max_depth,
                                n_min_leaf=self.n_min_leaf,
                                n_trials=self.n_trials)

    def _fit_training_data(self, X, y, max_depth=None, n_min_leaf=None, n_trials=None):
        if np.all(y == y[0]):
            return Leaf(y, self._n_classes, node_id=uuid.uuid4(),
                        description=self.build_node_description(y))

        if max_depth is not None and max_depth <= 0:
            return Leaf(y, self._n_classes, node_id=uuid.uuid4(),
                        description=self.build_node_description(y))

        split_parameters = self._find_split_parameters(X,
                                                       y,
                                                       n_min_leaf=n_min_leaf,
                                                       n_trials=n_trials)
        if split_parameters is None:
            return Leaf(y, self._n_classes, node_id=uuid.uuid4(),
                        description=self.build_node_description(y))

        split_dim, split_threshold = split_parameters
        mask_left = X[:, split_dim] <= split_threshold
        mask_right = np.logical_not(mask_left)

        left_child = self._fit_training_data(
            X[mask_left],
            y[mask_left],
            max_depth=max_depth - 1 if max_depth is not None else None)

        right_child = self._fit_training_data(
            X[mask_right],
            y[mask_right],
            max_depth=max_depth - 1 if max_depth is not None else None)

        description = 'feature:[{}] <= {}'.format(split_dim, split_threshold)
        self._root = Internal(
            dim=split_dim,
            threshold=split_threshold,
            left_child=left_child,
            right_child=right_child,
            node_id=uuid.uuid4(),
            description=description)
        return self._root

    def predict(self, X):
        y_predict = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            pointer = copy.copy(self._root);
            y_predict[i] = self._predict_data_points(X[i, :], pointer)
        return y_predict

    def predict_probability(self, X):
        y_predict = np.zeros((X.shape[0], self._n_classes))
        for i in range(X.shape[0]):
            pointer = copy.copy(self._root);
            y_predict[i,] = self._predict_data_points(X[i, :], pointer, emit_probability=True)
        return y_predict

    def _predict_data_points(self, X, node, emit_probability=False):
        if type(node) is Leaf:
            if emit_probability:
                return predict_node_probability(node.get_values(), self._n_classes)
            else:
                return predict_node_label(node.get_values(), self._n_classes)
        else:
            dim = node.dim
            feature = X[dim]
            if feature <= node.threshold:
                return self._predict_data_points(X, node.left_child, emit_probability)
            else:
                return self._predict_data_points(X, node.right_child, emit_probability)

    def visualize(self, file_name, file_format='png'):
        queue = [self._root]
        children = {}
        graph = gv.Digraph(format=file_format)

        while queue:
            vertex = queue.pop(0)
            graph.node(name=vertex.get_id(), label=vertex.get_description())
            if type(vertex) is Internal:
                queue.append(vertex.right_child)
                queue.append(vertex.left_child)
                children[vertex.get_id()] = (vertex.right_child.get_id(),
                                             vertex.left_child.get_id())

        for key, val in children.items():
            true_flag = True
            for child in val:
                if true_flag:
                    graph.edge(key, child, 'True')
                    true_flag = False
                else:
                    graph.edge(key, child, 'False')
        graph.render(file_name)

    def build_node_description(self, values):
        probabilities = predict_node_probability(values, self._n_classes)
        desc = ''
        for i in range(probabilities.shape[0]):
            desc += 'class: {}  prob: {} \n'.format(i, probabilities[i])
        desc += 'samples: {}\n'.format(values.shape[0])
        return desc
