import numpy as np
from indi.supervised.trees import ClassificationTree


class RandomForestClassifier(object):
    def __init__(self, n_trees=2, max_depth=None, n_min_leaf=1, n_trials=None):
        self.number_of_trees = n_trees
        self.max_depth = max_depth
        self.n_min_leaf = n_min_leaf
        self.n_trials = n_trials
        self.trees = []

    def fit(self, X, y):
        for i in range(self.number_of_trees):
            clf = ClassificationTree(max_depth=self.max_depth,
                                     n_min_leaf=self.n_min_leaf,
                                     n_trials=self.n_trials)
            X_bootstrap, y_bootstrap = self._bootstrap(X, y)
            clf.fit(X_bootstrap, y_bootstrap)
            self.trees.append(clf)

    def _bootstrap(self, X, Y):
        while True:
            sample = np.random.choice(X.shape[0], size=5000, replace=True)
            if np.unique(Y[sample]).shape[0] > 1:
                break
        return X[sample, :], Y[sample]

    def predict_probability(self, X):
        y_predicts = []
        for tree in self.trees:
            probabilities = tree.predict_probability(X)
            y_predicts.append(probabilities)
        return np.array(y_predicts).mean(axis=0)

    def predict(self, X):
        probabilities = self.predict_probability(X)
        return np.argmax(probabilities, axis=1)
