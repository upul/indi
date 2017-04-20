import numpy as np


class Node(object):
    def __init__(self, id=None, description=None):
        self.id = id
        self.description = description

    def set_id(self, id):
        self.id = id

    def get_id(self):
        return self.id

    def set_description(self, description):
        self.description = description

    def get_description(self):
        return self.description


class Leaf(Node):
    def __init__(self, values, id=None, description=None):
        Node.__init__(self, id, description)
        self.values = values

    def predict(self):
        probs = self.predict_probability()
        return np.argmax(probs)

    def predict_probability(self):
        return np.bincount(self.values) / self.values.shape[0]


class Internal(Node):
    def __init__(self, dim, threshold, left_child, right_child, id=None, description=None):
        Node.__init__(self, id, description)
        self.dim = dim
        self.threshold = threshold
        self.left_child = left_child