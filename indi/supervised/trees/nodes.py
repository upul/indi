import numpy as np


class Node(object):
    def __init__(self, id=None, description=None):
        self.id = id
        self.description = description

    def set_id(self, id):
        self.id = id

    def get_id(self):
        return str(self.id)

    def set_description(self, description):
        self.description = description

    def get_description(self):
        return self.description


class Leaf(Node):
    def __init__(self, values, n_classes=None, id=None, description=None):
        Node.__init__(self, id, description)
        self.values = values
        self._n_classes = n_classes

    def predict(self):
        probs = self.predict_probability()
        return np.argmax(probs)

    def predict_probability(self):
        return np.bincount(self.values, minlength=self._n_classes) / self.values.shape[0]

    def get_description(self):
        probs = self.predict_probability()
        desc = ''
        for i in range(probs.shape[0]):
            desc += 'class: {}  prob: {} \n'.format(i, probs[i])
        desc += 'samples: {}\n'.format(self.values.shape[0])
        return desc


class Internal(Node):
    def __init__(self, dim, threshold, left_child, right_child, id=None, description=None):
        Node.__init__(self, id, description)
        self.dim = dim
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child