class Node(object):
    def __init__(self, node_id=None, description=None):
        self.id = node_id
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
    def __init__(self, values, n_classes=None, node_id=None, description=None):
        Node.__init__(self, node_id, description)
        self.values = values

    def get_values(self):
        return self.values


class Internal(Node):
    def __init__(self, dim, threshold, left_child, right_child, node_id=None, description=None):
        Node.__init__(self, node_id, description)
        self.dim = dim
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
