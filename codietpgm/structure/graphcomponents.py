class Node:
    def __init__(self, name, node_type, distribution, model=None, observed=False, dynamic=False):
        self.name = name
        self.node_type = node_type
        self.distribution = distribution
        self.label = label
        self.time_index = time_index
        self.model = model
        self.observed = observed
        self.dynamic = dynamic

class Edge:
    def __init__(self, from_node, to_node):
        self.from_node = from_node
        self.to_node = to_node


