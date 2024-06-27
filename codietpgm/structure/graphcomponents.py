class Node:
    def __init__(self, name, node_type, distribution, model=None, observed=False, dynamic=False, time_index=None):
        self._name = name
        self._node_type = node_type
        self._distribution = distribution
        self._label = name
        self._time_index = time_index
        self._model = model
        self._observed = observed
        self._dynamic = dynamic


class Edge:
    def __init__(self, from_node, to_node):
        self._from_node = from_node
        self._to_node = to_node


