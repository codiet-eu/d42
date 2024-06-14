from abc import ABC

class ProbabilisticGraphicalModel(ABC):
    """
    The probabilistic graphical model consist of nodes that are connected by relations.
    """
    def __init__(self, nodes):
        """
        Constructs an instance.
        :param nodes: The set of nodes, one for each variable, in the model.
        """
        self._nodes = {node.name: node for node in nodes}