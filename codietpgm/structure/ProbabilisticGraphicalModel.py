class ProbabilisticGraphicalModel:
    def __init__(self, nodes):
        self._nodes = {node.name: node for node in nodes}