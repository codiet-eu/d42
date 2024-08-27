import numpy as np
import networkx as nx
from codietpgm.structure.ProbabilisticGraphicalModel import ProbabilisticGraphicalModel
from codietpgm.structure.transitionmodels import Transition
from codietpgm.structure.transitionmodels import GaussianModel
from codietpgm.structure.graphcomponents import Node


class BayesianNetwork(ProbabilisticGraphicalModel):
    def __init__(self, nodes, models=None):
        super().__init__(nodes)
        self._transitions = {}
        self._graph_t = nx.DiGraph()
        self._initialize_transitions(models)

    def initialize_transitions(self, models):
        for node in self._nodes.values():
            if node.dynamic:
                input_nodes_current = self.determine_input_nodes(node, self._graph_t)
                model = self.choose_model(node, input_nodes_current, models)
                self._transitions[node.name] = Transition(model, input_nodes_current)

    def determine_input_nodes(self, node, graph):
        return [self._nodes.get(n) for n in graph.predecessors(node.name)]

    def choose_model(self, node, input_nodes, models):
        model_type = models.get(node.name) if models and node.name in models else GaussianModel
        return model_type(input_nodes, self._backend)

    def step(self):
        new_values = {}
        for node_name, transition in self._transitions.items():
            node = self._nodes[node_name]
            if node.dynamic:
                data = [n.value for n in transition.input_nodes]
                new_values[node_name] = transition.evaluate(data)

    def update_structure(self, new_graph_t, new_graph_t_minus_one):
        self._graph_t = new_graph_t
        self.update_transitions()

    def update_transitions(self):
        for node in self._nodes.values():
            if node.dynamic:
                self.initialize_transitions(None)


