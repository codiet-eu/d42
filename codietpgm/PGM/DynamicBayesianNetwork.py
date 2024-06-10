import numpy as np
import networkx as nx
from itertools import permutations
import numpy as np
import networkx as nx


class DynamicBayesianNetwork:
    def __init__(self, nodes, static_nodes, backend='numpy', models=None, max_lag=1, autoregressive_lag=4):
        self.backend = GenericBackend(backend)
        self.nodes = {node.name: node for node in nodes}
        self.static_nodes = {node.name: node for node in static_nodes}
        self.transitions = {}
        self.graph_t = nx.DiGraph()
        self.graph_t_minus_one = nx.DiGraph()
        self.autoregressive_matrix = np.zeros((len(nodes), autoregressive_lag))
        self.max_lag = max_lag
        self.autoregressive_lag = autoregressive_lag
        self.initialize_transitions(models)

    def initialize_transitions(self, models):
        for node in self.nodes.values():
            if node.dynamic:
                input_nodes_current = self.determine_input_nodes(node, self.graph_t)
                input_nodes_previous = self.determine_input_nodes(node, self.graph_t_minus_one)
                model = self.choose_model(node, input_nodes_current + input_nodes_previous, models)
                self.transitions[node.name] = Transition(model, input_nodes_current + input_nodes_previous)

    def determine_input_nodes(self, node, graph):
        return [self.nodes.get(n, self.static_nodes.get(n)) for n in graph.predecessors(node.name)]

    def choose_model(self, node, input_nodes, models):
        model_type = models.get(node.name) if models and node.name in models else GaussianModel
        return model_type(input_nodes, self.backend)

    def step(self):
        new_values = {}
        for node_name, transition in self.transitions.items():
            node = self.nodes[node_name]
            if node.dynamic:
                data = [n.value for n in transition.input_nodes]
                new_values[node_name] = transition.evaluate(data)

        for name, value in new_values.items():
            node = self.nodes[name]
            new_node = Node(name=node.name, node_type=node.node_type, distribution=node.distribution,
                            model=node.model, observed=node.observed, dynamic=node.dynamic,
                            label=node.label, time_index=node.time_index + 1)
            new_node.value = value
            self.nodes[name] = new_node

    def update_structure(self, new_graph_t, new_graph_t_minus_one):
        self.graph_t = new_graph_t
        self.graph_t_minus_one = new_graph_t_minus_one
        self.update_transitions()

    def update_transitions(self):
        for node in self.nodes.values():
            if node.dynamic:
                self.initialize_transitions(None)
                
                
