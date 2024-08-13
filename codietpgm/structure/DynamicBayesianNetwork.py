import numpy as np
import networkx as nx
from codietpgm.structure.ProbabilisticGraphicalModel import ProbabilisticGraphicalModel
from codietpgm.structure.transitionmodels import Transition
from codietpgm.structure.transitionmodels import GaussianModel
from codietpgm.structure.graphcomponents import Node


'''
A class for storing dynamic bayesian networks. The network consists of nodes, some of which are static (inidicated by
False in node.dynamic flag). Then the main parameters are the causal graph at time t, and graph at time t-1, stored as
networkx structures. Then there are autoregresive matrix, and a matrix that stores ...
'''
class DynamicBayesianNetwork(ProbabilisticGraphicalModel):
    def __init__(self, nodes, models=None, max_lag=1, autoregressive_lag=4):
        super().__init__(nodes)
        self._nodes = nodes
        # so far, I will be trying to keep this without this cached ... static nodes should be subset of nodes, otherwise [n for n in nodes if not n.dynamic]
        #self._static_nodes = static_nodes  # Z does not depend on t
        self._transitions = {}
        self._graph_t = nx.DiGraph() #TODO ineffective, but keep it so far - better initialization would be preferred
        self._graph_t_minus_one = nx.DiGraph()
        self._autoregressive_matrix = np.zeros((len(nodes), autoregressive_lag))
        self._max_lag = max_lag
        self._autoregressive_lag = autoregressive_lag
        self._static_to_dynamic_matrix = np.zeros(sum([1 for n in nodes if n.dynamic]))  # another matrix from Z dimension to X dimension
        self._initialize_transitions(models)

    #TODO this should be private method, right? The only exposed methods shold be the constructor, and update_structure?
    def _initialize_transitions(self, models):
        for node in self.nodes.values():
            if node.dynamic:
                #model is complete mess - they are in models map, in the nodes as well ... this needs to be sorted out to remove them from one or the other
                # TODO: remove from the update methods ...
                input_nodes_current = self.determine_input_nodes(node, self._graph_t)
                input_nodes_previous = self.determine_input_nodes(node, self._graph_t_minus_one)
                model = self.choose_model(node, input_nodes_current + input_nodes_previous, models)
                self._transitions[node.name] = Transition(model, input_nodes_current + input_nodes_previous)

    def _determine_input_nodes(self, node, graph):
        #TODO why? if static nodes are supposed to be subset of nodes?
        return [self.nodes.get(n, self._static_nodes.get(n)) for n in graph.predecessors(node.name)]

    def _choose_model(self, node, input_nodes, models):
        # TODO node should be hashable with node.name as key, searching using name in dictionaries is unfriendly
        model_type = models.get(node.name) if models and node.name in models else GaussianModel
        return model_type(input_nodes, self.backend)

    def step(self, values):
        new_values = {}
        for node_name, transition in self._transitions.items():
            node = self._nodes[node_name]
            if node.dynamic:
                data = [values[n] for n in transition.input_nodes] # TODO how is the proper order of input nodes enforced?
                new_values[node_name] = transition.evaluate(data)

        for name, value in new_values.items():
            node = self._nodes[name]
            node.time_index = node.time_index + 1 #  copy constructor was here originally, no need for that, as the original value was rewritten

    # TODO, ok, here, we fill in a new graph of predecessor, however, how the transition parameters get to the "model class"
    def update_structure(self, new_graph_t=None, new_graph_t_minus_one=None, models=None):
        # polymorphism on this method to have the possibility to update only one / None as a parameter to note that nothig changed
        self._graph_t = new_graph_t
        self._graph_t_minus_one = new_graph_t_minus_one

        # original was private update transitions method for loop over all nodes times for loop over all nodes,
        # that did not make sense at all, also did not contain a way to pass new parameters
        self._initialize_transitions(models)

    def get_graph_t(self):
        return self._graph_t  # TODO view instead?, Is it Pythonish?

    def get_graph_t_minus_one(self):
        return self._graph_t_minus_one

    def get_autoregressive_matrix(self):
        return self._autoregressive_matrix

    def get_static_to_dynamic_matrix(self):
        return self._static_to_dynamic_matrix


                
