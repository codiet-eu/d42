import numpy as np
import networkx as nx
from codietpgm.structure.ProbabilisticGraphicalModel import ProbabilisticGraphicalModel
from codietpgm.structure.transitionmodels import Transition
from codietpgm.structure.transitionmodels import StatisticalModel
from codietpgm.structure.graphcomponents import Node, Edge

class DynamicBayesianNetwork(ProbabilisticGraphicalModel):
    def __init__(self, nodes, backend=np, models=None, max_lag=4, graph_t=None, static_dep=None, autoregressive_tensor=None):
        self._nodes = {(1 if node._dynamic else 0, node._time_index or 0, node._num): node for node in nodes}
        self.backend = backend
        self._transitions = {}
        self._static_nodes = [(0, 0, n._num) for n in nodes if not n._dynamic]
        self._active_nodes = [(1, 0, n._num) for n in nodes if n._dynamic and n._time_index == 0]
        self._nz = len(self._static_nodes)
        self._max_lag = max_lag
        self._nx = (len(nodes) - self._nz) / (max_lag + 1)

        # Initialize graph_t if not provided
        self._graph_t = nx.DiGraph() if graph_t is None else graph_t
        
        # Initialize static_dep if not provided
        self._static_dep = np.zeros((self._nz, int(self._nx))) if static_dep is None else static_dep
        
        # Initialize autoregressive_tensor if not provided
        self._autoregressive_tensor = np.zeros((max_lag, len(nodes), len(nodes)), dtype=int) if autoregressive_tensor is None else autoregressive_tensor
        
        # Initialize models as a local variable
        self._models = models if models else {n._num: ['Custom', 'Custom'] for n in nodes if n._dynamic and n._time_index == 0}

        # Initialize transitions
        self.initialize_transitions()

    def initialize_transitions(self):
        for b, l, n in self._active_nodes:
            node = self._nodes[(b, l, n)]
            input_nodes_current = self.determine_input_nodes(node, self._graph_t)
            input_nodes_previous = self.determine_input_nodes_time(node, self._autoregressive_tensor)
            model_type = self._models.get(node._num, ['Custom', 'Custom'])
            node._model = model_type
            input_nodes_static = self.determine_input_nodes_static(node, self._static_dep)
            input_nodes = self.get_input_nodes(input_nodes_static, input_nodes_current, input_nodes_previous)
            self._transitions[(b, l, n)] = Transition(node._model, input_nodes)

    def determine_input_nodes_static(self, node, statmat):
        input_node_indices = [row for row in range(statmat.shape[0]) if statmat[row, node._num] == 1]
        return input_node_indices

    def determine_input_nodes(self, node, graph):
        return [self._nodes[(1, 0, n)] for n in graph.predecessors(node._num) if (1, 0, n) in self._nodes]

    def determine_input_nodes_time(self, node, tensor):
        input_node_indices = [(l, n) for l in range(tensor.shape[0]) for n in range(tensor.shape[1]) if tensor[l, n, node._num] == 1]
        return input_node_indices

    def get_input_nodes(self, input_nodes_static, input_nodes_current, input_nodes_previous):
        selected_nodes = []
        for b, l, n in self._nodes:
            node = self._nodes[(b, l, n)]
            if b == 0 and n in input_nodes_static:
                selected_nodes.append(node)
            elif b == 1 and l == 0 and n in input_nodes_current:
                selected_nodes.append(node)
            elif b == 1 and any(l == (lag - 1) and n == num for lag, num in input_nodes_previous):
                selected_nodes.append(node)
        return selected_nodes

    def step(self):
        new_values = {}
        initialized_nodes = set()  # Track nodes that have been initialized
        nodes_to_initialize = [node for (b, l, n), node in self._nodes.items() if node._dynamic and node._time_index == 0]

        # Sort nodes by topological order in graph_t to ensure correct initialization sequence
        topo_order = list(nx.topological_sort(self._graph_t))
        nodes_to_initialize_sorted = [node for node in nodes_to_initialize if node._num in topo_order]

        # Initialize dynamic nodes at time 0 if they haven't been initialized
        for node in nodes_to_initialize_sorted:
            if node._value is None:
                # Check if all input nodes are initialized
                transition = self._transitions[(1, 0, node._num)]
                data = [input_node._value for input_node in transition.input_nodes]
                
                if all(value is not None for value in data):
                    # All input nodes are initialized; proceed to initialize this node
                    node._value = transition.evaluate(data)
                    print(f"Initialized node {node._name} with value {node._value}")
                    initialized_nodes.add((1, 0, node._num))
                else:
                    print(f"Cannot initialize node {node._name} yet; waiting for input nodes to be initialized.")

        # If all time-lag=0 nodes are initialized, proceed to create nodes at time-lag=1
        if len(initialized_nodes) == len(nodes_to_initialize):
            for (b, l, n), transition in self._transitions.items():
                if (b, l, n) in initialized_nodes:
                    node = self._nodes[(b, l, n)]
                    data = [n._value for n in transition.input_nodes]
                    new_values[(b, l, n)] = transition.evaluate(data)

            # Create new dynamic nodes at time 1 and copy transitions
            for (b, l, n), value in new_values.items():
                node = self._nodes[(b, l, n)]
                new_node = Node(
                    name=node._name, node_type=node._type, num=node._num,
                    distribution=node._distribution, model=node._model,
                    value=value, observed=node._observed, dynamic=node._dynamic,
                    time_index=node._time_index + 1
                )
                self._nodes[(b, l + 1, n)] = new_node

                # Copy transitions from time-lag=0 to time-lag=1
                old_transition = self._transitions[(1, 0, n)]
                new_input_nodes = self.get_input_nodes(
                    self.determine_input_nodes_static(new_node, self._static_dep),
                    [inp._num for inp in old_transition.input_nodes if inp._dynamic and inp._time_index == 0],
                    self.determine_input_nodes_time(new_node, self._autoregressive_tensor)
                )
                self._transitions[(1, l + 1, n)] = Transition(node._model, new_input_nodes)

    def update_structure(self, new_graph_t=None, new_static_dep=None, new_autoregressive_tensor=None):
        """
        Update the structure of the DBN including graph_t, static dependencies, and autoregressive tensor.

        Parameters:
        - new_graph_t: Updated directed graph for in-time dependencies.
        - new_static_dep: Updated static dependency matrix.
        - new_autoregressive_tensor: Updated autoregressive tensor for lagged dependencies.
        """
        if new_graph_t is not None:
            self._graph_t = new_graph_t
        if new_static_dep is not None:
            self._static_dep = new_static_dep
        if new_autoregressive_tensor is not None:
            self._autoregressive_tensor = new_autoregressive_tensor
        
        self.update_transitions()

def initialize_transitions(self):
    for b, l, n in self._active_nodes:
        node = self._nodes.get((b, l, n))
        if not node:
            continue  # Skip if the node is not found

        input_nodes_current = self.determine_input_nodes(node, self._graph_t)
        input_nodes_previous = self.determine_input_nodes_time(node, self._autoregressive_tensor)
        model_type = self._models.get(node._num, ['Custom', 'Custom'])
        node._model = model_type
        input_nodes_static = self.determine_input_nodes_static(node, self._static_dep)
        input_nodes = self.get_input_nodes(input_nodes_static, input_nodes_current, input_nodes_previous)

        if input_nodes:  # Ensure input nodes are not empty
            self._transitions[(b, l, n)] = Transition(node._model, input_nodes)




                
class GaussianDBN(DynamicBayesianNetwork):
    def __init__(self, nodes, model_type=['Gaussian','Gaussian'], max_lag=4, graph_t=None, static_dep=None, autoregressive_tensor=None):
        dynamic_nodes_lag0 = [(1, 0, node._num) for node in nodes if node._dynamic and node._time_index == 0]
        models = {num: model_type for _, _, num in dynamic_nodes_lag0}

        # Initialize Dynamic Bayesian Network
        super().__init__(nodes=nodes,
                         models=models,
                         max_lag=max_lag,
                         graph_t=graph_t,
                         static_dep=static_dep,
                         autoregressive_tensor=autoregressive_tensor)

class BinaryDBN(DynamicBayesianNetwork):
    def __init__(self, nodes, model_type='Linear', max_lag=4, graph_t=None, static_dep=None, autoregressive_tensor=None):
        dynamic_nodes_lag0 = [(1, 0, node._num) for node in nodes if node._dynamic and node._time_index == 0]
        models = {num: ['Bernoulli', model_type] for _, _, num in dynamic_nodes_lag0}

        # Initialize Dynamic Bayesian Network
        super().__init__(nodes=nodes,
                         models=models,
                         max_lag=max_lag,
                         graph_t=graph_t,
                         static_dep=static_dep,
                         autoregressive_tensor=autoregressive_tensor)

class LSEMDBN(DynamicBayesianNetwork):
    def __init__(self, nodes, model_type=['Real','LSEM'], max_lag=4, graph_t=None, static_dep=None, autoregressive_tensor=None):
        dynamic_nodes_lag0 = [(1, 0, node._num) for node in nodes if node._dynamic and node._time_index == 0]
        models = {num: model_type for _, _, num in dynamic_nodes_lag0}

        # Initialize Dynamic Bayesian Network
        super().__init__(nodes=nodes,
                         models=models,
                         max_lag=max_lag,
                         graph_t=graph_t,
                         static_dep=static_dep,
                         autoregressive_tensor=autoregressive_tensor)

