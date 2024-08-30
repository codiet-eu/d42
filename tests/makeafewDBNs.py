import random
import numpy as np
import networkx as nx

from codietpgm.structure.DynamicBayesianNetwork import DynamicBayesianNetwork, GaussianDBN, BinaryDBN, LSEMDBN
from codietpgm.structure.transitionmodels import Transition
from codietpgm.structure.transitionmodels import StatisticalModel
from codietpgm.structure.node import Node


def generate_random_nodes(s, n, lag, distribution_type):
    """
    Generate random nodes for a Dynamic Bayesian Network.

    Parameters:
    - s: Number of static nodes.
    - n: Number of dynamic nodes.
    - lag: Maximum time lag.
    - distribution_type: Type of distribution for nodes ('Gaussian', 'Bernoulli', 'Real').

    Returns:
    - List of Node objects.
    """
    nodes = []

    # Generate static nodes
    for i in range(s):
        node_name = f"Z_{i}"
        if distribution_type == 'Bernoulli':
            value = np.random.binomial(1,0.5)  # Randomly initialize value for static nodes
        elif distribution_type == 'Gaussian':
            value = np.random.normal()  # Randomly initialize value for static nodes
        elif distribution_type == 'Real':
            value = np.random.lognormal()  # Randomly initialize value for static nodes
        node = Node(
            name=node_name,
            node_type="static",
            num=i,
            distribution=distribution_type,
            value=value,
            dynamic=False
        )
        nodes.append(node)

    # Generate dynamic nodes with varying time indices from 0 to -lag
    node_counter = s  # Start numbering after static nodes
    for t in range(0, -lag - 1, -1):  # Time indices from 0 to -lag
        for i in range(n):
            node_name = f"X_{i}_t{t}"
            if t<0:
             if distribution_type == 'Bernoulli':
               value = np.random.binomial(1,0.5) 
             elif distribution_type == 'Gaussian':
               value = np.random.normal()  
             elif distribution_type == 'Real':
               value = np.random.lognormal()
            else:
             value = None
            node = Node(
                name=node_name,
                node_type="dynamic",
                num=node_counter,
                distribution=distribution_type,
                value=value,
                dynamic=True,
                time_index=t
            )
            nodes.append(node)
            node_counter += 1

    return nodes

def create_random_dependencies(nodes, lag):
    """
    Create random dependencies for static and dynamic nodes.

    Parameters:
    - nodes: List of Node objects.
    - lag: Maximum time lag.

    Returns:
    - Tuple (graph_t, static_dep, autoregressive_tensor) representing the DBN structure.
    """
    num_nodes = len(nodes)
    
    # Create a random directed acyclic graph (DAG) for in-time dependencies
    graph_t = nx.DiGraph()
    graph_t.add_nodes_from(range(num_nodes))
    edges = []

    for i in range(num_nodes):
        for j in range(i+1, num_nodes):  # Ensure acyclic by only adding edges from i to j where i < j
            if random.random() < 0.2:  # 20% chance of creating an edge
                edges.append((i, j))

    graph_t.add_edges_from(edges)

    # Create a random static dependency matrix
    static_dep = np.random.randint(0, 2, (num_nodes, num_nodes))

    # Create a random autoregressive tensor for lagged dependencies
    autoregressive_tensor = np.zeros((lag, num_nodes, num_nodes), dtype=int)
    for l in range(lag):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if random.random() < 0.1:  # 10% chance of dependency at this lag
                    autoregressive_tensor[l, i, j] = 1

    return graph_t, static_dep, autoregressive_tensor


def instantiate_random_gaussian_dbn(s, n, lag):
    """
    Instantiate a random Gaussian Dynamic Bayesian Network.

    Parameters:
    - s: Number of static nodes.
    - n: Number of dynamic nodes.
    - lag: Maximum time lag.

    Returns:
    - An instance of GaussianDBN.
    """
    nodes = generate_random_nodes(s, n, lag, 'Gaussian')
    graph_t, static_dep, autoregressive_tensor = create_random_dependencies(nodes, lag)
    
    # Instantiate the Gaussian DBN
    dbn = GaussianDBN(
        nodes=nodes,
        max_lag=lag,
        graph_t=graph_t,
        static_dep=static_dep,
        autoregressive_tensor=autoregressive_tensor
    )

    return dbn


def instantiate_random_binary_dbn(s, n, lag, model_type='Linear'):
    """
    Instantiate a random Binary Dynamic Bayesian Network.

    Parameters:
    - s: Number of static nodes.
    - n: Number of dynamic nodes.
    - lag: Maximum time lag.
    - model_type: Type of binary model ('LinearBinaryModel', 'CPDBinaryModel', 'NoisyOrModel').

    Returns:
    - An instance of BinaryDBN.
    """
    nodes = generate_random_nodes(s, n, lag, 'Bernoulli')
    graph_t, static_dep, autoregressive_tensor = create_random_dependencies(nodes, lag)
    
    # Instantiate the Binary DBN
    dbn = BinaryDBN(
        nodes=nodes,
        model_type=model_type,
        max_lag=lag,
        graph_t=graph_t,
        static_dep=static_dep,
        autoregressive_tensor=autoregressive_tensor
    )

    return dbn


def instantiate_random_lsem_dbn(s, n, lag):
    """
    Instantiate a random Linear Structural Equation Model Dynamic Bayesian Network.

    Parameters:
    - s: Number of static nodes.
    - n: Number of dynamic nodes.
    - lag: Maximum time lag.

    Returns:
    - An instance of LSEMDBN.
    """
    nodes = generate_random_nodes(s, n, lag, 'Real')
    graph_t, static_dep, autoregressive_tensor = create_random_dependencies(nodes, lag)
    
    # Instantiate the LSEM DBN
    dbn = LSEMDBN(
        nodes=nodes,
        max_lag=lag,
        graph_t=graph_t,
        static_dep=static_dep,
        autoregressive_tensor=autoregressive_tensor
    )

    return dbn



# Instantiate a random Gaussian DBN
gaussian_dbn = instantiate_random_gaussian_dbn(s=2, n=3, lag=2)

# Print basic information about the Gaussian DBN
print("Gaussian DBN Nodes:")
for node in gaussian_dbn.nodes.values():
    print(f"Node {node.name} - Dynamic: {node.dynamic}, Time Index: {node.time_index}, Model: {node.model}, Value: {node.value}")

# Perform a step in the Gaussian DBN
gaussian_dbn.step()
print("\nAfter stepping (Gaussian DBN):")
for node in gaussian_dbn.nodes.values():
    print(f"Node {node.name} - Value: {node.value}")

# Instantiate a random Binary DBN with a linear model
binary_dbn = instantiate_random_binary_dbn(s=2, n=3, lag=2)

# Print basic information about the Binary DBN
print("\nBinary DBN Nodes:")
for node in binary_dbn.nodes.values():
    print(f"Node {node.name} - Dynamic: {node.dynamic}, Time Index: {node.time_index}, Model: {node.model}, Value: {node.value}")

# Perform a step in the Binary DBN
binary_dbn.step()
print("\nAfter stepping (Binary DBN):")
for node in binary_dbn.nodes.values():
    print(f"Node {node.name} - Value: {node.value}")

# Instantiate a random LSEM DBN
lsem_dbn = instantiate_random_lsem_dbn(s=2, n=3, lag=2)

# Print basic information about the LSEM DBN
print("\nLSEM DBN Nodes:")
for node in lsem_dbn.nodes.values():
    print(f"Node {node.name} - Dynamic: {node.dynamic}, Time Index: {node.time_index}, Model: {node.model}, Value: {node.value}")

# Perform a step in the LSEM DBN
lsem_dbn.step()
print("\nAfter stepping (LSEM DBN):")
for node in lsem_dbn.nodes.values():
    print(f"Node {node.name} - Value: {node.value}")

