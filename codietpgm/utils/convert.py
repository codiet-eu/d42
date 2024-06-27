import numpy as np
import networkx as nx


def m2graph(dbn_matrix):
    """
    from adjacency matrix to directed graph
    """
    rows, cols = np.where(dbn_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    return graph
