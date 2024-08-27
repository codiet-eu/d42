from codietpgm.learners.BayesianNetworkLearner import BayesianNetworkLearner
from codietpgm.io.variableannotation import Type
import gurobipy as gp
from gurobipy import GRB
import networkx as nx

from codietpgm.structure.BayesianNetwork import BayesianNetwork


class MILPBN(BayesianNetworkLearner):
    """
    See : https://www.jmlr.org/papers/volume24/20-536/20-536.pdf
    """

    def __init__(self):
        super().__init__(True)
        self.variables = None
        self._model = None
        self.adjacency = None

    def learn_weights(self, data, lambda_n=0.5, eps = 1e-6):
        super().learn_weights(data)

        if lambda_n < 0:
            raise ValueError("Regularization parameters lambda_AW+- need to be non-negative.")

        # make sure we have dataframe with only continous static features
        data = data.project(data.variables_for_annotation(Type.CONTINUOUS))
        df = data.static_data_as_df()

        n, m = df.shape  # n=data samples num, m=n fetures
        x = df.to_numpy()

        # create gurobi model
        model = gp.Model("ILPBN")

        # include variables
        beta = model.addMVar((m, m), lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="beta")
        g = model.addMVar((m, m), vtype=GRB.BINARY, name="g")
        layer = model.addMVar(m, lb=1.0 - eps, ub=m + eps, vtype=GRB.CONTINUOUS, name="layer")

        # Add the constraints
        model.addConstrs((1 - m + m * g[j, k] <= layer[k] - layer[j] for j in range(m) for k in range(m)), name="(14a)")
        model.addConstrs((beta[j, k] * (1 - g[j, k]) == 0 for j in range(m) for k in range(m)), name="(13c)")

        # define the objective function
        model.setObjective(
            sum((x[d, k] - x[d, :] @ beta[:, k]) * (x[d, k] - x[d, :] @ beta[:, k]) for d in range(n) for k in range(m)) + lambda_n * g.sum(),
            GRB.MINIMIZE) # name="(13a)"

        model.optimize()

        variables = df.keys()
        graph = nx.DiGraph()
        graph.add_nodes_from(variables)
        graph.add_weighted_edges_from(self._get_edges(g, beta, variables))

        bn = BayesianNetwork()
        return graph  # TODO return bn insttead once working ...

    def _get_edges(g, beta, variables):
        edge_list = set()
        for i in range(len(variables)):
            for j in range(len(variables)):
                if int(g[i, j].item().X) == 1:
                    edge_list.add((variables[i], variables[j], beta[i, j].item().X))
        return edge_list
