from codietpgm.models.BayesianNetwork import BayesianNetwork
from codietpgm.io.variableannotation import Type
import gurobipy as gp
from gurobipy import GRB


class ILPBN(BayesianNetwork):
    """
    See : https://www.jmlr.org/papers/volume24/20-536/20-536.pdf
    """

    def __init__(self):
        super().__init__(True)
        self.variables = None
        self._model = None
        self.adjacency = None

    def learn_weights(self, data, lambda_n=0.5):
        super().learn_weights(data)

        # make sure we have dataframe with only continous static features
        data = data.project(data.variables_for_annotation(Type.CONTINUOUS))
        df = data.static_data_as_df()

        n, m = df.shape  # n=data samples num, m=n fetures
        x = df.to_numpy()

        # create gurobi model
        model = gp.Model("ILPBN")

        # include variables
        beta = model.addMVar((m, m), vtype=GRB.CONTINUOUS, name="beta")
        g = model.addMVar((m, m), vtype=GRB.BINARY, name="g")
        layer = model.addMVar(m, lb=1.0, ub=m, vtype=GRB.CONTINUOUS, name="layer")

        # Add the constraints
        model.addConstrs((1 - m + m * g[j, k] <= layer[k] - layer[j] for j in range(m) for k in range(m)), name="(14a)")
        model.addConstrs((beta[j, k] * (1 - g[j, k]) == 0 for j in range(m) for k in range(m)), name="(13c)")

        # define the objective function
        model.setObjective(
            sum((x[d, k] - x[d, :] @ beta[:, k]) * (x[d, k] - x[d, :] @ beta[:, k]) for d in range(n) for k in range(m)) + lambda_n * g.sum(),
            GRB.MINIMIZE) # name="(13a)"

        model.optimize()

        self._model = model
        self.variables = df.keys()
        self.adjacency = g

    def get_edges(self):
        edge_list = set()
        for i in range(len(self.variables)):
            for j in range(len(self.variables)):
                if int(self.adjacency[i, j].item().X) == 1: #TODO maybe a better conversion will be needed ...
                    edge_list.add((self.variables[i], self.variables[j]))
        return edge_list
