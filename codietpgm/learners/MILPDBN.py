from codietpgm.learners.BayesianNetworkLearner import BayesianNetworkLearner
from codietpgm.io.variableannotation import Type
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import networkx as nx

from codietpgm.structure.DynamicBayesianNetwork import DynamicBayesianNetwork
from codietpgm.structure.graphcomponents import Node
from codietpgm.structure.transitionmodels import CustomModel


class MILPDBN(BayesianNetworkLearner):
    """
    New method proposed in the draft.
    """

    def __init__(self):
        super().__init__(True)

    def learn_weights(self, data, lambda_wp=0.5, lamda_wm=0.5, lambda_ap=0.5, lamda_am=0.5, b_w=0.1, b_a=0.1, p=1,
                      eps=1e-6):
        super().learn_weights(data)

        if lamda_am < 0 or lambda_ap < 0 or lambda_wp < 0 or lambda_wp < 0:
            raise ValueError("Regularization parameters lambda_AW+- need to be non-negative.")
        if b_w < 0 or b_a < 0:
            raise ValueError("Bounds b_W and b_A need to be positive.")

        if p != 1:
            raise RuntimeError("p > 1 not yet implemented")

        # make sure we have dataframe with only continous static features
        data = data.project(data.variables_for_annotation(Type.CONTINUOUS))
        df = data.dynamic_variables()

        T, d = df[0].shape  # n=data samples num, d=num. fetures
        M = len(df)
        x = [dfi.to_numpy() for dfi in df]  # indexing x[sample][time, feature]

        # create gurobi model
        model = gp.Model("MILPDBN")

        # include variables
        ewp = model.addMVar((d, d), vtype=GRB.BINARY, name="EW+")
        ewm = model.addMVar((d, d), vtype=GRB.BINARY, name="EW-")
        eap = model.addMVar((d, d), vtype=GRB.BINARY, name="EA+")
        eam = model.addMVar((d, d), vtype=GRB.BINARY, name="EA-")
        wp = model.addMVar((d, d), lb=b_w, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="W+")
        wm = model.addMVar((d, d), lb=-GRB.INFINITY, ub=-b_w, vtype=GRB.CONTINUOUS, name="W-")
        ap = model.addMVar((d, d), lb=b_a, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="A+")
        am = model.addMVar((d, d), lb=-GRB.INFINITY, ub=-b_a, vtype=GRB.CONTINUOUS, name="A-")

        # layer variables
        layer = model.addMVar(d, lb=1 - eps, ub=d + eps, vtype=GRB.CONTINUOUS, name="layer")

        # helper variables, so that the contraint is quadratic
        hwp = model.addMVar((d, d), lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="helpwerW+")
        hwm = model.addMVar((d, d), lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="helpwerW-")
        hap = model.addMVar((d, d), lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="helpwerA+")
        ham = model.addMVar((d, d), lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="helpwerA-")

        # helper variables for the criterion
        critvar = model.addMVar((M, T, d), lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS,
                                name="criterion helper variable")

        # constraints
        # E_W+ + E_W- <= 1, and E_A+ + E_A- <= 1
        model.addConstrs((ewp[j, k] + ewm[j, k] <= 1 for j in range(d) for k in range(d)), name="E_W+ + E_W- <= 1")
        model.addConstrs((eap[j, k] + eam[j, k] <= 1 for j in range(d) for k in range(d)), name="E_A+ + E_A- <= 1")

        # DAG constraints
        model.addConstrs(((-eps if j == k else 1 - eps - d + d * (ewp[j, k] + ewm[j, k])) <= layer[k] - layer[j]
                          for j in range(d) for k in range(d)))
        model.addConstrs((ewm[j, j] == 0 for j in range(d)), name="self loop")
        model.addConstrs((ewp[j, j] == 0 for j in range(d)), name="self loop")

        # helper variables constraints
        model.addConstrs((ewp[j, k] * wp[j, k] == hwp[j, k] for j in range(d) for k in range(d)),
                         name="helper constraint on E_W+ and W+")
        model.addConstrs((ewm[j, k] * wm[j, k] == hwm[j, k] for j in range(d) for k in range(d)),
                         name="helper constraint on E_W- and W-")
        model.addConstrs((eap[j, k] * ap[j, k] == hap[j, k] for j in range(d) for k in range(d)),
                         name="helper constraint on E_A+ and A+")
        model.addConstrs((eam[j, k] * am[j, k] == ham[j, k] for j in range(d) for k in range(d)),
                         name="helper constraint on E_A- and A-")

        # criterion variable helper contraint
        model.addConstrs((critvar[m, t, i] == x[m][t, i]
                          - sum(x[m][t, j] * hwp[j, i] for j in range(d))
                          - sum(x[m][t, j] * hwm[j, i] for j in range(d))
                          - (0 if t == 0 else sum(x[m][t - 1, j] * hap[j, i] for j in range(d)))
                          - (0 if t == 0 else sum(x[m][t - 1, j] * ham[j, i] for j in range(d)))
                          for m in range(M) for t in range(T) for i in range(d)),
                         name="criterion helper variable")

        # define the objective function
        model.setObjective(
            sum((critvar[m, t, i] * critvar[m, t, i]) for m in range(M) for t in range(T) for i in range(d)) +
            # regularization
            lambda_wp * ewp.sum() + lamda_wm * ewm.sum() + lambda_ap * eap.sum() + lamda_am * eam.sum(),
            GRB.MINIMIZE)

        model.optimize()

        if model.status is not GRB.OPTIMAL:
            raise RuntimeError("Model is not optimal, should have been. This is likely an error in the MILP.")

        variables = df[0].keys()

        graph_prior = nx.DiGraph()  # W is prior
        graph_prior.add_nodes_from(variables)
        graph_prior.add_weighted_edges_from(self._get_W_edges(ewp, wp, ewm, wm, model, variables))

        graph_slices = nx.DiGraph()  # a is temporal
        graph_slices.add_nodes_from([variable + "_lag0" for variable in variables])
        graph_slices.add_weighted_edges_from(self._get_A_edges(eap, ap, eam, am, model, variables))

        nodes = [Node(name=node_name, dynamic=True, distribution="Linear, gaussian noise",
                      model=CustomModel(input_nodes=graph_prior.predecessors(), custom_function=None)) for node_name in
                 variables]  # TODO output node should be the node, but then ... how to encode prior distribution & time slice, giving up on this meess
        dbn = DynamicBayesianNetwork(nodes=nodes)
        dbn.update_structure(graph_prior, graph_slices, models=None)

        return dbn

    def _get_A_edges(self, eap, ap, eam, am, model, variables):
        edge_list = set()

        for i in range(len(variables)):
            for j in range(len(variables)):
                if int(self.ewp[i, j].item().X) == 1 or int(self.ewm[i, j].item().X) == 1:
                    edge_list.add(variables[i] + "_lag0", variables[j] + "lag_1",
                                  eap[i, j].item().X * ap[i, j].item().X + eam[i, j].item().X * am[i, j].item().X)
        return edge_list

    def _get_W_edges(self, ewp, wp, ewm, wm, model, variables):
        # TODO bit of copy-paste (diff is in the varialbes) could it be done better
        edge_list = set()

        for i in range(len(self.variables)):
            for j in range(len(self.variables)):
                if int(self.ewp[i, j].item().X) == 1 or int(self.ewm[i, j].item().X) == 1:
                    edge_list.add(variables[i], variables[j],
                                  ewp[i, j].item().X * wp[i, j].item().X + ewm[i, j].item().X * wm[i, j].item().X)
        return edge_list
