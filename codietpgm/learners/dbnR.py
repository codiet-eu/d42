from codietpgm.learners.DBNLearner import DBNLearner
from codietpgm.utils.runR import run_R
from codietpgm.utils.convert import m2graph


class dbnR(DBNLearner):
    def __init__(self, args):
        super().__init__(False)
        self._model = None
        self.args = {
            'num_time_slices': 2,
            'num_vars' : 3,
            # args for data
            'data': None, #Path to data
            'adjacency': None #Path to adjacency matrix with no header
        }
        self.args.update(args)

    def learn_structure(self, data):
        adj_structure = run_R("dbnr", self.args)
        self._model = m2graph(adj_structure)

    def learn_weights(self, data):
        raise RuntimeError("dbnr implementation provides only unweighted DBN.")
