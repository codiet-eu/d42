import unittest
import pandas as pd
import networkx as nx
from codietpgm.io.data import Data
from codietpgm.io.sample import Sample
from codietpgm.learners.MILPDBN import MILPDBN
from codietpgm.io.variableannotation import Type


class TestMILPDBN(unittest.TestCase):
    def setUp(self):
        self.data = Data([
            Sample(pd.DataFrame([[3, 4], [4, 5], [6, 7]], columns=["A", "B"]), {}),
            Sample(pd.DataFrame([[1, 2], [2, 3], [4, 5]], columns=["A", "B"]), {}),
            Sample(pd.DataFrame([[1, 2], [2, 3], [4, 5]], columns=["A", "B"]), {}),
            Sample(pd.DataFrame([[4, 5], [5, 6], [7, 8]], columns=["A", "B"]), {}),
            Sample(pd.DataFrame([[3, 4], [4, 5], [6, 7]], columns=["A", "B"]), {}),
            Sample(pd.DataFrame([[2, 3], [3, 4], [5, 6]], columns=["A", "B"]), {}),
            Sample(pd.DataFrame([[5, 6], [6, 7], [8, 9]], columns=["A", "B"]), {}),
            Sample(pd.DataFrame([[10, 11], [11, 12], [13, 14]], columns=["A", "B"]), {}),
            Sample(pd.DataFrame([[4, 5], [5, 6], [7, 8]], columns=["A", "B"]), {}),
            Sample(pd.DataFrame([[0, 1], [1, 2], [3, 4]], columns=["A", "B"]), {}),
        ], variables_annotation={"A": {Type.CONTINUOUS}, "B": {Type.CONTINUOUS}})

    def test_learn_weights(self):
        model = MILPDBN()
        dbn = model.learn_weights(self.data)
        self.assertEqual(nx.to_numpy_array(dbn.get_graph_t_minus_one(), nodelist=["A", "B"]), {("A", "B")})


if __name__ == '__main__':
    unittest.main()
