import unittest
import pandas as pd
from codietpgm.io.data import Data
from codietpgm.io.sample import Sample
from codietpgm.learners.ILPBN import ILPBN
from codietpgm.io.variableannotation import Type


class TestILPBN(unittest.TestCase):
    def setUp(self):
        self.data = Data([
            Sample(pd.DataFrame(), {"A": 3, "B": 4}),
            Sample(pd.DataFrame(), {"A": 1, "B": 2}),
            Sample(pd.DataFrame(), {"A": 1, "B": 2}),
            Sample(pd.DataFrame(), {"A": 4, "B": 5}),
            Sample(pd.DataFrame(), {"A": 3, "B": 4}),
            Sample(pd.DataFrame(), {"A": 2, "B": 3}),
            Sample(pd.DataFrame(), {"A": 5, "B": 6}),
            Sample(pd.DataFrame(), {"A": 10, "B": 11}),
            Sample(pd.DataFrame(), {"A": 4, "B": 5}),
            Sample(pd.DataFrame(), {"A": 0, "B": 1}),
        ], variables_annotation={"A": {Type.CONTINUOUS}, "B": {Type.CONTINUOUS}})

    def test_learn_weights(self):
        model = ILPBN()
        model.learn_weights(self.data)
        self.assertEqual(model.get_edges(), {("A", "B")})


if __name__ == '__main__':
    unittest.main()
