import unittest
import pandas as pd
import os
import time
import numpy as np
from codietpgm.io.data import Data
from codietpgm.io.sample import Sample
from codietpgm.learners.MILPDBN import MILPDBN
from codietpgm.io.variableannotation import Type


class TestMILPDBN(unittest.TestCase):
    def setUp(self):
        pass

    def test_learn_weights(self):
        # n = # variables, ts = # time steps, N = #samples
        for n, ts, N in (3, 10, 5), (3, 10, 30), (5, 20, 10), (5, 50, 50), (10, 40, 20), (10, 200, 100), (20, 50, 40), (20, 400, 400), (30, 100, 60), (30, 500, 600):
            datasetname = str(n) + "n_" + str(ts) + "ts_" + str(N) + "N"
            print(datasetname)
            start = time.time()
            ts = int(0.7*ts)
            df = pd.read_csv("data" + os.sep + datasetname + os.sep + "train_data_" + datasetname + ".csv", index_col = None)
            samples = []
            variables = ["X"+str(i) for i in range(1, n+1)]
            for i in range(N): #TODO should be nicer, looks like c++/Java
                data = np.zeros((ts, n))
                for t in range(ts):
                    for var in range(n):
                        data[t, var] = df.iloc[i]["X"+str(var+1)+"_t_"+str(t)]
                samples.append(Sample(pd.DataFrame(data, columns=variables), {}))

            self.data = Data(samples, variables_annotation={v: {Type.CONTINUOUS} for v in variables})

            model = MILPDBN()
            model.learn_weights(self.data, lambda_wp=0.05, lamda_wm=0.05, lambda_ap=0.05, lamda_am=0.05, b_w=0.1, b_a=0.1)

            matrix = model.get_A_adajcency()
            print(matrix)

            repeated_matrix = np.zeros((n*ts, n*ts))
            for t in range(ts - 1):
                start_row = 0 + t * n
                start_col = (t+1) * n
                repeated_matrix[start_row:(start_row + matrix.shape[0]), start_col:(start_col + matrix.shape[1])] = matrix

            matrix = model.get_W_adajcency()
            print(matrix)
            for t in range(ts):
                start_row_col = t * n
                repeated_matrix[start_row_col:(start_row_col + matrix.shape[0]), start_row_col:(start_row_col + matrix.shape[1])] = matrix

            print("Time : " + str(time.time() - start))

            df = pd.DataFrame(repeated_matrix)
            df = df.astype(bool).astype(int)
            df.index = range(len(df))
            df.columns = range(len(df.columns))
            df.to_csv("MILPDBN_adj_"+datasetname+".csv", index_label="")


if __name__ == '__main__':
    unittest.main()
