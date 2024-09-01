import unittest
import os
import pandas as pd
import numpy as np
from sklearn import metrics


class TestMILPDBN(unittest.TestCase):
    def setUp(self):
        pass

    def test_calc_stats(self):
        for n, ts, N in (3, 10, 5), (3, 10, 30), (5, 20, 10), (5, 50, 50), (10, 40, 20), (10, 200, 100), (20, 50, 40), (20, 400, 400), (30, 100, 60), (30, 500, 600):
            datasetname = str(n) + "n_" + str(ts) + "ts_" + str(N) + "N"
            print(datasetname)

            if os.path.isfile("MILPDBN_adj_"+datasetname+".csv"):
                num_vars = n
                num_time_slices = int(0.7*ts)

                #ground_truth = pd.read_csv(
                #    "data" + os.sep + datasetname + os.sep + "train_adj_" + datasetname + ".csv",
                #index_col=0).to_numpy()
                ground_truth = np.loadtxt(
                    "data" + os.sep + datasetname + os.sep + "train_adj_" + datasetname + ".csv",
                    delimiter=',', usecols=range(1, num_vars * num_time_slices + 1), skiprows=1)
                ground_truth = ground_truth.astype(int)
                print(ground_truth.shape)
                print(pd.DataFrame(ground_truth))

                adj = pd.read_csv("MILPDBN_adj_"+datasetname+".csv", index_col=0)
                print(adj.shape)
                print(adj)
                adj_arr = np.array(adj)
                adj_arr = adj_arr.astype(int)
                adj_mat_prior = adj_arr[:num_vars, :num_vars]
                adj_mat_transition = adj_arr[:num_vars, num_vars:2 * num_vars]
                adj_mat_final = np.zeros((num_vars * num_time_slices, num_vars * num_time_slices))

                for k in range(num_time_slices):
                    # filling the prior matrix
                    adj_mat_final[num_vars * k:num_vars * (k + 1), num_vars * k:num_vars * (k + 1)] = adj_mat_prior
                    # the transition
                    if k < num_time_slices - 1:
                        adj_mat_final[num_vars * k:num_vars * (k + 1),
                        num_vars * (k + 1):num_vars * (k + 2)] = adj_mat_transition

                bnstruct_flat = adj_mat_final.reshape(-1)
                gt_flat = ground_truth.reshape(-1)
                fpr, tpr, _ = metrics.roc_curve(gt_flat, bnstruct_flat)
                roc_auc = metrics.auc(fpr, tpr)
                precision, recall, _ = metrics.precision_recall_curve(gt_flat, bnstruct_flat)
                prc_auc = metrics.auc(recall, precision)
                ave_prec = metrics.average_precision_score(gt_flat, bnstruct_flat)
                print("ROC AUC")
                print(roc_auc)
                print("PRC AUC")
                print(prc_auc)

                def expected_shd_from_adj(adjacency, ground_truth):
                    """Compute the Expected Structural Hamming Distance for directed graphs.

                    Args:
                        adjacency (np.ndarray): A 2D numpy array the adjacency matrix (nodes, nodes).
                        ground_truth (np.ndarray): A 2D numpy array of the ground truth adjacency matrix (nodes, nodes).

                    Returns:
                        float: The expected Structural Hamming Distance.
                    """
                    diff = np.abs(adjacency - ground_truth)

                    # Sum the differences to get the SHD
                    shd = np.sum(diff, axis=None)

                    return shd

                print("SHD")
                print(expected_shd_from_adj(adj_mat_final, ground_truth))



if __name__ == '__main__':
    unittest.main()
