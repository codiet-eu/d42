from abc import ABC
from codietpgm.learners.DBNLearner import DBNLearner


class DBNOpt(ABC, DBNLearner):
    def __init__(self, dbn):
        self.dbn = dbn

    def optimize(self, data):
        # Placeholder: Compute cost and update structure or parameters
        cost = self.dbn.compute_likelihood(data)
        print("Cost:", cost)
        # Here you would update structures and parameters

    def update_structure_and_parameters(self):
        if self.dbn.linear:
            # Update adjacency matrices and coefficients
            pass  # Specific update logic here
