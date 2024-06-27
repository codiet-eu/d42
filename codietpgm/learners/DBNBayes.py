from abc import ABC
from codietpgm.learners.DBNLearner import DBNLearner


class DBNBayes(ABC, DBNLearner):
    def __init__(self, dbn):
        self.dbn = dbn

    def update_sampler_custom(self, node_name, custom_function):
        if node_name in self.dbn.nodes:
            model = self.dbn.nodes[node_name].model
            model.sampler.update_custom_sampling(custom_function)
        else:
            raise ValueError("Node not found in DBN")

    def update_hyperparameters(self, node_name, hyperparameters):
        if node_name in self.dbn.nodes:
            model = self.dbn.nodes[node_name].model
            model.update_sampler(hyperparameters)
        else:
            raise ValueError("Node not found in DBN")
