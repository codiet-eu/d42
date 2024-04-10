from abc import ABC, abstractmethod
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator
from PGM import PGM
from pgmpy.inference import VariableElimination
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import BicScore
from pgmpy.sampling import GibbsSampling


class BayesianNetwork(PGM):
    """Class for Probabilistic Graphical Models using Bayesian networks."""

    def __init__(self, structure_and_weights=False):
        """
        Creates a new instance of the Bayesian Network class.

        Parameters:
        structure_and_weights (bool): Indicates that the model is capable of learning the structure and weights together.
        """
        super().__init__(structure_and_weights)
        self._model = BayesianModel()

    def inference(self, query_variables):
        super().inference(query_variables)
        infer = VariableElimination(self._model)
        marginal_prob = infer.query(variables=query_variables)
        return marginal_prob

    def sample(self, num_samples):
        super().sample(num_samples)
        sampler = GibbsSampling(self._model)
        samples = sampler.sample(size=num_samples)
        return samples
    
    def learn_structure(self, data):
        super().learn_structure()
        hc = HillClimbSearch(data, scoring_method=BicScore(data))
        self._model = hc.estimate()

    def learn_weights(self, data):
        self.learn_structure()