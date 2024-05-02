from abc import ABC, abstractmethod
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator
from PGM import PGM
from pgmpy.inference import VariableElimination
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import BicScore
from pgmpy.sampling import GibbsSampling


class DynamicBayesianNetwork(PGM, ABC):
    """Abstract class for dynamic Bayesian networks (DBNs). Generally, DBNs consist of initial Bayesian network which
    encodes prior distribution, and a second network which encodes transition probabilities from one time slice to another."""

    def __init__(self, structure_and_weights=False):
        """
        Creates a new instance of the Dynamic Bayesian Network class.

        Parameters:
        structure_and_weights (bool): Indicates that the model is capable of learning the structure and weights together.
        """
        super().__init__(structure_and_weights)
