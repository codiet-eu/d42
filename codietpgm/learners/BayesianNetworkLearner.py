from abc import ABC
from codietpgm.learners.PGMLearner import PGMLearner


class BayesianNetworkLearner(PGMLearner, ABC):
    """Abstract class for Bayesian networks (BNs)."""

    def __init__(self, structure_and_weights=False):
        """
        Creates a new instance of the Bayesian Network class.

        Parameters:
        structure_and_weights (bool): Indicates that the model is capable of learning the structure and weights together.
        """
        super().__init__(structure_and_weights)
