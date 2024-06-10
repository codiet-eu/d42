from codietpgm.models.DynamicBayesianNetwork import DynamicBayesianNetwork

from causalnex.structure import dynotears as dnt


class DyNoTearsDBN(DynamicBayesianNetwork):
    """Class for Probabilistic Graphical Models using Bayesian networks."""

    def __init__(self):
        """
        Creates a new instance of the Bayesian Network class.

        Parameters:
        structure_and_weights (bool): Indicates that the model is capable of learning the structure and weights together.
        """
        super().__init__(False)
        self._model = None

    def learn_structure(self, data, tabuIntraSlice = True, p=1):
        super().learn_structure()
        # DyNoTears requires all the data to have the same columns, otherwise fails with an error
        variables = data.get_common_dynamic_variables()
        input = data.project(variables).dynamic_variables()
        tabu_edges = [(0, u, v) for u in variables for v in variables] if tabuIntraSlice else None
        self._model = dnt.from_pandas_dynamic(time_series=input, p=p, tabu_edges=tabu_edges)


    def learn_weights(self, data):
        raise RuntimeError("CausalNex implementation does not provide weight learning in the case of DyNoTears.")