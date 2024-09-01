from codietpgm.utils.statstools import Sampler


class Node:
    def __init__(self, name, node_type, num, distribution, model=None, value=None, observed=False, dynamic=False,
                 time_index=None, discrete_set=None):
        self._name = name  # e.g., 'X1(5)'
        self._type = node_type  # e.g., 'X1(t-1)'        
        self._num = num  # e.g., 1. if it is -1, it is not in the DBN model but an extra node
        self._distribution = distribution  # Bernoulli, multinomial, etc.
        self._label = name  # annotation, e.g., 'treatment'
        self._time_index = time_index  # 0 or -1 ...
        self._model = model
        self._value = value
        self._observed = observed  # bool
        self._dynamic = dynamic  # bool
        self._sampler = Sampler(1, distribution)  # a possibility to sample the distribution of an instance
        self._discrete_set = discrete_set  # Optional discrete set for multinomial values

    @property
    def name(self):
        """str: The name of the node. For, example 'X1(5)'. """
        return self._name

    @property
    def type(self):
        return self._type

    @property
    def num(self):
        return self._num

    @property
    def distribution(self):
        return self._distribution

    @property
    def label(self):
        return self._label

    @property
    def time_index(self):
        return self._time_index

    @property
    def model(self):
        return self._model

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        self._value = new_value

    @property
    def observed(self):
        return self._observed

    @property
    def dynamic(self):
        return self._dynamic

    @property
    def sampler(self):
        return self._sampler

    @property
    def discrete_set(self):
        return self._discrete_set




