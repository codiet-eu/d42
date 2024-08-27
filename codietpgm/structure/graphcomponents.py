from codietpgm.utils.statstools import Sampler

class Node:
    def __init__(self, name, node_type, num, distribution, model=None, value=None, observed=False, dynamic=False, time_index=None, \
    discrete_set=None):
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
        self._sampler = Sampler(1,distribution)  # a possibility to sample the distribution of an instance
        self._discrete_set = discrete_set  # Optional discrete set for multinomial values

    #TODO use properties



class Edge:
    def __init__(self, from_node, to_node):
        self._from_node = from_node
        self._to_node = to_node


