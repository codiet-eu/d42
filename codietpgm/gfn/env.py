import numpy as np
import gym
import bisect

from multiprocessing import get_context
from copy import deepcopy
from gym.spaces import Dict, Box, Discrete

from dag_gflownet.env import GFlowNetDAGEnv


class GFlowNetDBNEnv(GFlowNetDAGEnv):

    def __init__(
            self,
            num_envs,
            scorer,
            num_vars, 
            max_parents=None,
            num_workers=4,
            context=None,
            cache_max_size=10_000
            
        ):
        """
            GFlowNet environment for learning a distribution over    
            the transition network
        """

        self.num_vars = num_vars
        self.num_variables = self.num_vars * 2
        shape = (self.num_variables, self.num_variables)
        max_edges = self.num_vars ** 2

        observation_space = Dict({
            'adjacency': Box(low=0., high=1., shape=shape, dtype=np.int_),
            'mask': Box(low=0., high=1., shape=shape, dtype=np.int_),
            'num_edges': Discrete(max_edges),
            'score': Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float_),
            'order': Box(low=-1, high=max_edges, shape=shape, dtype=np.int_)
        })
        action_space = Discrete(self.num_variables ** 2 + 1)

        super().__init__(num_envs, scorer, max_parents, num_workers, context, cache_max_size)


    def reset(self):

        shape = (self.num_envs, self.num_variables, self.num_variables)  

        self._closure_T_intra_slice = np.eye(self.num_vars, dtype=np.bool_) 
        closure_T = np.ones((self.num_variables, self.num_variables), dtype=np.bool_)
        closure_T = np.tril(closure_T)

        for i in range(0, self.num_variables, self.num_vars): 
            closure_T[i:i+self.num_vars, i:i+self.num_vars] = self._closure_T_intra_slice
                
        closure_T[self.num_vars:self.num_variables, :self.num_variables] = 1
        closure_T[:self.num_vars, 2*self.num_vars:self.num_variables] = 1
        self._closure_T = np.tile(closure_T, (self.num_envs, 1, 1))
                
        self._state = {
            'adjacency': np.zeros(shape, dtype=np.int_), 
            'mask': 1 - self._closure_T,
            'num_edges': np.zeros((self.num_envs,), dtype=np.int_),
            'score': np.zeros((self.num_envs,), dtype=np.float_),
            'order': np.full(shape, -1, dtype=np.int_)
            }
        return deepcopy(self._state)

    def step(self, actions):

        sources, targets = divmod(actions, self.num_variables)
        keys, local_cache, data = self.local_scores_async(sources, targets)

        dones = (sources == self.num_variables)
        sources, targets = sources[~dones], targets[~dones]
        # Make sure that all the actions are valid
        if not np.all(self._state['mask'][~dones, sources, targets]):
            raise ValueError('Some actions are invalid: either the edge to be '
                             'added is already in the DAG, or adding this edge '
                             'would lead to a cycle.')

        # Update the adjacency matrices
        self._state['adjacency'][~dones, sources, targets] = 1
        self._state['adjacency'][dones] = 0

        # Update transitive closure of transpose
        source_rows = np.expand_dims(self._closure_T[~dones, sources, :], axis=1)
        target_cols = np.expand_dims(self._closure_T[~dones, :, targets], axis=2)
        self._closure_T[~dones] |= np.logical_and(source_rows, target_cols)  # Outer product

        # Intra/inter slice dependencides are learned in one slice/transition and copied to the others
        self._closure_T_intra_slice = np.eye(self.num_vars, dtype=np.bool_) # identity matrix for intra slice
        temp = np.ones((self.num_variables, self.num_variables), dtype=np.bool_)
        temp[:self.num_vars, :self.num_vars] = self._closure_T_intra_slice
        temp[:self.num_vars, self.num_vars:2*self.num_vars] = 0
        #only keep upper part 
        temp[self.num_vars:self.num_variables, :self.num_variables] = 1
     
        self._closure_T[dones] = temp

        # Update the masks
        self._state['mask'] = 1 - (self._state['adjacency'] + self._closure_T)
        self._state['mask'][self._state['mask']<0] = 0 # in case of negative values in lower part 
        self._state['mask'][:,:self.num_vars,:self.num_vars] = 0 #we only focus on learning the transition network's edges 

        # Update the masks (maximum number of parents)
        num_parents = np.sum(self._state['adjacency'], axis=1, keepdims=True)
        self._state['mask'] *= (num_parents < self.max_parents)

        # Update the order
        self._state['order'][~dones, sources, targets] = self._state['num_edges'][~dones]
        self._state['order'][dones] = -1

        self._shift_order = np.full((self.num_envs, self.num_variables, self.num_variables), fill_value = -1, dtype=np.int_) 
        for i in range(self.num_vars, self.num_variables, self.num_vars):
            self._shift_order[:,i:,i:] = self._state['order'][:,:-i,:-i]
        
        self._shift_order[:,:self.num_vars,:] = self._state['order'][:,:self.num_vars,:]
        self._state['order'] = self._shift_order

        # Update the number of edges
        self._state['num_edges'] += 1 
        self._state['num_edges'][dones] = 0
        
        # Get the difference of log-rewards. The environment returns the
        # delta-scores log R(G_t) - log R(G_{t-1}), corresponding to a local
        # change in the scores. This quantity can be used directly in the loss
        # function derived from the trajectory detailed loss.
        delta_scores = self.local_scores_wait(keys, local_cache, data)

        # Update the scores. The scores returned by the environments are scores
        # relative to the empty graph: score(G) - score(G_0).
        self._state['score'] += delta_scores
        self._state['score'][dones] = 0

        return (deepcopy(self._state), delta_scores, dones, {})

