import numpy as np
import math

from numpy.random import default_rng
from codietpgm.dag_gflownet.utils.replay_buffer import ReplayBuffer

from codietpgm.gfn.factories import min_max_normalize


class PrioritizedReplayBuffer(ReplayBuffer):

    def __init__(self,
                 capacity,
                 num_variables,
                 alpha=0.6,
                 beta=0.4,
                 epsilon_rb=1e-6,
                 decay_factor=0.90):

        super().__init__(capacity, num_variables)

        # Adding elements to prioritize the choice of actions based on some constraint algorithm 
        self.priority = np.full(capacity,
                                0.5)  # value should be in the range of the priorities that we could be getting to keep the exploration
        self.alpha = alpha  # prioritization exponent
        self.max_priority = 1.0  # priority value
        self.beta = beta  # importance sampling exponent
        self.epsilon_rb = epsilon_rb  # small positive constant to ensure all transitions have a non-zero probability of being sampled
        self.decay_factor = decay_factor  # decay factor for priorities, again for exploration

    def add(
            self,
            observations,
            actions,
            is_exploration,
            next_observations,
            delta_scores,
            dones,
            priority_actions=None,
            prev_indices=None
    ):

        indices = np.full((dones.shape[0],), -1, dtype=np.int_)
        if np.all(dones):
            return indices

        num_samples = np.sum(~dones)
        add_idx = np.arange(self._index, self._index + num_samples) % self.capacity
        self._is_full |= (self._index + num_samples >= self.capacity)
        self._index = (self._index + num_samples) % self.capacity
        indices[~dones] = add_idx

        data = {

            'adjacency': self.encode(observations['adjacency'][~dones]),
            'num_edges': observations['num_edges'][~dones],
            'actions': actions[~dones],
            'delta_scores': delta_scores[~dones],
            'mask': self.encode(observations['mask'][~dones]),
            'next_adjacency': self.encode(next_observations['adjacency'][~dones]),
            'next_mask': self.encode(next_observations['mask'][~dones]),

            # Extra keys for monitoring
            'is_exploration': is_exploration[~dones],
            'scores': observations['score'][~dones],

        }

        for name in data:
            shape = self._replay.dtype[name].shape
            self._replay[name][add_idx] = np.asarray(data[name].reshape(-1, *shape))

        if prev_indices is not None:
            self._prev[add_idx] = prev_indices[~dones]

        if priority_actions.size > 0:
            # priority actions taken from a constraint based method : didn't do tests with this because constraint-based algorithms take time to run 
            priority_indices = add_idx[np.isin(actions[~dones], priority_actions)]
            self.priority[priority_indices] = self.max_priority

        return indices

    def sample(self, batch_size, rng=default_rng()):

        probabilities = self.priority  # + 1e-6
        probabilities[np.isnan(probabilities)] = 0
        probabilities = np.resize(probabilities, len(self))

        probabilities /= np.sum(probabilities)

        indices = rng.choice(len(self), size=batch_size, replace=True, p=probabilities)
        samples = self._replay[indices]
        weights = (1 / (len(self) * probabilities[indices])) ** self.beta
        weights = weights / weights.max()

        samples['delta_scores'] = min_max_normalize(samples['delta_scores'])

        # Convert structured array into dictionary
        return {
            'indices': indices,
            'adjacency': self.decode(samples['adjacency']),
            'num_edges': samples['num_edges'],
            'actions': samples['actions'],
            'delta_scores': samples['delta_scores'],
            'mask': self.decode(samples['mask']),
            'next_adjacency': self.decode(samples['next_adjacency']),
            'next_mask': self.decode(samples['next_mask']),
            'weights': weights,
        }

    # for PER
    def update_priorities(self, indices, td_errors):

        indices = np.asarray(indices)
        td_errors = np.asarray(td_errors)

        # Compute priorities for each transition at specified indices
        priorities = np.abs(td_errors) ** self.alpha + self.epsilon_rb
        self.priority[indices] = priorities


class ReplayBufferSamplesnormalized(ReplayBuffer):

    def __init__(self,
                 capacity,
                 num_variables):
        super().__init__(capacity, num_variables)

    def sample(self, batch_size, rng=default_rng()):
        indices = rng.choice(len(self), size=batch_size, replace=False)
        samples = self._replay[indices]

        # normalize the delta scores 
        samples['delta_scores'] = min_max_normalize(samples['delta_scores'])

        # Convert structured array into dictionary
        return {
            'adjacency': self.decode(samples['adjacency']),
            'num_edges': samples['num_edges'],
            'actions': samples['actions'],
            'delta_scores': samples['delta_scores'],
            'mask': self.decode(samples['mask']),
            'next_adjacency': self.decode(samples['next_adjacency']),
            'next_mask': self.decode(samples['next_mask'])
        }
