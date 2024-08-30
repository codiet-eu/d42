import jax.numpy as jnp
import haiku as hk
import optax
import time
import matplotlib.pyplot as plt

from collections import namedtuple
from functools import partial
from jax import grad, random, vmap, jit

from codietpgm.dag_gflownet.nets.gflownet import gflownet
from codietpgm.dag_gflownet.utils.gflownet import uniform_log_policy
from codietpgm.dag_gflownet.utils.jnp_utils import batch_random_choice
from codietpgm.gfn.gfn_loss import detailed_balance_loss
from codietpgm.dag_gflownet.gflownet import DAGGFlowNet

class DBNGFlowNet_PER(DAGGFlowNet):
    """DAG-GFlowNet.
       see details in https://github.com/tristandeleu/jax-dag-gflownet/blob/master/dag_gflownet/gflownet.py
    """
    def __init__(self, model=None, delta=1., update_target_every=1000):
        super().__init__(model, delta, update_target_every)

    def loss(self, params, target_params, samples, weights=None):
        vmodel = vmap(self.model.apply, in_axes=(None, 0, 0))
        log_pi_t = vmodel(params, samples['adjacency'], samples['mask'])
        log_pi_tp1 = vmodel(
            target_params,
            samples['next_adjacency'],
            samples['next_mask']
        )

        return detailed_balance_loss(
            log_pi_t,
            log_pi_tp1,
            samples['actions'],
            samples['delta_scores'],
            samples['num_edges'],
            weights=weights, 
            delta=self.delta
        )

    # for PER
    def calculate_td_errors(self, 
                        params,
                        samples, 
                        mask, 
                        adjacency,
                        next_mask, 
                        next_adjacency, 
                        num_actions, 
                        discount_factor=0.99):

            ### compute the temporal difference (TD) errors 
            rewards = samples['delta_scores']
            actions = samples['actions']

            rewards = rewards.flatten()   
            actions = actions.flatten()  

            # compute Q-values for the current states using the online parameters
            q_values_current = self.get_q_values(params.online, mask, adjacency, num_actions)

            # compute Q-values for the next states using the target parameters
            q_values_next = self.get_q_values(params.target, next_mask, next_adjacency, num_actions)

            # calculate TD targets
            td_targets = rewards + discount_factor * jnp.amax(q_values_next, axis=1)
            # calculate TD errors
            td_errors = td_targets - q_values_current[jnp.arange(len(q_values_current)), actions]

            return td_errors

    def get_q_values(self, params, mask, adjacency, num_actions):

        masks = mask.astype(jnp.float32)
        adjacencies = adjacency.astype(jnp.float32)
        q_values = vmap(self.model.apply, in_axes=(None, 0, 0))(
            params,
            adjacencies,
            masks
        )
        return q_values
