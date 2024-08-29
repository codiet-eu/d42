import jax.numpy as jnp
import numpy as np
import optax
import networkx as nx
import pickle
import pandas as pd
import jax
import time
import matplotlib.pyplot as plt

from tqdm import trange
from numpy.random import default_rng
from pathlib import Path
from base import *

from gfn.env import GFlowNetDBNEnv
from gfn.factories import get_scorer, save_results
from dag_gflownet.gflownet import DAGGFlowNet
from gfn.gflownet import DBNGFlowNet_PER
from dag_gflownet.utils.replay_buffer import ReplayBuffer
from gfn.replay_buffer import PrioritizedReplayBuffer, ReplayBufferSamplesnormalized
from dag_gflownet.utils.gflownet import posterior_estimate
from dag_gflownet.utils import io

from DBNLearner import DBNLearner

from codietpgm.utils.convert import m2graph

##########################################################
#### Learning the transition network 
##########################################################

class GFN(DBNLearner):
    def __init__(self, args):

        self.default_args = {
            'num_envs': 8,
            'num_vars': 2,
            'num_time_slices': 2,
            'scorer_kwargs': {},
            'prior':'uniform',
            'prior_kwargs': {},
            'lr': 1e-5,
            'delta': 1.0,
            'batch_size': 32,
            'num_iterations': 100_000,
            'replay_capacity': 100_000,
            'alpha': 0.6, # for PER
            'beta': 0.4,  # for PER
            'epsilon_rb': 1e-6,  # for PER
            'decay_factor' : 0.9,  # for PER
            'prefill': 1000,
            'min_exploration': 0.1,
            'update_epsilon_every': 100,
            'discount_factor': 0.99,
            'num_samples_posterior': 1000,
            'update_target_every': 1000,
            'seed': 0,
            'num_workers': 4,
            'mp_context': 'spawn',
            'output_folder': 'output',
            'percentile': 99, #threshold to keep highest probabilities in the posterior matrix 
            # args for data
            'data': None, #Path to data
            'adjacency': None, #Path to adjacency matrix with no header
            'score': 'bge',
            'graph': None,
            'PER': False
        }
        self.default_args.update(args)
        

    def train(self, args=None):
        args = args or self.default_args  # use default args if None provided
        rng = default_rng(args['seed'])
        key = jax.random.PRNGKey(args['seed'])
        key, subkey = jax.random.split(key)
        
        scorer, data, graph = get_scorer(args, rng=rng)

        env = GFlowNetDBNEnv(
            num_envs=args['num_envs'],
            scorer=scorer,
            num_vars=args['num_vars']
        )

        transition_num_vars = args['num_vars']*2
        num_variables = env.num_variables
        PER = args['PER']
        
        if PER: 
            replay = PrioritizedReplayBuffer(
                args['replay_capacity'],
                transition_num_vars,
                args['alpha'],
                args['beta'],
                args['epsilon_rb'],
                args['decay_factor'],
                )
            gflownet = DBNGFlowNet_PER(
                delta=args['delta'],
                update_target_every=args['update_target_every']
            )
             
        else: 
            replay = ReplayBufferSamplesnormalized(
            args['replay_capacity'],
            transition_num_vars
            )
            gflownet = DAGGFlowNet(
                delta=args['delta'],
                update_target_every=args['update_target_every']
            )

        priority_actions = None
        discount_factor = args['discount_factor']

        optimizer = optax.adam(args['lr'])
        params, state = gflownet.init(
            subkey,
            optimizer,
            replay.dummy['adjacency'],
            replay.dummy['mask']

        )
        exploration_schedule = jax.jit(optax.linear_schedule(
            init_value=jnp.array(0.),
            end_value=jnp.array(1. - args['min_exploration']),
            transition_steps=args['num_iterations'] // 2,
            transition_begin=args['prefill'],
        ))

        indices = None
        observations = env.reset()
        num_actions = env.action_space
        history = {}

        with trange(args['prefill'] + args['num_iterations'], desc='Training') as pbar:
            for iteration in pbar:
                epsilon = exploration_schedule(iteration)
                actions, key, logs = gflownet.act(params.online, key, observations, epsilon)

                next_observations, delta_scores, dones, _ = env.step(np.asarray(actions))
                if PER: 
                    indices = replay.add(
                        observations,
                        actions,
                        logs['is_exploration'],
                        next_observations,
                        delta_scores,
                        dones,
                        priority_actions, 
                        prev_indices=indices
                        )
                else: 
                    indices = replay.add(
                        observations,
                        actions,
                        logs['is_exploration'],
                        next_observations,
                        delta_scores,
                        dones,
                        prev_indices=indices
                    )
                observations = next_observations

                if iteration >= args['prefill']:
                    samples = replay.sample(batch_size=args['batch_size'], rng=rng)
                    if PER: 

                        indices_sampled = samples['indices']
                        mask_sampled = samples['mask']
                        adjacency_sampled = samples['adjacency']
                        next_mask_sampled = samples['next_mask']
                        next_adjacency_sampled = samples['next_adjacency']
                        weights = samples['weights']
                        td_errors = gflownet.calculate_td_errors(params, samples, mask_sampled, adjacency_sampled, next_mask_sampled, next_adjacency_sampled, num_actions, discount_factor) 
                        replay.update_priorities(indices_sampled, td_errors)

                    params, state, logs = gflownet.step(params, state, samples)

                    pbar.set_postfix(loss=f"{logs['loss']:.2f}", epsilon=f"{epsilon:.2f}")
                    history[iteration] = logs['loss']

        posterior, _ = posterior_estimate(
            gflownet,
            params.online,
            env,
            key,
            num_samples=args['num_samples_posterior'],
            desc='Sampling from posterior'
        )

        # Save the transition matrix
        save_results(posterior, args)

        # Apply a threshold to the posterior and use m2graph 
        percentile = args['percentile']
        p_edge = np.mean(posterior, axis=0) 
        percentile_threshold = np.percentile(p_edge, percentile)
        filtered_posterior_trans = np.where(p_edge >= percentile_threshold if percentile_threshold == 1 else p_edge > percentile_threshold, p_edge, 0)

        posterior_graph = m2graph(filtered_posterior_trans)
        

