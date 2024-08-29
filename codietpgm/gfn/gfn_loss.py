import numpy as np
import jax.numpy as jnp
import optax

from tqdm.auto import trange
from jax import nn, lax, random

def detailed_balance_loss(
        log_pi_t,
        log_pi_tp1,
        actions,
        delta_scores,
        num_edges,
        weights=None,
        delta=1.
    ):
    r"""Detailed balance loss.

    see details in https://github.com/tristandeleu/jax-dag-gflownet/blob/master/dag_gflownet/utils/gflownet.py
    
    here we use the weighted error from the Prioritized Replay Experience to update the loss function 

    Returns
    -------
    loss : jnp.DeviceArray
        The detailed balance loss averaged over a batch of samples.

    logs : dict
        Additional information for logging purposes.
    """
    # Compute the forward log-probabilities
    log_pF = jnp.take_along_axis(log_pi_t, actions, axis=-1)

    # Compute the backward log-probabilities
    log_pB = -jnp.log1p(num_edges)

    error = (jnp.squeeze(delta_scores + log_pB - log_pF, axis=-1)
        + log_pi_t[:, -1] - lax.stop_gradient(log_pi_tp1[:, -1]))

    # if replay buffer is PER 
    if weights: 
        weighted_error = error * weights
        error = weighted_error 

    # loss with the weigthed error 
    loss = jnp.mean(optax.huber_loss(error, delta=delta))

    logs = {
        'error': error,
        'loss': loss,
    }
    return (loss, logs)


