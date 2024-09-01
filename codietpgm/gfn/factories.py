import pandas as pd
import numpy as np
import networkx as nx

from pathlib import Path
from numpy.random import default_rng
from codietpgm.gfn.bde_score import DBNBDeScore
from codietpgm.gfn.bge_score import DBNBGeScore
from codietpgm.dag_gflownet.scores.bde_score import BDeScore
from codietpgm.dag_gflownet.scores.bge_score import BGeScore
from codietpgm.dag_gflownet.scores.priors import *
from codietpgm.dag_gflownet.utils.factories import get_prior


def get_data(args, rng=default_rng()):

    adjacency = np.loadtxt(args['adjacency'], delimiter = ',')
    rows, cols = np.where(adjacency == 1)
    edges = zip(rows.tolist(), cols.tolist())
    graph = nx.DiGraph()
    graph.add_edges_from(edges)

    data = pd.read_csv(args['data'])

    score = args['score']

    return graph, data, score


def get_scorer(args, rng=default_rng()):

    # Get the data
    graph, data, score = get_data(args, rng=rng)

    # Get the prior
    prior = get_prior(args['prior'], **args['prior_kwargs'])

    num_vars = args['num_vars']
    num_time_slices = args['num_time_slices']

    # Get the scorer
    scores = {'bde': DBNBDeScore, 'bge': DBNBGeScore}
    scorer = scores[score](data, prior, num_vars, num_time_slices, **args['scorer_kwargs']) 

    return scorer, data, graph


def min_max_normalize(scores):
    min_score = np.min(scores)
    max_score = np.max(scores)
    return (scores - min_score) / (max_score - min_score)


def save_results(posterior, args=None):
    output_folder = Path(args['output_folder'])
    output_folder.mkdir(exist_ok=True)
    np.save(output_folder / 'posterior.npy', posterior)

