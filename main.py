from sbm_vbem.SBM import sbm
import numpy as np

from sklearn.metrics.cluster import adjusted_rand_score as ARI
from src import utils

# Matrix of the probability connection between two clusters
probability_mat = np.array([ [0.25,0.05,0.05],
                             [0.05,0.25,0.05],
                             [0.05,0.05,0.25]])

# Creation of the graph
(adj, adj_tilde, graph, pos, node_groups) = utils.generation(probability_mat, seed=None, n_groups=3)

# Estimation of the model (watch-out : it is difficult to recover from the random init !)
elbo, tau, _, count, time_list = sbm(A=adj,
                                     Q=3,
                                     max_iter=200,
                                     tau_init=None,
                                     type_init='random',
                                     seed=None,
                                     tol=1e-6,
                                     algo='vbem')

print('ARI : ', ARI(node_groups, tau.argmax(axis=1)))