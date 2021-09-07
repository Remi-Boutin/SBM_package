from package.SBM import sbm
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score as ARI
path = '/simulations/data/Scenario_C/Beta_0.21/'
random_gen = np.random.default_rng(seed=123)

for k in range(20):
    adj = np.load(path +  'Adj_7.npy')
    cl = np.load(path + 'True_groups_7.npy')
    elbo, tau, _, count, time_list = sbm(A=adj, Q=5,
                                         max_iter=1000,
                                         tau_init=None,
                                         type_init='random',
                                         random_gen=random_gen,
                                         algo='vbem')
    print('ARI : ', ARI(cl, tau.argmax(axis=1)))

#%%
