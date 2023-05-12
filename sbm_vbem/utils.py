import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score as ARI
import networkx as nx


def load_scenario(scenario='A', Q=5, beta=0.3, eps=0.01):
    import numpy as np

    if scenario == 'A':
        # Scenario 1
        probability_connection = beta * np.eye(Q) + eps * (1 - np.eye(Q))
        alpha = np.ones(Q) / Q

    elif scenario == 'B':
        # Scenario 2
        probability_connection = eps * np.ones((Q, Q))
        np.fill_diagonal(probability_connection, beta)
        probability_connection[0, :] = beta
        probability_connection[:, 0] = beta
        alpha = np.ones(Q) / Q

    elif scenario == 'C':
        # Scenario 3

        probability_connection = eps * np.ones((Q, Q))
        np.fill_diagonal(probability_connection, beta)
        probability_connection[0, :] = beta
        probability_connection[:, 0] = beta
        alpha = pow(0.7, np.arange(1, Q + 1))
        alpha /= alpha.sum()
    return probability_connection, alpha


def generation(probability_connection, random_gen = None, n_groups=5, n_nodes=100, seed=2021, alpha=None, verbose=False):
    from datetime import datetime
    import numpy as np
    from scipy.sparse import csr_matrix
    import networkx as nx

    """
    Generation of a graph following the Stochastic Block Model
    with the init params alpha = (1/Q, ..., 1/Q). The transition matrix is simplified
    and is of the form P = proba_intra * Id +  (1-proba_intra)(1_N -Id)
    where 1_N is the matrix filled with ones.

    To add : undirected case (and the speed up that comes with it)
    """

    if random_gen is None:
        random_gen = np.random.default_rng(seed=seed)

    t0 = datetime.now()
    probability_connection = probability_connection
    if alpha is None:
        alpha = np.ones(n_groups) / n_groups  # True alpha (same proba for each group)

    Z = random_gen.multinomial(1, alpha, size=n_nodes)  # Generation of the group of each node

    if verbose:
        print('True Propotion of groups : {}.'.format( Z.sum(axis=0)/n_nodes))

    assert Z.sum() == n_nodes, "Sum of Z is not equal to number of nodes"

    #  Creation of dictionaries to go from node to group and group to node
    node_indices, node_groups = np.nonzero(Z)
    group2nodes = {k: np.where(node_groups == k)[0] for k in range(n_groups)}
    nodes2group = {node_indices[k]: node_groups[k] for k in range(len(node_indices))}

    # Creation of the adj matrix. Init to the value 2 to check all the indices are overwritten
    A = np.ones((n_nodes, n_nodes)) * 2

    #### CHANGE TO LOOP OVER Q AND R

    # Generation of the adj matrix
    for q in range(n_groups):
        for r in range(n_groups):
            proba_connexion = probability_connection[q,r]
            A[np.ix_(group2nodes[q],group2nodes[r])] = random_gen.binomial(n=1, p=proba_connexion,
                                                                                   size=(len(group2nodes[q]),len(group2nodes[r])))

    np.fill_diagonal(A, 0)

    assert (A == 2).sum() == 0, "There is an error, some values in A are not " +\
                                "overwritten during the creation of the graph "

    assert A.diagonal().sum() == 0, 'The graph contains loops'

    adj_flatten = A.flatten()
    ### Adjacency to sparse with zero on the diagonal
    adj = csr_matrix(A)
    adj_coo = adj.tocoo()
    indices_ones = [adj_coo.row, adj_coo.col]

    one_minus_adj = np.ones_like(A)-A
    np.fill_diagonal(one_minus_adj, 0)

    adj = adj.toarray()
    A_tilde = np.ones((n_nodes, n_nodes)) - np.eye(n_nodes) - adj

    # Creation of sparse (1 - X) : useful for the E step, with zero on the diagonal
    ones_out_diag = np.ones_like(A)
    np.fill_diagonal(ones_out_diag,0)

    # Networkx graph creation
    graph = nx.from_numpy_matrix(A)
    generation_time = datetime.now() - t0

    if verbose :
        print('Time to generate the SBM model with {} nodes : {} seconds.'.format(n_nodes, generation_time))

    # Save plot parameters
    pos = nx.spring_layout(graph, k=0.05, iterations=30, seed=seed)
    node_groups = list(nodes2group.values())

    return (A, A_tilde, graph, pos, node_groups)


def init(A, Q, eps, random_gen, type_init='kmeans'):
    import os
    os.environ["OMP_NUM_THREADS"] = '1'

    import numpy as np
    from sklearn.cluster import KMeans

    M = A.shape[0]

    if type_init == 'kmeans':
        km = KMeans(n_clusters=Q, n_init=20, max_iter=15, tol=1e-5)
        mask = np.array( 1 - np.eye(M), dtype=bool)
        km.fit( np.minimum( A + A.T, 1))
        # One hot encoding (but prevent exactly one and zero values for stability reasons)
        tau = eps * np.ones( (M, Q) )
        tau[np.arange(km.labels_.size), km.labels_] = 1 - (Q-1) * eps
    
    elif type_init == 'random':
        # Uniformly select a category for each node
        tau = random_gen.choice(Q, size=M)
        tau = one_hot(tau, Q)
        tau += eps
        tau = tau / tau.sum(axis=1,keepdims=True)

    return tau


def plot(graph,
         pos,
         groups,
         title=None,
         node_size=30,
         edge_color='grey',
         alpha=0.8,
         width=0.7,
         compare_results=True):
    import matplotlib.pyplot as plt
    from networkx import draw

    """
    Plot either the results or the generated graph
    alpha = node and edge transparency
    width = width of edges
    """
    if compare_results :
        plt.figure()
        ax = plt.gca()

        if title is None:
            ax.set_title('Results', fontsize=15)
        else:
            ax.set_title(title, fontsize=15)

        draw(graph, width=width, pos=pos, node_size=node_size, alpha=alpha,edge_color=edge_color,
                node_color=[group for group in groups])
        plt.show()
        

def get_real_params(Pi, M, Q, true_groups):
    import numpy as np
    _, gamma = np.unique(true_groups, return_counts=True)
    
    kappa = np.zeros( (Q,Q,2))
    kappa[:,:,0] = Pi * M * (M-1)
    kappa[:,:,1] = (1-Pi) * M * (M-1)
    
    return kappa, gamma


def graph_database_creation(random_gen, path, n_repet=20, seed=2021):
    import numpy as np

    for scenario in ['A', 'B', 'C']:
        
        if not os.path.exists(os.path.join(path + 'Scenario_'+ scenario)):
            os.mkdir(os.path.join(path + 'Scenario_'+ scenario))
        
        for beta in np.arange(0.01, 0.42, 0.02):
            beta_display = np.round(beta, 2)

            if not os.path.exists(os.path.join(path + 'Scenario_'+ scenario, 'Beta_' + str(beta_display))):
                os.mkdir(os.path.join(path + 'Scenario_'+ scenario, 'Beta_' + str(beta_display)))
                
            load_scenario_args = {
                'Q' : 5,
                'beta' : beta,
                'eps' : 0.01,
                'scenario' : scenario
            }
            Pi, alpha = load_scenario(**load_scenario_args)

            generation_args = {
                'probability_connection': Pi,
                'alpha' : alpha,
                'random_gen' : random_gen,
                'n_groups' : 5,
                'n_nodes' : 100,
                'seed' : seed,
                'verbose' : False
            }
            for n in range(n_repet):
                (A,A_tilde, graph, pos, node_groups) = generation(**generation_args)
                np.save(os.path.join(path + 'Scenario_'+ scenario, 'Beta_' + str(beta_display), 'Adj_'+ str(n) ), A)
                np.save(os.path.join(path + 'Scenario_'+ scenario, 'Beta_' + str(beta_display), 'True_groups_'+ str(n) ), node_groups)



def plot_graph_results(graph, tau, tau_init, node_groups, elbo, pos):
    plt.figure(figsize=(18, 9))

    plt.subplot(221)
    nx.draw_networkx_edges(graph, alpha=0.5, edge_color='grey', pos=pos)
    nx.draw_networkx_nodes(graph, node_size=50, alpha=1, node_color=[i for i in node_groups], pos=pos)
    plt.title('True labels')

    plt.subplot(222)
    nx.draw_networkx_edges(graph, alpha=0.5, edge_color='grey', pos=pos)
    nx.draw_networkx_nodes(graph, node_size=50, alpha=1, node_color=[i for i in tau_init.argmax(axis=1)], pos=pos)
    plt.title('Init')

    plt.subplot(223)
    nx.draw_networkx_edges(graph, alpha=0.5, edge_color='grey', pos=pos)
    nx.draw_networkx_nodes(graph, node_size=50, alpha=1, node_color=[i for i in tau.argmax(axis=1)], pos=pos)
    plt.title('After training')

    plt.subplot(224)
    plt.plot(elbo[1:])
    plt.title('ELBO')

    print('ARI init :', ARI(node_groups, tau_init.argmax(axis=-1)))
    print('ARI after :', ARI(node_groups, tau.argmax(axis=-1)))


def plot_ari(ari, labels, path=None, title=None, betas=np.arange(0.01, 0.42, 0.02), plot_uncertainty=True, scenarii=['A', 'B', 'C'], savefig=False):
    import numpy as np
    fig, ax = plt.subplots(len(scenarii), 1, figsize=(8, 17))
    fig.patch.set_facecolor('white')

    for k, sc in enumerate(scenarii):
        for d in range(ari.shape[0]):
            ##### NCG #######
            mean = ari[d, k, :].mean(axis=-1)
            std = ari[d, k, :].std(axis=-1)
            ax[k].plot(betas, mean, label=labels[d])
            if plot_uncertainty:
                ax[k].fill_between(betas, mean - std, mean + std, alpha=0.3)
            if title is None:
                ax[k].set_title("ARI INIT Kmeans scenario {}".format(sc), fontsize=15)
            else :
                ax[k].set_title(title + " scenario {}".format(sc), fontsize=15)
            ax[k].grid(True)
            ax[k].legend()

    if savefig:
        if path is None:
            path='ari_results.png'
        plt.savefig(path, transparent=False)

        
def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])