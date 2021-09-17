import numpy as np
from scipy.special import softmax, digamma
from scipy.sparse import csr_matrix, coo_matrix


def gamma_update(tau):
    """ VEM update of the parameter of the variational Dirichlet distribution"""
    return tau.mean(axis=0)


def pi_update(tau, indices_ones, threshold=1e-16):
    """VEM update of pi"""
    Q = tau.shape[1]
    tau_sum = tau.sum(axis=0) - tau

    pi = (tau[indices_ones[0]].reshape(-1, Q, 1)
          * tau[indices_ones[1]].reshape(-1, 1, Q)
          ).sum(0) / (tau_sum.T @ tau)

    pi = np.maximum(pi, threshold)
    pi = np.minimum(pi, 1 - threshold )

    return pi


def tau_update(tau, pi, gamma, A, A_t):
    """VEM update of tau """
    # Computation of the constant
    M = A.shape[0]
    Q = gamma.shape[0]

    log_ratio_pi = np.log(pi) - np.log(1-pi)
    log_1_minus_pi = np.log(1-pi)
    mincut = np.log(np.finfo(np.float64).tiny)
    maxcut = np.log(np.finfo(np.float64).max) - np.log(Q)


    M1 = A.dot(tau)
    M2 = A_t.dot(tau)

    # sum_j tau_jq  for j neq i
    tau_sum_except_i = tau.sum(axis=0, keepdims=True) - tau

    log_tau = tau_sum_except_i @ (log_1_minus_pi + log_1_minus_pi.T)
    log_tau += M1.dot(log_ratio_pi.T) + M2.dot(log_ratio_pi)
    log_tau += np.log(gamma)
    log_tau -= np.max((1, log_tau.max()))  # For numerical stability

    # switch log tau to tau then normalize
    log_tau = np.minimum(log_tau, maxcut)
    log_tau = np.maximum(log_tau,  mincut)
    new_tau = np.array(np.exp(log_tau))
    new_tau = new_tau / new_tau.sum(axis=-1, keepdims=True)

    return new_tau


def ELBO(tau, gamma, pi, index):
    """
    index shape (m,2) : for each edge m, index[m] holds the nodes of the edges
    W shape (m, V) : for each edge, return the count of words of the document shared on that edge
    """
    from scipy.special import beta, loggamma
    Q = tau.shape[1]
    M = tau.shape[0]
    tau_sum = tau.sum(0)
    tau_prod = np.sum(tau[index[0]].reshape(-1, Q, 1) * tau[index[1]].reshape(-1, 1, Q),
                      axis=0)
    elbo = (tau_prod * (np.log(pi) - np.log(1 - pi))).sum()

    elbo += (((tau_sum.reshape((-1, 1)) * tau_sum) - tau.T @ tau)
            * np.log(1 - pi)).sum()

    elbo -= (tau * np.log(tau)).sum()
    elbo += tau.sum(axis=0) @ np.log(gamma)
    return elbo
