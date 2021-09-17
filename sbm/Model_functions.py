import numpy as np
from scipy.special import softmax, digamma


def gamma_update(tau, delta=1):
    """ VBEM update of the parameter of the variational Dirichlet distribution"""
    return delta + tau.sum(axis=0)


def kappa_update(tau, A, kappa, a=1, b=1):
    """VBEM update of the parameter of the Beta distributions of the probability matrix """
    M1 = A.dot(tau)
    M2 = tau.sum(axis=0, keepdims=True) - tau - M1

    kappa_qr_1 = a + tau.T @ M1
    kappa_qr_2 = b + tau.T @ M2
    return np.stack((kappa_qr_1,kappa_qr_2), out=kappa, axis=-1)


def tau2theta(tau):
    """Switch from tau in the simplex to unconstrained theta"""
    return np.log(tau[:, 0:-1]) - np.log(tau[:, -1]).reshape(-1, 1)


def theta2tau(theta, epsilon):
    """Switch from unconstrained theta to tau"""
    thetaQ = np.hstack((theta, np.zeros((theta.shape[0], 1))))
    tau = softmax(thetaQ, axis=-1)
    np.minimum(tau, 1 - epsilon, out=tau)
    np.maximum(tau, epsilon, out=tau)
    tau = np.array(tau)
    tau = tau / tau.sum(axis=-1, keepdims=True)
    return tau


def grad_tau(gamma, kappa, tau, A, A_tilde):
    """
    Return the gradient of the elbo w.r.t tau, shape : M x Q
    """
    M = tau.shape[0]
    Q = tau.shape[1]
    C1 = A @ tau + (tau.T @ A).T  # shape : M x Q
    C2 = A_tilde @ tau + (tau.T @ A_tilde).T  # shape M x Q
    grad_log_beta = digamma(kappa) - digamma(kappa.sum(axis=-1))[:, :, np.newaxis]  # shape Q x Q x 2
    grad = C1 @ grad_log_beta[:, :, 0] + C2 @ grad_log_beta[:, :, 1]  # shape M x Q
    grad = grad - np.log(tau) + digamma(gamma) - 1
    return grad


def norm_grad(grad, tau):
    """
    theta-Norm of the grad
    """
    '''
    norm = 0
    for i in range(tau.shape[0]):
        nabla_theta_i = np.diag(tau[i, :]) - (tau[i, :, np.newaxis] @ tau.T[np.newaxis, :, i])
        norm += grad[i, :] @ nabla_theta_i @ grad[i, :].T
    return norm
    '''
    nabla_theta = np.apply_along_axis(np.diag, 1, tau) - tau[:, :, np.newaxis] @ tau[:, np.newaxis, :]
    norm = tau[:,np.newaxis,:] @ nabla_theta @ tau[:, np.newaxis, :].swapaxes(1, 2)
    return norm.sum()




def get_directions(grad, new_norm, old_norm, old_direction):
    """
    grad (shape M x Q): grad of the elbo w.r.t tau
    new_norm : norm of the gradient given in input
    old_norm : norm of the previous gradient
    old_direction : previous direction
    """
    Q = grad.shape[1]

    # Compute the gradient with respect to tau and tau_star
    grad_tau_star = grad[:, :(Q - 1)] - grad[:, Q - 1].reshape(-1, 1)

    return grad_tau_star + (new_norm / old_norm) * old_direction


def ELBO(tau, gamma, kappa, delta=None, a=1, b=1):
    """
    ELBO when kappa and gamma are updated following the VBEM steps
    """
    from scipy.special import beta, loggamma
    Q = tau.shape[1]
    M = tau.shape[0]

    A = loggamma(kappa[:,:,0]) + loggamma(kappa[:,:,1]) - loggamma( kappa[:,:,0] + kappa[:,:,1] )
    A0 = 2 * loggamma(1) - loggamma(2)

    D = loggamma(gamma).sum() - loggamma(M + Q)
    D0 = Q * loggamma(1) - loggamma(Q)
    
    elbo = - (tau * np.log(tau)).sum()
    elbo += D - D0
    elbo += (A - A0).sum()


    return elbo


def tau_update(tau, gamma, kappa, A, delta=1):
    """VBEM update of tau """
    # Computation of the constant
    M = A.shape[0]
    Q = gamma.shape[0]

    mincut = np.log(np.finfo(np.float64).tiny)
    maxcut = np.log(np.finfo(np.float64).max) - np.log(Q)

    gamma_constant = digamma(gamma) - digamma(M + Q)
    kappa_2_constant = digamma(kappa[:,:,1]) - digamma(kappa.sum(axis=-1))
    kappa_1_minus_kappa_2 = digamma(kappa[:,:,0]) - digamma(kappa[:,:,1])
    M1 = A.dot(tau)
    M2 = A.T.dot(tau)

    # sum_j tau_jq  for j neq i
    tau_sum_except_i = tau.sum(axis=0, keepdims=True) - tau

    T1 = tau_sum_except_i @ ( kappa_2_constant.T + kappa_2_constant)
    T2 = M1 @ kappa_1_minus_kappa_2.T + M2 @ kappa_1_minus_kappa_2

    log_tau = gamma_constant + T1 + T2 - 1

    # switch log tau to tau then normalize
    log_tau = np.minimum(log_tau, maxcut)
    log_tau = np.maximum(log_tau,  mincut)
    new_tau = np.array(np.exp(log_tau))
    new_tau = new_tau / new_tau.sum(axis=-1, keepdims=True)

    return new_tau