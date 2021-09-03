from .utils import init
import time


def sbm(A,
        Q,
        algo='vbem',
        tau_init=None,
        e_iter=1,
        tol=1e-6,
        max_iter=50,
        type_init='random',
        random_gen=None,
        verbose=True):
    """
    Stochastic Block Model inference using either a variational-bayes EM with coordinate ascent optimisation (VBEM)
      or a Natural-Conjugate gradient optimisation (NCG)

    :param A: Binary adjacency matrix for directed graph
    :param algo: one of 'VBEM', 'NCG', 'VEM'
    :param tau_init: Optional, if given, tau serves as init for the estimation
    :param e_iter: Number of e estimation for the Variational methods
    :param tol: threshold of percentage of variation under which the estimation is stopped
    :param max_iter: maximum iteration
    :param type_init: either 'kmeans' or 'random'
    :param verbose: Print or not the convergence result (bool)
    :return: elbo (list), tau (arr), tau_init (arr), count (int), time_list (list)
    """
    import numpy as np

    count = 0
    t = 0
    c = 1
    epsilon = np.finfo(np.float64).tiny
    A_tilde = np.ones_like(A) - A - np.eye(A.shape[0])

    L_old = - np.inf

    elbo = []
    time_list = []

    old_norm = 1
    old_direction = 0
    BREAK = False


    #######################
    ####### INIT
    #######################

    if tau_init is None :
        tau_init = init(A=A,
                        Q=Q,
                        eps=np.finfo(np.float64).tiny,
                        type_init=type_init,
                        random_gen=random_gen
                        )
    ############
    #### LOAD FUNCTION CORRESPONDING TO THE ALGO
    ############

    if algo=='vbem' or algo=='ncg':
        from .Model_functions import tau2theta, theta2tau, tau_update, gamma_update, kappa_update, grad_tau, norm_grad, get_directions, ELBO
        theta_new = tau2theta(tau_init)
        gamma = gamma_update(tau_init)
        kappa = np.zeros((tau_init.shape[1], tau_init.shape[1], 2))  # Allows to use the "out" option from numpy functions
        kappa = kappa_update(tau_init, A, kappa=kappa)  # VBEM update

    elif algo=='vem':
        from .Model_functions_vem import gamma_update, pi_update, ELBO, tau_update
        from scipy.sparse import csr_matrix, coo_matrix
        A_csr = csr_matrix(A)
        A_csr_T = csr_matrix(A.T)
        A_coo = coo_matrix(A)
        indices_ones = [A_coo.row, A_coo.col]
        gamma = gamma_update(tau_init)
        pi = pi_update(tau_init, indices_ones, threshold=1e-16)

    Q = tau_init.shape[1] # Number of clusters
    M = tau_init.shape[0] # Number of nodes



    tau = tau_init
    #######################
    ####### Training
    #######################
    t0 = time.time()

    while True:

        if algo == 'ncg':
            gamma = gamma_update(tau)  # VBEM update
            kappa = kappa_update(tau, A, kappa=kappa)  # VBEM update
            L = ELBO(tau, gamma, kappa)

            # Compute the ELBO with the VBEM update for gamma and kappa

            if t > 0 and L_old != 0:
                if (L - L_old) / np.abs(L_old) < tol and L >= L_old:
                    elbo.append(L)
                    time_list.append(time.time() - t0)

                    BREAK = True
                    if verbose:
                        print('convergence has been reached')
            if BREAK:
                break

            if t == max_iter:
                if verbose:
                    print("Max iter reached")
                break

            # If the ELBO has increased, we compute another direction
            if L >= L_old:
                elbo.append(L)
                time_list.append(time.time() - t0)

                L_old = L
                t += 1
                rho = 1

                # Update d
                new_grad = grad_tau(gamma, kappa, tau, A, A_tilde)
                new_norm = norm_grad(new_grad, tau)
                new_direction = get_directions(new_grad, new_norm, old_norm, old_direction)

                theta_old = np.copy(theta_new)

                # Step in new direction
                theta_new = theta_old + rho * c * new_direction


            # Else, we diminush the step size until it increases
            else:
                count += 1
                rho /= 2
                c = np.abs((L - L_old) / L)
                theta_new = theta_old + rho * c * new_direction

            tau = theta2tau(theta_new, epsilon)  # Transform theta to tau

            if rho == 0:
                tau = theta2tau(theta_old, epsilon)
                print("Rho = 0")
                break

        elif algo == 'vbem':
            gamma = gamma_update(tau)  # VBEM update
            kappa = kappa_update(tau, A, kappa=kappa)  # VBEM update
            L = ELBO(tau, gamma, kappa)

            # Compute the ELBO with the VBEM update for gamma and kappa
            elbo.append(L)
            time_list.append(time.time() - t0)

            if t > 2 and L_old != 0:
                if (L - L_old) / np.abs(L_old) < tol and L >= L_old:
                    BREAK = True
                    if verbose:
                        print('convergence has been reached')
            if BREAK:
                break

            if t == max_iter:
                if verbose:
                    print("Max iter reached")
                break
            L_old = L

            for k in range(e_iter):
                tau = tau_update(tau, gamma, kappa, A)  # VBEM update

            t += 1

        elif algo == 'vem':
            L = ELBO(tau, gamma, pi, indices_ones)
            elbo.append(L)

            if count > 0:
                if (L - L_old) / np.abs(L_old) < tol and L > L_old:
                    if verbose:
                        print("Convergence has been reached")
                    BREAK = True
            if BREAK:
                break
            L_old = L

            if count == max_iter:
                if verbose:
                    print("Max iter reached")
                break

            # E-step
            for e in range(e_iter):
                tau = tau_update(tau, pi, gamma, A_csr, A_csr_T)

            # M-step
            gamma = gamma_update(tau)
            pi = pi_update(tau, indices_ones)

            count += 1

    return elbo, tau, tau_init, count, time_list