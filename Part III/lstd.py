import numpy as np
from scipy.optimize import minimize
from typing import List, Tuple


def perform_lstd(policy: callable, cost: callable, psi: callable, W: np.ndarray, data: Tuple[np.ndarray, np.ndarray], d: int) -> np.ndarray:
    """
    Perform LSTD algorithm to estimate the Q-function.

    Parameters:
    -----------
    policy: Callable representing the policy function π(x) -> u.
    cost: Callable representing the cost function c(x, u).
    psi: Callable representing the feature function ψ : (x, u) -> R^d.
    W: Regularization matrix.
    data: Tuple of data arrays (x_data, u_data) with shapes (n, N) and (m, N), respectively.
    d: Dimension of the feature space.

    Returns:
    --------
    theta_LSTD_N: Estimated Q-function parameter vector.
    """

    x_data, u_data = data
    
    N = x_data.shape[1]  # N is the number of data points
    
    R_N = np.zeros((d, d))
    psi_bar_gamma = np.zeros(d)

    # Compute R_N and psi_bar_gamma
    for k in range(N - 1):
        x_k = x_data[:, k]          # State at time k
        u_k = u_data[:, k]           # Action at time k
        x_k_plus = x_data[:, k + 1]  # State at time k+1

        gamma_k = cost(x_k, u_k)
        Upsilon_k_plus = psi(x_k, u_k[0]) - psi(x_k_plus, policy(x_k_plus))

        # Accumulate values for R_N and psi_bar_gamma
        R_N += np.outer(Upsilon_k_plus, Upsilon_k_plus) / N
        psi_bar_gamma += Upsilon_k_plus * gamma_k / N

    # Compute the LSTD solution
    theta_LSTD_N = np.linalg.solve(R_N + W/N, psi_bar_gamma)
    return theta_LSTD_N


def perform_lstd_PI(initial_policy : callable, cost : callable, psi : callable, W : np.ndarray, data : Tuple[np.ndarray], n_iter : int, action_space : tuple, d : int) -> np.ndarray:
    """
    Perform LSTD algorithm to estimate the Q-function with policy iteration
    
    Parameters:
    -----------
    initial_policy: Initial policy function.
    cost: Cost function.
    psi: Callable representing the feature function ψ : (x, u) -> R^d.
    W: Regularization matrix.
    data: Tuple of data arrays (x_data, u_data) with shapes (n, N) and (m, N), respectively.
    n_iter: Number of policy iterations.
    action_space: Tuple representing the bounds of the action space.
    d: Dimension of the feature space.

    Returns:
    --------
    theta_LSTD_N: Estimated Q-function parameter vector.
    
    """
    
    theta = None
    policy = initial_policy

   

    for iter in range(n_iter):
        theta = perform_lstd(policy, cost, psi, W, data, d)

        def improved_policy(x):
            sum = theta[4]*x[0] + theta[5]*x[3] + theta[6]*x[4]
            return - sum / (2*theta[3])
        
        policy = improved_policy

    return theta