import numpy as np
from typing import List, Tuple


def perform_lstd(policy: callable, cost: callable, psi: List[callable], W: np.ndarray, data: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Perform LSTD algorithm to estimate the Q-function.

    :param policy: Callable representing the policy function π(x) -> u.
    :param cost: Callable representing the cost function c(x, u).
    :param psi: List of feature functions.
    :param W: Regularization matrix.
    :param data: Tuple of data arrays (x_data, u_data) with shapes (n_traj, n, N) and (n_traj, m, N), respectively.
    
    :return: Estimated parameter vector θ_LSTD_N for the Q-function.
    """
    x_data, u_data = data

    if x_data.ndim == 2:
        x_data = x_data[np.newaxis, ...]
        u_data = u_data[np.newaxis, ...]

    N = x_data.shape[1]  # N is the number of data points
    d = len(psi)  # Number of basis functions

    # Define psi_vector to compute feature vector for given x and u
    psi_vector = lambda x, u: np.array([psi_i(x, u) for psi_i in psi])

    R_N = np.zeros((d, d))
    psi_bar_gamma = np.zeros(d)

    # Compute R_N and psi_bar_gamma
    for traj in range(x_data.shape[0]):

        for k in range(N - 1):
            x_k = x_data[traj, :, k]          # State at time k
            u_k = u_data[traj, :, k]           # Action at time k
            x_k_plus = x_data[traj, :, k + 1]  # State at time k+1

            gamma_k = cost(x_k, u_k)
            Upsilon_k_plus = psi_vector(x_k, u_k[0]) - psi_vector(x_k_plus, policy(x_k_plus))

            # Accumulate values for R_N and psi_bar_gamma
            R_N += np.outer(Upsilon_k_plus, Upsilon_k_plus) / (N*x_data.shape[0])
            psi_bar_gamma += Upsilon_k_plus * gamma_k / (N * x_data.shape[0])


    # Compute the LSTD solution
    theta_LSTD_N = np.linalg.solve(R_N + W/(N*x_data.shape[0]), psi_bar_gamma)

    return theta_LSTD_N


def perform_lstd_PI(initial_policy : callable, cost : callable, psi : List[callable], W : np.ndarray, data : Tuple[np.ndarray], n_iter : int, action_space : np.ndarray) -> np.ndarray:
    """
    Perform LSTD algorithm to estimate the Q-function with policy iteration

    :param phi: List of feature functions
    :param data: data pair (x, u) matrix with x : shape = (N, n) and u : shape = (N, m)

    :return: Estimated parameter vector θ_LSTD_N for the Q-function
    """

    theta_LSTD_N = None
    psi_vector = lambda x, u : np.array([psi_i(x, u) for psi_i in psi])
    policy = initial_policy

    for iter in range(n_iter):
        theta_LSTD_N = perform_lstd(policy, cost, psi, W, data)

        def improved_policy(x):
            return min(action_space, key = lambda u : psi_vector(x, u) @ theta_LSTD_N)

        policy = improved_policy

    return theta_LSTD_N, policy
