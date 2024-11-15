import numpy as np
from typing import Tuple, List

def add_functions(lam : float, func1 : callable, func2 : callable) -> callable:
    """
    Add two functions together and mutiply the first one by lam.

    :param lam: Memory parameter.
    :param func1: First function.
    :param func2: Second function.
    
    :return: Function that adds the two input functions.
    """

    return lambda *args, **kwargs: lam*func1(*args, **kwargs) + func2(*args, **kwargs)


def q_learning(lam : float, 
               data : np.ndarray, 
               step_size : callable, 
               theta0 : np.ndarray, 
               zeta0 : List[callable], 
               psi : List[callable], 
               cost : callable, 
               action_space : np.ndarray) -> np.ndarray:
    """
    Perform the Q(λ) learning algorithm to estimate the Q-function.

    :param lam: memory parameter.
    :param data: data pair (x, u) matrix with x : shape = (n_traj, n, N) and u : shape = (n_traj, 1,N)
    :param step_size: Step size function.
    :param theta0: Initial parameter vector.
    :param zeta0: Initial basis function
    :param psi: Final basis function.

    :return: Estimated parameter vector θ for the Q-function
    """


    if len(psi) != len(zeta0):
        raise ValueError("psi and zeta0 must have the same length")
    
    
    x_data, u_data = data

   
    if x_data.ndim == 2:
        x_data = x_data[np.newaxis, ...]
        u_data = u_data[np.newaxis, ...]

    psi_vector = lambda x, u: np.array([psi_i(x, u) for psi_i in psi])
    zeta_vector = lambda x, u: np.array([zeta_i(x, u) for zeta_i in zeta0])

    theta = theta0

    for traj in range(x_data.shape[0]):

        for k in range(1, x_data.shape[2]-1):

            x_k = x_data[traj, :, k]          # State at time k
            u_k = u_data[traj, 0, k]           # Action at time k
            x_k_plus = x_data[traj, :, k + 1]  # State at time k+1

            Q_theta = lambda x, u : np.dot(theta, zeta_vector(x, u))
            phi_theta = action_space[np.argmin([Q_theta(x_k_plus, u) for u in action_space])]

            D_k_plus = - Q_theta(x_k, u_k) + cost(x_k, u_k) + Q_theta(x_k_plus, phi_theta)
            theta += step_size(k) * D_k_plus * zeta_vector(x_k, u_k)
            zeta_vector = add_functions(lam, zeta_vector, psi_vector)

    return theta


