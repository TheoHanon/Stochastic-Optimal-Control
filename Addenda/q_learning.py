import numpy as np
from typing import Callable, Tuple
import model
import utils



def q_learning(
        lam : float,
        theta0 : np.ndarray,
        zeta0 : np.ndarray, 
        alpha : Callable[[int], float], 
        psi : Callable[[np.ndarray, np.ndarray], np.ndarray],
        data : Tuple[np.ndarray,np.ndarray]):
    
    x, u = data
    zeta = zeta0
    theta = theta0
    D = -np.dot(theta, psi(x[:, 0],u[:, 0])) + model.cost(x[:, 0],u[:, 0]) + np.dot(theta, psi(x[:, 0], policy_Q(x[:, 0],theta)))

    for k in range(len(x)-1):

        theta += alpha(k) * D * zeta
        zeta = lam * zeta + psi(x[:, k],u[:, k])
        D = -np.dot(theta, psi(x[:, k],u[:, k])) + model.cost(x[:, k],u[:, k]) + np.dot(theta, psi(x[:, k+1], policy_Q(x[:, k+1],theta)))    

    return theta




