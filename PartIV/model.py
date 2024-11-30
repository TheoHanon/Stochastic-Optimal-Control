import numpy as np
from typing import Tuple


#####################
# Model Parameters  #
#####################

THETA   = 1/2
KAPPA   = 1/2
OMEGA_A = 0.1
OMEGA_U = 0.1
GAMMA_U = 0.9
BETA_U  = 0.1 
BETA_A  = 0.1 
ALPHA_U = 1
SIGMA_D = 1
LAMBDA  = 1



A = np.array([
        [1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, THETA, THETA - KAPPA],
        [0, 0, 0, 1 - OMEGA_A, 0],
        [0, 0, 0, 0, 1 - OMEGA_U]])
B = np.array([1, GAMMA_U, (THETA - KAPPA) * GAMMA_U, 0, OMEGA_U * BETA_U])

Sigma_x = np.array([
    [0, 0, 0, 0, 0], 
    [0, SIGMA_D**2, THETA*SIGMA_D**2, 0, 0],
    [0, THETA*SIGMA_D**2, THETA**2*SIGMA_D**2, 0, 0],
    [0, 0, 0, BETA_A**2, 0],
    [0, 0, 0, 0, 0]])

Q = np.array([
    [LAMBDA*SIGMA_D**2, 0, 0, -1/2, -1/2], 
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [-1/2, 0, 0, 0, 0],
    [-1/2, 0, 0, 0, 0]])
S = np.array([
    [LAMBDA*THETA*SIGMA_D**2 - GAMMA_U/2, 0, 0, -THETA/2, -(THETA - KAPPA)/2]])

R = np.array([[LAMBDA*THETA**2*SIGMA_D**2 - (THETA - KAPPA)*GAMMA_U]])
C = np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0]])
L = np.block([[Q, S.T], [S, R]])

#####################
# Cost Functions    #
#####################

def cost(x: np.ndarray, u: np.ndarray) -> float:

    if np.isscalar(u):
        u = np.array([u])
        
    return np.concatenate((x, u)) @ L @ np.concatenate((x, u))

#####################
# Model Simulation  #
#####################


def linear_step(x: np.ndarray, u: np.ndarray, noise : np.ndarray) -> np.ndarray:
    # noise is (xi_d, xi_a)
    xi = np.array([0, SIGMA_D*noise[0], THETA*SIGMA_D*noise[0], BETA_A*noise[1], 0]) # noise constants
    return A @ x + B * u + xi 

def simulate_linear_model(x0: np.ndarray, policy : callable, n_step : int, with_noise : bool = False, std_const : float = 1) -> Tuple[np.ndarray, np.ndarray]:

    x = np.zeros((5, n_step))
    u = np.zeros((1, n_step-1))

    x[:,0] = x0

    for i in range(n_step-1):

        u[:, i] = policy(x[:,i])
        
        if with_noise:
            x[:,i+1] = A @ x[:,i] + B * u[:, i] + std_const * np.random.multivariate_normal(np.zeros(5), Sigma_x * std_const**2)
        else:
            x[:,i+1] = A @ x[:,i] + B * u[:, i]

    return (x, u)


def non_linear_step(x: np.ndarray, u: np.ndarray, noise : np.ndarray) -> np.ndarray:
    # noise is (xi_d, xi_a)
    F = np.array([
        x[0] + u[0],
        x[3] + x[4] + GAMMA_U* u[0] + ALPHA_U * u[0] * x[4],
        THETA * x[3] + (THETA - KAPPA) * x[4] + (THETA - KAPPA) * GAMMA_U * u[0] + (THETA - KAPPA) * ALPHA_U * u[0] * x[4],
        (1 - OMEGA_A) * x[3],
        (1 - OMEGA_U) * x[4] + OMEGA_U * BETA_U * u[0]
    ])

    xi = np.array([0, SIGMA_D*noise[0], THETA*SIGMA_D*noise[0], BETA_A*noise[1], 0]) # noise constants

    return F + xi 


def simulate_nl_model(x0: np.ndarray, policy : callable, n_step : int, with_noise : bool = False, std_const : float = 1) -> Tuple[np.ndarray, np.ndarray]:

    x = np.zeros((5, n_step))
    u = np.zeros((1, n_step-1))

    x[:,0] = x0

    for i in range(n_step-1):

        I = x[0,i]
        x_a = x[3,i]
        x_u = x[4,i]

        # Control input based on the LQG controller
        u[:, i] = policy(x[:, i])
     
        # Update state 
        F = np.array([
            I + u[0, i],
            x_a + x_u + GAMMA_U* u[0, i] + ALPHA_U * u[0, i] * x_u,
            THETA * x_a + (THETA - KAPPA) * x_u + (THETA - KAPPA) * GAMMA_U * u[0, i] + (THETA - KAPPA) * ALPHA_U * u[0, i] * x_u,
            (1 - OMEGA_A) * x_a,
            (1 - OMEGA_U) * x_u + OMEGA_U * BETA_U * u[0, i]
        ])

        if with_noise:
            xi_d = np.random.normal(0, std_const)
            xi_a = np.random.normal(0, std_const)

            xi = np.array([0, SIGMA_D*xi_d, THETA*SIGMA_D*xi_d, BETA_A*xi_a, 0]).reshape(-1)

            x[:, i+1] = F + xi
        else:
            x[:, i+1] = F

    return (x, u)



#####################
# LQR Parameters &  #
#     policy        #
#####################


K_star = np.array([0.85724945,  0.,          0.,         -0.40135333, -0.38022947])

def LQR_policy(x: np.ndarray) -> np.ndarray:
    return -K_star @ x


#####################
# Q-function        #
#####################

def Q_function(x, u, theta, psi_func : callable):
    return np.dot(psi_func(x, u), theta)



