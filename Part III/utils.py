from matplotlib import pyplot as plt
import numpy as np
from typing import List, Tuple, Callable



def show_trajectories(x, u, x0, legend_labels):
    """
    Create a plot of the trajectories of the system states and control inputs.

    :param x: List of state trajectories with shape (n_init, n_traj, x_dim, N_step) or (n_traj, x_dim, N_step) or (x_dim, N_step).
    :param u: List of control input trajectories with shape (n_init, n_traj, N_step) or (n_traj, N_step) or (N_step).
    :param x0: List of initial states with shape (n_init, x_dim) or (x_dim).
    
    """

    arr_x, arr_u, arr_x0 = np.array(x), np.array(u), np.array(x0)

    if len(arr_x.shape) == 2:
        arr_x = arr_x[np.newaxis, ...]
        arr_u = arr_u[np.newaxis, ...]

    if len(arr_x.shape) == 3:
        arr_x = arr_x[np.newaxis, ...]
        arr_u = arr_u[np.newaxis, ...]

    if len(arr_x0.shape) == 1:
        arr_x0 = arr_x0[np.newaxis, ...]
    
    colors = ["b", "g", "r", "c", "m", "y", "k", "w"]
    linestyles = ["-", "--", "-.", ":", "-.", "--", "-"]

    fig, axs = plt.subplots(2, 3, dpi = 150, sharex=True, figsize = (20, 10))
    axs = axs.ravel()
    ylabels = [r"$I_t$",r"$x^d_t$",r"$x^e_t$", r"$x^a_t$", r"$x^u_t$", r"$u_t$"]

    for ax, ylabel in zip(axs, ylabels) :
        ax.set_ylabel(ylabel, rotation = 0, labelpad = 10, fontsize = 14)
        ax.grid(True, which = "both", axis = "both", ls = "--")
        ax.set_xlabel("Time step", fontsize = 14) 

    for j, (x, u) in enumerate(zip(arr_x, arr_u)):

        color = colors[j]
        linestyle = linestyles[j] 
        
        for i in range(len(axs)):
   
            if i == len(axs) - 1:
                axs[i].plot(np.arange(0, u.shape[-1]), np.mean(u, axis = 0).squeeze(), color = color, linestyle = linestyle) 
            else :
                axs[i].plot(np.arange(0, x.shape[-1]), np.mean(x[:, i, :], axis = 0), color = color, linestyle = linestyle)

    fig.legend([f"{label_name} : $x_0 = [{x0[0]},{x0[1]}, {x0[2]}, {x0[3]},{x0[-1]}]$" for label_name ,x0 in zip(legend_labels, arr_x0)], loc = "upper center", ncol = 2, bbox_to_anchor=(0.5, 1.0),fontsize = 18, shadow = True)
    #fig.savefig("figures/trajectories_lin_vs_nl_PI.pdf")
    plt.show()

def compute_TD(data : np.ndarray, Q : Callable[[np.ndarray, np.ndarray, callable], float], phi : Callable[[np.ndarray], float], cost : Callable[[np.ndarray, np.ndarray], float], psi : Callable[[np.ndarray, np.ndarray], np.ndarray], theta : np.ndarray) -> np.ndarray:
    x, u = data
    TD = np.empty(x.shape[1] - 1)

    for k in range(x.shape[1]-1):

        x_k = x[:, k]
        u_k = u[:, k]
        x_k_plus = x[:, k+1]

        TD[k] = cost(x_k, u_k) + Q(x_k_plus, phi(x_k_plus), theta, psi) - Q(x_k, u_k, theta, psi)

    return TD