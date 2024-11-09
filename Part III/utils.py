from matplotlib import pyplot as plt
import numpy as np


def show_trajectories(x, u, x0):
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
    


    fig, axs = plt.subplots(2, 3, dpi = 150, sharex=True, figsize = (20, 10))
    axs = axs.ravel()
    ylabels = [r"$I_t$",r"$x^d_t$",r"$x^e_t$", r"$x^a_t$", r"$x^u_t$", r"$u_t$"]

    for ax, ylabel in zip(axs, ylabels) :
        ax.set_ylabel(ylabel, rotation = 0, labelpad = 10, fontsize = 14)
        ax.grid(True, which = "both", axis = "both", ls = "--")
        ax.set_xlabel("Time", fontsize = 14) 

    for i, (x, u) in enumerate(zip(arr_x, arr_u)):

        for i in range(len(axs)):
            if i == len(axs) - 1:
                axs[i].plot(np.arange(0, u.shape[-1]), np.mean(u, axis = 0).squeeze())    
            else :
                axs[i].plot(np.arange(0, x.shape[-1]), np.mean(x[:, i, :], axis = 0))


    fig.legend([f"$x_0 = [{x0[0]},{x0[1]}, {x0[2]}, {x0[3]},{x0[-1]}]$" for x0 in arr_x0], loc = "upper center", ncol = 2, bbox_to_anchor=(0.5, 1.0),fontsize = 18, shadow = True)
    # fig.savefig("figures/trajectories_nl.pdf")
    plt.show()