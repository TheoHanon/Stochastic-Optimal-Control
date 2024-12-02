from matplotlib import pyplot as plt
import numpy as np
from typing import List, Tuple, Callable


def show_trajectories(data_dict, title=None, use_labels=True, save=None):
    """
    Create a plot of the trajectories of the system states and control inputs.

    :param data_dict: Dictionary where keys are string representations of x0, 
                        and values are tuples (x, u).
                        - x: State trajectories with shape (n_traj, x_dim, N_step) or (x_dim, N_step).
                        - u: Control input trajectories with shape (n_traj, N_step) or (N_step).
    :param title: Title of the plot (optional).
    :param use_labels: Boolean to toggle using dictionary keys as labels.
    :param save: Filename to save the plot (optional, without extension).
    """
    colors = ["b", "g", "r", "c", "m", "y", "k", "w"]
    linestyles = ["-", "--", "-.", ":", "-.", "--", "-"]

    fig, axs = plt.subplots(2, 3, dpi=150, sharex=True, figsize=(20, 10))
    axs = axs.ravel()
    ylabels = [r"$I_t$", r"$x^d_t$", r"$x^e_t$", r"$x^a_t$", r"$x^u_t$", r"$u_t$"]

    for ax, ylabel in zip(axs, ylabels):
        ax.set_ylabel(ylabel, rotation=0, labelpad=10, fontsize=14)
        ax.grid(True, which="both", axis="both", ls="--")
        ax.set_xlabel("Time step", fontsize=14)

    for idx, (label, (x, u)) in enumerate(data_dict.items()):
        # Extract x0 from the label
        arr_x, arr_u = np.array(x), np.array(u)

        if len(arr_x.shape) == 2:
            arr_x = arr_x[np.newaxis, ...]
            arr_u = arr_u[np.newaxis, ...]

        if len(arr_x.shape) == 3:
            arr_x = arr_x[np.newaxis, ...]
            arr_u = arr_u[np.newaxis, ...]

        color = colors[idx % len(colors)]
        linestyle = linestyles[idx % len(linestyles)]

        for i in range(len(axs)):
            if i == len(axs) - 1:
                # Control input trajectory plotting
                axs[i].plot(
                    np.arange(0, arr_u.shape[-1]),
                    np.mean(arr_u, axis=0).squeeze(),  # Ensuring a 1D array
                    color=color,
                    linestyle=linestyle,
                )
            else:
                # State trajectory plotting
                axs[i].plot(
                    np.arange(0, arr_x.shape[-1]),
                    np.mean(arr_x[:, :, i], axis=0).squeeze(),  # Ensuring a 1D array
                    color=color,
                    linestyle=linestyle,
                )



    if use_labels:
        legend_labels = [
            label
            for label in data_dict.keys()
        ]
        fig.legend(legend_labels, loc="upper center", ncol=2, bbox_to_anchor=(0.508, 1.05), fontsize=18, shadow=True)

    if title:
        fig.suptitle(title, fontsize=20)

    if save is not None:
        fig.savefig(save, dpi=300, bbox_inches="tight")
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