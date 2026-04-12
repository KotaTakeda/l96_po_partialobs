#!/usr/bin/env python3
"""Run the Lorenz96 DA experiment from example.ipynb."""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from da.l96 import lorenz96
from da.scheme import rk4
from da.visualize import plot_loss
from da.etkf import ETKF
from da.po import PO
import visualize

plt.rcParams['text.usetex'] = False


def loss_sq(X, Y):
    """
    X, Y: (..., Nt, J) or broadcastable arrays.
    """
    return np.sum((X - Y) ** 2, axis=-1)


def stats(X):
    """
    X: (N_seed, Nt)
    """
    return X.mean(axis=0), X.std(axis=0)


def main():
    data_dir = "data/20260412v1"

    # ==========================================
    # Parameter Configurations
    # ==========================================
    # System Parameters
    J = 60          # dim of state space
    F = 8           # forcing
    dt = 0.01       # time step size
    N0 = 20 * 360   # number of spin-up time steps
    N = 20 * 50     # number of simulation time steps

    # Observation Parameters
    obs_per = 1     # observation interval in steps
    r = 1.0         # observation noise std

    # Data Assimilation Parameters
    m = 10          # ensemble size
    num_seeds = 20   # number of random seeds for simulation & plotting
    seed_list = np.arange(num_seeds)
    alpha_list = [0.0, 0.5, 2.0, 10.0, 100.0]  # multiplicative inflation factors
    methods = ["po_add", "po_proj"]  # DA methods

    # Visualization / Evaluation Parameters
    method = "po_add"          # default DA method to plot
    per_vis = 20               # plotting frequency
    per_ticklabel = per_vis * 20  # plot ticks parameter
    N_end = 50                 # plot until N_end steps in SE plot
    i_seed = 0                 # seed index for spatio-temporal plots
    alpha_list_vis_st = [0.0, 0.5, 2.0]  # alpha list for spatio-temporal plots
    k_ens = 0                  # ensemble index for spatio-temporal plots
    n_start = 0                # start step for spatio-temporal plots
    n_end = N                  # end step for spatio-temporal plots

    # ==========================================
    # Generate the true trajectory
    # ==========================================
    x0 = F * np.ones(J)
    x0[19] *= 1.001

    scheme = rk4
    p = (F,)

    try:
        x_true = np.load(f"{data_dir}/x_true_l96_full.npy")
        print("x_true loaded:", x_true.shape)
    except Exception:
        result = np.zeros((N0 + N, len(x0)))
        x = x0.copy()
        result[0] = x[:]

        for n in range(1, N0 + N):
            t = n * dt
            x = scheme(lorenz96, t, x, p, dt)
            result[n] = x[:]

        x_true = result[N0:]
        print(x_true.shape)
        np.save(f"{data_dir}/x_true_l96_full", x_true)

    # norm (normalized energy)
    norm = np.linalg.norm(x_true, axis=-1) / np.sqrt(J)
    t = np.arange(N) * dt
    fig3, ax3 = plt.subplots()
    ax3.grid(False)
    ax3.set_title('norm after spin-up')
    ax3.set_xlabel('$t$')
    ax3.set_ylabel('$ |u|/\\sqrt{J}$')
    ax3.set_ylim([0.0, 8.5])
    ax3.plot(t, norm, lw=0.5)
    fig3.tight_layout()

    # ==========================================
    # DA setting
    # ==========================================
    print("obs_per", obs_per, "steps")
    Dt = dt * obs_per

    def M(x, Dt):
        for _ in range(obs_per):
            x = rk4(lorenz96, 0, x, p, dt)
        return x

    H_diag = np.ones(J)
    H_diag[2::3] = 0
    H = np.diag(H_diag)
    H = H[H_diag != 0]

    Ny = np.linalg.matrix_rank(H)
    print("H.shape:", H.shape)
    print("diag of H:", H_diag)
    print("rank(H):", Ny)

    R = r ** 2 * np.eye(J)
    R = H @ R @ H.T
    print("obs noise std:", r)

    Nt = N
    x_true = np.load(f"{data_dir}/x_true_l96_full.npy")[:Nt:obs_per]

    print("ensemble size m:", m)
    P0 = 4**2 * np.eye(J)

    # ==========================================
    # Run DA over alpha
    # ==========================================
    for method_name in methods:
        print(method_name)
        paramname = f"m({m})-{method_name}-alpha({'='.join([str(alpha) for alpha in alpha_list])})-seeds({len(seed_list)})"
        try:
            Xa = np.load(f"{data_dir}/xa-{paramname}.npy")
            print("Xa loaded:", Xa.shape)
        except Exception:
            Xa = np.zeros((len(alpha_list), len(seed_list), Nt, m, J))
            Y = np.zeros((len(seed_list), Nt, H.shape[0]))
            for j, seed in enumerate(seed_list):
                np.random.seed(seed)
                y = (H @ x_true.T).T
                y += np.random.multivariate_normal(mean=np.zeros_like(y[0]), cov=R, size=len(y))
                Y[j] = y[:]

                X_0 = x_true[np.random.randint(len(x_true))] + np.random.multivariate_normal(
                    mean=np.zeros_like(x_true[0]), cov=P0, size=m
                )

                for i, alpha in enumerate(alpha_list):
                    print("alpha:", alpha)
                    np.random.seed(seed)

                    if method_name == "etkf":
                        da_instance = ETKF(M, H, R, alpha=alpha**2, store_ensemble=True)
                    elif method_name == "po_add":
                        da_instance = PO(
                            M,
                            H,
                            R,
                            alpha=alpha**2,
                            store_ensemble=True,
                            additive_inflation=True,
                        )
                    elif method_name == "po_proj":
                        da_instance = PO(
                            M,
                            H,
                            R,
                            alpha=alpha**2,
                            store_ensemble=True,
                            additive_inflation=True,
                            project_cov=True,
                        )
                    else:
                        raise ValueError(f"Unknown method: {method_name}")

                    da_instance.initialize(X_0)
                    for y_obs in y:
                        da_instance.forecast(Dt)
                        da_instance.update(y_obs)

                    Xa[i, j] = da_instance.Xa

            np.save(f"{data_dir}/xa-{paramname}", Xa)
            np.save(f"{data_dir}/y-seeds({len(seed_list)})", Y)

    # ==========================================
    # Plot errors and minimum eigenvalues
    # ==========================================
    from visualize import get_linestyle_cycle, get_marker_cycle

    print(f"Visualize until {per_vis * N_end} steps")
    print("load file")
    print(methods)

    fig1, ax1 = plt.subplots(figsize=(7, 4))
    fig2, ax2 = plt.subplots(figsize=(7, 4))

    num_alpha = len(alpha_list)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    time_ticks = np.arange(Nt // per_vis) * per_vis + 1
    time_ticklabels = np.arange(Nt // per_ticklabel + 1) * per_ticklabel
    time_ticklabels[0] = 1

    for k, method_name in enumerate(methods):
        paramname = f"m({m})-{method_name}-alpha({'='.join([str(alpha) for alpha in alpha_list])})-seeds({len(seed_list)})"
        print(paramname)
        Xa = np.load(f"{data_dir}/xa-{paramname}.npy")

        color = colors[k]
        line_cycle = get_linestyle_cycle()
        marker_cycle = get_marker_cycle()

        for i, xa in enumerate(Xa):
            alpha = alpha_list[i]
            if i in [3, 4]:
                print("skip alpha:", alpha)
                continue

            ls = next(line_cycle)
            marker = next(marker_cycle)

            se_wtm = loss_sq(x_true[None, :, None, :] - xa, 0) + loss_sq(x_true[None, :, None, :] @ H.T - xa @ H.T, 0)
            se_tm, _ = stats(se_wtm)
            se_t = se_tm.mean(axis=1)
            se_wt = se_wtm.mean(axis=2)
            print("t-averaged SE", se_tm[Nt // 2:].mean())

            ax1.plot(
                time_ticks[:N_end],
                se_t[::per_vis][:N_end],
                label=f"{method_name.replace('po_', '')} $\\alpha$={alpha}",
                lw=0.5,
                ls=ls,
                color=color,
                marker=marker,
                ms=5,
            )
            for se_k in se_wt:
                ax1.plot(time_ticks[:N_end], se_k[::per_vis][:N_end], lw=0.25, color=color, alpha=0.3)

            Pi = H.T @ H
            Q = np.eye(J) - Pi
            dX = xa - xa.mean(axis=2, keepdims=True)
            P = (dX.swapaxes(-2, -1) @ dX) / (m - 1)
            QPHt = Q @ P @ Pi.T
            HPHt = Pi @ P @ Pi.T
            rf = (np.linalg.norm(QPHt, axis=(2, 3)) / np.linalg.norm(HPHt, axis=(2, 3))).mean(axis=0)
            ax2.plot(time_ticks[:N_end], rf[::per_vis][:N_end], label=f"{method_name.replace('po_', '')} $\\alpha$={alpha}", lw=0.5, ls=ls, color=color, marker=marker, ms=5)

    ax1.plot(time_ticks[:N_end], 4 * Ny * (r**2) * np.ones_like(time_ticks[:N_end]), label='$ 4 N_y r^2 $', lw=0.5, c='black')

    ax1.set_xlabel("observation time step")
    ax1.set_title("The time series of $\\mathrm{MSE}$")
    ax1.set_ylabel(r"$ \frac{1}{m} \sum_{k=1}^m \mathbb{E} \|\delta^{(k)}\|^2 $")
    ax1.set_yscale("log")
    fig1.tight_layout()
    ax1.legend(bbox_to_anchor=(1.0, 1.0), loc='upper right')
    fig1.tight_layout()

    ax2.set_title("The off-diagonal ratio in the covariance")
    ax2.set_ylabel(r"$\|QP\Pi\|_F / \|\Pi P \Pi\|_F$")
    ax2.set_xlabel("observation time step")
    ax2.set_ylim([0.0, 2.0])
    ax2.legend(bbox_to_anchor=(1.0, 1.0), loc='upper right')
    fig2.tight_layout()

    fig1.savefig(f"{data_dir}/l96-po-inflation_Pse.pdf", transparent=True)
    fig2.savefig(f"{data_dir}/l96-po-inflation_offDiag.pdf", transparent=True)

    # ==========================================
    # Last covariance plots
    # ==========================================
    num_methods = len(methods)
    num_alphas = len(alpha_list)

    observed_idx = np.where(np.diag(H.T @ H) > 0.5)[0]
    unobserved_idx = np.where(np.diag(H.T @ H) <= 0.5)[0]
    sort_idx = np.concatenate([observed_idx, unobserved_idx])
    vmax = 1.0
    vmin = -1.0

    fig1, axes1 = plt.subplots(num_methods, num_alphas, figsize=(4 * num_alphas, 4 * num_methods))
    fig2, axes2 = plt.subplots(num_methods, num_alphas, figsize=(4 * num_alphas, 4 * num_methods))

    if num_methods == 1 and num_alphas == 1:
        axes1 = np.array([[axes1]])
        axes2 = np.array([[axes2]])
    elif num_methods == 1:
        axes1 = axes1[np.newaxis, :]
        axes2 = axes2[np.newaxis, :]
    elif num_alphas == 1:
        axes1 = axes1[:, np.newaxis]
        axes2 = axes2[:, np.newaxis]

    for r_idx, method_name in enumerate(methods):
        paramname = f"m({m})-{method_name}-alpha({'='.join([str(alpha) for alpha in alpha_list])})-seeds({len(seed_list)})"
        Xa = np.load(f"{data_dir}/xa-{paramname}.npy")
        for c, alpha in enumerate(alpha_list):
            xa = Xa[c, i_seed]
            dX = xa - xa.mean(axis=1, keepdims=True)
            P = (dX.swapaxes(-2, -1) @ dX) / (m - 1)
            P_last = P[-1]
            P_last /= P_last.max()

            ax1 = axes1[r_idx, c]
            im1 = ax1.imshow(P_last, cmap="coolwarm", interpolation="none", vmax=vmax, vmin=vmin)
            ax1.set_title(f"{method_name}, $\\alpha$={alpha}")
            ax1.set_xticks([])
            ax1.set_yticks([])
            if c == len(alpha_list) - 1:
                fig1.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

            P_rearranged = P_last[sort_idx][:, sort_idx]
            ax2 = axes2[r_idx, c]
            im2 = ax2.imshow(P_rearranged, cmap="coolwarm", interpolation="none", vmax=vmax, vmin=vmin)
            ax2.set_title(f"{method_name}, $\\alpha$={alpha}")
            ax2.set_xticks([])
            ax2.set_yticks([])
            if c == len(alpha_list) - 1:
                fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    fig1.suptitle("Normalized Last Covariance $P_n/\\max(P_{i,j})$ (Original)")
    fig1.tight_layout()
    fig2.suptitle("Normalized Last Covariance $P_n/\\max(P_{i,j})$ (Rearranged: Observed first, Unobserved last)")
    fig2.tight_layout()
    plt.show()

    fig2.savefig(f"{data_dir}/normalized_last_covariance")

    # ==========================================
    # Spatio-temporal Plot
    # ==========================================
    x_true = np.load(f"{data_dir}/x_true_l96_full.npy")
    Y = np.load(f"{data_dir}/y-seeds({num_seeds}).npy")
    y = Y[i_seed]

    num_methods = len(methods)
    num_alphas = len(alpha_list_vis_st)

    x1 = x_true[n_start:n_end]
    y_extend = (H.T @ y[n_start:n_end].T).T
    y_mask = np.ma.masked_where(y_extend == 0.0, y_extend)
    x2 = y_mask

    vmax = np.max(x1)
    vmin = np.min(x1)

    fig1, ax1 = plt.subplots(1, 2, figsize=(8, 4))
    im1 = ax1[0].imshow(x1, aspect=J / (n_end - n_start), vmax=vmax, vmin=vmin, origin="lower", interpolation="none")
    ax1[0].set_ylabel("obs steps $n$")
    ax1[0].set_xlabel("space $i$")
    ax1[0].set_title("x_true")

    ax1[1].imshow(x2, aspect=J / (n_end - n_start), vmax=vmax, vmin=vmin, origin="lower", interpolation="none")
    ax1[1].set_xlabel("space $i$")
    ax1[1].set_title("y_obs")
    ax1[1].set_yticks([])

    cax1 = fig1.add_axes([0.92, 0.155, 0.03, 0.675])
    fig1.colorbar(im1, cax=cax1)
    fig1.suptitle(f"True State & Observations: from t={n_start} to t={n_end}", fontsize=16)
    fig1.savefig(f"{data_dir}/spatio_temporal_true_obs")
    plt.show()

    fig2, axes2 = plt.subplots(num_methods, num_alphas, figsize=(8, 8 * num_methods / num_alphas))
    if num_methods == 1 and num_alphas == 1:
        axes2 = np.array([[axes2]])
    elif num_methods == 1:
        axes2 = axes2[np.newaxis, :]
    elif num_alphas == 1:
        axes2 = axes2[:, np.newaxis]

    for r_idx, m_name in enumerate(methods):
        paramname = f"m({m})-{m_name}-alpha({'='.join([str(alpha) for alpha in alpha_list])})-seeds({len(seed_list)})"
        Xa = np.load(f"{data_dir}/xa-{paramname}.npy")
        for a_idx, alpha_val in enumerate(alpha_list_vis_st):
            x_assim = Xa[a_idx, i_seed, :, k_ens]
            x3 = x_assim[n_start:n_end]

            ax = axes2[r_idx, a_idx]
            im2 = ax.imshow(x3, aspect=J / (n_end - n_start), vmax=vmax, vmin=vmin, origin="lower", interpolation="none")
            if r_idx == 0:
                ax.set_title(f"$\\alpha$={alpha_val}")
                ax.set_xticks([])
            elif r_idx == num_methods - 1:
                ax.set_xlabel("$i$")
            if a_idx == 0:
                ax.set_ylabel(m_name + "\n" + "$n$")
            else:
                ax.set_yticks([])
            if a_idx == num_alphas - 1:
                fig2.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    fig2.suptitle(f"Spatio-temporal Assimilations: from t={n_start} to t={n_end}", fontsize=16)
    fig2.tight_layout()
    fig2.savefig(f"{data_dir}/spatio_temporal_assimilations")
    plt.show()


if __name__ == "__main__":
    main()
