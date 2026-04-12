#!/usr/bin/env python3
"""Run the Lorenz96 DA experiment with helper functions for paramname and data loading.

This file factors paramname construction and data loading into helper functions and
separates plotting into an error-plot loop and an off-diagonal-ratio loop.
"""

import matplotlib.pyplot as plt
import numpy as np
from da.etkf import ETKF
from da.l96 import lorenz96
from da.po import PO
from da.scheme import rk4

import visualize  # provides get_linestyle_cycle, get_marker_cycle

plt.rcParams["text.usetex"] = False


def display_method(m):
    """Return a human-friendly display name for DA methods."""
    if m == "po_add":
        return "add"
    if m == "po_proj":
        return "add-proj"
    return m.replace("po_", "")


def make_paramname(m, method_name, alpha_list, seed_list_len):
    """Construct the paramname string used in file names.

    Example:
        m(10)-po_add-alpha(0.0=0.5=2.0)-seeds(20)
    """
    alpha_part = "=".join([str(alpha) for alpha in alpha_list])
    return f"m({m})-{method_name}-alpha({alpha_part})-seeds({seed_list_len})"


def try_load_xa(data_dir, paramname):
    """Try to load the xa file for a given paramname. Return ndarray or None."""
    path = f"{data_dir}/xa-{paramname}.npy"
    try:
        xa = np.load(path)
        print("Xa loaded:", path, xa.shape)
        return xa
    except Exception:
        return None


def load_xa(data_dir, paramname):
    """Load xa file and raise if missing (prints shape)."""
    path = f"{data_dir}/xa-{paramname}.npy"
    xa = np.load(path)
    print("Xa loaded:", path, xa.shape)
    return xa


def load_y(data_dir, num_seeds):
    """Load observations saved as y-seeds(NUM).npy."""
    path = f"{data_dir}/y-seeds({num_seeds}).npy"
    y = np.load(path)
    print("Y loaded:", path, y.shape)
    return y


def loss_sq(X, Y):
    """Squared difference summed over last axis. X, Y: (..., Nt, J) or broadcastable."""
    return np.sum((X - Y) ** 2, axis=-1)


def stats(X):
    """Return mean and std across first axis. X shape: (N_seed, Nt, ...)."""
    return X.mean(axis=0), X.std(axis=0)


def main():
    data_dir = "data/20260412v1"

    # ------------------------------------------
    # Parameters
    # ------------------------------------------
    # System
    J = 60
    F = 8
    dt = 0.01
    N0 = 20 * 360
    N = 20 * 50

    # Observation
    obs_per = 1
    r = 1.0

    # DA
    m = 10
    num_seeds = 20
    seed_list = np.arange(num_seeds)
    alpha_list = [0.0, 0.5, 2.0, 10.0, 100.0]
    methods = ["po_add", "po_proj"]

    # Visualization / eval
    per_vis = 20
    per_ticklabel = per_vis * 20
    N_end = 50
    i_seed = 0
    alpha_list_vis_st = [0.0, 0.5, 2.0]
    k_ens = 0
    n_start = 0
    n_end = N

    # ------------------------------------------
    # True trajectory (generate or load)
    # ------------------------------------------
    x0 = F * np.ones(J)
    x0[19] *= 1.001
    scheme = rk4
    p = (F,)

    try:
        x_true_full = np.load(f"{data_dir}/x_true_l96_full.npy")
        print("x_true loaded:", x_true_full.shape)
    except Exception:
        result = np.zeros((N0 + N, J))
        x = x0.copy()
        result[0] = x[:]
        for n in range(1, N0 + N):
            t = n * dt
            x = scheme(lorenz96, t, x, p, dt)
            result[n] = x[:]
        x_true_full = result[N0:]
        print("x_true generated:", x_true_full.shape)
        np.save(f"{data_dir}/x_true_l96_full", x_true_full)

    # normalized energy plot (after spin-up)
    norm = np.linalg.norm(x_true_full, axis=-1) / np.sqrt(J)
    tvec = np.arange(N) * dt
    fig3, ax3 = plt.subplots()
    ax3.grid(False)
    ax3.set_title("norm after spin-up")
    ax3.set_xlabel("$t$")
    ax3.set_ylabel("$ |u|/\\sqrt{J}$")
    ax3.set_ylim((0.0, 8.5))
    ax3.plot(tvec, norm, lw=0.5)
    fig3.tight_layout()

    # ------------------------------------------
    # DA setting
    # ------------------------------------------
    print("obs_per", obs_per, "steps")
    Dt = dt * obs_per

    def M(x, Dt_inner):
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

    R = r**2 * np.eye(J)
    R = H @ R @ H.T
    print("obs noise std:", r)

    Nt = N
    x_true = x_true_full[:Nt:obs_per]

    print("ensemble size m:", m)
    P0 = 4**2 * np.eye(J)

    # ------------------------------------------
    # Run DA (compute & save Xa, Y)
    # ------------------------------------------
    for method_name in methods:
        print("method:", method_name)
        paramname = make_paramname(m, method_name, alpha_list, len(seed_list))
        Xa = try_load_xa(data_dir, paramname)
        if Xa is None:
            Xa = np.zeros((len(alpha_list), len(seed_list), Nt, m, J))
            Y = np.zeros((len(seed_list), Nt, H.shape[0]))
            for j, seed in enumerate(seed_list):
                np.random.seed(int(seed))
                # observations with noise
                y = (H @ x_true.T).T
                y += np.random.multivariate_normal(
                    mean=np.zeros_like(y[0]), cov=R, size=len(y)
                )
                Y[j] = y[:]

                # initial ensemble
                X_0 = x_true[
                    np.random.randint(len(x_true))
                ] + np.random.multivariate_normal(
                    mean=np.zeros_like(x_true[0]), cov=P0, size=m
                )

                for i, alpha in enumerate(alpha_list):
                    print(" alpha:", alpha)
                    np.random.seed(int(seed))
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

    # ------------------------------------------
    # Plotting: separate loops
    #   - Loop A: error plots (fig1 / ax1)
    #   - Loop B: off-diagonal ratio plots (fig2 / ax2)
    # ------------------------------------------
    get_linestyle_cycle = visualize.get_linestyle_cycle
    get_marker_cycle = visualize.get_marker_cycle

    print(f"Visualize until {per_vis * N_end} steps")
    print("methods:", methods)

    # Error plot (MSE time series)
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    time_ticks = np.arange(Nt // per_vis) * per_vis + 1

    for k, method_name in enumerate(methods):
        paramname = make_paramname(m, method_name, alpha_list, len(seed_list))
        print(" loading param for errors:", paramname)
        Xa = load_xa(data_dir, paramname)

        color = colors[k % len(colors)]
        line_cycle = get_linestyle_cycle()
        marker_cycle = get_marker_cycle()

        for i, xa in enumerate(Xa):
            alpha = alpha_list[i]
            if i in [3, 4]:
                print(" skip alpha (errors):", alpha)
                continue

            ls = next(line_cycle)
            marker = next(marker_cycle)

            se_wtm = loss_sq(x_true[None, :, None, :] - xa, 0) + loss_sq(
                x_true[None, :, None, :] @ H.T - xa @ H.T, 0
            )
            se_tm, _ = stats(se_wtm)
            se_t = se_tm.mean(axis=1)
            se_wt = se_wtm.mean(axis=2)
            print(" t-averaged SE (errors)", se_tm[Nt // 2 :].mean())

            ax1.plot(
                time_ticks[:N_end],
                se_t[::per_vis][:N_end],
                label=f"{display_method(method_name)} $\\alpha$={alpha}",
                lw=0.5,
                ls=ls,
                color=color,
                marker=marker,
                ms=5,
            )
            for se_k in se_wt:
                ax1.plot(
                    time_ticks[:N_end],
                    se_k[::per_vis][:N_end],
                    lw=0.25,
                    color=color,
                    alpha=0.3,
                )

    # theoretical bound line
    ax1.plot(
        time_ticks[:N_end],
        4 * Ny * (r**2) * np.ones_like(time_ticks[:N_end]),
        label="$ 4 N_y r^2 $",
        lw=0.5,
        c="black",
    )

    ax1.set_xlabel("time step $n$")
    # ax1.set_title("The time series of $\\mathrm{MSE}$")
    ax1.set_ylabel(r"$ \frac{1}{m} \sum_{k=1}^m \mathbb{E} \|\delta^{(k)}\|^2 $")
    ax1.set_yscale("log")
    ax1.legend(bbox_to_anchor=(1.0, 0.85), loc="upper right", ncol=2)
    fig1.tight_layout()
    fig1.savefig(f"{data_dir}/l96-po-inflation_Pse.pdf", transparent=True)

    # Off-diagonal ratio plot
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    for k, method_name in enumerate(methods):
        paramname = make_paramname(m, method_name, alpha_list, len(seed_list))
        print(" loading param for off-diag:", paramname)
        Xa = load_xa(data_dir, paramname)

        color = colors[k % len(colors)]
        line_cycle = get_linestyle_cycle()
        marker_cycle = get_marker_cycle()

        for i, xa in enumerate(Xa):
            alpha = alpha_list[i]

            ls = next(line_cycle)
            marker = next(marker_cycle)

            Pi = H.T @ H
            Q = np.eye(J) - Pi
            dX = xa - xa.mean(axis=2, keepdims=True)
            P = (dX.swapaxes(-2, -1) @ dX) / (m - 1)
            QPHt = Q @ P @ Pi.T
            HPHt = Pi @ P @ Pi.T
            rf = (
                np.linalg.norm(QPHt, axis=(2, 3)) / np.linalg.norm(HPHt, axis=(2, 3))
            ).mean(axis=0)

            ax2.plot(
                time_ticks[:N_end],
                rf[::per_vis][:N_end],
                label=f"{display_method(method_name)} $\\alpha$={alpha}",
                lw=0.5,
                ls=ls,
                color=color,
                marker=marker,
                ms=5,
            )

    # ax2.set_title("The off-diagonal ratio in the covariance")
    ax2.set_ylabel(r"$\left|(I-\Pi)P\Pi\right|_F / \left|\Pi P \Pi\right|_F$")
    ax2.set_xlabel("time step $n$")
    ax2.set_ylim((0.0, 2.0))
    ax2.legend(bbox_to_anchor=(1.0, 1.0), loc="upper right", ncol=2)
    fig2.tight_layout()
    fig2.savefig(f"{data_dir}/l96-po-inflation_offDiag.pdf", transparent=True)

    # ------------------------------------------
    # Last covariance plots
    # ------------------------------------------
    num_methods = len(methods)
    num_alphas = len(alpha_list)

    observed_idx = np.where(np.diag(H.T @ H) > 0.5)[0]
    unobserved_idx = np.where(np.diag(H.T @ H) <= 0.5)[0]
    sort_idx = np.concatenate([observed_idx, unobserved_idx])
    vmax = 1.0
    vmin = -1.0

    fig1, axes1 = plt.subplots(
        num_methods,
        num_alphas,
        figsize=(8, 8 * num_methods / num_alphas),
        gridspec_kw={"hspace": 0.1, "wspace": 0.05},
    )
    fig2, axes2 = plt.subplots(
        num_methods, num_alphas, figsize=(8, 8 * num_methods / num_alphas)
    )

    if num_methods == 1 and num_alphas == 1:
        axes1 = np.array([[axes1]])
        axes2 = np.array([[axes2]])
    elif num_methods == 1:
        axes1 = axes1[np.newaxis, :]
        axes2 = axes2[np.newaxis, :]
    elif num_alphas == 1:
        axes1 = axes1[:, np.newaxis]
        axes2 = axes2[:, np.newaxis]

    # safe small placeholders for colorbars (will be replaced)
    first_ax1 = axes1.ravel()[0]
    first_ax2 = axes2.ravel()[0]
    im1 = first_ax1.imshow(
        np.zeros((1, 1)), cmap="coolwarm", vmax=vmax, vmin=vmin, interpolation="none"
    )
    im2 = first_ax2.imshow(
        np.zeros((1, 1)), cmap="coolwarm", vmax=vmax, vmin=vmin, interpolation="none"
    )

    for r_idx, method_name in enumerate(methods):
        paramname = make_paramname(m, method_name, alpha_list, len(seed_list))
        Xa = load_xa(data_dir, paramname)
        for c, alpha in enumerate(alpha_list):
            xa = Xa[c, i_seed]
            dX = xa - xa.mean(axis=1, keepdims=True)
            P = (dX.swapaxes(-2, -1) @ dX) / (m - 1)
            P_last = P[-1]
            P_last /= np.abs(P_last).max()

            ax1 = axes1[r_idx, c]
            im1 = ax1.imshow(
                P_last, cmap="coolwarm", interpolation="none", vmax=vmax, vmin=vmin
            )
            if r_idx == 0:
                ax1.set_title(f"$\\alpha$={alpha}")
            if c == 0:
                ax1.set_ylabel(display_method(method_name) + "\n" + "i")
            if r_idx == num_methods - 1:
                ax1.set_xlabel("j")
            ax1.set_xticks([])
            ax1.set_yticks([])

            P_rearranged = P_last[sort_idx][:, sort_idx]
            ax2 = axes2[r_idx, c]
            im2 = ax2.imshow(
                P_rearranged,
                cmap="coolwarm",
                interpolation="none",
                vmax=vmax,
                vmin=vmin,
            )
            if r_idx == 0:
                ax2.set_title(f"$\\alpha$={alpha}")
            if c == 0:
                ax2.set_ylabel(display_method(method_name) + "\n" + "i")
            if r_idx == num_methods - 1:
                ax2.set_xlabel("j")
            ax2.set_xticks([])
            ax2.set_yticks([])

    fig1.colorbar(im1, ax=axes1.ravel().tolist(), fraction=0.046, pad=0.04)
    fig2.colorbar(im2, ax=axes2.ravel().tolist(), fraction=0.046, pad=0.04)
    fig2.savefig(f"{data_dir}/normalized_last_covariance")

    # ------------------------------------------
    # Spatio-temporal: true & observations
    # ------------------------------------------
    print("Spatio-temporal Plot: State, Observations, Assimilations")
    x_true_full = np.load(f"{data_dir}/x_true_l96_full.npy")
    Y = load_y(data_dir, num_seeds)
    y = Y[i_seed]

    num_methods = len(methods)
    num_alphas = len(alpha_list_vis_st)

    x1 = x_true_full[n_start:n_end]
    y_extend = (H.T @ y[n_start:n_end].T).T
    y_mask = np.ma.masked_where(y_extend == 0.0, y_extend)
    x2 = y_mask

    vmax_ts = np.max(x1)
    vmin_ts = np.min(x1)

    fig1, ax1 = plt.subplots(1, 2, figsize=(8, 4))
    im1 = ax1[0].imshow(
        x1,
        aspect=J / (n_end - n_start),
        vmax=vmax_ts,
        vmin=vmin_ts,
        origin="lower",
        interpolation="none",
    )
    ax1[0].set_ylabel("time $n$")
    ax1[0].set_xlabel("space $i$")
    ax1[0].set_title("x_true")

    ax1[1].imshow(
        x2,
        aspect=J / (n_end - n_start),
        vmax=vmax_ts,
        vmin=vmin_ts,
        origin="lower",
        interpolation="none",
    )
    ax1[1].set_xlabel("space $i$")
    ax1[1].set_title("y_obs")
    ax1[1].set_yticks([])

    cax1 = fig1.add_axes((0.92, 0.155, 0.03, 0.675))
    fig1.colorbar(im1, cax=cax1)
    # fig1.suptitle(
    #     f"True State & Observations: from n={n_start + 1} to n={n_end}", fontsize=16
    # )
    fig1.savefig(f"{data_dir}/spatio_temporal_true_obs")

    # ------------------------------------------
    # Spatio-temporal: assimilations
    # ------------------------------------------
    fig2, axes2 = plt.subplots(
        num_methods,
        num_alphas,
        figsize=(8, 8 * num_methods / num_alphas),
        gridspec_kw={"hspace": 0.1, "wspace": 0.05},
    )
    if num_methods == 1 and num_alphas == 1:
        axes2 = np.array([[axes2]])
    elif num_methods == 1:
        axes2 = axes2[np.newaxis, :]
    elif num_alphas == 1:
        axes2 = axes2[:, np.newaxis]

    # placeholder for mappable
    im_spatio = None

    for r_idx, m_name in enumerate(methods):
        paramname = make_paramname(m, m_name, alpha_list, len(seed_list))
        Xa = load_xa(data_dir, paramname)
        for a_idx, alpha_val in enumerate(alpha_list_vis_st):
            x_assim = Xa[a_idx, i_seed, :, k_ens]
            x3 = x_assim[n_start:n_end]

            ax = axes2[r_idx, a_idx]
            im_spatio = ax.imshow(
                x3,
                aspect=J / (n_end - n_start),
                vmax=vmax_ts,
                vmin=vmin_ts,
                origin="lower",
                interpolation="none",
            )
            if r_idx == 0:
                ax.set_title(f"$\\alpha$={alpha_val}")
                ax.set_xticks([])
            elif r_idx == num_methods - 1:
                ax.set_xlabel("$i$")
                if a_idx == 0:
                    ax.set_xlabel("space $i$")
            if a_idx == 0:
                ax.set_ylabel(display_method(m_name) + "\n" + "$n$")
                if r_idx == num_methods - 1:
                    ax.set_ylabel(display_method(m_name) + "\n" + "time $n$")
            else:
                ax.set_yticks([])

    # fig2.suptitle(
    #     f"Spatio-temporal Assimilations: from n={n_start + 1} to n={n_end}", fontsize=16
    # )
    if im_spatio is not None:
        fig2.colorbar(im_spatio, ax=axes2.ravel().tolist(), fraction=0.046, pad=0.04)
    fig2.savefig(f"{data_dir}/spatio_temporal_assimilations")

    # ------------------------------------------
    # Spatio-temporal: absolute errors
    # ------------------------------------------
    print("Spatio-temporal Plot: Absolute Errors")
    x_true_full = np.load(f"{data_dir}/x_true_l96_full.npy")
    x_t = x_true_full[n_start:n_end]

    num_methods = len(methods)
    num_alphas = len(alpha_list_vis_st)

    # compute global vmax for consistent color scale
    global_vmax = 0.0
    for m_name in methods:
        paramname = make_paramname(m, m_name, alpha_list, len(seed_list))
        Xa = load_xa(data_dir, paramname)
        for a_idx in range(num_alphas):
            x_assim = Xa[a_idx, i_seed].mean(axis=1)
            e_assim = np.abs(x_assim[n_start:n_end] - x_t)
            global_vmax = max(global_vmax, np.max(e_assim))

    vmin_err = 0.0
    vmax_err = max(0.0, global_vmax)

    fig, axes = plt.subplots(
        num_methods,
        num_alphas,
        figsize=(8, 8 * num_methods / num_alphas),
        gridspec_kw={"hspace": 0.1, "wspace": 0.05},
    )
    if num_methods == 1 and num_alphas == 1:
        axes = np.array([[axes]])
    elif num_methods == 1:
        axes = axes[np.newaxis, :]
    elif num_alphas == 1:
        axes = axes[:, np.newaxis]

    im_err = None
    for r_idx, m_name in enumerate(methods):
        paramname = make_paramname(m, m_name, alpha_list, len(seed_list))
        Xa = load_xa(data_dir, paramname)
        for a_idx, alpha_val in enumerate(alpha_list_vis_st):
            x_assim = Xa[a_idx, i_seed, :, k_ens]
            e_assim = np.abs(x_assim[n_start:n_end] - x_t)

            ax = axes[r_idx, a_idx]
            im_err = ax.imshow(
                e_assim,
                aspect=J / (n_end - n_start),
                vmax=vmax_err,
                vmin=vmin_err,
                origin="lower",
                interpolation="none",
                cmap="flare_r",
            )
            if r_idx == 0:
                ax.set_title(f"$\\alpha$={alpha_val}")
                ax.set_xticks([])
            elif r_idx == num_methods - 1:
                ax.set_xlabel("$i$")
                if a_idx == 0:
                    ax.set_xlabel("space $i$")
            if a_idx == 0:
                ax.set_ylabel(display_method(m_name) + "\n" + "$n$")
                if r_idx == num_methods - 1:
                    ax.set_ylabel(display_method(m_name) + "\n" + "time $n$")
            else:
                ax.set_yticks([])

    if im_err is not None:
        fig.colorbar(im_err, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)

    # fig.suptitle(f"Absolute Errors: from n={n_start + 1} to n={n_end}", fontsize=16)
    fig.savefig(f"{data_dir}/absolute_errors_all")


if __name__ == "__main__":
    main()
