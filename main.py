#!/usr/bin/env python3
"""Reproduce the Lorenz 96 PO experiments reported in the manuscript.

The script runs the perturbed-observation (PO) filter with additive inflation
(`po_add`) and projected additive inflation (`po_proj`) for the partially
observed Lorenz 96 model, then saves the data products and figures used in the
paper.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from da.l96 import lorenz96
from da.po import PO
from da.scheme import rk4

import visualize

plt.rcParams["text.usetex"] = False


def display_method(method_name):
    """Return the label used in the manuscript figures."""
    if method_name == "po_add":
        return "add"
    if method_name == "po_proj":
        return "add-proj"
    return method_name.replace("po_", "")


def true_trajectory_path(data_dir):
    """Return the canonical cache path for the true trajectory."""
    return data_dir / "true_trajectory.npy"


def observations_path(data_dir):
    """Return the canonical cache path for observations."""
    return data_dir / "observations.npy"


def analysis_ensembles_path(data_dir, method_name):
    """Return the canonical cache path for analysis ensembles."""
    return data_dir / method_name / "analysis_ensembles.npy"


def parameters_path(data_dir):
    """Return the canonical path for run metadata."""
    return data_dir / "run_parameters.json"


def load_npy(path):
    """Load an ``.npy`` file and print its shape for traceability."""
    array = np.load(path)
    print("loaded:", path, array.shape)
    return array


def maybe_load_npy(path):
    """Load an ``.npy`` file if it exists."""
    if path.exists():
        return load_npy(path)
    return None


def save_run_parameters(data_dir, parameters):
    """Save experiment parameters as JSON."""
    path = parameters_path(data_dir)
    with path.open("w", encoding="utf-8") as f:
        json.dump(parameters, f, indent=2)
        f.write("\n")
    print("saved:", path)


def loss_sq(x, y):
    """Squared Euclidean loss summed over the state dimension."""
    return np.sum((x - y) ** 2, axis=-1)


def stats(x):
    """Return mean and std over the first axis."""
    return x.mean(axis=0), x.std(axis=0)


def build_observation_operator(state_dim):
    """Return the partial observation operator from Definition 2.1."""
    h_diag = np.ones(state_dim)
    h_diag[2::3] = 0
    h = np.diag(h_diag)
    return h[h_diag != 0]


def generate_true_trajectory(data_dir, x0, forcing, dt, spinup_steps, num_steps):
    """Load or generate the true Lorenz 96 trajectory after spin-up."""
    path = true_trajectory_path(data_dir)
    x_true = maybe_load_npy(path)
    if x_true is not None:
        return x_true

    scheme = rk4
    params = (forcing,)
    result = np.zeros((spinup_steps + num_steps, len(x0)))
    x = x0.copy()
    result[0] = x
    for n in range(1, spinup_steps + num_steps):
        x = scheme(lorenz96, n * dt, x, params, dt)
        result[n] = x

    x_true = result[spinup_steps:]
    np.save(path, x_true)
    print("generated:", path, x_true.shape)
    return x_true


def generate_observations_and_initial_ensembles(x_true, h, r, ensemble_size, seed_list):
    """Generate observations and initial ensembles for each seed.

    The initial ensemble follows the manuscript description: choose
    ``ensemble_size`` points from the attractor and perturb each by Gaussian
    noise with covariance ``16 I``.
    """
    obs_dim = h.shape[0]
    state_dim = x_true.shape[1]
    observation_cov = (r**2) * np.eye(obs_dim)
    initial_cov = (4**2) * np.eye(state_dim)

    observations = np.zeros((len(seed_list), len(x_true), obs_dim))
    initial_ensembles = np.zeros((len(seed_list), ensemble_size, state_dim))
    projected_truth = (h @ x_true.T).T

    for j, seed in enumerate(seed_list):
        rng = np.random.RandomState(int(seed))
        observations[j] = projected_truth + rng.multivariate_normal(
            mean=np.zeros(obs_dim),
            cov=observation_cov,
            size=len(x_true),
        )

        attractor_index = rng.randint(len(x_true))
        initial_ensembles[j] = x_true[attractor_index] + rng.multivariate_normal(
            mean=np.zeros(state_dim),
            cov=initial_cov,
            size=ensemble_size,
        )

    return observations, initial_ensembles


def run_assimilation(
    data_dir,
    x_true,
    h,
    forcing,
    r,
    dt,
    obs_per,
    ensemble_size,
    alpha_list,
    methods,
    seed_list,
    recompute=False,
):
    """Run the PO experiments and cache analysis ensembles and observations."""
    y_path = observations_path(data_dir)
    obs_dim = h.shape[0]
    state_dim = x_true.shape[1]
    observation_cov = (r**2) * np.eye(obs_dim)
    observations = None
    initial_ensembles = None

    if not recompute:
        observations = maybe_load_npy(y_path)

    if observations is None:
        observations, initial_ensembles = generate_observations_and_initial_ensembles(
            x_true=x_true,
            h=h,
            r=r,
            ensemble_size=ensemble_size,
            seed_list=seed_list,
        )
        np.save(y_path, observations)
        print("saved:", y_path, observations.shape)

    if initial_ensembles is None:
        _, initial_ensembles = generate_observations_and_initial_ensembles(
            x_true=x_true,
            h=h,
            r=r,
            ensemble_size=ensemble_size,
            seed_list=seed_list,
        )

    dt_obs = dt * obs_per
    params = (forcing,)

    def model_step(x, _dt_obs):
        for _ in range(obs_per):
            x = rk4(lorenz96, 0.0, x, params, dt)
        return x

    xa_dict = {}
    for method_name in methods:
        xa_path = analysis_ensembles_path(data_dir=data_dir, method_name=method_name)
        xa_path.parent.mkdir(parents=True, exist_ok=True)
        xa = None if recompute else maybe_load_npy(xa_path)
        if xa is None:
            xa = np.zeros(
                (len(alpha_list), len(seed_list), len(x_true), ensemble_size, state_dim)
            )
            for j, seed in enumerate(seed_list):
                y = observations[j]
                x0_ensemble = initial_ensembles[j]
                for i, alpha in enumerate(alpha_list):
                    np.random.seed(int(seed))
                    if method_name == "po_add":
                        da_instance = PO(
                            model_step,
                            h,
                            observation_cov,
                            alpha=alpha**2,
                            store_ensemble=True,
                            additive_inflation=True,
                        )
                    elif method_name == "po_proj":
                        da_instance = PO(
                            model_step,
                            h,
                            observation_cov,
                            alpha=alpha**2,
                            store_ensemble=True,
                            additive_inflation=True,
                            project_cov=True,
                        )
                    else:
                        raise ValueError(f"Unknown method: {method_name}")

                    da_instance.initialize(x0_ensemble.copy())
                    for y_obs in y:
                        da_instance.forecast(dt_obs)
                        da_instance.update(y_obs)
                    xa[i, j] = da_instance.Xa

            np.save(xa_path, xa)
            print("saved:", xa_path, xa.shape)

        xa_dict[method_name] = xa

    return observations, xa_dict


def plot_fig1_mse(
    data_dir,
    x_true,
    h,
    r,
    alpha_list_all,
    alpha_list_fig,
    methods,
    xa_dict,
    per_vis=20,
    num_points=50,
):
    """Plot Figure 1: MSE time series."""
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    time_ticks = np.arange(len(x_true) // per_vis) * per_vis + 1
    ny = np.linalg.matrix_rank(h)
    alpha_index_map = {alpha: alpha_list_all.index(alpha) for alpha in alpha_list_fig}

    for k, method_name in enumerate(methods):
        xa = xa_dict[method_name]
        color = colors[k % len(colors)]
        line_cycle = visualize.get_linestyle_cycle()
        marker_cycle = visualize.get_marker_cycle()

        for alpha in alpha_list_fig:
            alpha_idx = alpha_index_map[alpha]
            xa_alpha = xa[alpha_idx]
            ls = next(line_cycle)
            marker = next(marker_cycle)

            delta_sq = loss_sq(x_true[None, :, None, :] - xa_alpha, 0.0) + loss_sq(
                x_true[None, :, None, :] @ h.T - xa_alpha @ h.T,
                0.0,
            )
            delta_mean_t, _ = stats(delta_sq)
            delta_mean = delta_mean_t.mean(axis=1)
            delta_mean_per_seed = delta_sq.mean(axis=2)

            ax.plot(
                time_ticks[:num_points],
                delta_mean[::per_vis][:num_points],
                label=f"{display_method(method_name)} $\\alpha$={alpha}",
                lw=0.5,
                ls=ls,
                color=color,
                marker=marker,
                ms=5,
            )
            for delta_seed in delta_mean_per_seed:
                ax.plot(
                    time_ticks[:num_points],
                    delta_seed[::per_vis][:num_points],
                    lw=0.25,
                    color=color,
                    alpha=0.3,
                )

    ax.plot(
        time_ticks[:num_points],
        4 * ny * (r**2) * np.ones_like(time_ticks[:num_points]),
        label="$ 4 N_y r^2 $",
        lw=0.5,
        c="black",
    )
    ax.set_xlabel("time step $n$")
    ax.set_ylabel(r"$ \frac{1}{m} \sum_{k=1}^m \mathbb{E} \|\delta^{(k)}\|^2 $")
    ax.set_yscale("log")
    ax.legend(bbox_to_anchor=(1.0, 0.85), loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(data_dir / "fig1_mse.pdf", transparent=True)
    plt.close(fig)


def plot_fig2_abs_error(
    data_dir,
    x_true,
    alpha_list_all,
    alpha_list_fig,
    methods,
    xa_dict,
    sample_seed_index,
    sample_member_index,
):
    """Plot Figure 2: spatio-temporal absolute error of one ensemble member."""
    num_methods = len(methods)
    num_alphas = len(alpha_list_fig)
    state_dim = x_true.shape[1]

    global_vmax = 0.0
    for method_name in methods:
        xa = xa_dict[method_name]
        for alpha in alpha_list_fig:
            alpha_idx = alpha_list_all.index(alpha)
            x_assim = xa[alpha_idx, sample_seed_index, :, sample_member_index]
            e_assim = np.abs(x_assim - x_true)
            global_vmax = max(global_vmax, np.max(e_assim))

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

    im = None
    for r_idx, method_name in enumerate(methods):
        xa = xa_dict[method_name]
        for a_idx, alpha in enumerate(alpha_list_fig):
            alpha_idx = alpha_list_all.index(alpha)
            x_assim = xa[alpha_idx, sample_seed_index, :, sample_member_index]
            e_assim = np.abs(x_assim - x_true)

            ax = axes[r_idx, a_idx]
            im = ax.imshow(
                e_assim,
                aspect=state_dim / len(x_true),
                vmax=global_vmax,
                vmin=0.0,
                origin="lower",
                interpolation="none",
                cmap="flare_r",
            )
            if r_idx == 0:
                ax.set_title(f"$\\alpha$={alpha}")
                ax.set_xticks([])
            elif r_idx == num_methods - 1:
                ax.set_xlabel("space $i$")
            if a_idx == 0:
                ax.set_ylabel(display_method(method_name) + "\n" + "time $n$")
            else:
                ax.set_yticks([])

    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
    fig.savefig(data_dir / "fig2_abs_error.pdf", transparent=True)
    plt.close(fig)


def plot_fig3_covariance(
    data_dir,
    h,
    alpha_list_all,
    methods,
    xa_dict,
    sample_seed_index,
):
    """Plot Figure 3: normalized and rearranged covariance matrices."""
    num_methods = len(methods)
    num_alphas = len(alpha_list_all)
    observed_idx = np.where(np.diag(h.T @ h) > 0.5)[0]
    unobserved_idx = np.where(np.diag(h.T @ h) <= 0.5)[0]
    sort_idx = np.concatenate([observed_idx, unobserved_idx])
    vmax = 1.0
    vmin = -1.0

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

    im = axes.ravel()[0].imshow(
        np.zeros((1, 1)), cmap="coolwarm", vmax=vmax, vmin=vmin, interpolation="none"
    )
    for r_idx, method_name in enumerate(methods):
        xa = xa_dict[method_name]
        for c_idx, alpha in enumerate(alpha_list_all):
            xa_sample = xa[c_idx, sample_seed_index]
            d_x = xa_sample - xa_sample.mean(axis=1, keepdims=True)
            p = (d_x.swapaxes(-2, -1) @ d_x) / (xa_sample.shape[1] - 1)
            p_last = p[-1]
            scale = np.abs(p_last).max()
            if scale > 0.0:
                p_last = p_last / scale
            p_rearranged = p_last[sort_idx][:, sort_idx]

            ax = axes[r_idx, c_idx]
            im = ax.imshow(
                p_rearranged,
                cmap="coolwarm",
                interpolation="none",
                vmax=vmax,
                vmin=vmin,
            )
            if r_idx == 0:
                ax.set_title(f"$\\alpha$={alpha}")
            if c_idx == 0:
                ax.set_ylabel(display_method(method_name) + "\n" + "i")
            if r_idx == num_methods - 1:
                ax.set_xlabel("j")
            ax.set_xticks([])
            ax.set_yticks([])

    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
    fig.savefig(data_dir / "fig3_covariance.pdf", transparent=True)
    plt.close(fig)


def plot_fig4_offdiag_ratio(
    data_dir,
    h,
    alpha_list_all,
    methods,
    xa_dict,
    sample_seed_index,
    per_vis=20,
    num_points=50,
):
    """Plot Figure 4: off-diagonal/observed covariance ratio."""
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    time_ticks = np.arange(next(iter(xa_dict.values())).shape[2] // per_vis) * per_vis + 1
    pi = h.T @ h
    q = np.eye(pi.shape[0]) - pi

    for k, method_name in enumerate(methods):
        xa = xa_dict[method_name]
        color = colors[k % len(colors)]
        line_cycle = visualize.get_linestyle_cycle()
        marker_cycle = visualize.get_marker_cycle()

        for i, alpha in enumerate(alpha_list_all):
            xa_sample = xa[i, sample_seed_index]
            d_x = xa_sample - xa_sample.mean(axis=1, keepdims=True)
            p = (d_x.swapaxes(-2, -1) @ d_x) / (xa_sample.shape[1] - 1)
            num = np.linalg.norm(q @ p @ pi.T, axis=(1, 2))
            den = np.linalg.norm(pi @ p @ pi.T, axis=(1, 2))
            ratio = np.divide(
                num,
                den,
                out=np.full_like(num, np.nan),
                where=den > 0.0,
            )

            ax.plot(
                time_ticks[:num_points],
                ratio[::per_vis][:num_points],
                label=f"{display_method(method_name)} $\\alpha$={alpha}",
                lw=0.5,
                ls=next(line_cycle),
                color=color,
                marker=next(marker_cycle),
                ms=5,
            )

    ax.set_ylabel(r"$\left|(I-\Pi)P_n\Pi\right|_F / \left|\Pi P_n \Pi\right|_F$")
    ax.set_xlabel("time step $n$")
    ax.set_ylim((0.0, 2.0))
    ax.legend(bbox_to_anchor=(1.0, 1.0), loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(data_dir / "fig4_offdiag_ratio.pdf", transparent=True)
    plt.close(fig)


def plot_fig5_true_obs(data_dir, x_true, h, observations, sample_seed_index):
    """Plot Figure 5: true state and observations."""
    state_dim = x_true.shape[1]
    y = observations[sample_seed_index]
    y_extended = (h.T @ y.T).T
    y_mask = np.ma.masked_where(y_extended == 0.0, y_extended)
    vmax = np.max(x_true)
    vmin = np.min(x_true)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    im = axes[0].imshow(
        x_true,
        aspect=state_dim / len(x_true),
        vmax=vmax,
        vmin=vmin,
        origin="lower",
        interpolation="none",
    )
    axes[0].set_ylabel("time $n$")
    axes[0].set_xlabel("space $i$")
    axes[0].set_title("true state")

    axes[1].imshow(
        y_mask,
        aspect=state_dim / len(x_true),
        vmax=vmax,
        vmin=vmin,
        origin="lower",
        interpolation="none",
    )
    axes[1].set_xlabel("space $i$")
    axes[1].set_title("observation")
    axes[1].set_yticks([])

    cax = fig.add_axes((0.92, 0.155, 0.03, 0.675))
    fig.colorbar(im, cax=cax)
    fig.savefig(data_dir / "fig5_true_obs.pdf", transparent=True)
    plt.close(fig)


def plot_fig6_analysis_states(
    data_dir,
    x_true,
    alpha_list_all,
    alpha_list_fig,
    methods,
    xa_dict,
    sample_seed_index,
    sample_member_index,
):
    """Plot Figure 6: sample-path analysis states."""
    num_methods = len(methods)
    num_alphas = len(alpha_list_fig)
    state_dim = x_true.shape[1]
    vmax = np.max(x_true)
    vmin = np.min(x_true)

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

    im = None
    for r_idx, method_name in enumerate(methods):
        xa = xa_dict[method_name]
        for a_idx, alpha in enumerate(alpha_list_fig):
            alpha_idx = alpha_list_all.index(alpha)
            x_assim = xa[alpha_idx, sample_seed_index, :, sample_member_index]
            ax = axes[r_idx, a_idx]
            im = ax.imshow(
                x_assim,
                aspect=state_dim / len(x_true),
                vmax=vmax,
                vmin=vmin,
                origin="lower",
                interpolation="none",
            )
            if r_idx == 0:
                ax.set_title(f"$\\alpha$={alpha}")
                ax.set_xticks([])
            elif r_idx == num_methods - 1:
                ax.set_xlabel("space $i$")
            if a_idx == 0:
                ax.set_ylabel(display_method(method_name) + "\n" + "time $n$")
            else:
                ax.set_yticks([])

    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
    fig.savefig(data_dir / "fig6_analysis_states.pdf", transparent=True)
    plt.close(fig)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the Lorenz 96 PO experiments and generate manuscript figures."
    )
    parser.add_argument(
        "--data-dir",
        default="data/reproduce",
        help="Directory used for cached arrays and generated figures.",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Regenerate cached arrays even if they already exist.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    state_dim = 60
    forcing = 8.0
    dt = 0.01
    spinup_steps = 20 * 360
    num_steps = 20 * 50

    obs_per = 1
    obs_noise_std = 1.0

    ensemble_size = 10
    seed_list = np.arange(20)
    alpha_list_all = [0.0, 0.5, 2.0, 10.0, 100.0]
    alpha_list_fig = [0.0, 0.5, 2.0]
    methods = ["po_add", "po_proj"]

    sample_seed_index = 0
    sample_member_index = 0
    # The manuscript uses k = 1 for the sample-path plots; Python indexing is 0-based.

    save_run_parameters(
        data_dir=data_dir,
        parameters={
            "state_dim": state_dim,
            "forcing": forcing,
            "dt": dt,
            "spinup_steps": spinup_steps,
            "num_steps": num_steps,
            "obs_per": obs_per,
            "obs_noise_std": obs_noise_std,
            "ensemble_size": ensemble_size,
            "seed_list": seed_list.tolist(),
            "alpha_list_all": alpha_list_all,
            "alpha_list_fig": alpha_list_fig,
            "methods": methods,
            "sample_seed_index": sample_seed_index,
            "sample_member_index": sample_member_index,
            "recompute": args.recompute,
        },
    )

    x0 = forcing * np.ones(state_dim)
    x0[19] *= 1.001
    x_true = generate_true_trajectory(
        data_dir=data_dir,
        x0=x0,
        forcing=forcing,
        dt=dt,
        spinup_steps=spinup_steps,
        num_steps=num_steps,
    )

    h = build_observation_operator(state_dim)
    print("H.shape:", h.shape)
    print("rank(H):", np.linalg.matrix_rank(h))

    observations, xa_dict = run_assimilation(
        data_dir=data_dir,
        x_true=x_true,
        h=h,
        forcing=forcing,
        r=obs_noise_std,
        dt=dt,
        obs_per=obs_per,
        ensemble_size=ensemble_size,
        alpha_list=alpha_list_all,
        methods=methods,
        seed_list=seed_list,
        recompute=args.recompute,
    )

    plot_fig1_mse(
        data_dir=data_dir,
        x_true=x_true,
        h=h,
        r=obs_noise_std,
        alpha_list_all=alpha_list_all,
        alpha_list_fig=alpha_list_fig,
        methods=methods,
        xa_dict=xa_dict,
    )
    plot_fig2_abs_error(
        data_dir=data_dir,
        x_true=x_true,
        alpha_list_all=alpha_list_all,
        alpha_list_fig=alpha_list_fig,
        methods=methods,
        xa_dict=xa_dict,
        sample_seed_index=sample_seed_index,
        sample_member_index=sample_member_index,
    )
    plot_fig3_covariance(
        data_dir=data_dir,
        h=h,
        alpha_list_all=alpha_list_all,
        methods=methods,
        xa_dict=xa_dict,
        sample_seed_index=sample_seed_index,
    )
    plot_fig4_offdiag_ratio(
        data_dir=data_dir,
        h=h,
        alpha_list_all=alpha_list_all,
        methods=methods,
        xa_dict=xa_dict,
        sample_seed_index=sample_seed_index,
    )
    plot_fig5_true_obs(
        data_dir=data_dir,
        x_true=x_true,
        h=h,
        observations=observations,
        sample_seed_index=sample_seed_index,
    )
    plot_fig6_analysis_states(
        data_dir=data_dir,
        x_true=x_true,
        alpha_list_all=alpha_list_all,
        alpha_list_fig=alpha_list_fig,
        methods=methods,
        xa_dict=xa_dict,
        sample_seed_index=sample_seed_index,
        sample_member_index=sample_member_index,
    )


if __name__ == "__main__":
    main()
