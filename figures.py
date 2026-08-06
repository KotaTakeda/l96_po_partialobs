"""Manuscript error metrics, uncertainty display convention, figures and style.

This module owns everything that defines how the paper *quantifies* and
*renders* the state-estimation error:

* the weighted error used throughout the manuscript,
* the seed-variability band (arithmetic mean with the empirical 2.5th--97.5th
  percentiles) drawn in every figure and reported in the summary tables,
* the plot style and the marker/line-style cycles, and
* the Figure 1--7 renderers.

:mod:`main` imports from this module, and this module imports nothing from the
project, so the dependency stays one-directional.
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np

# Imported for its side effect: seaborn registers the "flare" colormap family
# with matplotlib, which Figure 2 uses via cmap="flare_r".
import seaborn  # noqa: F401

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

# Resolved relative to the working directory, i.e. the repository root.
plt.style.use("vis.mplstyle")
plt.rcParams["text.usetex"] = False

MARKERS = [".", "*", "^", "v", "s", "h", "o", "x", "d", "p"]

# https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
LINESTYLES = ["-", "--", ":", "-."]


def get_marker_cycle():
    """Return an endless cycle of marker styles."""
    return itertools.cycle(MARKERS)


def get_linestyle_cycle():
    """Return an endless cycle of line styles."""
    return itertools.cycle(LINESTYLES)


EMPIRICAL_BAND_PERCENTILES = (2.5, 97.5)

WEIGHTED_ERROR_YLABEL = (
    r"$\frac{1}{m}\sum_{k=1}^m\mathbb{E}\left[\|\boldsymbol{\delta}_n^{(k)}\|^2\right]$"
)


def display_method(method_name):
    """Return the label used in the manuscript figures."""
    if method_name == "po_add":
        return "add"
    if method_name == "po_proj":
        return "add-proj"
    return method_name.replace("po_", "")


def loss_sq(x, y):
    """Squared Euclidean loss summed over the state dimension."""
    return np.sum((x - y) ** 2, axis=-1)


def mean_empirical_percentile_band(values, axis=0):
    """Return the mean and pointwise 2.5--97.5% empirical percentile band."""
    values = np.asarray(values, dtype=float)
    lower, upper = np.percentile(
        values,
        EMPIRICAL_BAND_PERCENTILES,
        axis=axis,
    )
    return values.mean(axis=axis), lower, upper


def member_mean_weighted_error(x_true, h, xa_seed):
    """Return the manuscript weighted error for one seed at every time.

    ``xa_seed`` has shape ``(time, member, state)``. This function is the
    single implementation of the error used by Figures 1 and 5 and by their
    table summaries.
    """
    delta = xa_seed - x_true[:, None, :]
    weighted_error = loss_sq(delta, 0.0) + loss_sq(delta @ h.T, 0.0)
    return weighted_error.mean(axis=1)


def weighted_error_per_seed(x_true, h, xa):
    """Return the manuscript weighted-error time series for every seed."""
    return np.stack(
        [member_mean_weighted_error(x_true, h, xa_seed) for xa_seed in xa],
        axis=0,
    )


def method_alpha_grid(num_methods, num_alphas):
    """Create a (method x alpha) grid of panels sharing the manuscript layout.

    Returns ``(fig, axes)`` where ``axes`` is always 2-D and indexed as
    ``axes[method_index, alpha_index]``, even when either dimension is 1.
    """
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
    return fig, axes


def add_grid_colorbar(fig, im, axes):
    """Attach a single colorbar spanning every panel of a grid figure."""
    if im is None:
        return
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)


def plot_fig1_mse(
    data_dir,
    x_true,
    h,
    r,
    alpha_list_all,
    alpha_list_fig,
    methods,
    xa_dict,
    show_sample_paths=False,
    per_vis=20,
    num_points=50,
):
    """Plot Figure 1 with a 2.5--97.5% seed range and optional paths."""
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    time_ticks = np.arange(len(x_true) // per_vis) * per_vis + 1
    ny = np.linalg.matrix_rank(h)
    alpha_index_map = {alpha: alpha_list_all.index(alpha) for alpha in alpha_list_fig}

    for k, method_name in enumerate(methods):
        xa = xa_dict[method_name]
        color = colors[k % len(colors)]
        line_cycle = get_linestyle_cycle()
        marker_cycle = get_marker_cycle()

        for alpha in alpha_list_fig:
            alpha_idx = alpha_index_map[alpha]
            xa_alpha = xa[alpha_idx]
            ls = next(line_cycle)
            marker = next(marker_cycle)

            delta_mean_per_seed = weighted_error_per_seed(x_true, h, xa_alpha)
            plot_times = time_ticks[:num_points]
            sampled = delta_mean_per_seed[:, ::per_vis][:, :num_points]
            plot_mean, band_low, band_high = mean_empirical_percentile_band(sampled)

            ax.fill_between(
                plot_times,
                band_low,
                band_high,
                color=color,
                alpha=0.15,
                linewidth=0.0,
                zorder=2,
            )
            if show_sample_paths:
                for seed_path in sampled:
                    ax.plot(
                        plot_times,
                        seed_path,
                        lw=0.2,
                        ls=ls,
                        color=color,
                        alpha=0.3,
                        zorder=1,
                    )

            ax.plot(
                plot_times,
                plot_mean,
                label=f"{display_method(method_name)} $\\alpha$={alpha}",
                lw=0.5,
                ls=ls,
                color=color,
                marker=marker,
                ms=5,
                zorder=3,
            )

    ax.plot(
        time_ticks[:num_points],
        4 * ny * (r**2) * np.ones_like(time_ticks[:num_points]),
        label="$ 4 N_y r^2 $",
        lw=0.5,
        c="black",
    )
    ax.set_xlabel("time step $n$", loc="center")
    ax.set_ylabel(WEIGHTED_ERROR_YLABEL)
    ax.set_yscale("log")
    ax.legend(bbox_to_anchor=(1.0, 0.85), loc="upper right", ncol=2)
    fig.tight_layout()
    ax.xaxis.set_label_coords(0.5, -0.11)
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

    fig, axes = method_alpha_grid(num_methods, num_alphas)

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
                ax.set_ylabel(display_method(method_name) + "\n" + "time step $n$")
            else:
                ax.set_yticks([])

    add_grid_colorbar(fig, im, axes)
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

    fig, axes = method_alpha_grid(num_methods, num_alphas)

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
                ax.set_ylabel(display_method(method_name) + "\n" + "$i$")
            if r_idx == num_methods - 1:
                ax.set_xlabel("$j$")
            ax.set_xticks([])
            ax.set_yticks([])

    add_grid_colorbar(fig, im, axes)
    fig.savefig(data_dir / "fig3_covariance.pdf", transparent=True)
    plt.close(fig)


def plot_fig4_offdiag_ratio(
    data_dir,
    h,
    alpha_list_all,
    methods,
    xa_dict,
    show_sample_paths=False,
    per_vis=20,
    num_points=50,
):
    """Plot Figure 4 with a 2.5--97.5% seed range and optional paths."""
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    time_ticks = (
        np.arange(next(iter(xa_dict.values())).shape[2] // per_vis) * per_vis + 1
    )
    pi = h.T @ h
    q = np.eye(pi.shape[0]) - pi

    for k, method_name in enumerate(methods):
        xa = xa_dict[method_name]
        color = colors[k % len(colors)]
        line_cycle = get_linestyle_cycle()
        marker_cycle = get_marker_cycle()

        for i, alpha in enumerate(alpha_list_all):
            ratios = []
            for xa_seed in xa[i]:  # xa_seed: (n_time, ensemble_size, state_dim)
                d_x = xa_seed - xa_seed.mean(axis=1, keepdims=True)
                p = (d_x.swapaxes(-2, -1) @ d_x) / (xa_seed.shape[1] - 1)
                num = np.linalg.norm(q @ p @ pi.T, axis=(-2, -1))  # (n_time,)
                den = np.linalg.norm(pi @ p @ pi.T, axis=(-2, -1))  # (n_time,)
                ratios.append(
                    np.divide(
                        num,
                        den,
                        out=np.full_like(num, np.nan),
                        where=den > 0.0,
                    )
                )
            ratio = np.asarray(ratios)  # (n_seeds, n_time)
            ls = next(line_cycle)
            marker = next(marker_cycle)
            sampled = ratio[:, ::per_vis][:, :num_points]
            plot_mean, band_low, band_high = mean_empirical_percentile_band(sampled)

            ax.fill_between(
                time_ticks[:num_points],
                band_low,
                band_high,
                color=color,
                alpha=0.15,
                linewidth=0.0,
                zorder=2,
            )
            if show_sample_paths:
                for seed_path in sampled:
                    ax.plot(
                        time_ticks[:num_points],
                        seed_path,
                        lw=0.2,
                        ls=ls,
                        color=color,
                        alpha=0.3,
                        zorder=1,
                    )

            ax.plot(
                time_ticks[:num_points],
                plot_mean,
                label=f"{display_method(method_name)} $\\alpha$={alpha}",
                lw=0.5,
                ls=ls,
                color=color,
                marker=marker,
                ms=5,
                zorder=3,
            )

    ax.set_ylabel(r"$\mathbb{E}[\mathcal{R}_n]$")
    ax.set_xlabel("time step $n$")
    ax.set_ylim((0.0, 2.0))
    ax.legend(bbox_to_anchor=(1.0, 1.0), loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(data_dir / "fig4_offdiag_ratio.pdf", transparent=True)
    plt.close(fig)


def plot_fig5_projection_comparison(
    pdf_path,
    ensemble_sizes,
    all_results,
    methods,
):
    """Plot Figure 5: two-panel covariance-projection comparison.

    Panel (a) shows the direct unobserved analysis-increment RMS, which is
    identically zero for ``add-proj`` by construction; panel (b) shows the
    per-seed paired ``add``/``add-proj`` weighted-error ratio.
    """
    xs = np.array(ensemble_sizes)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, (ax_increment, ax_ratio) = plt.subplots(
        1, 2, figsize=(9, 4), layout="constrained"
    )

    # Panel (a): ensemble-mean unobserved analysis increment RMS vs m.
    # add-proj is identically zero by construction; plot it explicitly at zero.
    ax = ax_increment
    method_markers = ("o", "s")
    for k, method in enumerate(methods):
        color = colors[k % len(colors)]
        means, band_lows, band_highs = [], [], []
        for x_m, m_res in zip(xs, all_results):
            if method == "po_proj":
                n_seeds = len(m_res["po_add"]["mean_increment_rms_unobserved"])
                vals = np.zeros(n_seeds)
            else:
                vals = m_res[method]["mean_increment_rms_unobserved"]
            ax.scatter(
                np.full(len(vals), x_m),
                vals,
                color=color,
                alpha=0.25,
                s=12,
                zorder=2,
            )
            mean, p2_5, p97_5 = mean_empirical_percentile_band(vals)
            means.append(mean)
            band_lows.append(p2_5)
            band_highs.append(p97_5)
        means = np.array(means)
        band_lows = np.array(band_lows)
        band_highs = np.array(band_highs)
        ax.errorbar(
            xs,
            means,
            yerr=np.vstack((means - band_lows, band_highs - means)),
            label=display_method(method),
            color=color,
            marker=method_markers[k],
            ms=5,
            lw=1.5,
            capsize=3,
            zorder=3,
            linestyle="none",
        )
    ax.set_xlabel("ensemble size $m$")
    ax.set_ylabel(
        r"$\left[\frac{1}{N_t}\sum_{n=1}^{N_t}"
        r"\left(I_n^{\mathrm{U}}\right)^2\right]^{1/2}$"
    )
    ax.set_yscale("linear")
    ax.set_ylim(bottom=0.0)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(int(v)) for v in xs])
    ax.legend()
    ax.set_title(r"(a) Analysis increment", loc="left")

    # Panel (b): per-seed paired weighted-error ratio vs m.
    ax = ax_ratio
    ratio_means, ratio_lows, ratio_highs = [], [], []
    for x_m, m_res in zip(xs, all_results):
        ratios = (
            m_res["po_add"]["weighted_norm_error"]
            / m_res["po_proj"]["weighted_norm_error"]
        )
        ax.scatter(
            np.full(len(ratios), x_m),
            ratios,
            color="black",
            alpha=0.25,
            s=12,
            zorder=2,
        )
        mean, p2_5, p97_5 = mean_empirical_percentile_band(ratios)
        ratio_means.append(mean)
        ratio_lows.append(p2_5)
        ratio_highs.append(p97_5)
    ratio_means = np.array(ratio_means)
    ratio_lows = np.array(ratio_lows)
    ratio_highs = np.array(ratio_highs)
    ax.errorbar(
        xs,
        ratio_means,
        yerr=np.vstack((ratio_means - ratio_lows, ratio_highs - ratio_means)),
        color="black",
        marker="o",
        ms=5,
        lw=1.5,
        capsize=3,
        zorder=3,
        linestyle="none",
    )
    ax.axhline(1.0, color="black", lw=0.8, ls="--", zorder=1)
    ax.set_xlabel("ensemble size $m$")
    ax.set_ylabel(
        r"$\sum_{n=1}^{N_t}\sum_{k=1}^{m}"
        r"\|\boldsymbol{\delta}_n^{(k),\mathrm{add}}\|^2"
        r"/\sum_{n=1}^{N_t}\sum_{k=1}^{m}"
        r"\|\boldsymbol{\delta}_n^{(k),\mathrm{add-proj}}\|^2$"
    )
    ax.set_yscale("linear")
    ax.set_ylim(bottom=0.0)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(int(v)) for v in xs])
    ax.set_title(r"(b) MSE ratio (add/add-proj)", loc="left")

    fig.savefig(pdf_path, transparent=True)
    plt.close(fig)
    print("saved:", pdf_path)


def plot_fig6_true_obs(data_dir, x_true, h, observations, sample_seed_index):
    """Plot Figure 6: true state and observations."""
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
    axes[0].set_ylabel("time step $n$")
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
    fig.savefig(data_dir / "fig6_true_obs.pdf", transparent=True)
    plt.close(fig)


def plot_fig7_analysis_states(
    data_dir,
    x_true,
    alpha_list_all,
    alpha_list_fig,
    methods,
    xa_dict,
    sample_seed_index,
    sample_member_index,
):
    """Plot Figure 7: sample-path analysis states."""
    num_methods = len(methods)
    num_alphas = len(alpha_list_fig)
    state_dim = x_true.shape[1]
    vmax = np.max(x_true)
    vmin = np.min(x_true)

    fig, axes = method_alpha_grid(num_methods, num_alphas)

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
                ax.set_ylabel(display_method(method_name) + "\n" + "time step $n$")
            else:
                ax.set_yticks([])

    add_grid_colorbar(fig, im, axes)
    fig.savefig(data_dir / "fig7_analysis_states.pdf", transparent=True)
    plt.close(fig)
