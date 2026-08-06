"""Reproduce the Lorenz 96 PO experiments reported in the manuscript.

The script runs the perturbed-observation (PO) filter with additive inflation
(`po_add`) and projected additive inflation (`po_proj`) for the partially
observed Lorenz 96 model, then saves the data products and figures used in the
paper.
"""

import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
from da.l96 import lorenz96
from da.po import PO
from da.scheme import rk4

from figures import (
    display_method,
    mean_empirical_percentile_band,
    plot_fig1_mse,
    plot_fig2_abs_error,
    plot_fig3_covariance,
    plot_fig4_offdiag_ratio,
    plot_fig5_projection_comparison,
    plot_fig6_true_obs,
    plot_fig7_analysis_states,
    weighted_error_per_seed,
)

FIG1_SUMMARY_BASENAME = "fig1_summary.csv"
FIG5_SUMMARY_BASENAME = "fig5_summary.csv"
TIME_AVERAGE_START = 501


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


def initial_ensembles_path(data_dir):
    """Return the canonical cache path for initial ensembles."""
    return data_dir / "initial_ensembles.npy"


def _complete_marker_path(data_dir):
    """Return the path of the bundle-completion marker for data_dir."""
    return data_dir / ".complete"


def load_npy(path):
    """Load an ``.npy`` file and print its shape for traceability."""
    array = np.load(path)
    print("loaded:", path, array.shape)
    return array


def save_run_parameters(data_dir, parameters):
    """Save experiment parameters as JSON."""
    path = parameters_path(data_dir)
    with path.open("w", encoding="utf-8") as f:
        json.dump(parameters, f, indent=2)
        f.write("\n")
    print("saved:", path)






def time_average_per_seed(values):
    """Average seed-wise time series over analysis steps 501--1000."""
    return np.asarray(values)[:, TIME_AVERAGE_START - 1 :].mean(axis=1)


def time_average_per_seed_all(values):
    """Average seed-wise time series over all available analysis steps (n=1,...,N).

    Used exclusively for Figure 5 / projection-comparison aggregation so that
    Figure 1 summary (which uses time_average_per_seed) stays isolated.
    """
    return np.asarray(values).mean(axis=1)


def write_fig1_summary_csv(
    data_dir,
    x_true,
    h,
    r,
    alpha_list_all,
    alpha_list_fig,
    methods,
    xa_dict,
):
    """Write table-ready time-aggregated statistics corresponding to Figure 1.

    The Figure 1 weighted error is first averaged over analysis times from
    ``TIME_AVERAGE_START`` onward within each seed. The CSV then reports
    the arithmetic mean and empirical 2.5th--97.5th percentiles of those
    seed-wise time averages.
    """
    rows = []
    alpha_index_map = {alpha: alpha_list_all.index(alpha) for alpha in alpha_list_fig}
    for method_name in methods:
        for alpha in alpha_list_fig:
            alpha_idx = alpha_index_map[alpha]
            values = weighted_error_per_seed(
                x_true,
                h,
                xa_dict[method_name][alpha_idx],
            )
            seed_time_means = time_average_per_seed(values)
            mean, p2_5, p97_5 = mean_empirical_percentile_band(seed_time_means)
            rows.append(
                {
                    "method": display_method(method_name),
                    "alpha": alpha,
                    "mse": mean,
                    "p2_5": p2_5,
                    "p97_5": p97_5,
                    "num_seeds": values.shape[0],
                    "analysis_time_start": TIME_AVERAGE_START,
                    "analysis_time_end": values.shape[1],
                    "reference_4_Ny_r2": 4 * np.linalg.matrix_rank(h) * r**2,
                }
            )

    path = data_dir / FIG1_SUMMARY_BASENAME
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("saved:", path)


def build_observation_operator(state_dim, pattern="two_of_three"):
    """Return the partial observation operator from Definition 2.1."""
    _unobserved = {
        "two_of_three": slice(2, None, 3),
        "alternating": slice(1, None, 2),
    }
    h_diag = np.ones(state_dim)
    h_diag[_unobserved[pattern]] = 0
    h = np.diag(h_diag)
    return h[h_diag != 0]


def compute_diagnostics(Xf, Xa, unobserved_idx):
    """Return the direct unobserved analysis-increment time series for one run.

    Parameters
    ----------
    Xf, Xa : (time, member, state)
    unobserved_idx : 1-D index array

    Returns
    -------
    dict with one key of shape (time,).
    """
    xa_mean = Xa.mean(axis=1)  # (time, state)
    xf_mean = Xf.mean(axis=1)  # (time, state)

    mean_increment_sq_unobserved = (
        (xa_mean[:, unobserved_idx] - xf_mean[:, unobserved_idx]) ** 2
    ).mean(axis=-1)

    return {
        "mean_increment_sq_unobserved": mean_increment_sq_unobserved,
    }


def generate_true_trajectory(
    data_dir, x0, forcing, dt, spinup_steps, num_steps, recompute=False
):
    """Load or generate the true Lorenz 96 trajectory after spin-up."""
    path = true_trajectory_path(data_dir)
    if not recompute:
        return load_npy(path)
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
    diagnostics_path=None,
    unobserved_idx=None,
    load_analysis_ensembles=True,
):
    """Run the PO experiments and cache analysis ensembles and observations."""
    y_path = observations_path(data_dir)
    obs_dim = h.shape[0]
    state_dim = x_true.shape[1]
    observation_cov = (r**2) * np.eye(obs_dim)

    if diagnostics_path is not None:
        diagnostics_path = Path(diagnostics_path)
        unobserved_idx = np.asarray(unobserved_idx)

    ie_path = initial_ensembles_path(data_dir)

    if not recompute:
        observations = load_npy(y_path)
        initial_ensembles = load_npy(ie_path)
    else:
        observations, initial_ensembles = generate_observations_and_initial_ensembles(
            x_true=x_true,
            h=h,
            r=r,
            ensemble_size=ensemble_size,
            seed_list=seed_list,
        )
        np.save(y_path, observations)
        print("saved:", y_path, observations.shape)
        np.save(ie_path, initial_ensembles)
        print("saved:", ie_path, initial_ensembles.shape)

    if (
        diagnostics_path is not None
        and not recompute
        and not load_analysis_ensembles
        and diagnostics_path.exists()
    ):
        return observations, {}

    dt_obs = dt * obs_per
    params = (forcing,)

    def model_step(x, _dt_obs):
        for _ in range(obs_per):
            x = rk4(lorenz96, 0.0, x, params, dt)
        return x

    n_methods = len(methods)
    n_alphas = len(alpha_list)
    n_seeds = len(seed_list)
    n_time = len(x_true)

    _diag_shape = (n_methods, n_alphas, n_seeds, n_time)
    _residual_shape = (n_methods, n_alphas, n_seeds)

    _need_compute_diag = diagnostics_path is not None and recompute
    _force_recompute_xa = _need_compute_diag
    if diagnostics_path is not None and not _need_compute_diag:
        print("loaded diagnostics:", diagnostics_path)

    _diag_arrays = None
    _member_increment_max_abs_unobserved = None
    if diagnostics_path is not None and _need_compute_diag:
        _diag_arrays = {k: np.full(_diag_shape, np.nan) for k in DIAG_KEYS}
        _member_increment_max_abs_unobserved = np.full(_residual_shape, np.nan)

    _method_extra_kwargs = {
        "po_add": {},
        "po_proj": {"project_cov": True},
    }

    xa_dict = {}
    for m_idx, method_name in enumerate(methods):
        xa_path = analysis_ensembles_path(data_dir=data_dir, method_name=method_name)
        xa_path.parent.mkdir(parents=True, exist_ok=True)
        if recompute or _force_recompute_xa:
            xa = np.zeros(
                (len(alpha_list), len(seed_list), len(x_true), ensemble_size, state_dim)
            )
            for j, seed in enumerate(seed_list):
                y = observations[j]
                x0_ensemble = initial_ensembles[j]
                for i, alpha in enumerate(alpha_list):
                    np.random.seed(int(seed))
                    da_instance = PO(
                        model_step,
                        h,
                        observation_cov,
                        alpha=alpha**2,
                        store_ensemble=True,
                        additive_inflation=True,
                        **_method_extra_kwargs[method_name],
                    )

                    da_instance.initialize(x0_ensemble.copy())
                    for y_obs in y:
                        da_instance.forecast(dt_obs)
                        da_instance.update(y_obs)
                    xa[i, j] = da_instance.Xa

                    if _diag_arrays is not None:
                        _Xf_arr = np.array(da_instance.Xf)
                        _Xa_arr = np.array(da_instance.Xa)
                        _diag = compute_diagnostics(_Xf_arr, _Xa_arr, unobserved_idx)
                        for _k, _v in _diag.items():
                            _diag_arrays[_k][m_idx, i, j] = _v
                        _max_abs = float(
                            np.abs(
                                _Xa_arr[:, :, unobserved_idx]
                                - _Xf_arr[:, :, unobserved_idx]
                            ).max()
                        )
                        _member_increment_max_abs_unobserved[m_idx, i, j] = _max_abs

            np.save(xa_path, xa)
            print("saved:", xa_path, xa.shape)
        else:
            xa = load_npy(xa_path)

        xa_dict[method_name] = xa

    if diagnostics_path is not None and _need_compute_diag:
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            diagnostics_path,
            **_diag_arrays,
            member_increment_max_abs_unobserved=_member_increment_max_abs_unobserved,
            method_names=np.array(methods),
            alphas=np.array(alpha_list),
            seeds=np.array(seed_list),
            time_index=np.arange(n_time),
        )
        print("saved diagnostics:", diagnostics_path)

    return observations, xa_dict



# ---------------------------------------------------------------------------
# Experiment configuration – one dict per experiment condition.
# To add a new condition, define another dict and pass it to run_experiment().
# ---------------------------------------------------------------------------

CACHE_SCHEMA_VERSION = 1

PAPER_EXPERIMENT = {
    "state_dim": 60,
    "forcing": 8.0,
    "dt": 0.01,
    "spinup_steps": 20 * 360,
    "num_steps": 20 * 50,
    "obs_per": 1,
    "obs_noise_std": 1.0,
    "ensemble_size": 10,
    "seed_list": list(range(20)),
    "alpha_list_all": [0.0, 0.5, 2.0, 10.0, 100.0],
    "alpha_list_fig": [0.0, 0.5, 2.0],
    "methods": ["po_add", "po_proj"],
    "sample_seed_index": 0,
    # The manuscript uses k = 1 for sample-path plots; Python indexing is 0-based.
    "sample_member_index": 0,
    "observation_pattern": "two_of_three",
}

# ---------------------------------------------------------------------------
# Projection-comparison experiment constants
# ---------------------------------------------------------------------------

COMPARISON_SUBDIR = "projection_comparison"
COMPARISON_M_DIR_FMT = "m{ensemble_size}"
COMPARISON_DIAGNOSTICS_BASENAME = "diagnostics.npz"
COMPARISON_SUMMARY_BASENAME = "summary.csv"
COMPARISON_PDF_BASENAME = "fig5_projection_comparison.pdf"
COMPARISON_ENSEMBLE_SIZES = (10, 20, 40, 80)

PRODUCTION_METHODS = ("po_add", "po_proj")
DIAG_KEYS = ("mean_increment_sq_unobserved",)

PROJECTION_COMPARISON_EXPERIMENT = {
    **PAPER_EXPERIMENT,
    "observation_pattern": "alternating",
    "alpha_list_all": [0.5],
    "alpha_list_fig": [0.5],
    "seed_list": list(range(20)),
    "diagnostics_filename": COMPARISON_DIAGNOSTICS_BASENAME,
    "make_standard_figures": False,
}


def run_experiment(cfg, data_dir, recompute=False, show_sample_paths=False):
    """Run a single experiment described by cfg and save results to data_dir."""
    data_dir.mkdir(parents=True, exist_ok=True)

    _marker = _complete_marker_path(data_dir)
    if recompute:
        if _marker.exists():
            _marker.unlink()
    elif not _marker.exists():
        recompute = True

    state_dim = cfg["state_dim"]
    forcing = cfg["forcing"]
    dt = cfg["dt"]
    spinup_steps = cfg["spinup_steps"]
    num_steps = cfg["num_steps"]
    obs_per = cfg["obs_per"]
    obs_noise_std = cfg["obs_noise_std"]
    ensemble_size = cfg["ensemble_size"]
    seed_list = np.array(cfg["seed_list"])
    alpha_list_all = cfg["alpha_list_all"]
    alpha_list_fig = cfg["alpha_list_fig"]
    methods = cfg["methods"]
    sample_seed_index = cfg["sample_seed_index"]
    sample_member_index = cfg["sample_member_index"]
    observation_pattern = cfg.get("observation_pattern", "two_of_three")

    h = build_observation_operator(state_dim, pattern=observation_pattern)
    print("H.shape:", h.shape)
    print("rank(H):", np.linalg.matrix_rank(h))

    parameters = {
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
        "recompute": recompute,
        "observation_pattern": observation_pattern,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
    }

    _git_rev = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True
    )
    if _git_rev.returncode == 0:
        parameters["git_commit"] = _git_rev.stdout.strip()
        _git_status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
        )
        parameters["git_dirty"] = _git_status.returncode == 0 and bool(
            _git_status.stdout.strip()
        )
    else:
        parameters["git_commit"] = None
        parameters["git_dirty"] = None

    if recompute:
        save_run_parameters(data_dir, parameters)

    x0 = forcing * np.ones(state_dim)
    x0[19] *= 1.001
    x_true = generate_true_trajectory(
        data_dir=data_dir,
        x0=x0,
        forcing=forcing,
        dt=dt,
        spinup_steps=spinup_steps,
        num_steps=num_steps,
        recompute=recompute,
    )

    unobserved_idx = np.where(np.diag(h.T @ h) <= 0.5)[0]
    diagnostics_filename = cfg.get("diagnostics_filename")
    diagnostics_path = (
        (data_dir / diagnostics_filename) if diagnostics_filename else None
    )

    _make_standard_figures = cfg.get("make_standard_figures", True)
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
        recompute=recompute,
        diagnostics_path=diagnostics_path,
        unobserved_idx=unobserved_idx if diagnostics_path is not None else None,
        load_analysis_ensembles=_make_standard_figures,
    )

    if _make_standard_figures:
        write_fig1_summary_csv(
            data_dir=data_dir,
            x_true=x_true,
            h=h,
            r=obs_noise_std,
            alpha_list_all=alpha_list_all,
            alpha_list_fig=alpha_list_fig,
            methods=methods,
            xa_dict=xa_dict,
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
            show_sample_paths=show_sample_paths,
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
            show_sample_paths=show_sample_paths,
        )
        plot_fig6_true_obs(
            data_dir=data_dir,
            x_true=x_true,
            h=h,
            observations=observations,
            sample_seed_index=sample_seed_index,
        )
        plot_fig7_analysis_states(
            data_dir=data_dir,
            x_true=x_true,
            alpha_list_all=alpha_list_all,
            alpha_list_fig=alpha_list_fig,
            methods=methods,
            xa_dict=xa_dict,
            sample_seed_index=sample_seed_index,
            sample_member_index=sample_member_index,
        )

    # Atomic bundle marker write: all computation and figures succeeded
    _tmp = data_dir / ".complete.tmp"
    _tmp.write_text("bundle complete\n", encoding="utf-8")
    _tmp.replace(_marker)


def _load_weighted_error_timeseries(m_dir, method, alpha_idx=0):
    """Return seed-wise member-mean weighted-error time series.

    The manuscript norm is expanded computationally as the full-state squared
    error plus the observed-component squared error.  The returned array has
    shape ``(n_seeds, n_time)``.
    """
    x_true = np.load(true_trajectory_path(m_dir), mmap_mode="r")
    xa = np.load(analysis_ensembles_path(m_dir, method), mmap_mode="r")
    with (m_dir / "run_parameters.json").open(encoding="utf-8") as f:
        run_params = json.load(f)
    h = build_observation_operator(
        run_params["state_dim"],
        pattern=run_params.get("observation_pattern", "alternating"),
    )
    return weighted_error_per_seed(x_true, h, xa[alpha_idx])


def _aggregate_diagnostics_one_m(npz_path):
    """Return the two final per-seed scalar diagnostics for one ensemble size."""
    alpha_idx = 0
    m_dir = Path(npz_path).parent

    with np.load(npz_path) as npz:
        seeds = npz["seeds"]
        result = {"seeds": seeds}
        for m_idx, method in enumerate(PRODUCTION_METHODS):
            miu = npz["mean_increment_sq_unobserved"][m_idx, alpha_idx]
            result[method] = {
                "mean_increment_rms_unobserved": np.sqrt(
                    time_average_per_seed_all(miu)
                ),
            }

    for method in PRODUCTION_METHODS:
        weighted_timeseries = _load_weighted_error_timeseries(m_dir, method, alpha_idx)
        result[method]["weighted_norm_error"] = time_average_per_seed_all(
            weighted_timeseries
        )

    return result


def _write_summary_csv(csv_path, result_groups, methods):
    """Write long-form summary CSV to csv_path.

    Aggregate rows report the arithmetic seed mean and the empirical 2.5th and
    97.5th seed percentiles. Per-seed rows leave the percentile columns empty.
    """
    _FIELDS = [
        "record_type",
        "ensemble_size",
        "alpha",
        "seed",
        "method",
        "metric",
        "estimate",
        "p2_5",
        "p97_5",
    ]

    def _fmt(v):
        return repr(float(v))

    rows = []
    for group in result_groups:
        alpha = group["alpha"]
        metrics = group["metrics"]
        for m, m_res in zip(group["ensemble_sizes"], group["results"]):
            seeds = m_res["seeds"]
            for metric in metrics:
                for method in methods:
                    vals = m_res[method][metric]
                    for seed_val, value in zip(seeds, vals):
                        rows.append(
                            {
                                "record_type": "per_seed",
                                "ensemble_size": m,
                                "alpha": alpha,
                                "seed": int(seed_val),
                                "method": method,
                                "metric": metric,
                                "estimate": _fmt(value),
                                "p2_5": "",
                                "p97_5": "",
                            }
                        )
                    mean, p2_5, p97_5 = mean_empirical_percentile_band(vals)
                    rows.append(
                        {
                            "record_type": "aggregate",
                            "ensemble_size": m,
                            "alpha": alpha,
                            "seed": "",
                            "method": method,
                            "metric": metric,
                            "estimate": _fmt(mean),
                            "p2_5": _fmt(p2_5),
                            "p97_5": _fmt(p97_5),
                        }
                    )

            ratios = (
                m_res["po_add"]["weighted_norm_error"]
                / m_res["po_proj"]["weighted_norm_error"]
            )
            for seed_val, ratio in zip(seeds, ratios):
                rows.append(
                    {
                        "record_type": "per_seed",
                        "ensemble_size": m,
                        "alpha": alpha,
                        "seed": int(seed_val),
                        "method": "po_add/po_proj",
                        "metric": "weighted_norm_error_ratio",
                        "estimate": _fmt(ratio),
                        "p2_5": "",
                        "p97_5": "",
                    }
                )
            mean, p2_5, p97_5 = mean_empirical_percentile_band(ratios)
            rows.append(
                {
                    "record_type": "aggregate",
                    "ensemble_size": m,
                    "alpha": alpha,
                    "seed": "",
                    "method": "po_add/po_proj",
                    "metric": "weighted_norm_error_ratio",
                    "estimate": _fmt(mean),
                    "p2_5": _fmt(p2_5),
                    "p97_5": _fmt(p97_5),
                }
            )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print("saved:", csv_path)


def write_fig5_summary_csv(data_dir, ensemble_sizes, all_results, alpha, num_steps):
    """Write table-ready statistics corresponding to Figure 5.

    One row per ensemble size, holding the two quantities shown in the figure:
    the direct unobserved analysis-increment RMS for ``add`` (panel (a)) and the
    per-seed paired ``add``/``add-proj`` weighted-error ratio (panel (b)). Both
    are already time-averaged over all analysis times by
    ``_aggregate_diagnostics_one_m``. The reported values are the arithmetic
    seed mean and the empirical 2.5th--97.5th percentiles.

    The ``add-proj`` increment is identically zero by construction and is
    therefore stated in the manuscript caption rather than given a column.
    """
    rows = []
    for m, m_res in zip(ensemble_sizes, all_results):
        increments = m_res["po_add"]["mean_increment_rms_unobserved"]
        ratios = (
            m_res["po_add"]["weighted_norm_error"]
            / m_res["po_proj"]["weighted_norm_error"]
        )
        inc_mean, inc_low, inc_high = mean_empirical_percentile_band(increments)
        ratio_mean, ratio_low, ratio_high = mean_empirical_percentile_band(ratios)
        rows.append(
            {
                "ensemble_size": m,
                "alpha": alpha,
                "increment_rms": inc_mean,
                "increment_rms_p2_5": inc_low,
                "increment_rms_p97_5": inc_high,
                "mse_ratio": ratio_mean,
                "mse_ratio_p2_5": ratio_low,
                "mse_ratio_p97_5": ratio_high,
                "num_seeds": len(m_res["seeds"]),
                "analysis_time_start": 1,
                "analysis_time_end": num_steps,
            }
        )

    path = Path(data_dir) / FIG5_SUMMARY_BASENAME
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("saved:", path)


def run_projection_comparison(
    data_dir,
    recompute=False,
    cfg=None,
    ensemble_sizes=None,
):
    """Run and aggregate the projection-comparison experiment behind Figure 5."""
    if cfg is None:
        cfg = PROJECTION_COMPARISON_EXPERIMENT
    if ensemble_sizes is None:
        ensemble_sizes = COMPARISON_ENSEMBLE_SIZES
    data_dir = Path(data_dir)
    ensemble_sizes = tuple(ensemble_sizes)
    base_dir = data_dir / COMPARISON_SUBDIR
    methods = cfg["methods"]

    for m in ensemble_sizes:
        m_cfg = {**cfg, "ensemble_size": m}
        m_dir = base_dir / COMPARISON_M_DIR_FMT.format(ensemble_size=m)
        run_experiment(m_cfg, m_dir, recompute=recompute)

    all_results = []
    for m in ensemble_sizes:
        m_dir = base_dir / COMPARISON_M_DIR_FMT.format(ensemble_size=m)
        npz_path = m_dir / COMPARISON_DIAGNOSTICS_BASENAME
        all_results.append(_aggregate_diagnostics_one_m(npz_path))

    csv_path = base_dir / COMPARISON_SUMMARY_BASENAME
    _write_summary_csv(
        csv_path,
        [
            {
                "alpha": cfg["alpha_list_all"][0],
                "ensemble_sizes": ensemble_sizes,
                "results": all_results,
                "metrics": (
                    "mean_increment_rms_unobserved",
                    "weighted_norm_error",
                ),
            },
        ],
        methods,
    )

    write_fig5_summary_csv(
        data_dir,
        ensemble_sizes,
        all_results,
        alpha=cfg["alpha_list_all"][0],
        num_steps=cfg["num_steps"],
    )

    plot_fig5_projection_comparison(
        data_dir / COMPARISON_PDF_BASENAME,
        ensemble_sizes,
        all_results,
        methods,
    )


def run_reproduction(
    data_dir,
    *,
    num_steps=PAPER_EXPERIMENT["num_steps"],
    spinup_steps=PAPER_EXPERIMENT["spinup_steps"],
    seed_list=None,
    baseline_alpha=None,
    comparison_sizes=COMPARISON_ENSEMBLE_SIZES,
    recompute=False,
    show_sample_paths=False,
):
    """Internal entry point for reproducibility and smoke testing.

    Defaults reproduce the full paper experiments.  For a quick smoke test pass
    smaller values explicitly:

        run_reproduction(
            data_dir,
            num_steps=20, spinup_steps=7200, seed_list=(0, 1),
            baseline_alpha=(0.5,), comparison_sizes=(10, 20),
        )

    Parameters
    ----------
    data_dir : path-like
    num_steps : int
    spinup_steps : int
    seed_list : sequence of int or None
        Seeds shared by baseline and comparison. Defaults to all paper seeds.
    baseline_alpha : sequence of float or None
        Inflation parameters for the baseline experiment. Defaults to all paper alphas.
    comparison_sizes : sequence of int
        Ensemble sizes for the projection-comparison experiment.
    recompute : bool
    """
    if seed_list is None:
        seed_list = PAPER_EXPERIMENT["seed_list"]
    if baseline_alpha is None:
        _alpha_list_all = PAPER_EXPERIMENT["alpha_list_all"]
        _alpha_list_fig = PAPER_EXPERIMENT["alpha_list_fig"]
    else:
        _alpha_list_all = list(baseline_alpha)
        _alpha_list_fig = list(baseline_alpha)
    data_dir = Path(data_dir)
    baseline_cfg = {
        **PAPER_EXPERIMENT,
        "num_steps": num_steps,
        "spinup_steps": spinup_steps,
        "alpha_list_all": _alpha_list_all,
        "alpha_list_fig": _alpha_list_fig,
        "seed_list": list(seed_list),
    }
    run_experiment(
        baseline_cfg,
        data_dir,
        recompute=recompute,
        show_sample_paths=show_sample_paths,
    )
    comparison_cfg = {
        **PROJECTION_COMPARISON_EXPERIMENT,
        "num_steps": num_steps,
        "spinup_steps": spinup_steps,
        "seed_list": list(seed_list),
    }
    run_projection_comparison(
        data_dir,
        recompute=recompute,
        cfg=comparison_cfg,
        ensemble_sizes=tuple(comparison_sizes),
    )


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
    parser.add_argument(
        "--show-sample-paths",
        action="store_true",
        help="Overlay seed-wise paths on the time-series figures.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_reproduction(
        args.data_dir,
        recompute=args.recompute,
        show_sample_paths=args.show_sample_paths,
    )


if __name__ == "__main__":
    main()
