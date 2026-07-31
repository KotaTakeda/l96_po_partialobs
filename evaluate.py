#!/usr/bin/env python3
"""Preparatory evaluation of cached Lorenz 96 partial-observation experiments.

The cached analysis ensembles are left unchanged. Forecast ensembles are
reconstructed from the seed-reproduced initial ensemble at the first cycle
and from the preceding analysis ensemble thereafter, using the same Lorenz 96
model step as ``main.py``. This script is a diagnostic prototype; the
manuscript figures and public reproduction code will be produced from a
separately polished implementation after the experiment design is fixed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path

import matplotlib
import numpy as np
from da.l96 import lorenz96
from da.scheme import rk4

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402


EVALUATION_START = 0
EVALUATION_STOP = 1000
INITIAL_ENSEMBLE_VARIANCE = 16.0
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 0
CI_LEVEL = 0.95
QC_TOLERANCE = 1.0e-10
EXPECTED_METHODS = ("po_add", "po_proj")
EXPECTED_SCALAR_PARAMETERS = {
    "state_dim": 60,
    "forcing": 8.0,
    "dt": 0.01,
    "spinup_steps": 7200,
    "num_steps": 1000,
    "obs_per": 1,
    "obs_noise_std": 1.0,
}
OUTPUT_FILENAMES = (
    "evaluation_results.npz",
    "evaluation_summary.csv",
    "evaluation.pdf",
)

METHOD_LABELS = {
    "po_add": "add",
    "po_proj": "add-proj",
}
METHOD_COLORS = {
    "po_add": "#107D79",
    "po_proj": "#FF9933",
}

SCOPES = ("all", "observed", "unobserved")
STAGES = ("forecast", "analysis")

METRIC_NAMES = [
    "analysis_member_theory_norm_sq",
    "analysis_member_theory_norm_sq_over_reference_4no_r2",
]
for scope in SCOPES:
    for stage in STAGES:
        METRIC_NAMES.extend(
            [
                f"{stage}_mean_mse_{scope}",
                f"{stage}_member_mse_{scope}",
                f"{stage}_spread_{scope}",
            ]
        )
METRIC_NAMES.extend(
    [
        "forecast_cross_cov_ratio",
        "analysis_cross_cov_ratio",
        "gain_unobserved_rms",
        "innovation_rms",
        "mean_increment_sq_unobserved",
        "member_increment_sq_unobserved",
        "po_free_mean_increment_sq_unobserved",
        "po_free_member_increment_sq_unobserved",
        "mean_relative_error_reduction_unobserved",
    ]
)
METRIC_INDEX = {name: index for index, name in enumerate(METRIC_NAMES)}


def parse_args() -> argparse.Namespace:
    """Parse public input and plotting options."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the preparatory evaluation of cached PO experiments without "
            "modifying main.py or the input cache."
        )
    )
    parser.add_argument(
        "--data-dir",
        default="data/reproduce",
        help="Directory containing run_parameters.json and cached .npy arrays.",
    )
    parser.add_argument(
        "--linear-mse",
        action="store_true",
        help="Use linear y-axes for the page-4 MSE_U and MSE_O panels.",
    )
    return parser.parse_args()


def load_parameters(data_dir: Path) -> dict:
    """Load the experiment configuration."""
    path = data_dir / "run_parameters.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing run parameters: {path}")
    with path.open(encoding="utf-8") as stream:
        parameters = json.load(stream)
    return parameters


def ensure_expected_configuration(parameters: dict) -> None:
    """Validate the shared model settings and experiment axes."""
    required = {
        "state_dim",
        "forcing",
        "dt",
        "spinup_steps",
        "num_steps",
        "obs_per",
        "obs_noise_std",
        "ensemble_size",
        "seed_list",
        "alpha_list_all",
        "methods",
    }
    missing = sorted(required.difference(parameters))
    if missing:
        raise ValueError(f"Missing run-parameter keys: {missing}")

    methods = tuple(parameters["methods"])
    alphas = np.asarray(parameters["alpha_list_all"], dtype=float)
    seeds = np.asarray(parameters["seed_list"], dtype=int)
    checks = {
        "methods": methods == EXPECTED_METHODS,
        "alphas_one_dimensional": alphas.ndim == 1,
        "alphas_nonempty": alphas.size > 0,
        "alphas_finite": bool(np.isfinite(alphas).all()),
        "alphas_nonnegative": bool(np.all(alphas >= 0.0)),
        "alphas_unique": len(np.unique(alphas)) == len(alphas),
        "seeds_one_dimensional": seeds.ndim == 1,
        "seeds_nonempty": seeds.size > 0,
        "seeds_unique": len(np.unique(seeds)) == len(seeds),
        "ensemble_size_integer": isinstance(parameters["ensemble_size"], int),
        "ensemble_size_at_least_two": (
            isinstance(parameters["ensemble_size"], int)
            and parameters["ensemble_size"] >= 2
        ),
    }
    for name, expected in EXPECTED_SCALAR_PARAMETERS.items():
        checks[name] = parameters[name] == expected
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "The cache configuration is incompatible with the preparatory "
            f"evaluator; failed checks: {failed}"
        )
    if EVALUATION_START != 0:
        raise ValueError("The evaluation interval must start at Python index 0.")
    if EVALUATION_STOP - EVALUATION_START != 1000:
        raise ValueError("The required evaluation interval must contain 1000 steps.")


def observed_state_indices(parameters: dict) -> tuple[np.ndarray, np.ndarray]:
    """Load the observation partition, with a fallback for the original cache."""
    state_dim = int(parameters["state_dim"])
    if "observed_indices_python_zero_based" in parameters:
        observed = np.asarray(
            parameters["observed_indices_python_zero_based"],
            dtype=int,
        )
    else:
        observed_mask = np.ones(state_dim, dtype=bool)
        observed_mask[2::3] = False
        observed = np.flatnonzero(observed_mask)
    if observed.ndim != 1 or observed.size == 0:
        raise ValueError("Observed indices must be a nonempty one-dimensional list.")
    if len(np.unique(observed)) != len(observed):
        raise ValueError("Observed indices contain duplicates.")
    if np.any(observed < 0) or np.any(observed >= state_dim):
        raise ValueError("Observed indices fall outside the state dimension.")
    if not np.all(observed[:-1] < observed[1:]):
        raise ValueError(
            "Observed indices must be strictly increasing to match observation rows."
        )
    observed_mask = np.zeros(state_dim, dtype=bool)
    observed_mask[observed] = True
    unobserved = np.flatnonzero(~observed_mask)
    return observed, unobserved


def reconstruct_initial_ensembles(
    truth: np.ndarray,
    observations: np.ndarray,
    observed: np.ndarray,
    seeds: np.ndarray,
    obs_noise_std: float,
    ensemble_size: int,
    qc: dict[str, float],
) -> np.ndarray:
    """Reproduce cached observations and the initial ensembles from each seed."""
    state_dim = truth.shape[1]
    observation_dim = len(observed)
    projected_truth = np.asarray(truth[:, observed])
    observation_covariance = (obs_noise_std**2) * np.eye(observation_dim)
    initial_covariance = INITIAL_ENSEMBLE_VARIANCE * np.eye(state_dim)
    initial_ensembles = np.empty(
        (len(seeds), ensemble_size, state_dim),
        dtype=float,
    )
    max_observation_error = 0.0

    for seed_index, seed in enumerate(seeds):
        rng = np.random.RandomState(int(seed))
        regenerated_observations = (
            projected_truth
            + rng.multivariate_normal(
                np.zeros(observation_dim),
                observation_covariance,
                size=len(truth),
            )
        )
        difference = np.abs(
            regenerated_observations - observations[seed_index]
        )
        max_observation_error = max(
            max_observation_error,
            float(np.max(difference)),
        )
        if not np.array_equal(
            regenerated_observations,
            observations[seed_index],
        ):
            raise AssertionError(
                "Seed-based observation regeneration did not exactly match "
                f"the cache for seed={int(seed)}; "
                f"max absolute error={float(np.max(difference)):.6g}."
            )

        attractor_index = rng.randint(len(truth))
        initial_ensembles[seed_index] = (
            truth[attractor_index]
            + rng.multivariate_normal(
                np.zeros(state_dim),
                initial_covariance,
                size=ensemble_size,
            )
        )

    require_finite("reconstructed initial ensembles", initial_ensembles)
    qc["observation_regeneration_max_absolute_error"] = (
        max_observation_error
    )
    return initial_ensembles


def lorenz96_batch(_time: float, state: np.ndarray, forcing: float) -> np.ndarray:
    """Evaluate Lorenz 96 independently along the last array axis."""
    return (
        (np.roll(state, -1, axis=-1) - np.roll(state, 2, axis=-1))
        * np.roll(state, 1, axis=-1)
        - state
        + forcing
    )


def forecast_batch(
    analysis: np.ndarray,
    forcing: float,
    dt: float,
    obs_per: int,
) -> np.ndarray:
    """Advance every time/member sample by one observation interval."""
    forecast = np.array(analysis, dtype=float, copy=True)
    for _ in range(obs_per):
        forecast = rk4(
            lorenz96_batch,
            0.0,
            forecast,
            (forcing,),
            dt,
        )
    return forecast


def forecast_scalar_reference(
    analysis: np.ndarray,
    forcing: float,
    dt: float,
    obs_per: int,
) -> np.ndarray:
    """Use the literal ``main.py`` member loop for one QC sample."""
    forecast = np.array(analysis, dtype=float, copy=True)
    for member_index, member in enumerate(forecast):
        state = member
        for _ in range(obs_per):
            state = rk4(lorenz96, 0.0, state, (forcing,), dt)
        forecast[member_index] = state
    return forecast


def require_finite(name: str, array: np.ndarray) -> None:
    """Fail immediately when a required input or diagnostic is non-finite."""
    if not np.isfinite(array).all():
        bad_count = int(array.size - np.count_nonzero(np.isfinite(array)))
        raise FloatingPointError(f"{name} contains {bad_count} non-finite values.")


def check_identity(
    name: str,
    left: np.ndarray,
    right: np.ndarray,
    qc: dict[str, float],
) -> None:
    """Check a numerical identity with the declared absolute/relative tolerance."""
    difference = np.abs(left - right)
    max_absolute = float(np.max(difference))
    scale = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1.0)
    max_scaled = float(np.max(difference / scale))
    qc[f"{name}_max_absolute_error"] = max(
        qc.get(f"{name}_max_absolute_error", 0.0),
        max_absolute,
    )
    qc[f"{name}_max_scaled_error"] = max(
        qc.get(f"{name}_max_scaled_error", 0.0),
        max_scaled,
    )
    if not np.allclose(
        left,
        right,
        atol=QC_TOLERANCE,
        rtol=QC_TOLERANCE,
    ):
        raise AssertionError(
            f"{name} failed at tolerance {QC_TOLERANCE:g}; "
            f"max absolute error={max_absolute:.6g}, "
            f"max scaled error={max_scaled:.6g}."
        )


def covariance_blocks(
    ensemble: np.ndarray,
    observed: np.ndarray,
    unobserved: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw sample-covariance blocks C_OO and C_UO."""
    ensemble_size = ensemble.shape[1]
    mean = ensemble.mean(axis=1, keepdims=True)
    anomalies = ensemble - mean
    anomalies_observed = anomalies[:, :, observed]
    anomalies_unobserved = anomalies[:, :, unobserved]
    covariance_observed = np.einsum(
        "tmi,tmj->tij",
        anomalies_observed,
        anomalies_observed,
        optimize=True,
    )
    covariance_cross = np.einsum(
        "tmi,tmj->tij",
        anomalies_unobserved,
        anomalies_observed,
        optimize=True,
    )
    covariance_observed /= ensemble_size - 1
    covariance_cross /= ensemble_size - 1
    return covariance_observed, covariance_cross


def cross_covariance_ratio(
    covariance_observed: np.ndarray,
    covariance_cross: np.ndarray,
    name: str,
) -> np.ndarray:
    """Compute ||C_UO||_F / ||C_OO||_F for each time."""
    denominator = np.linalg.norm(covariance_observed, axis=(1, 2))
    if np.any(denominator <= 0.0):
        count = int(np.count_nonzero(denominator <= 0.0))
        raise FloatingPointError(
            f"{name} has {count} non-positive observed-block norms."
        )
    return np.linalg.norm(covariance_cross, axis=(1, 2)) / denominator


def scope_statistics(
    ensemble: np.ndarray,
    truth: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ensemble-mean MSE, member-average MSE, and sample spread."""
    subset = ensemble[:, :, indices]
    truth_subset = truth[:, indices]
    ensemble_mean = subset.mean(axis=1)
    mean_mse = np.mean((ensemble_mean - truth_subset) ** 2, axis=1)
    member_mse = np.mean(
        (subset - truth_subset[:, None, :]) ** 2,
        axis=(1, 2),
    )
    anomalies = subset - ensemble_mean[:, None, :]
    spread = np.sum(anomalies**2, axis=(1, 2))
    spread /= (ensemble.shape[1] - 1) * len(indices)
    return mean_mse, member_mse, spread


def evaluate_chunk(
    analysis_previous: np.ndarray,
    analysis: np.ndarray,
    truth: np.ndarray,
    observations: np.ndarray,
    observed: np.ndarray,
    unobserved: np.ndarray,
    method: str,
    alpha: float,
    forcing: float,
    dt: float,
    obs_per: int,
    obs_noise_std: float,
    qc: dict[str, float],
) -> np.ndarray:
    """Evaluate one method/alpha/seed block over all requested times."""
    forecast = forecast_batch(
        analysis_previous,
        forcing=forcing,
        dt=dt,
        obs_per=obs_per,
    )
    require_finite("reconstructed forecast", forecast)

    time_count, ensemble_size, state_dim = analysis.shape
    if forecast.shape != (time_count, ensemble_size, state_dim):
        raise ValueError(
            f"Forecast shape {forecast.shape} does not match analysis "
            f"shape {analysis.shape}."
        )

    result = np.empty((len(METRIC_NAMES), time_count), dtype=float)
    reference = 4.0 * len(observed) * obs_noise_std**2

    analysis_error = analysis - truth[:, None, :]
    theory_norm_sq = np.mean(
        np.sum(analysis_error**2, axis=2)
        + np.sum(analysis_error[:, :, observed] ** 2, axis=2),
        axis=1,
    )
    result[METRIC_INDEX["analysis_member_theory_norm_sq"]] = theory_norm_sq
    result[
        METRIC_INDEX[
            "analysis_member_theory_norm_sq_over_reference_4no_r2"
        ]
    ] = theory_norm_sq / reference

    scope_indices = {
        "all": np.arange(state_dim),
        "observed": observed,
        "unobserved": unobserved,
    }
    stage_ensembles = {
        "forecast": forecast,
        "analysis": analysis,
    }
    for scope, indices in scope_indices.items():
        for stage, ensemble in stage_ensembles.items():
            mean_mse, member_mse, spread = scope_statistics(
                ensemble,
                truth,
                indices,
            )
            result[METRIC_INDEX[f"{stage}_mean_mse_{scope}"]] = mean_mse
            result[METRIC_INDEX[f"{stage}_member_mse_{scope}"]] = member_mse
            result[METRIC_INDEX[f"{stage}_spread_{scope}"]] = spread
            decomposition = mean_mse + (ensemble_size - 1) / ensemble_size * spread
            check_identity(
                f"{stage}_{scope}_member_mse_decomposition",
                member_mse,
                decomposition,
                qc,
            )

    forecast_cov_observed, forecast_cov_cross = covariance_blocks(
        forecast,
        observed,
        unobserved,
    )
    analysis_cov_observed, analysis_cov_cross = covariance_blocks(
        analysis,
        observed,
        unobserved,
    )
    result[METRIC_INDEX["forecast_cross_cov_ratio"]] = cross_covariance_ratio(
        forecast_cov_observed,
        forecast_cov_cross,
        "forecast covariance",
    )
    result[METRIC_INDEX["analysis_cross_cov_ratio"]] = cross_covariance_ratio(
        analysis_cov_observed,
        analysis_cov_cross,
        "analysis covariance",
    )

    if method == "po_add":
        identity_observed = np.eye(len(observed))[None, :, :]
        gain_system = forecast_cov_observed + (
            alpha**2 + obs_noise_std**2
        ) * identity_observed
        gain_unobserved = np.linalg.solve(
            gain_system,
            np.swapaxes(forecast_cov_cross, 1, 2),
        )
        gain_unobserved = np.swapaxes(gain_unobserved, 1, 2)
    elif method == "po_proj":
        gain_unobserved = np.zeros(
            (time_count, len(unobserved), len(observed)),
            dtype=float,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    result[METRIC_INDEX["gain_unobserved_rms"]] = np.sqrt(
        np.mean(gain_unobserved**2, axis=(1, 2))
    )

    forecast_mean = forecast.mean(axis=1)
    analysis_mean = analysis.mean(axis=1)
    innovation = observations - forecast_mean[:, observed]
    result[METRIC_INDEX["innovation_rms"]] = np.sqrt(
        np.mean(innovation**2, axis=1)
    )

    actual_increment_unobserved = (
        analysis[:, :, unobserved] - forecast[:, :, unobserved]
    )
    actual_mean_increment_unobserved = (
        analysis_mean[:, unobserved] - forecast_mean[:, unobserved]
    )
    po_free_member_increment_unobserved = np.einsum(
        "tuo,tmo->tmu",
        gain_unobserved,
        observations[:, None, :] - forecast[:, :, observed],
        optimize=True,
    )
    po_free_mean_increment_unobserved = np.einsum(
        "tuo,to->tu",
        gain_unobserved,
        innovation,
        optimize=True,
    )
    check_identity(
        "po_free_increment_mean",
        po_free_member_increment_unobserved.mean(axis=1),
        po_free_mean_increment_unobserved,
        qc,
    )

    result[METRIC_INDEX["mean_increment_sq_unobserved"]] = np.mean(
        actual_mean_increment_unobserved**2,
        axis=1,
    )
    result[METRIC_INDEX["member_increment_sq_unobserved"]] = np.mean(
        actual_increment_unobserved**2,
        axis=(1, 2),
    )
    result[METRIC_INDEX["po_free_mean_increment_sq_unobserved"]] = np.mean(
        po_free_mean_increment_unobserved**2,
        axis=1,
    )
    result[METRIC_INDEX["po_free_member_increment_sq_unobserved"]] = np.mean(
        po_free_member_increment_unobserved**2,
        axis=(1, 2),
    )

    forecast_mean_error_unobserved = (
        forecast_mean[:, unobserved] - truth[:, unobserved]
    )
    analysis_mean_error_unobserved = (
        analysis_mean[:, unobserved] - truth[:, unobserved]
    )
    forecast_mean_mse_unobserved = np.mean(
        forecast_mean_error_unobserved**2,
        axis=1,
    )
    analysis_mean_mse_unobserved = np.mean(
        analysis_mean_error_unobserved**2,
        axis=1,
    )
    if np.any(forecast_mean_mse_unobserved <= 0.0):
        raise ValueError(
            "Relative unobserved MSE improvement requires strictly positive "
            "forecast ensemble-mean MSE."
        )
    mean_relative_error_reduction = (
        forecast_mean_mse_unobserved - analysis_mean_mse_unobserved
    ) / forecast_mean_mse_unobserved
    result[METRIC_INDEX["mean_relative_error_reduction_unobserved"]] = (
        mean_relative_error_reduction
    )
    relative_error_reduction_identity = (
        -2.0
        * np.mean(
            forecast_mean_error_unobserved
            * actual_mean_increment_unobserved,
            axis=1,
        )
        - np.mean(actual_mean_increment_unobserved**2, axis=1)
    ) / forecast_mean_mse_unobserved
    check_identity(
        "mean_relative_error_reduction_unobserved",
        mean_relative_error_reduction,
        relative_error_reduction_identity,
        qc,
    )

    if method == "po_proj":
        max_qk = float(np.max(np.abs(gain_unobserved)))
        max_increment = float(np.max(np.abs(actual_increment_unobserved)))
        qc["po_proj_qk_max_absolute"] = max(
            qc.get("po_proj_qk_max_absolute", 0.0),
            max_qk,
        )
        qc["po_proj_unobserved_increment_max_absolute"] = max(
            qc.get("po_proj_unobserved_increment_max_absolute", 0.0),
            max_increment,
        )
        if max_qk > QC_TOLERANCE:
            raise AssertionError(
                f"Projected QK is nonzero: max absolute value={max_qk:.6g}."
            )
        if max_increment > QC_TOLERANCE:
            raise AssertionError(
                "Projected analysis changed an unobserved component: "
                f"max absolute increment={max_increment:.6g}."
            )

    require_finite("chunk diagnostics", result)
    return result


def metric_definitions() -> dict[str, str]:
    """Return human-readable definitions for every stored diagnostic."""
    definitions = {
        "analysis_member_theory_norm_sq": (
            "Member average of ||X_k^a-x||_2^2 + "
            "||H(X_k^a-x)||_2^2; sums, rather than averages, over state "
            "components."
        ),
        "analysis_member_theory_norm_sq_over_reference_4no_r2": (
            "analysis_member_theory_norm_sq divided by 4*N_O*r^2. The "
            "denominator is the reference line used in the original Figure 1, "
            "not the literal bound stated in the theorem."
        ),
        "forecast_cross_cov_ratio": (
            "Raw forecast sample-covariance ratio "
            "||C_UO^f||_F/||C_OO^f||_F, before inflation or projection."
        ),
        "analysis_cross_cov_ratio": (
            "Raw analysis sample-covariance ratio "
            "||C_UO^a||_F/||C_OO^a||_F."
        ),
        "gain_unobserved_rms": (
            "Root mean square of entries of QK. For add, "
            "K_U=C_UO^f[C_OO^f+(alpha^2+r^2)I]^{-1}; for add-proj, K_U=0."
        ),
        "innovation_rms": (
            "Root mean square over observed components of y-H*xbar^f."
        ),
        "mean_increment_sq_unobserved": (
            "Mean over U of (xbar_U^a-xbar_U^f)^2, including the finite-member "
            "mean of perturbed-observation noise."
        ),
        "member_increment_sq_unobserved": (
            "Mean over members and U of (X_U^a-X_U^f)^2."
        ),
        "po_free_mean_increment_sq_unobserved": (
            "Mean over U of [K_U(y-H*xbar^f)]^2; the ensemble-mean increment "
            "with the perturbed-observation contribution removed."
        ),
        "po_free_member_increment_sq_unobserved": (
            "Mean over members and U of [K_U(y-H*X_k^f)]^2; the member-wise "
            "increment with the perturbed-observation contribution removed."
        ),
        "mean_relative_error_reduction_unobserved": (
            "Instantaneous relative analysis-step improvement "
            "1-MSE_U(xbar^a,x)/MSE_U(xbar^f,x). Positive values indicate "
            "improvement and negative values indicate deterioration. This is "
            "not a comparison of long-time errors between methods. In CSV "
            "summaries and PDF panel 4, each seed is instead aggregated as "
            "1-time_mean(MSE_U^a)/time_mean(MSE_U^f), so its sign is "
            "consistent with the reported time-mean MSEs."
        ),
    }
    for scope in SCOPES:
        for stage in STAGES:
            definitions[f"{stage}_mean_mse_{scope}"] = (
                f"Mean squared error of the {stage} ensemble mean over "
                f"{scope} state components."
            )
            definitions[f"{stage}_member_mse_{scope}"] = (
                f"Mean squared error over ensemble members and {scope} state "
                f"components at the {stage} stage."
            )
            definitions[f"{stage}_spread_{scope}"] = (
                f"Trace of the {stage} sample covariance divided by the number "
                f"of {scope} components (sample normalization m-1)."
            )
    if set(definitions) != set(METRIC_NAMES):
        missing = sorted(set(METRIC_NAMES).difference(definitions))
        extra = sorted(set(definitions).difference(METRIC_NAMES))
        raise AssertionError(
            f"Metric-definition mismatch; missing={missing}, extra={extra}."
        )
    return definitions


def bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Bootstrap a mean with seeds as the resampling unit."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError(
            "Bootstrap input must contain at least two seed values; "
            f"received shape {values.shape}."
        )
    indices = rng.integers(
        0,
        values.size,
        size=(BOOTSTRAP_SAMPLES, values.size),
    )
    replicates = values[indices].mean(axis=1)
    tail = (1.0 - CI_LEVEL) / 2.0
    low, high = np.quantile(replicates, [tail, 1.0 - tail])
    return float(values.mean()), float(low), float(high)


def aggregate_diagnostics_by_seed(diagnostics: np.ndarray) -> np.ndarray:
    """Aggregate time series, preserving MSE-ratio consistency."""
    per_seed_values = diagnostics.mean(axis=-1)
    forecast_mse = diagnostics[
        METRIC_INDEX["forecast_mean_mse_unobserved"]
    ].mean(axis=-1)
    analysis_mse = diagnostics[
        METRIC_INDEX["analysis_mean_mse_unobserved"]
    ].mean(axis=-1)
    if np.any(forecast_mse <= 0.0):
        raise ValueError(
            "Aggregated relative unobserved MSE improvement requires "
            "strictly positive time-mean forecast MSE."
        )
    per_seed_values[
        METRIC_INDEX["mean_relative_error_reduction_unobserved"]
    ] = 1.0 - analysis_mse / forecast_mse
    return per_seed_values


def build_summary(
    diagnostics: np.ndarray,
    methods: tuple[str, ...],
    alphas: np.ndarray,
    seeds: np.ndarray,
) -> tuple[
    list[dict[str, object]],
    dict[tuple[str, str, float], tuple[float, float, float]],
    dict[tuple[str, float], tuple[float, float, float]],
]:
    """Build long-form rows and retain estimates needed by the PDF."""
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    overall_estimates = {}
    paired_ratio_estimates = {}
    time_count = diagnostics.shape[-1]

    per_seed_means = aggregate_diagnostics_by_seed(diagnostics)
    for metric_index, metric in enumerate(METRIC_NAMES):
        is_relative_improvement = (
            metric == "mean_relative_error_reduction_unobserved"
        )
        per_seed_statistic = (
            "one_minus_ratio_of_time_mean_mse"
            if is_relative_improvement
            else "time_mean"
        )
        overall_statistic = (
            "mean_of_seed_relative_mse_improvements"
            if is_relative_improvement
            else "mean_of_seed_time_means"
        )
        for method_index, method in enumerate(methods):
            for alpha_index, alpha in enumerate(alphas):
                values = per_seed_means[
                    metric_index,
                    method_index,
                    alpha_index,
                ]
                for seed_index, seed in enumerate(seeds):
                    rows.append(
                        {
                            "record_type": "per_seed",
                            "metric": metric,
                            "statistic": per_seed_statistic,
                            "method": method,
                            "alpha": float(alpha),
                            "seed": int(seed),
                            "value": float(values[seed_index]),
                            "ci_low": "",
                            "ci_high": "",
                            "n": time_count,
                        }
                    )
                estimate = bootstrap_mean_ci(values, rng)
                overall_estimates[(metric, method, float(alpha))] = estimate
                rows.append(
                    {
                        "record_type": "overall",
                        "metric": metric,
                        "statistic": overall_statistic,
                        "method": method,
                        "alpha": float(alpha),
                        "seed": "",
                        "value": estimate[0],
                        "ci_low": estimate[1],
                        "ci_high": estimate[2],
                        "n": len(seeds),
                    }
                )

        for alpha_index, alpha in enumerate(alphas):
            add_values = per_seed_means[metric_index, 0, alpha_index]
            projected_values = per_seed_means[metric_index, 1, alpha_index]
            difference = add_values - projected_values
            difference_estimate = bootstrap_mean_ci(difference, rng)
            rows.append(
                {
                    "record_type": "paired",
                    "metric": metric,
                    "statistic": "mean_difference",
                    "method": "po_add-po_proj",
                    "alpha": float(alpha),
                    "seed": "",
                    "value": difference_estimate[0],
                    "ci_low": difference_estimate[1],
                    "ci_high": difference_estimate[2],
                    "n": len(seeds),
                }
            )

            if np.all(add_values > 0.0) and np.all(projected_values > 0.0):
                log_ratio = np.log(add_values / projected_values)
                mean_log, low_log, high_log = bootstrap_mean_ci(log_ratio, rng)
                ratio_estimate = (
                    float(np.exp(mean_log)),
                    float(np.exp(low_log)),
                    float(np.exp(high_log)),
                )
                paired_ratio_estimates[(metric, float(alpha))] = ratio_estimate
                rows.append(
                    {
                        "record_type": "paired",
                        "metric": metric,
                        "statistic": "geometric_mean_ratio",
                        "method": "po_add/po_proj",
                        "alpha": float(alpha),
                        "seed": "",
                        "value": ratio_estimate[0],
                        "ci_low": ratio_estimate[1],
                        "ci_high": ratio_estimate[2],
                        "n": len(seeds),
                    }
                )
    return rows, overall_estimates, paired_ratio_estimates


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write the required long-form summary table."""
    fieldnames = [
        "record_type",
        "metric",
        "statistic",
        "method",
        "alpha",
        "seed",
        "value",
        "ci_low",
        "ci_high",
        "n",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def alpha_label(alpha: float) -> str:
    """Format an alpha value compactly."""
    return f"{alpha:g}"


def preferred_alphas(
    alphas: np.ndarray,
    preferred: tuple[float, ...],
    max_count: int,
) -> tuple[float, ...]:
    """Select preferred available alpha values, then fill in input order."""
    selected = []
    for candidate in preferred:
        matches = np.flatnonzero(
            np.isclose(alphas, candidate, rtol=0.0, atol=1.0e-12)
        )
        if matches.size:
            selected.append(float(alphas[int(matches[0])]))
    for alpha in alphas:
        value = float(alpha)
        if value not in selected:
            selected.append(value)
        if len(selected) == max_count:
            break
    return tuple(selected[:max_count])


def configure_plot_style() -> None:
    """Set a self-contained, LaTeX-free style for the evaluation report."""
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
            "pdf.fonttype": 42,
        }
    )


def plot_theory_page(
    pdf: PdfPages,
    diagnostics: np.ndarray,
    methods: tuple[str, ...],
    alphas: np.ndarray,
    time_index: np.ndarray,
    reference: float,
    condition_label: str,
) -> None:
    """Page 1: theory-norm time series and between-seed variation."""
    selected_alphas = preferred_alphas(
        alphas,
        preferred=(0.0, 0.25, 0.5, 2.0),
        max_count=3,
    )
    fig, axes = plt.subplots(
        1,
        len(selected_alphas),
        figsize=(11.69, 8.27),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    metric_index = METRIC_INDEX["analysis_member_theory_norm_sq"]
    manuscript_time = time_index + 1
    tick_candidates = [
        int(manuscript_time[0]),
        400,
        600,
        800,
        int(manuscript_time[-1]),
    ]
    time_ticks = sorted(
        {
            tick
            for tick in tick_candidates
            if manuscript_time[0] <= tick <= manuscript_time[-1]
        }
    )

    for axis, alpha in zip(axes, selected_alphas, strict=True):
        alpha_index = int(
            np.flatnonzero(
                np.isclose(alphas, alpha, rtol=0.0, atol=1.0e-12)
            )[0]
        )
        for method_index, method in enumerate(methods):
            values = diagnostics[metric_index, method_index, alpha_index]
            mean = values.mean(axis=0)
            low, high = np.quantile(values, [0.1, 0.9], axis=0)
            color = METHOD_COLORS[method]
            axis.fill_between(
                manuscript_time,
                low,
                high,
                color=color,
                alpha=0.18,
                linewidth=0.0,
            )
            axis.plot(
                manuscript_time,
                mean,
                color=color,
                label=METHOD_LABELS[method],
            )
        axis.axhline(
            reference,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label=r"reference $4N_Or^2$",
        )
        axis.set_title(rf"$\alpha={alpha_label(alpha)}$")
        axis.set_xlabel("assimilation step $n$")
        axis.set_xlim(manuscript_time[0], manuscript_time[-1])
        axis.set_xticks(time_ticks)
        axis.set_yscale("log")
        axis.grid(alpha=0.2, linewidth=0.5)

    axes[0].set_ylabel(
        r"member mean of $\|X_k^a-x\|^2+\|H(X_k^a-x)\|^2$"
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "Preparatory evaluation - Page 1: analysis error in the theorem norm",
        fontsize=15,
        y=0.96,
    )
    fig.text(0.5, 0.925, condition_label, ha="center", va="center", fontsize=9)
    fig.text(
        0.5,
        0.065,
        (
            "Lines are seed means; shaded bands are the 10th-90th seed "
            "percentiles (not confidence intervals). "
            "The reference reproduces the original Figure 1 convention and "
            "is not the literal bound stated in the theorem."
        ),
        ha="center",
        va="center",
        fontsize=8,
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.14,
        top=0.82,
        wspace=0.12,
    )
    pdf.savefig(fig)
    plt.close(fig)


def plot_mse_ratio_page(
    pdf: PdfPages,
    diagnostics: np.ndarray,
    alphas: np.ndarray,
    paired_ratio_estimates: dict[
        tuple[str, float],
        tuple[float, float, float],
    ],
    condition_label: str,
) -> None:
    """Page 2: paired add/add-proj analysis-MSE ratios."""
    scopes = ("all", "observed", "unobserved")
    fig, axes = plt.subplots(
        1,
        len(scopes),
        figsize=(11.69, 8.27),
        sharey=True,
    )
    x = np.arange(len(alphas), dtype=float)
    seed_jitter = np.linspace(-0.10, 0.10, diagnostics.shape[3])
    ratio_extent = []

    for axis, scope in zip(axes, scopes, strict=True):
        metric = f"analysis_mean_mse_{scope}"
        metric_index = METRIC_INDEX[metric]
        per_seed = diagnostics[metric_index].mean(axis=-1)
        estimates = []
        lows = []
        highs = []
        for alpha_index, alpha in enumerate(alphas):
            ratio_by_seed = per_seed[0, alpha_index] / per_seed[1, alpha_index]
            ratio_extent.extend(ratio_by_seed.tolist())
            axis.scatter(
                x[alpha_index] + seed_jitter,
                ratio_by_seed,
                s=10,
                color="#777777",
                alpha=0.32,
                edgecolors="none",
            )
            estimate, low, high = paired_ratio_estimates[
                (metric, float(alpha))
            ]
            estimates.append(estimate)
            lows.append(low)
            highs.append(high)
            ratio_extent.extend([low, high])
        estimates_array = np.asarray(estimates)
        error = np.vstack(
            [
                estimates_array - np.asarray(lows),
                np.asarray(highs) - estimates_array,
            ]
        )
        axis.errorbar(
            x,
            estimates_array,
            yerr=error,
            fmt="o",
            color="#222222",
            capsize=3,
            markersize=5,
            label="paired geometric mean and 95% CI",
        )
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
        axis.set_xticks(x, [alpha_label(alpha) for alpha in alphas])
        axis.set_xlabel(r"inflation $\alpha$")
        axis.set_title(scope.capitalize())
        axis.set_yscale("log")
        axis.grid(alpha=0.2, linewidth=0.5)

    ratio_extent_array = np.asarray(ratio_extent)
    require_finite("Page 2 ratio extent", ratio_extent_array)
    if np.any(ratio_extent_array <= 0.0):
        raise ValueError("Page 2 log-scale limits require positive ratios.")
    log_low = float(np.log(ratio_extent_array.min()))
    log_high = float(np.log(ratio_extent_array.max()))
    log_padding = max(0.08 * (log_high - log_low), 0.02)
    ratio_limits = (
        float(np.exp(log_low - log_padding)),
        float(np.exp(log_high + log_padding)),
    )
    for axis in axes:
        axis.set_ylim(ratio_limits)

    axes[0].set_ylabel(
        "analysis ensemble-mean MSE ratio\nadd / add-proj"
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        frameon=False,
    )
    fig.suptitle(
        "Preparatory evaluation - Page 2: paired analysis-MSE comparison",
        fontsize=15,
        y=0.96,
    )
    fig.text(0.5, 0.925, condition_label, ha="center", va="center", fontsize=9)
    fig.text(
        0.5,
        0.065,
        (
            "Each gray point is one seed after averaging over n=1,...,1000. "
            "Error bars use 10,000 paired seed bootstraps with seed 0. "
            "Ratios above 1 indicate larger analysis MSE for add."
        ),
        ha="center",
        va="center",
        fontsize=8,
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.14,
        top=0.82,
        wspace=0.18,
    )
    pdf.savefig(fig)
    plt.close(fig)


def plot_mechanism_page(
    pdf: PdfPages,
    diagnostics: np.ndarray,
    methods: tuple[str, ...],
    alphas: np.ndarray,
    overall_estimates: dict[
        tuple[str, str, float],
        tuple[float, float, float],
    ],
    condition_label: str,
) -> None:
    """Page 3: four-stage summary of information transfer into U."""
    has_quarter_alpha = bool(
        np.any(np.isclose(alphas, 0.25, rtol=0.0, atol=1.0e-12))
    )
    selected_alphas = preferred_alphas(
        alphas,
        preferred=(0.25, 0.5, 2.0),
        max_count=3 if has_quarter_alpha else 2,
    )
    panels = (
        (
            "forecast_cross_cov_ratio",
            r"1. Raw forecast $R_{UO}$",
            r"$\|C_{UO}^f\|_F/\|C_{OO}^f\|_F$",
        ),
        (
            "gain_unobserved_rms",
            r"2. Unobserved gain $QK$",
            r"entrywise RMS of $QK$",
        ),
        (
            "po_free_mean_increment_sq_unobserved",
            "3. Mean-squared PO-free mean U-increment",
            r"mean$_U\{[K_U(y-H\bar{x}^f)]^2\}$",
        ),
        (
            "mean_relative_error_reduction_unobserved",
            r"4. Relative immediate $G_U^{\mathrm{mean}}$",
            r"$1-\mathrm{MSE}_U(\bar{x}^a)/\mathrm{MSE}_U(\bar{x}^f)$",
        ),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    x = np.arange(len(selected_alphas), dtype=float)
    method_offsets = (-0.14, 0.14)
    seed_jitter = np.linspace(-0.055, 0.055, diagnostics.shape[3])
    per_seed_values = aggregate_diagnostics_by_seed(diagnostics)

    for axis, (metric, title, ylabel) in zip(
        axes.ravel(),
        panels,
        strict=True,
    ):
        metric_index = METRIC_INDEX[metric]
        for method_index, method in enumerate(methods):
            color = METHOD_COLORS[method]
            estimates = []
            lows = []
            highs = []
            for selected_index, alpha in enumerate(selected_alphas):
                alpha_index = int(
                    np.flatnonzero(
                        np.isclose(
                            alphas,
                            alpha,
                            rtol=0.0,
                            atol=1.0e-12,
                        )
                    )[0]
                )
                seed_values = per_seed_values[
                    metric_index,
                    method_index,
                    alpha_index,
                ]
                center = x[selected_index] + method_offsets[method_index]
                axis.scatter(
                    center + seed_jitter,
                    seed_values,
                    color=color,
                    alpha=0.24,
                    s=10,
                    edgecolors="none",
                )
                estimate, low, high = overall_estimates[
                    (metric, method, float(alpha))
                ]
                estimates.append(estimate)
                lows.append(low)
                highs.append(high)
            estimates_array = np.asarray(estimates)
            error = np.vstack(
                [
                    estimates_array - np.asarray(lows),
                    np.asarray(highs) - estimates_array,
                ]
            )
            axis.errorbar(
                x + method_offsets[method_index],
                estimates_array,
                yerr=error,
                fmt="o",
                color=color,
                capsize=3,
                markersize=5,
                label=METHOD_LABELS[method],
            )
        if metric == "mean_relative_error_reduction_unobserved":
            axis.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        elif metric == "forecast_cross_cov_ratio":
            axis.set_ylim(bottom=0.0)
        axis.set_xticks(
            x,
            [rf"$\alpha={alpha_label(alpha)}$" for alpha in selected_alphas],
        )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2, linewidth=0.5)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "Preparatory evaluation - Page 3: information transfer into U",
        fontsize=15,
        y=0.97,
    )
    fig.text(0.5, 0.935, condition_label, ha="center", va="center", fontsize=9)
    fig.text(
        0.5,
        0.045,
        (
            "Seed-level summaries use n=1,...,1000; panel 4 is "
            r"$1-\overline{\mathrm{MSE}^a_U}/"
            r"\overline{\mathrm{MSE}^f_U}$; large markers show the mean and "
            "95% bootstrap CI.\n"
            "Panel 3 removes perturbed-observation noise; panel 4 uses the "
            "actual analysis and includes its finite-member mean.\n"
            "The fourth diagnostic is an immediate within-method change, not "
            "a long-time between-method comparison."
        ),
        ha="center",
        va="center",
        fontsize=8,
    )
    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        bottom=0.13,
        top=0.84,
        hspace=0.34,
        wspace=0.25,
    )
    pdf.savefig(fig)
    plt.close(fig)


def plot_unobserved_error_timeseries_page(
    pdf: PdfPages,
    diagnostics: np.ndarray,
    methods: tuple[str, ...],
    alphas: np.ndarray,
    time_index: np.ndarray,
    condition_label: str,
    linear_mse: bool,
) -> None:
    """Page 4: time-resolved U-improvement and O/U forecast-analysis MSE."""
    alpha = preferred_alphas(
        alphas,
        preferred=(0.25, 0.5, 2.0),
        max_count=1,
    )[0]
    alpha_idx = int(
        np.flatnonzero(
            np.isclose(alphas, alpha, rtol=0.0, atol=1.0e-12)
        )[0]
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.69, 8.27))
    ax_g, ax_mse_u, ax_mse_o = axes
    time_ticks = time_index + 1
    g_idx = METRIC_INDEX[
        "mean_relative_error_reduction_unobserved"
    ]

    for k, method_name in enumerate(methods):
        color = METHOD_COLORS[method_name]
        g = diagnostics[g_idx, k, alpha_idx]
        g_mean = g.mean(axis=0)
        g_low, g_high = np.quantile(
            g,
            [0.1, 0.9],
            axis=0,
        )
        ax_g.fill_between(
            time_ticks,
            g_low,
            g_high,
            color=color,
            alpha=0.14,
            linewidth=0.0,
        )
        ax_g.plot(
            time_ticks,
            g_mean,
            color=color,
            label=METHOD_LABELS[method_name],
        )

        for ax, scope in (
            (ax_mse_u, "unobserved"),
            (ax_mse_o, "observed"),
        ):
            for stage, ls in (("forecast", "--"), ("analysis", "-")):
                mse_idx = METRIC_INDEX[f"{stage}_mean_mse_{scope}"]
                mse = diagnostics[mse_idx, k, alpha_idx]
                mse_mean = mse.mean(axis=0)
                mse_low, mse_high = np.quantile(mse, [0.1, 0.9], axis=0)
                ax.fill_between(
                    time_ticks,
                    mse_low,
                    mse_high,
                    color=color,
                    alpha=0.07,
                    linewidth=0.0,
                )
                ax.plot(
                    time_ticks,
                    mse_mean,
                    color=color,
                    linestyle=ls,
                    linewidth=1.2,
                    label=f"{METHOD_LABELS[method_name]} {stage}",
                )

    ax_g.axhline(
        0.0,
        color="black",
        linestyle=":",
        linewidth=0.8,
    )
    ax_g.set_yscale("symlog", linthresh=1.0e-3)
    ax_g.set_title(
        rf"Relative forecast-to-analysis change, "
        rf"$\alpha={alpha_label(alpha)}$"
    )
    ax_g.set_xlabel("assimilation step $n$")
    ax_g.set_ylabel(
        r"$G_{U,n}^{\mathrm{mean}}"
        r"=1-\mathrm{MSE}_{U,n}^a/\mathrm{MSE}_{U,n}^f$"
    )
    ax_g.grid(alpha=0.2, linewidth=0.5)
    for ax, scope in (
        (ax_mse_u, "U"),
        (ax_mse_o, "O"),
    ):
        if not linear_mse:
            ax.set_yscale("log")
        ax.set_title(
            rf"Forecast and analysis $\mathrm{{MSE}}_{scope}$, "
            rf"$\alpha={alpha_label(alpha)}$"
        )
        ax.set_xlabel("assimilation step $n$")
        ax.set_ylabel(rf"ensemble-mean $\mathrm{{MSE}}_{scope}$")
        ax.grid(alpha=0.2, linewidth=0.5)

    g_handles, g_labels = ax_g.get_legend_handles_labels()
    mse_handles, mse_labels = ax_mse_u.get_legend_handles_labels()
    fig.legend(
        g_handles + mse_handles,
        g_labels + mse_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.865),
        ncol=6,
        frameon=False,
    )
    fig.suptitle(
        "Preparatory evaluation - Page 4: time-resolved O/U error",
        fontsize=15,
        y=0.95,
    )
    fig.text(0.5, 0.91, condition_label, ha="center", va="center", fontsize=9)
    fig.text(
        0.5,
        0.055,
        (
            "Lines are seed means; shaded bands are the 10th-90th seed "
            "percentiles. The G panel uses a symmetric log scale around zero; "
            f"the MSE panels use a {'linear' if linear_mse else 'log'} scale. "
            "Forecast is dashed and analysis is solid."
        ),
        ha="center",
        va="center",
        fontsize=8,
    )
    fig.subplots_adjust(
        left=0.06,
        right=0.98,
        bottom=0.13,
        top=0.77,
        wspace=0.28,
    )
    pdf.savefig(fig)
    plt.close(fig)


def write_evaluation_pdf(
    path: Path,
    diagnostics: np.ndarray,
    methods: tuple[str, ...],
    alphas: np.ndarray,
    time_index: np.ndarray,
    reference: float,
    overall_estimates: dict[
        tuple[str, str, float],
        tuple[float, float, float],
    ],
    paired_ratio_estimates: dict[
        tuple[str, float],
        tuple[float, float, float],
    ],
    condition_label: str,
    linear_mse: bool,
) -> None:
    """Create the four-page preparatory evaluation report."""
    configure_plot_style()
    with PdfPages(
        path,
        metadata={
            "Title": "Preparatory Lorenz 96 PO evaluation",
            "Subject": (
                "Diagnostic prototype; not the manuscript or public "
                "reproduction figure set"
            ),
            "Creator": "evaluate.py",
        },
    ) as pdf:
        plot_theory_page(
            pdf,
            diagnostics,
            methods,
            alphas,
            time_index,
            reference,
            condition_label,
        )
        plot_mse_ratio_page(
            pdf,
            diagnostics,
            alphas,
            paired_ratio_estimates,
            condition_label,
        )
        plot_mechanism_page(
            pdf,
            diagnostics,
            methods,
            alphas,
            overall_estimates,
            condition_label,
        )
        plot_unobserved_error_timeseries_page(
            pdf,
            diagnostics,
            methods,
            alphas,
            time_index,
            condition_label,
            linear_mse,
        )


def repository_metadata(repository: Path) -> tuple[str, bool]:
    """Return the current Git commit and dirty-worktree state."""
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return sha, bool(status.strip())


def source_sha256() -> str:
    """Hash this evaluator because it may be uncommitted at execution time."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def build_metadata(
    data_dir: Path,
    parameters: dict,
    truth: np.ndarray,
    observations: np.ndarray,
    analysis_shapes: dict[str, tuple[int, ...]],
    observed: np.ndarray,
    unobserved: np.ndarray,
    diagnostics: np.ndarray,
    time_index: np.ndarray,
    qc: dict[str, float],
    git_sha: str,
    git_dirty: bool,
    linear_mse: bool,
) -> dict:
    """Assemble provenance, definitions, settings, and QC results."""
    reference = 4.0 * len(observed) * float(parameters["obs_noise_std"]) ** 2
    return {
        "schema_version": 1,
        "artifact_status": "preparatory_diagnostic",
        "intended_use": (
            "Metric and experiment-design validation only. Manuscript figures "
            "and public reproduction code will be generated from a separately "
            "polished implementation after the design is fixed."
        ),
        "repository_git_sha": git_sha,
        "repository_dirty_at_evaluation": git_dirty,
        "evaluate_py_sha256": source_sha256(),
        "data_dir": str(data_dir.resolve()),
        "input_shapes": {
            "true_trajectory": list(truth.shape),
            "observations": list(observations.shape),
            "analysis_ensembles": {
                method: list(shape) for method, shape in analysis_shapes.items()
            },
        },
        "input_dtypes": {
            "true_trajectory": str(truth.dtype),
            "observations": str(observations.dtype),
            "analysis_ensembles": "float64",
        },
        "run_parameters": parameters,
        "observation_partition": {
            "definition": (
                "Observed indices are read from run_parameters.json when "
                "present; the original main.py pattern is used only as a "
                "backward-compatible fallback."
            ),
            "observed_indices_python_zero_based": observed.tolist(),
            "unobserved_indices_python_zero_based": unobserved.tolist(),
            "observed_count": int(len(observed)),
            "unobserved_count": int(len(unobserved)),
        },
        "evaluation": {
            "python_slice": [
                int(EVALUATION_START),
                int(EVALUATION_STOP),
            ],
            "python_time_index": [
                int(time_index[0]),
                int(time_index[-1]),
            ],
            "manuscript_assimilation_steps": [
                int(time_index[0] + 1),
                int(time_index[-1] + 1),
            ],
            "time_count": int(len(time_index)),
            "page_4_mse_y_scale": "linear" if linear_mse else "log",
            "forecast_reconstruction": (
                "Xf[0] = M(Xinit) and Xf[n] = M(Xa[n-1]) for n>=1, using "
                "the same seed-reconstructed initial ensemble and RK4 "
                "Lorenz 96 step as main.py."
            ),
            "initial_ensemble_variance": INITIAL_ENSEMBLE_VARIANCE,
            "initial_ensemble_reconstruction": (
                "For each seed, regenerate the complete observation sequence, "
                "require exact equality with observations.npy, then continue "
                "the same RandomState stream to reconstruct Xinit."
            ),
            "qc_atol": QC_TOLERANCE,
            "qc_rtol": QC_TOLERANCE,
        },
        "bootstrap": {
            "resampling_unit": "seed",
            "pairing": (
                "Method comparisons resample aligned add/add-proj seed pairs."
            ),
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "ci_level": CI_LEVEL,
            "ratio_method": (
                "Bootstrap the seed-wise log ratio of time means, then "
                "exponentiate."
            ),
        },
        "diagnostics": {
            "axis_order": [
                "metric",
                "method",
                "alpha",
                "seed",
                "time",
            ],
            "shape": list(diagnostics.shape),
            "metric_definitions": metric_definitions(),
        },
        "reference_4no_r2": {
            "value": reference,
            "formula": "4*N_O*r^2",
            "interpretation": (
                "Reference used in the original Figure 1 after ignoring theta. "
                "The theorem instead uses N=min(m-1,N_O) and includes the "
                "factor 1/(1-theta)."
            ),
            "theorem_rank_parameter": int(
                min(int(parameters["ensemble_size"]) - 1, len(observed))
            ),
        },
        "quality_control": {
            "status": "passed",
            **{key: float(value) for key, value in sorted(qc.items())},
        },
    }


def validate_input_shapes(
    truth: np.ndarray,
    observations: np.ndarray,
    analysis_arrays: dict[str, np.ndarray],
    parameters: dict,
    observed: np.ndarray,
    unobserved: np.ndarray,
) -> None:
    """Check every cache shape and the observed/unobserved partition."""
    state_dim = int(parameters["state_dim"])
    num_steps = int(parameters["num_steps"])
    ensemble_size = int(parameters["ensemble_size"])
    alpha_count = len(parameters["alpha_list_all"])
    seed_count = len(parameters["seed_list"])
    observed_count = observations.shape[-1]

    expected_truth_shape = (num_steps, state_dim)
    expected_observation_shape = (seed_count, num_steps, len(observed))
    expected_analysis_shape = (
        alpha_count,
        seed_count,
        num_steps,
        ensemble_size,
        state_dim,
    )
    if truth.shape != expected_truth_shape:
        raise ValueError(
            f"Truth shape {truth.shape}; expected {expected_truth_shape}."
        )
    if observations.shape != expected_observation_shape:
        raise ValueError(
            f"Observation shape {observations.shape}; "
            f"expected {expected_observation_shape}."
        )
    for method, analysis in analysis_arrays.items():
        if analysis.shape != expected_analysis_shape:
            raise ValueError(
                f"{method} analysis shape {analysis.shape}; "
                f"expected {expected_analysis_shape}."
            )

    combined = np.sort(np.concatenate([observed, unobserved]))
    expected_combined = np.arange(state_dim)
    if np.intersect1d(observed, unobserved).size:
        raise ValueError("Observed and unobserved index sets overlap.")
    if not np.array_equal(combined, expected_combined):
        raise ValueError("Observed/unobserved indices do not cover the state.")
    if observed_count != len(observed):
        raise ValueError(
            f"Observation dimension {observed_count} does not match "
            f"N_O={len(observed)}."
        )
    if len(unobserved) == 0:
        raise ValueError("At least one unobserved state is required.")


def main() -> None:
    """Run preparatory evaluation, runtime QC, and three-file export."""
    args = parse_args()
    data_dir = Path(args.data_dir)
    parameters = load_parameters(data_dir)
    ensure_expected_configuration(parameters)

    methods = tuple(parameters["methods"])
    alphas = np.asarray(parameters["alpha_list_all"], dtype=float)
    seeds = np.asarray(parameters["seed_list"], dtype=int)
    observed, unobserved = observed_state_indices(parameters)
    time_index = np.arange(EVALUATION_START, EVALUATION_STOP, dtype=int)

    truth_path = data_dir / "true_trajectory.npy"
    observations_path = data_dir / "observations.npy"
    truth = np.load(truth_path, mmap_mode="r")
    observations = np.load(observations_path, mmap_mode="r")
    analysis_arrays = {
        method: np.load(
            data_dir / method / "analysis_ensembles.npy",
            mmap_mode="r",
        )
        for method in methods
    }
    validate_input_shapes(
        truth,
        observations,
        analysis_arrays,
        parameters,
        observed,
        unobserved,
    )
    require_finite("true trajectory", truth)
    require_finite("observations", observations)

    qc: dict[str, float] = {}
    initial_ensembles = reconstruct_initial_ensembles(
        truth=truth,
        observations=observations,
        observed=observed,
        seeds=seeds,
        obs_noise_std=float(parameters["obs_noise_std"]),
        ensemble_size=int(parameters["ensemble_size"]),
        qc=qc,
    )
    first_analysis_previous = initial_ensembles[0]
    vectorized_forecast = forecast_batch(
        first_analysis_previous[None, :, :],
        forcing=float(parameters["forcing"]),
        dt=float(parameters["dt"]),
        obs_per=int(parameters["obs_per"]),
    )[0]
    scalar_forecast = forecast_scalar_reference(
        first_analysis_previous,
        forcing=float(parameters["forcing"]),
        dt=float(parameters["dt"]),
        obs_per=int(parameters["obs_per"]),
    )
    if not np.allclose(
        vectorized_forecast,
        scalar_forecast,
        atol=QC_TOLERANCE,
        rtol=QC_TOLERANCE,
    ):
        max_error = float(np.max(np.abs(vectorized_forecast - scalar_forecast)))
        raise AssertionError(
            "Vectorized forecast does not reproduce the main.py member loop; "
            f"max absolute error={max_error:.6g}."
        )

    expected_shape = (
        len(METRIC_NAMES),
        len(methods),
        len(alphas),
        len(seeds),
        len(time_index),
    )
    diagnostics = np.empty(expected_shape, dtype=float)
    qc["vectorized_forecast_max_absolute_error"] = float(
        np.max(np.abs(vectorized_forecast - scalar_forecast))
    )

    print(
        "Evaluating",
        f"{len(methods)} methods x {len(alphas)} alphas x "
        f"{len(seeds)} seeds x {len(time_index)} times",
    )
    for method_index, method in enumerate(methods):
        analysis_cache = analysis_arrays[method]
        for alpha_index, alpha in enumerate(alphas):
            for seed_index, _seed in enumerate(seeds):
                full_analysis = analysis_cache[alpha_index, seed_index]
                require_finite(
                    f"{method}, alpha={alpha:g}, seed={int(_seed)} analysis",
                    full_analysis,
                )
                analysis_previous = np.empty_like(
                    full_analysis[EVALUATION_START:EVALUATION_STOP]
                )
                analysis_previous[0] = initial_ensembles[seed_index]
                analysis_previous[1:] = full_analysis[
                    EVALUATION_START : EVALUATION_STOP - 1
                ]
                analysis = full_analysis[
                    EVALUATION_START:EVALUATION_STOP
                ]
                truth_chunk = truth[EVALUATION_START:EVALUATION_STOP]
                observation_chunk = observations[
                    seed_index,
                    EVALUATION_START:EVALUATION_STOP,
                ]
                diagnostics[
                    :,
                    method_index,
                    alpha_index,
                    seed_index,
                    :,
                ] = evaluate_chunk(
                    analysis_previous=analysis_previous,
                    analysis=analysis,
                    truth=truth_chunk,
                    observations=observation_chunk,
                    observed=observed,
                    unobserved=unobserved,
                    method=method,
                    alpha=float(alpha),
                    forcing=float(parameters["forcing"]),
                    dt=float(parameters["dt"]),
                    obs_per=int(parameters["obs_per"]),
                    obs_noise_std=float(parameters["obs_noise_std"]),
                    qc=qc,
                )
            print(
                f"  completed {method}, alpha={alpha_label(float(alpha))} "
                f"({len(seeds)} seeds)"
            )

    if diagnostics.shape != expected_shape:
        raise AssertionError(
            f"Diagnostic shape {diagnostics.shape}; expected {expected_shape}."
        )
    require_finite("complete diagnostics", diagnostics)

    rows, overall_estimates, paired_ratio_estimates = build_summary(
        diagnostics,
        methods,
        alphas,
        seeds,
    )
    required_ratio_keys = {
        (f"analysis_mean_mse_{scope}", float(alpha))
        for scope in SCOPES
        for alpha in alphas
    }
    missing_ratio_keys = required_ratio_keys.difference(paired_ratio_estimates)
    if missing_ratio_keys:
        raise AssertionError(
            "Positive MSE ratios were not available for "
            f"{sorted(missing_ratio_keys)}."
        )

    repository = Path(__file__).resolve().parent
    git_sha, git_dirty = repository_metadata(repository)
    metadata = build_metadata(
        data_dir=data_dir,
        parameters=parameters,
        truth=truth,
        observations=observations,
        analysis_shapes={
            method: tuple(array.shape)
            for method, array in analysis_arrays.items()
        },
        observed=observed,
        unobserved=unobserved,
        diagnostics=diagnostics,
        time_index=time_index,
        qc=qc,
        git_sha=git_sha,
        git_dirty=git_dirty,
        linear_mse=args.linear_mse,
    )
    metadata_json = json.dumps(
        metadata,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    output_dir = data_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    unexpected_existing = {
        path.name for path in output_dir.iterdir()
    }.difference(OUTPUT_FILENAMES)
    if unexpected_existing:
        raise RuntimeError(
            "Refusing to mix evaluation outputs with unexpected files: "
            f"{sorted(unexpected_existing)}"
        )

    results_path = output_dir / "evaluation_results.npz"
    summary_path = output_dir / "evaluation_summary.csv"
    pdf_path = output_dir / "evaluation.pdf"
    np.savez_compressed(
        results_path,
        diagnostics=diagnostics,
        metric_names=np.asarray(METRIC_NAMES),
        methods=np.asarray(methods),
        alphas=alphas,
        seeds=seeds,
        time_index=time_index,
        metadata_json=np.asarray(metadata_json),
    )
    write_summary_csv(summary_path, rows)
    reference = 4.0 * len(observed) * float(parameters["obs_noise_std"]) ** 2
    observation_fraction = Fraction(len(observed), int(parameters["state_dim"]))
    condition_label = (
        f"Observation density: {len(observed)}/{int(parameters['state_dim'])} "
        f"= {observation_fraction.numerator}/{observation_fraction.denominator}; "
        f"noise std: {float(parameters['obs_noise_std']):g}; "
        f"ensemble size: {int(parameters['ensemble_size'])}"
    )
    write_evaluation_pdf(
        path=pdf_path,
        diagnostics=diagnostics,
        methods=methods,
        alphas=alphas,
        time_index=time_index,
        reference=reference,
        overall_estimates=overall_estimates,
        paired_ratio_estimates=paired_ratio_estimates,
        condition_label=condition_label,
        linear_mse=args.linear_mse,
    )

    actual_outputs = {path.name for path in output_dir.iterdir()}
    if actual_outputs != set(OUTPUT_FILENAMES):
        raise AssertionError(
            f"Output set {sorted(actual_outputs)} does not equal "
            f"{sorted(OUTPUT_FILENAMES)}."
        )
    for output_path in (results_path, summary_path, pdf_path):
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise AssertionError(f"Missing or empty output: {output_path}")
        print("saved:", output_path)
    print("All runtime checks passed.")


if __name__ == "__main__":
    main()
