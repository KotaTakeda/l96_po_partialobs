# Error Analysis of the Stochastic Ensemble Kalman Filter for the Partially Observed Lorenz 96 Model

This repository contains the reproduction code for the numerical experiments in

> K. Takeda, *Observed-unobserved information transfer in ensemble Kalman filtering for the partially observed Lorenz 96 model*, under review.

[`main.py`](./main.py) runs the perturbed-observation EnKF with additive
inflation (`add`) and projected additive inflation (`add-proj`). It generates
the manuscript figures and the covariance-projection comparison.

| Module | Role |
| --- | --- |
| [`main.py`](./main.py) | Experiment configuration, assimilation runs, caching, summary tables, command line |
| [`figures.py`](./figures.py) | Weighted mean squared error, seed-variability band, plot style, and the Figure 1--7 renderers |

`main.py` imports `figures.py`, and `figures.py` imports nothing from this
repository, so the dependency is one-directional.

## Setup

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The dependencies are `numpy`, `matplotlib`, `seaborn`, and
[`da_py`](https://github.com/KotaTakeda/da_py), which provides the Lorenz 96
model and the data assimilation routines. `seaborn` is required because
Figure 2 uses its `flare` colormap.

Run the commands below from the repository root: the plot style is loaded as
`vis.mplstyle`, which is resolved relative to the working directory.

## Reproduce all figures

```bash
.venv/bin/python main.py --data-dir data/reproduce
```

The command runs or reuses the following data directories:

1. Figures 1--4 and 6--7: $2/3$-pattern observations, $m=10$,
   $\alpha=0,0.5,2,10,100$, and 20 seeds. Figures 1, 2, and 7 display only
   $\alpha=0,0.5,2$; Figures 3 and 4 use all five values.
2. Figure 5 projection comparison: $1/2$-pattern (alternating) observations, $\alpha=0.5$,
   $m=10,20,40,80$, and 20 paired seeds.

The common parameters are the state dimension $J=60$, the forcing $F=8$, the
integration time step $\Delta t=0.01$, observation-noise standard deviation $r=1$, and seeds $0,\ldots,19$.
Figures 2 and 7 show seed 0 and ensemble member $k=1$ (Python index 0).

By default, all uncertainty displays use the mean over seeds and the
empirical 2.5th--97.5th percentile range across the 20 seeds. In time-series
figures this range is evaluated pointwise.

To additionally overlay all seed-wise sample paths with opacity 0.3:

```bash
.venv/bin/python main.py --data-dir data/reproduce --show-sample-paths
```

This display-only option does not change or recompute the experiment cache. It
overwrites the same PDF files with the sample-path overlay version.

To regenerate every data directory instead of reusing it:

```bash
.venv/bin/python main.py --data-dir data/reproduce --recompute
```

## Cache contract

Each data directory passed to `run_experiment` is cached as a single unit: it is written, reused, and regenerated as a whole,
never file by file. The public interface is the command line above; internal
functions and configuration dictionaries may change.

- A successful data directory has a `.complete` marker.
- If the marker exists, arrays are reused directly.
- If the marker is absent, the whole data directory is recomputed.
- `--recompute` regenerates the whole data directory.
- `run_parameters.json` records provenance; it is not a second validation
  state machine.
- If a marked data directory is damaged or incomplete, rerun with `--recompute`.

Do not mix individual cache files between data directories. In
particular, the projection comparisons use paired seeds and must retain their
common generation conditions.

## Outputs

A fresh run creates the following main outputs under `data/reproduce/`:

```text
data/reproduce/
├── fig1_mse.pdf
├── fig1_summary.csv
├── fig2_abs_error.pdf
├── fig3_covariance.pdf
├── fig4_offdiag_ratio.pdf
├── fig5_projection_comparison.pdf
├── fig5_summary.csv
├── fig6_true_obs.pdf
├── fig7_analysis_states.pdf
└── projection_comparison/
    ├── summary.csv
    ├── m10/
    ├── m20/
    ├── m40/
    └── m80/
```

The tree lists only the figures and tables. Alongside them, `data/reproduce/`
is itself a data directory, and every data directory also contains
`run_parameters.json`, `true_trajectory.npy`, `observations.npy`,
`initial_ensembles.npy`, one `analysis_ensembles.npy` per method under
`po_add/` and `po_proj/`, and a `.complete` marker. Projection-comparison
data directories additionally contain `diagnostics.npz`, whose arrays are

| Key | Shape | Content |
| --- | --- | --- |
| `mean_increment_sq_unobserved` | (method, alpha, seed, time) | Squared unobserved ensemble-mean analysis increment, averaged over the unobserved components |
| `member_increment_max_abs_unobserved` | (method, alpha, seed) | Largest absolute per-member unobserved increment, kept as a numerical quality-control residual |
| `method_names`, `alphas`, `seeds`, `time_index` | — | Coordinates for the arrays above |

A complete run from scratch writes about 3.6 GB, most of it the analysis
ensembles, and takes roughly 40 minutes of wall-clock time on an Apple M4 Max
(7 minutes of CPU time; the run is I/O bound, dominated by writing the cached
arrays).
Reusing the cache instead regenerates the figures and tables in seconds.

### Manuscript figures

| Output | Content |
| --- | --- |
| `fig1_mse.pdf` | Time series of the weighted mean squared error of the manuscript |
| `fig1_summary.csv` | Table-ready aggregate statistics corresponding to Figure 1 |
| `fig2_abs_error.pdf` | Spatio-temporal absolute error |
| `fig3_covariance.pdf` | Normalized, rearranged covariance matrices |
| `fig4_offdiag_ratio.pdf` | Off-diagonal/observed covariance ratio |
| `fig5_projection_comparison.pdf` | Covariance-projection comparison |
| `fig5_summary.csv` | Table-ready aggregate statistics corresponding to Figure 5 |
| `fig6_true_obs.pdf` | Truth and observations |
| `fig7_analysis_states.pdf` | Sample analysis states |

Figure 1 plots

$$
\frac{1}{m}\sum_{k=1}^m
\left(
|\boldsymbol{\delta}_n^{(k)}|^2+
|\Pi\boldsymbol{\delta}_n^{(k)}|^2
\right)
$$

on a logarithmic axis. The emphasized curve is the mean over seeds and the
shaded region is the pointwise empirical 2.5th--97.5th percentile range. The
`--show-sample-paths` option additionally overlays all 20 seed-wise results.

`fig1_summary.csv` is a machine-readable source for the manuscript table. For each
method and inflation value, the Figure 1 weighted mean squared error is first averaged over
analysis times $n=501,\ldots,1000$ within each seed. The `mse` column reports
the mean and the empirical 2.5th and 97.5th percentiles of those seed-wise time averages, along
with the number of seeds, the included analysis-time range, and the reference
level $4N_y r^2$. Thus, its percentiles summarize time-aggregated seed values;
they are distinct from the pointwise percentile ranges shown in Figure 1.

Figure 4 plots the seed mean of the off-diagonal/observed covariance ratio

$$
\mathcal{R}_n = \frac{|(I-\Pi)P_n\Pi|_F}{|\Pi P_n\Pi|_F},
$$

as defined in the manuscript (Section 4, preceding Figure 4). The emphasized
curves are the mean over seeds, which approximates $\mathbb{E}[\mathcal{R}_n]$,
and the shaded regions are the pointwise empirical 2.5th--97.5th percentile
ranges. The seed-wise paths are optional.

### Projection-comparison figure

`fig5_projection_comparison.pdf` has two panels, both using the
$1/2$-pattern observation experiment, $\alpha=0.5$, $m\in\{10,20,40,80\}$, and 20
seeds. All scalar time averages use all available analysis times $n=1,\ldots,1000$.

- **(a)** Plots the root mean square over time of the analysis increment on the
  unobserved subspace,

  $$
  \left[\frac{1}{N_t}\sum_{n=1}^{N_t}\left(I_n^{\mathrm{U}}\right)^2\right]^{1/2},
  \qquad
  I_n^{\mathrm{U}}=
  \left(
  \frac{1}{\#\mathcal{I}^{\mathrm{U}}}\sum_{i\in\mathcal{I}^{\mathrm{U}}}
  \left|\left(\overline{\widehat{v}}_n\right)^i-\left(\overline{v}_n\right)^i\right|^2
  \right)^{1/2},
  $$

  where $\mathcal{I}^{\mathrm{U}}$ indexes the unobserved components and the
  overline denotes the ensemble mean, following the manuscript definition.

  Seed values, their mean, and the empirical 2.5th--97.5th percentile
  range are displayed for both `add` and `add-proj`. The `add-proj` value is
  exactly zero by construction (projection removes the direct unobserved analysis
  increment), so its seed points, mean marker, and error bars all appear at zero.

- **(b)** Plots the per-seed paired ratio of the time-averaged weighted mean squared
  error,
  `add`/`add-proj`. Scalar time averages use all available analysis times
  $n=1,\ldots,1000$.
  Black points are seed-wise paired ratios and the black error bar gives their
  mean and empirical 2.5th--97.5th percentile range. The vertical
  axis is linear starting at zero; the dashed line marks ratio 1.

The figure compares how covariance projection changes the information-transfer
mechanism and accuracy in selected conditions. It is not intended as a general
algorithm-ranking benchmark. A nonzero `add` increment in panel (a) measures
the amount of direct transfer; by itself it does not prove an immediate error
improvement.

`fig5_summary.csv` is a machine-readable source for the manuscript table accompanying
Figure 5, in the same role that `fig1_summary.csv` plays for Figure 1. It is
wide-form with one row per ensemble size and the columns

```text
ensemble_size,alpha,increment_rms,increment_rms_p2_5,increment_rms_p97_5,
mse_ratio,mse_ratio_p2_5,mse_ratio_p97_5,num_seeds,analysis_time_start,analysis_time_end
```

`increment_rms` is the panel (a) quantity for `add` and `mse_ratio` is the
panel (b) paired ratio; both are reported as the mean over seeds with the
empirical 2.5th and 97.5th percentiles, time-averaged over all analysis times
$n=1,\ldots,1000$. The `add-proj` increment is identically zero by construction
and therefore has no column.

The two files differ in role: `fig5_summary.csv` is the wide, table-ready
extract of exactly what Figure 5 shows, whereas
`projection_comparison/summary.csv` below is the long-form record retaining
every per-seed value and additional metrics.

### Numerical summary

`projection_comparison/summary.csv` is the long-form record of the same
experiment. Its columns are
`record_type,ensemble_size,alpha,seed,method,metric,estimate,p2_5,p97_5`.
Aggregate rows report the mean over seeds and the empirical 2.5th and 97.5th
percentiles; per-seed rows leave the percentile columns empty.
All scalar time averages in this summary use all available analysis times
$n=1,\ldots,1000$. (Figure 1 summary uses $n=501,\ldots,1000$; see `fig1_summary.csv`.)

The retained metrics are:

- `mean_increment_rms_unobserved`;
- `weighted_norm_error`;
- `weighted_norm_error_ratio`, the seed-paired `add/add-proj` ratio.

`diagnostics.npz` additionally stores
`member_increment_max_abs_unobserved` as a numerical quality-control residual.
No fixed threshold is used to turn this residual into an exception.

<!-- ## Reproducibility

Runs are seeded, so rerunning the same command reproduces the reported values.
Two caveats are worth stating.

The inflated cases reproduce to machine precision. Rerunning everything with
`--recompute` reproduced `fig5_summary.csv` and
`projection_comparison/summary.csv` bit for bit, and reproduced the
$\alpha=0.5$ and $\alpha=2$ entries of `fig1_summary.csv` to the last digit.

The uninflated case $\alpha=0$ does not. Its time-averaged error agreed only to
about four significant digits (a relative difference of $2\times10^{-4}$).
This is expected rather than a defect: without inflation the sample covariance
is rank deficient and the filter does not contract, so rounding differences,
for instance from the multithreaded summation order in the BLAS backend, grow
with the chaotic dynamics. It is the same instability that the $\alpha=0$
curves in Figure 1 are there to show, and it does not affect any reported
conclusion, since the $\alpha=0$ error exceeds the reference level $4N_yr^2$ by
more than an order of magnitude in either case.

Set `OMP_NUM_THREADS=1` (and `VECLIB_MAXIMUM_THREADS=1` on macOS) before
running if you need a stricter bitwise match for the uninflated case. -->
