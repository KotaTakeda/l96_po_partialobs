# Error Analysis of the Stochastic Ensemble Kalman Filter for the Partially Observed Lorenz 96 Model

This repository contains the reproduction code for the numerical experiments in the manuscript:

> K. Takeda,
> *Error analysis of the stochastic ensemble Kalman filter for the partially observed Lorenz 96 model with and without the covariance projection*,
> under review.


The script [`main.py`](./main.py) runs the perturbed observation (PO) filter with

- additive inflation (`add`)
- projected additive inflation (`add-proj`)

for the partially observed Lorenz 96 model, and saves the cached arrays and manuscript figures.

## Setup

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

This installs the standard scientific Python stack used in the experiments and also installs [`da_py`](https://github.com/KotaTakeda/da_py) from GitHub, which provides the Lorenz 96 model and data assimilation implementations used by this repository.

## Usage

Run the experiment with:

```bash
python main.py --data-dir data/reproduce
```

By default, the script reuses cached `.npy` files in `--data-dir` when they already exist and regenerates only missing files.

Force regeneration of the cached arrays with:

```bash
python main.py --data-dir data/reproduce --recompute
```

The main experiment is executed in one run with

- `J = 60`
- `F = 8.0`
- `dt = 0.01`
- observation interval `h = 0.01`
- observation noise standard deviation `r = 1.0`
- ensemble size `m = 10`
- seeds `0, ..., 19`
- inflation parameters `alpha = 0.0, 0.5, 2.0, 10.0, 100.0`

The sample-path plots use `seed = 0` and ensemble member `k = 1` in the manuscript notation, which corresponds to index `0` in Python.

## Outputs

The script stores cached arrays such as

- `run_parameters.json`
- `true_trajectory.npy`
- `observations.npy`
- `po_add/analysis_ensembles.npy`
- `po_proj/analysis_ensembles.npy`

and saves the figures below into `--data-dir`.

| Manuscript figure | Output file |
| --- | --- |
| Figure 1: MSE time series | `fig1_mse.pdf` |
| Figure 2: Spatio-temporal absolute error | `fig2_abs_error.pdf` |
| Figure 3: Normalized and rearranged covariance matrices | `fig3_covariance.pdf` |
| Figure 4: Off-diagonal ratio time series | `fig4_offdiag_ratio.pdf` |
| Figure 5: True state and observation | `fig5_true_obs.pdf` |
| Figure 6: Analysis states | `fig6_analysis_states.pdf` |

## Notes

- The observation operator follows the partial observation pattern used in the manuscript: two observed components followed by one unobserved component.
- Figure 1 uses the MSE following the notation in the paper as $\|\bm{\delta}\|^2 = |\bm{\delta}|^2 + |\Pi\bm{\delta}|^2$, where $|\cdot|$ denotes the standard Euclidean norm.
