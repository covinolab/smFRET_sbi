# smFRET-SBI

Simulation-based inference (SBI) for single-molecule FRET experiments. This package infers the parameters of a molecular system — its free-energy landscape, diffusion coefficient, and photophysical properties — directly from photon count time-series, without relying on hand-crafted likelihood functions.

## Overview

The pipeline has three stages:

1. **Simulate** — A Cython-accelerated smFRET simulator generates synthetic photon traces from a spline-parameterised free-energy potential and full FRET photophysics (cross-talk, background, detector efficiencies).
2. **Train** — Neural density estimators learn the mapping from simulated observations to posterior distributions over the parameters of interest.
3. **Analyse** — Trained estimators are applied to experimental (or held-out) data to obtain posterior samples.

## Project Structure

```
smFRET_sbi/
├── src/
│   ├── smFRET_simulator.pyx       # Core simulator (Cython, calls GSL splines)
│   ├── gsl_spline.pyx             # GSL cubic-spline wrapper
│   ├── _gsl_spline.pxd            # Cython declaration for GSL spline
│   ├── gsl_random_numbers.pxd     # Cython declaration for GSL RNG
│   ├── build_cython.py            # Compile Cython extensions in-place
│   ├── simulator.py               # Python wrapper around the Cython simulator
│   ├── priors.py                  # Prior definitions (uniform + correlated spline)
│   ├── utils.py                   # Spline-knot construction & evaluation helpers
│   ├── data_loaders.py            # HDF5 iterable dataset with chunk-based I/O
│   ├── temporal_encoders.py       # 1-D CNN embedding network (ResNet-style)
│   ├── generate_simulations.py    # CLI for parallel data generation → HDF5
│   └── train_density_estimator.py # NPE training script (normalizing flow)
├── conf/
│   ├── simulator/                 # Simulator & prior configs (Hydra, YAML)
│   │   ├── default.yaml           # Default config (tracked)
│   │   └── test.yaml              # Test config (tracked)
│   └── training/                  # Training configs (Hydra, YAML)
│       ├── density_estimator.yaml # Default training config (tracked)
│       └── test.yaml              # Test training config (tracked)
├── analysis/                      # Evaluation & exploration notebooks
│   ├── eval_posterior.ipynb        # Posterior visualisation
│   └── setup_prior.ipynb          # Prior inspection & setup
├── data/                          # Generated training data (HDF5, not tracked)
├── outputs/                       # Hydra-managed training outputs (not tracked)
└── README.md
```

> **Note:** Only `default.yaml` and `test.yaml` configs are tracked in git. Experiment-specific config variants (e.g. `experiment.yaml`) are gitignored — Hydra logs the full resolved config in each run's output directory.

## Simulator

The core simulator (`smFRET_simulator.pyx`) integrates Langevin dynamics on a spline-defined free-energy potential and produces two-channel (green/red) photon count traces:

1. **Dynamics** — Overdamped Langevin where the force derived from potential $U(x)$ parametirzed by  cubic spline interpolation. (dx = -D * dU/dx * dt + sqrt(2D*dt)*N(0,1))
2. **FRET efficiency** — $E(r) = R_0^6 / (R_0^6 + r^6)$.
3. **Photon detection** — Poisson-distributed photon counts with cross-talk, background, and quantum-efficiency corrections.

The free-energy landscape is parameterised by spline knot values drawn from a correlated multivariate normal prior, enabling flexible, non-parametric potentials.

### Parameters

| Group | Parameters | Description |
|-------|-----------|-------------|
| Dynamics | `logD` | Log-diffusion coefficient |
| Potential | `spline_coeffs` (×`num_knots`) | Spline knot values defining $U(x)$ |
| FRET | `R0`, `kD` | Förster radius, donor excitation rate |
| Background | `k_gb`, `k_rb` | Green/red background photon rates |
| Detectors | `eta_g`, `eta_r` | Green/red quantum efficiencies |
| Cross-talk | `C_gg`, `C_rr`, `C_gr`, `C_rg` | Spectral bleed-through matrix |

### Fixed Simulator Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `total_time` | 10 000 | Total simulation time |
| `saving_frequency` | 10 000 | Photon binning window |
| `time_step` | 5×10⁻⁵ | Integration time step |
| `x_min` / `x_max` | 4.0 / 14.0 | Position of first and last knot |
| `boundary_condition` | `linear` | Setup of boundary conditions |
| `num_boundary_knots` | 3 | Number of boundary knots |

## Installation

### Prerequisites

- Python ≥ 3.9
- GSL (GNU Scientific Library) — required by the Cython simulator
- CUDA-capable GPU (optional, for faster training)

### Build Cython Extensions

```bash
cd src
python build_cython.py
```

This compiles `smFRET_simulator.pyx` and `gsl_spline.pyx` into shared libraries (`.so`) that can be imported from Python.

### Python Dependencies

Key packages: `torch`, `sbi`, `hydra-core`, `h5py`, `einops`, `numpy`, `cython`.

## Usage

### 1. Generate Training Data

```bash
python src/generate_simulations.py \
    --config conf/simulator/default.yaml \
    --format h5
```

Override config values from the command line:

```bash
python src/generate_simulations.py \
    --config conf/simulator/default.yaml \
    --num-simulations 200000 \
    --num-workers 80 \
    --output-dir data/my_run \
    --chunk-size 128
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config`, `-c` | `conf/simulator/default.yaml` | Path to YAML config |
| `--num-simulations`, `-n` | from config | Override simulation count |
| `--num-workers`, `-w` | from config | Override parallel workers |
| `--output-dir`, `-o` | from config | Override output directory |
| `--format`, `-f` | `h5` | Output format: `h5`, `pt`, or `both` |
| `--chunk-size` | batch size | HDF5 chunk size |
| `--timestamp` | off | Add timestamp to output dir |

Outputs an HDF5 file under `data/` with `theta` (parameters) and `x` (photon traces) datasets, chunked for efficient streaming during training.

### 2. Train a Density Estimator

Training uses [Hydra](https://hydra.cc) for configuration management.

**Neural Posterior Estimation (NPE)** — learns $p(\theta \mid x)$ with a normalizing flow (MAF):

```bash
python src/train_density_estimator.py
```

Override any config value from the command line:

```bash
python src/train_density_estimator.py training.num_epochs=300 embedding.output_dim=256
```

Select a different temporal encoder (CNN / Transformer) and tune only that encoder's hyperparameters:

```bash
# Transformer encoder
python src/train_density_estimator.py embedding=transformer embedding.depth=6 embedding.num_heads=8

# CNN encoder
python src/train_density_estimator.py embedding=cnn embedding.dim=16 embedding.kernel_size=9
```

Encoder configs live in `conf/training/embedding/` (Hydra config group):

- `embedding=cnn` with `embedding.*` CNN hyperparameters
- `embedding=transformer` with `embedding.*` Transformer hyperparameters

Training logs are written to TensorBoard under the Hydra output directory (`outputs/density_estimator/<date>/<time>/`).

### 3. Evaluate

Use the notebooks in `analysis/` to load trained models and inspect posteriors on test data:

- `eval_posterior.ipynb` — sample and visualise the posterior
- `setup_prior.ipynb` — inspect and configure priors

## Data Loading

`H5Dataset` (iterable PyTorch dataset) streams data from HDF5 with chunk-aligned contiguous reads. Chunks are partitioned across DataLoader workers for parallel decompression.

## What's Tracked in Git

| Tracked | Not tracked |
|---------|-------------|
| All `src/*.py`, `src/*.pyx`, `src/*.pxd` | `src/build/`, `__pycache__/`, `*.so`, `*.c` (generated) |
| `conf/**/default.yaml`, `conf/**/test.yaml` | Experiment config variants |
| `analysis/*.ipynb` (code only, outputs stripped) | `data/`, `outputs/` |
| `README.md`, `.gitignore` | `*.pt`, `*.pkl`, images |