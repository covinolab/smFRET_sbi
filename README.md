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
│   ├── temporal_encoders.py       # 1-D CNN / Transformer / LRU embedding networks
│   ├── generate_simulations.py    # CLI for parallel data generation → HDF5
│   ├── merge_h5_shards.py         # Merge sharded HDF5 files into one dataset
│   ├── train_density_estimator.py # NPE training script (normalizing flow)
│   └── train_ratio_estimator.py   # Likelihood-ratio estimation (alternative SBI)
├── conf/
│   ├── simulator/                 # Simulator & prior configs (Hydra, YAML)
│   │   ├── default.yaml           # Default config (tracked)
│   │   └── test.yaml              # Test config (tracked)
│   └── training/                  # Training configs (Hydra, YAML)
│       ├── density_estimator.yaml # Default NPE training config (tracked)
│       ├── ratio_estimator.yaml   # Likelihood-ratio training config (tracked)
│       ├── test.yaml              # Test training config (tracked)
│       └── embedding/             # Encoder config group
│           ├── cnn.yaml           # CNN encoder hyperparameters
│           ├── transformer.yaml   # Transformer encoder hyperparameters
│           └── lru.yaml           # LRU encoder hyperparameters
├── analysis/                      # Evaluation & exploration notebooks
│   ├── eval_posterior.ipynb           # Posterior visualisation
│   ├── eval_likelihood_ratios.ipynb   # Likelihood-ratio evaluation
│   └── setup_prior.ipynb              # Prior inspection & setup
├── data/                          # Generated training data (HDF5, not tracked)
├── outputs/                       # Hydra-managed training outputs (not tracked)
└── README.md
```

> **Note:** Only `default.yaml` and `test.yaml` configs are tracked in git. Experiment-specific config variants (e.g. `experiment.yaml`) are gitignored — Hydra logs the full resolved config in each run's output directory.

## Simulator

The core simulator (`smFRET_simulator.pyx`) integrates Langevin dynamics on a spline-defined free-energy potential and produces two-channel (green/red) photon count traces:

1. **Dynamics** — Overdamped Langevin where the force is derived from potential $U(x)$ parameterized by cubic spline interpolation. (dx = -D * dU/dx * dt + sqrt(2D*dt)*N(0,1))
2. **FRET efficiency** — $E(r) = R_0^6 / (R_0^6 + r^6)$.
3. **Photon detection** — Poisson-distributed photon counts with cross-talk, background, and quantum-efficiency corrections.

The free-energy landscape is parameterised by spline knot height increments (Δᵢ = φᵢ₊₁ − φᵢ), each drawn independently from a uniform prior. Absolute knot values are reconstructed for simulation and visualization via cumulative sum and then zero-mean centred for identifiability.

### Parameters

| Group | Parameters | Description |
|-------|-----------|-------------|
| Dynamics | `logD` | Log-diffusion coefficient |
| Potential | `delta_phi` (×`num_knots−1`) | Knot height increments defining $U(x)$ |
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
| `spline_type` | `cubic` | Interpolation type for the free-energy potential |

## Installation

### Prerequisites

- Python ≥ 3.9
- GSL (GNU Scientific Library) — required by the Cython simulator
- CUDA-capable GPU (optional, for faster training)

### Python Dependencies

```bash
pip install -r requirements.txt
```

Key packages: `torch`, `sbi`, `hydra-core`, `h5py`, `einops`, `numpy`, `cython`.

### Build Cython Extensions

```bash
cd src
python build_cython.py
```

This compiles `smFRET_simulator.pyx` and `gsl_spline.pyx` into shared libraries (`.so`) that can be imported from Python.

## Usage

### 1. Generate Training Data

```bash
python src/generate_simulations.py \
    --config conf/simulator/default.yaml \
    --num-workers 80
```

Override additional config values from the command line:

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
| `--config`, `-c` | *(required)* | Path to YAML config |
| `--num-workers`, `-w` | *(required)* | Number of parallel worker processes |
| `--num-simulations`, `-n` | from config | Override simulation count |
| `--output-dir`, `-o` | from config | Override output directory |
| `--format`, `-f` | `h5` | Output format: `h5`, `pt`, or `both` |
| `--chunk-size` | batch size | HDF5 chunk size |
| `--timestamp` | off | Add timestamp to output dir |

Outputs an HDF5 file under `data/` with `theta` (parameters) and `x` (photon traces) datasets, chunked for efficient streaming during training.

### 2. Train a Density Estimator

Training uses [Hydra](https://hydra.cc) for configuration management.

**Neural Posterior Estimation (NPE)** — learns $p(\theta \mid x)$ with a normalizing flow (MAF with Rational-Quadratic Spline transforms):

```bash
python src/train_density_estimator.py
```

Override any config value from the command line:

```bash
python src/train_density_estimator.py training.num_epochs=300 embedding.output_dim=256
```

Select a different temporal encoder and tune its hyperparameters:

```bash
# CNN encoder (default)
python src/train_density_estimator.py embedding=cnn embedding.dim=16 embedding.kernel_size=9

# Transformer encoder
python src/train_density_estimator.py embedding=transformer embedding.depth=6 embedding.num_heads=8

# LRU encoder
python src/train_density_estimator.py embedding=lru
```

Encoder configs live in `conf/training/embedding/` (Hydra config group): `cnn`, `transformer`, `lru`.

**Multi-GPU training** with DDP:

```bash
torchrun --nproc_per_node=8 src/train_density_estimator.py
```

Training logs are written to TensorBoard under the Hydra output directory (`outputs/<date>/<time>/`).

### 2b. Train a Likelihood-Ratio Estimator (alternative)

As an alternative to NPE, a likelihood-ratio estimator can be trained to discriminate between the prior and the posterior:

```bash
python src/train_ratio_estimator.py
```

This uses the config at `conf/training/ratio_estimator.yaml` and supports the same Hydra overrides as the density estimator.

### 2c. Merge HDF5 Shards

If data was generated in parallel shards, merge them before training:

```bash
python src/merge_h5_shards.py --input-dir data/exp_1_data_shards --output data/exp_1_data/data.h5
```

### 3. Evaluate

Use the notebooks in `analysis/` to load trained models and inspect results:

- `eval_posterior.ipynb` — sample and visualise the posterior
- `eval_likelihood_ratios.ipynb` — evaluate likelihood-ratio estimators
- `setup_prior.ipynb` — inspect and configure priors

## Data Loading

`H5Dataset` (iterable PyTorch dataset) streams data from HDF5 with chunk-aligned contiguous reads. Chunks are partitioned across DataLoader workers for parallel decompression.

## What's Tracked in Git

| Tracked | Not tracked |
|---------|-------------|
| All `src/*.py`, `src/*.pyx`, `src/*.pxd` | `src/build/`, `__pycache__/`, `*.so`, `*.c` (generated) |
| `conf/**/default.yaml`, `conf/**/test.yaml`, `conf/**/ratio_estimator.yaml`, `conf/training/embedding/*.yaml` | Experiment-specific config variants |
| `analysis/*.ipynb` (code only, outputs stripped) | `data/`, `outputs/` |
| `README.md`, `.gitignore` | `*.pt`, `*.pkl`, images |