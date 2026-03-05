# smFRET-SBI

Simulation-based inference (SBI) for single-molecule FRET experiments. This package infers the parameters of a molecular system—its free-energy landscape, diffusion coefficient, and photophysical properties—directly from photon count time-series, without relying on hand-crafted likelihood functions.

## Overview

The pipeline has three stages:

1. **Simulate** – A Cython-accelerated smFRET simulator generates synthetic photon traces from a spline-parameterised free-energy potential and full FRET photophysics (cross-talk, background, detector efficiencies).
2. **Train** – Neural density or ratio estimators learn the mapping from simulated observations to posterior distributions over the parameters of interest.
3. **Analyse** – Trained estimators are applied to experimental (or held-out) data to obtain posterior samples or likelihood ratios.

## Project Structure

```
smFRET_sbi/
├── conf/
│   ├── simulator/           # Simulator & prior configuration (YAML)
│   │   └── default.yaml
│   └── training/            # Training configs managed by Hydra
│       ├── density_estimator.yaml
│       └── ratio_estimator.yaml
├── src/
│   ├── smFRET_simulator.pyx # Core simulator (Cython, calls GSL splines)
│   ├── gsl_spline.pyx       # GSL cubic-spline wrapper
│   ├── build_cython.py      # Compile Cython extensions in-place
│   ├── simulator.py         # Python wrapper around the Cython simulator
│   ├── priors.py            # Prior definitions (uniform + correlated spline)
│   ├── utils.py             # Spline-knot construction & evaluation helpers
│   ├── data_loaders.py      # HDF5 iterable dataset with chunk-based I/O
│   ├── temporal_encoders.py  # 1-D CNN embedding network (ResNet-style with optional attention)
│   ├── generate_simulations.py  # CLI for parallel data generation → HDF5
│   ├── train_density_estimator.py  # NPE training script (normalizing flow)
│   └── train_ratio_estimator.py    # NRE training script (classifier)
├── analysis/                # Evaluation notebooks (posterior & likelihood-ratio)
├── data/                    # Generated training data (HDF5 / PT)
├── outputs/                 # Hydra-managed training outputs & checkpoints
├── launch_data_gen.sh       # SLURM job script for data generation
└── README.md
```

## Simulator

The core simulator (`smFRET_simulator.pyx`) integrates Langevin dynamics on a spline-defined free-energy potential and produces two-channel (green/red) photon count traces:

1. **Dynamics** – Overdamped Langevin equation: $dx = D \, F(x) \, dt + \sqrt{2D \, dt}\;\eta$, where $F(x) = -dU/dx$ is the force derived from a cubic-spline potential $U(x)$.
2. **FRET efficiency** – $E(r) = R_0^6 / (R_0^6 + r^6)$.
3. **Photon detection** – Poisson-distributed photon counts with cross-talk, background, and quantum-efficiency corrections.

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

This compiles `smFRET_simulator.pyx` and `gsl_spline.pyx` into shared libraries that can be imported from Python.

### Python Dependencies

Key packages: `torch`, `sbi`, `hydra-core`, `h5py`, `einops`, `numpy`, `cython`.

## Usage

### 1. Generate Training Data

```bash
# Single run
python src/generate_simulations.py \
    --config conf/simulator/default.yaml \
    --format h5 \
    --num-simulations 200000 \
    --num-workers 80

# Or via SLURM
sbatch launch_data_gen.sh
```

Outputs an HDF5 file with `theta` (parameters) and `x` (photon traces) datasets, chunked for efficient streaming during training.

### 2. Train an Estimator

Both training scripts use [Hydra](https://hydra.cc) for configuration management.

**Neural Posterior Estimation (NPE)** – learns $p(\theta \mid x)$ with a normalizing flow (MAF):

```bash
python src/train_density_estimator.py
```

**Neural Ratio Estimation (NRE)** – learns the likelihood-to-evidence ratio with a classifier:

```bash
python src/train_ratio_estimator.py
```

Override any config value from the command line:

```bash
python src/train_density_estimator.py training.num_epochs=300 embedding.output_dim=256
```

Training logs are written to TensorBoard under the Hydra output directory.

### 3. Evaluate

Use the notebooks in `analysis/` to load trained models and inspect posteriors or likelihood ratios on test data:

- `eval_posterior.ipynb` – sample and visualise the posterior
- `eval_likelihood_ratio.ipynb` – compute and plot likelihood ratios

## Architecture

The observation encoder is a **1-D temporal CNN** (`TemporalCNN`) that maps variable-length two-channel photon traces to a fixed-size embedding vector. It uses residual blocks with GroupNorm, Mish activations, strided downsampling, and optional linear attention layers.

This embedding feeds into either:

- A **Masked Autoregressive Flow (MAF)** for density estimation (NPE), or
- A **ResNet classifier** for ratio estimation (NRE).

## Data Loading

For large datasets, an `H5Dataset` (iterable PyTorch dataset) streams data from HDF5 with chunk-aligned contiguous reads. Chunks are partitioned across DataLoader workers for parallel decompression. Smaller datasets can be loaded entirely into RAM from `.pt` files.