# smFRET-SBI

**Simulation-Based Inference for single-molecule FRET.**

This repository implements a full pipeline for inferring the parameters of a molecular system — its free-energy landscape, diffusion coefficient, and photophysical properties — directly from photon-count time series recorded in single-molecule FRET experiments, without the need for hand-crafted likelihood functions.

The pipeline has three stages:

1. **Simulate** — a Cython + GSL accelerated simulator generates synthetic two-channel photon traces from a spline-parameterised free-energy potential and full FRET photophysics (cross-talk, background, detector efficiencies).
2. **Train** — a neural density or likelihood-ratio estimator learns the mapping from simulated observations to the posterior over the parameters of interest.
3. **Analyse** — the trained estimator is applied to experimental (or held-out synthetic) data to obtain posterior samples.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [The Simulator](#the-simulator)
  - [Forward Model](#forward-model)
  - [Spline Parameterisation](#spline-parameterisation)
  - [Simulator Modes](#simulator-modes)
  - [Prior](#prior)
  - [Simulator Configuration](#simulator-configuration)
  - [Generating Training Data](#generating-training-data)
- [Training](#training)
  - [Neural Posterior Estimation (NPE)](#neural-posterior-estimation-npe)
  - [Neural Ratio Estimation (NRE/BNRE)](#neural-ratio-estimation-nrebnre)
  - [Temporal Encoders](#temporal-encoders)
  - [Training Configuration](#training-configuration)
  - [Data Loading](#data-loading)
  - [Multi-GPU Training (DDP)](#multi-gpu-training-ddp)
  - [Checkpointing & Resuming](#checkpointing--resuming)
  - [Logging](#logging)
- [Analysis](#analysis)
- [Repository Conventions](#repository-conventions)

---

## Project Structure

```
smFRET_sbi/
├── src/
│   ├── smFRET_simulator.pyx       # Cython core: equilibrium + binding simulators
│   ├── gsl_spline.pyx             # GSL cubic/steffen/akima spline wrapper
│   ├── _gsl_spline.pxd            # GSL spline C declarations
│   ├── gsl_random_numbers.pxd     # GSL RNG C declarations
│   ├── build_cython.py            # Compile .pyx → .so (autodetects GSL)
│   ├── simulator.py               # Python wrappers around the Cython simulator
│   ├── priors.py                  # Prior construction (logD + gradient knots + photophysics)
│   ├── utils.py                   # Spline utilities, DDP helpers, checkpointing, seeding
│   ├── data_loaders.py            # H5Dataset — chunk-streaming IterableDataset
│   ├── temporal_encoders.py       # CNN / Transformer / LRU embedding networks
│   ├── generate_simulations.py    # CLI: parallel simulation → sharded HDF5
│   ├── merge_h5_shards.py         # Merge HDF5 shards (validates config consistency)
│   ├── train_density_estimator.py # NPE training (MAF + RQ-spline)
│   └── train_ratio_estimator.py   # NRE/BNRE training (ResNet contrastive classifier)
├── conf/
│   ├── simulator/                 # Simulator + prior configs (Hydra, YAML)
│   │   ├── default.yaml           # Baseline equilibrium config
│   │   ├── exp_*.yaml             # Experiment-specific configs (e.g. exp_6 = binding mode)
│   │   ├── calibrated.yaml        # Calibrated against a reference dataset
│   │   └── set_equilibrium.yaml   # Set-based (N iid traces per parameter set)
│   └── training/                  # Training configs (Hydra, YAML)
│       ├── density_estimator.yaml # Default NPE training config
│       ├── density_estimator_set.yaml # Set-based NPE training
│       ├── ratio_estimator.yaml   # NRE/BNRE training
│       ├── exp_*.yaml             # Experiment-specific training configs
│       └── embedding/             # Encoder hyperparameter group
│           ├── cnn.yaml
│           ├── transformer.yaml
│           └── lru.yaml
├── analysis/                      # Evaluation and inference notebooks (see below)
├── data/                          # Generated training data (HDF5, gitignored)
├── outputs/                       # Hydra-managed training runs (gitignored)
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites

- Python ≥ 3.9
- **GSL (GNU Scientific Library)** — required by the Cython simulator for cubic-spline interpolation and random number generation
- CUDA-capable GPU (recommended for training)

### Python dependencies

```bash
pip install -r requirements.txt
```

Core packages: `torch`, `sbi`, `hydra-core`, `h5py`, `einops`, `numpy`, `cython`.

### Build the Cython extensions

```bash
cd src
python build_cython.py
```

This compiles [smFRET_simulator.pyx](src/smFRET_simulator.pyx) and [gsl_spline.pyx](src/gsl_spline.pyx) into shared libraries (`.so`). The build script autodetects GSL via `pkg-config`, then falls back to `EBROOTGSL` (EasyBuild) and `GSL_DIR` environment variables.

---

## The Simulator

### Forward model

The simulator produces two-channel (green / red) photon count traces by integrating the following physics:

1. **Overdamped Langevin dynamics** on a spline-parameterised free-energy potential $U(x)$:

$$
dx = -D\,\frac{dU}{dx}\,dt + \sqrt{2D\,dt}\;\eta_t,\qquad \eta_t \sim \mathcal{N}(0,1)
$$

2. **FRET efficiency** at each binned position:

$$
E(x) = \frac{R_0^{\,6}}{R_0^{\,6} + x^{\,6}}
$$

3. **Photon emission** — during each saving window the number of donor/acceptor-channel photons arriving at the detectors is drawn from Poisson distributions. The emission rates are modulated by the donor excitation rate $k_D$, the (time-averaged) FRET efficiency, the detector quantum efficiencies $\eta_g,\eta_r$, per-channel background rates $k_{gb},k_{rb}$, and a $2\times2$ cross-talk matrix $\{C_{gg},C_{rr},C_{gr},C_{rg}\}$ that accounts for spectral bleed-through.

Compiled with `-O3 -ffast-math`, a single equilibrium trace takes ≈ tens of milliseconds, enabling multi-hundred-thousand-trace training datasets to be generated across CPU workers.

### Spline parameterisation

The free-energy landscape is parameterised by **force-gradient knot values** $g_0, g_1, \dots, g_{n-1}$ placed on a uniform grid between `x_min` and `x_max`. The potential is reconstructed by integrating the force:

$$
U(x) = -\int_{x_0}^{x} F(x') \, dx', \qquad F(x) = \text{spline}(x; g_0\ldots g_{n-1})
$$

This gradient parameterisation avoids the additive zero-point ambiguity of potential-height parameterisations: any constant shift in $U$ leaves the dynamics unchanged, which would otherwise make the inference problem non-identifiable.

The spline interpolation type (`cubic`, `steffen`, `akima`) is chosen via `simulator.spline_type`. Boundary behaviour beyond `x_min` / `x_max` is controlled by `boundary_condition` together with `num_boundary_knots`, which adds additional knots to define how the force extrapolates (e.g. `linear` extrapolation outside the support of the interior knots).

Knot helper functions live in [utils.py](src/utils.py): `make_spline_knots`, `integrate_force_to_energy`, `eval_free_energy`, `eval_force`.

### Simulator modes

Two modes are available via `simulator.mode`:

| Mode | Function | Output | When to use |
|------|----------|--------|-------------|
| `equilibrium` | [`smFRET_simulator`](src/smFRET_simulator.pyx) | Fixed-length trace `[T, 2]` | Long unbiased trajectories — sufficient for well-populated landscapes. |
| `binding` | [`binding_simulator`](src/smFRET_simulator.pyx) | Variable-length trace `[T, 2]` + `length` | Rare transitions (B → A) — uses Doob h-transform with two-way shooting from the transition state to sample transition paths efficiently. |

**Binding mode details.** Given a candidate potential $U(x)$, the simulator auto-detects the wells $x_A, x_B$ and barrier $x_{TS}$ from the landscape via a fine `landmark_grid_n` scan. It then generates reactive paths using a guided Langevin integrator with a force clamped at `F_guide_max_factor = 0.1` of the natural force magnitude, rejecting non-committed trajectories according to `k_persist` consecutive bins inside a basin. Paths have variable length — each trace is zero-padded to `L_max_bins` and a `lengths` dataset is stored alongside `theta` and `x` in the HDF5 file, so the training scripts can reconstruct a binary mask channel (see [Temporal Encoders](#temporal-encoders)).

### Prior

The full prior factorises over 12 parameter groups (total dimension `num_knots + 11`):

| Parameter | Prior | Description |
|-----------|-------|-------------|
| `logD` | Uniform(`logD_lims`) | log₁₀ diffusion coefficient |
| `g_0 … g_{n-1}` | Uniform | `num_knots` force-gradient knot values; each is Uniform(`centerᵢ − scale`, `centerᵢ + scale`). `center: null` gives a symmetric zero-mean prior, or you can pass a list to bias the prior around an expected force profile (e.g. a restoring harmonic). |
| `R0`, `kD` | Uniform | Förster radius, donor excitation rate |
| `k_gb`, `k_rb` | Uniform | Green / red background photon rates |
| `eta_g`, `eta_r` | Uniform | Green / red detector quantum efficiencies |
| `C_gg`, `C_rr`, `C_gr`, `C_rg` | Uniform | Cross-talk matrix entries |

The parameter vector ordering used throughout the code (see [priors.py](src/priors.py)) is:

```
[logD, g_0, …, g_{n-1}, R0, kD, k_gb, k_rb, eta_g, eta_r, C_gg, C_rr, C_gr, C_rg]
```

Helper functions:
- `generate_spline_fret_prior(...)` — full prior (12 groups)
- `generate_posterior_prior_from_config(...)` — reduced prior for posterior sampling restricted to `[logD, g_0..g_{n-1}, R0]`
- `generate_fret_prior_from_config(cfg)` — build the prior from a YAML config

### Simulator configuration

Configs live under [conf/simulator/](conf/simulator/) and are loaded via OmegaConf. Key fields:

```yaml
prior:
  logD_lims: [-0.5, 2.0]
  R0_lims:   [6.0, 12.0]
  kD_lims:   [0.1, 10.0]
  k_gb_lims: [0.0, 0.25]
  k_rb_lims: [0.0, 0.25]
  eta_g_lims: [0.7, 1.0]
  eta_r_lims: [0.7, 1.0]
  C_gg_lims: [0.8, 1.0]
  C_rr_lims: [0.8, 1.0]
  C_gr_lims: [0.0, 0.2]
  C_rg_lims: [0.0, 0.2]
  num_knots: 5                 # n force-gradient knots
  center: null                 # null → symmetric prior; list of length n → biased prior
  scale: 10.0                  # half-width of each knot's uniform box

simulator:
  mode: equilibrium            # or "binding"
  total_time: 10000.0
  saving_frequency: 10000      # photon-binning window (simulation steps per bin)
  time_step: 5e-5
  spline_type: cubic           # cubic | steffen | akima
  boundary_condition: linear
  num_boundary_knots: 3
  max_force: 30.0              # soft force magnitude cap (numerical stability)
  x_min: 4.0
  x_max: 14.0
  x_init: [4.0, 14.0]          # range from which the initial position is sampled

  binding:                     # only used if mode == "binding"
    L_min_bins: 200
    L_max_bins: 400
    k_persist: 20
    landmark_grid_n: 400
    N_max: 200000000

data_generation:
  num_simulations: 200000
  batch_size: 100              # simulations per worker batch
  chunk_size: 128              # HDF5 chunk size (samples per chunk)
  num_traces_per_set: 1        # N iid traces per parameter set (>1 = set-based mode)
  max_traj_retries: 5          # retry budget for failed trajectories (binding mode)

output:
  directory: ./data/base
```

**Set-based mode.** Setting `num_traces_per_set > 1` records *N* iid traces per sampled parameter vector, producing observation tensors of shape `[N, T, 2]`. This is useful when the experimentalist observes many independent molecules under the same physical conditions; the encoder can then aggregate across the set to sharpen the posterior. See [conf/simulator/set_equilibrium.yaml](conf/simulator/set_equilibrium.yaml).

### Generating training data

```bash
python src/generate_simulations.py \
    --config conf/simulator/default.yaml \
    --num-workers 80
```

Common flags:

| Flag | Description |
|------|-------------|
| `--config`, `-c` | *(required)* path to YAML config |
| `--num-workers`, `-w` | *(required)* parallel simulation workers |
| `--num-simulations`, `-n` | override simulation count |
| `--output-dir`, `-o` | override output directory |
| `--chunk-size` | override HDF5 chunk size |
| `--timestamp` | append timestamp to output directory name |
| `--debug` | verbose logging |

**Implementation highlights** (see [generate_simulations.py](src/generate_simulations.py)):

- A **worker pool** (via `multiprocessing.Pool`) runs batches of simulations; a dedicated **writer process** drains a bounded queue and appends results to a gzip-compressed HDF5 file with shuffle filter.
- Workers seed `numpy`'s RNG from their PID so that each process samples independent trajectories.
- Each parameter vector is retried up to `max_traj_retries` times if the simulator returns `None` (e.g. rare failed binding paths); unfilled simulations after `max_rounds` are trimmed at the end and the HDF5 file is resized.
- Metadata (source config, generation timestamp, CLI overrides) is written to `metadata.yaml` and `config.yaml` alongside `data.h5`.

**Output structure:**

```
data/<run>/
├── data.h5              # theta, x  (and lengths, in binding mode)
├── config.yaml          # exact simulator config used
└── metadata.yaml        # provenance (timestamp, overrides, etc.)
```

For very large runs, you can generate data in parallel shards on a cluster and then merge them:

```bash
python src/merge_h5_shards.py --input-dir data/my_run_shards --output data/my_run/data.h5
```

`merge_h5_shards.py` validates that all shards share the same HDF5 dataset shapes, dtypes, chunking, and compression before concatenating.

---

## Training

Two complementary SBI approaches are supported. Both use the same [Hydra](https://hydra.cc) configuration system and share the same embedding network architectures.

### Neural Posterior Estimation (NPE)

[`train_density_estimator.py`](src/train_density_estimator.py) learns the amortised posterior $q_\phi(\theta \mid x)$ directly as a conditional **normalising flow**, specifically a **Masked Autoregressive Flow with Rational-Quadratic Spline transforms** (`maf_rqs` from the `sbi` package).

At inference time, calling `posterior.sample((n,), x=x_obs)` draws `n` posterior samples conditional on an observation `x_obs` — no MCMC required.

```bash
python src/train_density_estimator.py
```

Override any config value from the CLI:

```bash
# Train for more epochs with a bigger embedding
python src/train_density_estimator.py training.num_epochs=300 embedding.output_dim=256

# Use a different data directory / experiment config
python src/train_density_estimator.py data.data_dir=data/exp_1 training.num_epochs=100

# Select a different encoder (Hydra config group)
python src/train_density_estimator.py embedding=transformer embedding.depth=6 embedding.num_heads=8
python src/train_density_estimator.py embedding=lru
```

### Neural Ratio Estimation (NRE/BNRE)

[`train_ratio_estimator.py`](src/train_ratio_estimator.py) learns a discriminator that estimates the likelihood-to-evidence ratio $r(x, \theta) \propto p(\theta \mid x)/p(\theta)$ via a contrastive classifier trained with `num_atoms` negative samples per positive pair. It supports the **Balanced Neural Ratio Estimator (BNRE)** regularisation via `regularization_strength` (logged to TensorBoard hparams).

```bash
python src/train_ratio_estimator.py
```

The ratio estimator is useful for hypothesis-testing style analyses and for diagnostics — posterior sampling then requires an MCMC sweep (see [eval_likelihood_ratios.ipynb](analysis/eval_likelihood_ratios.ipynb)).

### Temporal Encoders

All estimators rely on an embedding network that maps a photon-count trace of shape `[T, C]` (with `C = 2` for `(green, red)` or `C = 3` when a mask channel is added in binding mode) to a fixed-size feature vector of dimension `output_dim`. Three encoder families are shipped under [conf/training/embedding/](conf/training/embedding/):

**CNN** ([cnn.yaml](conf/training/embedding/cnn.yaml)) — a residual 1-D convolutional U-net backbone with optional linear attention and a statistics-pooling head. Defaults:

```yaml
name: cnn
dim: 32
dim_mults: [1, 1, 2, 2, 4]   # channel multipliers per stage
attention: true
padding_mode: reflect
kernel_size: 32
downsample: 0
output_dim: 128
```

**Transformer** ([transformer.yaml](conf/training/embedding/transformer.yaml)) — a Conv1d stem (stride-2 downsampling) followed by positional encoding, a `CLS` token, and a stack of standard `nn.TransformerEncoder` layers:

```yaml
name: transformer
dim: 32
kernel_sizes: [4, 4]
stride: 2
num_heads: 4
depth: 4
mlp_dim: 128
dropout: 0.1
output_dim: 128
```

**LRU** ([lru.yaml](conf/training/embedding/lru.yaml)) — a custom bidirectional Linear Recurrent Unit (Orvieto-Smith–style) with complex-diagonal state dynamics, gated skip connections, and log-parameterised eigenvalues for stability:

```yaml
name: lru
state_dim: 32
hidden_dim: 64
num_blocks: 2
r_min: 0.3
r_max: 0.99
phase_max: 6.2830
dropout: 0.05
output_dim: 64
```

The appropriate encoder is instantiated by `build_temporal_encoder(...)` (or `build_set_embedding(...)` when `num_traces_per_set > 1`, which wraps the single-trace encoder with a permutation-invariant aggregation head). Both helpers live in [temporal_encoders.py](src/temporal_encoders.py).

**Variable-length handling.** In binding mode the dataset carries a `lengths` array; the training scripts detect this automatically via `H5Dataset.has_lengths`, append a binary mask channel `(green, red, mask)`, and set the encoder input dimension to 3. Encoders are expected to learn to ignore padded regions (there is no explicit attention masking).

### Training configuration

Defaults for [density_estimator.yaml](conf/training/density_estimator.yaml):

```yaml
defaults:
  - embedding: cnn         # select an encoder config group
  - _self_

training:
  batch_size: 32
  validation_fraction: 0.1
  num_epochs: 50
  seed: 42
  compile: true                       # torch.compile wraps the loss
  grad_clip: 10.0                     # clip gradient global norm
  checkpoint_interval_minutes: 30
  resume_from: null

optimizer:
  lr: 1e-3
  weight_decay: 0.01                  # AdamW

scheduler:                            # OneCycleLR (omit to use constant LR)
  max_lr: 5e-3
  pct_start: 0.3
  final_div_factor: 1e4

embedding:                            # inline encoder overrides (merged onto the selected group)
  dim: 8
  dim_mults: [1, 2]
  attention: false
  kernel_size: 7
  padding_mode: reflect
  output_dim: 128

flow:
  model: "maf_rqs"                    # MAF + Rational-Quadratic Spline
  hidden_features: 128
  num_transforms: 3
  num_blocks: 2
  dropout_probability: 0.1
  use_batch_norm: false

data:
  data_dir: "data/base"
  chunk_size: 512                     # HDF5 chunk size used for I/O
  chunk_step: 1                       # how many chunks to load into RAM at once
  num_workers: 6
  prefetch_factor: 3

artifacts:
  model_filename: "npe_density_estimator.pkl"
  checkpoint_filename: "checkpoint.pt"
```

Hydra writes run artefacts to `outputs/<date>/<time>/`, including the fully resolved config (`.hydra/config.yaml`), a copy of the dataset config (`.hydra/data_conf.yaml`), checkpoints, the best model, and TensorBoard event files.

### Data loading

Training uses [`H5Dataset`](src/data_loaders.py) — a PyTorch `IterableDataset` that streams HDF5 with **chunk-aligned contiguous reads** for maximum I/O throughput:

1. The chunk list (excluding those assigned to validation) is shuffled each epoch.
2. `chunk_step` chunks are loaded into RAM at once; samples inside are shuffled.
3. Batches are yielded from the shuffled in-memory buffer.
4. When wrapped in a PyTorch `DataLoader(num_workers=N)`, chunks are partitioned round-robin across workers, so each worker performs independent, parallel GZIP decompression.

Train / validation split is performed **by chunk** (not by random indices) using a fixed seed (`torch.Generator().manual_seed(42)`), so the split is reproducible and consistent across DDP ranks. The training script partitions training chunks across ranks with `_trim_ddp_train_chunks(...)` so that every rank runs the same number of steps per epoch (asserted via `_assert_equal_train_steps`).

### Multi-GPU training (DDP)

Launch with `torchrun`:

```bash
torchrun --nproc_per_node=8 src/train_density_estimator.py
```

Implementation notes:
- `setup_ddp()` initialises `nccl` and sets the per-rank CUDA device from `LOCAL_RANK`.
- The loss function is wrapped in a `DDP`-wrapped `nn.Module` (`_LossForward`) so that gradient synchronisation is automatic.
- Only the main process (`rank == 0`) writes TensorBoard logs, checkpoints, and the best model pickle.
- Loss and sample counts are reduced via `dist.all_reduce` before logging.
- `torch.compile` is applied to the wrapped loss after DDP (`training.compile: true`).

### Checkpointing & resuming

- **Rolling checkpoint** every `training.checkpoint_interval_minutes` minutes (rank 0 only) — written to `checkpoint.pt` in the Hydra output directory. Contains optimizer state, scheduler state, current epoch, best validation loss, and best model weights.
- **Best model** is saved as a pickled `DirectPosterior`/estimator (`last_npe_density_estimator.pkl`) whenever a new validation minimum is reached.
- On successful completion, the intermediate rolling checkpoint and the `last_*` pickle are deleted; the final `npe_density_estimator.pkl` contains the best model.
- Resume by setting `training.resume_from=path/to/checkpoint.pt` (supports relative paths resolved against the original launch directory).

### Logging

Each run writes a TensorBoard event file under `outputs/<date>/<time>/tensorboard/`. Scalars tracked per epoch include:

- `Loss/train`, `Loss/val`, `Loss/gap`
- `Norm/Grad_Norm`, `Norm/Grad_Norm_Encoder`, `Norm/Grad_Norm_Flow`
- `Norm/Param_Norm_Encoder`, `Norm/Param_Norm_Flow`
- `Learning_rate`, `Epoch_time_seconds`, `Throughput/samples_per_sec`
- `Optimization/clip_fraction` (fraction of steps whose gradient hit the clipping threshold)
- `Embedding/probe_mean`, `Embedding/probe_std`, `Embedding/probe_l2` — embedding statistics for a fixed probe batch, useful for detecting collapse or saturation
- `hparams/*` — all hyperparameters plus the final best validation loss, written via `SummaryWriter.add_hparams`

The git commit hash of the repository at launch time is logged at the start of every run to tie models to their source state.

---

## Analysis

The [analysis/](analysis/) directory contains Jupyter notebooks and a helper script for evaluating trained models and running inference on experimental data:

- `eval_posterior.ipynb` — sample and visualise the NPE posterior
- `eval_likelihood_ratios.ipynb` — evaluate NRE/BNRE likelihood-ratio estimators (posterior via MCMC)
- `setup_prior.ipynb` — interactive prior inspection and calibration
- `infer_real_data.ipynb`, `run_real_data_analysis.py`, `plot_real_data_analysis.ipynb` — pipelines for applying trained estimators to experimental traces

These are provided as-is for reference and day-to-day analysis; start from whichever notebook matches your use case.

---

## Repository Conventions

- **Configs:** only `default.yaml`, `test.yaml`, and generic encoder configs are committed. Experiment-specific variants (`exp_*.yaml`, `calibrated.yaml`, etc.) are tracked only where useful; Hydra logs the *resolved* config for every run in the run output directory, so no experiment is ever un-reproducible.
- **Data and outputs** (`data/`, `outputs/`) are gitignored.
- **Generated Cython artefacts** (`src/build/`, `*.so`, `*.c`, `__pycache__/`) are gitignored.
- **Parameter vector ordering** is fixed in [priors.py](src/priors.py) and must match the unpacking logic in [simulator.py](src/simulator.py) (`_unpack_physics_params`).
