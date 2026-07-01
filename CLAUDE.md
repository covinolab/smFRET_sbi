# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This is a PhD research codebase. Keep edits precise. (Claude Code auto-loads `CLAUDE.md`, not `Claude.md` — rename this file if you want it picked up automatically.)

## Project

smFRET-SBI — Simulation-based inference of free-energy landscapes, diffusion coefficients, and photophysics from single-molecule FRET **photon streams**. Three stages: (1) Cython+GSL simulator → (2) neural posterior/ratio estimator training (sbi + Hydra + PyTorch DDP) → (3) evaluation on synthetic and experimental data.

The primary scientific target is **fast (microsecond) transition-path dynamics** in `binding` mode; weight method choices toward fast-dynamics fidelity.

## Build & Run

```bash
# 0. Always use the project venv (Python 3.10 with all deps; the default conda python lacks omegaconf etc.)
source ~/Venvs/fret_sbi/bin/activate      # or call ~/Venvs/fret_sbi/bin/python directly

# 1. Compile the Cython simulator once (autodetects GSL via pkg-config → EBROOTGSL → GSL_DIR)
cd src && python build_cython.py

# 2. Generate training data (writes data.h5 + config.yaml + metadata.yaml)
python src/generate_simulations.py -c conf/simulator/test_6.yaml -w 80

# 3. Merge parallel shards (validates shape/dtype/chunk/compression consistency first)
python src/merge_h5_shards.py --input-dir data/run_shards --output data/run/data.h5

# 4. Train (Hydra; select a config from conf/training/ with --config-name)
python src/train_density_estimator.py --config-name test_6
python src/train_density_estimator.py embedding=photon_dt_ssm data.data_dir=data/exp_7 training.num_epochs=100
torchrun --nproc_per_node=8 src/train_density_estimator.py --config-name test_6   # DDP
python src/train_ratio_estimator.py                                               # NRE/BNRE

# 5. Evaluate a trained posterior (model_dir holds data_config.yaml + *_density_estimator.pkl)
python analysis/run_eval_posterior.py <model_dir> --set kD=300 --out-dir results/kD_300
```

Hydra writes each run to `outputs/<date>/<time>/` with the resolved config (`.hydra/`), a copy of the dataset config (`data_config.yaml`), TensorBoard logs, checkpoints, and the best-model pickle. `hydra.job.chdir=false` — the process stays in the repo root, so relative `data.data_dir` paths resolve there.

### Tests

```bash
~/Venvs/fret_sbi/bin/python -m pytest src/test_nuisance_per_set.py src/test_photon_dt_ssm.py -q
~/Venvs/fret_sbi/bin/python -m pytest analysis/test_fe_widget.py analysis/test_run_eval_posterior.py -q
# Parity tests double as standalone scripts (run from src/): python test_equilibrium_event_parity.py
```

There is no central test runner, lint, or CI config — invoke pytest on the specific files above. The `test_*equilibrium*`/`test_*binding*` parity tests require the compiled `.so` (step 1).

## Architecture

### The pipeline is photon-event-based (this is the key thing to internalize)

The model does **not** see binned two-channel intensity traces. The Cython simulator emits raw photon **arrival times** per channel; everything downstream is a per-photon event stream. (The old README/Claude docs describing `[T, 2]` Poisson-count traces and CNN/Transformer/LRU encoders are stale.)

- **Simulator** ([smFRET_simulator.pyx](src/smFRET_simulator.pyx)) returns `(G_times, R_times)` — variable-length arrival-time arrays for the donor (green) and acceptor (red) channels. Times are in **milliseconds**.
- **Wrappers** ([simulator.py](src/simulator.py)) interleave both channels by time into `(interphoton_times, channels, length)`, zero-padded to `N_max_photons`. `channels` is `uint8` (0=green/donor, 1=red/acceptor); `interphoton_times[k] = t_k − t_{k−1}` (first entry 0).
- **HDF5 schema v2** ([generate_simulations.py](src/generate_simulations.py)): datasets `theta`, `interphoton_times`, `channels`, `lengths` (set-mode adds a `num_traces_per_set` axis). `f.attrs` carries `schema_version`, `n_max_photons`, `num_traces_per_set`.
- **Data loader** ([data_loaders.py](src/data_loaders.py)) `H5Dataset._arrivals_to_x` builds the model input `x` of shape `(..., N_max, 4)`: channels `(log(Δt+ε), G_onehot, R_onehot, mask)`. The mask (channel 3) is the validity flag; padded positions are zeroed. `transition_dim` is therefore **always 4**. When building an observation from real data, go through `_arrivals_to_x` so the shape/units match training.

### Parameters, prior splitting, and nuisance marginalization

Canonical parameter vector (length `num_knots + 9`), defined in [priors.py](src/priors.py):

```
[logD, g_0..g_{n-1}, R0, kD, k_gb, k_rb, eta_g, eta_r, C_gr, C_rg]
```

- The landscape is parameterised by **force-gradient knot values** `g_i` (NOT potential heights); the potential is recovered by `integrate_force_to_energy()` in [utils.py](src/utils.py). This removes the additive zero-point ambiguity of `U(x)`.
- Crosstalk is conserved by construction: only leak fractions `C_gr`, `C_rg` are sampled; the simulator wrapper derives `C_gg = 1 − C_gr`, `C_rr = 1 − C_rg` (`_unpack_physics_params`).

**The posterior is over a *subset* of parameters; the rest are nuisances marginalized at data-generation time.** `prior.inferred_params` (a token list, e.g. `[logD, knots, R0]`; `knots` is all-or-nothing) selects what goes into the posterior. `generate_split_priors_from_config` returns `(inferred_prior, nuisance_prior, inferred_idx, nuisance_idx)`; during generation each inferred draw is paired with freshly-sampled nuisances via `assemble_canonical_theta`, simulated, and **only the inferred portion is stored in `theta`** (so `theta_dim = len(inferred_idx)`). The network thus learns a posterior already marginalized over the nuisances it never sees.

Nuisance draw policy (set-based mode, `num_traces_per_set > 1`):
- `nuisance_per_set: false` (default) → resample nuisances **per trace** (PT).
- `nuisance_per_set: true` → draw nuisances **once per set**, shared across all traces — models one fixed detector/photophysics (PS).
- `nuisance_per_trace_params: [kD, ...]` → those params stay per-trace even under per-set (PTkD), since they vary molecule-to-molecule.

Prior helpers, by use case: `generate_split_priors_from_config` (generation), `generate_fret_prior_from_config` (full 10-group prior), `generate_posterior_prior_from_config` (reduced `[logD, knots, R0]`, dim `num_knots+2` — includes R0), `generate_nuisance_prior_from_config` (7 photophysics nuisances). Note the split prior and `posterior_prior` differ: when R0 is pinned (`R0_lims: [x, x]`, R0 not in `inferred_params`) use the split-prior path, not the R0-bearing `posterior_prior`, or the DirectPosterior dimension won't match.

### Simulator modes (`simulator.mode`)

- `equilibrium` — long fixed-duration free Langevin run (`smFRET_simulator`). Initial position sampled from `x_init`.
- `binding` — Doob h-transform / two-way committor shooting to sample rare B→A transition paths (`binding_simulator`). Wells/barrier auto-detected from the landscape; guiding force clamped; per-leg retries (`max_shoot_attempts`) inside the `.pyx`, plus an outer retry-on-`None` loop (`max_traj_retries`) in the wrapper. This is the mode for the fast transition-path science.

Spline interpolation (`spline_type`: `cubic`/`steffen`/`akima`; models use `steffen`) and boundary extrapolation (`boundary_condition` `[left, right]` ∈ `linear`/`quadratic`/`open`, plus `num_boundary_knots`) are configured per run. `open` (zero force outside the grid) is binding-only. A force-knot fit to a target FE shape is affine-exact only for cubic splines — with steffen it is approximate, so verify against the realized profile.

### Encoders (summary networks)

All operate on the `(N, 4)` photon stream and handle variable length via the mask channel (no separate attention-mask plumbing). Selected by `embedding.name` via `build_temporal_encoder` in [temporal_encoders.py](src/temporal_encoders.py):

- `photon_tcn` (default) — dilated causal TCN + masked attention pool ([temporal_encoders.py](src/temporal_encoders.py) `PhotonTCN`).
- `photon_dt_ssm` — Δt-aware diagonal SSM (`exp(−α·Δt)` decay), bidirectional, with `alpha_init_range_ms` log-spaced to cover μs→100 ms (`PhotonDtSSM` / `DtSSM`).
- `fret_ssm`, `fret_ssm_deep` — SSM stacks with an optional self-supervised **next-photon NLL** auxiliary loss.
- `smfret_photon` — tokenizer → local conv → FastSSM stack → Set-Transformer (ISAB+PMA) pool, with next-photon head ([smfret_photon_encoder.py](src/smfret_photon_encoder.py)).

**Aux-loss hook:** encoders exposing `last_aux_loss` + `aux_loss_weight` (FretSSM family, smfret_photon) get that NLL added to the NPE gradient inside `_train_one_epoch`; it is a no-op for encoders without those attributes. The reported train loss is the NPE loss only.

**Set mode** (`num_traces_per_set > 1`): `build_set_embedding` wraps the per-trace encoder in `SetEmbedding` (per-trace encode → permutation-invariant aggregate). `cfg.data.num_traces_per_set` must match the dataset's trailing shape or the training script raises before building the flow.

### Inference & training internals

- **NPE** ([train_density_estimator.py](src/train_density_estimator.py)): conditional normalizing flow (`maf_rqs`) via `sbi.neural_nets.posterior_nn` with the encoder as `embedding_net`, `z_score_x='none'`. Best model pickled as a `DirectPosterior`-bearing estimator; `posterior.sample((n,), x=x_obs)` needs no MCMC.
- **NRE/BNRE** ([train_ratio_estimator.py](src/train_ratio_estimator.py)): contrastive ResNet classifier; `regularization_strength` enables BNRE. Posterior requires MCMC at eval time.
- **Data I/O**: `H5Dataset` is an `IterableDataset` doing chunk-aligned contiguous reads. Train/val split is **by chunk** with a fixed seed (consistent across DDP ranks); DDP trims/partitions chunks so every rank runs equal steps (`_assert_equal_train_steps`).
- **DDP**: loss wrapped in a `DDP(_LossForward)` module; `torch.compile` applied after DDP; only rank 0 logs/checkpoints; losses reduced with `dist.all_reduce`.
- **Checkpointing**: rolling time-based `checkpoint.pt` + `last_*.pkl` on new val-min; both deleted on clean finish, leaving the final best pickle. Resume with `training.resume_from=...`.
- TensorBoard logs include per-component grad/param norms, throughput, clip fraction, and fixed-probe embedding stats (`Embedding/probe_*`) for collapse detection. The launch-time git commit is logged.

## Config conventions & gotchas

- Configs live in `conf/simulator/` (prior + forward model) and `conf/training/` (training + `embedding/` group); `conf/analysis/` parameterises evaluation runs. Training default `config_name` is `density_estimator`; pick experiment configs with `--config-name`.
- **`conf/simulator/default.yaml` is partly stale**: its prior block uses `spline_mean`/`spline_scale`/`prior_type`/`igmrf_*` keys that **no code reads**. The code reads `prior.scale`, `prior.center` (null → symmetric zero-mean; list → per-knot center), `prior.num_knots`, and `prior.inferred_params`. **Use the `test_*.yaml` configs as the canonical, working examples** (e.g. [test_6.yaml](conf/simulator/test_6.yaml)). `scale` may be a scalar or a length-`num_knots` list of per-knot half-widths.
- Posterior FE shapes are bounded by the knot prior box (`center ± scale`); to recover flexible intermediates the generating prior must be wide/flexible enough.
- **Knot prior type** (`prior.knot_prior`, default `independent`): `independent` = per-knot uniforms `g_i ~ U(center_i ± scale_i)` (the historical prior). `gaussian` = one correlated multivariate normal over the knot vector — mean `center`, per-knot marginal std `scale`, RBF covariance `K_ij = scale_i·scale_j·exp(−(i−j)²/(2·correlation_length²))`. Use it to add knots for resolution while `prior.correlation_length` (knot-index units; larger = smoother) controls landscape smoothness. It slots into the same `MultipleIndependent` (a single `event_shape=[num_knots]` component), so all prior helpers and `.sample()`/`.log_prob()` call sites are unchanged. The Gaussian is **unbounded**, so eval OOD warnings ([run_eval_posterior.py](analysis/run_eval_posterior.py)) and the FE widget ([fe_widget.py](analysis/fe_widget.py)) use `center ± prior.n_sigma·scale` (default 3) as the effective box; posterior rejection/MAP simply run unconstrained (this also side-steps the leakage-stall). See [test_gaussian_knots.yaml](conf/simulator/test_gaussian_knots.yaml).
- The parameter ordering in [priors.py](src/priors.py) must stay in sync with `_unpack_physics_params` in [simulator.py](src/simulator.py).

## Analysis (`analysis/`)

- `run_eval_posterior.py` — CLI to evaluate **one** trained posterior under one setting: builds ground-truth FE cases (prior-centre, prior samples, explicit `--knots`), simulates observations with a worker pool **before** any CUDA work (never fork a Pool after a CUDA context — that hung multi-case eval previously), samples the posterior + MAP (MAP started from the posterior mean), and writes per-case `parameters.npz`/`map_estimate.npz`/`ground.npz` + plots + `run_meta.json`. Headline metric is `param_dist_l2`. Sweeps = shell loops over `--set`/`--out-dir` (see `launch_systematic_eval.sh`, `launch_test6_nuisance_eval.sh`); notebooks only **load** the saved files.
- Notebooks: `eval_posterior.ipynb` (auto-detects nuisance mode incl. PTkD), `eval_likelihood_ratios.ipynb`, `setup_prior.ipynb`, `infer_real_data.ipynb`, `diagnose_sim_vs_exp.ipynb`. `fe_widget.py` builds FE profiles from a model's `data_config.yaml` (CPU/numpy only). `summarize_nuisance_eval.py` aggregates sweep outputs.
- If posterior sampling stalls at ~0% acceptance, the operating point likely pushed the posterior outside the prior box (e.g. asymmetric crosstalk) — not a deadlock.

## Repository conventions

- `data/`, `outputs/`, compiled Cython artefacts (`src/build/`, `*.so`, `*.c`, `*.html`, `__pycache__/`) are gitignored.
- Every run records its resolved Hydra config and a copy of the dataset config, so any model traces back to its exact source state and data.
