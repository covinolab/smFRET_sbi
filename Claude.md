Guidance for Claude Code when working in this repository. Keep edits precise; this is a PhD research codebase.
Project
smFRET-SBI — Simulation-based inference (SBI) of free-energy landscapes, diffusion coefficients, and photophysics from single-molecule FRET photon-count time series. Three stages: (1) Cython simulator → (2) neural posterior/ratio estimator training → (3) analysis on experimental data.
Stack: PyTorch, sbi (mackelab), Hydra, Cython + GSL, HDF5, SLURM/DDP.
Repo Map
src/                          # All source code
  smFRET_simulator.pyx        # Core Cython simulator (uses GSL)
                              #   - smFRET_simulator()  : equilibrium Langevin
                              #   - binding_simulator() : Doob h-transform B→A paths
  gsl_spline.pyx              # GSL cubic/steffen/akima spline wrapper
  _gsl_spline.pxd             # GSL spline C decls
  gsl_random_numbers.pxd      # GSL RNG C decls
  build_cython.py             # Compile .pyx → .so (run before first use)

  simulator.py                # Python wrappers around Cython sim
                              #   - smFRET_simulator_wrapper       (equilibrium)
                              #   - binding_simulator_wrapper      (transition paths)
                              #   - *_from_config() variants
  priors.py                   # generate_spline_fret_prior() → MultipleIndependent
                              # Param order (total dim = num_knots + 9):
                              #   [logD, g₀..g_{n-1}, R0, kD, k_gb, k_rb,
                              #    eta_g, eta_r, C_gr, C_rg]
                              # Crosstalk diagonals derived in the simulator
                              # wrapper: C_gg = 1 - C_gr, C_rr = 1 - C_rg.
  utils.py                    # make_spline_knots, integrate_force_to_energy,
                              # eval_free_energy/force, DDP helpers,
                              # checkpoint save/load, seed, git-commit logging
  data_loaders.py             # H5Dataset (IterableDataset, chunked streaming)
                              # has_lengths → yields (theta, x, lengths)
  temporal_encoders.py        # TemporalCNN, TemporalTransformer,
                              # custom LRU/LRUBlock/LRUEmbedding (picklable)
  generate_simulations.py     # CLI: parallel sim → HDF5 (eq + binding modes)
  merge_h5_shards.py          # Merge sharded HDF5 (validates config consistency)
  train_density_estimator.py  # NPE: MAF + RQ-spline (maf_rqs). DDP, OneCycleLR,
                              # TensorBoard, ckpt resume. Auto mask channel.
  train_ratio_estimator.py    # NRE/BNRE: ResNet classifier, contrastive loss.
                              # Same auto mask channel support.

conf/                         # Hydra configs (YAML)
  simulator/{default,test}.yaml
  training/{density_estimator,ratio_estimator,test}.yaml
  training/embedding/{cnn,transformer,lru}.yaml

analysis/                     # Jupyter notebooks (outputs stripped in git)
  setup_prior.ipynb           # Binning, t_TP extraction, prior calibration
  eval_posterior.ipynb        # Posterior sampling/visualisation
  eval_likelihood_ratios.ipynb

data/, outputs/               # gitignored. data/ = HDF5 sims; outputs/ = Hydra runsCore Concepts
Forward model. Overdamped Langevin on a cubic-spline potential U(x) → FRET efficiency E(r) = R₀⁶/(R₀⁶ + r⁶) → Poisson photon counts in 2 channels (green/red) with cross-talk, background, and quantum efficiencies.
Spline parameterisation. The landscape is parameterised by force-gradient knot values g₀…g_{n−1} (NOT potential heights). Potential is recovered by integrate_force_to_energy(). This avoids the additive zero-point ambiguity of U(x). (The README still mentions "knot height increments Δᵢ" — this is stale; the code uses force gradients.)
Two simulator modes (simulator.mode in config):

equilibrium — long fixed-length traces, free Langevin.
binding — Doob h-transform conditioned B→A transition paths. Variable-length, zero-padded to max_bins, with a lengths dataset stored alongside theta/x in HDF5. Well minima x_A, x_B auto-detected from the potential per simulation. Guiding force clamped via F_guide_max_factor=0.1.
Variable-length handling. Binding traces use a mask channel: input dim becomes (green, red, mask) = 3 instead of (green, red) = 2. H5Dataset.has_lengths triggers this automatically; both training scripts call _unpack_batch() / _augment_with_mask() and increase transition_dim 2→3. Encoders rely on learning to ignore padded regions (no explicit attention masking).
Inference methods.

NPE (train_density_estimator.py) — normalizing flow p(θ|x) with MAF + RQ-spline (maf_rqs).
NRE/BNRE (train_ratio_estimator.py) — ResNet contrastive classifier with num_atoms, optional BNRE regulariser (regularization_strength logged to TB hparams).
Temporal encoders (conf/training/embedding/):

cnn — residual Conv1d + optional linear attention + StatsPool. Defaults: dim=32, mults=[1,1,2,2,4], kernel=32.
transformer — Conv1d stem → posenc → CLS → nn.TransformerEncoder. Defaults: dim=32, depth=4, heads=4.
lru — custom bidirectional LRU with gated skip connections (Smith et al. 2023 style). Defaults: state_dim=32, hidden_dim=64, r_min=0.3, r_max=0.99, output_dim=64.
Build & Run
# 0. Activate the project virtualenv (always do this first)
source ~/Venvs/fret_sbi/bin/activate

# 1. Compile Cython once
cd src && python build_cython.py

# 2. Generate training data
python src/generate_simulations.py -c conf/simulator/default.yaml -w 80
# (binding mode also writes a `lengths` HDF5 dataset)

# 3. Merge shards if generated in parallel
python src/merge_h5_shards.py --input-dir data/run_shards --output data/run/data.h5

# 4. Train (Hydra-configured)
python src/train_density_estimator.py
python src/train_density_estimator.py embedding=transformer embedding.depth=6
torchrun --nproc_per_node=8 src/train_density_estimator.py    # DDP

python src/train_ratio_estimator.pyHydra outputs land in outputs/<date>/<time>/ with full resolved config + TensorBoard logs + checkpoints.