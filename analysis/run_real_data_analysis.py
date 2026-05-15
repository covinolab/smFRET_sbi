"""Batch real-data analysis runner for the Sturzenegger 2018 (Schuler) dataset.

Loops over every experimental condition and performs K bootstrap-resampled
rounds of posterior inference + MAP estimation + PPC simulation, persisting
every artifact under a Hydra-managed timestamped results tree. A standalone
script: all trajectory-handling helpers live here so it can be run without
touching the src/ package (it still imports priors / simulator builders that
are used by the training pipeline).
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import h5py
import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sbi.inference.posteriors import DirectPosterior
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from priors import (  # noqa: E402
    generate_fret_prior_from_config,
    generate_posterior_prior_from_config,
    generate_split_priors_from_config,
    assemble_canonical_theta,
)
from simulator import binding_simulator_wrapper_from_config  # noqa: E402
from utils import map_with_cached_embedding  # noqa: E402

ACCEPTOR_ID = 1
DONOR_ID = 2


def get_trajectory_data(traj_file):
    """Load a single trajectory from an HDF5 file."""
    with h5py.File(traj_file, "r") as f:
        detector = f["Detectors"][:]
        interphoton_ds = f["InterPhotonTimes"]
        interphoton_times = interphoton_ds[:]
        macro_time_unit_ps = interphoton_ds.attrs["MacroTimeUnitInPicoSeconds"]
    return interphoton_times, macro_time_unit_ps, detector


def photons_to_model_input(interphoton_times, macro_time_unit_ps, detector,
                           N_max_photons, sim_bin_width_ms=None, L_max_bins=None):
    """Convert raw photon stream to PhotonTCN-ready `x` tensor.

    `sim_bin_width_ms` and `L_max_bins` are kept as kw-args for backward
    compatibility with old call sites but are unused now — the network
    consumes per-photon arrival times directly.

    Returns
    -------
    x : torch.Tensor, shape (N_max_photons, 4), float32
        Channels: (log(Δt + ε), G_onehot, R_onehot, mask). Padded with zeros.
    n_valid : int
        Number of real photons (≤ N_max_photons).
    """
    macro_time_unit_ps = np.int64(macro_time_unit_ps)
    n_total = int(interphoton_times.shape[0])
    if n_total == 0:
        return torch.zeros(N_max_photons, 4, dtype=torch.float32), 0

    # Δt in MILLISECONDS — matches the simulator's time convention (the project
    # uses ms throughout; see setup_prior.ipynb cell 31, "All times in ms").
    # Experimental files store Δt as macrotime units × MacroTimeUnitInPicoseconds:
    #   ps → ms = × 1e-9.
    ipt_ms = interphoton_times.astype(np.int64) * macro_time_unit_ps * 1e-9
    # Channels: G=0 (DONOR), R=1 (ACCEPTOR). Keep the same convention as the simulator.
    chans = (detector == ACCEPTOR_ID).astype(np.uint8)

    # Truncate or pad to N_max_photons.
    n_valid = min(n_total, N_max_photons)
    ipt_padded = np.zeros(N_max_photons, dtype=np.float32)
    chans_padded = np.zeros(N_max_photons, dtype=np.uint8)
    ipt_padded[:n_valid] = ipt_ms[:n_valid].astype(np.float32)
    chans_padded[:n_valid] = chans[:n_valid]
    # First interphoton time is 0 by convention (matches simulator output).
    if n_valid > 0:
        ipt_padded[0] = 0.0

    # Build features the same way the H5Dataset loader does for sim data
    # (see data_loaders.H5Dataset._arrivals_to_x). Channels: log Δt, G_oh, R_oh, mask.
    LOG_IPT_EPS = np.float32(1e-12)
    log_ipt = np.log(ipt_padded + LOG_IPT_EPS)
    g_oh = (chans_padded == 0).astype(np.float32)
    r_oh = (chans_padded == 1).astype(np.float32)
    idx = np.arange(N_max_photons, dtype=np.int64)
    mask_bool = idx < n_valid
    mask_f = mask_bool.astype(np.float32)
    log_ipt = log_ipt * mask_f
    g_oh = g_oh * mask_f
    r_oh = r_oh * mask_f
    x = np.stack([log_ipt, g_oh, r_oh, mask_f], axis=-1)
    return torch.from_numpy(x), int(n_valid)


def list_condition_trace_files(data_root: Path, condition_name: str) -> list[Path]:
    return sorted((data_root / condition_name).glob("*.h5"))


def load_all_condition_traces(trace_files, N_max_photons):
    """Load every trajectory for one condition into model-input tensors.

    Returns (tensors, valid_lens, kept_paths) — traces with 0 photons are dropped.
    Each tensor has shape (N_max_photons, 4) — see `photons_to_model_input`.
    """
    tensors, valid_lens, kept = [], [], []
    for f in trace_files:
        try:
            ipt, mtu, det = get_trajectory_data(f)
        except (OSError, KeyError) as err:
            print(f"  [skip] {f.name}: {err}")
            continue
        tensor, n_valid = photons_to_model_input(ipt, mtu, det, N_max_photons=N_max_photons)
        if n_valid > 0:
            tensors.append(tensor)
            valid_lens.append(n_valid)
            kept.append(f)
    return tensors, valid_lens, kept


def bootstrap_indices(n_pool: int, n_draw: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(n_pool, size=n_draw, replace=True)


def build_observation(tensors, indices, device):
    stacked = torch.stack([tensors[i] for i in indices])        # (n_obs, L, 3)
    return stacked.unsqueeze(0).to(device)                      # (1, n_obs, L, 3)


def simulate_ppc_trajectories(simulate, fret_prior, map_theta, inferred_idx,
                              num_trajectories, L_max_bins, max_retries):
    """Simulate PPC trajectories from MAP with photophysics drawn from fret_prior."""
    sim_tensors, sim_valid = [], []
    attempts = 0
    max_attempts = num_trajectories * max_retries
    while len(sim_tensors) < num_trajectories and attempts < max_attempts:
        attempts += 1
        params_16 = fret_prior.sample((1,)).squeeze().clone()
        params_16[inferred_idx] = map_theta

        result = simulate(params_16)
        if result is None:
            continue
        trace, length = result
        tensor_3ch = torch.zeros(L_max_bins, 3, dtype=torch.float32)
        if isinstance(trace, torch.Tensor):
            tensor_3ch[:, :2] = trace
        else:
            tensor_3ch[:, :2] = torch.from_numpy(trace)
        tensor_3ch[:length, 2] = 1.0
        sim_tensors.append(tensor_3ch)
        sim_valid.append(int(length))
    return sim_tensors, sim_valid


def batched_log_prob(posterior, samples, obs, batch_size, device):
    """Evaluate posterior log-probability of `samples` in chunks of `batch_size`."""
    samples_dev = samples.to(device)
    log_probs = []
    with torch.no_grad():
        for start in tqdm(
            range(0, samples_dev.shape[0], batch_size), desc="log_prob batches"
        ):
            chunk = samples_dev[start:start + batch_size]
            log_probs.append(posterior.log_prob(chunk, x=obs).cpu())
    return torch.cat(log_probs)


def save_round_artifacts(
    out_dir: Path,
    posterior_samples: torch.Tensor,
    log_probs: torch.Tensor,
    top_idx: torch.Tensor,
    map_estimate: torch.Tensor,
    map_log_prob: float,
    ppc_tensor: torch.Tensor,
    ppc_valid_bins: torch.Tensor,
    trace_manifest: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(posterior_samples, out_dir / "posterior_samples.pt")
    torch.save(log_probs, out_dir / "posterior_log_probs.pt")
    torch.save(top_idx, out_dir / "top_idx.pt")
    torch.save(map_estimate, out_dir / "map_estimate.pt")
    torch.save(torch.tensor(map_log_prob), out_dir / "map_log_prob.pt")
    torch.save(ppc_tensor, out_dir / "ppc_trajectories.pt")
    torch.save(ppc_valid_bins, out_dir / "ppc_valid_bins.pt")
    with open(out_dir / "trace_manifest.json", "w") as f:
        json.dump(trace_manifest, f, indent=2)


@hydra.main(version_base=None, config_path="../conf/analysis", config_name="real_data")
def main(cfg: DictConfig) -> None:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[runner] output dir: {output_dir}")

    device = cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"
    print(f"[runner] device: {device}")

    data_cfg_path = str(cfg.model.data_config_path)
    data_cfg = OmegaConf.load(data_cfg_path)
    num_knots = data_cfg.prior.num_knots
    prior_posterior, _, inferred_idx, _ = generate_split_priors_from_config(
        data_cfg_path, device=device
    )
    theta_dim = int(inferred_idx.numel())
    sim_bin_width_ms = data_cfg.simulator.time_step * data_cfg.simulator.saving_frequency
    L_max_bins = data_cfg.simulator.binding.L_max_bins
    print(f"[runner] model expects (n_obs, {L_max_bins}, 3) @ {sim_bin_width_ms} ms bins; theta_dim={theta_dim}")

    estimator_path = Path(cfg.model.estimator_path)
    if not estimator_path.exists():
        fallback = estimator_path.parent / "last_npe_set_estimator.pkl"
        if fallback.exists():
            print(f"[runner] {estimator_path.name} not found; using {fallback.name}")
            estimator_path = fallback
        else:
            raise FileNotFoundError(f"No estimator found at {estimator_path} or {fallback}")
    with open(estimator_path, "rb") as f:
        density_estimator = pickle.load(f)
    density_estimator = density_estimator.to(device)
    density_estimator.eval()

    fret_prior = generate_fret_prior_from_config(data_cfg_path, device="cpu")
    simulate = binding_simulator_wrapper_from_config(data_cfg_path)
    posterior = DirectPosterior(density_estimator, prior=prior_posterior)

    data_root = Path(cfg.data.root)
    if cfg.data.conditions == "all":
        conditions = sorted([d.name for d in data_root.iterdir() if d.is_dir()])
    else:
        conditions = list(cfg.data.conditions)
    print(f"[runner] conditions to process ({len(conditions)}): {conditions}")

    run_meta = {
        "estimator_path": str(estimator_path),
        "data_config_path": str(cfg.model.data_config_path),
        "data_root": str(data_root),
        "conditions": conditions,
        "num_rounds": int(cfg.rounds.num_rounds),
        "base_seed": int(cfg.rounds.base_seed),
        "num_traces_per_round": int(cfg.data.num_traces_per_round),
        "sim_bin_width_ms": float(sim_bin_width_ms),
        "L_max_bins": int(L_max_bins),
        "theta_dim": int(theta_dim),
        "device": device,
    }
    with open(output_dir / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    for condition_name in conditions:
        cond_dir = output_dir / condition_name
        cond_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[cond] === {condition_name} ===")

        trace_files = list_condition_trace_files(data_root, condition_name)
        print(f"[cond] {len(trace_files)} raw trajectories on disk")
        if len(trace_files) == 0:
            print("[cond] no trajectories — skipping")
            continue

        t0 = time.time()
        tensors, valid_bins, kept_paths = load_all_condition_traces(
            trace_files, sim_bin_width_ms, L_max_bins
        )
        print(f"[cond] {len(tensors)} valid trajectories loaded ({time.time()-t0:.1f}s)")
        if len(tensors) == 0:
            print("[cond] no valid trajectories — skipping")
            continue

        with open(cond_dir / "condition_meta.json", "w") as f:
            json.dump({
                "condition": condition_name,
                "n_trajectories_on_disk": len(trace_files),
                "n_valid": len(tensors),
                "valid_bins_min": int(min(valid_bins)),
                "valid_bins_max": int(max(valid_bins)),
                "valid_bins_median": float(np.median(valid_bins)),
                "all_trace_paths": [str(p) for p in kept_paths],
            }, f, indent=2)

        for round_idx in range(int(cfg.rounds.num_rounds)):
            round_seed = int(cfg.rounds.base_seed) + round_idx
            rng = np.random.default_rng(round_seed)
            torch.manual_seed(round_seed)

            round_dir = cond_dir / f"round_{round_idx:03d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            print(f"[round] {condition_name} round_{round_idx:03d} (seed={round_seed})")

            idx = bootstrap_indices(len(tensors), int(cfg.data.num_traces_per_round), rng)
            obs = build_observation(tensors, idx, device)

            samples = posterior.sample(
                (int(cfg.inference.num_posterior_samples),),
                x=obs,
                max_sampling_batch_size=int(cfg.inference.posterior_batch_size),
            ).cpu()

            log_probs = batched_log_prob(
                posterior,
                samples,
                obs,
                batch_size=int(cfg.inference.log_prob_batch_size),
                device=device,
            )
            top_k = int(cfg.inference.map_num_init_samples)
            top_log_probs, top_idx = torch.topk(log_probs, k=top_k)
            top_samples = samples[top_idx]

            map_theta, map_log_prob = map_with_cached_embedding(
                density_estimator,
                prior_posterior,
                obs,
                num_iter=int(cfg.inference.map_num_iter),
                num_init_samples=top_k,
                num_to_optimize=top_k,
                learning_rate=float(cfg.inference.map_learning_rate),
                init_samples=top_samples,
                show_progress_bars=True,
            )
            print(f"[round]   MAP = {map_theta.numpy().round(3)}  log p = {map_log_prob:.2f}")

            sim_tensors, sim_valid = simulate_ppc_trajectories(
                simulate,
                fret_prior,
                map_theta,
                inferred_idx=inferred_idx,
                num_trajectories=int(cfg.ppc.num_trajectories),
                L_max_bins=L_max_bins,
                max_retries=int(cfg.ppc.max_retries_per_trace),
            )
            if len(sim_tensors) < int(cfg.ppc.num_trajectories):
                print(f"[round]   WARNING: only {len(sim_tensors)}/{cfg.ppc.num_trajectories} PPC sims succeeded")
            ppc_tensor = torch.stack(sim_tensors) if sim_tensors else torch.empty(0, L_max_bins, 3)
            ppc_valid = torch.tensor(sim_valid, dtype=torch.int64)

            trace_manifest = {
                "seed": round_seed,
                "round_idx": round_idx,
                "condition": condition_name,
                "bootstrap_indices": idx.tolist(),
                "trace_paths": [str(kept_paths[i]) for i in idx],
                "n_valid_bins": [int(valid_bins[i]) for i in idx],
            }

            save_round_artifacts(
                round_dir,
                posterior_samples=samples,
                log_probs=log_probs,
                top_idx=top_idx,
                map_estimate=map_theta,
                map_log_prob=float(map_log_prob),
                ppc_tensor=ppc_tensor,
                ppc_valid_bins=ppc_valid,
                trace_manifest=trace_manifest,
            )
            print(f"[round]   saved -> {round_dir}")

    print(f"\n[runner] done. results at {output_dir}")


if __name__ == "__main__":
    main()
