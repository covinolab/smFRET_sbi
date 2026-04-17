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

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from priors import generate_fret_prior_from_config, generate_posterior_prior_from_config  # noqa: E402
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
                           sim_bin_width_ms, L_max_bins):
    """Bin photon arrivals at simulator resolution, return (L_max_bins, 3) tensor."""
    bin_width_ps = np.int64(sim_bin_width_ms * 1e9)
    macro_time_unit_ps = np.int64(macro_time_unit_ps)

    interphoton_times_ps = interphoton_times.astype(np.int64) * macro_time_unit_ps
    photon_times_ps = np.cumsum(interphoton_times_ps, dtype=np.int64)

    total_time_ps = photon_times_ps[-1]
    n_bins = max(1, int(np.ceil(total_time_ps / bin_width_ps)))

    n_valid = min(n_bins, L_max_bins)
    bin_edges = np.linspace(0, n_valid * bin_width_ps, n_valid + 1)

    donor_counts, _ = np.histogram(photon_times_ps[detector == DONOR_ID], bins=bin_edges)
    acceptor_counts, _ = np.histogram(photon_times_ps[detector == ACCEPTOR_ID], bins=bin_edges)

    out = torch.zeros(L_max_bins, 3, dtype=torch.float32)
    out[:n_valid, 0] = torch.from_numpy(donor_counts.astype(np.float32))
    out[:n_valid, 1] = torch.from_numpy(acceptor_counts.astype(np.float32))
    out[:n_valid, 2] = 1.0
    return out, n_valid


def list_condition_trace_files(data_root: Path, condition_name: str) -> list[Path]:
    return sorted((data_root / condition_name).glob("*.h5"))


def load_all_condition_traces(trace_files, sim_bin_width_ms, L_max_bins):
    """Load every trajectory for one condition into model-input tensors.

    Returns (tensors, valid_bins, kept_paths) — traces with 0 valid bins are dropped.
    """
    tensors, valid_bins, kept = [], [], []
    for f in trace_files:
        try:
            ipt, mtu, det = get_trajectory_data(f)
        except (OSError, KeyError) as err:
            print(f"  [skip] {f.name}: {err}")
            continue
        tensor, n_valid = photons_to_model_input(ipt, mtu, det, sim_bin_width_ms, L_max_bins)
        if n_valid > 0:
            tensors.append(tensor)
            valid_bins.append(n_valid)
            kept.append(f)
    return tensors, valid_bins, kept


def bootstrap_indices(n_pool: int, n_draw: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(n_pool, size=n_draw, replace=True)


def build_observation(tensors, indices, device):
    stacked = torch.stack([tensors[i] for i in indices])        # (n_obs, L, 3)
    return stacked.unsqueeze(0).to(device)                      # (1, n_obs, L, 3)


def simulate_ppc_trajectories(simulate, fret_prior, map_theta, num_knots,
                              num_trajectories, L_max_bins, max_retries):
    """Simulate PPC trajectories from MAP with photophysics drawn from fret_prior."""
    sim_tensors, sim_valid = [], []
    attempts = 0
    max_attempts = num_trajectories * max_retries
    while len(sim_tensors) < num_trajectories and attempts < max_attempts:
        attempts += 1
        nuisance = fret_prior.sample((1,)).squeeze()
        params_16 = nuisance.clone()
        params_16[0] = map_theta[0]
        params_16[1:num_knots + 1] = map_theta[1:-1]
        params_16[num_knots + 1] = map_theta[-1]

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


def save_round_artifacts(
    out_dir: Path,
    posterior_samples: torch.Tensor,
    map_estimate: torch.Tensor,
    ppc_tensor: torch.Tensor,
    ppc_valid_bins: torch.Tensor,
    trace_manifest: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(posterior_samples, out_dir / "posterior_samples.pt")
    torch.save(map_estimate, out_dir / "map_estimate.pt")
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
    theta_dim = num_knots + 2
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

    prior_posterior = generate_posterior_prior_from_config(data_cfg_path, device=device)
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

            map_theta, map_log_prob = map_with_cached_embedding(
                density_estimator,
                prior_posterior,
                obs,
                num_iter=int(cfg.inference.map_num_iter),
                num_init_samples=int(cfg.inference.map_num_init_samples) * 100,
                num_to_optimize=int(cfg.inference.map_num_init_samples),
                learning_rate=float(cfg.inference.map_learning_rate),
                init_samples=samples,
            )
            print(f"[round]   MAP = {map_theta.numpy().round(3)}  log p = {map_log_prob:.2f}")

            sim_tensors, sim_valid = simulate_ppc_trajectories(
                simulate,
                fret_prior,
                map_theta,
                num_knots=num_knots,
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
                map_estimate=map_theta,
                ppc_tensor=ppc_tensor,
                ppc_valid_bins=ppc_valid,
                trace_manifest=trace_manifest,
            )
            print(f"[round]   saved -> {round_dir}")

    print(f"\n[runner] done. results at {output_dir}")


if __name__ == "__main__":
    main()
