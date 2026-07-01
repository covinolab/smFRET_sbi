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
import warnings
from pathlib import Path

import h5py
import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sbi.inference.posteriors import DirectPosterior
from sbi.utils import within_support
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from priors import (  # noqa: E402
    generate_fret_prior_from_config,
    generate_split_priors_from_config,
)
from data_loaders import H5Dataset  # noqa: E402  (_arrivals_to_x: photon arrivals -> (N_max, 4))
from simulator import binding_simulator_wrapper_from_config  # noqa: E402
from utils import map_with_cached_embedding  # noqa: E402

ACCEPTOR_ID = 1   # detector id for the acceptor (red) channel in the raw HDF5
DONOR_ID = 2      # detector id for the donor (green) channel
LOG_IPT_EPS = H5Dataset.LOG_IPT_EPS


def get_trajectory_data(traj_file):
    """Load a single trajectory from an HDF5 file."""
    with h5py.File(traj_file, "r") as f:
        detector = f["Detectors"][:]
        interphoton_ds = f["InterPhotonTimes"]
        interphoton_times = interphoton_ds[:]
        macro_time_unit_ps = interphoton_ds.attrs["MacroTimeUnitInPicoSeconds"]
    return interphoton_times, macro_time_unit_ps, detector


def photons_to_model_input(interphoton_times, macro_time_unit_ps, detector, N_max_photons):
    """Convert a raw photon stream to the model's per-photon `x` tensor (N_max_photons, 4).

    Matches the TRAINING representation exactly by delegating to
    `H5Dataset._arrivals_to_x` — channels `[log(Δt + ε), donor_onehot, acceptor_onehot,
    mask]`. Δt is converted to MILLISECONDS (the simulator's unit; experimental files store
    Δt as macrotime units × MacroTimeUnitInPicoseconds, so ps → ms is × 1e-9). Channel
    coding follows the simulator: 0 = donor (green), 1 = acceptor (red). The first
    interphoton gap is set to 0 (no predecessor). Streams longer than N_max_photons are
    truncated; shorter ones are zero-padded (the mask channel marks valid photons).

    Returns
    -------
    x : torch.Tensor, shape (N_max_photons, 4), float32
    n_valid : int
        Number of real photons (≤ N_max_photons).
    """
    n_total = int(np.asarray(interphoton_times).shape[0])
    if n_total == 0:
        return torch.zeros(N_max_photons, 4, dtype=torch.float32), 0

    ipt_ms = np.asarray(interphoton_times, dtype=np.float64) * float(macro_time_unit_ps) * 1e-9
    ipt_ms[0] = 0.0  # first photon has no predecessor (matches the simulator's ipt[0] = 0)
    chans = (np.asarray(detector) == ACCEPTOR_ID).astype(np.uint8)  # 1=acceptor, 0=donor

    n_valid = min(n_total, N_max_photons)
    ipt_padded = np.zeros(N_max_photons, dtype=np.float32)
    chans_padded = np.zeros(N_max_photons, dtype=np.uint8)
    ipt_padded[:n_valid] = ipt_ms[:n_valid].astype(np.float32)
    chans_padded[:n_valid] = chans[:n_valid]

    x = H5Dataset._arrivals_to_x(
        ipt_padded[None, :], chans_padded[None, :], np.asarray([n_valid], dtype=np.int64)
    )[0]  # (N_max_photons, 4)
    return x, int(n_valid)


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
    stacked = torch.stack([tensors[i] for i in indices])        # (n_obs, N_max, 4)
    return stacked.unsqueeze(0).to(device)                      # (1, n_obs, N_max, 4)


def simulate_ppc_trajectories(simulate, fret_prior, map_theta, inferred_idx,
                              num_trajectories, N_max_photons, max_retries):
    """Simulate PPC trajectories from the MAP, with photophysics drawn from `fret_prior`.

    The binding simulator returns per-photon `(interphoton_times, channels, length)` (chans
    0=donor, 1=acceptor); we convert each to the SAME `(N_max_photons, 4)` representation as
    the observations (H5Dataset._arrivals_to_x) so sim and real can be reduced identically.
    A full prior draw already pins R0 and randomises the photophysics nuisances; the inferred
    dims (logD + knots) are overwritten with the MAP.
    """
    sim_tensors, sim_valid = [], []
    attempts = 0
    max_attempts = num_trajectories * max_retries
    while len(sim_tensors) < num_trajectories and attempts < max_attempts:
        attempts += 1
        params_16 = fret_prior.sample((1,)).squeeze().clone()
        params_16[inferred_idx] = map_theta.to(params_16.dtype)

        result = simulate(params_16)
        if result is None:
            continue
        ipt, chans, length = result
        ipt = ipt.cpu().numpy() if torch.is_tensor(ipt) else np.asarray(ipt)
        chans = chans.cpu().numpy() if torch.is_tensor(chans) else np.asarray(chans)
        x = H5Dataset._arrivals_to_x(
            ipt[None, :], chans[None, :], np.asarray([int(length)], dtype=np.int64)
        )[0]  # (N_max_photons, 4)
        sim_tensors.append(x)
        sim_valid.append(int(length))
    return sim_tensors, sim_valid


def bounded_sample(density_estimator, inferred_prior, obs, num_samples, batch, theta_dim,
                   device, max_oversample=20, giveup_proposals=2000):
    """Rejection-sample the posterior with a HARD proposal cap so it can never hang.

    Mirrors run_eval_posterior.py / infer_real_data.ipynb. `DirectPosterior.sample` loops
    forever when the posterior leaks out of the prior box (the `0.000% proposal samples
    accepted` failure) — the likeliest outcome for a real, out-of-distribution condition,
    which would otherwise hang the entire batch run. Draw raw flow samples
    (`density_estimator.sample(condition=...)`), keep the in-support ones, stop once
    `num_samples` are collected OR `num_samples*max_oversample` proposals are spent (give up
    early if nothing is accepted after `giveup_proposals`). Returns
    `(samples_cpu, acceptance, complete, n_proposals)`.
    """
    x = obs.to(device)
    max_prop = max(int(num_samples * max_oversample), batch)
    kept, n_prop, n_kept = [], 0, 0
    with torch.no_grad():
        while n_kept < num_samples and n_prop < max_prop:
            s = density_estimator.sample((batch,), condition=x).reshape(-1, theta_dim)
            n_prop += s.shape[0]
            ins = within_support(inferred_prior, s)
            if bool(ins.any()):
                sk = s[ins].cpu()
                kept.append(sk); n_kept += sk.shape[0]
            if n_kept == 0 and n_prop >= giveup_proposals:
                break
    samples = torch.cat(kept)[:num_samples] if kept else torch.empty((0, theta_dim))
    acc = (n_kept / n_prop) if n_prop else 0.0
    return samples, float(acc), bool(samples.shape[0] >= num_samples), int(n_prop)


def batched_log_prob(posterior, samples, obs, batch_size, device):
    """Evaluate posterior log-probability of `samples` in chunks (relative use only ->
    norm_posterior=False). Returns an empty tensor when `samples` is empty (leaked case)."""
    if samples.shape[0] == 0:
        return torch.empty(0)
    samples_dev = samples.to(device)
    log_probs = []
    with torch.no_grad():
        for start in tqdm(
            range(0, samples_dev.shape[0], batch_size), desc="log_prob batches"
        ):
            chunk = samples_dev[start:start + batch_size]
            log_probs.append(posterior.log_prob(chunk, x=obs, norm_posterior=False).cpu())
    return torch.cat(log_probs)


def save_round_artifacts(
    out_dir: Path,
    posterior_samples: torch.Tensor,
    log_probs: torch.Tensor,
    top_idx: torch.Tensor,
    map_estimate: torch.Tensor,
    map_log_prob: float,
    ppc_tensor: torch.Tensor,
    ppc_valid: torch.Tensor,
    trace_manifest: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(posterior_samples, out_dir / "posterior_samples.pt")
    torch.save(log_probs, out_dir / "posterior_log_probs.pt")
    torch.save(top_idx, out_dir / "top_idx.pt")
    torch.save(map_estimate, out_dir / "map_estimate.pt")
    torch.save(torch.tensor(map_log_prob), out_dir / "map_log_prob.pt")
    torch.save(ppc_tensor, out_dir / "ppc_trajectories.pt")        # (n_sim, N_max_photons, 4)
    torch.save(ppc_valid, out_dir / "ppc_valid.pt")                # photon counts per sim
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
    num_knots = int(data_cfg.prior.num_knots)
    # Prior over the INFERRED params only (logD + knots; R0 pinned -> not inferred), matching
    # the density estimator's output dim. Used both for sampling support and the MAP optimiser.
    inferred_prior, _, inferred_idx, _ = generate_split_priors_from_config(
        data_cfg_path, device=device
    )
    theta_dim = int(inferred_idx.numel())
    # The model consumes the per-photon representation (N_max_photons, 4), not binned counts.
    N_max_photons = int(data_cfg.simulator.N_max_photons)
    R0_fixed = 0.5 * (float(data_cfg.prior.R0_lims[0]) + float(data_cfg.prior.R0_lims[1]))
    # Safeguard / MAP knobs (.get keeps older configs working).
    max_oversample = int(cfg.inference.get("max_oversample", 20))
    giveup_proposals = int(cfg.inference.get("sample_giveup_proposals", 2000))
    map_init = str(cfg.inference.get("map_init", "posterior_mean"))
    print(f"[runner] model expects per-photon (n_obs, {N_max_photons}, 4); "
          f"theta_dim={theta_dim}; R0 pinned at {R0_fixed:.3f}; MAP init={map_init}")

    estimator_path = Path(cfg.model.estimator_path)
    if not estimator_path.exists():
        cands = ["npe_density_estimator.pkl", "last_npe_density_estimator.pkl",
                 "npe_set_estimator.pkl", "last_npe_set_estimator.pkl"]
        for name in cands:
            fb = estimator_path.parent / name
            if fb.exists():
                print(f"[runner] {estimator_path.name} not found; using {fb.name}")
                estimator_path = fb
                break
        else:
            raise FileNotFoundError(
                f"No estimator at {estimator_path} or fallbacks {cands} in {estimator_path.parent}")
    with open(estimator_path, "rb") as f:
        density_estimator = pickle.load(f)
    density_estimator = density_estimator.to(device)
    density_estimator.eval()

    fret_prior = generate_fret_prior_from_config(data_cfg_path, device="cpu")
    simulate = binding_simulator_wrapper_from_config(data_cfg_path)
    posterior = DirectPosterior(density_estimator, prior=inferred_prior)

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
        "N_max_photons": int(N_max_photons),
        "theta_dim": int(theta_dim),
        "R0_fixed": float(R0_fixed),
        "map_init": map_init,
        "max_oversample": int(max_oversample),
        "sample_giveup_proposals": int(giveup_proposals),
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
        tensors, valid_photons, kept_paths = load_all_condition_traces(
            trace_files, N_max_photons
        )
        print(f"[cond] {len(tensors)} valid trajectories loaded ({time.time()-t0:.1f}s)")
        if len(tensors) == 0:
            print("[cond] no valid trajectories — skipping")
            continue
        n_trunc = int(sum(1 for v in valid_photons if v >= N_max_photons))
        if n_trunc:
            print(f"[cond] {n_trunc} trajectories hit the {N_max_photons}-photon cap (truncated)")

        with open(cond_dir / "condition_meta.json", "w") as f:
            json.dump({
                "condition": condition_name,
                "n_trajectories_on_disk": len(trace_files),
                "n_valid": len(tensors),
                "valid_photons_min": int(min(valid_photons)),
                "valid_photons_max": int(max(valid_photons)),
                "valid_photons_median": float(np.median(valid_photons)),
                "n_truncated_at_cap": n_trunc,
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

            # Bounded rejection sampling: never hangs even if this condition is OOD (the
            # posterior leaks out of the prior box) — it flags the round LEAKED and moves on.
            samples, acceptance, complete, n_prop = bounded_sample(
                density_estimator, inferred_prior, obs,
                num_samples=int(cfg.inference.num_posterior_samples),
                batch=int(cfg.inference.posterior_batch_size),
                theta_dim=theta_dim, device=device,
                max_oversample=max_oversample, giveup_proposals=giveup_proposals,
            )
            num_samp = int(samples.shape[0])
            leaked = num_samp < int(cfg.inference.num_posterior_samples)
            if leaked:
                print(f"[round]   LEAKED: {num_samp}/{cfg.inference.num_posterior_samples} "
                      f"in-support samples (acceptance {acceptance:.2%}, {n_prop} proposals) "
                      f"-> real data OOD for this condition")

            log_probs = batched_log_prob(
                posterior, samples, obs,
                batch_size=int(cfg.inference.log_prob_batch_size), device=device,
            )

            # MAP init: no in-support samples -> multi-start from prior draws; "posterior_mean"
            # -> single start at the sample mean (stable for peaky posteriors); "top_k" ->
            # multi-start from the best-log-prob samples.
            if num_samp == 0:
                top_idx = torch.empty(0, dtype=torch.long)
                top_samples = torch.empty((0, theta_dim))
                init_samples = inferred_prior.sample(
                    (min(int(cfg.inference.map_num_init_samples), 100),)).cpu()
                n_init = n_opt = int(init_samples.shape[0])
            else:
                k = min(int(cfg.inference.map_num_init_samples), num_samp)
                _, top_idx = torch.topk(log_probs, k=k)
                top_samples = samples[top_idx]
                if map_init == "posterior_mean":
                    init_samples = samples.mean(dim=0, keepdim=True)
                    if not bool(within_support(inferred_prior, init_samples.to(device)).item()):
                        init_samples = top_samples[:1]
                    n_init = n_opt = 1
                else:  # "top_k"
                    init_samples = top_samples
                    n_init = n_opt = int(top_samples.shape[0])

            map_theta, map_log_prob = map_with_cached_embedding(
                density_estimator,
                inferred_prior,
                obs,
                num_iter=int(cfg.inference.map_num_iter),
                num_init_samples=n_init,
                num_to_optimize=n_opt,
                learning_rate=float(cfg.inference.map_learning_rate),
                init_samples=init_samples,
                show_progress_bars=True,
            )
            print(f"[round]   MAP = {map_theta.numpy().round(3)}  log p = {map_log_prob:.2f}"
                  f"  (init={map_init if num_samp else 'prior'})")

            sim_tensors, sim_valid = simulate_ppc_trajectories(
                simulate,
                fret_prior,
                map_theta,
                inferred_idx=inferred_idx,
                num_trajectories=int(cfg.ppc.num_trajectories),
                N_max_photons=N_max_photons,
                max_retries=int(cfg.ppc.max_retries_per_trace),
            )
            if len(sim_tensors) < int(cfg.ppc.num_trajectories):
                print(f"[round]   WARNING: only {len(sim_tensors)}/{cfg.ppc.num_trajectories} PPC sims succeeded")
            ppc_tensor = torch.stack(sim_tensors) if sim_tensors else torch.empty(0, N_max_photons, 4)
            ppc_valid = torch.tensor(sim_valid, dtype=torch.int64)

            trace_manifest = {
                "seed": round_seed,
                "round_idx": round_idx,
                "condition": condition_name,
                "bootstrap_indices": idx.tolist(),
                "trace_paths": [str(kept_paths[i]) for i in idx],
                "n_valid_photons": [int(valid_photons[i]) for i in idx],
                "acceptance": float(acceptance),
                "n_posterior_samples": int(num_samp),
                "leaked": bool(leaked),
                "map_log_prob": float(map_log_prob),
            }

            save_round_artifacts(
                round_dir,
                posterior_samples=samples,
                log_probs=log_probs,
                top_idx=top_idx,
                map_estimate=map_theta,
                map_log_prob=float(map_log_prob),
                ppc_tensor=ppc_tensor,
                ppc_valid=ppc_valid,
                trace_manifest=trace_manifest,
            )
            print(f"[round]   saved -> {round_dir}")

    print(f"\n[runner] done. results at {output_dir}")


if __name__ == "__main__":
    main()
