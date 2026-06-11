import argparse
import logging
import os
import queue as _queue
import shutil
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from functools import partial

import torch
import h5py
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
import yaml

logger = logging.getLogger(__name__)

from simulator import smFRET_simulator_wrapper as fret_simulator
from simulator import binding_simulator_wrapper as binding_simulator
from priors import (
    generate_split_priors_from_config,
    assemble_canonical_theta,
)
from utils import _validate_bc


# ── Multiprocessing workers ─────────────────────────────────────────────────

_worker_sim_fn = None
_worker_nuisance_prior = None
_worker_out_queue = None
_worker_num_traces = 1
_worker_max_traj_retries = 50
_worker_inferred_idx = None
_worker_nuisance_idx = None
_worker_nuisance_per_set = False


def _init_worker(out_queue, sim_fn, sim_kwargs, nuisance_prior,
                 num_traces=1, max_traj_retries=1,
                 inferred_idx=None, nuisance_idx=None,
                 nuisance_per_set=False):
    """Initialise per-worker shared state (called once per pool worker)."""
    global _worker_sim_fn, _worker_nuisance_prior, _worker_out_queue
    global _worker_num_traces, _worker_max_traj_retries
    global _worker_inferred_idx, _worker_nuisance_idx
    global _worker_nuisance_per_set
    _worker_nuisance_prior = nuisance_prior
    _worker_out_queue = out_queue
    _worker_num_traces = num_traces
    _worker_max_traj_retries = max_traj_retries
    _worker_inferred_idx = inferred_idx
    _worker_nuisance_idx = nuisance_idx
    _worker_nuisance_per_set = nuisance_per_set
    _worker_sim_fn = partial(sim_fn, **sim_kwargs)
    np.random.seed(os.getpid() % (2**32))
    torch.set_num_threads(1)


def _worker_simulate_batch(batch_theta):
    """Simulate a batch, push results to the writer queue, return success count.

    Each simulator call returns (interphoton_times, channels, length) padded
    to N_max_photons. For set-based mode (num_traces_per_set > 1), each
    "simulation" stacks N traces; the inferred portion is shared across the
    set. Nuisance is resampled per trace by default; if ``nuisance_per_set`` is
    set, the nuisance vector is drawn ONCE per set and shared across all traces
    (matches a real measurement with one fixed detector/photophysics).
    """
    results_theta, results_ipt, results_chans, results_lengths = [], [], [], []
    N = _worker_num_traces

    for i in range(len(batch_theta)):
        ipt_set, chans_set, lengths_set = [], [], []
        ok = True
        # Per-SET nuisance: one draw shared by every trace in the set (and reused across
        # retries — the simulator is stochastic, so retrying the same theta can still
        # succeed). Per-TRACE (default): redraw inside the loop below.
        set_nuisance = (_worker_nuisance_prior.sample((1,)).squeeze(0)
                        if _worker_nuisance_per_set else None)
        for _ in range(N):
            result = None
            for _attempt in range(_worker_max_traj_retries):
                nuisance = (set_nuisance if _worker_nuisance_per_set
                            else _worker_nuisance_prior.sample((1,)).squeeze(0))
                theta_for_sim = assemble_canonical_theta(
                    batch_theta[i], nuisance,
                    _worker_inferred_idx, _worker_nuisance_idx,
                )
                result = _worker_sim_fn(theta_for_sim)
                if result is not None:
                    break
            if result is None:
                ok = False
                break
            ipt, chans, length = result
            ipt_set.append(ipt)
            chans_set.append(chans)
            lengths_set.append(length)
        if not ok:
            continue

        if N > 1:
            results_ipt.append(np.stack(ipt_set, axis=0))
            results_chans.append(np.stack(chans_set, axis=0))
            results_lengths.append(np.array(lengths_set, dtype=np.int32))
        else:
            results_ipt.append(ipt_set[0])
            results_chans.append(chans_set[0])
            results_lengths.append(lengths_set[0])
        results_theta.append(batch_theta[i].numpy())

    if not results_theta:
        return 0

    thetas = np.stack(results_theta)
    ipts = np.stack(results_ipt)
    chans_arr = np.stack(results_chans)
    if N > 1:
        lengths_arr = np.stack(results_lengths)
    else:
        lengths_arr = np.array(results_lengths, dtype=np.int32)
    payload = (thetas, ipts, chans_arr, lengths_arr)
    while True:
        try:
            _worker_out_queue.put(payload, timeout=30)
            break
        except _queue.Full:
            continue
    return len(results_theta)


# ── HDF5 writer process ────────────────────────────────────────────────────

def _writer_process(h5_path, num_simulations, theta_dim, n_max_photons,
                    chunk_size, in_queue, done_queue, written_counter,
                    num_traces_per_set=1):
    """Dedicated writer process: drains the queue and appends batches to HDF5.

    Schema (v2, per-photon events):
        /theta              float32  [N, theta_dim]
        /interphoton_times  float32  [N, N_max] or [N, num_traces_per_set, N_max]
        /channels           uint8    same trailing shape as interphoton_times
        /lengths            int32    [N] or [N, num_traces_per_set]
    """
    if num_traces_per_set > 1:
        events_shape = (num_traces_per_set, n_max_photons)
        lengths_shape = (num_traces_per_set,)
    else:
        events_shape = (n_max_photons,)
        lengths_shape = ()

    theta_chunks = (max(1, min(chunk_size, num_simulations)), theta_dim)
    events_chunks = (max(1, min(chunk_size, num_simulations)),) + events_shape
    full_lengths_shape = (num_simulations,) + lengths_shape
    lengths_chunks = (max(1, min(chunk_size, num_simulations)),) + lengths_shape
    idx = 0

    with h5py.File(h5_path, "w", libver="latest") as f:
        f.create_dataset(
            "theta",
            shape=(num_simulations, theta_dim),
            maxshape=(num_simulations, theta_dim),
            dtype="float32",
            chunks=theta_chunks,
            compression="gzip", compression_opts=1, shuffle=True,
        )
        f.create_dataset(
            "interphoton_times",
            shape=(num_simulations,) + events_shape,
            maxshape=(num_simulations,) + events_shape,
            dtype="float32",
            chunks=events_chunks,
            compression="gzip", compression_opts=1, shuffle=True,
        )
        f.create_dataset(
            "channels",
            shape=(num_simulations,) + events_shape,
            maxshape=(num_simulations,) + events_shape,
            dtype="uint8",
            chunks=events_chunks,
            compression="gzip", compression_opts=1, shuffle=True,
        )
        f.create_dataset(
            "lengths",
            shape=full_lengths_shape,
            maxshape=full_lengths_shape,
            dtype="int32",
            chunks=lengths_chunks,
            compression="gzip", compression_opts=1,
        )

        while True:
            item = in_queue.get()
            if item is None:
                break

            theta_np, ipts_np, chans_np, lengths_np = item
            n = theta_np.shape[0]

            f["theta"][idx : idx + n] = theta_np
            f["interphoton_times"][idx : idx + n] = ipts_np
            f["channels"][idx : idx + n] = chans_np
            f["lengths"][idx : idx + n] = lengths_np
            idx += n
            with written_counter.get_lock():
                written_counter.value = idx

        # Trim if some simulations returned None
        if idx < num_simulations:
            for ds_name in ("theta", "interphoton_times", "channels", "lengths"):
                f[ds_name].resize(idx, axis=0)

        f.attrs["schema_version"] = 2
        f.attrs["write_idx"] = idx
        f.attrs["num_simulations"] = idx
        f.attrs["n_max_photons"] = n_max_photons
        f.attrs["num_traces_per_set"] = num_traces_per_set
        f.attrs["generated_at"] = datetime.now().isoformat()
        f.attrs["theta_shape"] = list(f["theta"].shape)

    done_queue.put(idx)


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate FRET simulation training data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", "-c", type=str, required=True,
                        help="Path to YAML configuration file")
    parser.add_argument("--num-simulations", "-n", type=int, default=None,
                        help="Override number of simulations from config")
    parser.add_argument("--num-workers", "-w", type=int, required=True,
                        help="Number of parallel worker processes")
    parser.add_argument("--output-dir", "-o", type=str, default=None,
                        help="Override output directory from config")
    parser.add_argument("--timestamp", action="store_true",
                        help="Add timestamp to output directory name")
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="HDF5 chunk size (defaults to batch_size)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    return parser.parse_args()


# ── Simulation orchestration ───────────────────────────────────────────────

def _validate_simulator_returns_events(sim_fn, sim_kwargs, prior, nuisance_prior,
                                       inferred_idx, nuisance_idx,
                                       n_max_photons, max_attempts=100):
    """Sanity-check that the simulator returns a (ipt, channels, length) tuple
    with the expected shapes. Returns once a single successful simulation is
    observed; raises if `max_attempts` fail.
    """
    fn = partial(sim_fn, **sim_kwargs)
    for _ in range(max_attempts):
        inferred = prior.sample((1,)).squeeze(0)
        nuisance = nuisance_prior.sample((1,)).squeeze(0)
        theta = assemble_canonical_theta(inferred, nuisance, inferred_idx, nuisance_idx)
        result = fn(theta)
        if result is None:
            continue
        if not (isinstance(result, tuple) and len(result) == 3):
            raise RuntimeError(
                "Simulator must return (interphoton_times, channels, length); "
                f"got {type(result).__name__} of length "
                f"{len(result) if hasattr(result, '__len__') else 'N/A'}"
            )
        ipt, chans, length = result
        if ipt.shape != (n_max_photons,) or chans.shape != (n_max_photons,):
            raise RuntimeError(
                f"Simulator returned shapes ipt={ipt.shape}, chans={chans.shape}; "
                f"expected ({n_max_photons},) for both."
            )
        return
    raise RuntimeError(
        f"Could not get a successful simulation after {max_attempts} attempts"
    )


def _prepare_output_dir(output_dir_arg, cfg_output_dir, use_timestamp):
    """Create and return the output directory path."""
    output_dir = Path(output_dir_arg or cfg_output_dir)
    if use_timestamp:
        output_dir = output_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _run_simulation_pool(prior, nuisance_prior, batch_size, num_workers,
                         out_queue, sim_fn, sim_kwargs, num_simulations,
                         written_counter, writer_process=None,
                         num_traces_per_set=1, max_traj_retries=1,
                         max_rounds=500,
                         inferred_idx=None, nuisance_idx=None,
                         nuisance_per_set=False):
    """Run the worker pool, retrying until the simulation budget is met."""
    pool = mp.Pool(
        num_workers,
        initializer=_init_worker,
        initargs=(out_queue, sim_fn, sim_kwargs, nuisance_prior,
                  num_traces_per_set, max_traj_retries,
                  inferred_idx, nuisance_idx, nuisance_per_set),
    )
    total_simulated = 0
    round_num = 0

    sim_bar = tqdm(total=num_simulations, desc="Simulating", position=0,
                   leave=True, unit="sim")
    write_bar = tqdm(total=num_simulations, desc="Writing   ", position=1,
                     leave=True, unit="sim")

    try:
        while total_simulated < num_simulations:
            round_num += 1
            if round_num > max_rounds:
                logger.warning("Reached max retry rounds (%d), stopping with "
                               "%d/%d simulations", max_rounds,
                               total_simulated, num_simulations)
                break

            remaining = num_simulations - total_simulated
            if round_num > 1:
                logger.debug("Round %d: sampling %d more to fill remaining "
                             "budget...", round_num, remaining)

            all_theta = prior.sample((remaining,))
            batches = [
                all_theta[i * batch_size : min((i + 1) * batch_size, remaining)]
                for i in range((remaining + batch_size - 1) // batch_size)
            ]

            for count in pool.imap_unordered(_worker_simulate_batch, batches):
                total_simulated += count
                sim_bar.n = total_simulated
                sim_bar.refresh()
                with written_counter.get_lock():
                    write_bar.n = written_counter.value
                write_bar.refresh()
                if writer_process is not None and not writer_process.is_alive():
                    pool.terminate()
                    pool.join()
                    raise RuntimeError(
                        f"Writer process died (exit code {writer_process.exitcode}). "
                        f"Simulated {total_simulated}/{num_simulations} before failure."
                    )
    finally:
        pool.close()
        pool.join()

    return total_simulated, sim_bar, write_bar


def generate_simulations(cfg, args):
    """Generate simulations using a queue-based writer pattern.

    Returns (h5_path, actual_count, output_dir).
    """
    num_simulations = args.num_simulations or cfg.data_generation.num_simulations
    num_workers = args.num_workers
    batch_size = cfg.data_generation.batch_size
    chunk_size = args.chunk_size or cfg.data_generation.chunk_size

    num_traces_per_set = int(cfg.data_generation.get("num_traces_per_set", 1))
    max_traj_retries = int(cfg.data_generation.get("max_traj_retries", 1))
    nuisance_per_set = bool(cfg.data_generation.get("nuisance_per_set", False))
    n_max_photons = int(cfg.simulator.N_max_photons)

    prior, nuisance_prior, inferred_idx, nuisance_idx = (
        generate_split_priors_from_config(cfg, device="cpu")
    )
    theta_dim = int(inferred_idx.numel())

    # Mode dispatch — build sim_fn and sim_kwargs
    mode = str(cfg.simulator.get("mode", "equilibrium")).lower()
    bc = _validate_bc(cfg.simulator.boundary_condition, allow_open=(mode == "binding"))
    sim_kwargs = dict(
        dt=cfg.simulator.time_step,
        N_max_photons=n_max_photons,
        x_min=cfg.simulator.x_min,
        x_max=cfg.simulator.x_max,
        num_knots=cfg.prior.num_knots,
        max_force=cfg.simulator.max_force,
        boundary_condition=bc,
        spline_type=cfg.simulator.spline_type,
        num_boundary_knots=cfg.simulator.num_boundary_knots,
    )
    if mode == "binding":
        binding_cfg = cfg.simulator.binding
        sim_kwargs.update(
            saving_freq=cfg.simulator.saving_frequency,
            L_min_bins=binding_cfg.L_min_bins,
            L_max_bins=binding_cfg.L_max_bins,
            epsilon=binding_cfg.get("epsilon", 0.05),
            N_max=binding_cfg.get("N_max", 200000000),
            max_shoot_attempts=binding_cfg.get("max_shoot_attempts", 100),
            center_jitter_frac=binding_cfg.get("center_jitter_frac", 0.10),
            n_committor_grid=binding_cfg.get("n_committor_grid", 4000),
            open_left=int(bc[0] == "open"),
            open_right=int(bc[1] == "open"),
        )
        sim_fn = binding_simulator
        print("Mode: binding (per-photon arrival times)")
    else:
        sim_kwargs.update(
            T=cfg.simulator.total_time,
            x_init=tuple(cfg.simulator.x_init),
        )
        sim_fn = fret_simulator
        print("Mode: equilibrium (per-photon arrival times)")

    if num_traces_per_set > 1:
        _nuis_mode = ("nuisance drawn ONCE per set (shared across traces)"
                      if nuisance_per_set else "per-trace nuisance resampling")
        print(f"Set-based mode: {num_traces_per_set} traces per set "
              f"(shared inferred params, {_nuis_mode})")
    if max_traj_retries > 1:
        print(f"Per-trajectory retries: {max_traj_retries}")

    # Sanity-check the simulator returns the expected (ipt, channels, length).
    print("Running test simulation to validate output format...")
    _validate_simulator_returns_events(
        sim_fn, sim_kwargs, prior, nuisance_prior,
        inferred_idx, nuisance_idx, n_max_photons,
    )
    print(f"  theta dim: {theta_dim}, N_max_photons: {n_max_photons}")

    # Prepare output
    output_dir = _prepare_output_dir(args.output_dir, cfg.output.directory, args.timestamp)
    h5_path = str(output_dir / "data.h5")

    set_dim = num_traces_per_set if num_traces_per_set > 1 else 1
    batch_x_mb = (n_max_photons * set_dim * (4 + 1)) * batch_size / (1024 ** 2)
    print(f"Starting {num_simulations} simulations with {num_workers} workers "
          f"(batch_size={batch_size})...")
    print(f"  per-batch result size ~{batch_x_mb:.1f} MB (events+channels)", flush=True)

    # Start dedicated writer process
    out_queue = mp.Queue(maxsize=2 * num_workers)
    done_queue = mp.Queue(maxsize=1)
    written_counter = mp.Value('i', 0)
    writer = mp.Process(
        target=_writer_process,
        args=(h5_path, num_simulations, theta_dim, n_max_photons, chunk_size,
              out_queue, done_queue, written_counter),
        kwargs=dict(num_traces_per_set=num_traces_per_set),
        daemon=False,
    )
    writer.start()

    # Run simulations
    _, sim_bar, write_bar = _run_simulation_pool(
        prior, nuisance_prior, batch_size, num_workers, out_queue, sim_fn,
        sim_kwargs, num_simulations, written_counter, writer_process=writer,
        num_traces_per_set=num_traces_per_set,
        max_traj_retries=max_traj_retries,
        inferred_idx=inferred_idx, nuisance_idx=nuisance_idx,
        nuisance_per_set=nuisance_per_set,
    )

    # Signal writer to finish
    out_queue.put(None)
    writer.join()
    # Final drain of the writer counter after the writer has flushed.
    with written_counter.get_lock():
        write_bar.n = written_counter.value
    write_bar.refresh()
    sim_bar.close()
    write_bar.close()
    if writer.exitcode != 0:
        raise RuntimeError(f"Writer process failed with exit code {writer.exitcode}")

    actual_count = int(done_queue.get())
    if actual_count < num_simulations:
        print(f"  Note: {num_simulations - actual_count} simulations failed, "
              f"trimmed to {actual_count}")
    print(f"  HDF5 file written: {h5_path}  ({actual_count} simulations, gzip level 1)")

    return h5_path, actual_count, output_dir


def save_metadata(output_dir, cfg, config_path, args, actual_count, h5_path):
    """Save a copy of the config and a metadata summary alongside the data."""
    shutil.copy2(config_path, output_dir / "config.yaml")

    metadata = {
        "generated_at": datetime.now().isoformat(),
        "num_simulations": actual_count,
        "source_config": str(Path(config_path).resolve()),
        "output_file": str(h5_path),
        "command_line_overrides": {
            "num_simulations": args.num_simulations,
            "num_workers": args.num_workers,
            "output_dir": args.output_dir,
        },
    }
    with open(output_dir / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)

    print("  config.yaml (copy of configuration)")
    print("  metadata.yaml (generation info)")


# ── Entry point ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    print(f"Loading config from: {config_path}")
    config = OmegaConf.load(config_path)

    h5_path, actual_count, output_dir = generate_simulations(config, args)
    save_metadata(output_dir, config, str(config_path), args, actual_count, h5_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
