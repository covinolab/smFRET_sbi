import argparse
import os
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

from simulator import smFRET_simulator_wrapper as fret_simulator
from simulator import binding_simulator_wrapper as binding_simulator
from priors import generate_fret_prior_from_config


# ── Multiprocessing workers ─────────────────────────────────────────────────

_worker_sim_fn = None
_worker_theta_dim = None
_worker_out_queue = None
_worker_has_lengths = False


def _init_worker(out_queue, sim_fn, sim_kwargs, theta_dim, has_lengths):
    """Initialise per-worker shared state (called once per pool worker)."""
    global _worker_sim_fn, _worker_theta_dim, _worker_out_queue, _worker_has_lengths
    _worker_theta_dim = theta_dim
    _worker_out_queue = out_queue
    _worker_has_lengths = has_lengths
    _worker_sim_fn = partial(sim_fn, **sim_kwargs)
    np.random.seed(os.getpid() % (2**32))
    torch.set_num_threads(1)


def _worker_simulate_batch(batch_theta):
    """Simulate a batch, push results to the writer queue, return success count."""
    results_theta, results_x, results_lengths = [], [], []

    for i in range(len(batch_theta)):
        result = _worker_sim_fn(batch_theta[i])
        if result is None:
            continue
        if _worker_has_lengths:
            trace, length = result
            results_x.append(trace)
            results_lengths.append(length)
        else:
            results_x.append(result)
        results_theta.append(batch_theta[i][:_worker_theta_dim].numpy())

    if not results_theta:
        return 0

    thetas, xs = np.stack(results_theta), np.stack(results_x)
    if _worker_has_lengths:
        _worker_out_queue.put((thetas, xs, np.array(results_lengths, dtype=np.int32)))
    else:
        _worker_out_queue.put((thetas, xs))
    return len(results_theta)


# ── HDF5 writer process ────────────────────────────────────────────────────

def _writer_process(h5_path, num_simulations, theta_dim, x_shape, chunk_size,
                    in_queue, done_queue, written_counter, has_lengths=False):
    """Dedicated writer process: drains the queue and appends batches to HDF5."""
    theta_chunks = (max(1, min(chunk_size, num_simulations)), theta_dim)
    x_chunks = (max(1, min(chunk_size, num_simulations)),) + x_shape
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
            "x",
            shape=(num_simulations,) + x_shape,
            maxshape=(num_simulations,) + x_shape,
            dtype="float32",
            chunks=x_chunks,
            compression="gzip", compression_opts=1, shuffle=True,
        )
        if has_lengths:
            f.create_dataset(
                "lengths",
                shape=(num_simulations,),
                maxshape=(num_simulations,),
                dtype="int32",
                chunks=(max(1, min(chunk_size, num_simulations)),),
                compression="gzip", compression_opts=1,
            )

        while True:
            item = in_queue.get()
            if item is None:
                break

            if has_lengths:
                theta_np, x_np, lengths_np = item
            else:
                theta_np, x_np = item
            n = theta_np.shape[0]

            f["theta"][idx : idx + n] = theta_np
            f["x"][idx : idx + n] = x_np
            if has_lengths:
                f["lengths"][idx : idx + n] = lengths_np
            idx += n
            with written_counter.get_lock():
                written_counter.value = idx

        # Trim if some simulations returned None
        if idx < num_simulations:
            f["theta"].resize(idx, axis=0)
            f["x"].resize(idx, axis=0)
            if has_lengths:
                f["lengths"].resize(idx, axis=0)

        f.attrs["write_idx"] = idx
        f.attrs["num_simulations"] = idx
        f.attrs["generated_at"] = datetime.now().isoformat()
        f.attrs["theta_shape"] = list(f["theta"].shape)
        f.attrs["x_shape"] = list(f["x"].shape)
        if has_lengths:
            f.attrs["has_lengths"] = True

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
    parser.add_argument("--format", "-f", type=str, choices=["pt", "h5", "both"],
                        default="h5", help="Output format: 'h5', 'pt', or 'both'")
    return parser.parse_args()


# ── Configuration helpers ───────────────────────────────────────────────────

def _get_common_sim_kwargs(cfg):
    """Extract simulator kwargs shared between equilibrium and binding modes."""
    return dict(
        dt=cfg.simulator.time_step,
        saving_freq=cfg.simulator.saving_frequency,
        x_min=cfg.simulator.x_min,
        x_max=cfg.simulator.x_max,
        num_knots=cfg.prior.num_knots,
        max_force=cfg.simulator.max_force,
        boundary_condition=cfg.simulator.boundary_condition,
        spline_type=cfg.simulator.spline_type,
        num_boundary_knots=cfg.simulator.num_boundary_knots,
    )


def _get_sim_kwargs(cfg):
    """Build kwargs for the equilibrium simulator."""
    return {
        **_get_common_sim_kwargs(cfg),
        "T": cfg.simulator.total_time,
        "x_init": tuple(cfg.simulator.x_init),
    }


def _get_binding_sim_kwargs(cfg):
    """Build kwargs for the binding simulator."""
    binding_cfg = cfg.simulator.get("binding", {})
    return {
        **_get_common_sim_kwargs(cfg),
        "L_min_bins":      binding_cfg.L_min_bins,
        "L_max_bins":      binding_cfg.L_max_bins,
        "k_persist":       binding_cfg.k_persist,
        "landmark_grid_n": binding_cfg.get("landmark_grid_n", 400),
        "N_max":           binding_cfg.get("N_max", 200000000),
    }


# ── Simulation orchestration ───────────────────────────────────────────────

def _determine_output_shape(sim_fn, sim_kwargs, prior, has_lengths=False,
                            max_attempts=100):
    """Run test simulations to discover the observation tensor shape."""
    fn = partial(sim_fn, **sim_kwargs)
    for _ in range(max_attempts):
        theta = prior.sample((1,)).squeeze(0)
        result = fn(theta)
        if result is not None:
            if has_lengths:
                trace, _ = result
                return tuple(trace.shape)
            return tuple(result.shape)
    raise RuntimeError(
        f"Could not determine output shape after {max_attempts} test simulations"
    )


def _prepare_output_dir(output_dir_arg, cfg_output_dir, use_timestamp):
    """Create and return the output directory path."""
    output_dir = Path(output_dir_arg or cfg_output_dir)
    if use_timestamp:
        output_dir = output_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _run_simulation_pool(prior, batch_size, num_workers, out_queue, sim_fn,
                         sim_kwargs, theta_dim, num_simulations,
                         written_counter, has_lengths=False, max_rounds=50):
    """Run the worker pool, retrying until the simulation budget is met."""
    pool = mp.Pool(
        num_workers,
        initializer=_init_worker,
        initargs=(out_queue, sim_fn, sim_kwargs, theta_dim, has_lengths),
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
                tqdm.write(f"  Warning: reached max retry rounds ({max_rounds}), "
                           f"stopping with {total_simulated}/{num_simulations} simulations")
                break

            remaining = num_simulations - total_simulated
            if round_num > 1:
                tqdm.write(f"  Round {round_num}: sampling {remaining} more to fill "
                           f"remaining budget...")

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
    finally:
        pool.close()
        pool.join()

    return total_simulated, sim_bar, write_bar


def _export_pt(h5_path, output_dir):
    """Export theta and x datasets from HDF5 as PyTorch .pt files."""
    with h5py.File(h5_path, "r") as f:
        torch.save(torch.from_numpy(f["theta"][:]), output_dir / "theta.pt")
        torch.save(torch.from_numpy(f["x"][:]), output_dir / "x.pt")
    print("  Saved theta.pt and x.pt")


def generate_simulations(cfg, args):
    """Generate simulations using a queue-based writer pattern.

    Returns (h5_path, actual_count, output_dir).
    """
    num_simulations = args.num_simulations or cfg.data_generation.num_simulations
    num_workers = args.num_workers or cfg.data_generation.num_workers
    batch_size = cfg.data_generation.batch_size
    chunk_size = args.chunk_size or cfg.data_generation.chunk_size
    theta_dim = cfg.prior.num_knots + 1

    prior = generate_fret_prior_from_config(cfg, device="cpu")

    # Mode dispatch
    mode = str(cfg.simulator.get("mode", "equilibrium")).lower()
    has_lengths = (mode == "binding")
    if mode == "binding":
        sim_fn, sim_kwargs = binding_simulator, _get_binding_sim_kwargs(cfg)
        print("Mode: binding (h-transform, variable-length traces)")
    else:
        sim_fn, sim_kwargs = fret_simulator, _get_sim_kwargs(cfg)
        print("Mode: equilibrium (fixed-length traces)")

    # Discover observation shape
    print("Running test simulation to determine output shape...")
    x_shape = _determine_output_shape(sim_fn, sim_kwargs, prior, has_lengths=has_lengths)
    print(f"  theta dim: {theta_dim}, observation shape: {x_shape}")

    # Prepare output
    output_dir = _prepare_output_dir(args.output_dir, cfg.output.directory, args.timestamp)
    h5_path = str(output_dir / "data.h5")

    batch_x_mb = int(np.prod(x_shape)) * 4 * batch_size / (1024 ** 2)
    print(f"Starting {num_simulations} simulations with {num_workers} workers "
          f"(batch_size={batch_size})...")
    print(f"  x_shape={x_shape}, per-batch result size ~{batch_x_mb:.1f} MB", flush=True)

    # Start dedicated writer process
    out_queue = mp.Queue()
    done_queue = mp.Queue(maxsize=1)
    written_counter = mp.Value('i', 0)
    writer = mp.Process(
        target=_writer_process,
        args=(h5_path, num_simulations, theta_dim, x_shape, chunk_size,
              out_queue, done_queue, written_counter),
        kwargs=dict(has_lengths=has_lengths),
        daemon=False,
    )
    writer.start()

    # Run simulations
    _, sim_bar, write_bar = _run_simulation_pool(
        prior, batch_size, num_workers, out_queue, sim_fn, sim_kwargs,
        theta_dim, num_simulations, written_counter, has_lengths=has_lengths,
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

    # Optionally export .pt copies
    if args.format in ("pt", "both"):
        _export_pt(h5_path, output_dir)
    if args.format == "pt":
        Path(h5_path).unlink()
        h5_path = None

    return h5_path, actual_count, output_dir


def save_metadata(output_dir, cfg, config_path, args, actual_count, h5_path):
    """Save a copy of the config and a metadata summary alongside the data."""
    shutil.copy2(config_path, output_dir / "config.yaml")

    metadata = {
        "generated_at": datetime.now().isoformat(),
        "num_simulations": actual_count,
        "source_config": str(Path(config_path).resolve()),
        "output_file": str(h5_path) if h5_path else None,
        "format": args.format,
        "command_line_overrides": {
            "num_simulations": args.num_simulations,
            "num_workers": args.num_workers,
            "output_dir": args.output_dir,
            "format": args.format,
        },
    }
    with open(output_dir / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)

    print("  config.yaml (copy of configuration)")
    print("  metadata.yaml (generation info)")


# ── Entry point ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

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
