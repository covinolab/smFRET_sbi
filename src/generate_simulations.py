import argparse
import os
import shutil
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
import torch
import h5py
import numpy as np
from omegaconf import OmegaConf
import yaml
from functools import partial
from simulator import smFRET_simulator_wrapper as fret_simulator
from priors import generate_fret_prior_from_config


# ─── Worker globals (initialised once per pool worker by _init_worker) ───────
_worker_sim_fn = None
_worker_theta_dim = None
_worker_out_queue = None


def _init_worker(out_queue, sim_kwargs, theta_dim):
    """Set up shared state for each pool worker."""
    global _worker_sim_fn, _worker_theta_dim, _worker_out_queue
    _worker_theta_dim = theta_dim
    _worker_out_queue = out_queue
    _worker_sim_fn = partial(fret_simulator, **sim_kwargs)
    np.random.seed(os.getpid() % (2**32))
    torch.set_num_threads(1)  # prevent OpenMP deadlocks in forked workers


def _worker_simulate_batch(batch_theta):
    """
    Worker function: simulate a batch, put results on an unbounded queue,
    and return only a tiny count (avoids large-result pipe deadlocks).
    """
    results_theta = []
    results_x = []

    for i in range(len(batch_theta)):
        x = _worker_sim_fn(batch_theta[i])
        if x is not None:
            results_theta.append(batch_theta[i][:_worker_theta_dim].numpy())
            results_x.append(x)

    if not results_theta:
        return 0

    thetas = np.stack(results_theta)
    xs = np.stack(results_x)

    _worker_out_queue.put((thetas, xs))
    return len(results_theta)


def _writer_process(h5_path, num_simulations, theta_dim, x_shape, chunk_size, in_queue, done_queue):
    """
    Dedicated writer process: keeps HDF5 file open for the entire run
    and appends batches received via the queue.
    """
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
            compression="gzip",
            compression_opts=1,
            shuffle=True,
        )
        f.create_dataset(
            "x",
            shape=(num_simulations,) + x_shape,
            maxshape=(num_simulations,) + x_shape,
            dtype="float32",
            chunks=x_chunks,
            compression="gzip",
            compression_opts=1,
            shuffle=True,
        )

        while True:
            item = in_queue.get()

            if item is None:          # sentinel → all workers done
                break

            theta_np, x_np = item
            n = theta_np.shape[0]

            f["theta"][idx : idx + n] = theta_np
            f["x"][idx : idx + n] = x_np

            idx += n
            print(
                f"\r  Writing: {idx}/{num_simulations} written",
                end="",
                flush=True,
            )

        print()

        # Trim if some simulations returned None
        if idx < num_simulations:
            f["theta"].resize(idx, axis=0)
            f["x"].resize(idx, axis=0)

        f.attrs["write_idx"] = idx
        f.attrs["num_simulations"] = idx
        f.attrs["generated_at"] = datetime.now().isoformat()
        f.attrs["theta_shape"] = list(f["theta"].shape)
        f.attrs["x_shape"] = list(f["x"].shape)

    done_queue.put(idx)


# ─── CLI ─────────────────────────────────────────────────────────────────────
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate FRET simulation training data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        default="conf/simulator/default.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--num-simulations", "-n",
        type=int,
        default=None,
        help="Override number of simulations from config",
    )
    parser.add_argument(
        "--num-workers", "-w",
        type=int,
        default=None,
        help="Override number of workers from config",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Override output directory from config",
    )
    parser.add_argument(
        "--timestamp",
        action="store_true",
        help="Add timestamp to output directory name",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="HDF5 chunk size (number of samples per chunk). Defaults to batch_size.",
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["pt", "h5", "both"],
        default="h5",
        help="Output format: 'h5' (HDF5, default), 'pt' (PyTorch), or 'both'",
    )

    return parser.parse_args()


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _get_sim_kwargs(cfg):
    """Build the keyword-argument dict for the simulator from the config."""
    return dict(
        T=cfg.simulator.total_time,
        saving_freq=cfg.simulator.saving_frequency,
        dt=cfg.simulator.time_step,
        x_min=cfg.simulator.x_min,
        x_max=cfg.simulator.x_max,
        x_init=tuple(cfg.simulator.x_init),
        num_knots=cfg.prior.num_knots,
        max_G0=cfg.simulator.max_G0,
        boundary_condition=cfg.simulator.boundary_condition,
        spline_type=cfg.simulator.spline_type,
    )


def _determine_output_shape(sim_kwargs, prior, max_attempts=100):
    """Run a test simulation to discover the observation tensor shape."""
    sim_fn = partial(fret_simulator, **sim_kwargs)
    for _ in range(max_attempts):
        theta = prior.sample((1,)).squeeze(0)
        x = sim_fn(theta)
        if x is not None:
            return tuple(x.shape)
    raise RuntimeError(
        f"Could not determine output shape after {max_attempts} test simulations"
    )


# ─── Main generation routine ────────────────────────────────────────────────
def generate_simulations(cfg, args):
    """
    Generate simulations using a queue-based writer pattern.

    Workflow
    --------
    1. Start a dedicated writer process that owns the HDF5 file.
    2. Sample all parameter sets from the prior.
    3. Workers simulate batches and push results to a queue.
    4. The writer drains the queue and writes sequentially (no lock needed).
    5. Trim datasets if some simulations returned None.
    """
    num_simulations = args.num_simulations or cfg.data_generation.num_simulations
    num_workers = args.num_workers or cfg.data_generation.num_workers
    batch_size = cfg.data_generation.batch_size
    chunk_size = args.chunk_size or cfg.data_generation.chunk_size
    theta_dim = cfg.prior.num_knots + 1

    prior = generate_fret_prior_from_config(cfg, device="cpu")
    sim_kwargs = _get_sim_kwargs(cfg)

    # ── Discover observation shape ──
    print("Running test simulation to determine output shape...")
    x_shape = _determine_output_shape(sim_kwargs, prior)
    print(f"  theta dim: {theta_dim}, observation shape: {x_shape}")

    # ── Prepare output directory ──
    output_dir = Path(args.output_dir or cfg.output.directory)
    if args.timestamp:
        output_dir = output_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    h5_path = str(output_dir / "data.h5")

    # ── Sample parameters from the prior ──
    print(f"Sampling {num_simulations} parameter sets from prior...")
    all_theta = prior.sample((num_simulations,))

    num_batches = (num_simulations + batch_size - 1) // batch_size
    batches = [
        all_theta[i * batch_size : min((i + 1) * batch_size, num_simulations)]
        for i in range(num_batches)
    ]

    # ── Start the dedicated writer process ──
    # No maxsize — workers never block on put(), avoiding deadlock.
    # Memory is bounded: at most num_batches items in flight (~few hundred MB).
    out_queue = mp.Queue()
    done_queue = mp.Queue(maxsize=1)

    writer = mp.Process(
        target=_writer_process,
        args=(h5_path, num_simulations, theta_dim, x_shape, chunk_size,
              out_queue, done_queue),
        daemon=False,
    )
    writer.start()

    # ── Run simulations ──
    print(
        f"Starting {num_simulations} simulations with {num_workers} workers "
        f"(batch_size={batch_size}, {num_batches} batches)..."
    )
    # Print expected data sizes
    single_x_bytes = 4  # float32
    for d in x_shape:
        single_x_bytes *= d
    batch_x_mb = (single_x_bytes * batch_size) / (1024 * 1024)
    print(f"  x_shape={x_shape}, per-batch result size ~{batch_x_mb:.1f} MB", flush=True)

    total_simulated = 0
    pool = mp.Pool(
        num_workers,
        initializer=_init_worker,
        initargs=(out_queue, sim_kwargs, theta_dim),
    )
    for count in pool.imap_unordered(_worker_simulate_batch, batches):
        total_simulated += count
        print(
            f"\r  Progress: {total_simulated}/{num_simulations} simulated",
            end="",
            flush=True,
        )

    pool.close()
    pool.join()

    print()

    # Signal writer to finish, then wait
    out_queue.put(None)
    writer.join()
    if writer.exitcode != 0:
        raise RuntimeError(f"Writer process failed with exit code {writer.exitcode}")

    actual_count = int(done_queue.get())
    if actual_count < num_simulations:
        print(
            f"  Note: {num_simulations - actual_count} simulations failed, "
            f"trimmed to {actual_count}"
        )
    print(f"  HDF5 file written: {h5_path}  ({actual_count} simulations, gzip level 1)")

    # ── Optionally export .pt copies ──
    if args.format in ("pt", "both"):
        with h5py.File(h5_path, "r") as f:
            torch.save(torch.from_numpy(f["theta"][:]), output_dir / "theta.pt")
            torch.save(torch.from_numpy(f["x"][:]), output_dir / "x.pt")
        print("  Saved theta.pt and x.pt")

    if args.format == "pt":
        Path(h5_path).unlink()
        h5_path = None

    return h5_path, actual_count, output_dir


def save_metadata(output_dir, cfg, config_path, args, actual_count, h5_path):
    """Save a copy of the config and a metadata summary."""
    # Copy config
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

    print(f"  config.yaml (copy of configuration)")
    print(f"  metadata.yaml (generation info)")


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
