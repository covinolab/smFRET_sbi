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
_worker_lock = None
_worker_h5_path = None
_worker_sim_fn = None
_worker_theta_dim = None


def _init_worker(lock, h5_path, sim_kwargs, theta_dim):
    """Set up shared state for each pool worker."""
    global _worker_lock, _worker_h5_path, _worker_sim_fn, _worker_theta_dim
    _worker_lock = lock
    _worker_h5_path = h5_path
    _worker_theta_dim = theta_dim
    _worker_sim_fn = partial(fret_simulator, **sim_kwargs)
    # Give each worker a unique random seed so simulations are independent
    np.random.seed(os.getpid() % (2**32))


def _worker_simulate_batch_and_write(batch_theta):
    """
    Worker function: simulate a batch of parameters and write the results
    directly into the shared HDF5 file (one lock-protected write per batch).
    """
    results_theta = []
    results_x = []

    for i in range(len(batch_theta)):
        x = _worker_sim_fn(batch_theta[i])
        if x is not None:
            results_theta.append(batch_theta[i][:_worker_theta_dim])
            results_x.append(x)

    if not results_theta:
        return 0

    thetas = torch.stack(results_theta)
    xs = torch.stack(results_x)

    # ── Acquire lock and write directly into the HDF5 file ──
    with _worker_lock:
        with h5py.File(
            _worker_h5_path, "a",
            libver="latest",
            rdcc_nbytes=64 * 1024 * 1024,  # 64 MB chunk cache
            rdcc_nslots=521,
        ) as f:
            idx = int(f.attrs["write_idx"])
            n = len(thetas)
            f["theta"][idx : idx + n] = thetas.numpy()
            f["x"][idx : idx + n] = xs.numpy()
            f.attrs["write_idx"] = idx + n

    return n


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
        "--suffix",
        type=str,
        default=None,
        help="Add suffix to output filenames (e.g., 'small' -> data_small.h5)",
    )
    parser.add_argument(
        "--timestamp",
        action="store_true",
        help="Add timestamp to output directory name",
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


def _compress_h5(src_path, dst_path, compression="gzip", compression_opts=1):
    """
    Copy an uncompressed HDF5 file into a new compressed one.
    This is much faster than compressing on every batch write because
    compression runs in a single sequential pass over contiguous data.
    """
    with h5py.File(src_path, "r") as fin, h5py.File(dst_path, "w", libver="latest") as fout:
        for key in fin:
            data = fin[key]
            chunks = (min(512, data.shape[0]),) + data.shape[1:]
            fout.create_dataset(
                key,
                data=data[:],
                chunks=chunks,
                compression=compression,
                compression_opts=compression_opts,
            )
        # Copy attributes
        for attr_key, attr_val in fin.attrs.items():
            fout.attrs[attr_key] = attr_val


# ─── Main generation routine ────────────────────────────────────────────────
def generate_simulations(cfg, args):
    """
    Generate simulations with each worker writing directly into HDF5.

    Workflow
    --------
    1. Pre-allocate an HDF5 file with datasets of the correct shape.
    2. Sample all parameter sets from the prior (small memory footprint).
    3. Distribute batches to a multiprocessing pool.
    4. Each worker simulates its batch and writes results straight into the
       HDF5 file (guarded by a shared lock).
    5. Trim datasets if some simulations returned None.
    """
    num_simulations = args.num_simulations or cfg.data_generation.num_simulations
    num_workers = args.num_workers or cfg.data_generation.num_workers
    batch_size = cfg.data_generation.batch_size
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

    suffix = f"_{args.suffix}" if args.suffix else ""
    h5_path = str(output_dir / f"data{suffix}.h5")

    # ── Pre-allocate HDF5 datasets (uncompressed for fast writes) ──
    # Align chunk size to batch_size so each write fills whole chunks
    # and avoids expensive read-modify-write cycles on partial chunks.
    theta_chunks = (min(batch_size, num_simulations), theta_dim)
    x_chunks = (min(batch_size, num_simulations),) + x_shape

    with h5py.File(h5_path, "w", libver="latest") as f:
        f.create_dataset(
            "theta",
            shape=(num_simulations, theta_dim),
            maxshape=(num_simulations, theta_dim),
            dtype="float32",
            chunks=theta_chunks,
            # No compression during writes — compress in a final pass
        )
        f.create_dataset(
            "x",
            shape=(num_simulations,) + x_shape,
            maxshape=(num_simulations,) + x_shape,
            dtype="float32",
            chunks=x_chunks,
            # No compression during writes — compress in a final pass
        )
        f.attrs["write_idx"] = 0

    # ── Sample parameters from the prior ──
    print(f"Sampling {num_simulations} parameter sets from prior...")
    all_theta = prior.sample((num_simulations,))

    # Split into worker batches
    num_batches = (num_simulations + batch_size - 1) // batch_size
    batches = [
        all_theta[i * batch_size : min((i + 1) * batch_size, num_simulations)]
        for i in range(num_batches)
    ]

    # ── Run simulations ──
    print(
        f"Starting {num_simulations} simulations with {num_workers} workers "
        f"(batch_size={batch_size}, {num_batches} batches)..."
    )
    manager = mp.Manager()
    lock = manager.Lock()

    total_written = 0
    with mp.Pool(
        num_workers,
        initializer=_init_worker,
        initargs=(lock, h5_path, sim_kwargs, theta_dim),
    ) as pool:
        for count in pool.imap_unordered(
            _worker_simulate_batch_and_write, batches
        ):
            total_written += count
            print(
                f"\r  Progress: {total_written}/{num_simulations} written",
                end="",
                flush=True,
            )

    print()

    # ── Finalise the HDF5 file ──
    with h5py.File(h5_path, "a") as f:
        actual_count = int(f.attrs["write_idx"])
        if actual_count < num_simulations:
            print(
                f"  Note: {num_simulations - actual_count} simulations failed, "
                f"resizing datasets to {actual_count}"
            )
            f["theta"].resize(actual_count, axis=0)
            f["x"].resize(actual_count, axis=0)
        f.attrs["num_simulations"] = actual_count
        f.attrs["generated_at"] = datetime.now().isoformat()
        f.attrs["theta_shape"] = list(f["theta"].shape)
        f.attrs["x_shape"] = list(f["x"].shape)

    # ── Compress the final file ──
    compressed_path = h5_path.replace(".h5", "_compressed.h5")
    print(f"  Compressing {h5_path} -> {compressed_path} ...")
    _compress_h5(h5_path, compressed_path)
    os.replace(compressed_path, h5_path)
    print(f"  HDF5 file written: {h5_path}  ({actual_count} simulations, gzip)")

    # ── Optionally export .pt copies ──
    if args.format in ("pt", "both"):
        theta_base = cfg.output.theta_filename.replace(".pt", "")
        x_base = cfg.output.x_filename.replace(".pt", "")
        if args.suffix:
            theta_base = f"{theta_base}_{args.suffix}"
            x_base = f"{x_base}_{args.suffix}"

        with h5py.File(h5_path, "r") as f:
            torch.save(torch.from_numpy(f["theta"][:]), output_dir / f"{theta_base}.pt")
            torch.save(torch.from_numpy(f["x"][:]), output_dir / f"{x_base}.pt")
        print(f"  Saved {theta_base}.pt and {x_base}.pt")

    # If the user only wanted .pt, remove the intermediate HDF5
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
            "suffix": args.suffix,
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
