import argparse
import shutil
from pathlib import Path
from datetime import datetime
import torch
import h5py
from functools import partial
from sbi.utils.user_input_checks import process_simulator
from sbi.inference import simulate_for_sbi
from smFRET_sbi.src.simulator import smFRET_simulator_wrapper as fret_simulator
from smFRET_sbi.src.priors import generate_spline_fret_prior_from_config, load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate FRET simulation training data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="conf/simulator/default.yaml",
        help="Path to YAML configuration file"
    )
    
    # Override options for quick experiments
    parser.add_argument(
        "--num-simulations", "-n",
        type=int,
        default=None,
        help="Override number of simulations from config"
    )
    
    parser.add_argument(
        "--num-workers", "-w",
        type=int,
        default=None,
        help="Override number of workers from config"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Override output directory from config"
    )
    
    parser.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="Add suffix to output filenames (e.g., 'small' -> theta_small.pt)"
    )
    
    parser.add_argument(
        "--timestamp",
        action="store_true",
        help="Add timestamp to output directory name"
    )
    
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=['pt', 'h5', 'both'],
        default='pt',
        help="Output format: 'pt' (PyTorch), 'h5' (HDF5), or 'both'"
    )
    
    return parser.parse_args()


def generate_simulations(config: dict, args) -> tuple:
    """Generate simulations based on configuration."""
    
    sim_cfg = config['simulator']
    data_cfg = config['data_generation']
    
    num_simulations = args.num_simulations or data_cfg['num_simulations']
    num_workers = args.num_workers or data_cfg['num_workers']
    batch_size = data_cfg['batch_size']
    
    prior = generate_spline_fret_prior_from_config(config)
    
    simulator = partial(
        fret_simulator,
        T=sim_cfg['total_time'],
        deltax=sim_cfg['deltax'],
        saving_freq=sim_cfg['saving_frequency'],
        dt=sim_cfg['time_step']
    )

    simulator = process_simulator(simulator, is_numpy_simulator=False, prior=prior)
    
    theta, x = simulate_for_sbi(
        simulator,
        prior,
        num_simulations=num_simulations,
        num_workers=num_workers,
        simulation_batch_size=batch_size
    )
    
    return theta[:, 0:sim_cfg['num_spline_knots']+1], x


def save_outputs(theta, x, config: dict, config_path: str, args):
    
    output_cfg = config['output']
    
    output_dir = Path(args.output_dir or output_cfg['directory'])
    
    if args.timestamp:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = output_dir / timestamp
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Base filenames (without extension)
    theta_base = output_cfg['theta_filename'].replace('.pt', '')
    x_base = output_cfg['x_filename'].replace('.pt', '')
    
    if args.suffix:
        theta_base = f"{theta_base}_{args.suffix}"
        x_base = f"{x_base}_{args.suffix}"
    
    print(f"\nSaved outputs to {output_dir}:")
    
    # Save as PyTorch tensors
    if args.format in ['pt', 'both']:
        theta_path = output_dir / f"{theta_base}.pt"
        x_path = output_dir / f"{x_base}.pt"
        torch.save(theta, theta_path)
        torch.save(x, x_path)
        print(f"  {theta_base}.pt: {theta.shape}")
        print(f"  {x_base}.pt: {x.shape}")
    
    # Save as HDF5
    if args.format in ['h5', 'both']:
        h5_path = output_dir / f"data{('_' + args.suffix) if args.suffix else ''}.h5"
        with h5py.File(h5_path, 'w') as f:
            # No compression for fast reads; chunks aligned to sample boundaries
            # Chunk size of 1000 samples balances I/O efficiency and memory
            theta_chunks = (min(1000, theta.shape[0]),) + theta.shape[1:]
            x_chunks = (min(1000, x.shape[0]),) + x.shape[1:]
            
            f.create_dataset('theta', data=theta.numpy(), chunks=theta_chunks)
            f.create_dataset('x', data=x.numpy(), chunks=x_chunks)
            # Store metadata as attributes
            f.attrs['generated_at'] = datetime.now().isoformat()
            f.attrs['num_simulations'] = theta.shape[0]
            f.attrs['theta_shape'] = list(theta.shape)
            f.attrs['x_shape'] = list(x.shape)
            f.attrs['source_config'] = str(Path(config_path).resolve())
        print(f"  data{('_' + args.suffix) if args.suffix else ''}.h5: theta{list(theta.shape)}, x{list(x.shape)} (uncompressed)")
    
    config_copy_path = output_dir / "config.yaml"
    shutil.copy2(config_path, config_copy_path)
    print(f"  config.yaml (copy of configuration)")
    
    metadata = {
        'generated_at': datetime.now().isoformat(),
        'num_simulations': theta.shape[0],
        'theta_shape': list(theta.shape),
        'x_shape': list(x.shape),
        'source_config': str(Path(config_path).resolve()),
        'command_line_overrides': {
            'num_simulations': args.num_simulations,
            'num_workers': args.num_workers,
            'output_dir': args.output_dir,
            'suffix': args.suffix,
        }
    }
    
    metadata_path = output_dir / "metadata.yaml"
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)
    print(f"  metadata.yaml (generation info)")


def main():
    args = parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)
    
    # Generate simulations
    theta, x = generate_simulations(config, args)
    
    # Save outputs
    save_outputs(theta, x, config, str(config_path), args)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
