from functools import partial
import torch
import numpy as np
from utils import make_spline_knots, integrate_force_to_energy
from smFRET_simulator import smFRET_simulator, binding_simulator
from omegaconf import OmegaConf

def smFRET_simulator_wrapper(
                params, 
                dt=5e-4, 
                saving_freq=1, 
                T=10000.0, 
                x_init=(3.0, 7.0),
                x_min=0.0,
                x_max=10.0,
                num_knots=10,
                max_force=10.0,
                boundary_condition='diffusive',
                spline_type="cubic",
                num_boundary_knots=4,
                return_torch=False
    ):
            
        x_init = np.random.uniform(low=x_init[0], high=x_init[1])
        x_knots, y_knots = make_spline_knots(
            params=params,
            x_min=x_min,
            x_max=x_max,
            num_knots=num_knots,
            max_force=max_force,
            boundary_condition=boundary_condition,  
            num_boundary_knots=num_boundary_knots
        )

        kG_photons, kR_photons = smFRET_simulator(
                x0=x_init,
                Dx=10**params[0],
                x_knots=x_knots,
                y_knots=y_knots,
                R0=params[num_knots+1],
                kD=params[num_knots+2],
                k_gb=params[num_knots+3],
                k_rb=params[num_knots+4],
                eta_g=params[num_knots+5],
                eta_r=params[num_knots+6],
                C_gg=params[num_knots+7],
                C_rr=params[num_knots+8],
                C_gr=params[num_knots+9],
                C_rg=params[num_knots+10],
                T=T,
                saving_freq=saving_freq,
                dt=dt,
                spline_type=spline_type

        )
        if kG_photons is None or kR_photons is None:
            return None

        if return_torch:
            kG_photons = torch.from_numpy(kG_photons).float()
            kR_photons = torch.from_numpy(kR_photons).float()
            return torch.stack([kG_photons, kR_photons], dim=1)

        return np.stack([kG_photons, kR_photons], axis=1).astype(np.float32)


def smFRET_simulator_wrapper_from_config(cfg: str):
    """
    Generate a smFRET simulator wrapper from a configuration file or dict.
    
    Expected config structure:
        simulator:
            total_time: 5000.0
            saving_frequency: 2000
            time_step: 5e-4
            spline_type: "cubic"
            boundary_condition: "bound"
    """

    cfg = OmegaConf.load(cfg) if isinstance(cfg, str) else cfg

    total_time = cfg.simulator.total_time
    saving_frequency = cfg.simulator.saving_frequency
    time_step = cfg.simulator.time_step
    spline_type = cfg.simulator.spline_type
    boundary_condition = cfg.simulator.boundary_condition
    num_knots = cfg.prior.num_knots
    max_force = cfg.simulator.max_force
    x_min = cfg.simulator.x_min
    x_max = cfg.simulator.x_max
    num_boundary_knots = cfg.simulator.num_boundary_knots
    x_init = tuple(cfg.simulator.x_init)

    return partial(
        smFRET_simulator_wrapper,
        dt=time_step,
        saving_freq=saving_frequency,
        T=total_time,
        x_init=x_init,
        x_min=x_min,
        x_max=x_max,
        num_knots=num_knots,
        max_force=max_force,
        boundary_condition=boundary_condition,
        spline_type=spline_type,
        num_boundary_knots=num_boundary_knots,
        return_torch=True
    )


def binding_simulator_wrapper(
                params,
                dt=5e-5,
                saving_freq=1,
                x_min=4.0,
                x_max=14.0,
                num_knots=5,
                max_force=30.0,
                boundary_condition='linear',
                spline_type="cubic",
                num_boundary_knots=3,
                n_grid=500,
                F_guide_max_factor=0.1,
                N_max=200000000,
                min_steps=0,
                max_bins=None,
                return_torch=False
    ):
        x_knots, y_knots = make_spline_knots(
            params=params,
            x_min=x_min,
            x_max=x_max,
            num_knots=num_knots,
            max_force=max_force,
            boundary_condition=boundary_condition,
            num_boundary_knots=num_boundary_knots
        )

        # Find well minima for x_A and x_B
        # Left well: between first boundary knot and first interior knot
        # Right well: between last interior knot and first right boundary knot
        nb = num_boundary_knots
        left_grid = np.linspace(x_knots[nb - 1], x_knots[nb], 100)
        right_grid = np.linspace(x_knots[-nb - 1], x_knots[-nb], 100)
        U_left = integrate_force_to_energy(x_knots, y_knots, left_grid, spline_type)
        U_right = integrate_force_to_energy(x_knots, y_knots, right_grid, spline_type)
        x_A = float(left_grid[np.argmin(U_left)])
        x_B = float(right_grid[np.argmin(U_right)])

        Dx = 10 ** params[0]
        x0 = x_B - 0.01 * (x_B - x_A)

        kG_photons, kR_photons = binding_simulator(
                x0=x0,
                Dx=Dx,
                x_knots=x_knots,
                y_knots=y_knots,
                R0=params[num_knots+1],
                kD=params[num_knots+2],
                k_gb=params[num_knots+3],
                k_rb=params[num_knots+4],
                eta_g=params[num_knots+5],
                eta_r=params[num_knots+6],
                C_gg=params[num_knots+7],
                C_rr=params[num_knots+8],
                C_gr=params[num_knots+9],
                C_rg=params[num_knots+10],
                saving_freq=saving_freq,
                dt=dt,
                x_A=x_A,
                x_B=x_B,
                n_grid=n_grid,
                F_guide_max_factor=F_guide_max_factor,
                N_max=N_max,
                min_steps=min_steps,
                spline_type=spline_type
        )
        if kG_photons is None or kR_photons is None:
            return None

        actual_length = len(kG_photons)

        # Pad to max_bins and return (trace, length) tuple
        if max_bins is not None:
            if actual_length > max_bins:
                return None  # reject unexpectedly long trajectories
            trace = np.zeros((max_bins, 2), dtype=np.float32)
            trace[:actual_length, 0] = kG_photons
            trace[:actual_length, 1] = kR_photons
            if return_torch:
                return torch.from_numpy(trace).float(), actual_length
            return trace, actual_length

        if return_torch:
            kG_photons = torch.from_numpy(kG_photons).float()
            kR_photons = torch.from_numpy(kR_photons).float()
            return torch.stack([kG_photons, kR_photons], dim=1)

        return np.stack([kG_photons, kR_photons], axis=1).astype(np.float32)


def binding_simulator_wrapper_from_config(cfg: str):
    """
    Generate a binding simulator wrapper from a configuration file or dict.
    """
    cfg = OmegaConf.load(cfg) if isinstance(cfg, str) else cfg

    binding_cfg = cfg.simulator.get("binding", {})

    return partial(
        binding_simulator_wrapper,
        dt=cfg.simulator.time_step,
        saving_freq=cfg.simulator.saving_frequency,
        x_min=cfg.simulator.x_min,
        x_max=cfg.simulator.x_max,
        num_knots=cfg.prior.num_knots,
        max_force=cfg.simulator.max_force,
        boundary_condition=cfg.simulator.boundary_condition,
        spline_type=cfg.simulator.spline_type,
        num_boundary_knots=cfg.simulator.num_boundary_knots,
        n_grid=binding_cfg.get("n_grid", 500),
        F_guide_max_factor=binding_cfg.get("F_guide_max_factor", 0.1),
        N_max=binding_cfg.get("N_max", 200000000),
        min_steps=binding_cfg.get("min_steps", 0),
        max_bins=binding_cfg.get("max_bins", None),
        return_torch=True
    )