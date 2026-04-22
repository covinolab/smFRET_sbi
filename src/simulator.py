from functools import partial
import torch
import numpy as np
from utils import make_spline_knots, _validate_bc
from smFRET_simulator import smFRET_simulator, binding_simulator
from omegaconf import OmegaConf


def _unpack_physics_params(params, num_knots):
    """Unpack physics parameters, deriving the conserved crosstalk diagonals.

    The prior samples only the leak fractions C_gr and C_rg; photon
    conservation on each emission channel fixes C_gg = 1 − C_gr and
    C_rr = 1 − C_rg.
    """
    C_gr = float(params[num_knots + 7])
    C_rg = float(params[num_knots + 8])
    return dict(
        R0=params[num_knots + 1],
        kD=params[num_knots + 2],
        k_gb=params[num_knots + 3],
        k_rb=params[num_knots + 4],
        eta_g=params[num_knots + 5],
        eta_r=params[num_knots + 6],
        C_gg=1.0 - C_gr,
        C_rr=1.0 - C_rg,
        C_gr=C_gr,
        C_rg=C_rg,
    )


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
                boundary_condition=("linear", "linear"),
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

        phys = _unpack_physics_params(params, num_knots)
        kG_photons, kR_photons = smFRET_simulator(
                x0=x_init,
                Dx=10**params[0],
                x_knots=x_knots,
                y_knots=y_knots,
                **phys,
                T=T,
                saving_freq=saving_freq,
                dt=dt,
                spline_type=spline_type,
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
    boundary_condition = _validate_bc(cfg.simulator.boundary_condition, allow_open=False)
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
                boundary_condition=("linear", "linear"),
                spline_type="cubic",
                num_boundary_knots=3,
                L_min_bins=200,
                L_max_bins=400,
                k_persist=20,
                landmark_grid_n=400,
                N_max=200000000,
                open_left=0,
                open_right=0,
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

        phys = _unpack_physics_params(params, num_knots)
        kG_photons, kR_photons = binding_simulator(
                Dx=10 ** params[0],
                x_knots=x_knots,
                y_knots=y_knots,
                **phys,
                saving_freq=saving_freq,
                dt=dt,
                x_min=x_min,
                x_max=x_max,
                L_min_bins=L_min_bins,
                L_max_bins=L_max_bins,
                k_persist=k_persist,
                landmark_grid_n=landmark_grid_n,
                N_max=N_max,
                spline_type=spline_type,
                open_left=open_left,
                open_right=open_right,
        )
        if kG_photons is None or kR_photons is None:
            return None

        actual_length = len(kG_photons)

        # Pad to L_max_bins and return (trace, length) tuple
        trace = np.zeros((L_max_bins, 2), dtype=np.float32)
        trace[:actual_length, 0] = kG_photons
        trace[:actual_length, 1] = kR_photons
        if return_torch:
            return torch.from_numpy(trace).float(), actual_length
        return trace, actual_length


def binding_simulator_wrapper_from_config(cfg: str):
    """
    Generate a binding simulator wrapper from a configuration file or dict.
    """
    cfg = OmegaConf.load(cfg) if isinstance(cfg, str) else cfg

    binding_cfg = cfg.simulator.get("binding", {})
    left_bc, right_bc = _validate_bc(cfg.simulator.boundary_condition)

    return partial(
        binding_simulator_wrapper,
        dt=cfg.simulator.time_step,
        saving_freq=cfg.simulator.saving_frequency,
        x_min=cfg.simulator.x_min,
        x_max=cfg.simulator.x_max,
        num_knots=cfg.prior.num_knots,
        max_force=cfg.simulator.max_force,
        boundary_condition=(left_bc, right_bc),
        spline_type=cfg.simulator.spline_type,
        num_boundary_knots=cfg.simulator.num_boundary_knots,
        L_min_bins=binding_cfg.L_min_bins,
        L_max_bins=binding_cfg.L_max_bins,
        k_persist=binding_cfg.k_persist,
        landmark_grid_n=binding_cfg.get("landmark_grid_n", 400),
        N_max=binding_cfg.get("N_max", 200000000),
        open_left=int(left_bc == "open"),
        open_right=int(right_bc == "open"),
        return_torch=True
    )