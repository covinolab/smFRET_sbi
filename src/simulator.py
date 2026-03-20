from functools import partial
import torch
import numpy as np
from utils import make_spline_knots
from smFRET_simulator import smFRET_simulator
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
                max_G0=10.0,
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
            max_G0=max_G0,
            boundary_condition=boundary_condition,  
            num_boundary_knots=num_boundary_knots
        )

        kG_photons, kR_photons = smFRET_simulator(
                x0=x_init,
                Dx=10**params[0],
                x_knots=x_knots,
                y_knots=y_knots,
                R0=params[1+num_knots],
                kD=params[2+num_knots],
                k_gb=params[3+num_knots],
                k_rb=params[4+num_knots],
                eta_g=params[5+num_knots],
                eta_r=params[6+num_knots],
                C_gg=params[7+num_knots],
                C_rr=params[8+num_knots],
                C_gr=params[9+num_knots],
                C_rg=params[10+num_knots],
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
    max_G0 = cfg.simulator.max_G0
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
        max_G0=max_G0,
        boundary_condition=boundary_condition,
        spline_type=spline_type,
        num_boundary_knots=num_boundary_knots,
        return_torch=True
    )