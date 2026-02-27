import torch
import numpy as np
from utils import make_spline_knots
from smFRET_simulator import smFRET_simulator

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
                spline_type="cubic"
    ):
            
        x_init = np.random.uniform(low=x_init[0], high=x_init[1])
        x_knots, y_knots = make_spline_knots(
            params=params,
            x_min=x_min,
            x_max=x_max,
            num_knots=num_knots,
            max_G0=max_G0,
            boundary_condition=boundary_condition
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

        kG_photons = torch.from_numpy(kG_photons).float()
        kR_photons = torch.from_numpy(kR_photons).float()
        return torch.stack([kG_photons, kR_photons], dim=1)