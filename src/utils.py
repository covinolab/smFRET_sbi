from functools import partial
import numpy as np
import torch
from omegaconf import OmegaConf
from gsl_spline import spline

def make_spline_knots(
        params,
        x_min=0.0, 
        x_max=10.0, 
        num_knots=10, 
        max_G0=10.0, 
        num_boundary_knots=3,
        boundary_condition='bound'
    ):

    knot_spacing = (x_max - x_min) / (num_knots - 1)
    x_knots = np.linspace(x_min - num_boundary_knots * knot_spacing, x_max + num_boundary_knots * knot_spacing, num_knots + 2 * num_boundary_knots)
    y_knots = np.zeros(num_knots + 2 * num_boundary_knots) # +4 for boundary nodes
    for i in range(1, num_boundary_knots+1):
        if boundary_condition == 'linear':
            y_knots[-num_boundary_knots + i] = i * (max_G0 / num_boundary_knots) + params[num_knots]
            y_knots[num_boundary_knots - i - 1 ] = i * (max_G0 / num_boundary_knots) + params[1]
        
        elif boundary_condition == 'quadratic':
            y_knots[-num_boundary_knots + i - 1] = (i / num_boundary_knots)**2 * max_G0 + params[num_knots]
            y_knots[num_boundary_knots - i] =  (i / num_boundary_knots)**2 * max_G0 + params[1]

    y_knots[num_boundary_knots:-num_boundary_knots] = params[1:1+num_knots]
    return x_knots, y_knots


def eval_spline(
        params,
        x_eval,
        N_knots, 
        x_min, 
        x_max, 
        boundary_condition, 
        max_G0, 
        spline_type="cubic",
        num_boundary_knots=4
    ):

    if isinstance(params, torch.Tensor):
        params = params.detach().cpu().numpy()
    
    x_knots, y_knots = make_spline_knots(
        params=params,
        x_min=x_min,
        x_max=x_max,
        num_knots=N_knots,
        max_G0=max_G0,
        boundary_condition=boundary_condition,
        num_boundary_knots=num_boundary_knots
    )

    y = spline(x_knots, y_knots, x_eval, spline_type=spline_type)

    return x_eval, y


def eval_spline_from_config(cfg):
    cfg = OmegaConf.load(cfg) if isinstance(cfg, str) else cfg
    return partial(eval_spline,
        N_knots=cfg.prior.num_knots,
        x_min=cfg.simulator.x_min,
        x_max=cfg.simulator.x_max,
        boundary_condition=cfg.simulator.boundary_condition,
        max_G0=cfg.simulator.max_G0,
        spline_type=cfg.simulator.spline_type,
        num_boundary_knots=cfg.simulator.num_boundary_knots
    )