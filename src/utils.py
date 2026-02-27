import numpy as np
import torch
from gsl_spline import spline

def make_spline_knots(
        params,
        x_min=0.0, 
        x_max=10.0, 
        num_knots=10, 
        max_G0=10.0, 
        boundary_condition='bound'
    ):
    
    knot_spacing = (x_max - x_min) / (num_knots - 1)
    x_knots = np.linspace(x_min - 2 * knot_spacing, x_max + 2 * knot_spacing, num_knots + 4) # +4 for boundary nodes
    y_knots = np.zeros(num_knots + 4) # +4 for boundary nodes

    if boundary_condition == 'diffusive':
        y_knots[-1] = 0.0
        y_knots[-2] = 0.0

    elif boundary_condition == 'bound':
        y_knots[-1] = 2 * max_G0 + params[num_knots]
        y_knots[-2] = max_G0 + params[num_knots]
    
    y_knots[0] = 2 * max_G0 + params[1]
    y_knots[1] = max_G0 + params[1]  

    y_knots[2:-2] = params[1:1+num_knots]
    return x_knots, y_knots


def eval_spline(
        params,
        x_eval,
        N_knots, 
        x_min, 
        x_max, 
        boundary_condition, 
        max_G0, 
        spline_type="cubic"
    ):
    
    x_knots, y_knots = make_spline_knots(
        params=params,
        x_min=x_min,
        x_max=x_max,
        num_knots=N_knots,
        max_G0=max_G0,
        boundary_condition=boundary_condition
    )

    y = spline(x_knots, y_knots, x_eval, spline_type=spline_type)

    return x_eval, y