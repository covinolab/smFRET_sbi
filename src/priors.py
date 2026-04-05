from typing import Union, List

from sklearn.preprocessing import scale
from omegaconf import OmegaConf
import torch
import torch.distributions as dists
from sbi.utils.user_input_checks import MultipleIndependent


def generate_spline_prior(
        centers: Union[torch.Tensor, None] = None, 
        scale: float = 1.0, 
        num_knots: int = 10, 
        device: str = "cpu"
    ) -> List[dists.Uniform]:

    if centers is None:
        centers = torch.zeros(num_knots)
    
    prior_spline = [
        dists.Uniform(
            torch.tensor(
                [
                    centers[i] - scale], device=device), torch.tensor([centers[i] + scale], device=device)) for i in range(num_knots)
                ]
    return prior_spline

def generate_spline_fret_prior(
    num_knots: int,
    logD_lims: tuple = (-2.0, 2.0),
    R0_lims: tuple = (3.0, 8.0),
    kD_lims: tuple = (0.01, 1.0),
    k_gb_lims: tuple = (0.0, 0.1),
    k_rb_lims: tuple = (0.0, 0.1),
    eta_g_lims: tuple = (0.5, 1.0),
    eta_r_lims: tuple = (0.5, 1.0),
    C_gg_lims: tuple = (0.8, 1.0),
    C_rr_lims: tuple = (0.8, 1.0),
    C_gr_lims: tuple = (0.0, 0.2),
    C_rg_lims: tuple = (0.0, 0.2),
    center: Union[torch.Tensor, List[float], None] = None,
    scale: float = 1.0,
    device: str = "cpu",
) -> MultipleIndependent:
    """
    Build the full FRET prior: logD + gradient knots + photophysics.

    Parameter order: [logD, g₀…gₙ₋₁, R0, kD, k_gb, k_rb, eta_g, eta_r,
                      C_gg, C_rr, C_gr, C_rg]
    Total length: num_knots + 11

    Args:
        num_knots: Number of spline knots n; prior holds n gradient knot values.
        logD_lims … C_rg_lims: (low, high) bounds for each parameter.
        center: Optional force template of length n (knot force values).
            None → flat, symmetric prior (zero mean force).
        scale: Half-width per knot's uniform box.
        device: Torch device.

    Examples
    --------
    Flat prior, all force knots in [-10, 10]:
        >>> prior = generate_spline_fret_prior(num_knots=5, scale=10.0)

    Prior expecting a central restoring force (harmonic-like):
        >>> center = torch.tensor([4., 2., 0., -2., -4.])
        >>> prior = generate_spline_fret_prior(num_knots=5, scale=3.0, center=center)
    """
    prior_logD = dists.Uniform(
        torch.tensor([logD_lims[0]], device=device),
        torch.tensor([logD_lims[1]], device=device),
    )
    
    if center is not None:
        center = torch.as_tensor(center, dtype=torch.float32, device=device)
    
    prior_spline = generate_spline_prior(centers=center, scale=scale, num_knots=num_knots, device=device)

    phys_lims = [
        R0_lims, kD_lims, k_gb_lims, k_rb_lims,
        eta_g_lims, eta_r_lims, C_gg_lims, C_rr_lims, C_gr_lims, C_rg_lims,
    ]
    phys_priors = [
        dists.Uniform(torch.tensor([lo], device=device), torch.tensor([hi], device=device))
        for lo, hi in phys_lims
    ]

    return MultipleIndependent([prior_logD, *prior_spline, *phys_priors], device=device)


def generate_fret_prior_from_config(cfg, device: str = "cpu") -> MultipleIndependent:
    """
    Build the full FRET prior from an OmegaConf config object.

    Relevant config keys under ``prior:``:
        num_knots, scale, center (null = flat), logD_lims … C_rg_lims.
    """
    cfg = OmegaConf.load(cfg) if isinstance(cfg, str) else cfg

    center = None
    if cfg.prior.center is not None:
        center = list(cfg.prior.center)

    return generate_spline_fret_prior(
        num_knots=cfg.prior.num_knots,
        logD_lims=tuple(cfg.prior.logD_lims),
        R0_lims=tuple(cfg.prior.R0_lims),
        kD_lims=tuple(cfg.prior.kD_lims),
        k_gb_lims=tuple(cfg.prior.k_gb_lims),
        k_rb_lims=tuple(cfg.prior.k_rb_lims),
        eta_g_lims=tuple(cfg.prior.eta_g_lims),
        eta_r_lims=tuple(cfg.prior.eta_r_lims),
        C_gg_lims=tuple(cfg.prior.C_gg_lims),
        C_rr_lims=tuple(cfg.prior.C_rr_lims),
        C_gr_lims=tuple(cfg.prior.C_gr_lims),
        C_rg_lims=tuple(cfg.prior.C_rg_lims),
        center=center,
        scale=cfg.prior.scale,
        device=device,
    )


def generate_posterior_prior_from_config(cfg, device: str = "cpu") -> MultipleIndependent:
    """
    Reduced prior (logD + gradient knots only) for posterior sampling.

    Returns MultipleIndependent([prior_logD, prior_spline]).
    """
    cfg = OmegaConf.load(cfg) if isinstance(cfg, str) else cfg

    scale = cfg.prior.scale
    center = cfg.prior.center if cfg.prior.center is not None else None
    num_knots = cfg.prior.num_knots

    prior_logD = dists.Uniform(
        torch.tensor([cfg.prior.logD_lims[0]], device=device),
        torch.tensor([cfg.prior.logD_lims[1]], device=device),
    )

    prior_spline = generate_spline_prior(centers=center, scale=scale, num_knots=num_knots, device=device)

    return MultipleIndependent([prior_logD, *prior_spline], device=device)
