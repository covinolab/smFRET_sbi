from pathlib import Path
from typing import Union, List
from omegaconf import OmegaConf
import torch
import torch.distributions as dists
from sbi import utils as utils
from sbi.utils.user_input_checks import MultipleIndependent


def make_covariance_matrix(n: int, scale: float, correlation: float, device: str = "cpu") -> torch.Tensor:
    """
    Create a covariance matrix with exponential decay correlation.
    
    Args:
        n: Size of the covariance matrix (n x n).
        scale: Scaling factor for the covariance values.
        correlation: Correlation factor determining the decay of covariance.
        device: Device to store the tensor.
    
    Returns:
        Covariance matrix of size (n x n).
    """
    i, j = torch.meshgrid(torch.arange(n, device=device), torch.arange(n, device=device), indexing='ij')
    cov = scale * torch.exp(-correlation * torch.abs(i - j).float())
    return cov


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
    spline_mean: Union[torch.Tensor, List[float], None] = None,
    spline_scale: float = 1.0,
    spline_correlation: float = 0.5,
    device: str = "cpu"
) -> Union[MultipleIndependent]:
    """
    Generate a prior for FRET with spline-based potential.
    
    Parameter order: [logD, spline_params..., R0, kD, k_gb, k_rb, eta_g, eta_r, C_gg, C_rr, C_gr, C_rg]
    
    This replaces the single deltaG parameter with num_knots spline parameters
    that define the potential landscape via a correlated multivariate normal.
    
    Args:
        num_knots: Number of spline knots (determines spline parameter dimension).
        logD_lims: (low, high) bounds for log10(D).
        R0_lims: (low, high) bounds for Förster radius.
        kD_lims: (low, high) bounds for donor excitation rate.
        k_gb_lims: (low, high) bounds for green background rate.
        k_rb_lims: (low, high) bounds for red background rate.
        eta_g_lims: (low, high) bounds for green detector efficiency.
        eta_r_lims: (low, high) bounds for red detector efficiency.
        C_gg_lims: (low, high) bounds for green->green cross-talk.
        C_rr_lims: (low, high) bounds for red->red cross-talk.
        C_gr_lims: (low, high) bounds for green->red cross-talk.
        C_rg_lims: (low, high) bounds for red->green cross-talk.
        spline_scale: Scale of the spline covariance matrix.
        spline_correlation: Correlation decay factor for spline covariance.
        normalize_splines: If True, normalize spline params to zero mean when sampling.
        device: Device for tensors.
    
    Returns:
        Prior distribution (SplineFRETPrior if normalize_splines=True, else MultipleIndependent).
    
    Example:
        >>> prior = generate_spline_fret_prior(num_knots=10)
        >>> theta = prior.sample((1000,))  # Shape: (1000, 21)
    """
    # Prior on log10(D)
    prior_logD = dists.Uniform(
        torch.tensor([logD_lims[0]], device=device),
        torch.tensor([logD_lims[1]], device=device)
    )
    
    # Prior on spline parameters (correlated Gaussian)
    if spline_mean is None:
        spline_mean = torch.zeros(num_knots, device=device)

    spline_cov = make_covariance_matrix(num_knots, spline_scale, spline_correlation, device)
    spline_cov = spline_cov.to(device)
    spline_mean = spline_mean.to(device)
    prior_spline = dists.MultivariateNormal(loc=spline_mean, covariance_matrix=spline_cov)
    
    # Photophysics priors (uniform)
    prior_R0 = dists.Uniform(
        torch.tensor([R0_lims[0]], device=device),
        torch.tensor([R0_lims[1]], device=device)
    )
    prior_kD = dists.Uniform(
        torch.tensor([kD_lims[0]], device=device),
        torch.tensor([kD_lims[1]], device=device)
    )
    prior_k_gb = dists.Uniform(
        torch.tensor([k_gb_lims[0]], device=device),
        torch.tensor([k_gb_lims[1]], device=device)
    )
    prior_k_rb = dists.Uniform(
        torch.tensor([k_rb_lims[0]], device=device),
        torch.tensor([k_rb_lims[1]], device=device)
    )
    prior_eta_g = dists.Uniform(
        torch.tensor([eta_g_lims[0]], device=device),
        torch.tensor([eta_g_lims[1]], device=device)
    )
    prior_eta_r = dists.Uniform(
        torch.tensor([eta_r_lims[0]], device=device),
        torch.tensor([eta_r_lims[1]], device=device)
    )
    prior_C_gg = dists.Uniform(
        torch.tensor([C_gg_lims[0]], device=device),
        torch.tensor([C_gg_lims[1]], device=device)
    )
    prior_C_rr = dists.Uniform(
        torch.tensor([C_rr_lims[0]], device=device),
        torch.tensor([C_rr_lims[1]], device=device)
    )
    prior_C_gr = dists.Uniform(
        torch.tensor([C_gr_lims[0]], device=device),
        torch.tensor([C_gr_lims[1]], device=device)
    )
    prior_C_rg = dists.Uniform(
        torch.tensor([C_rg_lims[0]], device=device),
        torch.tensor([C_rg_lims[1]], device=device)
    )
    
    priors = [
        prior_logD,
        prior_spline,
        prior_R0,
        prior_kD,
        prior_k_gb,
        prior_k_rb,
        prior_eta_g,
        prior_eta_r,
        prior_C_gg,
        prior_C_rr,
        prior_C_gr,
        prior_C_rg,
    ]

    return MultipleIndependent(priors)


def generate_fret_prior_from_config(
    cfg: OmegaConf, 
    device: str = "cpu"
) -> Union[MultipleIndependent]:
    """
    Generate a spline FRET prior from a configuration file or dict.
    
    Expected config structure:
        prior:
            num_knots: 10
            logD_lims: [-2.0, 2.0]
            R0_lims: [3.0, 8.0]
            kD_lims: [0.01, 1.0]
            k_gb_lims: [0.0, 0.1]
            k_rb_lims: [0.0, 0.1]
            eta_g_lims: [0.5, 1.0]
            eta_r_lims: [0.5, 1.0]
            C_gg_lims: [0.8, 1.0]
            C_rr_lims: [0.8, 1.0]
            C_gr_lims: [0.0, 0.2]
            C_rg_lims: [0.0, 0.2]
            spline_scale: 1.0
            spline_correlation: 0.5
            normalize_splines: true
    
    Args:
        config: Path to YAML config or dict with 'prior' section.
        device: Device for tensors.
    
    Returns:
        SplineFRETPrior or MultipleIndependent distribution.
    """

    if cfg.prior.spline_mean is not None:
        spline_mean = torch.tensor(cfg.prior.spline_mean, device=device)
        spline_mean = spline_mean - spline_mean.mean()  # Normalize to zero mean
    else:
        spline_mean = cfg.prior.spline_mean  # This will be None, and the prior function will default to zero mean

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
        spline_mean=spline_mean,
        spline_scale=cfg.prior.spline_scale,
        spline_correlation=cfg.prior.spline_correlation,
        device=device
    )


def generate_posterior_prior_from_config(cfg: str, device: str = "cpu") -> MultipleIndependent:
    """
    Generate a prior for posterior sampling from config.
    
    This is a wrapper around generate_fret_prior_from_config that ensures the returned
    distribution is always a MultipleIndependent, which is required for some inference algorithms.
    
    Args:
        cfg: Config with 'prior' section defining the FRET prior parameters.
        device: Device for tensors.

    Returns:
        MultipleIndependent distribution representing the FRET prior.
    """

    cfg = OmegaConf.load(cfg)

    prior_logD = dists.Uniform(
        torch.tensor([cfg.prior.logD_lims[0]], device=device),
        torch.tensor([cfg.prior.logD_lims[1]], device=device)
    )

    # Prior on spline parameters (correlated Gaussian)
    if cfg.prior.spline_mean is not None:
        spline_mean = torch.tensor(cfg.prior.spline_mean)
        spline_mean = spline_mean - spline_mean.mean()  # Normalize to zero mean
    else:
        spline_mean = torch.zeros(cfg.prior.num_knots)

    spline_cov = make_covariance_matrix(cfg.prior.num_knots, cfg.prior.spline_scale, cfg.prior.spline_correlation, device)
    prior_spline = dists.MultivariateNormal(loc=spline_mean.to(device), covariance_matrix=spline_cov.to(device))

    return MultipleIndependent([prior_logD, prior_spline], device=device)
