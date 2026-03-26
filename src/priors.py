from typing import Union, List
from omegaconf import OmegaConf
import torch
import torch.distributions as dists
from sbi.utils.user_input_checks import MultipleIndependent


def make_igmrf_covariance_matrix(n: int, roughness_scale: float, order: int = 1, device: str = "cpu") -> torch.Tensor:
    """
    Create a covariance matrix for an Intrinsic Gaussian Markov Random Field (IGMRF).

    Uses a finite-difference precision matrix K = D'D (where D is the order-th difference
    operator), regularized by a small diagonal to make it numerically proper. The resulting
    prior regularizes roughness (differences between adjacent knots) rather than absolute
    knot heights, and is naturally shift-invariant.

    Args:
        n: Number of knots.
        roughness_scale: Scales the covariance (larger = more variation allowed).
        order: Order of the difference operator (1 = first differences, 2 = curvature penalty).
        device: Device to store the tensor.

    Returns:
        Covariance matrix of size (n x n).
    """
    D = torch.zeros(n - 1, n, device=device)
    for i in range(n - 1):
        D[i, i] = -1.0
        D[i, i + 1] = 1.0

    if order == 2:
        if n < 3:
            raise ValueError("Need at least 3 knots for order-2 IGMRF.")
        D2 = torch.zeros(n - 2, n - 1, device=device)
        for i in range(n - 2):
            D2[i, i] = -1.0
            D2[i, i + 1] = 1.0
        D = D2 @ D  # shape (n-2, n)
    elif order != 1:
        raise ValueError("order must be 1 or 2.")

    K = D.T @ D  # Precision matrix, shape (n, n), rank-deficient (shift-invariant)
    K_reg = K + 1e-4 * torch.eye(n, device=device)
    return roughness_scale * torch.linalg.inv(K_reg)


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
    return scale * torch.exp(-correlation * torch.abs(i - j).float())


def _make_spline_prior(
    num_knots: int,
    spline_scale: float,
    prior_type: str,
    igmrf_order: int = 1,
    spline_mean: Union[torch.Tensor, None] = None,
    spline_correlation: float = 0.5,
    device: str = "cpu",
) -> dists.Distribution:
    """
    Build the spline-knot prior distribution.

    Args:
        num_knots: Number of spline knots.
        spline_scale: Scale / half-range for the knot prior.
        prior_type: "uniform" — independent Uniform(-scale, +scale) per knot;
                    "igmrf"   — first/second-difference IGMRF (shift-invariant);
                    "gp"      — exponential-decay GP (original).
        igmrf_order: Difference order for IGMRF (1 or 2). Ignored for other types.
        spline_mean: Mean vector for GP / IGMRF priors. Ignored for "uniform".
        spline_correlation: Decay factor for GP covariance. Ignored for other types.
        device: Torch device.

    Returns:
        Distribution with event_shape = (num_knots,).
    """
    if prior_type == "uniform":
        return dists.Independent(
            dists.Uniform(
                torch.full((num_knots,), -spline_scale, device=device),
                torch.full((num_knots,), spline_scale, device=device),
            ),
            reinterpreted_batch_ndims=1,
        )

    if spline_mean is None:
        spline_mean = torch.zeros(num_knots, device=device)
    spline_mean = spline_mean.to(device)

    if prior_type == "igmrf":
        cov = make_igmrf_covariance_matrix(num_knots, spline_scale, igmrf_order, device)
    elif prior_type == "gp":
        cov = make_covariance_matrix(num_knots, spline_scale, spline_correlation, device)
    else:
        raise ValueError(f"prior_type must be 'uniform', 'igmrf', or 'gp', got '{prior_type}'.")

    return dists.MultivariateNormal(loc=spline_mean, covariance_matrix=cov.to(device))


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
    prior_type: str = "uniform",
    igmrf_order: int = 1,
    device: str = "cpu",
) -> MultipleIndependent:
    """
    Generate a prior for FRET with spline-based potential.

    Parameter order: [logD, spline_params..., R0, kD, k_gb, k_rb, eta_g, eta_r, C_gg, C_rr, C_gr, C_rg]

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
        spline_mean: Mean for GP/IGMRF knot prior. Ignored for prior_type="uniform".
        spline_scale: Scale for the knot prior (covariance scale for GP/IGMRF,
            half-range [-scale, +scale] for uniform).
        spline_correlation: Decay factor for GP covariance. Ignored for other types.
        prior_type: "uniform" — independent Uniform(-scale, +scale) per knot (default);
                    "igmrf"   — first/second-difference IGMRF (shift-invariant MVN);
                    "gp"      — exponential-decay GP (original MVN).
        igmrf_order: Difference order for IGMRF (1 or 2). Ignored for other types.
        device: Device for tensors.

    Returns:
        MultipleIndependent prior distribution.

    Example:
        >>> prior = generate_spline_fret_prior(num_knots=5)
        >>> theta = prior.sample((1000,))  # Shape: (1000, 16)
    """
    prior_logD = dists.Uniform(
        torch.tensor([logD_lims[0]], device=device),
        torch.tensor([logD_lims[1]], device=device),
    )

    prior_spline = _make_spline_prior(
        num_knots, spline_scale, prior_type, igmrf_order, spline_mean, spline_correlation, device
    )

    phys_lims = [R0_lims, kD_lims, k_gb_lims, k_rb_lims, eta_g_lims, eta_r_lims,
                 C_gg_lims, C_rr_lims, C_gr_lims, C_rg_lims]
    phys_priors = [
        dists.Uniform(torch.tensor([lo], device=device), torch.tensor([hi], device=device))
        for lo, hi in phys_lims
    ]

    return MultipleIndependent([prior_logD, prior_spline, *phys_priors])


def generate_fret_prior_from_config(
    cfg: OmegaConf,
    device: str = "cpu",
) -> MultipleIndependent:
    """
    Generate a spline FRET prior from an OmegaConf config object.

    Expected config structure:
        prior:
            num_knots: 5
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
            spline_scale: 10.0
            spline_correlation: 1.0  # only used for prior_type=gp
            prior_type: uniform      # "uniform", "igmrf", or "gp"
            igmrf_order: 1           # only used for prior_type=igmrf

    Args:
        cfg: OmegaConf config with 'prior' section.
        device: Device for tensors.

    Returns:
        MultipleIndependent prior distribution.
    """
    spline_mean = None
    if cfg.prior.spline_mean is not None:
        spline_mean = torch.tensor(cfg.prior.spline_mean, device=device)
        spline_mean = spline_mean - spline_mean.mean()

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
        spline_correlation=OmegaConf.select(cfg, "prior.spline_correlation", default=0.5),
        prior_type=OmegaConf.select(cfg, "prior.prior_type", default="uniform"),
        igmrf_order=OmegaConf.select(cfg, "prior.igmrf_order", default=1),
        device=device,
    )


def generate_posterior_prior_from_config(cfg: str, device: str = "cpu") -> MultipleIndependent:
    """
    Generate a reduced prior (logD + spline only) for posterior sampling from a config file path.

    Args:
        cfg: Path to YAML config file with 'prior' section.
        device: Device for tensors.

    Returns:
        MultipleIndependent([prior_logD, prior_spline]).
    """
    cfg = OmegaConf.load(cfg)

    prior_logD = dists.Uniform(
        torch.tensor([cfg.prior.logD_lims[0]], device=device),
        torch.tensor([cfg.prior.logD_lims[1]], device=device),
    )

    spline_mean = None
    if cfg.prior.spline_mean is not None:
        spline_mean = torch.tensor(cfg.prior.spline_mean)
        spline_mean = spline_mean - spline_mean.mean()

    prior_spline = _make_spline_prior(
        num_knots=cfg.prior.num_knots,
        spline_scale=cfg.prior.spline_scale,
        prior_type=OmegaConf.select(cfg, "prior.prior_type", default="uniform"),
        igmrf_order=OmegaConf.select(cfg, "prior.igmrf_order", default=1),
        spline_mean=spline_mean,
        spline_correlation=OmegaConf.select(cfg, "prior.spline_correlation", default=0.5),
        device=device,
    )

    return MultipleIndependent([prior_logD, prior_spline], device=device)
