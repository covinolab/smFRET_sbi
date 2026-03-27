from typing import Union, List
from omegaconf import OmegaConf
import torch
import torch.distributions as dists
from sbi.utils.user_input_checks import MultipleIndependent


class FreeEnergyPrior(torch.distributions.Distribution):
    """
    Uniform prior over the n-1 increments of a free energy profile.

    The parameters ARE the increments Δᵢ = φᵢ₊₁ − φᵢ (i = 0 … n-2).
    The n knot values are not stored in the parameter vector — they are
    reconstructed inside the simulator via cumsum + zero-mean centring.

    This makes the prior shift-invariant by construction and keeps
    sample / log_prob trivially consistent: each increment is drawn from
    an independent Uniform box and log_prob is the sum of their log densities.

    Rough shape encoding
    --------------------
    Pass a length-n knot-height template as ``center``.  Its first differences
    become the midpoints of the per-increment uniform boxes:
        center_diff_i = center[i+1] − center[i]
    ``center=None`` → all midpoints zero → symmetric, shape-agnostic prior.

    Config mapping
    --------------
    num_knots  →  n   (number of physical knots; prior has n-1 parameters)
    scale      →  half-width of each increment's uniform box
    center     →  optional knot-height template (null = flat)

    Parameters
    ----------
    num_knots : int
        Number of physical knots n.  The distribution has event_shape=(n-1,).
    scale : float
        Half-width of each increment's uniform box.
    center : Tensor (n,) or None
        Shape template as knot heights; internally converted to n-1 increment
        centres.  None → zeros (no shape preference).
    device : str
    """

    arg_constraints = {}
    has_rsample = True

    def __init__(
        self,
        num_knots: int,
        scale: float,
        center: Union[torch.Tensor, List[float], None] = None,
        device: str = "cpu",
    ):
        self.num_knots = num_knots
        self.scale = float(scale)

        if isinstance(center, torch.Tensor):
            device_ = center.device
        else:
            device_ = torch.device(device)

        if center is not None:
            center = torch.as_tensor(center, dtype=torch.float32, device=device_)
            if center.shape != torch.Size([num_knots]):
                raise ValueError(
                    f"center must have shape ({num_knots},), got {tuple(center.shape)}."
                )
            center = center - center.mean()
            center_diffs = center[1:] - center[:-1]
        else:
            center_diffs = torch.zeros(num_knots - 1, device=device_)

        self.center = center

        self._base = dists.Independent(
            dists.Uniform(center_diffs - self.scale, center_diffs + self.scale),
            reinterpreted_batch_ndims=1,
        )
        super().__init__(
            batch_shape=torch.Size([]),
            event_shape=torch.Size([num_knots - 1]),
        )

    def rsample(self, sample_shape=torch.Size()) -> torch.Tensor:
        return self._base.rsample(sample_shape)

    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:
        with torch.no_grad():
            return self._base.sample(sample_shape)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return self._base.log_prob(value)

    @property
    def support(self):
        return dists.constraints.independent(dists.constraints.real, 1)
    
    def to(self, device):
        return self.__class__(
            num_knots=self.num_knots,
            scale=self.scale,
            center=self.center.to(device) if self.center is not None else None,
            device=device,
        )


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
    Build the full FRET prior: logD + free-energy increments + photophysics.

    Parameter order: [logD, Δ₁…Δₙ₋₁, R0, kD, k_gb, k_rb, eta_g, eta_r,
                      C_gg, C_rr, C_gr, C_rg]
    Total length: num_knots + 10  (one fewer than before)

    Args:
        num_knots: Number of physical spline knots n; prior holds n-1 increments.
        logD_lims … C_rg_lims: (low, high) bounds for each parameter.
        center: Optional shape template of length n (knot heights).
            Its first differences set the midpoints of the increment uniform boxes.
            None → flat, symmetric prior.
        scale: Half-width per increment.
        device: Torch device.

    Examples
    --------
    Flat prior, all increments in [-10, 10]:
        >>> prior = generate_spline_fret_prior(num_knots=5, scale=10.0)

    Prior expecting a central barrier (~4 kT rise then fall):
        >>> center = torch.tensor([-2., 0., 4., 0., -2.])
        >>> prior = generate_spline_fret_prior(num_knots=5, scale=3.0, center=center)
    """
    prior_logD = dists.Uniform(
        torch.tensor([logD_lims[0]], device=device),
        torch.tensor([logD_lims[1]], device=device),
    )
    prior_spline = FreeEnergyPrior(num_knots=num_knots, scale=scale, center=center, device=device)

    phys_lims = [
        R0_lims, kD_lims, k_gb_lims, k_rb_lims,
        eta_g_lims, eta_r_lims, C_gg_lims, C_rr_lims, C_gr_lims, C_rg_lims,
    ]
    phys_priors = [
        dists.Uniform(torch.tensor([lo], device=device), torch.tensor([hi], device=device))
        for lo, hi in phys_lims
    ]

    return MultipleIndependent([prior_logD, prior_spline, *phys_priors], device=device)


def generate_fret_prior_from_config(cfg: OmegaConf, device: str = "cpu") -> MultipleIndependent:
    """
    Build the full FRET prior from an OmegaConf config object.

    Relevant config keys under ``prior:``:
        num_knots, scale, center (null = flat), logD_lims … C_rg_lims.
    """
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
    Reduced prior (logD + free-energy increments only) for posterior sampling.

    Returns MultipleIndependent([prior_logD, prior_spline]).
    """
    cfg = OmegaConf.load(cfg) if isinstance(cfg, str) else cfg

    prior_logD = dists.Uniform(
        torch.tensor([cfg.prior.logD_lims[0]], device=device),
        torch.tensor([cfg.prior.logD_lims[1]], device=device),
    )

    center = None
    if cfg.prior.center is not None:
        center = torch.as_tensor(cfg.prior.center, dtype=torch.float32, device=device)
        center = center - center.mean()

    prior_spline = FreeEnergyPrior(
        num_knots=cfg.prior.num_knots, scale=cfg.prior.scale, center=center, device=device
    )

    return MultipleIndependent([prior_logD, prior_spline], device=device)
