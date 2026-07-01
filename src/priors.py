from typing import Union, List, Sequence

from sklearn.preprocessing import scale
from omegaconf import OmegaConf
import torch
import torch.distributions as dists
from sbi.utils.user_input_checks import MultipleIndependent


PHYS_PARAM_NAMES = ["R0", "kD", "k_gb", "k_rb", "eta_g", "eta_r", "C_gr", "C_rg"]
DEFAULT_INFERRED = ["logD", "knots", "R0"]
_VALID_TOKENS = ("logD", "knots", *PHYS_PARAM_NAMES)


def _canonical_indices(num_knots: int) -> dict:
    """Token -> list of canonical-theta positions in the 16-dim layout.

    Canonical layout (length num_knots + 9):
        [logD, g_0..g_{n-1}, R0, kD, k_gb, k_rb, eta_g, eta_r, C_gr, C_rg]
    """
    out = {"logD": [0], "knots": list(range(1, num_knots + 1))}
    for i, name in enumerate(PHYS_PARAM_NAMES):
        out[name] = [num_knots + 1 + i]
    return out


def _resolve_inferred_tokens(cfg) -> List[str]:
    raw = cfg.prior.get("inferred_params", DEFAULT_INFERRED)
    tokens = list(raw) if raw is not None else list(DEFAULT_INFERRED)
    if not tokens:
        raise ValueError(
            "inferred_params is empty. At least one token is required."
        )
    seen = set()
    for tok in tokens:
        if tok not in _VALID_TOKENS:
            raise ValueError(
                f"Unknown inferred_params token: {tok!r}. "
                f"Valid: {list(_VALID_TOKENS)}"
            )
        if tok in seen:
            raise ValueError(f"Duplicate inferred_params token: {tok!r}")
        seen.add(tok)
    canonical_order = {tok: i for i, tok in enumerate(("logD", "knots", *PHYS_PARAM_NAMES))}
    return sorted(tokens, key=canonical_order.__getitem__)


def _build_dists_for_token(cfg, token, num_knots, device, center, scale_):
    if token == "logD":
        return [dists.Uniform(
            torch.tensor([cfg.prior.logD_lims[0]], device=device),
            torch.tensor([cfg.prior.logD_lims[1]], device=device),
        )]
    if token == "knots":
        return _knot_prior_dists(cfg, center, scale_, num_knots, device)
    lo, hi = tuple(cfg.prior[f"{token}_lims"])
    return [dists.Uniform(
        torch.tensor([lo], device=device),
        torch.tensor([hi], device=device),
    )]


def generate_split_priors_from_config(cfg, device: str = "cpu"):
    """Return (inferred_prior, nuisance_prior, inferred_idx, nuisance_idx).

    Honors the optional ``prior.inferred_params`` config key, which lists
    the canonical parameter tokens to put in the inference posterior
    (default: ``['logD', 'knots', 'R0']``). Anything not listed becomes a
    per-trace nuisance.

    Valid tokens: 'logD', 'knots', 'R0', 'kD', 'k_gb', 'k_rb', 'eta_g',
    'eta_r', 'C_gr', 'C_rg'. Knots are all-or-nothing; the 'knots' token
    expands to all g_0..g_{n-1}. The user-supplied token order is sorted
    into canonical order internally so theta dim ordering is predictable.

    The two LongTensors give canonical theta positions in the same order
    as the components appear inside their respective MultipleIndependent
    priors. Pair with ``assemble_canonical_theta`` to scatter samples
    back into a canonical 16-dim theta the simulator wrapper consumes.

    Under the default split this returns priors that are component-for-
    component identical to ``generate_posterior_prior_from_config`` and
    ``generate_nuisance_prior_from_config``.
    """
    cfg = OmegaConf.load(cfg) if isinstance(cfg, str) else cfg
    num_knots = cfg.prior.num_knots
    center = list(cfg.prior.center) if cfg.prior.center is not None else None
    scale_ = cfg.prior.scale

    inferred_tokens = _resolve_inferred_tokens(cfg)
    canonical = _canonical_indices(num_knots)

    inferred_dists, inferred_idx = [], []
    for tok in inferred_tokens:
        inferred_dists.extend(
            _build_dists_for_token(cfg, tok, num_knots, device, center, scale_)
        )
        inferred_idx.extend(canonical[tok])

    inferred_set = set(inferred_tokens)
    nuisance_dists, nuisance_idx = [], []
    for tok in ("logD", "knots", *PHYS_PARAM_NAMES):
        if tok not in inferred_set:
            nuisance_dists.extend(
                _build_dists_for_token(cfg, tok, num_knots, device, center, scale_)
            )
            nuisance_idx.extend(canonical[tok])

    if not nuisance_dists:
        raise ValueError(
            "Empty nuisance set is not supported. Leave at least one "
            "parameter out of inferred_params."
        )

    return (
        MultipleIndependent(inferred_dists, device=device),
        MultipleIndependent(nuisance_dists, device=device),
        torch.tensor(inferred_idx, dtype=torch.long),
        torch.tensor(nuisance_idx, dtype=torch.long),
    )


def assemble_canonical_theta(
    inferred: torch.Tensor,
    nuisance: torch.Tensor,
    inferred_idx: torch.Tensor,
    nuisance_idx: torch.Tensor,
) -> torch.Tensor:
    """Scatter (inferred, nuisance) into a single canonical-order theta.

    Under the default split (inferred_idx = [0..num_knots+1], nuisance_idx
    = [num_knots+2..num_knots+8]) this is bit-identical to
    ``torch.cat([inferred, nuisance])``.
    """
    total = int(inferred_idx.numel() + nuisance_idx.numel())
    out = torch.zeros(total, dtype=inferred.dtype, device=inferred.device)
    out[inferred_idx] = inferred
    out[nuisance_idx] = nuisance.to(device=out.device, dtype=out.dtype)
    return out


def _normalize_scale(scale, num_knots: int) -> List[float]:
    """Broadcast a prior half-width into one positive value per knot.

    ``scale`` may be a scalar (shared half-width for every knot) or a sequence
    of length ``num_knots`` (a per-knot half-width). Returns a plain list of
    floats so the caller can index it knot-by-knot.
    """
    if isinstance(scale, (int, float)):
        scales = [float(scale)] * num_knots
    else:
        scales = [float(s) for s in scale]
        if len(scales) != num_knots:
            raise ValueError(
                f"prior.scale has length {len(scales)} but num_knots={num_knots}; "
                f"pass a scalar or a sequence of length {num_knots}."
            )
    if any(s <= 0.0 for s in scales):
        raise ValueError(f"prior.scale entries must be positive; got {scales}")
    return scales


def generate_spline_prior(
        centers: Union[torch.Tensor, None] = None,
        scale: Union[float, Sequence[float]] = 1.0,
        num_knots: int = 10,
        device: str = "cpu"
    ) -> List[dists.Uniform]:
    """Independent uniform prior per knot: g_i ~ U(center_i - s_i, center_i + s_i).

    ``scale`` is either a scalar (same half-width ``s_i`` for every knot) or a
    sequence of length ``num_knots`` giving a separate half-width per knot.
    """
    if centers is None:
        centers = torch.zeros(num_knots)

    scales = _normalize_scale(scale, num_knots)

    prior_spline = [
        dists.Uniform(
            torch.tensor([centers[i] - scales[i]], device=device),
            torch.tensor([centers[i] + scales[i]], device=device),
        )
        for i in range(num_knots)
    ]
    return prior_spline


def _rbf_knot_covariance(
        scales: Union[torch.Tensor, Sequence[float]],
        correlation_length: float,
        device: str = "cpu",
        jitter: float = 1e-6,
    ) -> torch.Tensor:
    """RBF (squared-exponential) covariance over knots, in knot-index units.

    ``K_ij = sigma_i * sigma_j * exp(-(i - j)^2 / (2 * l^2))`` where the marginal
    std ``sigma_i`` is ``scales[i]`` and ``l`` is ``correlation_length``. Inner
    knots are equally spaced in x (see ``make_spline_knots``), so an index-space
    length maps linearly to physical x. A small ``jitter`` is added to the diagonal
    so the matrix is comfortably positive-definite for a float32 Cholesky.
    """
    if correlation_length <= 0.0:
        raise ValueError(
            f"correlation_length must be positive; got {correlation_length}"
        )
    scales = torch.as_tensor(scales, dtype=torch.float32, device=device).reshape(-1)
    n = scales.numel()
    idx = torch.arange(n, dtype=torch.float32, device=device)
    d2 = (idx[:, None] - idx[None, :]) ** 2
    corr = torch.exp(-d2 / (2.0 * float(correlation_length) ** 2))
    K = (scales[:, None] * scales[None, :]) * corr
    if jitter:
        K = K + float(jitter) * torch.eye(n, dtype=torch.float32, device=device)
    return K


def generate_correlated_spline_prior(
        centers: Union[torch.Tensor, Sequence[float], None] = None,
        scale: Union[float, Sequence[float]] = 1.0,
        num_knots: int = 10,
        correlation_length: float = 1.0,
        device: str = "cpu",
        jitter: float = 1e-6,
    ) -> List[dists.MultivariateNormal]:
    """Correlated Gaussian prior over the knot vector: g ~ N(center, K_rbf).

    Mean is ``centers`` (zeros if None), the marginal std of knot ``i`` is
    ``scale`` (scalar or per-knot, reused as-is from the uniform prior), and the
    RBF ``correlation_length`` couples neighbouring knots to control landscape
    smoothness. Returned as a **single-element list** ``[MultivariateNormal]`` so
    callers can ``*``-spread it exactly where ``generate_spline_prior`` returned a
    list of per-knot Uniforms; the one multivariate component has
    ``event_shape=[num_knots]`` and slots into ``MultipleIndependent`` directly.
    """
    if centers is None:
        loc = torch.zeros(num_knots, dtype=torch.float32, device=device)
    else:
        loc = torch.as_tensor(centers, dtype=torch.float32, device=device).reshape(-1)
    if loc.numel() != num_knots:
        raise ValueError(
            f"center has length {loc.numel()} but num_knots={num_knots}."
        )
    scales = _normalize_scale(scale, num_knots)
    cov = _rbf_knot_covariance(scales, correlation_length, device=device, jitter=jitter)
    return [dists.MultivariateNormal(loc=loc, covariance_matrix=cov)]


def _knot_prior_dists(cfg, center, scale_, num_knots, device):
    """Dispatch the ``knots`` block to independent-uniform or correlated-Gaussian.

    Reads ``prior.knot_prior`` (default ``independent``) and, for the Gaussian
    case, ``prior.correlation_length`` (default 1.0). Returns a list of prior
    components covering exactly ``num_knots`` sample dimensions in either case.
    """
    knot_prior = cfg.prior.get("knot_prior", "independent")
    if knot_prior == "gaussian":
        return generate_correlated_spline_prior(
            centers=center, scale=scale_, num_knots=num_knots,
            correlation_length=cfg.prior.get("correlation_length", 1.0),
            device=device,
        )
    if knot_prior != "independent":
        raise ValueError(
            f"Unknown prior.knot_prior: {knot_prior!r}. "
            f"Valid: 'independent', 'gaussian'."
        )
    return generate_spline_prior(
        centers=center, scale=scale_, num_knots=num_knots, device=device
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
    C_gr_lims: tuple = (0.0, 0.2),
    C_rg_lims: tuple = (0.0, 0.2),
    center: Union[torch.Tensor, List[float], None] = None,
    scale: Union[float, Sequence[float]] = 1.0,
    knot_prior: str = "independent",
    correlation_length: float = 1.0,
    device: str = "cpu",
) -> MultipleIndependent:
    """
    Build the full FRET prior: logD + gradient knots + photophysics.

    The crosstalk matrix is conserved by construction: only the two leak
    fractions C_gr (green → red) and C_rg (red → green) are sampled; the
    diagonals are derived downstream as C_gg = 1 − C_gr and C_rr = 1 − C_rg.

    Parameter order: [logD, g₀…gₙ₋₁, R0, kD, k_gb, k_rb, eta_g, eta_r,
                      C_gr, C_rg]
    Total length: num_knots + 9

    Args:
        num_knots: Number of spline knots n; prior holds n gradient knot values.
        logD_lims … C_rg_lims: (low, high) bounds for each parameter.
        center: Optional force template of length n (knot force values).
            None → flat, symmetric prior (zero mean force).
        scale: Half-width of each knot's uniform box. Either a scalar (shared
            across all knots) or a length-n sequence for a per-knot half-width.
        knot_prior: 'independent' (per-knot uniforms, default) or 'gaussian'
            (correlated RBF multivariate normal over the knot vector).
        correlation_length: RBF length in knot-index units; only used when
            ``knot_prior='gaussian'``.
        device: Torch device.

    Examples
    --------
    Flat prior, all force knots in [-10, 10]:
        >>> prior = generate_spline_fret_prior(num_knots=5, scale=10.0)

    Prior expecting a central restoring force (harmonic-like):
        >>> center = torch.tensor([4., 2., 0., -2., -4.])
        >>> prior = generate_spline_fret_prior(num_knots=5, scale=3.0, center=center)

    Per-knot half-widths (tighter at the ends, looser in the middle):
        >>> prior = generate_spline_fret_prior(
        ...     num_knots=5, scale=[1., 2., 3., 2., 1.], center=center)
    """
    prior_logD = dists.Uniform(
        torch.tensor([logD_lims[0]], device=device),
        torch.tensor([logD_lims[1]], device=device),
    )
    
    if center is not None:
        center = torch.as_tensor(center, dtype=torch.float32, device=device)

    if knot_prior == "gaussian":
        prior_spline = generate_correlated_spline_prior(
            centers=center, scale=scale, num_knots=num_knots,
            correlation_length=correlation_length, device=device,
        )
    elif knot_prior == "independent":
        prior_spline = generate_spline_prior(
            centers=center, scale=scale, num_knots=num_knots, device=device
        )
    else:
        raise ValueError(
            f"Unknown knot_prior: {knot_prior!r}. Valid: 'independent', 'gaussian'."
        )

    phys_lims = [
        R0_lims, kD_lims, k_gb_lims, k_rb_lims,
        eta_g_lims, eta_r_lims, C_gr_lims, C_rg_lims,
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
        C_gr_lims=tuple(cfg.prior.C_gr_lims),
        C_rg_lims=tuple(cfg.prior.C_rg_lims),
        center=center,
        scale=cfg.prior.scale,
        knot_prior=cfg.prior.get("knot_prior", "independent"),
        correlation_length=cfg.prior.get("correlation_length", 1.0),
        device=device,
    )


def generate_posterior_prior_from_config(cfg, device: str = "cpu") -> MultipleIndependent:
    """
    Reduced prior (logD + gradient knots + R0) for posterior sampling.

    Returns MultipleIndependent([prior_logD, *prior_spline, prior_R0]).
    Parameter order: [logD, g₀…gₙ₋₁, R0]. Total length: num_knots + 2.
    """
    cfg = OmegaConf.load(cfg) if isinstance(cfg, str) else cfg

    scale = cfg.prior.scale
    center = list(cfg.prior.center) if cfg.prior.center is not None else None
    num_knots = cfg.prior.num_knots

    prior_logD = dists.Uniform(
        torch.tensor([cfg.prior.logD_lims[0]], device=device),
        torch.tensor([cfg.prior.logD_lims[1]], device=device),
    )

    prior_spline = _knot_prior_dists(cfg, center, scale, num_knots, device)

    prior_R0 = dists.Uniform(
        torch.tensor([cfg.prior.R0_lims[0]], device=device),
        torch.tensor([cfg.prior.R0_lims[1]], device=device),
    )

    return MultipleIndependent([prior_logD, *prior_spline, prior_R0], device=device)


def generate_nuisance_prior_from_config(cfg, device: str = "cpu") -> MultipleIndependent:
    """
    Photophysics-only prior over the 7 nuisance parameters that vary per trace.

    Parameter order: [kD, k_gb, k_rb, eta_g, eta_r, C_gr, C_rg]. Total length: 7.
    Shares bounds with ``generate_fret_prior_from_config`` via the same config keys.
    """
    cfg = OmegaConf.load(cfg) if isinstance(cfg, str) else cfg

    phys_lims = [
        tuple(cfg.prior.kD_lims),
        tuple(cfg.prior.k_gb_lims),
        tuple(cfg.prior.k_rb_lims),
        tuple(cfg.prior.eta_g_lims),
        tuple(cfg.prior.eta_r_lims),
        tuple(cfg.prior.C_gr_lims),
        tuple(cfg.prior.C_rg_lims),
    ]
    phys_priors = [
        dists.Uniform(torch.tensor([lo], device=device), torch.tensor([hi], device=device))
        for lo, hi in phys_lims
    ]

    return MultipleIndependent(phys_priors, device=device)
