import logging
import math
import os
import random
import subprocess
import types
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from sbi.inference.posteriors import MCMCPosterior
from sbi.inference.potentials import ratio_estimator_based_potential
from sbi.utils import within_support
from sbi.utils.sbiutils import gradient_ascent, mcmc_transform
from torch import nn

from gsl_spline import eval_spline, integrate_spline

logger = logging.getLogger(__name__)


def git_commit(repo_dir: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    best_val_loss,
    best_model_state_dict,
    best_epoch,
    logger: Optional[logging.Logger] = None,
):
    """Save a resumable training checkpoint."""
    scheduler_state_dict = scheduler.state_dict() if scheduler is not None else None
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler_state_dict,
            "best_val_loss": best_val_loss,
            "best_model_state_dict": best_model_state_dict,
            "best_epoch": best_epoch,
        },
        path,
    )
    if logger is not None:
        logger.info(
            f"Checkpoint saved: epoch {epoch}, best_val={best_val_loss:.4f}, path={path}"
        )


def load_checkpoint(path, model, optimizer, scheduler, device, logger: Optional[logging.Logger] = None):
    """Load checkpoint and return (start_epoch, best_val_loss, best_model_state_dict, best_epoch)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    ckpt_scheduler_state = ckpt.get("scheduler_state_dict", None)
    if scheduler is not None and ckpt_scheduler_state is not None:
        scheduler.load_state_dict(ckpt_scheduler_state)
    elif scheduler is not None and ckpt_scheduler_state is None:
        if logger is not None:
            logger.warning("Checkpoint has no scheduler state, but scheduler is enabled in config.")
    elif scheduler is None and ckpt_scheduler_state is not None:
        if logger is not None:
            logger.info(
                "Checkpoint contains scheduler state, but scheduler is disabled in config; ignoring scheduler state."
            )
    if logger is not None:
        logger.info(
            f"Resumed from checkpoint: epoch {ckpt['epoch']}, best_val={ckpt['best_val_loss']:.4f}, path={path}"
        )
    return (
        ckpt["epoch"],
        ckpt["best_val_loss"],
        ckpt["best_model_state_dict"],
        ckpt.get("best_epoch", ckpt["epoch"]),
    )


def setup_ddp() -> bool:
    """Initialize DDP if launched via torchrun. Returns True if DDP is active."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        return True
    return False


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def get_device() -> torch.device:
    if dist.is_initialized():
        return torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def count_model_parameters(model) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def set_training_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_embedding_name(cfg_embedding) -> str:
    name = getattr(cfg_embedding, "name", None)
    if name is not None:
        return str(name)
    if hasattr(cfg_embedding, "kernel_sizes") or hasattr(cfg_embedding, "depth"):
        return "transformer"
    return "cnn"


def l2_norm(tensors) -> float:
    norms = [tensor.detach().float().norm(2) for tensor in tensors if tensor is not None]
    if not norms:
        return 0.0
    return torch.norm(torch.stack(norms), p=2).cpu().item()


def parameter_norm(parameters) -> float:
    return l2_norm([param for param in parameters if param.requires_grad])


def gradient_norm(parameters) -> float:
    return l2_norm(
        [param.grad for param in parameters if param.requires_grad and param.grad is not None]
    )


def split_trainable_parameters(model, encoder):
    encoder_param_ids = {id(param) for param in encoder.parameters() if param.requires_grad}
    encoder_params = []
    other_params = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if id(param) in encoder_param_ids:
            encoder_params.append(param)
        else:
            other_params.append(param)
    return encoder_params, other_params


def resolve_data_dir(data_dir, orig_cwd, logger: Optional[logging.Logger] = None) -> Path:
    data_dir = Path(data_dir)
    if data_dir.is_absolute():
        return data_dir

    orig_candidate = Path(orig_cwd) / data_dir
    if orig_candidate.exists():
        return orig_candidate

    repo_candidate = Path(__file__).resolve().parents[1] / data_dir
    if repo_candidate.exists():
        if logger is not None:
            logger.info(f"Resolved relative data_dir via repo root: {repo_candidate}")
        return repo_candidate

    return orig_candidate


def chunk_num_samples(chunk_index: int, chunk_size: int, num_examples: int) -> int:
    start = chunk_index * chunk_size
    end = min(start + chunk_size, num_examples)
    return end - start


def trim_ddp_train_chunks(train_chunk_indices, chunk_size: int, num_examples: int, world_size: int):
    full_train_chunks = [
        chunk_index
        for chunk_index in train_chunk_indices
        if chunk_num_samples(chunk_index, chunk_size, num_examples) == chunk_size
    ]
    dropped_partial_chunks = len(train_chunk_indices) - len(full_train_chunks)
    usable_chunk_count = len(full_train_chunks) - (len(full_train_chunks) % world_size)
    dropped_remainder_chunks = len(full_train_chunks) - usable_chunk_count

    if usable_chunk_count <= 0:
        raise ValueError(
            "DDP training requires at least one full chunk per rank after trimming uneven chunks. "
            f"Got {len(train_chunk_indices)} candidate train chunks, {len(full_train_chunks)} full chunks, "
            f"world_size={world_size}."
        )

    return (
        full_train_chunks[:usable_chunk_count],
        dropped_partial_chunks,
        dropped_remainder_chunks,
    )


def assert_equal_train_steps(local_steps: int, device: torch.device):
    local_steps_tensor = torch.tensor([local_steps], device=device, dtype=torch.long)
    gathered_steps = [torch.zeros_like(local_steps_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_steps, local_steps_tensor)
    steps_per_rank = [int(steps.item()) for steps in gathered_steps]
    if len(set(steps_per_rank)) != 1:
        raise RuntimeError(
            "DDP ranks have different numbers of train steps per epoch, which would deadlock NCCL. "
            f"steps_per_rank={steps_per_rank}"
        )
    return steps_per_rank


_ALLOWED_BCS = ("linear", "quadratic", "open")


def _validate_bc(bc, allow_open=True):
    """Normalise a 2-element boundary-condition spec into a (left, right) tuple.

    Raises ValueError on length mismatch, unknown value, or — when
    ``allow_open=False`` — any ``"open"`` entry.
    """
    try:
        items = list(bc)
    except TypeError:
        raise ValueError(
            f"boundary_condition must be a 2-element list [left, right]; got {bc!r}"
        )
    if len(items) != 2:
        raise ValueError(
            f"boundary_condition must have exactly 2 elements, got {len(items)}: {bc!r}"
        )
    left, right = str(items[0]), str(items[1])
    for side, val in (("left", left), ("right", right)):
        if val not in _ALLOWED_BCS:
            raise ValueError(
                f"Unsupported boundary_condition on {side} side: {val!r} "
                f"(allowed: {_ALLOWED_BCS})"
            )
        if not allow_open and val == "open":
            raise ValueError(
                f"boundary_condition {val!r} is not allowed on the {side} side "
                f"for this simulator.mode (equilibrium mode cannot contain "
                f"particles with an open boundary)"
            )
    return left, right


def check_config_consistency(cfg):
    """Validate logical self-consistency of the inferred / nuisance config keys.

    Catches contradictions between ``prior.inferred_params`` and
    ``data_generation.nuisance_per_trace_params`` (the per-set/per-trace knobs),
    plus settings that are valid but have no effect.

    Raises
    ------
    ValueError
        On any hard contradiction:
          * a param listed in BOTH ``prior.inferred_params`` and
            ``data_generation.nuisance_per_trace_params`` (an inferred param is
            shared across the set and learned — it cannot also be a per-trace
            nuisance);
          * an unknown token in ``nuisance_per_trace_params`` (not a physics
            nuisance, e.g. a typo or ``logD``/``knots``);
          * duplicate entries in ``nuisance_per_trace_params``.

    Logs a warning (does not raise) for no-op settings:
          * ``nuisance_per_trace_params`` set while ``nuisance_per_set`` is false;
          * ``nuisance_per_trace_params`` set while ``num_traces_per_set <= 1``;
          * ``nuisance_per_set`` true while ``num_traces_per_set <= 1``.

    Returns
    -------
    list[str]
        The warning messages that were emitted (empty if none).
    """
    from priors import PHYS_PARAM_NAMES, DEFAULT_INFERRED  # local import avoids any import cycle

    cfg = OmegaConf.load(cfg) if isinstance(cfg, str) else cfg

    prior_cfg = cfg.get("prior", {}) or {}
    dgen = cfg.get("data_generation", {}) or {}

    inferred = list(prior_cfg.get("inferred_params", None) or DEFAULT_INFERRED)
    per_trace = list(dgen.get("nuisance_per_trace_params", None) or [])
    nuisance_per_set = bool(dgen.get("nuisance_per_set", False))
    num_traces_per_set = int(dgen.get("num_traces_per_set", 1))

    errors, warns = [], []

    # 1. unknown tokens in nuisance_per_trace_params (must be physics nuisances)
    bad = [p for p in per_trace if p not in PHYS_PARAM_NAMES]
    if bad:
        errors.append(
            f"nuisance_per_trace_params contains non-physics-nuisance token(s) {bad}; "
            f"valid options are {PHYS_PARAM_NAMES}."
        )

    # 2. duplicates in nuisance_per_trace_params
    dupes = sorted({p for p in per_trace if per_trace.count(p) > 1})
    if dupes:
        errors.append(f"nuisance_per_trace_params has duplicate entries: {dupes}.")

    # 3. a param cannot be both inferred and a per-trace nuisance
    both = [p for p in per_trace if p in set(inferred)]
    if both:
        errors.append(
            f"param(s) {both} appear in BOTH prior.inferred_params and "
            f"data_generation.nuisance_per_trace_params. An inferred param is shared "
            f"across a set and learned, so it cannot also be a per-trace nuisance — "
            f"remove it from one of the two lists."
        )

    # 4. no-op settings (valid, but have no effect) -> warn only
    if per_trace and not nuisance_per_set:
        warns.append(
            "nuisance_per_trace_params is set but nuisance_per_set is false; every "
            "nuisance is already drawn per trace, so the list has no effect."
        )
    if per_trace and num_traces_per_set <= 1:
        warns.append(
            "nuisance_per_trace_params is set but num_traces_per_set <= 1 (single-trace "
            "mode); per-set vs per-trace nuisance is moot, so the list has no effect."
        )
    if nuisance_per_set and num_traces_per_set <= 1:
        warns.append(
            "nuisance_per_set is true but num_traces_per_set <= 1; with one trace per "
            "set there is nothing to share a nuisance across."
        )

    if errors:
        raise ValueError(
            "Config self-consistency check failed:\n  - " + "\n  - ".join(errors)
        )
    for w in warns:
        logger.warning("config consistency: %s", w)
    return warns


def _boundary_slots(edge_value, direction, bc, max_force, N):
    """Return N boundary-knot force values ordered innermost → outermost.

    ``direction`` is +1 for the left side (force grows positive outward) and
    −1 for the right side (force grows negative outward).
    """
    if bc == "open":
        return np.zeros(N)
    slots = np.empty(N)
    for j in range(1, N + 1):
        if bc == "linear":
            delta = direction * j * (max_force / N)
        else:  # "quadratic" — _validate_bc has already ruled out other values
            delta = direction * (j / N) ** 2 * max_force
        slots[j - 1] = edge_value + delta
    return slots


def make_spline_knots(
    params,
    x_min=0.0,
    x_max=10.0,
    num_knots=10,
    max_force=10.0,
    num_boundary_knots=3,
    boundary_condition=("linear", "linear"),
):
    left_bc, right_bc = _validate_bc(boundary_condition)
    g_inner = np.asarray(params[1:num_knots + 1], dtype=np.float64)
    knot_spacing = (x_max - x_min) / (num_knots - 1)
    N = num_boundary_knots
    x_knots = np.linspace(
        x_min - N * knot_spacing,
        x_max + N * knot_spacing,
        num_knots + 2 * N,
    )
    y_knots = np.zeros(num_knots + 2 * N)
    y_knots[N:-N] = g_inner
    y_knots[:N] = _boundary_slots(g_inner[0], +1, left_bc, max_force, N)[::-1]
    y_knots[-N:] = _boundary_slots(g_inner[-1], -1, right_bc, max_force, N)
    return x_knots, y_knots


def integrate_force_to_energy(x_knots, g_knots, x_eval, spline_type="cubic"):
    """Recover U(x) = -∫F(x)dx via the GSL spline integral.

    Integrates the force spline analytically (exact for the piecewise
    polynomial) using gsl_spline_eval_integ_e. The additive constant is
    fixed by pinning the minimum of U to zero.

    Parameters
    ----------
    x_knots : array_like
        x-positions of the force spline knots (with boundary knots).
    g_knots : array_like
        Force values at the knots (output of make_spline_knots).
    x_eval : np.ndarray
        Positions at which to evaluate U(x).
    spline_type : str
        GSL spline type ("cubic", "steffen", or "akima").

    Returns
    -------
    U : np.ndarray, shape (len(x_eval),)
        Free energy profile, with min(U) = 0.
    """
    U = -integrate_spline(
        np.asarray(x_knots, dtype=np.float64),
        np.asarray(g_knots, dtype=np.float64),
        np.asarray(x_eval, dtype=np.float64),
        spline_type,
    )
    return U


def eval_free_energy(
    params,
    x_eval,
    N_knots,
    x_min,
    x_max,
    boundary_condition,
    max_force,
    spline_type="cubic",
    num_boundary_knots=4,
):
    if isinstance(params, torch.Tensor):
        params = params.detach().cpu().numpy()

    x_knots, y_knots = make_spline_knots(
        params=params,
        x_min=x_min,
        x_max=x_max,
        num_knots=N_knots,
        max_force=max_force,
        boundary_condition=boundary_condition,
        num_boundary_knots=num_boundary_knots,
    )

    y = integrate_force_to_energy(x_knots, y_knots, x_eval, spline_type=spline_type)
    return x_eval, y


def eval_free_energy_from_config(cfg):
    cfg = OmegaConf.load(cfg) if isinstance(cfg, str) else cfg
    return partial(
        eval_free_energy,
        N_knots=cfg.prior.num_knots,
        x_min=cfg.simulator.x_min,
        x_max=cfg.simulator.x_max,
        boundary_condition=cfg.simulator.boundary_condition,
        max_force=cfg.simulator.max_force,
        spline_type=cfg.simulator.spline_type,
        num_boundary_knots=cfg.simulator.num_boundary_knots,
    )


def eval_force(
    params,
    x_eval,
    N_knots,
    x_min,
    x_max,
    boundary_condition,
    max_force,
    spline_type="cubic",
    num_boundary_knots=4,
):
    if isinstance(params, torch.Tensor):
        params = params.detach().cpu().numpy()

    x_knots, y_knots = make_spline_knots(
        params=params,
        x_min=x_min,
        x_max=x_max,
        num_knots=N_knots,
        max_force=max_force,
        boundary_condition=boundary_condition,
        num_boundary_knots=num_boundary_knots,
    )

    y = eval_spline(x_knots, y_knots, x_eval, spline_type=spline_type)
    return x_eval, y

def eval_force_from_config(cfg):
    cfg = OmegaConf.load(cfg) if isinstance(cfg, str) else cfg
    return partial(
        eval_force,
        N_knots=cfg.prior.num_knots,
        x_min=cfg.simulator.x_min,
        x_max=cfg.simulator.x_max,
        boundary_condition=cfg.simulator.boundary_condition,
        max_force=cfg.simulator.max_force,
        spline_type=cfg.simulator.spline_type,
        num_boundary_knots=cfg.simulator.num_boundary_knots,
    )


def _install_cached_x_embedding(ratio_estimator, observation, device):
    """Pre-embed x_o once and monkeypatch combine_theta_and_x to reuse it."""
    with torch.no_grad():
        x_o_embedded = ratio_estimator.embedding_net_x(
            observation.to(device).reshape(-1, *ratio_estimator.x_shape)
        )
    ratio_estimator._x_cached_embedded = x_o_embedded

    _original_combine = ratio_estimator.__class__.combine_theta_and_x

    def _fast_combine(self, theta, x):
        theta_shape_size = math.prod(self.theta_shape)
        n_iid = self._x_cached_embedded.shape[0]

        if theta.numel() % theta_shape_size != 0:
            return _original_combine(self, theta, x)

        n = theta.numel() // theta_shape_size
        if n % n_iid != 0:
            return _original_combine(self, theta, x)

        embedded_theta = self.embedding_net_theta(theta.reshape(n, *self.theta_shape))
        embedded_x = self._x_cached_embedded.repeat_interleave(n // n_iid, dim=0)
        prefix_shape = theta.shape[: -len(self.theta_shape)]
        return torch.cat([embedded_theta, embedded_x], dim=-1).reshape(*prefix_shape, -1)

    ratio_estimator.combine_theta_and_x = types.MethodType(_fast_combine, ratio_estimator)


def _build_cached_ratio_potentials(ratio_estimators, prior, observation, device):
    """Move each estimator to ``device``, install cached x embeddings, and build
    per-member ``RatioBasedPotential``s. Returns ``(potentials, theta_transform)``.
    The ``theta_transform`` is constructed explicitly on ``device`` to avoid
    device mismatches when a pickled estimator was saved on CPU.
    """

    observation_dev = observation.to(device)
    potentials = []
    for re in ratio_estimators:
        re.to(device)
        _install_cached_x_embedding(re, observation_dev, device)
        potential, _ = ratio_estimator_based_potential(re, prior, observation_dev)
        potentials.append(potential)

    theta_transform = mcmc_transform(prior, device=device)
    return potentials, theta_transform


def build_cached_posterior(
    ratio_estimator,
    prior,
    observation,
    num_chains=50,
    thin=2,
    device="cuda",
):
    """MCMCPosterior backed by a single ratio estimator with cached x embedding."""
    return build_cached_ensembled_posterior(
        [ratio_estimator],
        prior,
        observation,
        num_chains=num_chains,
        thin=thin,
        device=device,
    )


def build_cached_ensembled_posterior(
    ratio_estimators,
    prior,
    observation,
    num_chains=50,
    thin=2,
    weights=None,
    device="cuda",
):
    """Product-of-experts MCMCPosterior over a list of cached ratio estimators.

    Each per-member ``RatioBasedPotential`` already adds ``prior.log_prob(theta)``,
    so the weighted sum subtracts ``(sum(weights) - 1) * log_prior`` to keep the
    prior contribution at exactly one.
    """

    if len(ratio_estimators) == 0:
        raise ValueError("ratio_estimators must be non-empty.")
    if weights is None:
        weights = [1.0] * len(ratio_estimators)
    if len(weights) != len(ratio_estimators):
        raise ValueError("weights must match ratio_estimators in length.")

    potentials, theta_transform = _build_cached_ratio_potentials(
        ratio_estimators, prior, observation, device
    )
    prior_correction = float(sum(weights)) - 1.0

    def ensemble_potential(theta, x_o=None, track_gradients=True):
        # x_o is ignored: each member holds its own cached x embedding.
        weighted_terms = [
            w * p(theta, track_gradients=track_gradients)
            for w, p in zip(weights, potentials)
        ]
        log_post = torch.stack(weighted_terms, dim=0).sum(dim=0)
        if prior_correction != 0.0:
            log_post = log_post - prior_correction * prior.log_prob(theta)
        return log_post

    return MCMCPosterior(
        ensemble_potential,
        proposal=prior,
        theta_transform=theta_transform,
        num_chains=num_chains,
        thin=thin,
        method="slice_np_vectorized",
        device=device,
    )


def map_with_cached_embedding(
    density_estimator,
    prior,
    x,
    num_iter: int,
    num_init_samples: int,
    num_to_optimize: int,
    learning_rate: float,
    init_samples: Optional[torch.Tensor] = None,
    show_progress_bars: bool = False,
):
    """MAP estimator for an NPE density estimator that embeds ``x`` exactly once.

    Equivalent to ``DirectPosterior.map()`` but hoists the (expensive)
    embedding-net forward pass out of the gradient loop: the embedding runs
    once up front, then the inner log_prob path is fed the cached embedded
    context on every gradient step. For ``maf_rqs`` (NFlowsFlow) the
    embedding net lives at ``density_estimator.net._embedding_net``; for
    ZukoFlow it lives at ``density_estimator._embedding_net``. Both are
    temporarily swapped to ``nn.Identity`` inside a try/finally so that the
    wrapped ``log_prob(theta, condition=emb_x)`` becomes a no-op embedding
    plus the usual flow transform + base distribution log prob.
    """
    device = next(density_estimator.parameters()).device
    density_estimator.eval()
    x = x.to(device)

    with torch.no_grad():
        emb_x = density_estimator.embedding_net(x).detach()

    if hasattr(density_estimator.net, "_embedding_net"):
        embed_holder, embed_attr = density_estimator.net, "_embedding_net"
    else:
        embed_holder, embed_attr = density_estimator, "_embedding_net"
    original = getattr(embed_holder, embed_attr)
    setattr(embed_holder, embed_attr, nn.Identity())

    try:
        def potential_fn(theta):
            theta_ = theta.to(device)
            if theta_.dim() == 1:
                theta_ = theta_.unsqueeze(0)
            theta_batched = theta_.unsqueeze(1)
            log_p = density_estimator.log_prob(theta_batched, condition=emb_x)
            log_p = log_p.squeeze(1)
            in_support = within_support(prior, theta_)
            return torch.where(
                in_support,
                log_p,
                torch.tensor(-float("inf"), device=log_p.device, dtype=log_p.dtype),
            )

        if init_samples is None:
            init_samples = prior.sample((num_init_samples,))
        init_samples = init_samples.to(device).detach()

        theta_transform = mcmc_transform(prior, device=str(device))

        argmax_, max_val = gradient_ascent(
            potential_fn=potential_fn,
            inits=init_samples,
            theta_transform=theta_transform,
            num_iter=num_iter,
            num_to_optimize=num_to_optimize,
            learning_rate=learning_rate,
            show_progress_bars=show_progress_bars,
        )
    finally:
        setattr(embed_holder, embed_attr, original)

    return argmax_.detach().cpu().squeeze(0), float(max_val.detach().cpu().item())