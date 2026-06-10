#!/usr/bin/env python
"""Evaluate ONE trained NPE posterior under a given set of settings (CLI).

Given a trained-model output directory (the folder holding ``data_config.yaml`` and a
``*_density_estimator.pkl``), build a set of ground-truth free-energy cases:

  * the **prior-centre** double well (the spline-centre knots == the prior mean),
  * ``--n-prior-samples`` shapes drawn from the model's prior (seeded by ``--seed``),
  * any explicit ``--knots`` shapes you pass.

The **ground-truth nuisance parameters** (logD, R0, kD, k_gb, k_rb, eta_g, eta_r,
C_gr, C_rg) are controlled with ``--set PARAM=VALUE`` and applied to every case. For
each case it simulates one observation set, samples the posterior + finds the MAP, and
saves the raw results (samples, MAP, ground truth, free energy) + the free-energy plot
into ``<out-dir>/cases/<case>/``.

This script does ONE thing: evaluate one posterior under one set of settings. Sweeps
and multi-model comparisons are done by calling it MULTIPLE times (e.g. a kD sweep is a
shell loop over ``--set kD=...`` with different ``--out-dir``) and loading the saved
``posterior.npz`` files elsewhere.

Everything physical (num_knots, priors, photophysics, simulator settings) is read from
the model's ``data_config.yaml`` — nothing is hardcoded.

Pool / CUDA safety: all observations are simulated with the worker pool *before* any
GPU work, then the pool is closed. No multiprocessing fork ever happens after a CUDA
context exists.

Examples
--------
    python analysis/run_eval_posterior.py <model_dir>
    python analysis/run_eval_posterior.py <model_dir> --set kD=300 --out-dir results/kD_300
    python analysis/run_eval_posterior.py <model_dir> --knots "5.5,-4.5,0.3,0,-0.3,4.5,-5.5"
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # headless: never tries to open a window
import matplotlib.pyplot as plt
import numpy as np
import torch
from multiprocessing import Pool
from omegaconf import OmegaConf
from tqdm import tqdm

# ── Project imports (src/ on path) ───────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.append(str(_SRC))

from sbi import analysis as analysis
from sbi.inference.posteriors import DirectPosterior
from sbi.utils import within_support

from simulator import (
    smFRET_simulator_wrapper_from_config,
    binding_simulator_wrapper_from_config,
)
from priors import (
    generate_fret_prior_from_config,
    generate_split_priors_from_config,
    PHYS_PARAM_NAMES,
    _normalize_scale,
)
from utils import eval_free_energy_from_config, eval_force_from_config, map_with_cached_embedding
from data_loaders import H5Dataset


# =============================================================================
# Context: everything derived once from the model's data_config.yaml
# =============================================================================
@dataclass
class Ctx:
    data_cfg: Any
    data_cfg_path: str
    device: str
    # priors / indices
    prior_sim_cpu: Any
    inferred_idx: torch.Tensor
    inferred_labels: list
    # evaluators (numpy/CPU)
    eval_force: Any
    eval_free_energy: Any
    # dims
    num_knots: int
    canonical_dim: int
    theta_dim: int
    # knot grid / plotting grid
    x_min: float
    x_max: float
    knot_x: np.ndarray
    x_eval: np.ndarray
    # prior box (per-knot aware)
    prior_center: np.ndarray
    prior_scale: np.ndarray
    # set in phase B
    density_estimator: Any = None
    inferred_prior: Any = None


def build_context(cfg_path: str, device: str) -> Ctx:
    data_cfg_path = str(Path(cfg_path) / "data_config.yaml")
    data_cfg = OmegaConf.load(data_cfg_path)

    # Sampling prior lives on CPU so we can draw ground-truth cases before any CUDA use.
    prior_sim_cpu = generate_fret_prior_from_config(data_cfg_path, device="cpu")
    _, _, inferred_idx, _ = generate_split_priors_from_config(data_cfg_path, device="cpu")

    eval_force = eval_force_from_config(data_cfg_path)
    eval_free_energy = eval_free_energy_from_config(data_cfg_path)

    num_knots = int(data_cfg.prior.num_knots)
    canonical_dim = num_knots + 9
    theta_dim = int(inferred_idx.numel())

    tok_names = ["logD"] + [f"g{i}" for i in range(num_knots)] + list(PHYS_PARAM_NAMES)
    inferred_labels = [tok_names[i] for i in inferred_idx.tolist()]

    x_min = float(data_cfg.simulator.x_min)
    x_max = float(data_cfg.simulator.x_max)
    knot_x = np.linspace(x_min, x_max, num_knots)
    knot_spacing = (x_max - x_min) / (num_knots - 1)
    x_margin = 0.6 * int(data_cfg.simulator.num_boundary_knots) * knot_spacing
    x_eval = np.linspace(x_min - x_margin, x_max + x_margin, 500)

    prior_center = np.asarray(
        list(data_cfg.prior.center) if data_cfg.prior.center is not None
        else [0.0] * num_knots, dtype=float
    )
    # Per-knot half-widths (scalar configs broadcast to a uniform array).
    prior_scale = np.asarray(_normalize_scale(data_cfg.prior.scale, num_knots), dtype=float)

    return Ctx(
        data_cfg=data_cfg, data_cfg_path=data_cfg_path, device=device,
        prior_sim_cpu=prior_sim_cpu, inferred_idx=inferred_idx, inferred_labels=inferred_labels,
        eval_force=eval_force, eval_free_energy=eval_free_energy,
        num_knots=num_knots, canonical_dim=canonical_dim, theta_dim=theta_dim,
        x_min=x_min, x_max=x_max, knot_x=knot_x, x_eval=x_eval,
        prior_center=prior_center, prior_scale=prior_scale,
    )


# =============================================================================
# (a) Parameters & free-energy shapes
# =============================================================================
def _lims_mid(ctx: Ctx, name: str) -> float:
    lo, hi = tuple(ctx.data_cfg.prior[f"{name}_lims"])
    return 0.5 * (float(lo) + float(hi))


def default_canonical_from_config(ctx: Ctx) -> torch.Tensor:
    """Canonical theta with every non-knot parameter at its config midpoint.

    For pinned params (lims [a, a]) the midpoint is exactly ``a`` -> stays in support.
    Knots are left at 0 and filled per case.
    """
    theta = torch.zeros(ctx.canonical_dim, dtype=torch.float64)
    theta[0] = _lims_mid(ctx, "logD")
    for i, name in enumerate(PHYS_PARAM_NAMES):
        theta[ctx.num_knots + 1 + i] = _lims_mid(ctx, name)
    return theta


def canonical_index(ctx: Ctx) -> dict:
    """Map each controllable nuisance name -> its canonical-theta index.

    Covers ``logD`` and the 8 ``PHYS_PARAM_NAMES``. Knots (the free-energy shape) are
    never addressable here — they are set via the ``knots`` argument, not as nuisances.
    """
    return {"logD": 0,
            **{nm: ctx.num_knots + 1 + i for i, nm in enumerate(PHYS_PARAM_NAMES)}}


def prior_range_violations(ctx: Ctx, theta) -> list:
    """Ground-truth params that fall OUTSIDE the model's prior range.

    Returns ``[(name, value, lo, hi), ...]`` (empty == fully in range). Checks ``logD``
    against ``logD_lims``, each spline knot ``g_i`` against its per-knot box
    (``center_i ± scale_i``), and the 8 physical params against their ``*_lims``. This
    is the data-generating ground truth: if it lies outside what the NPE was trained on,
    the simulated observation is out-of-distribution and the posterior is unreliable.
    """
    th = theta.detach().cpu().double().numpy() if torch.is_tensor(theta) else np.asarray(theta, float)
    out = []
    lo, hi = (float(x) for x in ctx.data_cfg.prior.logD_lims)
    if not (lo <= th[0] <= hi):
        out.append(("logD", float(th[0]), lo, hi))
    for i in range(ctx.num_knots):
        lo = float(ctx.prior_center[i] - ctx.prior_scale[i])
        hi = float(ctx.prior_center[i] + ctx.prior_scale[i])
        if not (lo <= th[1 + i] <= hi):
            out.append((f"g{i}", float(th[1 + i]), lo, hi))
    for j, nm in enumerate(PHYS_PARAM_NAMES):
        lo, hi = (float(x) for x in ctx.data_cfg.prior[f"{nm}_lims"])
        v = float(th[ctx.num_knots + 1 + j])
        if not (lo <= v <= hi):
            out.append((nm, v, lo, hi))
    return out


def make_case(ctx: Ctx, name, knots, **overrides) -> dict:
    """Build one ground-truth case -> dict(name, true_params, knots, in_support, violations).

    ``knots`` is the raw knot-force vector (len == num_knots). ``**overrides`` set
    non-knot canonical params (the fixed nuisance ground truth, e.g. ``kD=300``). The
    ground truth is checked against the model's prior range; out-of-range cases warn and
    set ``in_support=False`` (the caller may skip them) but never raise.
    """
    g = np.asarray(knots, dtype=np.float64)
    if g.shape != (ctx.num_knots,):
        raise ValueError(f"[{name}] expected {ctx.num_knots} knots, got {g.shape[0]}.")

    theta = default_canonical_from_config(ctx)
    theta[1:ctx.num_knots + 1] = torch.from_numpy(g)

    ci = canonical_index(ctx)
    for key, val in overrides.items():
        if key not in ci:
            raise ValueError(f"[{name}] unknown override {key!r}; valid: {list(ci)}")
        theta[ci[key]] = float(val)

    violations = prior_range_violations(ctx, theta)
    in_support = (len(violations) == 0)
    if violations:
        warnings.warn(f"[{name}] ground truth OUT OF PRIOR RANGE: " +
                      "; ".join(f"{n}={v:.4g} not in [{lo:.4g}, {hi:.4g}]"
                                for n, v, lo, hi in violations))
    return {"name": name, "true_params": theta, "knots": g,
            "in_support": in_support, "violations": violations}


def make_prior_sample_case(ctx: Ctx, name, theta: torch.Tensor) -> dict:
    """Wrap a full canonical theta drawn from the prior into a case dict."""
    theta = theta.detach().cpu().double()
    g = theta[1:ctx.num_knots + 1].numpy().copy()
    in_support = bool(within_support(ctx.prior_sim_cpu, theta.unsqueeze(0)).item())
    return {"name": name, "true_params": theta, "knots": g, "in_support": in_support}


# =============================================================================
# (a2) External controls: fixed nuisance values (--set) + explicit knots (--knots)
# =============================================================================
def parse_set(items) -> dict:
    """``["R0=6.0", "kD=400"]`` -> ``{"R0": 6.0, "kD": 400.0}`` (fixed nuisance GT)."""
    out = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"--set expects NAME=VALUE, got {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = float(v)
    return out


def parse_knots(items, num_knots) -> list:
    """``["myshape=v1,..,vN", "v1,..,vN"]`` -> ``[("myshape", arr), ("explicit_1", arr)]``.

    Each item is an optional ``NAME=`` prefix + a comma-separated knot-force vector of
    length ``num_knots`` (errors otherwise — explicit knots are deliberately
    model-specific). Unnamed shapes get the name ``explicit_<i>``.
    """
    out = []
    for i, it in enumerate(items or []):
        if "=" in it:
            name, body = it.split("=", 1)
            name = name.strip()
        else:
            name, body = f"explicit_{i}", it
        vals = [float(v) for v in body.split(",") if v != ""]
        if len(vals) != num_knots:
            raise ValueError(f"--knots {name!r}: expected {num_knots} values, got {len(vals)}.")
        out.append((name, np.asarray(vals, dtype=np.float64)))
    return out


def build_cases(ctx: Ctx, *, prior_center=True, n_prior_samples=0,
                knot_specs=(), fixed=None, seed=None) -> list:
    """Build the flat ground-truth case list for one call.

    prior_center (the prior-mean double well) + ``n_prior_samples`` seeded prior draws +
    each explicit ``(name, knots)`` in ``knot_specs``. The ``fixed`` nuisance values
    OVERRIDE the default/prior nuisance on every case (the data-generating ground truth);
    each case carries ``gt_nuisance=dict(fixed)``.
    """
    fixed = dict(fixed or {})
    specs = []  # (name, knots_array)
    if prior_center:
        specs.append(("prior_center", np.asarray(ctx.prior_center, dtype=np.float64)))
    if n_prior_samples > 0:
        if seed is not None:
            torch.manual_seed(int(seed))  # reproducible prior draws
        for i, theta in enumerate(ctx.prior_sim_cpu.sample((n_prior_samples,))):
            knots = theta.detach().cpu().double()[1:ctx.num_knots + 1].numpy().copy()
            specs.append((f"prior_sample_{i}", knots))
    specs.extend(knot_specs)

    cases = []
    for name, g in specs:
        case = make_case(ctx, name, knots=g, **fixed)
        case["gt_nuisance"] = dict(fixed)
        cases.append(case)
    return cases


# =============================================================================
# (b) Simulator -> (N_max, 4) model input  (module-level for multiprocessing fork)
# =============================================================================
_FRET_SIMULATOR = None  # set in main() before the pool is forked


def _init_worker(seed_base: int):
    np.random.seed((seed_base + os.getpid()) % (2 ** 31 - 1))


def _arrivals_to_x_single(ipt, chans, length):
    ipt = ipt.detach().cpu().numpy() if torch.is_tensor(ipt) else np.asarray(ipt)
    chans = chans.detach().cpu().numpy() if torch.is_tensor(chans) else np.asarray(chans)
    length = np.asarray([int(length)], dtype=np.int64)
    x = H5Dataset._arrivals_to_x(ipt[None, :], chans[None, :], length)
    return x[0]


def fret_simulator(params):
    """Run the Cython sim once; return a (N_max, 4) tensor or None on failure."""
    result = _FRET_SIMULATOR(params)
    if result is None:
        return None
    ipt, chans, length = result
    return _arrivals_to_x_single(ipt, chans, length)


def build_observation(pool, true_params, n_obs, n_pool, max_attempts=None):
    """Simulate via the pool in batches until ``n_obs`` successes -> (1, n_obs, N_max, 4).

    Every trace in the set shares the same ``true_params`` (the case's fixed ground
    truth). Sim-failure rates vary a lot between ground-truth shapes, so we top up in
    batches of ``n_pool`` (reusing the same pool — never re-forking) until enough traces
    succeed or ``max_attempts`` is hit.
    """
    max_attempts = max_attempts if max_attempts is not None else n_pool * 8
    results, attempts = [], 0
    while len(results) < n_obs and attempts < max_attempts:
        batch = pool.map(fret_simulator, [true_params] * n_pool)
        attempts += n_pool
        results.extend(r for r in batch if r is not None)
    if len(results) < n_obs:
        raise RuntimeError(f"Only {len(results)}/{n_obs} simulations succeeded in {attempts} "
                           f"attempts; increase --n-simulations-pool or --max-sim-attempts.")
    return torch.stack(results[:n_obs]).unsqueeze(0)  # (1, n_obs, N_max, 4)


# =============================================================================
# (c) Posterior inference
# =============================================================================
def to_canonical(ctx: Ctx, theta_inferred: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(ctx.canonical_dim, dtype=theta_inferred.dtype)
    out[ctx.inferred_idx] = theta_inferred.to(out.dtype)
    return out


def infer_posterior(ctx: Ctx, posterior, obs, args, show_progress=True) -> dict:
    """Sample the posterior, score the samples, find the MAP."""
    obs = obs.to(ctx.device)
    samples = posterior.sample(
        (args.num_posterior_samples,), x=obs,
        max_sampling_batch_size=args.posterior_batch_size,
        show_progress_bars=show_progress,
    ).cpu()

    samples_dev = samples.to(ctx.device)
    log_probs = []
    with torch.no_grad():
        rng = range(0, samples_dev.shape[0], args.log_prob_batch_size)
        for start in (tqdm(rng, desc="log_prob", leave=False) if show_progress else rng):
            chunk = samples_dev[start:start + args.log_prob_batch_size]
            # norm_posterior=False: skip the (rejection-based) leakage correction — we
            # only use these log-probs RELATIVELY (top-k seeding + HPD band quantile), and
            # the correction explodes when the posterior leaks out of the prior support.
            log_probs.append(posterior.log_prob(chunk, x=obs, norm_posterior=False).cpu())
    log_probs = torch.cat(log_probs)

    k = min(args.top_k, samples.shape[0])
    top_samples = samples[torch.topk(log_probs, k=k).indices]

    # MAP initialization. Default ("top_k") seeds a multi-start gradient ascent from the
    # top-log-prob samples. "posterior_mean" starts a SINGLE optimization at the posterior
    # sample mean (the mean of the in-support posterior samples). For a roughly unimodal
    # posterior this lands in the same basin far more cheaply (1 trajectory vs top_k).
    if args.map_init == "posterior_mean":
        init_samples = samples.mean(dim=0, keepdim=True)  # (1, theta_dim)
        if not bool(within_support(ctx.inferred_prior, init_samples.to(ctx.device)).item()):
            warnings.warn("posterior mean is outside the prior support; falling back to the "
                          "single best-log-prob sample for the MAP initialization.")
            init_samples = top_samples[:1]
        n_init = n_opt = 1
    else:  # "top_k"
        init_samples = top_samples
        n_init = n_opt = k

    map_estimate = map_with_cached_embedding(
        density_estimator=ctx.density_estimator, prior=ctx.inferred_prior, x=obs,
        num_iter=args.map_num_iter, num_init_samples=n_init, num_to_optimize=n_opt,
        learning_rate=args.map_learning_rate, init_samples=init_samples,
        show_progress_bars=show_progress,
    )
    return {"samples": samples, "log_probs": log_probs, "map_estimate": map_estimate}


# =============================================================================
# (d) Plotting & metrics
# =============================================================================
def _align(y, ref, mask=None):
    """Vertical shift (display only) so y best matches ref over ``mask``."""
    if mask is not None:
        return y + np.mean(ref[mask] - y[mask])
    return y + np.mean(ref - y)


def plot_fe_comparison(ctx: Ctx, result, ax=None, n_band=300, hpd_frac=0.68):
    """Ground-truth vs MAP free energy with a posterior credible band + FRET-E twin axis.

    Ground truth is the anchor (min-subtracted). The MAP curve is vertically aligned to the
    ground truth over the inner region [x_min, x_max], and the posterior band is aligned to
    that MAP curve. Alignment is display only (free energy is defined up to an additive
    constant) and never modifies the inferred parameters.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    samples = result["samples"]
    log_probs = result["log_probs"]
    inner = (ctx.x_eval >= ctx.x_min) & (ctx.x_eval <= ctx.x_max)

    # Ground truth is the anchor (min-subtracted to 0).
    _, y_true = ctx.eval_free_energy(params=result["true_params"], x_eval=ctx.x_eval)
    y_true = y_true - y_true.min()

    # MAP free energy aligned to the GROUND TRUTH over the inner region (display only).
    map_canonical = to_canonical(ctx, result["map_estimate"][0].cpu())
    _, y_map = ctx.eval_free_energy(params=map_canonical, x_eval=ctx.x_eval)
    y_map = _align(y_map, y_true, inner)

    # Posterior band: each sample aligned to the (true-aligned) MAP over the inner region.
    # Track the data range over the VISIBLE x-window while plotting, so the ΔG y-limits can
    # be scaled to show the whole landscape (ground truth + MAP + band) without clipping.
    xlim = (2.5, 10.0)
    vis = (ctx.x_eval >= xlim[0]) & (ctx.x_eval <= xlim[1])
    y_lo, y_hi = np.inf, -np.inf

    def _track(y):
        nonlocal y_lo, y_hi
        if vis.any():
            y_lo = min(y_lo, float(np.nanmin(y[vis])))
            y_hi = max(y_hi, float(np.nanmax(y[vis])))

    keep = log_probs >= torch.quantile(log_probs, 1.0 - hpd_frac)
    band = samples[keep]
    for i in range(min(n_band, band.shape[0])):
        _, y = ctx.eval_free_energy(params=to_canonical(ctx, band[i]), x_eval=ctx.x_eval)
        y = _align(y, y_map, inner)
        ax.plot(ctx.x_eval, y, color="gray", alpha=0.03)
        _track(y)

    ax.plot(ctx.x_eval, y_map, color="black", lw=2, label="MAP")
    ax.plot(ctx.x_eval, y_true, color="r", lw=2, label="Ground truth")
    _track(y_map)
    _track(y_true)

    ax.axvspan(ctx.x_min, ctx.x_max, color="0.95", zorder=0)
    ax.set_xlabel("Distance (nm)", fontsize=13)
    ax.set_ylabel(r"$\Delta G\ [k_B T]$", fontsize=13)
    # Scale the ΔG axis to the full visible landscape with a small margin (replaces the old
    # hard-coded (-3, 8) so tall/deep free-energy curves are always shown in full).
    if np.isfinite(y_lo) and np.isfinite(y_hi) and y_hi > y_lo:
        margin = 0.08 * (y_hi - y_lo)
        ax.set_ylim(y_lo - margin, y_hi + margin)
    else:
        ax.set_ylim(-3, 8)
    ax.set_xlim(*xlim)
    ax.set_title(result["name"])
    ax.grid(alpha=0.3)

    R0 = float(result["true_params"][ctx.num_knots + 1])
    r = np.clip(ctx.x_eval, 1e-9, None)
    E = 1.0 / (1.0 + (r / R0) ** 6)
    E[ctx.x_eval <= 0] = np.nan
    ax2 = ax.twinx()
    ax2.plot(ctx.x_eval, E, color="tab:blue", ls="--", lw=1.5)
    ax2.set_ylabel("FRET efficiency", color="tab:blue")
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax.legend(loc="upper right", fontsize=10)
    return ax


def plot_pairplot(ctx: Ctx, result):
    def _bounds(i):
        if i == 0:
            return list(ctx.data_cfg.prior.logD_lims)
        if 1 <= i <= ctx.num_knots:
            c = ctx.prior_center[i - 1]
            s = ctx.prior_scale[i - 1]
            return [float(c - s), float(c + s)]
        name = PHYS_PARAM_NAMES[i - ctx.num_knots - 1]
        return list(ctx.data_cfg.prior[f"{name}_lims"])

    limits = [_bounds(int(i)) for i in ctx.inferred_idx.tolist()]
    fig, _ = analysis.pairplot(
        result["samples"], figsize=(7, 7),
        points=[result["true_params"][ctx.inferred_idx].float(),
                result["map_estimate"][0].cpu()],
        points_offdiag={"markersize": 6}, points_colors=["r", "y"],
        limits=limits, labels=ctx.inferred_labels,
    )
    fig.suptitle(result["name"], y=1.02)
    return fig


def fe_rmse(ctx: Ctx, result) -> float:
    map_canonical = to_canonical(ctx, result["map_estimate"][0].cpu())
    _, y_map = ctx.eval_free_energy(params=map_canonical, x_eval=ctx.x_eval)
    _, y_true = ctx.eval_free_energy(params=result["true_params"], x_eval=ctx.x_eval)
    y_map = y_map - y_map.min()
    y_true = y_true - y_true.min()
    inner = (ctx.x_eval >= ctx.x_min) & (ctx.x_eval <= ctx.x_max)
    return float(np.sqrt(np.mean((y_map[inner] - y_true[inner]) ** 2)))


# =============================================================================
# Persistence
# =============================================================================
def save_case_results(ctx: Ctx, case, result, case_dir: Path, args):
    case_dir.mkdir(parents=True, exist_ok=True)

    # FE comparison (own figure).
    fig, ax = plt.subplots(figsize=(7, 4.2))
    plot_fe_comparison(ctx, result, ax=ax, n_band=args.n_band, hpd_frac=args.hpd_frac)
    fig.tight_layout(); fig.savefig(case_dir / "fe_comparison.png", dpi=150); plt.close(fig)

    # Pairplot.
    try:
        figp = plot_pairplot(ctx, result)
        figp.savefig(case_dir / "pairplot.png", dpi=150, bbox_inches="tight"); plt.close(figp)
    except Exception as e:  # pairplot can be finicky; don't lose the run over a plot
        warnings.warn(f"[{case['name']}] pairplot failed: {e}")

    # Numerical results.
    map_canonical = to_canonical(ctx, result["map_estimate"][0].cpu())
    _, U_map = ctx.eval_free_energy(params=map_canonical, x_eval=ctx.x_eval)
    _, U_true = ctx.eval_free_energy(params=result["true_params"], x_eval=ctx.x_eval)
    np.savez_compressed(
        case_dir / "posterior.npz",
        samples=result["samples"].numpy(),
        log_probs=result["log_probs"].numpy(),
        map_theta=result["map_estimate"][0].cpu().numpy(),
        map_logprob=float(result["map_estimate"][1]),
        true_params=case["true_params"].numpy(),
        knots=case["knots"],
        inferred_idx=ctx.inferred_idx.numpy(),
        x_eval=ctx.x_eval,
        U_map=U_map - U_map.min(),
        U_true=U_true - U_true.min(),
        fe_rmse=fe_rmse(ctx, result),
        in_support=case["in_support"],
    )
    if args.save_obs:
        np.save(case_dir / "observation.npy", result["obs"].numpy())


# =============================================================================
# (e) Engine helpers: simulator, observation-set size, estimator loading
# =============================================================================
_ESTIMATOR_CANDIDATES = ["npe_density_estimator.pkl", "last_npe_density_estimator.pkl",
                         "npe_set_estimator.pkl", "last_npe_set_estimator.pkl"]


def _make_simulator(ctx: Ctx):
    """Build the (picklable) per-trace simulator wrapper for this model's config."""
    mode = ctx.data_cfg.simulator.mode
    if mode == "equilibrium":
        return smFRET_simulator_wrapper_from_config(ctx.data_cfg_path)
    if mode == "binding":
        return binding_simulator_wrapper_from_config(ctx.data_cfg_path)
    raise ValueError(f"Unknown simulator.mode: {mode!r}")


def _resolve_n_obs(ctx: Ctx, args) -> int:
    """Observation-set size for one model: the NPE's trained set size unless overridden."""
    trained = int(ctx.data_cfg.data_generation.num_traces_per_set)
    n = args.n_observations if args.n_observations is not None else trained
    if n != trained:
        warnings.warn(f"n_observations={n} != trained num_traces_per_set={trained} for "
                      f"{Path(ctx.data_cfg_path).parent.name}; the posterior may reject the shape.")
    return n


def _load_estimator_and_posterior(ctx: Ctx, cfg_path, device) -> DirectPosterior:
    """Load the density estimator onto ``device`` and return a DirectPosterior. CUDA path."""
    cfg_path = Path(cfg_path)
    for fname in _ESTIMATOR_CANDIDATES:
        if (cfg_path / fname).exists():
            with open(cfg_path / fname, "rb") as f:
                ctx.density_estimator = pickle.load(f)
            print(f"[model] {cfg_path.name}: loaded {fname}")
            break
    else:
        raise FileNotFoundError(f"No density estimator pickle in {cfg_path}. "
                                f"Looked for: {_ESTIMATOR_CANDIDATES}")
    ctx.density_estimator = ctx.density_estimator.to(device).eval()
    ctx.inferred_prior = generate_split_priors_from_config(ctx.data_cfg_path, device=device)[0]
    return DirectPosterior(ctx.density_estimator, prior=ctx.inferred_prior)


def _safe(name: str) -> str:
    """Filesystem-safe case name (case names carry '=' and '.' from sweep tags)."""
    return "".join(ch if (ch.isalnum() or ch in "._=+-") else "_" for ch in str(name))


def _print_cases(label, cases):
    print(f"[cases] {label}")
    for c in cases:
        flag = "in range" if c["in_support"] else "OUT OF RANGE"
        gt = {k: v for k, v in c["gt_nuisance"].items()}
        line = f"   {c['name']:>30s}  [{flag:>12s}]  gt_nuisance={gt or '{}'}"
        if c.get("violations"):
            line += "  !! " + ", ".join(f"{n}={v:.4g}∉[{lo:.4g},{hi:.4g}]"
                                        for n, v, lo, hi in c["violations"])
        print(line)


def _result_from_case(ctx: Ctx, posterior, c, args, seed) -> dict:
    """Run posterior + MAP for one case (obs already in ``c['obs']``)."""
    if seed is not None:
        torch.manual_seed(seed)
    print(f"[infer] {c['name']}: posterior ({args.num_posterior_samples}) + MAP ...")
    res = infer_posterior(ctx, posterior, c["obs"], args)
    return {"name": c["name"], "true_params": c["true_params"], "obs": c["obs"].cpu(), **res}


# =============================================================================
# CLI
# =============================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Evaluate ONE trained NPE posterior under a given set of settings: "
                    "prior-centre + N prior-sample + explicit-knots ground-truth shapes, "
                    "with fixed ground-truth nuisance values via --set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Examples:\n"
               "  run_eval_posterior.py <model_dir>\n"
               "  run_eval_posterior.py <model_dir> --set kD=300 --out-dir results/kD_300\n"
               "  run_eval_posterior.py <model_dir> --knots \"5.5,-4.5,0.3,0,-0.3,4.5,-5.5\"\n"
               "A kD sweep or a multi-model comparison is just this script called repeatedly.",
    )
    p.add_argument("output_dir",
                   help="Trained-model output dir (holds data_config.yaml + *_estimator.pkl).")
    p.add_argument("--out-dir", default=None,
                   help="Where to write results (default: <output_dir>/eval/<timestamp>/).")
    # ground-truth shapes
    p.add_argument("--n-prior-samples", type=int, default=5,
                   help="Ground-truth cases drawn from the model's prior (seeded by --seed).")
    p.add_argument("--no-prior-center", action="store_true", help="Skip the prior-centre reference case.")
    p.add_argument("--knots", nargs="+", default=None, metavar="[NAME=]v1,..,vN",
                   help="Explicit ground-truth shape(s): a knot-force vector of length num_knots.")
    p.add_argument("--skip-out-of-range", action="store_true",
                   help="Drop cases whose ground truth falls outside the model's prior range "
                        "(instead of running them anyway).")
    # nuisance control
    p.add_argument("--set", nargs="+", default=None, metavar="PARAM=VALUE",
                   help="Fix ground-truth nuisance value(s) for the whole run, e.g. R0=6.0 kD=400.")
    # observation / simulation
    p.add_argument("--n-observations", type=int, default=None,
                   help="Traces per observation set (default: the model's trained num_traces_per_set).")
    p.add_argument("--n-simulations-pool", type=int, default=200, help="Simulation batch size per top-up round.")
    p.add_argument("--max-sim-attempts", type=int, default=None,
                   help="Max total sim attempts per case before giving up (default: 8x --n-simulations-pool).")
    p.add_argument("--n-sim-workers", type=int, default=24, help="Multiprocessing workers for the simulator.")
    # posterior
    p.add_argument("--num-posterior-samples", type=int, default=1000)
    p.add_argument("--posterior-batch-size", type=int, default=100)
    p.add_argument("--log-prob-batch-size", type=int, default=50)
    p.add_argument("--top-k", type=int, default=500, help="Top-log-prob samples seeding the MAP search.")
    # MAP
    p.add_argument("--map-num-iter", type=int, default=500)
    p.add_argument("--map-learning-rate", type=float, default=0.001)
    p.add_argument("--map-init", choices=["top_k", "posterior_mean"], default="top_k",
                   help="MAP gradient-ascent initialization: 'top_k' (default) multi-starts "
                        "from the top-log-prob samples; 'posterior_mean' starts a single "
                        "optimization at the posterior sample mean.")
    # plotting
    p.add_argument("--n-band", type=int, default=300, help="Posterior FE curves drawn in the credible band.")
    p.add_argument("--hpd-frac", type=float, default=0.68, help="Top log-prob fraction kept for the band.")
    # misc
    p.add_argument("--device", default=None, help="cuda|cpu (default: cuda if available).")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for posterior sampling + prior-sample draws (use -1 to stay stochastic). "
                        "NOTE: simulated photons are PID-seeded per worker and not bit-reproducible.")
    p.add_argument("--save-obs", action="store_true", help="Also save the raw observation tensors (large).")
    return p.parse_args(argv)


# =============================================================================
# Run
# =============================================================================
def main(argv=None):
    global _FRET_SIMULATOR
    args = parse_args(argv)
    cfg_path = Path(args.output_dir).resolve()
    if not (cfg_path / "data_config.yaml").exists():
        raise FileNotFoundError(f"{cfg_path}/data_config.yaml not found — is this a trained-model output dir?")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    seed = None if (args.seed is not None and args.seed < 0) else args.seed

    print(f"[setup] model dir: {cfg_path}\n[setup] device:    {device}")
    ctx = build_context(str(cfg_path), device)
    n_obs = _resolve_n_obs(ctx, args)
    print(f"[setup] mode={ctx.data_cfg.simulator.mode}  num_knots={ctx.num_knots}  "
          f"theta_dim={ctx.theta_dim}  inferred={ctx.inferred_labels}  n_obs={n_obs}")

    # Ground-truth cases + fixed nuisance settings.
    fixed = parse_set(args.set)
    knot_specs = parse_knots(args.knots, ctx.num_knots)
    if (not args.no_prior_center) and ctx.data_cfg.prior.center is None:
        warnings.warn("prior.center is null; the prior-centre case is a flat (zero-force) landscape.")
    cases = build_cases(ctx, prior_center=not args.no_prior_center,
                        n_prior_samples=args.n_prior_samples,
                        knot_specs=knot_specs, fixed=fixed, seed=seed)
    if not cases:
        raise ValueError("No cases to run (prior centre disabled, no prior samples, no --knots).")
    _print_cases(cfg_path.name, cases)

    if args.skip_out_of_range:
        kept = [c for c in cases if c["in_support"]]
        dropped = [c["name"] for c in cases if not c["in_support"]]
        if dropped:
            warnings.warn(f"--skip-out-of-range: dropping {len(dropped)} out-of-range case(s): {dropped}")
        cases = kept
        if not cases:
            raise ValueError("All cases were out of prior range; nothing to run.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (cfg_path / "eval" / timestamp)
    cases_dir = out_dir / "cases"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output] {out_dir}")

    # ── Phase A: simulate ALL observations, then close the pool BEFORE any CUDA work.
    print(f"[simulate] {len(cases)} cases x {n_obs} obs ({args.n_sim_workers} workers) ...")
    _FRET_SIMULATOR = _make_simulator(ctx)
    pool = Pool(processes=args.n_sim_workers, initializer=_init_worker, initargs=((seed or 0),))
    try:
        for c in cases:
            c["obs"] = build_observation(pool, c["true_params"], n_obs,
                                         args.n_simulations_pool, args.max_sim_attempts)
            print(f"   {c['name']}: obs {tuple(c['obs'].shape)}")
    finally:
        pool.close(); pool.join()
    print("[simulate] pool closed (no fork after this point).")

    # ── Phase B: CUDA — load estimator, infer + save per case.
    posterior = _load_estimator_and_posterior(ctx, cfg_path, device)
    results = []
    for c in cases:
        res = _result_from_case(ctx, posterior, c, args, seed)
        results.append(res)
        save_case_results(ctx, c, res, cases_dir / _safe(c["name"]), args)

    # Combined FE-comparison grid (montage of the per-case plots).
    ncols = min(3, len(results))
    nrows = math.ceil(len(results) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.2 * nrows), squeeze=False)
    for ax, res in zip(axes.ravel(), results):
        plot_fe_comparison(ctx, res, ax=ax, n_band=args.n_band, hpd_frac=args.hpd_frac)
    for ax in axes.ravel()[len(results):]:
        ax.set_visible(False)
    fig.tight_layout(); fig.savefig(out_dir / "fe_comparison_grid.png", dpi=150); plt.close(fig)

    # run_meta.json: the call's exact settings + per-case headline results.
    rmses = {res["name"]: fe_rmse(ctx, res) for res in results}
    meta = {
        "timestamp": timestamp, "model_dir": str(cfg_path), "data_config": ctx.data_cfg_path,
        "device": device, "mode": str(ctx.data_cfg.simulator.mode), "num_knots": ctx.num_knots,
        "n_obs": n_obs, "seed": seed, "inferred_labels": ctx.inferred_labels,
        "set": fixed, "n_prior_samples": args.n_prior_samples,
        "explicit_knots": [name for name, _ in knot_specs],
        "cases": [{"name": c["name"], "in_support": c["in_support"],
                   "violations": [list(v) for v in c.get("violations", [])],
                   "fe_rmse_kBT": rmses[c["name"]]} for c in cases],
        "args": vars(args),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))

    print("\n[summary]  case / FE RMSE [kBT]")
    for res in results:
        print(f"   {res['name']:>30s}  {rmses[res['name']]:.3f}")
    print(f"\nDone. Results in: {out_dir}")
    return out_dir


if __name__ == "__main__":
    main()
