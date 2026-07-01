"""Interactive free-energy profile builder for the by-hand posterior-eval notebook.

This module is **self-contained**: it does not import ``run_eval_posterior`` and does not
touch the trained-model pipeline. It operates on the objects the notebook already builds
(``eval_free_energy``, ``eval_force``, ``num_knots``, the knot prior box, the plotting
grid ``x_eval``) and produces the same ``(name, knots, overrides)`` data the notebook's
``make_case`` consumes. It also provides ``save_systematic_eval`` to persist results in
the exact ``outputs/systematic_eval`` layout written by ``run_eval_posterior.py`` (whose
``save_case_results`` / ``run_meta.json`` writer is the format spec we match).

Two construction modes, the two the user asked for:
  * **Direct force knots** — one slider per inner force-spline knot ``g_i`` (bounded to the
    per-knot prior box ``center_i ± scale_i``); you set every spline parameter yourself.
  * **Energy landscape** — height sliders define a target free energy ``U(x)`` (a cubic
    spline through the control points); the force-spline knots are then *fit* to it by a
    box-constrained least squares (``fit_force_knots``), the generalization of the old
    ``knots_from_wells`` to an arbitrary target shape.

The free energy is **affine in the inner knot forces** (``U(x) = A @ g + base``), so the
fit is a single ``scipy.optimize.lsq_linear`` solve bounded to the prior box — always in
support, exact for the piecewise-polynomial force spline.
"""
from __future__ import annotations

import json
import math
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import lsq_linear


# =============================================================================
# Core math: target U(x) -> best in-box force-spline knots
# =============================================================================
def _normalize_box(values, num_knots):
    """Broadcast a scalar / length-1 / per-knot prior-box edge to a (num_knots,) array."""
    v = np.asarray(values, dtype=float).ravel()
    if v.size == 1:
        return np.full(num_knots, float(v[0]))
    if v.size == num_knots:
        return v.copy()
    raise ValueError(f"expected a scalar or {num_knots} values, got {v.size}.")


def box_sigma_from_config(data_cfg):
    """Effective knot-box half-width multiplier for a data config.

    ``1.0`` for the (bounded) uniform knot prior; ``prior.n_sigma`` (default 3.0) for a
    Gaussian knot prior, whose ``scale`` is the marginal std and whose box is unbounded.
    Pass the result as ``box_sigma`` to ``ProfileBuilder`` / ``fit_force_knots``.
    """
    if str(data_cfg.prior.get("knot_prior", "independent")) == "gaussian":
        return float(data_cfg.prior.get("n_sigma", 3.0))
    return 1.0


def build_design(eval_free_energy, *, x_eval, x_min, x_max, num_knots, canonical_dim):
    """Affine model of the integrated free energy in the inner knot forces.

    Returns ``(A, base, inner)`` where ``U(x) = A @ g + base`` over ``x_eval`` (the
    boundary ramps are baked into ``base`` and the per-knot columns), and ``inner`` is the
    ``[x_min, x_max]`` mask. ``A`` depends only on the config, so build it once and reuse
    it for every live re-fit.
    """
    zero = np.zeros(canonical_dim)
    base = np.asarray(eval_free_energy(params=zero, x_eval=x_eval)[1], dtype=float)
    A = np.empty((np.asarray(x_eval).size, num_knots))
    for i in range(num_knots):
        e = zero.copy()
        e[1 + i] = 1.0
        A[:, i] = np.asarray(eval_free_energy(params=e, x_eval=x_eval)[1], dtype=float) - base
    inner = (np.asarray(x_eval) >= x_min) & (np.asarray(x_eval) <= x_max)
    return A, base, inner


def fit_force_knots(eval_free_energy, U_target, *, x_eval, x_min, x_max, num_knots,
                    canonical_dim, prior_center, prior_scale, box_sigma=1.0, design=None):
    """Best in-box force-spline knots whose integrated free energy matches ``U_target``.

    Solves a box-constrained least squares for the inner knot forces ``g`` bounded to the
    per-knot prior box (``center_i ± box_sigma·scale_i``), fitting only the inner region
    ``[x_min, x_max]`` (the boundary margins are fixed by the boundary condition). One free
    additive constant is fit too (free energy is defined up to a constant). Generalizes the
    notebook's ``knots_from_wells`` to an arbitrary target ``U(x)`` on ``x_eval``.

    ``box_sigma`` widens the box; use ``n_sigma`` under a Gaussian knot prior (where
    ``scale`` is the marginal std), or the default ``1.0`` for the uniform prior box.
    """
    if design is None:
        design = build_design(eval_free_energy, x_eval=x_eval, x_min=x_min, x_max=x_max,
                              num_knots=num_knots, canonical_dim=canonical_dim)
    A, base, inner = design
    n_in = int(inner.sum())
    A_aug = np.hstack([A[inner], np.ones((n_in, 1))])
    pc = _normalize_box(prior_center, num_knots)
    ps = float(box_sigma) * _normalize_box(prior_scale, num_knots)
    lb = np.r_[pc - ps, -np.inf]
    ub = np.r_[pc + ps, np.inf]
    target = (np.asarray(U_target, dtype=float) - base)[inner]
    sol = lsq_linear(A_aug, target, bounds=(lb, ub))
    return sol.x[:num_knots]


def energy_knots_to_target_U(x_control, heights, x_eval, x_min, x_max):
    """Target ``U(x)`` from energy control knots: a cubic spline through ``(x_control, heights)``.

    Evaluated on ``clip(x_eval, x_min, x_max)`` — only the inner region is fit by
    ``fit_force_knots``, so values outside ``[x_min, x_max]`` are clamped (irrelevant).
    """
    cs = CubicSpline(np.asarray(x_control, dtype=float), np.asarray(heights, dtype=float))
    return cs(np.clip(np.asarray(x_eval, dtype=float), x_min, x_max))


# =============================================================================
# Metrics + persistence in the systematic_eval layout
# =============================================================================
def _to_np(x):
    """numpy view of a numpy array or a (possibly CUDA) torch tensor."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_metrics(result, *, inferred_idx, to_canonical=None):
    """Per-case scalars matching ``run_eval_posterior.compute_metrics``.

    ``map_logprob`` (NPE log-density at the MAP), ``param_dist_l2`` (||theta_MAP -
    theta_true|| in the inferred subspace) and ``H_post`` (posterior differential entropy,
    -mean(log q)). ``result`` carries ``true_params`` (canonical), ``map_estimate``
    (theta_inferred, logprob) and ``log_probs``.
    """
    idx = np.asarray(inferred_idx).astype(int).ravel()
    map_theta = _to_np(result["map_estimate"][0]).astype(float).ravel()
    true_inf = _to_np(result["true_params"]).astype(float).ravel()[idx]
    log_probs = _to_np(result["log_probs"]).astype(float).ravel()
    return {
        "map_logprob": float(result["map_estimate"][1]),
        "param_dist_l2": float(np.linalg.norm(map_theta - true_inf)),
        "H_post": float(-log_probs.mean()),
    }


def _safe(name):
    """Filesystem-safe case name (mirrors run_eval_posterior._safe)."""
    return "".join(ch if (ch.isalnum() or ch in "._=+-") else "_" for ch in str(name))


def save_systematic_eval(out_dir, results, cases, *, meta, eval_free_energy, x_eval,
                         inferred_idx, to_canonical, num_knots, prior_entropy=None,
                         plot_fe_comparison=None, plot_pairplot=None, save_obs=False,
                         n_band=300, hpd_frac=0.68):
    """Write results in the exact ``outputs/systematic_eval`` layout.

    Per case ``cases/<name>/``: ``parameters.npz`` (samples, log_probs, inferred_idx,
    in_support), ``map_estimate.npz`` (map_theta, map_logprob, U_map min-subtracted),
    ``ground.npz`` (true_params, knots, x_eval, U_true min-subtracted), optional
    ``observation.npy``, and (if the plot callables are given) ``fe_comparison.png`` +
    ``pairplot.png``. Top level: ``fe_comparison_grid.png`` + ``run_meta.json``.

    Dependency-injected (``eval_free_energy``, ``to_canonical``, plot callables) so this
    file never imports the trained-model pipeline. ``meta`` is merged into ``run_meta.json``
    (the caller supplies model_dir / mode / n_obs / seed / inferred_labels / etc.).
    Returns the resolved ``out_dir`` Path.
    """
    import matplotlib.pyplot as plt  # local import: notebook owns the backend

    out_dir = Path(out_dir)
    cases_dir = out_dir / "cases"
    out_dir.mkdir(parents=True, exist_ok=True)

    by_name = {c["name"]: c for c in cases}
    inferred_idx_np = np.asarray(inferred_idx).astype(int).ravel()
    per_case_meta = []

    for res in results:
        case = by_name[res["name"]]
        case_dir = cases_dir / _safe(res["name"])
        case_dir.mkdir(parents=True, exist_ok=True)

        # FE comparison + pairplot (optional, via the notebook's own plot functions).
        if plot_fe_comparison is not None:
            fig, ax = plt.subplots(figsize=(7, 4.2))
            plot_fe_comparison(res, ax=ax, n_band=n_band, hpd_frac=hpd_frac)
            fig.tight_layout()
            fig.savefig(case_dir / "fe_comparison.png", dpi=150)
            plt.close(fig)
        if plot_pairplot is not None:
            try:
                figp = plot_pairplot(res)
                figp.savefig(case_dir / "pairplot.png", dpi=150, bbox_inches="tight")
                plt.close(figp)
            except Exception as e:  # don't lose the run over a finicky plot
                warnings.warn(f"[{res['name']}] pairplot failed: {e}")

        # Free-energy curves (min-subtracted), same as save_case_results.
        map_theta_np = _to_np(res["map_estimate"][0])
        map_canonical = to_canonical(map_theta_np)
        _, U_map = eval_free_energy(params=map_canonical, x_eval=x_eval)
        _, U_true = eval_free_energy(params=res["true_params"], x_eval=x_eval)
        U_map = np.asarray(U_map, dtype=float)
        U_true = np.asarray(U_true, dtype=float)

        np.savez_compressed(
            case_dir / "parameters.npz",
            samples=_to_np(res["samples"]),
            log_probs=_to_np(res["log_probs"]),
            inferred_idx=inferred_idx_np,
            in_support=case["in_support"],
        )
        np.savez_compressed(
            case_dir / "map_estimate.npz",
            map_theta=map_theta_np,
            map_logprob=float(res["map_estimate"][1]),
            U_map=U_map - U_map.min(),
        )
        np.savez_compressed(
            case_dir / "ground.npz",
            true_params=_to_np(case["true_params"]),
            knots=_to_np(case["knots"]),
            x_eval=np.asarray(x_eval),
            U_true=U_true - U_true.min(),
        )
        if save_obs and res.get("obs") is not None:
            np.save(case_dir / "observation.npy", _to_np(res["obs"]))

        m = compute_metrics(res, inferred_idx=inferred_idx_np) if case["in_support"] else {}
        per_case_meta.append({
            "name": res["name"], "in_support": bool(case["in_support"]),
            "map_logprob": m.get("map_logprob"),
            "param_dist_l2": m.get("param_dist_l2"),
            "H_post": m.get("H_post"),
        })

    # Combined FE-comparison grid (montage of the per-case plots).
    if plot_fe_comparison is not None and results:
        ncols = min(3, len(results))
        nrows = math.ceil(len(results) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.2 * nrows), squeeze=False)
        for ax, res in zip(axes.ravel(), results):
            plot_fe_comparison(res, ax=ax, n_band=n_band, hpd_frac=hpd_frac)
        for ax in axes.ravel()[len(results):]:
            ax.set_visible(False)
        fig.tight_layout()
        fig.savefig(out_dir / "fe_comparison_grid.png", dpi=150)
        plt.close(fig)

    run_meta = dict(meta)
    run_meta.setdefault("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_meta.setdefault("source", "fe_widget")
    if prior_entropy is not None:
        run_meta.setdefault("H_prior", float(prior_entropy))
    run_meta["cases"] = per_case_meta
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2, default=str))
    return out_dir


# =============================================================================
# The interactive widget
# =============================================================================
class ProfileBuilder:
    """ipywidgets UI to build free-energy profiles two ways, with a live preview.

    Build it from the notebook's globals, e.g.::

        builder = ProfileBuilder(
            eval_free_energy=eval_free_energy, eval_force=eval_force,
            num_knots=num_knots, canonical_dim=canonical_dim, x_eval=x_eval,
            x_min=x_min, x_max=x_max, knot_x=knot_x,
            prior_center=prior_center, prior_scale=prior_scale,
            param_lims={"logD": tuple(data_cfg.prior.logD_lims),
                        "R0":   tuple(data_cfg.prior.R0_lims)},
        )
        builder.display()

    Add ~5 profiles, then ``cases = builder.to_cases(make_case)`` feeds the notebook's
    existing pipeline. ``param_lims`` maps each exposed ground-truth override to its
    ``(lo, hi)`` bounds; pinned params (lo == hi) are skipped (left at the config value).
    """

    def __init__(self, *, eval_free_energy, eval_force, num_knots, canonical_dim,
                 x_eval, x_min, x_max, knot_x, prior_center, prior_scale,
                 param_lims=None, n_energy_knots=7, continuous_update=False,
                 box_sigma=1.0):
        import ipywidgets as widgets  # imported here so the module imports without a frontend

        self.W = widgets
        self.eval_free_energy = eval_free_energy
        self.eval_force = eval_force
        self.num_knots = int(num_knots)
        self.canonical_dim = int(canonical_dim)
        self.x_eval = np.asarray(x_eval, dtype=float)
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.knot_x = np.asarray(knot_x, dtype=float)
        self.prior_center = _normalize_box(prior_center, self.num_knots)
        self.prior_scale = _normalize_box(prior_scale, self.num_knots)
        # box_sigma widens the box to center +/- box_sigma*scale. Pass n_sigma under a
        # Gaussian knot prior (scale = marginal std); default 1.0 for the uniform box.
        self.box_sigma = float(box_sigma)
        self.lo = self.prior_center - self.box_sigma * self.prior_scale
        self.hi = self.prior_center + self.box_sigma * self.prior_scale
        self.param_lims = dict(param_lims or {})
        self.n_energy_knots = int(n_energy_knots)
        self._cu = bool(continuous_update)

        # Affine design matrix for the (energy-mode) force-knot fit — built once.
        self._design = build_design(
            eval_free_energy, x_eval=self.x_eval, x_min=self.x_min, x_max=self.x_max,
            num_knots=self.num_knots, canonical_dim=self.canonical_dim,
        )

        self.profiles = []          # list of {name, knots (np), overrides (dict)}
        self._fitted_knots = None   # last energy-mode fit
        self._build_widgets()

    # ---- profile -> case conversion -----------------------------------------
    def current_knots(self):
        """The active force-knot vector for the selected mode (np array, len num_knots)."""
        if self.mode_toggle.value == "Direct force knots":
            return np.array([s.value for s in self.knot_sliders], dtype=float)
        return np.asarray(self._fitted_knots, dtype=float)

    def current_overrides(self):
        """Ground-truth nuisance overrides (only the non-pinned, exposed params)."""
        return {name: float(sl.value) for name, sl in self.override_sliders.items()}

    def to_cases(self, make_case):
        """Turn the accumulated profiles into the notebook's ``case`` dicts.

        ``make_case`` is the notebook's function ``make_case(name, knots=..., **overrides)``.
        """
        return [make_case(p["name"], knots=p["knots"], **p["overrides"])
                for p in self.profiles]

    def profiles_overrides(self):
        """``{name: overrides}`` for stashing in run_meta.json."""
        return {p["name"]: p["overrides"] for p in self.profiles}

    # ---- in-support check (knots within the prior box) ----------------------
    def _knot_violations(self, knots):
        return np.where((knots < self.lo - 1e-9) | (knots > self.hi + 1e-9))[0]

    # ---- widget construction -------------------------------------------------
    def _build_widgets(self):
        W = self.W
        self.mode_toggle = W.ToggleButtons(
            options=["Direct force knots", "Energy landscape"],
            value="Direct force knots", description="Mode:",
            tooltips=["Edit every force-spline knot directly",
                      "Shape U(x); fit force knots to it"],
        )

        # Direct mode: one slider per inner force knot, bounded to the prior box.
        self.knot_sliders = []
        for i in range(self.num_knots):
            lo, hi, c = float(self.lo[i]), float(self.hi[i]), float(self.prior_center[i])
            step = max((hi - lo) / 200.0, 1e-3)
            self.knot_sliders.append(W.FloatSlider(
                value=c, min=lo, max=hi, step=step, description=f"g{i}",
                continuous_update=self._cu, readout_format=".2f",
                layout=W.Layout(width="92%"),
            ))
        self.knot_box = W.VBox(self.knot_sliders)

        # Energy mode: height sliders at fixed x control points across [x_min, x_max].
        self.x_control = np.linspace(self.x_min, self.x_max, self.n_energy_knots)
        self.energy_sliders = []
        for xc in self.x_control:
            self.energy_sliders.append(W.FloatSlider(
                value=0.0, min=-2.0, max=12.0, step=0.1,
                description=f"U(x={xc:.2f})", continuous_update=self._cu,
                readout_format=".1f", layout=W.Layout(width="92%"),
            ))
        self.energy_box = W.VBox(
            [W.HTML("<b>Target free energy</b> (k<sub>B</sub>T) at fixed x control points "
                    "&mdash; force knots are fit to this shape:")]
            + self.energy_sliders
        )
        self.energy_box.layout.display = "none"  # hidden until energy mode

        # Ground-truth nuisance overrides (skip pinned params; midpoint default).
        self.override_sliders = {}
        adv = []
        for name, (lo, hi) in self.param_lims.items():
            lo, hi = float(lo), float(hi)
            if hi <= lo:   # pinned -> not adjustable, left at the config value
                continue
            sl = W.FloatSlider(value=0.5 * (lo + hi), min=lo, max=hi,
                               step=max((hi - lo) / 200.0, 1e-4), description=name,
                               continuous_update=self._cu, readout_format=".3f",
                               layout=W.Layout(width="92%"))
            self.override_sliders[name] = sl
            adv.append(sl)
        self.override_box = W.Accordion(children=[W.VBox(adv)]) if adv else W.VBox([])
        if adv:
            self.override_box.set_title(0, "Ground-truth nuisance overrides (optional)")
            self.override_box.selected_index = None

        # Profile management.
        self.name_text = W.Text(value="profile_0", description="Name:",
                                layout=W.Layout(width="60%"))
        self.add_btn = W.Button(description="Add / Update", button_style="primary",
                                icon="plus")
        self.remove_btn = W.Button(description="Remove selected", button_style="warning",
                                   icon="trash")
        self.profile_list = W.Select(options=[], description="Profiles:", rows=6,
                                     layout=W.Layout(width="60%"))
        self.count_label = W.HTML()
        self.status_label = W.HTML()

        # Live preview as a PNG Image widget. Frontend-independent: we render the figure
        # to PNG bytes and set them here, so the preview never depends on a matplotlib
        # display formatter being active (which caused it to show as plain text, or to
        # render duplicated). The Image always shows exactly one picture.
        self.preview_img = W.Image(format="png", layout=W.Layout(width="100%"))

        # Wiring.
        self.mode_toggle.observe(self._on_mode_change, names="value")
        for sl in self.knot_sliders + self.energy_sliders:
            sl.observe(self._on_change, names="value")
        self.add_btn.on_click(self._on_add)
        self.remove_btn.on_click(self._on_remove)

        self._update()  # initial draw

    # ---- callbacks -----------------------------------------------------------
    def _on_mode_change(self, _):
        direct = self.mode_toggle.value == "Direct force knots"
        self.knot_box.layout.display = "" if direct else "none"
        self.energy_box.layout.display = "none" if direct else ""
        self._update()

    def _on_change(self, _):
        self._update()

    def _update(self):
        # In energy mode, refit force knots from the current target U(x).
        if self.mode_toggle.value == "Energy landscape":
            heights = np.array([s.value for s in self.energy_sliders], dtype=float)
            U_target = energy_knots_to_target_U(self.x_control, heights, self.x_eval,
                                                self.x_min, self.x_max)
            self._fitted_knots = fit_force_knots(
                self.eval_free_energy, U_target, x_eval=self.x_eval, x_min=self.x_min,
                x_max=self.x_max, num_knots=self.num_knots, canonical_dim=self.canonical_dim,
                prior_center=self.prior_center, prior_scale=self.prior_scale,
                box_sigma=self.box_sigma,
                design=self._design,
            )
        else:
            U_target = None

        knots = self.current_knots()
        viol = self._knot_violations(knots)
        in_support = len(viol) == 0
        if in_support:
            self.status_label.value = ("<span style='color:green'>&#10003; in support "
                                       "(all knots within the prior box)</span>")
        else:
            self.status_label.value = (f"<span style='color:#c00'>&#9888; out of support: "
                                       f"knots {viol.tolist()} outside the prior box</span>")
        self._draw(knots, U_target=U_target, in_support=in_support)

    def _draw(self, knots, U_target=None, in_support=True):
        # Render to PNG bytes (OO Figure + Agg canvas, never pyplot) and push them into the
        # Image widget. No display formatter, no clear_output, no pyplot registry -> the
        # preview always shows exactly one picture in any frontend.
        import io
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        theta = np.zeros(self.canonical_dim)
        theta[1:self.num_knots + 1] = knots
        _, U = self.eval_free_energy(params=theta, x_eval=self.x_eval)
        _, F = self.eval_force(params=theta, x_eval=self.x_eval)
        U = np.asarray(U, dtype=float)
        U = U - U.min()
        F = np.asarray(F, dtype=float)
        inner = (self.x_eval >= self.x_min) & (self.x_eval <= self.x_max)

        fig = Figure(figsize=(6.4, 3.8))
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        # Free energy (left).
        ax.plot(self.x_eval, U, color="tab:orange", lw=2, label="Realized U(x)")
        ax.scatter(self.knot_x, np.interp(self.knot_x, self.x_eval, U),
                   color="tab:orange", s=18, zorder=5)
        if U_target is not None:
            Ut = np.asarray(U_target, dtype=float)
            shift = np.mean((U - Ut)[inner])  # align target to realized over inner region
            ax.plot(self.x_eval[inner], (Ut + shift)[inner], color="tab:red", lw=1.5,
                    ls="--", label="Target U(x)")
            ax.scatter(self.x_control,
                       np.interp(self.x_control, self.x_eval, Ut + shift),
                       color="tab:red", s=20, marker="s", zorder=6)
        ax.axvspan(self.x_min, self.x_max, color="0.92", zorder=0)
        ax.set_xlabel("Distance (nm)")
        ax.set_ylabel(r"$\Delta G\ [k_B T]$", color="tab:orange")
        ax.tick_params(axis="y", labelcolor="tab:orange")
        ax.grid(alpha=0.3)
        U_in = U[inner]
        ax.set_ylim(-0.5, max(3.0, float(U_in.max()) * 1.25))
        ax.set_xlim(2.5, 10.0)
        ax.set_title("in support" if in_support else "OUT OF SUPPORT",
                     color="green" if in_support else "tab:red")
        ax.legend(loc="upper left", fontsize=8)

        # Force + prior box (right twin).
        axf = ax.twinx()
        axf.plot(self.x_eval, F, color="tab:blue", lw=1, alpha=0.4)
        axf.fill_between(self.knot_x, self.lo, self.hi, color="tab:blue", alpha=0.12,
                         step="mid")
        axf.scatter(self.knot_x, knots, color="tab:blue", s=20, zorder=5)
        axf.set_ylabel("Force (knots + prior box)", color="tab:blue")
        axf.tick_params(axis="y", labelcolor="tab:blue")
        axf.set_ylim(float(self.lo.min()) - 1.0, float(self.hi.max()) + 1.0)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        self.preview_img.value = buf.getvalue()

    def _refresh_list(self):
        self.profile_list.options = [p["name"] for p in self.profiles]
        self.count_label.value = f"<b>{len(self.profiles)}</b> profile(s) queued"

    def _on_add(self, _):
        name = self.name_text.value.strip() or f"profile_{len(self.profiles)}"
        entry = {"name": name, "knots": self.current_knots().copy(),
                 "overrides": self.current_overrides()}
        for i, p in enumerate(self.profiles):
            if p["name"] == name:       # update in place
                self.profiles[i] = entry
                break
        else:
            self.profiles.append(entry)
        self._refresh_list()
        # Auto-advance the default name so the next Add doesn't overwrite.
        self.name_text.value = f"profile_{len(self.profiles)}"

    def _on_remove(self, _):
        sel = self.profile_list.value
        if sel is None:
            return
        self.profiles = [p for p in self.profiles if p["name"] != sel]
        self._refresh_list()

    def display(self):
        from IPython.display import display
        W = self.W
        left = W.VBox([
            self.mode_toggle, self.knot_box, self.energy_box, self.override_box,
            W.HBox([self.name_text, self.add_btn]),
            W.HBox([self.profile_list, self.remove_btn]),
            self.count_label, self.status_label,
        ], layout=W.Layout(width="48%"))
        right = W.VBox([W.HTML("<b>Live preview</b>"), self.preview_img],
                       layout=W.Layout(width="50%"))
        self._refresh_list()
        display(W.HBox([left, right]))
