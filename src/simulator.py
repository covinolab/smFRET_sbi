from functools import partial
import torch
import numpy as np
from utils import make_spline_knots, _validate_bc
from smFRET_simulator import smFRET_simulator, binding_simulator
from omegaconf import OmegaConf


def _unpack_physics_params(params, num_knots):
    """Unpack physics parameters, deriving the conserved crosstalk diagonals.

    The prior samples only the leak fractions C_gr and C_rg; photon
    conservation on each emission channel fixes C_gg = 1 − C_gr and
    C_rr = 1 − C_rg.
    """
    C_gr = float(params[num_knots + 7])
    C_rg = float(params[num_knots + 8])
    return dict(
        R0=params[num_knots + 1],
        kD=params[num_knots + 2],
        k_gb=params[num_knots + 3],
        k_rb=params[num_knots + 4],
        eta_g=params[num_knots + 5],
        eta_r=params[num_knots + 6],
        C_gg=1.0 - C_gr,
        C_rr=1.0 - C_rg,
        C_gr=C_gr,
        C_rg=C_rg,
    )


def smFRET_simulator_wrapper(
                params,
                dt=5e-4,
                N_max_photons=8192,
                T=10000.0,
                x_init=(3.0, 7.0),
                x_min=0.0,
                x_max=10.0,
                num_knots=10,
                max_force=10.0,
                boundary_condition=("linear", "linear"),
                spline_type="cubic",
                num_boundary_knots=4,
                return_torch=False
    ):
        """Run the equilibrium simulator and return padded per-photon events.

        Returns
        -------
        (interphoton_times, channels, length) or None on failure.
            - interphoton_times : (N_max_photons,) float32, padded with 0
            - channels          : (N_max_photons,) uint8, 0=G/1=R, padded with 0
            - length            : python int, number of valid photons
            If `return_torch` is True, arrays come back as torch tensors.
        """

        x_init = np.random.uniform(low=x_init[0], high=x_init[1])
        x_knots, y_knots = make_spline_knots(
            params=params,
            x_min=x_min,
            x_max=x_max,
            num_knots=num_knots,
            max_force=max_force,
            boundary_condition=boundary_condition,
            num_boundary_knots=num_boundary_knots
        )

        phys = _unpack_physics_params(params, num_knots)
        G_times, R_times = smFRET_simulator(
                x0=x_init,
                Dx=10**params[0],
                x_knots=x_knots,
                y_knots=y_knots,
                **phys,
                T=T,
                N_max_photons=N_max_photons,
                dt=dt,
                spline_type=spline_type,
        )
        if G_times is None or R_times is None:
            return None

        n_g = G_times.shape[0]
        n_r = R_times.shape[0]
        n_total = n_g + n_r
        if n_total > N_max_photons:
            return None

        times = np.empty(n_total, dtype=np.float64)
        chans = np.empty(n_total, dtype=np.uint8)
        times[:n_g] = G_times
        chans[:n_g] = 0
        times[n_g:] = R_times
        chans[n_g:] = 1
        order = np.argsort(times, kind="stable")
        times = times[order]
        chans = chans[order]

        ipt = np.zeros(n_total, dtype=np.float32)
        if n_total > 1:
            ipt[1:] = (times[1:] - times[:-1]).astype(np.float32)

        ipt_padded = np.zeros(N_max_photons, dtype=np.float32)
        chans_padded = np.zeros(N_max_photons, dtype=np.uint8)
        ipt_padded[:n_total] = ipt
        chans_padded[:n_total] = chans

        if return_torch:
            return (
                torch.from_numpy(ipt_padded),
                torch.from_numpy(chans_padded),
                int(n_total),
            )
        return ipt_padded, chans_padded, int(n_total)


def smFRET_simulator_wrapper_from_config(cfg: str):
    """
    Generate a smFRET simulator wrapper from a configuration file or dict.

    Expected config structure:
        simulator:
            total_time: 5000.0
            N_max_photons: 8192
            time_step: 5e-4
            spline_type: "cubic"
            boundary_condition: "bound"
    """

    cfg = OmegaConf.load(cfg) if isinstance(cfg, str) else cfg

    total_time = cfg.simulator.total_time
    N_max_photons = int(cfg.simulator.N_max_photons)
    time_step = cfg.simulator.time_step
    spline_type = cfg.simulator.spline_type
    boundary_condition = _validate_bc(cfg.simulator.boundary_condition, allow_open=False)
    num_knots = cfg.prior.num_knots
    max_force = cfg.simulator.max_force
    x_min = cfg.simulator.x_min
    x_max = cfg.simulator.x_max
    num_boundary_knots = cfg.simulator.num_boundary_knots
    x_init = tuple(cfg.simulator.x_init)

    return partial(
        smFRET_simulator_wrapper,
        dt=time_step,
        N_max_photons=N_max_photons,
        T=total_time,
        x_init=x_init,
        x_min=x_min,
        x_max=x_max,
        num_knots=num_knots,
        max_force=max_force,
        boundary_condition=boundary_condition,
        spline_type=spline_type,
        num_boundary_knots=num_boundary_knots,
        return_torch=True
    )


def binding_simulator_wrapper(
                params,
                dt=5e-5,
                saving_freq=1,
                N_max_photons=8192,
                x_min=4.0,
                x_max=14.0,
                num_knots=5,
                max_force=30.0,
                boundary_condition=("linear", "linear"),
                spline_type="steffen",
                num_boundary_knots=3,
                L_min_bins=200,
                L_max_bins=400,
                epsilon=0.05,
                N_max=200000000,
                max_shoot_attempts=100,
                max_traj_retries=5,
                center_jitter_frac=0.10,
                n_committor_grid=4000,
                open_left=0,
                open_right=0,
                return_torch=False
    ):
        """Run the binding simulator and return padded per-photon events.

        Returns
        -------
        (interphoton_times, channels, length) or None on failure.
            - interphoton_times : (N_max_photons,) float32, padded with 0
            - channels          : (N_max_photons,) uint8, 0=G/1=R, padded with 0
            - length            : python int, number of valid photons
        """
        x_knots, y_knots = make_spline_knots(
            params=params,
            x_min=x_min,
            x_max=x_max,
            num_knots=num_knots,
            max_force=max_force,
            boundary_condition=boundary_condition,
            num_boundary_knots=num_boundary_knots
        )

        phys = _unpack_physics_params(params, num_knots)

        # Outer retry-on-None loop. Each .pyx call is a single shoot attempt
        # at the trace level: the per-leg `max_shoot_attempts` happens inside,
        # but RNG-dependent failures (no_rmid_crossing, trace_too_short,
        # shooting_failed, per-leg event budget overflow) still occur.
        G_times = R_times = None
        for _ in range(max(1, int(max_traj_retries))):
            G_times, R_times = binding_simulator(
                    Dx=10 ** params[0],
                    x_knots=x_knots,
                    y_knots=y_knots,
                    **phys,
                    saving_freq=saving_freq,
                    dt=dt,
                    x_min=x_min,
                    x_max=x_max,
                    L_min_bins=L_min_bins,
                    L_max_bins=L_max_bins,
                    N_max_photons=N_max_photons,
                    epsilon=epsilon,
                    N_max=N_max,
                    spline_type=spline_type,
                    open_left=open_left,
                    open_right=open_right,
                    n_committor_grid=n_committor_grid,
                    max_shoot_attempts=max_shoot_attempts,
                    center_jitter_frac=center_jitter_frac,
            )
            if G_times is not None and R_times is not None:
                break

        if G_times is None or R_times is None:
            return None

        n_g = G_times.shape[0]
        n_r = R_times.shape[0]
        n_total = n_g + n_r
        if n_total > N_max_photons:
            return None

        times = np.empty(n_total, dtype=np.float64)
        chans = np.empty(n_total, dtype=np.uint8)
        times[:n_g] = G_times
        chans[:n_g] = 0
        times[n_g:] = R_times
        chans[n_g:] = 1
        order = np.argsort(times, kind="stable")
        times = times[order]
        chans = chans[order]

        ipt = np.zeros(n_total, dtype=np.float32)
        if n_total > 1:
            ipt[1:] = (times[1:] - times[:-1]).astype(np.float32)

        ipt_padded = np.zeros(N_max_photons, dtype=np.float32)
        chans_padded = np.zeros(N_max_photons, dtype=np.uint8)
        ipt_padded[:n_total] = ipt
        chans_padded[:n_total] = chans

        if return_torch:
            return (
                torch.from_numpy(ipt_padded),
                torch.from_numpy(chans_padded),
                int(n_total),
            )
        return ipt_padded, chans_padded, int(n_total)


def binding_simulator_wrapper_from_config(cfg: str):
    """
    Generate a binding simulator wrapper from a configuration file or dict.
    """
    cfg = OmegaConf.load(cfg) if isinstance(cfg, str) else cfg

    binding_cfg = cfg.simulator.get("binding", {})
    left_bc, right_bc = _validate_bc(cfg.simulator.boundary_condition)

    return partial(
        binding_simulator_wrapper,
        dt=cfg.simulator.time_step,
        saving_freq=cfg.simulator.saving_frequency,
        N_max_photons=int(cfg.simulator.N_max_photons),
        x_min=cfg.simulator.x_min,
        x_max=cfg.simulator.x_max,
        num_knots=cfg.prior.num_knots,
        max_force=cfg.simulator.max_force,
        boundary_condition=(left_bc, right_bc),
        spline_type=cfg.simulator.spline_type,
        num_boundary_knots=cfg.simulator.num_boundary_knots,
        L_min_bins=binding_cfg.L_min_bins,
        L_max_bins=binding_cfg.L_max_bins,
        epsilon=binding_cfg.get("epsilon", 0.05),
        N_max=binding_cfg.get("N_max", 200000000),
        max_shoot_attempts=binding_cfg.get("max_shoot_attempts", 100),
        max_traj_retries=binding_cfg.get("max_traj_retries", 5),
        center_jitter_frac=binding_cfg.get("center_jitter_frac", 0.10),
        n_committor_grid=binding_cfg.get("n_committor_grid", 4000),
        open_left=int(left_bc == "open"),
        open_right=int(right_bc == "open"),
        return_torch=True
    )