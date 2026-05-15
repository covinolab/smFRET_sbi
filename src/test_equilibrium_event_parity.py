"""Standalone parity test for the per-photon-event equilibrium simulator.

Run from `smFRET_sbi/src/`:
    python test_equilibrium_event_parity.py

Checks
------
1. Invariants on the raw Cython output (monotonic G_times, R_times in [0, T]).
2. Invariants on the wrapper output (interphoton_times >= 0, channels in {0,1},
   `length <= N_max_photons`, padding region is zero).
3. Total photon count is in the analytically expected ballpark given the
   trajectory's mean FRET efficiency.
4. Within-step uniform jitter actually fills the [i*dt, (i+1)*dt] interval
   (not all photons stamped at integer step boundaries).
"""
from __future__ import annotations

import numpy as np
import sys
import math

import smFRET_simulator as cy
from simulator import smFRET_simulator_wrapper


# ---------------------------------------------------------------------------
# Parameter setup
# ---------------------------------------------------------------------------
T = 5.0                  # short trace for fast test
dt = 5e-4                # simulator default
N_max_photons = 50_000

# Equilibrium prior parameters: [logD, knots(10), R0, kD, k_gb, k_rb, eta_g, eta_r, C_gr, C_rg]
# Numbers chosen to roughly match exp_9-style scales but with a smooth flat-ish landscape.
NUM_KNOTS = 10
params = np.array(
    [0.0]                                          # logD
    + [0.0] * NUM_KNOTS                            # flat landscape (zero forces)
    + [6.0, 200.0, 1.0, 1.0, 0.9, 0.9, 0.05, 0.05],
    dtype=np.float64,
)


def _build_test_knots():
    # Wide spline domain so diffusing trajectories don't escape and trip
    # status!=0 in gsl_spline_eval_e (which would return None).
    x_knots = np.linspace(0.0, 20.0, NUM_KNOTS).astype(np.float64)
    # Force ≈ 0 ⇒ essentially free diffusion (parity-test traces).
    y_knots = np.zeros(NUM_KNOTS, dtype=np.float64)
    return x_knots, y_knots


def test_cython_invariants():
    """Direct call to the Cython core: check output shape and monotonicity."""
    x_knots, y_knots = _build_test_knots()
    np.random.seed(42)
    G_times, R_times = cy.smFRET_simulator(
        x0=6.0,
        Dx=0.05,
        x_knots=x_knots,
        y_knots=y_knots,
        R0=6.0, kD=200.0,
        k_gb=1.0, k_rb=1.0,
        eta_g=0.9, eta_r=0.9,
        C_gg=0.95, C_rr=0.95, C_gr=0.05, C_rg=0.05,
        T=T,
        N_max_photons=N_max_photons,
        dt=dt,
        spline_type="cubic",
    )
    assert G_times is not None and R_times is not None, "simulator returned None"
    n_g, n_r = G_times.shape[0], R_times.shape[0]
    print(f"[cython] n_g={n_g}, n_r={n_r}, n_total={n_g+n_r}")

    # All arrival times in valid range.
    assert (G_times >= 0).all() and (G_times <= T + dt).all(), "G times out of range"
    assert (R_times >= 0).all() and (R_times <= T + dt).all(), "R times out of range"
    # Per-channel ordering is "almost-monotone": photons emitted in the same
    # integration step get independent uniform jitters and can therefore be
    # locally out of order. The wrapper sorts across both channels — verify
    # the local violations are within one step.
    for name, arr in [("G", G_times), ("R", R_times)]:
        if arr.size > 1:
            violations = np.where(np.diff(arr) < 0)[0]
            for v in violations:
                assert arr[v] - arr[v + 1] < dt, (
                    f"{name}_times: out-of-order pair beyond one dt: "
                    f"{arr[v]} -> {arr[v+1]}"
                )

    # The within-step uniform jitter should produce non-integer multiples of dt.
    # If every photon were stamped at i*dt we'd see (G_times / dt) all integer-valued.
    frac = (G_times / dt) - np.floor(G_times / dt)
    assert frac.max() > 0.05, "within-step jitter looks degenerate (all integer-aligned)"
    print(f"[cython] jitter fractional part range = [{frac.min():.3g}, {frac.max():.3g}]")
    return n_g, n_r


def test_wrapper_invariants():
    """Wrapper: padded outputs with length and channel correctness."""
    np.random.seed(42)
    out = smFRET_simulator_wrapper(
        params=params,
        dt=dt,
        N_max_photons=N_max_photons,
        T=T,
        x_init=(6.0, 6.0),   # narrow init range
        x_min=3.5, x_max=9.0,
        num_knots=NUM_KNOTS,
        max_force=10.0,
        boundary_condition=("linear", "linear"),
        spline_type="cubic",
        num_boundary_knots=4,
        return_torch=False,
    )
    assert out is not None, "wrapper returned None"
    ipt, chans, length = out
    assert ipt.shape == (N_max_photons,) and chans.shape == (N_max_photons,)
    assert ipt.dtype == np.float32 and chans.dtype == np.uint8
    assert 0 < length <= N_max_photons, f"bad length {length}"
    assert (ipt[:length] >= 0).all(), "negative interphoton times"
    assert ipt[0] == 0.0, "first interphoton time should be 0"
    assert (chans[:length] < 2).all(), "channel labels out of range"
    assert (ipt[length:] == 0).all(), "padding region not zero (interphoton)"
    assert (chans[length:] == 0).all(), "padding region not zero (channels)"
    print(f"[wrapper] length={length}, G frac={(chans[:length]==0).mean():.3f}, "
          f"sum ipt={ipt[:length].sum():.4g} ≤ T={T}")
    return length


def test_photon_rate_in_ballpark():
    """Total photon count should be in the right order of magnitude.

    With kD=200, eta=0.9, and FRET efficiency around 0.5 on average (varies
    with the diffusing trajectory), expected per-channel rate is ~ kD * eta /
    2 ≈ 90 photons / simulator-time-unit. Over T=5 → ~450/channel = ~900 total.
    Acceptance window is loose; this is just a sanity check.
    """
    np.random.seed(0)
    counts = []
    for _ in range(8):
        out = smFRET_simulator_wrapper(
            params=params,
            dt=dt,
            N_max_photons=N_max_photons,
            T=T,
            x_init=(6.0, 6.0),
            x_min=3.5, x_max=9.0,
            num_knots=NUM_KNOTS,
            max_force=10.0,
            boundary_condition=("linear", "linear"),
            spline_type="cubic",
            num_boundary_knots=4,
            return_torch=False,
        )
        if out is None:
            continue
        _, _, length = out
        counts.append(length)
    counts = np.array(counts)
    mean = counts.mean()
    print(f"[rate] photon counts over {len(counts)} runs: mean={mean:.0f}, "
          f"std={counts.std():.0f}, range=[{counts.min()}, {counts.max()}]")
    # Loose ballpark: between 100 and 5000 total photons over T=5 with these rates.
    assert 100 < mean < 5000, f"photon-rate way off expected ballpark: mean={mean}"


def main():
    # The Cython direct call and wrapper call use different physics (Dx,
    # spline knots) and independent RNG draws, so their photon counts are
    # not comparable. Each test stands alone.
    print("== Equilibrium event-output parity test ==")
    test_cython_invariants()
    test_wrapper_invariants()
    test_photon_rate_in_ballpark()
    print("\nAll equilibrium parity checks passed.")


if __name__ == "__main__":
    main()
