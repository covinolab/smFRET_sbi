import cython
import numpy as np

from libc.math cimport sqrt, pow, exp
cimport numpy as cnp
cimport _gsl_spline as _gsl_spline
cimport gsl_random_numbers as grn
from libc.stdlib cimport malloc, free

# Required for direct numpy C-API calls (PyArray_ZEROS, etc.) in
# binding_simulator's photon-emission allocation.
cnp.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def smFRET_simulator(
        double x0,
        double Dx,
        cnp.ndarray[double] x_knots,
        cnp.ndarray[double] y_knots,
        double R0,
        double kD,
        double k_gb,
        double k_rb,
        double eta_g,
        double eta_r,
        double C_gg,
        double C_rr,
        double C_gr,
        double C_rg,
        double T,
        int N_max_photons,
        double dt,
        str spline_type='cubic'
        ):
    """
    Spline-based potential integrator with complete FRET photophysics.

    Implements:
    1. Langevin dynamics: dx = D*F*dt + sqrt(2*D*dt)*η
       where F = -dU/dx from cubic spline potential U(x)
    2. FRET efficiency: E(r) = R0^6 / (R0^6 + r^6)
    3. Cross-talk matrix for photon detection:
       - φ_g(r) = C_gg*(1-E(r)) + C_rg*E(r)
       - φ_r(r) = C_gr*(1-E(r)) + C_rr*E(r)
    4. Poisson photon statistics:
       - n_g ~ Poisson(η_g * (k_D * φ_g(r) + k_gb) * dt)
       - n_r ~ Poisson(η_r * (k_D * φ_r(r) + k_rb) * dt)
       Within each `dt` step, n_g/n_r arrival times are drawn uniformly on
       [i*dt, (i+1)*dt] — statistically exact under the homogeneous-Poisson
       assumption the simulator already commits to per step.

    Parameters
    ----------
    N_max_photons : int
        Per-channel budget for arrival-time arrays. If exceeded mid-trace,
        the simulation aborts and returns (None, None).
    dt : float
        Integration time step.
    T : float
        Total simulation time.

    Returns
    -------
    G_times : np.ndarray
        Green-channel photon arrival times (variable length ≤ N_max_photons).
    R_times : np.ndarray
        Red-channel photon arrival times (variable length ≤ N_max_photons).
    """

    # Initialize loop variables
    cdef long i
    cdef int N_knots = len(x_knots)
    cdef long N = <long>(T / dt)

    # Initialize random number generator
    cdef grn.gsl_rng_type * rng_type
    cdef grn.gsl_rng * rng
    cdef long seed = np.random.randint(low=1, high=2**63)

    grn.gsl_rng_env_setup()
    rng_type = grn.gsl_rng_default
    rng = grn.gsl_rng_alloc(rng_type)
    grn.gsl_rng_set(rng, seed)

    # Initialize spline knots as C arrays
    cdef double *x_k = <double *> malloc(N_knots * sizeof(double))
    cdef double *y_k = <double *> malloc(N_knots * sizeof(double))

    for i in range(N_knots):
        x_k[i] = x_knots[i]
        y_k[i] = y_knots[i]

    # Setup spline interpolation
    cdef _gsl_spline.gsl_interp_accel *acc
    acc = _gsl_spline.gsl_interp_accel_alloc()
    cdef _gsl_spline.gsl_spline *spline

    if spline_type == "cubic":
        spline = _gsl_spline.gsl_spline_alloc(_gsl_spline.gsl_interp_cspline, N_knots)
    elif spline_type == "steffen":
        spline = _gsl_spline.gsl_spline_alloc(_gsl_spline.gsl_interp_steffen, N_knots)
    elif spline_type == "akima":
        spline = _gsl_spline.gsl_spline_alloc(_gsl_spline.gsl_interp_akima, N_knots)
    else:
        raise ValueError(f"Unsupported spline type: {spline_type}")

    _gsl_spline.gsl_spline_init(spline, x_k, y_k, N_knots)

    # Langevin dynamics coefficients (overdamped limit, kT=1)
    cdef double Ax = Dx * dt          # Drift coefficient
    cdef double Bx = sqrt(2.0 * Ax)   # Diffusion coefficient

    # Pre-compute R0^6 for FRET efficiency
    cdef double R0_6 = pow(R0, 6.0)

    # State variables
    cdef double xold = x0
    cdef double xnew, Fx, spline_val
    cdef double r, r_6, E, phi_g, phi_r, lambda_g, lambda_r
    cdef unsigned int n_g, n_r, k_emit
    cdef int status = 0
    cdef bint budget_exceeded = False

    # Output arrays: per-photon arrival times (variable length, capped at N_max_photons).
    cdef cnp.ndarray[double] G_times = np.empty(N_max_photons, dtype=np.double)
    cdef cnp.ndarray[double] R_times = np.empty(N_max_photons, dtype=np.double)
    cdef long n_g_total = 0
    cdef long n_r_total = 0

    # Main integration loop
    for i in range(1, N):

        # Evaluate force directly from gradient spline: F = g(x)
        status = _gsl_spline.gsl_spline_eval_e(spline, xold, acc, &spline_val)
        if status != 0:
            break

        Fx = spline_val

        # Langevin integration step
        xnew = xold + Ax * Fx + Bx * grn.gsl_ran_gaussian_ziggurat(rng, 1.0)

        # Use position as distance r for FRET (ensure positive)
        r = xnew
        if r < 0.0:
            r = 0.0
        r_6 = pow(r, 6.0)

        # FRET efficiency: E(r) = R0^6 / (R0^6 + r^6)
        E = R0_6 / (R0_6 + r_6)

        # Cross-talk corrected detection probabilities
        phi_g = C_gg * (1.0 - E) + C_rg * E
        phi_r = C_gr * (1.0 - E) + C_rr * E

        # Poisson rates for photon emission
        lambda_g = eta_g * (kD * phi_g + k_gb) * dt
        lambda_r = eta_r * (kD * phi_r + k_rb) * dt

        # Sample photon counts from Poisson distribution
        n_g = grn.gsl_ran_poisson(rng, lambda_g)
        n_r = grn.gsl_ran_poisson(rng, lambda_r)

        # Emit arrival times for each photon in this step.
        # Times are uniform on [i*dt, (i+1)*dt] (homogeneous Poisson within dt).
        if n_g > 0:
            if n_g_total + <long>n_g > N_max_photons:
                budget_exceeded = True
                break
            for k_emit in range(n_g):
                G_times[n_g_total] = (<double>i + grn.gsl_rng_uniform(rng)) * dt
                n_g_total += 1
        if n_r > 0:
            if n_r_total + <long>n_r > N_max_photons:
                budget_exceeded = True
                break
            for k_emit in range(n_r):
                R_times[n_r_total] = (<double>i + grn.gsl_rng_uniform(rng)) * dt
                n_r_total += 1

        xold = xnew

    # Free allocated memory
    _gsl_spline.gsl_spline_free(spline)
    _gsl_spline.gsl_interp_accel_free(acc)
    grn.gsl_rng_free(rng)
    free(x_k)
    free(y_k)

    if status != 0 or budget_exceeded:
        return None, None

    return G_times[:n_g_total], R_times[:n_r_total]


# ============================================================================
# Binding simulator — committor-based two-way shooting
# ----------------------------------------------------------------------------
# Faithful Cython port of `binding_simulator_walkthrough_committor` from
# smFRET_sbi/analysis/simulator_walkthrough/walkthrough_simulator.py.
#
# Pipeline (no Python in the hot path):
#   Phase 0   — spline + RNG setup, Langevin coefficients
#   Phase 0.5 — analytic committor + shoot anchor x_TS
#   Phase 1   — pre-draw target window length
#   Phase 1.5 — pre-evaluate force on a uniform grid (O(1) lookup)
#   Phase 2   — two-way shooting with single-r_mid-crossing invariant
#   Phase 3   — concatenate as [B-reversed, A]
#   Phase 4   — locate r_mid crossing, crop with ±jitter
#   Phase 5   — emit Poisson photons
# ============================================================================


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int _compute_committor_c(
        _gsl_spline.gsl_spline *force_spline,
        _gsl_spline.gsl_interp_accel *acc,
        double x_min,
        double x_max,
        double knot_spacing,
        double x_knot_lo,
        double x_knot_hi,
        int open_left,
        int open_right,
        double epsilon,
        int n_grid,
        double *U_scratch,
        double *expU_scratch,
        double *cum_scratch,
        double *out_x_q_lo,
        double *out_x_q_hi,
        double *out_x_q_half,
        ):
    """Analytic 1D committor q(x) = ∫_a^x e^U / ∫_a^b e^U.

    U(x) = -∫_{x_lo}^x F ds, evaluated on a uniform grid spanning the spline
    domain (extended into open pads by 2·knot_spacing where the BC is open).
    Anchors `a`, `b` are leftmost / rightmost strict local minima of U on the
    inner domain (open side → grid edge instead). Cumulative-trapezoid with
    a U.max pivot for numerical stability.

    Outputs three level-set positions of q. Returns:
        0  ok
       -1  no strict local min (or no interior, or Z ≤ 0)
       -2  no second basin on a confining side
       -3  ε too tight (level sets collapse)
    """
    cdef double open_ext = 2.0 * knot_spacing
    cdef double x_lo = (x_min - open_ext) if open_left else (x_min - knot_spacing)
    cdef double x_hi = (x_max + open_ext) if open_right else (x_max + knot_spacing)
    if x_lo < x_knot_lo:
        x_lo = x_knot_lo
    if x_hi > x_knot_hi:
        x_hi = x_knot_hi
    if x_hi - x_lo <= 0.0 or n_grid < 4:
        return -1
    cdef double dx = (x_hi - x_lo) / (n_grid - 1)

    cdef int j
    cdef double xj
    cdef double val
    cdef int status
    cdef double U_min = 1e300

    # Build U(x) on the grid. Normalize to U.min = 0 (matches Python).
    for j in range(n_grid):
        xj = x_lo + j * dx
        status = _gsl_spline.gsl_spline_eval_integ_e(force_spline, x_lo, xj, acc, &val)
        if status != 0:
            return -1
        U_scratch[j] = -val
        if U_scratch[j] < U_min:
            U_min = U_scratch[j]
    for j in range(n_grid):
        U_scratch[j] -= U_min

    # Inner-domain index bounds.
    cdef int j_int_lo = <int>((x_min - x_lo) / dx)
    cdef int j_int_hi = <int>((x_max - x_lo) / dx)
    if j_int_lo < 0:
        j_int_lo = 0
    if j_int_hi >= n_grid:
        j_int_hi = n_grid - 1
    if j_int_hi - j_int_lo < 2:
        return -1

    # Pick anchors based on BC.
    cdef int j_A = -1
    cdef int j_B = -1
    cdef int k
    if open_left:
        j_A = 0
    else:
        for j in range(j_int_lo + 1, j_int_hi):
            if U_scratch[j] < U_scratch[j-1] and U_scratch[j] < U_scratch[j+1]:
                j_A = j
                break
        if j_A < 0:
            j_A = j_int_lo
            for k in range(j_int_lo + 1, j_int_hi + 1):
                if U_scratch[k] < U_scratch[j_A]:
                    j_A = k

    if open_right:
        j_B = n_grid - 1
    else:
        for j in range(j_int_hi - 1, j_int_lo, -1):
            if j > j_A and U_scratch[j] < U_scratch[j-1] and U_scratch[j] < U_scratch[j+1]:
                j_B = j
                break
        if j_B < 0:
            return -2

    if j_B <= j_A:
        return -1

    # Compute exp(U − U.max) on [j_A, j_B] for stability, then cumulative
    # trapezoid → Z*q.
    cdef int N_sub = j_B - j_A + 1
    cdef double U_max = U_scratch[j_A]
    for j in range(j_A + 1, j_B + 1):
        if U_scratch[j] > U_max:
            U_max = U_scratch[j]
    for j in range(N_sub):
        expU_scratch[j] = exp(U_scratch[j_A + j] - U_max)
    cum_scratch[0] = 0.0
    for j in range(1, N_sub):
        cum_scratch[j] = cum_scratch[j-1] + 0.5 * (expU_scratch[j-1] + expU_scratch[j]) * dx
    cdef double Z = cum_scratch[N_sub - 1]
    if Z <= 0.0:
        return -1

    cdef double eps_norm     = epsilon * Z
    cdef double eps_norm_hi  = (1.0 - epsilon) * Z
    cdef double half_norm    = 0.5 * Z

    # x_q_lo: largest x with q < ε  →  index `searchsorted(eps_norm) − 1`
    # clamped to [0, N_sub-1] (matches the Python max(0, ...) clamp).
    cdef int i_lo = N_sub - 1
    for j in range(N_sub):
        if cum_scratch[j] >= eps_norm:
            i_lo = j - 1
            break
    if i_lo < 0:
        i_lo = 0

    # x_q_hi: smallest x with q > 1−ε  →  searchsorted(eps_norm_hi), clamped.
    cdef int i_hi = N_sub - 1
    for j in range(N_sub):
        if cum_scratch[j] >= eps_norm_hi:
            i_hi = j
            break

    out_x_q_lo[0] = x_lo + (j_A + i_lo) * dx
    out_x_q_hi[0] = x_lo + (j_A + i_hi) * dx

    # x_q_half: linear interp at q = 0.5
    cdef double q_val, q_prev, t
    out_x_q_half[0] = x_lo + (j_A + (N_sub - 1)) * dx  # default: far end
    for j in range(1, N_sub):
        q_val  = cum_scratch[j]
        q_prev = cum_scratch[j-1]
        if q_prev <= half_norm and half_norm <= q_val:
            if q_val > q_prev:
                t = (half_norm - q_prev) / (q_val - q_prev)
            else:
                t = 0.0
            out_x_q_half[0] = x_lo + (j_A + j - 1) * dx + t * dx
            break

    if out_x_q_hi[0] <= out_x_q_lo[0]:
        return -3
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int _locate_TS_global_c(
        _gsl_spline.gsl_spline *force_spline,
        _gsl_spline.gsl_interp_accel *acc,
        double x_min,
        double x_max,
        double knot_spacing,
        double x_knot_lo,
        double x_knot_hi,
        int open_left,
        int open_right,
        double prominence_tol,
        int n_grid,
        double *U_scratch,
        double *out_x_TS,
        ):
    """Argmax of U(x) along the path from basin A to the open boundary.

    Mirrors the Python `locate_TS_global`: basin A is the strict local min
    closest to the *confining* side of the interior (fallback: interior
    argmin); the path search extends 2·knot_spacing past the open side.

    For [open, open] BC this defaults to the open_right behavior to match
    `_pick_shoot_point` in the walkthrough.
    """
    cdef int use_right = 0 if (open_left and not open_right) else 1
    cdef double open_ext = 2.0 * knot_spacing
    cdef double x_lo, x_hi
    if use_right:
        x_lo = x_min - knot_spacing
        x_hi = x_max + open_ext
    else:
        x_lo = x_min - open_ext
        x_hi = x_max + knot_spacing
    if x_lo < x_knot_lo:
        x_lo = x_knot_lo
    if x_hi > x_knot_hi:
        x_hi = x_knot_hi
    if x_hi - x_lo <= 0.0 or n_grid < 4:
        return -1

    cdef double dx = (x_hi - x_lo) / (n_grid - 1)
    cdef int j
    cdef double xj, val
    cdef int status
    cdef double U_min = 1e300

    for j in range(n_grid):
        xj = x_lo + j * dx
        status = _gsl_spline.gsl_spline_eval_integ_e(force_spline, x_lo, xj, acc, &val)
        if status != 0:
            return -1
        U_scratch[j] = -val
        if U_scratch[j] < U_min:
            U_min = U_scratch[j]
    for j in range(n_grid):
        U_scratch[j] -= U_min

    cdef int j_int_lo = <int>((x_min - x_lo) / dx)
    cdef int j_int_hi = <int>((x_max - x_lo) / dx)
    if j_int_lo < 0:
        j_int_lo = 0
    if j_int_hi >= n_grid:
        j_int_hi = n_grid - 1
    if j_int_hi - j_int_lo < 2:
        return -1

    # Basin A on the confining side; fallback to interior argmin.
    cdef int j_A = -1
    cdef int k
    if use_right:
        for j in range(j_int_lo + 1, j_int_hi):
            if U_scratch[j] < U_scratch[j-1] and U_scratch[j] < U_scratch[j+1]:
                j_A = j
                break
    else:
        for j in range(j_int_hi - 1, j_int_lo, -1):
            if U_scratch[j] < U_scratch[j-1] and U_scratch[j] < U_scratch[j+1]:
                j_A = j
                break
    if j_A < 0:
        j_A = j_int_lo
        for k in range(j_int_lo + 1, j_int_hi + 1):
            if U_scratch[k] < U_scratch[j_A]:
                j_A = k

    # Path: from j_A toward the open side, extended by open_ext.
    cdef int j_path_lo, j_path_hi
    if use_right:
        j_path_lo = j_A
        j_path_hi = <int>((x_max + open_ext - x_lo) / dx)
    else:
        j_path_lo = <int>((x_min - open_ext - x_lo) / dx)
        j_path_hi = j_A
    if j_path_lo < 0:
        j_path_lo = 0
    if j_path_hi >= n_grid:
        j_path_hi = n_grid - 1
    if j_path_hi - j_path_lo < 1:
        return -1

    cdef int j_TS = j_path_lo
    cdef double U_TS = U_scratch[j_path_lo]
    for j in range(j_path_lo + 1, j_path_hi + 1):
        if U_scratch[j] > U_TS:
            U_TS = U_scratch[j]
            j_TS = j

    if U_TS - U_scratch[j_A] < prominence_tol:
        return -1
    out_x_TS[0] = x_lo + j_TS * dx
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int _shoot_leg_committor_c(
        double *out,
        int target_len,
        double x_TS,
        double x_q_lo,
        double x_q_hi,
        double r_mid,
        int saving_freq,
        long raw_step_budget,
        double Ax,
        double Bx,
        _gsl_spline.gsl_spline *force_spline,
        _gsl_spline.gsl_interp_accel *acc_force,
        double x_knot_lo,
        double x_knot_hi,
        int open_left,
        int open_right,
        # FRET physics for per-fine-step photon emission:
        double R0_6,
        double kD,
        double k_gb,
        double k_rb,
        double eta_g,
        double eta_r,
        double C_gg,
        double C_rr,
        double C_gr,
        double C_rg,
        double dt,
        # Per-leg event buffers (leg-local fine-step time, seconds from leg start):
        double *G_times,
        double *R_times,
        int N_max_events_per_channel,
        grn.gsl_rng *rng,
        int *out_basin,
        int *out_q_set_bin,
        int *out_n_bins,
        int *out_n_g,
        int *out_n_r,
        ):
    """Hot loop: unbiased Langevin from x_TS with strict single-r_mid-crossing.

    Position trajectory is stored every `saving_freq` fine steps; photons are
    emitted at every fine step using the updated FRET rate, with within-step
    uniform jitter (statistically exact under the homogeneous-Poisson-per-dt
    assumption the simulator already makes).

    Termination priority:
      4 = event_budget_exceeded (photon buffer full mid-leg)
      3 = escape_fail        (spline boundary, confining side)
      2 = pre_commit_flicker (2nd r_mid crossing BEFORE q-set entry)
      0 = ok                 (post-commit r_mid recross → halt at last clean bin,
                              OR reached target_len after q-set entry)
      1 = no_q_set           (reached target_len, or step budget, without q-set)
    """
    cdef double x = x_TS
    cdef long step = 0
    cdef int bin_idx = 0
    cdef int q_set_reached = 0
    cdef int q_set_bin = -1
    cdef int basin_code = -1

    cdef int rmid_last_side = 0 if x_TS < r_mid else 1
    cdef int rmid_transitioned = 0
    cdef int rmid_cur

    cdef double F
    cdef int spline_status

    cdef int n_g_leg = 0
    cdef int n_r_leg = 0
    cdef double r_fret, r_6, E, phi_g, phi_r, lambda_g, lambda_r
    cdef double t_step_start
    cdef unsigned int n_g, n_r, kk

    out_basin[0] = -1
    out_q_set_bin[0] = -1
    out_n_bins[0] = 0
    out_n_g[0] = 0
    out_n_r[0] = 0

    while step < raw_step_budget:
        step += 1

        # Direct spline evaluation (cubic / Steffen / akima per setup).
        if x < x_knot_lo:
            if not open_left:
                out_n_bins[0] = bin_idx
                out_n_g[0] = n_g_leg
                out_n_r[0] = n_r_leg
                return 3
            F = 0.0
        elif x > x_knot_hi:
            if not open_right:
                out_n_bins[0] = bin_idx
                out_n_g[0] = n_g_leg
                out_n_r[0] = n_r_leg
                return 3
            F = 0.0
        else:
            spline_status = _gsl_spline.gsl_spline_eval_e(
                force_spline, x, acc_force, &F)
            if spline_status != 0:
                out_n_bins[0] = bin_idx
                out_n_g[0] = n_g_leg
                out_n_r[0] = n_r_leg
                return 3

        x = x + Ax * F + Bx * grn.gsl_ran_gaussian_ziggurat(rng, 1.0)

        # Per-fine-step photon emission (FRET rate evaluated at updated x).
        # Leg-local fine-step time of this update: [t_step_start, t_step_start + dt).
        r_fret = x if x > 0.0 else 0.0
        r_6 = pow(r_fret, 6.0)
        E = R0_6 / (R0_6 + r_6)
        phi_g = C_gg * (1.0 - E) + C_rg * E
        phi_r = C_gr * (1.0 - E) + C_rr * E
        lambda_g = eta_g * (kD * phi_g + k_gb) * dt
        lambda_r = eta_r * (kD * phi_r + k_rb) * dt
        n_g = grn.gsl_ran_poisson(rng, lambda_g)
        n_r = grn.gsl_ran_poisson(rng, lambda_r)
        if n_g > 0 or n_r > 0:
            t_step_start = <double>(step - 1) * dt
            if n_g > 0:
                if n_g_leg + <int>n_g > N_max_events_per_channel:
                    out_n_bins[0] = bin_idx
                    out_n_g[0] = n_g_leg
                    out_n_r[0] = n_r_leg
                    return 4
                for kk in range(n_g):
                    G_times[n_g_leg] = t_step_start + grn.gsl_rng_uniform(rng) * dt
                    n_g_leg += 1
            if n_r > 0:
                if n_r_leg + <int>n_r > N_max_events_per_channel:
                    out_n_bins[0] = bin_idx
                    out_n_g[0] = n_g_leg
                    out_n_r[0] = n_r_leg
                    return 4
                for kk in range(n_r):
                    R_times[n_r_leg] = t_step_start + grn.gsl_rng_uniform(rng) * dt
                    n_r_leg += 1

        if step % saving_freq == 0:
            if bin_idx >= target_len:
                out_n_bins[0] = bin_idx
                out_n_g[0] = n_g_leg
                out_n_r[0] = n_r_leg
                if q_set_reached:
                    out_basin[0] = basin_code
                    out_q_set_bin[0] = q_set_bin
                    return 0
                return 1

            out[bin_idx] = x

            # r_mid (E=0.5) crossing detection. First crossing = legitimate
            # transition. Any subsequent crossing pre-q-set is a flicker;
            # post-q-set is a recross (halt at last clean bin).
            rmid_cur = 0 if x < r_mid else 1
            if rmid_cur != rmid_last_side:
                if q_set_reached:
                    out_n_bins[0] = bin_idx
                    out_n_g[0] = n_g_leg
                    out_n_r[0] = n_r_leg
                    out_basin[0] = basin_code
                    out_q_set_bin[0] = q_set_bin
                    return 0
                if not rmid_transitioned:
                    rmid_transitioned = 1
                    rmid_last_side = rmid_cur
                else:
                    out_n_bins[0] = bin_idx
                    out_n_g[0] = n_g_leg
                    out_n_r[0] = n_r_leg
                    return 2

            # q-set entry → commitment & basin code.
            if not q_set_reached:
                if x <= x_q_lo:
                    q_set_reached = 1
                    q_set_bin = bin_idx
                    basin_code = 0
                elif x >= x_q_hi:
                    q_set_reached = 1
                    q_set_bin = bin_idx
                    basin_code = 1

            bin_idx += 1

    # Raw step budget exhausted.
    out_n_bins[0] = bin_idx
    out_n_g[0] = n_g_leg
    out_n_r[0] = n_r_leg
    if q_set_reached:
        out_basin[0] = basin_code
        out_q_set_bin[0] = q_set_bin
        return 0
    return 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int _find_rmid_crossing_c(
        double *concat,
        int n,
        double r_mid,
        int anchor_idx,
        int *out_t,
        int *out_half_max,
        ):
    """Locate the r_mid sign-change nearest `anchor_idx`. Mirrors the Python
    `_find_rmid_crossing`: argmin |sign_flip_idx − anchor_idx|, then `t` is
    the index of the first bin past the flip.
    """
    if n < 2:
        return -1
    cdef int j
    cdef int s_prev = 0 if concat[0] < r_mid else 1
    cdef int s_cur
    cdef int best_t = -1
    cdef int best_dist = n + 1
    cdef int diff_idx, dist

    for j in range(1, n):
        s_cur = 0 if concat[j] < r_mid else 1
        if s_cur != s_prev:
            diff_idx = j - 1
            dist = diff_idx - anchor_idx
            if dist < 0:
                dist = -dist
            if dist < best_dist:
                best_dist = dist
                best_t = j  # t = diff_idx + 1
        s_prev = s_cur

    if best_t < 0:
        return -1
    out_t[0] = best_t
    cdef int hl = best_t
    cdef int hr = n - best_t
    out_half_max[0] = hl if hl < hr else hr
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def binding_simulator(
        double Dx,
        cnp.ndarray[double] x_knots,
        cnp.ndarray[double] y_knots,
        double R0,
        double kD,
        double k_gb,
        double k_rb,
        double eta_g,
        double eta_r,
        double C_gg,
        double C_rr,
        double C_gr,
        double C_rg,
        int saving_freq,
        double dt,
        double x_min,
        double x_max,
        int L_min_bins,
        int L_max_bins,
        int N_max_photons,
        double epsilon=0.05,
        long N_max=200000000,
        str spline_type='steffen',
        int open_left=0,
        int open_right=0,
        int n_committor_grid=4000,
        int max_shoot_attempts=100,
        double center_jitter_frac=0.10,
        ):
    """Committor-based two-way shooting binding simulator (per-photon output).

    Places a shoot anchor x_TS, shoots two unbiased Langevin legs with
    per-fine-step photon emission, concatenates [B-reversed, A], crops a
    transition window around the r_mid crossing, and returns per-photon
    arrival times for the cropped window.

    Returns (G_times, R_times) on success — arrival times in seconds from the
    start of the cropped window. (None, None) on any failure (committor
    degenerate, no barrier, shooting timeout, no r_mid crossing, cropped
    window shorter than L_min_bins, or per-leg event buffer overflow).
    """
    cdef long i
    cdef int N_knots = len(x_knots)

    # All C-side allocations declared up front; freed in `finally`.
    cdef double *x_k = NULL
    cdef double *y_k = NULL
    cdef double *U_scratch = NULL
    cdef double *expU_scratch = NULL
    cdef double *cum_scratch = NULL
    cdef double *leg_fwd = NULL
    cdef double *leg_bwd = NULL
    cdef double *full_concat = NULL
    cdef double *G_fwd_buf = NULL
    cdef double *R_fwd_buf = NULL
    cdef double *G_bwd_buf = NULL
    cdef double *R_bwd_buf = NULL
    cdef _gsl_spline.gsl_spline *force_spline = NULL
    cdef _gsl_spline.gsl_interp_accel *acc_force = NULL
    cdef grn.gsl_rng *rng = NULL
    cdef grn.gsl_rng_type *rng_type

    cdef cnp.ndarray[double] G_times_out = None
    cdef cnp.ndarray[double] R_times_out = None

    # Scalars / state used across phases.
    cdef long seed
    cdef double knot_spacing, x_knot_lo, x_knot_hi
    cdef double Ax, Bx, bin_dt, r_mid, R0_6
    cdef double x_q_lo = 0.0, x_q_hi = 0.0, x_q_half = 0.0
    cdef double x_TS = 0.0
    cdef int committor_status, ts_status, crossing_status
    cdef int L_j_target, crop_half, jitter_max, target_half
    cdef int basin_fwd = -1, basin_bwd = -1
    cdef int q_set_bin_fwd = -1, q_set_bin_bwd = -1
    cdef int n_bins_fwd = 0, n_bins_bwd = 0
    cdef int n_g_fwd = 0, n_r_fwd = 0
    cdef int n_g_bwd = 0, n_r_bwd = 0
    cdef int shoot_status, attempt, target_basin_bwd
    cdef int accepted_fwd = 0, accepted_bwd = 0
    cdef double *leg_A
    cdef double *leg_B
    cdef double *G_A_buf
    cdef double *R_A_buf
    cdef double *G_B_buf
    cdef double *R_B_buf
    cdef int n_g_A_leg, n_r_A_leg, n_g_B_leg, n_r_B_leg
    cdef int leg_a_len, leg_b_len, n_concat
    cdef int t_rmid_full = 0, half_max = 0
    cdef int half_safe, eff_jitter_max, offset, L_j, start
    cdef double T_B, T_A, crop_t_start, crop_t_end
    cdef double t_event, t_event_concat
    cdef int n_g_kept, n_r_kept, k
    cdef cnp.npy_intp cnp_dims[1]

    try:
        # ===== Phase 0: spline + RNG =====
        seed = np.random.randint(low=1, high=2**63)
        grn.gsl_rng_env_setup()
        rng_type = grn.gsl_rng_default
        rng = grn.gsl_rng_alloc(rng_type)
        grn.gsl_rng_set(rng, seed)

        x_k = <double *> malloc(N_knots * sizeof(double))
        y_k = <double *> malloc(N_knots * sizeof(double))
        for i in range(N_knots):
            x_k[i] = x_knots[i]
            y_k[i] = y_knots[i]

        acc_force = _gsl_spline.gsl_interp_accel_alloc()
        if spline_type == "cubic":
            force_spline = _gsl_spline.gsl_spline_alloc(_gsl_spline.gsl_interp_cspline, N_knots)
        elif spline_type == "steffen":
            force_spline = _gsl_spline.gsl_spline_alloc(_gsl_spline.gsl_interp_steffen, N_knots)
        elif spline_type == "akima":
            force_spline = _gsl_spline.gsl_spline_alloc(_gsl_spline.gsl_interp_akima, N_knots)
        else:
            raise ValueError(f"Unsupported spline type: {spline_type}")
        _gsl_spline.gsl_spline_init(force_spline, x_k, y_k, N_knots)

        x_knot_lo = x_knots[0]
        x_knot_hi = x_knots[N_knots - 1]
        knot_spacing = x_knots[1] - x_knots[0]

        Ax = Dx * dt
        Bx = sqrt(2.0 * Ax)
        bin_dt = saving_freq * dt
        r_mid = R0
        R0_6 = pow(R0, 6.0)

        # ===== Phase 0.5: committor + shoot anchor =====
        U_scratch    = <double *> malloc(n_committor_grid * sizeof(double))
        expU_scratch = <double *> malloc(n_committor_grid * sizeof(double))
        cum_scratch  = <double *> malloc(n_committor_grid * sizeof(double))

        committor_status = _compute_committor_c(
            force_spline, acc_force,
            x_min, x_max, knot_spacing, x_knot_lo, x_knot_hi,
            open_left, open_right, epsilon, n_committor_grid,
            U_scratch, expU_scratch, cum_scratch,
            &x_q_lo, &x_q_hi, &x_q_half,
        )
        if committor_status != 0:
            return None, None

        if not open_left and not open_right:
            x_TS = x_q_half
        else:
            ts_status = _locate_TS_global_c(
                force_spline, acc_force,
                x_min, x_max, knot_spacing, x_knot_lo, x_knot_hi,
                open_left, open_right, 0.0, n_committor_grid,
                U_scratch, &x_TS,
            )
            if ts_status != 0:
                return None, None

        # ===== Phase 1: pre-draw target window length =====
        # gsl_rng_uniform_int(r, n) returns 0..n-1; shift to [L_min, L_max].
        L_j_target  = L_min_bins + <int>grn.gsl_rng_uniform_int(
            rng, <unsigned long int>(L_max_bins - L_min_bins + 1))
        crop_half   = L_j_target // 2
        jitter_max  = <int>(center_jitter_frac * crop_half)
        target_half = crop_half + jitter_max
        if target_half < 2:
            return None, None

        # ===== Phase 2: two-way shooting with per-fine-step photon emission =====
        leg_fwd = <double *> malloc(target_half * sizeof(double))
        leg_bwd = <double *> malloc(target_half * sizeof(double))
        # Per-leg event buffers (leg-local time, seconds from leg start).
        # Sized to N_max_photons per channel per leg: the wrapper picks
        # N_max_photons generously to cover both pre-crop and cropped events.
        G_fwd_buf = <double *> malloc(N_max_photons * sizeof(double))
        R_fwd_buf = <double *> malloc(N_max_photons * sizeof(double))
        G_bwd_buf = <double *> malloc(N_max_photons * sizeof(double))
        R_bwd_buf = <double *> malloc(N_max_photons * sizeof(double))

        for attempt in range(max_shoot_attempts):
            shoot_status = _shoot_leg_committor_c(
                leg_fwd, target_half, x_TS, x_q_lo, x_q_hi, r_mid,
                saving_freq, N_max, Ax, Bx,
                force_spline, acc_force, x_knot_lo, x_knot_hi,
                open_left, open_right,
                R0_6, kD, k_gb, k_rb, eta_g, eta_r,
                C_gg, C_rr, C_gr, C_rg, dt,
                G_fwd_buf, R_fwd_buf, N_max_photons,
                rng,
                &basin_fwd, &q_set_bin_fwd, &n_bins_fwd,
                &n_g_fwd, &n_r_fwd,
            )
            if shoot_status == 0:
                accepted_fwd = 1
                break
        if not accepted_fwd:
            return None, None

        target_basin_bwd = 1 - basin_fwd
        for attempt in range(max_shoot_attempts):
            shoot_status = _shoot_leg_committor_c(
                leg_bwd, target_half, x_TS, x_q_lo, x_q_hi, r_mid,
                saving_freq, N_max, Ax, Bx,
                force_spline, acc_force, x_knot_lo, x_knot_hi,
                open_left, open_right,
                R0_6, kD, k_gb, k_rb, eta_g, eta_r,
                C_gg, C_rr, C_gr, C_rg, dt,
                G_bwd_buf, R_bwd_buf, N_max_photons,
                rng,
                &basin_bwd, &q_set_bin_bwd, &n_bins_bwd,
                &n_g_bwd, &n_r_bwd,
            )
            if shoot_status == 0 and basin_bwd == target_basin_bwd:
                accepted_bwd = 1
                break
        if not accepted_bwd:
            return None, None

        # ===== Phase 3: identify A/B legs (positions and event buffers) =====
        if basin_fwd == 0:
            leg_A = leg_fwd
            leg_a_len = n_bins_fwd
            leg_B = leg_bwd
            leg_b_len = n_bins_bwd
            G_A_buf = G_fwd_buf
            R_A_buf = R_fwd_buf
            n_g_A_leg = n_g_fwd
            n_r_A_leg = n_r_fwd
            G_B_buf = G_bwd_buf
            R_B_buf = R_bwd_buf
            n_g_B_leg = n_g_bwd
            n_r_B_leg = n_r_bwd
        else:
            leg_A = leg_bwd
            leg_a_len = n_bins_bwd
            leg_B = leg_fwd
            leg_b_len = n_bins_fwd
            G_A_buf = G_bwd_buf
            R_A_buf = R_bwd_buf
            n_g_A_leg = n_g_bwd
            n_r_A_leg = n_r_bwd
            G_B_buf = G_fwd_buf
            R_B_buf = R_fwd_buf
            n_g_B_leg = n_g_fwd
            n_r_B_leg = n_r_fwd

        n_concat = leg_a_len + leg_b_len
        if n_concat < 2:
            return None, None
        full_concat = <double *> malloc(n_concat * sizeof(double))
        for i in range(leg_b_len):
            full_concat[i] = leg_B[leg_b_len - 1 - i]
        for i in range(leg_a_len):
            full_concat[leg_b_len + i] = leg_A[i]

        # ===== Phase 4: r_mid crossing + jittered crop (positions, in bins) =====
        crossing_status = _find_rmid_crossing_c(
            full_concat, n_concat, r_mid, leg_b_len,
            &t_rmid_full, &half_max,
        )
        if crossing_status != 0:
            return None, None

        half_safe = crop_half if crop_half < half_max else half_max
        eff_jitter_max = half_max - half_safe
        if jitter_max < eff_jitter_max:
            eff_jitter_max = jitter_max
        if eff_jitter_max < 0:
            eff_jitter_max = 0
        if eff_jitter_max > 0:
            offset = <int>grn.gsl_rng_uniform_int(
                rng, <unsigned long int>(2 * eff_jitter_max + 1)) - eff_jitter_max
        else:
            offset = 0

        L_j = 2 * half_safe
        if L_j < L_min_bins:
            return None, None

        start = t_rmid_full - half_safe + offset
        if start < 0:
            start = 0
        if start + L_j > n_concat:
            start = n_concat - L_j
        if start < 0:
            return None, None

        # ===== Phase 5: build per-photon event arrays from per-leg buffers =====
        # Concatenated time frame:
        #   t in [0, T_B]            ←  reversed B-leg, mapped via t_concat = T_B - t_leg
        #   t in [T_B, T_B + T_A]    ←  forward A-leg, mapped via t_concat = T_B + t_leg
        # where T_B = leg_b_len * bin_dt, T_A = leg_a_len * bin_dt (the saved
        # position trajectory's implicit time span). Events beyond a leg's
        # saved time span are dropped (these come from fine steps the loop
        # took between the last save and the termination check).
        # Crop window in concat time:
        #   crop_t_start = start * bin_dt
        #   crop_t_end   = (start + L_j) * bin_dt
        # Output time = t_concat - crop_t_start.
        T_B = <double>leg_b_len * bin_dt
        T_A = <double>leg_a_len * bin_dt
        crop_t_start = <double>start * bin_dt
        crop_t_end   = <double>(start + L_j) * bin_dt

        # First pass: count cropped events per channel to size the output arrays.
        n_g_kept = 0
        for k in range(n_g_B_leg):
            t_event = G_B_buf[k]
            if t_event > T_B:
                continue
            t_event_concat = T_B - t_event
            if t_event_concat >= crop_t_start and t_event_concat < crop_t_end:
                n_g_kept += 1
        for k in range(n_g_A_leg):
            t_event = G_A_buf[k]
            if t_event > T_A:
                continue
            t_event_concat = T_B + t_event
            if t_event_concat >= crop_t_start and t_event_concat < crop_t_end:
                n_g_kept += 1
        n_r_kept = 0
        for k in range(n_r_B_leg):
            t_event = R_B_buf[k]
            if t_event > T_B:
                continue
            t_event_concat = T_B - t_event
            if t_event_concat >= crop_t_start and t_event_concat < crop_t_end:
                n_r_kept += 1
        for k in range(n_r_A_leg):
            t_event = R_A_buf[k]
            if t_event > T_A:
                continue
            t_event_concat = T_B + t_event
            if t_event_concat >= crop_t_start and t_event_concat < crop_t_end:
                n_r_kept += 1

        # Second pass: fill output numpy arrays.
        cnp_dims[0] = n_g_kept
        G_times_out = <cnp.ndarray>cnp.PyArray_EMPTY(1, cnp_dims, cnp.NPY_DOUBLE, 0)
        cnp_dims[0] = n_r_kept
        R_times_out = <cnp.ndarray>cnp.PyArray_EMPTY(1, cnp_dims, cnp.NPY_DOUBLE, 0)

        n_g_kept = 0
        for k in range(n_g_B_leg):
            t_event = G_B_buf[k]
            if t_event > T_B:
                continue
            t_event_concat = T_B - t_event
            if t_event_concat >= crop_t_start and t_event_concat < crop_t_end:
                G_times_out[n_g_kept] = t_event_concat - crop_t_start
                n_g_kept += 1
        for k in range(n_g_A_leg):
            t_event = G_A_buf[k]
            if t_event > T_A:
                continue
            t_event_concat = T_B + t_event
            if t_event_concat >= crop_t_start and t_event_concat < crop_t_end:
                G_times_out[n_g_kept] = t_event_concat - crop_t_start
                n_g_kept += 1
        n_r_kept = 0
        for k in range(n_r_B_leg):
            t_event = R_B_buf[k]
            if t_event > T_B:
                continue
            t_event_concat = T_B - t_event
            if t_event_concat >= crop_t_start and t_event_concat < crop_t_end:
                R_times_out[n_r_kept] = t_event_concat - crop_t_start
                n_r_kept += 1
        for k in range(n_r_A_leg):
            t_event = R_A_buf[k]
            if t_event > T_A:
                continue
            t_event_concat = T_B + t_event
            if t_event_concat >= crop_t_start and t_event_concat < crop_t_end:
                R_times_out[n_r_kept] = t_event_concat - crop_t_start
                n_r_kept += 1

        return G_times_out, R_times_out

    finally:
        free(x_k)
        free(y_k)
        free(U_scratch)
        free(expU_scratch)
        free(cum_scratch)
        free(leg_fwd)
        free(leg_bwd)
        free(full_concat)
        free(G_fwd_buf)
        free(R_fwd_buf)
        free(G_bwd_buf)
        free(R_bwd_buf)
        if force_spline != NULL:
            _gsl_spline.gsl_spline_free(force_spline)
        if acc_force != NULL:
            _gsl_spline.gsl_interp_accel_free(acc_force)
        if rng != NULL:
            grn.gsl_rng_free(rng)
