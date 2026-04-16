import cython
import numpy as np

from libc.math cimport sqrt, pow, exp
cimport numpy as cnp
cimport _gsl_spline as _gsl_spline
cimport gsl_random_numbers as grn
from libc.stdlib cimport malloc, free
from libc.time cimport time


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
        int saving_freq,
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

    Parameters
    ----------
    x0 : float
        Initial position of the particle.
    Dx : float
        Diffusion coefficient D (related to friction by D = kT/γ).
    x_knots : np.ndarray
        x-positions of spline knots defining the free energy landscape.
    y_knots : np.ndarray
        y-positions (free energy values) of spline knots.
    R0 : float
        Förster radius.
    kD : float
        Donor excitation rate.
    k_gb : float
        Green photon background rate.
    k_rb : float
        Red photon background rate.
    eta_g : float
        Green detector efficiency.
    eta_r : float
        Red detector efficiency.
    C_gg : float
        Probability green photon detected by green detector.
    C_rr : float
        Probability red photon detected by red detector.
    C_gr : float
        Probability green photon detected by red detector (cross-talk).
    C_rg : float
        Probability red photon detected by green detector (cross-talk).
    T : float
        Total simulation time.
    saving_freq : int
        Number of integration steps per saved time bin.
    dt : float
        Integration time step.

    Returns
    -------
    kG_photons : np.ndarray
        Green channel photon counts per time bin.
    kR_photons : np.ndarray
        Red channel photon counts per time bin.
    """

    # Initialize loop variables
    cdef long i
    cdef int N_knots = len(x_knots)
    cdef long N = <long>(T / dt)
    cdef long n_save = N // saving_freq

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
    cdef unsigned int n_g, n_r
    cdef int status = 0
    cdef long idx

    # Output arrays for photon counts
    cdef cnp.ndarray[double] kG_photons = np.zeros(n_save, dtype=np.double)
    cdef cnp.ndarray[double] kR_photons = np.zeros(n_save, dtype=np.double)

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

        # Accumulate photons in time bin
        idx = i // saving_freq
        if idx < n_save:
            kG_photons[idx] += n_g
            kR_photons[idx] += n_r

        xold = xnew

    # Free allocated memory
    _gsl_spline.gsl_spline_free(spline)
    _gsl_spline.gsl_interp_accel_free(acc)
    grn.gsl_rng_free(rng)
    free(x_k)
    free(y_k)

    if status != 0:
        return None, None

    return kG_photons, kR_photons


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int _shoot_leg(
        double *out,
        int max_bins,
        int saving_freq,
        double x_start,
        double x_TS,
        int k_persist,
        long raw_step_budget,
        double Ax,
        double Bx,
        _gsl_spline.gsl_spline *force_spline,
        _gsl_spline.gsl_interp_accel *acc_force,
        grn.gsl_rng *rng,
        int *out_len
        ):
    """
    Shoot one Langevin leg from x_start under the unbiased SDE.

    Returns the basin code committed to:
        0 = basin A (x < x_TS)
        1 = basin B (x >= x_TS)
       -1 = failed (no commitment within max_bins or raw_step_budget)
    *out_len is set to the number of saved bins written to `out`.
    """
    cdef double x = x_start
    cdef double sval, F
    cdef long step = 0
    cdef int bin_idx = 0
    cdef int in_count = 0
    cdef int last_side = -1
    cdef int cur_side
    cdef int status = 0

    out_len[0] = 0

    while step < raw_step_budget:
        step += 1

        status = _gsl_spline.gsl_spline_eval_e(force_spline, x, acc_force, &sval)
        if status != 0:
            out_len[0] = bin_idx
            return -1
        F = sval

        x = x + Ax * F + Bx * grn.gsl_ran_gaussian_ziggurat(rng, 1.0)

        if step % saving_freq == 0:
            if bin_idx >= max_bins:
                out_len[0] = bin_idx
                return -1
            out[bin_idx] = x
            bin_idx += 1

            cur_side = 0 if x < x_TS else 1
            if cur_side == last_side:
                in_count += 1
            else:
                in_count = 1
                last_side = cur_side

            if in_count >= k_persist:
                out_len[0] = bin_idx
                return cur_side

    out_len[0] = bin_idx
    return -1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _pad_in_basin(
        double *leg,
        int current_len,
        int target_len,
        int saving_freq,
        double Ax,
        double Bx,
        _gsl_spline.gsl_spline *force_spline,
        _gsl_spline.gsl_interp_accel *acc_force,
        grn.gsl_rng *rng
        ):
    """
    Continue the unbiased SDE from leg[current_len - 1], writing one bin every
    `saving_freq` raw steps until the leg has `target_len` bins. No commitment
    checks, no rejection — just keeps the trajectory going inside whatever
    basin it has already committed to.
    """
    cdef double x
    cdef double sval, F
    cdef int bin_idx
    cdef int s

    if current_len <= 0 or current_len >= target_len:
        return

    x = leg[current_len - 1]
    bin_idx = current_len

    while bin_idx < target_len:
        for s in range(saving_freq):
            _gsl_spline.gsl_spline_eval_e(force_spline, x, acc_force, &sval)
            F = sval
            x = x + Ax * F + Bx * grn.gsl_ran_gaussian_ziggurat(rng, 1.0)
        leg[bin_idx] = x
        bin_idx += 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _locate_landmarks(
        _gsl_spline.gsl_spline *force_spline,
        _gsl_spline.gsl_interp_accel *acc,
        double x_min,
        double x_max,
        int n,
        double *x_A_out,
        double *x_B_out,
        double *x_TS_out
        ):
    """
    Locate the two basin minima and the barrier top of U(x) on [x_min, x_max].

    U(x) = -∫_{x_min}^{x} F(s) ds is gridded with n points. x_A is argmin on
    the left half, x_B is argmin on the right half, x_TS is argmax on
    [x_A, x_B].
    """
    cdef int j
    cdef int mid = n // 2
    cdef double dx = (x_max - x_min) / (n - 1)
    cdef double xj, U
    cdef int j_A = 0
    cdef int j_B = n - 1
    cdef double U_min_left = 1e300
    cdef double U_min_right = 1e300
    cdef int j_TS = 0
    cdef double U_max = -1e300

    for j in range(n):
        xj = x_min + j * dx
        U = -_gsl_spline.gsl_spline_eval_integ(force_spline, x_min, xj, acc)
        if j <= mid:
            if U < U_min_left:
                U_min_left = U
                j_A = j
        if j >= mid:
            if U < U_min_right:
                U_min_right = U
                j_B = j

    x_A_out[0] = x_min + j_A * dx
    x_B_out[0] = x_min + j_B * dx

    for j in range(j_A, j_B + 1):
        xj = x_min + j * dx
        U = -_gsl_spline.gsl_spline_eval_integ(force_spline, x_min, xj, acc)
        if U > U_max:
            U_max = U
            j_TS = j

    x_TS_out[0] = x_min + j_TS * dx


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
        int k_persist,
        int landmark_grid_n,
        long N_max=200000000,
        str spline_type='cubic'
        ):
    """
    Two-way shooting transition-path simulator (unbiased Langevin).

    Locates the basin minima x_A, x_B and the barrier top x_TS from the force
    spline, then shoots two independent unbiased Langevin legs from x_TS.
    Each leg is run until it commits to a basin (k_persist consecutive saved
    bins on one side of x_TS) or fails (exceeds half * saving_freq saved
    steps or N_max raw steps). If both legs commit to the same basin the
    trajectory is non-reactive and rejected. Otherwise each leg is padded by
    additional in-basin diffusion to reach exactly L_max_bins // 2 bins, the
    two legs are concatenated (with the "backward" leg reversed) so the
    transition sits at the midpoint, then a window of random length
    L_j ∈ [L_min_bins, L_max_bins] is taken centered on the transition and
    used to emit FRET photons.

    Parameters
    ----------
    Dx : float
        Diffusion coefficient.
    x_knots, y_knots : np.ndarray
        Force spline knot positions and force values.
    R0, kD, k_gb, k_rb, eta_g, eta_r, C_gg, C_rr, C_gr, C_rg : float
        FRET photophysics and detection-channel parameters.
    saving_freq : int
        Raw integration steps per saved bin.
    dt : float
        Raw integration time step.
    x_min, x_max : float
        Interior search interval used to locate x_A, x_B, x_TS.
    L_min_bins : int
        Smallest truncated window length, in saved bins.
    L_max_bins : int
        Largest truncated window length, in saved bins. Also fixes the
        total length of the two-sided shoot (each leg gets L_max_bins // 2).
    k_persist : int
        Consecutive bins inside a basin required to declare commitment.
    landmark_grid_n : int
        Grid size used by _locate_landmarks for finding x_A, x_B, x_TS.
    N_max : long
        Per-leg hard cap on raw integration steps (safety guard).
    spline_type : str
        GSL spline type ('cubic', 'steffen', or 'akima').

    Returns
    -------
    kG_photons, kR_photons : np.ndarray or (None, None)
        Green and red photon counts per saved bin (length L_j), or
        (None, None) if either leg failed to commit or both legs landed
        in the same basin (non-reactive).
    """

    cdef long i
    cdef int N_knots = len(x_knots)
    cdef int half = L_max_bins // 2

    # Random number generator
    cdef grn.gsl_rng_type * rng_type
    cdef grn.gsl_rng * rng
    cdef long seed = np.random.randint(low=1, high=2**63)

    grn.gsl_rng_env_setup()
    rng_type = grn.gsl_rng_default
    rng = grn.gsl_rng_alloc(rng_type)
    grn.gsl_rng_set(rng, seed)

    # Copy knots to C arrays
    cdef double *x_k = <double *> malloc(N_knots * sizeof(double))
    cdef double *y_k = <double *> malloc(N_knots * sizeof(double))
    for i in range(N_knots):
        x_k[i] = x_knots[i]
        y_k[i] = y_knots[i]

    # Force spline
    cdef _gsl_spline.gsl_interp_accel *acc_force = _gsl_spline.gsl_interp_accel_alloc()
    cdef _gsl_spline.gsl_spline *force_spline

    if spline_type == "cubic":
        force_spline = _gsl_spline.gsl_spline_alloc(_gsl_spline.gsl_interp_cspline, N_knots)
    elif spline_type == "steffen":
        force_spline = _gsl_spline.gsl_spline_alloc(_gsl_spline.gsl_interp_steffen, N_knots)
    elif spline_type == "akima":
        force_spline = _gsl_spline.gsl_spline_alloc(_gsl_spline.gsl_interp_akima, N_knots)
    else:
        free(x_k)
        free(y_k)
        _gsl_spline.gsl_interp_accel_free(acc_force)
        grn.gsl_rng_free(rng)
        raise ValueError(f"Unsupported spline type: {spline_type}")

    _gsl_spline.gsl_spline_init(force_spline, x_k, y_k, N_knots)

    # Locate landmarks (x_A, x_B, x_TS) directly from the force spline
    cdef double x_A_val = 0.0
    cdef double x_B_val = 0.0
    cdef double x_TS_val = 0.0
    _locate_landmarks(force_spline, acc_force, x_min, x_max, landmark_grid_n,
                      &x_A_val, &x_B_val, &x_TS_val)

    # Langevin coefficients
    cdef double Ax = Dx * dt
    cdef double Bx = sqrt(2.0 * Ax)

    # Allocate leg buffers and full-trajectory buffer
    cdef double *leg_fwd = <double *> malloc(half * sizeof(double))
    cdef double *leg_bwd = <double *> malloc(half * sizeof(double))
    cdef double *full = <double *> malloc(L_max_bins * sizeof(double))

    cdef int leg_len_fwd = 0
    cdef int leg_len_bwd = 0
    cdef int basin_fwd = -1
    cdef int basin_bwd = -1
    cdef int max_shoot_attempts = 100
    cdef int attempt
    cdef int accepted = 0

    # Retry shooting until the two legs commit to opposite basins. Each
    # attempt is an independent unbiased draw from the unbiased SDE at x_TS,
    # so retrying is just rejection sampling at the shooting stage and
    # introduces no bias. Scratch buffers leg_fwd / leg_bwd are reused.
    for attempt in range(max_shoot_attempts):
        basin_fwd = _shoot_leg(leg_fwd, half, saving_freq, x_TS_val, x_TS_val,
                               k_persist, N_max, Ax, Bx,
                               force_spline, acc_force, rng, &leg_len_fwd)
        if basin_fwd < 0:
            continue
        basin_bwd = _shoot_leg(leg_bwd, half, saving_freq, x_TS_val, x_TS_val,
                               k_persist, N_max, Ax, Bx,
                               force_spline, acc_force, rng, &leg_len_bwd)
        if basin_bwd < 0:
            continue
        if basin_fwd != basin_bwd:
            accepted = 1
            break

    if not accepted:
        free(leg_fwd)
        free(leg_bwd)
        free(full)
        _gsl_spline.gsl_spline_free(force_spline)
        _gsl_spline.gsl_interp_accel_free(acc_force)
        grn.gsl_rng_free(rng)
        free(x_k)
        free(y_k)
        return None, None

    # Pad each leg to exactly `half` bins of in-basin diffusion
    _pad_in_basin(leg_fwd, leg_len_fwd, half, saving_freq, Ax, Bx,
                  force_spline, acc_force, rng)
    _pad_in_basin(leg_bwd, leg_len_bwd, half, saving_freq, Ax, Bx,
                  force_spline, acc_force, rng)

    # Concatenate legs into `full` so that basin A is on the left, basin B on
    # the right, with the saddle crossing centered at index `half`.
    cdef int idx
    if basin_bwd == 0:
        # backward leg committed to A → reverse it into the left half
        for idx in range(half):
            full[idx] = leg_bwd[half - 1 - idx]
        for idx in range(half):
            full[half + idx] = leg_fwd[idx]
    else:
        # forward leg committed to A → reverse it into the left half
        for idx in range(half):
            full[idx] = leg_fwd[half - 1 - idx]
        for idx in range(half):
            full[half + idx] = leg_bwd[idx]

    # ─── Single-transition crop ───────────────────────────────────────────
    # Locate the central crossing nearest `half`. _shoot_leg only guarantees
    # the LAST k_persist bins of a leg are on the committed side, so the
    # first bins of either leg may not yet straddle x_TS exactly at `half`.
    cdef int t_idx = half
    cdef int offset = 0
    cdef int found = 0
    cdef int s_l, s_r
    while True:
        if half - offset - 1 >= 0 and half - offset < L_max_bins:
            s_l = 0 if full[half - offset - 1] < x_TS_val else 1
            s_r = 0 if full[half - offset]     < x_TS_val else 1
            if s_l != s_r:
                t_idx = half - offset
                found = 1
                break
        if half + offset >= 0 and half + offset + 1 < L_max_bins:
            s_l = 0 if full[half + offset]     < x_TS_val else 1
            s_r = 0 if full[half + offset + 1] < x_TS_val else 1
            if s_l != s_r:
                t_idx = half + offset + 1
                found = 1
                break
        offset += 1
        if half - offset - 1 < 0 and half + offset + 1 >= L_max_bins:
            break

    if not found:
        # No crossing anywhere — should not happen given the orientation
        # guarantee, but reject defensively.
        free(leg_fwd)
        free(leg_bwd)
        free(full)
        _gsl_spline.gsl_spline_free(force_spline)
        _gsl_spline.gsl_interp_accel_free(acc_force)
        grn.gsl_rng_free(rng)
        free(x_k)
        free(y_k)
        return None, None

    # Walk outward from t_idx until the next *persistent* recrossing on each
    # side. A recrossing counts only when k_persist consecutive bins land on
    # the opposite side — single-bin diffusive blips across x_TS don't end
    # the safe window. This mirrors the commitment criterion in _shoot_leg.
    cdef int side_left  = 0 if full[t_idx - 1] < x_TS_val else 1
    cdef int side_right = 0 if full[t_idx]     < x_TS_val else 1
    cdef int s_cur
    cdef int other_run

    cdef int left_safe = t_idx          # default: no recrossing found
    cdef int j = t_idx - 1
    other_run = 0
    while j >= 0:
        s_cur = 0 if full[j] < x_TS_val else 1
        if s_cur == side_left:
            other_run = 0
        else:
            other_run += 1
            if other_run >= k_persist:
                # Persistent recrossing run occupies indices [j, j + k_persist).
                # Safe region is everything strictly above j + k_persist - 1.
                left_safe = (t_idx - 1) - (j + k_persist - 1)
                break
        j -= 1

    cdef int right_safe = L_max_bins - t_idx   # default: no recrossing found
    j = t_idx
    other_run = 0
    while j < L_max_bins:
        s_cur = 0 if full[j] < x_TS_val else 1
        if s_cur == side_right:
            other_run = 0
        else:
            other_run += 1
            if other_run >= k_persist:
                # Persistent recrossing run occupies indices [j - k_persist + 1, j].
                # Safe region is everything strictly below j - k_persist + 1.
                right_safe = (j - k_persist + 1) - t_idx
                break
        j += 1

    # Symmetric crop: usable half-width is min(left_safe, right_safe).
    cdef int half_safe = left_safe if left_safe < right_safe else right_safe
    cdef int max_safe = 2 * half_safe
    if max_safe > L_max_bins:
        max_safe = L_max_bins

    if max_safe < L_min_bins:
        # Cannot fit a symmetric, single-transition window of the minimum length.
        free(leg_fwd)
        free(leg_bwd)
        free(full)
        _gsl_spline.gsl_spline_free(force_spline)
        _gsl_spline.gsl_interp_accel_free(acc_force)
        grn.gsl_rng_free(rng)
        free(x_k)
        free(y_k)
        return None, None

    cdef int L_j = int(np.random.randint(L_min_bins, max_safe + 1))
    cdef int start = t_idx - (L_j // 2)
    if start < 0:
        start = 0
    if start + L_j > L_max_bins:
        start = L_max_bins - L_j

    # --- Photon emission over the selected window ---
    cdef double bin_dt = saving_freq * dt
    cdef double R0_6 = pow(R0, 6.0)
    cdef double xi
    cdef double r, r_6, E, phi_g, phi_r, lambda_g, lambda_r
    cdef unsigned int n_g, n_r

    cdef cnp.ndarray[double] kG_photons = np.zeros(L_j, dtype=np.double)
    cdef cnp.ndarray[double] kR_photons = np.zeros(L_j, dtype=np.double)

    for idx in range(L_j):
        xi = full[start + idx]

        r = xi
        if r < 0.0:
            r = 0.0
        r_6 = pow(r, 6.0)

        E = R0_6 / (R0_6 + r_6)

        phi_g = C_gg * (1.0 - E) + C_rg * E
        phi_r = C_gr * (1.0 - E) + C_rr * E

        lambda_g = eta_g * (kD * phi_g + k_gb) * bin_dt
        lambda_r = eta_r * (kD * phi_r + k_rb) * bin_dt

        n_g = grn.gsl_ran_poisson(rng, lambda_g)
        n_r = grn.gsl_ran_poisson(rng, lambda_r)

        kG_photons[idx] = n_g
        kR_photons[idx] = n_r

    # --- Cleanup ---
    free(leg_fwd)
    free(leg_bwd)
    free(full)
    _gsl_spline.gsl_spline_free(force_spline)
    _gsl_spline.gsl_interp_accel_free(acc_force)
    grn.gsl_rng_free(rng)
    free(x_k)
    free(y_k)

    return kG_photons, kR_photons