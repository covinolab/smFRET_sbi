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
def binding_simulator(
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
        int saving_freq,
        double dt,
        double x_A,
        double x_B,
        int n_grid=500,
        double F_guide_max_factor=0.1,
        long N_max=200000000,
        long min_steps=0,
        str spline_type='cubic'
        ):
    """
    Doob h-transform conditioned binding transition path simulator.

    Generates trajectories conditioned on completing a B->A binding transition
    (large distance -> small distance) using the Doob h-transform. Every
    trajectory is guaranteed to reach x_A, producing unbiased transition paths.

    Photon arrival times are collected during the simulation and binned
    into time bins of width saving_freq * dt after the trajectory completes.

    Parameters
    ----------
    x0 : float
        Initial position (should be near x_B).
    Dx : float
        Diffusion coefficient.
    x_knots, y_knots : np.ndarray
        Force spline knot positions and force values.
    R0 : float
        Forster radius.
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
    C_gg, C_rr, C_gr, C_rg : float
        Cross-talk matrix elements.
    saving_freq : int
        Integration steps per saved time bin.
    dt : float
        Integration time step.
    x_A : float
        Absorbing boundary at bound state (small r, high FRET).
    x_B : float
        Starting boundary at unbound state (large r, low FRET).
    n_grid : int
        Grid points for precomputing U(x) and I(x).
    F_guide_max_factor : float
        Clamping factor for guiding force magnitude.
    N_max : long
        Safety cutoff in number of steps (default 2e8).
    spline_type : str
        GSL spline type ('cubic', 'steffen', or 'akima').

    Returns
    -------
    kG_photons : np.ndarray or None
        Green channel photon counts per time bin.
    kR_photons : np.ndarray or None
        Red channel photon counts per time bin.
    """

    # --- Setup: variables and force spline ---
    cdef long i, j, k
    cdef int N_knots = len(x_knots)

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

    # --- Phase A: Precomputation of U(x), I(x), J(x) ---
    cdef double dx_grid = (x_B - x_A) / (n_grid - 1)
    cdef double *x_grid = <double *> malloc(n_grid * sizeof(double))
    cdef double *U_grid = <double *> malloc(n_grid * sizeof(double))
    cdef double *expU_grid = <double *> malloc(n_grid * sizeof(double))
    cdef double *I_grid = <double *> malloc(n_grid * sizeof(double))
    cdef double *J_grid = <double *> malloc(n_grid * sizeof(double))

    for j in range(n_grid):
        x_grid[j] = x_A + j * dx_grid

    for j in range(n_grid):
        U_grid[j] = -_gsl_spline.gsl_spline_eval_integ(force_spline, x_A, x_grid[j], acc_force)

    for j in range(n_grid):
        expU_grid[j] = exp(U_grid[j])

    I_grid[0] = 0.0
    for j in range(1, n_grid):
        I_grid[j] = I_grid[j - 1] + 0.5 * (expU_grid[j - 1] + expU_grid[j]) * dx_grid

    cdef double Z = I_grid[n_grid - 1]
    for j in range(n_grid):
        J_grid[j] = Z - I_grid[j]

    cdef _gsl_spline.gsl_interp_accel *acc_U = _gsl_spline.gsl_interp_accel_alloc()
    cdef _gsl_spline.gsl_spline *U_spline = _gsl_spline.gsl_spline_alloc(
        _gsl_spline.gsl_interp_cspline, n_grid)
    _gsl_spline.gsl_spline_init(U_spline, x_grid, U_grid, n_grid)

    cdef _gsl_spline.gsl_interp_accel *acc_J = _gsl_spline.gsl_interp_accel_alloc()
    cdef _gsl_spline.gsl_spline *J_spline = _gsl_spline.gsl_spline_alloc(
        _gsl_spline.gsl_interp_cspline, n_grid)
    _gsl_spline.gsl_spline_init(J_spline, x_grid, J_grid, n_grid)

    # --- Phase B: Integration loop with photon collection ---
    cdef double Ax = Dx * dt
    cdef double Bx = sqrt(2.0 * Ax)
    cdef double R0_6 = pow(R0, 6.0)
    cdef double F_guide_max = F_guide_max_factor / (Dx * dt)

    cdef double xold = x0
    cdef double xnew, F_phys, F_guide, F_total, U_val, J_val
    cdef double r, r_6, E, phi_g, phi_r, lambda_g, lambda_r
    cdef double spline_val, t
    cdef unsigned int n_g, n_r
    cdef int status = 0
    cdef int binding_complete = 0
    cdef long n_steps = 0

    # Collect photon arrival times in Python lists
    green_arrival_times = []
    red_arrival_times = []

    for i in range(1, N_max):

        # Physical force from force spline
        status = _gsl_spline.gsl_spline_eval_e(force_spline, xold, acc_force, &spline_val)
        if status != 0:
            break
        F_phys = spline_val

        # Guiding force for B->A binding
        U_val = _gsl_spline.gsl_spline_eval(U_spline, xold, acc_U)
        J_val = _gsl_spline.gsl_spline_eval(J_spline, xold, acc_J)

        if J_val > 1e-15:
            F_guide = -2.0 * exp(U_val) / J_val
        else:
            F_guide = -F_guide_max

        if F_guide < -F_guide_max:
            F_guide = -F_guide_max

        F_total = F_phys + F_guide

        # Euler-Maruyama step
        xnew = xold + Ax * F_total + Bx * grn.gsl_ran_gaussian_ziggurat(rng, 1.0)

        # Boundary handling
        if xnew >= x_B:
            xnew = x_B - 1e-10
        if xnew <= x_A:
            xnew = x_A
            binding_complete = 1

        # Photophysics (identical to smFRET_simulator)
        r = xnew
        if r < 0.0:
            r = 0.0
        r_6 = pow(r, 6.0)

        E = R0_6 / (R0_6 + r_6)

        phi_g = C_gg * (1.0 - E) + C_rg * E
        phi_r = C_gr * (1.0 - E) + C_rr * E

        lambda_g = eta_g * (kD * phi_g + k_gb) * dt
        lambda_r = eta_r * (kD * phi_r + k_rb) * dt

        n_g = grn.gsl_ran_poisson(rng, lambda_g)
        n_r = grn.gsl_ran_poisson(rng, lambda_r)

        # Record photon arrival times
        t = i * dt
        for k in range(n_g):
            green_arrival_times.append(t)
        for k in range(n_r):
            red_arrival_times.append(t)

        n_steps = i
        xold = xnew

        if binding_complete:
            break

    # --- Phase C: Cleanup ---
    _gsl_spline.gsl_spline_free(J_spline)
    _gsl_spline.gsl_interp_accel_free(acc_J)
    _gsl_spline.gsl_spline_free(U_spline)
    _gsl_spline.gsl_interp_accel_free(acc_U)
    _gsl_spline.gsl_spline_free(force_spline)
    _gsl_spline.gsl_interp_accel_free(acc_force)
    grn.gsl_rng_free(rng)
    free(x_k)
    free(y_k)
    free(x_grid)
    free(U_grid)
    free(expU_grid)
    free(I_grid)
    free(J_grid)

    if status != 0 or not binding_complete:
        return None, None

    # Safety: reject too-short transition paths
    if min_steps > 0 and n_steps < min_steps:
        return None, None

    # --- Phase D: Bin photon arrival times ---
    cdef double bin_width = saving_freq * dt
    cdef long n_bins = (n_steps // saving_freq) + 1
    cdef long idx

    cdef cnp.ndarray[double] kG_photons = np.zeros(n_bins, dtype=np.double)
    cdef cnp.ndarray[double] kR_photons = np.zeros(n_bins, dtype=np.double)

    for t in green_arrival_times:
        idx = <long>(t / bin_width)
        if idx >= n_bins:
            idx = n_bins - 1
        kG_photons[idx] += 1.0

    for t in red_arrival_times:
        idx = <long>(t / bin_width)
        if idx >= n_bins:
            idx = n_bins - 1
        kR_photons[idx] += 1.0

    return kG_photons, kR_photons