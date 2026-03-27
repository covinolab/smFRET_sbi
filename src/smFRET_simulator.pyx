import cython
import numpy as np

from libc.math cimport sqrt, pow
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