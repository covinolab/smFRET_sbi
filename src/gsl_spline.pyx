import cython
cimport _gsl_spline as _gsl_spline
from math import *
from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as cnp


@cython.boundscheck(False)
@cython.wraparound(False)
def eval_spline(
        cnp.ndarray[double] x_knots,
        cnp.ndarray[double] y_knots,
        cnp.ndarray[double] x_eval,
        str spline_type='cubic'
        ):
    """Cython wrapper for GSL spline interpolation
    
    Parameters
    ----------
    x_knots : array_like
        x values of the knots
    y_knots : array_like
        y values of the knots
    x_eval : array_like
        x values where the spline is evaluated
    
    Returns
    -------
    y_eval : array_like
        y values of the spline at x_eval
    """

    cdef int N_knots = len(x_knots)
    cdef int N_eval = len(x_eval)

    cdef double *x_k = <double *> malloc(N_knots * sizeof(double))
    cdef double *y_k = <double *> malloc(N_knots * sizeof(double))

    for i  from 0 <= i < N_knots:
        x_k[i] = x_knots[i]
        y_k[i] = y_knots[i]

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

    cdef cnp.ndarray[double] y_eval = np.empty(N_eval, dtype=np.double)
    cdef double tmp_y
    cdef int status

    for i  from 0 <= i < N_eval:
        status = _gsl_spline.gsl_spline_eval_e(spline, x_eval[i], acc, &tmp_y)
        if status != 0:
            break
        y_eval[i] = tmp_y

    _gsl_spline.gsl_spline_free (spline)
    _gsl_spline.gsl_interp_accel_free (acc)
    free(x_k)
    free(y_k)

    if status != 0:
        return None

    return y_eval


@cython.boundscheck(False)
@cython.wraparound(False)
def integrate_spline(
        cnp.ndarray[double] x_knots,
        cnp.ndarray[double] y_knots,
        cnp.ndarray[double] x_eval,
        str spline_type='cubic'
        ):
    """Cumulative integral of a spline from x_eval[0] to each x_eval[i].

    Uses gsl_spline_eval_integ_e, which computes the exact definite integral
    of the piecewise polynomial representation (not numerical quadrature).

    Returns
    -------
    result : ndarray, shape (len(x_eval),)
        result[i] = ∫_{x_eval[0]}^{x_eval[i]} y(x) dx
    """

    cdef int N_knots = len(x_knots)
    cdef int N_eval = len(x_eval)

    cdef double *x_k = <double *> malloc(N_knots * sizeof(double))
    cdef double *y_k = <double *> malloc(N_knots * sizeof(double))

    for i in range(N_knots):
        x_k[i] = x_knots[i]
        y_k[i] = y_knots[i]

    cdef _gsl_spline.gsl_interp_accel *acc
    acc = _gsl_spline.gsl_interp_accel_alloc()
    cdef _gsl_spline.gsl_spline *sp

    if spline_type == "cubic":
        sp = _gsl_spline.gsl_spline_alloc(_gsl_spline.gsl_interp_cspline, N_knots)
    elif spline_type == "steffen":
        sp = _gsl_spline.gsl_spline_alloc(_gsl_spline.gsl_interp_steffen, N_knots)
    elif spline_type == "akima":
        sp = _gsl_spline.gsl_spline_alloc(_gsl_spline.gsl_interp_akima, N_knots)
    else:
        raise ValueError(f"Unsupported spline type: {spline_type}")

    _gsl_spline.gsl_spline_init(sp, x_k, y_k, N_knots)

    cdef cnp.ndarray[double] result = np.zeros(N_eval, dtype=np.double)
    cdef double tmp_integ
    cdef double x0 = x_eval[0]
    cdef int status = 0

    for i in range(N_eval):
        status = _gsl_spline.gsl_spline_eval_integ_e(sp, x0, x_eval[i], acc, &tmp_integ)
        if status != 0:
            break
        result[i] = tmp_integ

    _gsl_spline.gsl_spline_free(sp)
    _gsl_spline.gsl_interp_accel_free(acc)
    free(x_k)
    free(y_k)

    if status != 0:
        return None

    return result