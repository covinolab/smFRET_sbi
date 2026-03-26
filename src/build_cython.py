#!/usr/bin/env python
"""
Simple script to compile Cython code in-place without packaging.

Usage:
    python build_cython.py

This will compile smFRET_simulator.pyx into a shared library (.so) 
that can be imported directly from Python.
"""

import os
import subprocess
import numpy as np
from Cython.Build import cythonize
from setuptools import Extension
from setuptools.dist import Distribution

# Get the directory containing this script
src_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(src_dir)


def _gsl_dirs():
    """Detect GSL include/library dirs without hardcoding paths.

    Resolution order:
      1. pkg-config (works when GSL module sets PKG_CONFIG_PATH)
      2. EBROOTGSL  (set by EasyBuild/Lmod when the GSL module is loaded)
      3. GSL_DIR    (user-set environment variable)
      4. empty      (fall back to compiler defaults, original behaviour)
    """
    # 1. pkg-config
    try:
        cflags = subprocess.check_output(
            ["pkg-config", "--cflags-only-I", "gsl"],
            stderr=subprocess.DEVNULL, text=True,
        ).split()
        libs = subprocess.check_output(
            ["pkg-config", "--libs-only-L", "gsl"],
            stderr=subprocess.DEVNULL, text=True,
        ).split()
        inc = [f[2:] for f in cflags if f.startswith("-I")]
        lib = [f[2:] for f in libs if f.startswith("-L")]
        if inc or lib:
            return inc, lib
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # 2. EBROOTGSL (EasyBuild)
    # 3. GSL_DIR   (generic convention)
    for env_var in ("EBROOTGSL", "GSL_DIR"):
        root = os.environ.get(env_var, "").strip()
        if root:
            return [os.path.join(root, "include")], [os.path.join(root, "lib")]

    return [], []


gsl_include_dirs, gsl_library_dirs = _gsl_dirs()

# Define the extension
extensions = [
    Extension(
        name="smFRET_simulator",  # Module name (no package prefix)
        sources=["smFRET_simulator.pyx"],
        libraries=["gsl", "gslcblas", "m"],  # GSL libraries
        include_dirs=[np.get_include()] + gsl_include_dirs,
        library_dirs=gsl_library_dirs,
        extra_compile_args=["-O3", "-ffast-math"],
    ),
    Extension(
        name="gsl_spline",  # GSL spline Python wrapper
        sources=["gsl_spline.pyx"],
        libraries=["gsl", "gslcblas", "m"],
        include_dirs=[np.get_include()] + gsl_include_dirs,
        library_dirs=gsl_library_dirs,
        extra_compile_args=["-O3", "-ffast-math"],
    ),
]

# Cythonize with settings
ext_modules = cythonize(
    extensions,
    compiler_directives={
        "language_level": "3",
        "boundscheck": False,
        "wraparound": False,
        "cdivision": True,
    },
)

# Build in-place
dist = Distribution({"ext_modules": ext_modules})
dist.parse_config_files()

cmd = dist.get_command_obj("build_ext")
cmd.inplace = True  # Build in current directory
cmd.ensure_finalized()
cmd.run()

print("\n✓ Build complete! You can now: from smFRET_simulator import smFRET_simulator")
