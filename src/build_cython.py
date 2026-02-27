#!/usr/bin/env python
"""
Simple script to compile Cython code in-place without packaging.

Usage:
    python build_cython.py

This will compile smFRET_simulator.pyx into a shared library (.so) 
that can be imported directly from Python.
"""

import os
import sys
import numpy as np
from Cython.Build import cythonize
from setuptools import Extension
from setuptools.dist import Distribution

# Get the directory containing this script
src_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(src_dir)

# Define the extension
extensions = [
    Extension(
        name="smFRET_simulator",  # Module name (no package prefix)
        sources=["smFRET_simulator.pyx"],
        libraries=["gsl", "gslcblas", "m"],  # GSL libraries
        include_dirs=[
            np.get_include(),
            # Add custom include paths here if needed:
            # "/usr/local/include",
        ],
        library_dirs=[
            # Add custom library paths here if needed:
            # "/usr/local/lib",
        ],
        extra_compile_args=["-O3", "-ffast-math"],
    ),
    Extension(
        name="gsl_spline",  # GSL spline Python wrapper
        sources=["gsl_spline.pyx"],
        libraries=["gsl", "gslcblas", "m"],
        include_dirs=[np.get_include()],
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
    annotate=False,  # Generate .html showing Python/C interaction
)

# Build in-place
dist = Distribution({"ext_modules": ext_modules})
dist.parse_config_files()

cmd = dist.get_command_obj("build_ext")
cmd.inplace = True  # Build in current directory
cmd.ensure_finalized()
cmd.run()

print("\n✓ Build complete! You can now: from smFRET_simulator import smFRET_simulator")
