# MENHAG, Mohamed Gamal Zaid, University of New Hampshire. MIT license, see LICENSE.
"""MENHAG: Modulation ENgine for Harmonic ASAD Generation.

Polarization responses of wide field-of-view Compton polarimeters (POLAR,
POLAR-2, LEAP and similar instruments) from a small set of simulated
inputs. An ASAD is the azimuthal scattering angle distribution, the
histogram of the azimuthal direction into which photons Compton scatter in
the detector; the polarization response of an instrument is the set of
ASADs over energy and polarization angle. This package is the code behind
the accompanying paper, released so that anyone can generate responses for
their own instrument.

Two response modes:

1. Mode 1, the unpolarized-only mode: the unpolarized ASAD at each energy
   is multiplied by the analytic modulation kernel built from the gated,
   frame-transformed Klein-Nishina cross-section. It needs no polarized
   simulation and assumes no polarization-dependent instrumental effects.
2. Mode 2, the two-angle mode: the unpolarized ASAD plus fully polarized
   ASADs at two polarization angles per energy (0 and 45 degrees by
   default; more angles are combined by least squares) determine the ASAD
   at every other polarization angle, so the polarization-dependent
   instrumental effects present in the simulations carry into the
   response.

Quick start::

    import numpy as np
    from menhag import build_single

    pol_counts, pol_errors = build_single(
        counts, bin_edges,             # your unpolarized ASAD
        energy=200.0,                  # keV
        delta=26.4, lambda_=0.0,       # source position [deg]
        phi0=45.0,                     # polarization angle [deg]
        threshold={"type": "erf", "mu": 25.0, "sigma": 75.0},
    )

The threshold gate is always an instrument-specific input; the values
above are the POLAR example of the paper, never a default. See the README
for the run modes, the I/O adapters, the worked notebooks and how to cite.
"""

from .asad import ASAD, ResponseSet, poisson_errors
from .config import ResponseConfig
from .engine import (
    build_ideal,
    build_response,
    build_single,
    build_spectrum_response,
    generate_expected_polarized_response,
    predict_polarized_asad,
)
from .exceptions import (
    AdapterError,
    BasisError,
    ConfigError,
    EmptyComptonWindowError,
    MenhagError,
    MissingEnergyError,
    ThresholdError,
)
from .adapters import ArrayAdapter, InstrumentAdapter
from .gates import (
    EnergyDependentErfGate,
    ErfGate,
    GaussianSampledGate,
    SharpGate,
    TwoDepositErfGate,
    load_gate_json,
    parse_threshold,
)
from .m2 import M2GRB
from .basis import (
    PolarizedBasis,
    blind_test,
    build_basis_response,
    check_basis_angles,
    holdout_validate,
    reconstruct,
    reconstruct_var,
)
from .validation import calculate_reduced_chi_squared, compare_asads
from . import palette

# The single version string of the package; pyproject.toml reads it from
# here and CITATION.cff carries the same value at release.
__version__ = "1.0.0"
__author__ = "Mohamed Gamal Zaid"

__all__ = [
    "__version__",
    "__author__",
    "palette",
    # containers
    "ASAD",
    "ResponseSet",
    "poisson_errors",
    # configuration
    "ResponseConfig",
    # mode 1 (analytic kernel)
    "build_single",
    "build_ideal",
    "build_response",
    "build_spectrum_response",
    "predict_polarized_asad",
    "generate_expected_polarized_response",
    "M2GRB",
    # mode 2 (polarized basis)
    "PolarizedBasis",
    "check_basis_angles",
    "reconstruct",
    "reconstruct_var",
    "holdout_validate",
    "blind_test",
    "build_basis_response",
    # adapters (in-memory; file adapters live in menhag.adapters)
    "InstrumentAdapter",
    "ArrayAdapter",
    # validation
    "calculate_reduced_chi_squared",
    "compare_asads",
    # threshold gates
    "ErfGate",
    "SharpGate",
    "TwoDepositErfGate",
    "EnergyDependentErfGate",
    "GaussianSampledGate",
    "parse_threshold",
    "load_gate_json",
    # exceptions
    "MenhagError",
    "ThresholdError",
    "EmptyComptonWindowError",
    "MissingEnergyError",
    "BasisError",
    "AdapterError",
    "ConfigError",
]
