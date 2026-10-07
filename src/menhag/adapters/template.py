# MENHAG, Mohamed Gamal Zaid, University of New Hampshire. MIT license, see LICENSE.
"""Template adapter: copy this file and edit two functions for any format.

If your unpolarized ASADs do not live in POLAR-style ROOT files, FITS
tables, or the documented CSV schema, this file is the intended starting
point. Copy it next to your analysis code (or subclass in place), then fill
in the two blocks marked

    # ============================ EDIT HERE ============================

Everything menhag needs from you:

1. ``load_unpolarized(energy)`` must return an :class:`menhag.asad.ASAD`
   holding, for that energy:
       - ``counts``:    1-D array of bin contents,
       - ``bin_edges``: 1-D array of bin edges in DEGREES of the measured
                        azimuthal scattering angle eta (length = bins + 1),
       - ``errors``:    per-bin uncertainties, or None to fall back to the
                        Poisson convention sigma = sqrt(n * f) / f with the
                        run's norm_factor.
   The generated polarized ASADs will inherit exactly this binning.

2. ``save_response(response)`` receives a :class:`menhag.asad.ResponseSet`:
       - ``response.unpolarized``  is {energy_kev: ASAD},
       - ``response.polarized``    is {(energy_kev, phi0_deg): ASAD},
       - ``response.pol_fraction`` is the polarization fraction,
   and should write them wherever and however you like.

Optionally, implement ``load_polarized(energy, phi0)`` the same way as
``load_unpolarized`` to enable mode 2 (simulated polarized ASADs at two
or more polarization angles per energy instead of the analytic kernel).

That is the whole contract. Run your build exactly as with the built-in
adapters::

    from my_adapter import MyInstrumentAdapter
    from menhag import ResponseConfig, build_response

    adapter = MyInstrumentAdapter("my_inputs/", "my_response.npz")
    build_response(config, adapter, adapter)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..asad import ASAD, ResponseSet
from ..exceptions import MissingEnergyError
from .base import InstrumentAdapter


class TemplateAdapter(InstrumentAdapter):
    """Skeleton adapter: fill in the EDIT HERE blocks for your format."""

    def __init__(self, input_path: str, output_path: Optional[str] = None) -> None:
        # Keep whatever state you need (paths, open handles, calibration).
        self.input_path = input_path
        self.output_path = output_path

    # ------------------------------------------------------------------ #
    def load_unpolarized(self, energy: float) -> ASAD:
        """Return the unpolarized ASAD at ``energy`` keV."""
        # ============================ EDIT HERE ============================
        # Replace this block with code that reads YOUR file format and
        # fills the three arrays below. The example shows the expected
        # shapes for a 60-bin ASAD over the full circle.
        #
        # counts    = ...  # shape (60,)
        # bin_edges = ...  # shape (61,), degrees, e.g. np.linspace(0, 360, 61)
        # errors    = ...  # shape (60,) or None for Poisson errors
        #
        # If the requested energy is not available, raise so menhag fails
        # loudly instead of interpolating:
        #     raise MissingEnergyError(f"No ASAD at {energy} keV")
        raise MissingEnergyError(
            "TemplateAdapter.load_unpolarized has not been filled in yet; "
            "copy menhag/adapters/template.py and edit the EDIT HERE block."
        )
        # =========================== END EDIT ==============================
        return ASAD(
            counts=np.asarray(counts, dtype=float),
            bin_edges=np.asarray(bin_edges, dtype=float),
            errors=None if errors is None else np.asarray(errors, dtype=float),
            energy=float(energy),
        )

    # ------------------------------------------------------------------ #
    def save_response(self, response: ResponseSet) -> None:
        """Persist the generated response in your format of choice."""
        # ============================ EDIT HERE ============================
        # Walk the response and write it out. A minimal NumPy example:
        #
        # payload = {}
        # for energy, asad in response.unpolarized.items():
        #     payload[f"unp_{int(energy)}"] = asad.counts
        #     payload[f"unp_{int(energy)}_err"] = asad.errors
        #     payload[f"unp_{int(energy)}_edges"] = asad.bin_edges
        # for (energy, phi0), asad in response.polarized.items():
        #     stem = f"pol_{int(energy)}_{int(round(phi0))}"
        #     payload[stem] = asad.counts
        #     payload[stem + "_err"] = asad.errors
        # np.savez(self.output_path, **payload)
        raise NotImplementedError(
            "TemplateAdapter.save_response has not been filled in yet; "
            "copy menhag/adapters/template.py and edit the EDIT HERE block."
        )
        # =========================== END EDIT ==============================
