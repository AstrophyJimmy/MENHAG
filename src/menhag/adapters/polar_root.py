# MENHAG, Mohamed Gamal Zaid, University of New Hampshire. MIT license, see LICENSE.
"""ROOT histogram I/O in the POLAR response layout (the package default).

Layout
------
A response file is a flat ROOT file of TH1 histograms keyed

    sim_{energy:03d}_{pol:03d}_{angle:03d}

where ``energy`` is the photon energy in keV, ``pol`` is the polarization
degree in percent (000 for unpolarized, 100 for fully polarized), and
``angle`` is the polarization angle phi0 in integer degrees. Examples:
``sim_200_000_000`` is the unpolarized ASAD at 200 keV and
``sim_200_100_045`` is the fully polarized ASAD at 200 keV with
phi0 = 45 degrees. The histogram x axis is the measured azimuthal
scattering angle eta in degrees.

Adapting to your own key scheme
-------------------------------
The key template is an ordinary Python format string configurable through
``key_template`` (fields: ``energy``, ``pol``, ``angle``); if your layout
differs by more than the key names (directories, trees, several files),
copy :mod:`menhag.adapters.template` and edit it instead.

Dependencies
------------
Requires ``uproot`` (and ``hist`` for writing histograms that carry the
propagated per-bin errors); both are pulled in by
``pip install menhag[root]``. Imports are lazy so CSV, FITS, and in-memory
users never need ROOT tooling.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

from ..asad import ASAD, ResponseSet, poisson_errors
from ..exceptions import AdapterError, MissingEnergyError
from .base import InstrumentAdapter

logger = logging.getLogger("menhag")

DEFAULT_KEY_TEMPLATE = "sim_{energy:03d}_{pol:03d}_{angle:03d}"


def _import_uproot():
    try:
        import uproot
    except ImportError as exc:
        raise AdapterError(
            "The ROOT adapter needs the optional dependency 'uproot'. "
            "Install it with: pip install menhag[root]"
        ) from exc
    return uproot


def read_asad_root(path: str, key: str, energy: Optional[float] = None) -> ASAD:
    """Read one TH1 from a ROOT file into an :class:`ASAD`.

    Bin errors stored in the histogram (Sumw2) are preserved; if the file
    carries none, uproot reports sqrt(contents), which matches the Poisson
    convention with norm_factor = 1.

    Examples
    --------
    ``read_asad_root("response.root", "sim_200_000_000", energy=200.0)``
    """
    uproot = _import_uproot()
    if not os.path.exists(path):
        raise AdapterError(f"ROOT file not found: {path}")
    with uproot.open(path) as handle:
        try:
            hist_obj = handle[key]
        except KeyError:
            raise MissingEnergyError(
                f"Histogram key '{key}' not found in {path}. Check the "
                "key_template against the file contents "
                "(uproot.open(path).keys() lists them)."
            ) from None
        counts = np.asarray(hist_obj.values(), dtype=float)
        errors = np.asarray(hist_obj.errors(), dtype=float)
        edges = np.asarray(hist_obj.axis().edges(), dtype=float)
    return ASAD(
        counts=counts,
        bin_edges=edges,
        errors=errors,
        energy=energy,
        meta={"source": path, "key": key},
    )


def write_response_root(
    path: str,
    response: ResponseSet,
    key_template: str = DEFAULT_KEY_TEMPLATE,
    include_unpolarized: bool = True,
) -> None:
    """Write a :class:`ResponseSet` as a POLAR-style ROOT histogram file.

    Histograms are written with weighted storage so the propagated per-bin
    errors are preserved in the file (requires the 'hist' package; without
    it a plain histogram is written, whose stored errors degrade to
    sqrt(contents), and a warning is emitted).

    Parameters
    ----------
    path : str
        Output ROOT file (recreated).
    response : ResponseSet
        The generated response. Spectrum-integrated entries (energy None)
        are written with energy code 000.
    key_template : str
        Format string with ``energy``, ``pol``, ``angle`` integer fields.
    include_unpolarized : bool
        Also write the unpolarized inputs (pol code 000, angle 000), so the
        output file is a complete, self-contained response.
    """
    uproot = _import_uproot()
    try:
        import hist as hist_lib
    except ImportError:
        hist_lib = None
        logger.warning(
            "The 'hist' package is not installed; ROOT output will not carry "
            "the propagated errors (stored errors fall back to "
            "sqrt(contents)). Install with: pip install menhag[root]"
        )

    pol_code = int(round(100.0 * response.pol_fraction))

    def _writable(asad: ASAD):
        if hist_lib is not None and asad.errors is not None:
            h = hist_lib.Hist.new.Variable(
                asad.bin_edges, name="eta", label="eta [deg]"
            ).Weight()
            view = h.view()
            view["value"] = asad.counts
            view["variance"] = asad.errors**2
            return h
        return (asad.counts, asad.bin_edges)

    with uproot.recreate(path) as out:
        if include_unpolarized:
            for energy, asad in response.unpolarized.items():
                key = key_template.format(
                    energy=int(round(energy)) if energy is not None else 0,
                    pol=0,
                    angle=0,
                )
                out[key] = _writable(asad)
        for (energy, phi0), asad in response.polarized.items():
            angle_code = int(round(phi0))
            if abs(angle_code - phi0) > 1e-9:
                logger.warning(
                    "Polarization angle %.3f deg rounded to %d in the ROOT "
                    "key; use a custom key_template if this is not wanted.",
                    phi0,
                    angle_code,
                )
            key = key_template.format(
                energy=int(round(energy)) if energy is not None else 0,
                pol=pol_code,
                angle=angle_code,
            )
            out[key] = _writable(asad)
    logger.info("Wrote ROOT response: %s", path)


class PolarRootAdapter(InstrumentAdapter):
    """Default adapter: POLAR-style ROOT files (see module docstring).

    Parameters
    ----------
    input_path : str
        ROOT file holding the unpolarized (and, for mode 2, polarized)
        input ASADs.
    output_path : str, optional
        Destination for :meth:`save_response`; required only when saving.
    key_template : str
        Histogram key format string; edit to match your file's naming.
    input_pol_code : int
        ``pol`` field value of polarized inputs (default 100, fully
        polarized simulations).
    norm_factor : float, optional
        Count normalization f of the stored histograms (raw counts are
        value times f). When given, the per-bin errors of every ASAD
        read through this adapter are the Poisson errors
        sqrt(n f) / f of the stored contents, replacing whatever error
        array the file carries. Give it whenever the file stores
        normalized contents whose stored errors are not the Poisson
        errors of the normalized values (the POLAR response files store
        errors equal to sqrt of the normalized contents, which is not
        the statistical error of a histogram divided by f). ``None``
        keeps the file's own errors.

    Examples
    --------
    >>> adapter = PolarRootAdapter("polar_sims.root", "response.root")  # doctest: +SKIP
    """

    def __init__(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        key_template: str = DEFAULT_KEY_TEMPLATE,
        input_pol_code: int = 100,
        norm_factor: Optional[float] = None,
    ) -> None:
        self.input_path = input_path
        self.output_path = output_path
        self.key_template = key_template
        self.input_pol_code = input_pol_code
        self.norm_factor = None if norm_factor is None else float(norm_factor)

    def _with_errors(self, asad: ASAD) -> ASAD:
        if self.norm_factor is None:
            return asad
        return ASAD(
            counts=asad.counts,
            bin_edges=asad.bin_edges,
            errors=poisson_errors(asad.counts, self.norm_factor),
            energy=asad.energy,
            meta=dict(asad.meta, norm_factor=self.norm_factor),
        )

    def load_unpolarized(self, energy: float) -> ASAD:
        key = self.key_template.format(energy=int(round(energy)), pol=0, angle=0)
        return self._with_errors(read_asad_root(self.input_path, key, energy=energy))

    def load_polarized(self, energy: float, phi0: float) -> ASAD:
        key = self.key_template.format(
            energy=int(round(energy)),
            pol=self.input_pol_code,
            angle=int(round(phi0)),
        )
        return self._with_errors(read_asad_root(self.input_path, key, energy=energy))

    def save_response(self, response: ResponseSet) -> None:
        if self.output_path is None:
            raise AdapterError(
                "PolarRootAdapter has no output_path; pass one to save."
            )
        write_response_root(self.output_path, response, self.key_template)
