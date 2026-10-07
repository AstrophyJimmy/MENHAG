# MENHAG, Mohamed Gamal Zaid, University of New Hampshire. MIT license, see LICENSE.
"""FITS table I/O for ASADs and responses (optional alternative to ROOT).

Layout
------
Each ASAD is one binary-table HDU with columns

    ETA_LO   lower bin edge of eta, degrees
    ETA_HI   upper bin edge of eta, degrees
    COUNTS   bin contents
    ERROR    per-bin uncertainty

and header keywords ``ENERGY`` (keV), ``POLFRAC`` (polarization fraction),
``POLANG`` (phi0, degrees), ``DELTA`` and ``LAMBDA`` (source incidence,
degrees), and ``MENHAGV`` (package version). HDU names follow

    ASAD_{energy:04d}_{pol:03d}_{angle:03d}

mirroring the ROOT key scheme: ``pol`` is the polarization degree in percent
(000 unpolarized) and ``angle`` is phi0 in integer degrees. The template is
configurable, like the ROOT one.

This is a plain, self-describing table layout, not an OGIP response
format. The other FITS layout the package reads is the .prsp polarization
response of polpy: :class:`FitsAdapter` detects it from its extensions
(INPAVALS and the two SPECRESP matrices) and reads it through
:class:`menhag.adapters.prsp.PrspAdapter`, in either of the two angle
orders that layout has carried.

Requires ``astropy`` (``pip install menhag[fits]``); imported lazily.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

from ..asad import ASAD, ResponseSet
from ..exceptions import AdapterError, MissingEnergyError
from .base import InstrumentAdapter

logger = logging.getLogger("menhag")

DEFAULT_EXT_TEMPLATE = "ASAD_{energy:04d}_{pol:03d}_{angle:03d}"


def _import_fits():
    try:
        from astropy.io import fits
    except ImportError as exc:
        raise AdapterError(
            "The FITS adapter needs the optional dependency 'astropy'. "
            "Install it with: pip install menhag[fits]"
        ) from exc
    return fits


def read_asad_fits(path: str, extname: str, energy: Optional[float] = None) -> ASAD:
    """Read one ASAD table HDU from a FITS file.

    Examples
    --------
    ``read_asad_fits("response.fits", "ASAD_0200_000_000", energy=200.0)``
    """
    fits = _import_fits()
    if not os.path.exists(path):
        raise AdapterError(f"FITS file not found: {path}")
    with fits.open(path) as hdul:
        if extname not in hdul:
            raise MissingEnergyError(
                f"HDU '{extname}' not found in {path}. Check the extension "
                "template against the file contents."
            )
        hdu = hdul[extname]
        data = hdu.data
        lo = np.asarray(data["ETA_LO"], dtype=float)
        hi = np.asarray(data["ETA_HI"], dtype=float)
        counts = np.asarray(data["COUNTS"], dtype=float)
        errors = np.asarray(data["ERROR"], dtype=float)
        header_energy = hdu.header.get("ENERGY")
    edges = np.concatenate([lo, hi[-1:]])
    if energy is None and header_energy is not None:
        energy = float(header_energy)
    return ASAD(
        counts=counts,
        bin_edges=edges,
        errors=errors,
        energy=energy,
        meta={"source": path, "extname": extname},
    )


def write_response_fits(
    path: str,
    response: ResponseSet,
    ext_template: str = DEFAULT_EXT_TEMPLATE,
    include_unpolarized: bool = True,
    overwrite: bool = True,
) -> None:
    """Write a :class:`ResponseSet` as a multi-extension FITS file.

    See the module docstring for the table layout. Spectrum-integrated
    entries (energy None) are written with energy code 0000.
    """
    fits = _import_fits()
    from .. import __version__

    def _hdu(asad: ASAD, name: str, pol_fraction: float, phi0: float):
        errors = (
            asad.errors
            if asad.errors is not None
            else np.zeros_like(asad.counts)
        )
        columns = fits.ColDefs(
            [
                fits.Column(name="ETA_LO", format="D", unit="deg",
                            array=asad.bin_edges[:-1]),
                fits.Column(name="ETA_HI", format="D", unit="deg",
                            array=asad.bin_edges[1:]),
                fits.Column(name="COUNTS", format="D", array=asad.counts),
                fits.Column(name="ERROR", format="D", array=errors),
            ]
        )
        hdu = fits.BinTableHDU.from_columns(columns, name=name)
        header = hdu.header
        if asad.energy is not None:
            header["ENERGY"] = (float(asad.energy), "photon energy [keV]")
        header["POLFRAC"] = (float(pol_fraction), "polarization fraction")
        header["POLANG"] = (float(phi0), "polarization angle phi0 [deg]")
        if "delta_deg" in response.meta:
            header["DELTA"] = (float(response.meta["delta_deg"]),
                               "source zenith angle [deg]")
        if "lambda_deg" in response.meta:
            header["LAMBDA"] = (float(response.meta["lambda_deg"]),
                                "source azimuth angle [deg]")
        header["MENHAGV"] = (__version__, "menhag version")
        return hdu

    pol_code = int(round(100.0 * response.pol_fraction))
    hdus = [fits.PrimaryHDU()]
    if include_unpolarized:
        for energy, asad in response.unpolarized.items():
            name = ext_template.format(
                energy=int(round(energy)) if energy is not None else 0,
                pol=0,
                angle=0,
            )
            hdus.append(_hdu(asad, name, 0.0, 0.0))
    for (energy, phi0), asad in response.polarized.items():
        name = ext_template.format(
            energy=int(round(energy)) if energy is not None else 0,
            pol=pol_code,
            angle=int(round(phi0)),
        )
        hdus.append(_hdu(asad, name, response.pol_fraction, phi0))
    fits.HDUList(hdus).writeto(path, overwrite=overwrite)
    logger.info("Wrote FITS response: %s", path)


def is_prsp(path: str) -> bool:
    """Whether a FITS file is a .prsp polarization response (it carries
    the INPAVALS extension and the two SPECRESP matrices)."""
    fits = _import_fits()
    if not os.path.exists(path):
        return False
    with fits.open(path) as hdul:
        names = {h.name for h in hdul}
    return {"INPAVALS", "SPECRESP POLMATRIX", "SPECRESP UNPOLMATRIX"} <= names


class FitsAdapter(InstrumentAdapter):
    """Adapter reading and writing the FITS layout described above.

    When ``input_path`` is a .prsp polarization response instead, the
    loads go through :class:`menhag.adapters.prsp.PrspAdapter`, which
    reads the angle order from the file's PALAYOUT keyword or takes it
    from ``layout``; the output of :meth:`save_response` is always the
    table layout of this module (use :class:`PrspAdapter` to write a
    .prsp).

    Parameters
    ----------
    input_path : str
        FITS file holding the input ASADs.
    output_path : str, optional
        Destination for :meth:`save_response`.
    ext_template : str
        HDU name format string with ``energy``, ``pol``, ``angle`` fields.
    input_pol_code : int
        ``pol`` field value of polarized inputs (mode 2), default 100.
    layout : str, optional
        The angle order of a .prsp input without a PALAYOUT keyword.
    norm_factor : float, optional
        Passed to the prsp adapter for a .prsp input (Poisson errors of
        the stored matrices); ignored for the table layout, which stores
        its errors.
    """

    def __init__(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        ext_template: str = DEFAULT_EXT_TEMPLATE,
        input_pol_code: int = 100,
        layout: Optional[str] = None,
        norm_factor: Optional[float] = None,
    ) -> None:
        self.input_path = input_path
        self.output_path = output_path
        self.ext_template = ext_template
        self.input_pol_code = input_pol_code
        self._prsp = None
        if is_prsp(input_path):
            from .prsp import PrspAdapter

            self._prsp = PrspAdapter(input_path, layout=layout, norm_factor=norm_factor)
            logger.info("%s is a .prsp polarization response; reading it as one.",
                        input_path)

    def load_unpolarized(self, energy: float) -> ASAD:
        if self._prsp is not None:
            return self._prsp.load_unpolarized(energy)
        name = self.ext_template.format(energy=int(round(energy)), pol=0, angle=0)
        return read_asad_fits(self.input_path, name, energy=energy)

    def load_polarized(self, energy: float, phi0: float) -> ASAD:
        if self._prsp is not None:
            return self._prsp.load_polarized(energy, phi0)
        name = self.ext_template.format(
            energy=int(round(energy)),
            pol=self.input_pol_code,
            angle=int(round(phi0)),
        )
        return read_asad_fits(self.input_path, name, energy=energy)

    def save_response(self, response: ResponseSet) -> None:
        if self.output_path is None:
            raise AdapterError("FitsAdapter has no output_path; pass one to save.")
        write_response_fits(self.output_path, response, self.ext_template)
