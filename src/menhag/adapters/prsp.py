# MENHAG, Mohamed Gamal Zaid, University of New Hampshire. MIT license, see LICENSE.
"""The .prsp polarization-response layout of polpy: export and input adapter.

A .prsp file is the FITS layout read by the polpy plug-in for threeML
(the format document of the threeML/polpy repository): INEBOUNDS (the
Monte Carlo energy bin edges), INPAVALS (the polarization angles of the
polarized entries), SABOUNDS (the scattering-angle bins), and the two
image extensions SPECRESP POLMATRIX (nE x nPA x nSA, fully polarized) and
SPECRESP UNPOLMATRIX (nE x nSA). Values are float32.

Two plug-ins read this layout, and they disagree on which polarized entry
belongs to which angle:

- ``"polpy-1.2.1"``: the current polpy release (threeML/polpy, 1.2.1 on
  PyPI). Entry j of the polarized matrix is the ASAD simulated at
  INPAVALS[j]. This is the order of the format document and of the
  response files the POLAR collaboration delivers, and it is the default
  of every writer in this module.
- ``"polarpy"``: the earlier plug-in (grburgess/polarpy), the one behind
  the fits of the accompanying paper. Entry j holds the ASAD at
  180 - INPAVALS[j] degrees, the mirror image along the angle axis. A fit
  against a file in the wrong order returns the polarization angle
  reflected through 90 degrees and leaves the polarization fraction
  untouched, so the order is not a matter of convention within the errors.

Every writer records the layout it used in the header keyword PALAYOUT of
the INPAVALS and SPECRESP POLMATRIX extensions, and every reader honours
that keyword. A file without the keyword (a collaboration file, or one
written by another tool) is read in the layout the caller names, or in the
default layout with a logged warning.

A POLAR response holds, per energy, 61 unique simulations: one unpolarized
and 60 fully polarized at 3 degree steps from 0 to 177 degrees. The
polarization angle is defined modulo 180 degrees, so a 180 degree entry
carries no new information; the writers here add it as a copy of the
0 degree ASAD, the closed boundary polpy expects.

Requires astropy (``pip install menhag[fits]``); :func:`convert_prsp` also
needs uproot (``pip install menhag[root]``). Both are imported lazily.
"""

from __future__ import annotations

import datetime
import logging
import os
from typing import Optional

import numpy as np

from ..asad import ASAD, ResponseSet, poisson_errors
from ..exceptions import AdapterError, MissingEnergyError
from .base import InstrumentAdapter

logger = logging.getLogger("menhag")

LAYOUT_CURRENT = "polpy-1.2.1"
LAYOUT_PAPER = "polarpy"
LAYOUTS = (LAYOUT_CURRENT, LAYOUT_PAPER)
DEFAULT_LAYOUT = LAYOUT_CURRENT
LAYOUT_KEYWORD = "PALAYOUT"
_LAYOUT_COMMENT = {
    LAYOUT_CURRENT: "entry j holds the ASAD at PA_IN[j]",
    LAYOUT_PAPER: "entry j holds the ASAD at 180 - PA_IN[j]",
}
_ANGLE_TOL = 1e-6


def _import_uproot():
    try:
        import uproot
    except ImportError as exc:
        raise AdapterError(
            "The prsp export from ROOT needs the optional dependency "
            "'uproot'. Install it with: pip install menhag[root]"
        ) from exc
    return uproot


def _import_fits():
    try:
        from astropy.io import fits
    except ImportError as exc:
        raise AdapterError(
            "The prsp adapter needs the optional dependency 'astropy'. "
            "Install it with: pip install menhag[fits]"
        ) from exc
    return fits


def check_layout(layout: str) -> str:
    """Return the layout name, or raise for a name that is not one of
    :data:`LAYOUTS`."""
    if layout not in LAYOUTS:
        raise AdapterError(
            f"Unknown .prsp layout {layout!r}; the layouts are {LAYOUTS}: "
            f"'{LAYOUT_CURRENT}' for the current polpy release and "
            f"'{LAYOUT_PAPER}' for the earlier plug-in behind the fits of "
            "the accompanying paper."
        )
    return layout


def layout_index_map(pa_vals, layout: str) -> np.ndarray:
    """Which simulated angle each file entry holds.

    Returns an integer array ``src`` such that file entry ``j`` holds the
    ASAD simulated at ``pa_vals[src[j]]``. In the current layout ``src`` is
    the identity; in the paper layout it is the reflection
    ``pa -> 180 - pa``, which needs an angle set closed under that
    reflection (the 3 degree grid from 0 to 180 degrees is).
    """
    pa_vals = np.asarray(pa_vals, dtype=float)
    layout = check_layout(layout)
    if layout == LAYOUT_CURRENT:
        return np.arange(pa_vals.size)
    src = np.empty(pa_vals.size, dtype=int)
    for j, pa in enumerate(pa_vals):
        hits = np.flatnonzero(np.abs(pa_vals - (180.0 - pa)) < _ANGLE_TOL)
        if hits.size != 1:
            raise AdapterError(
                f"The '{LAYOUT_PAPER}' layout stores entry j at "
                "180 - PA_IN[j] degrees, so the angle set must be closed "
                f"under that reflection; {180.0 - pa:g} degrees is missing "
                f"for the entry at {pa:g} degrees."
            )
        src[j] = hits[0]
    return src


def uproot_th1_to_nparray(file_name, hist_name):
    """Read one TH1 histogram's bin contents as float32."""
    uproot = _import_uproot()
    with uproot.open(file_name) as open_file:
        hist = open_file[hist_name]
        values, edges = hist.to_numpy()
    return values.astype(np.float32)


def closed_angle_grid(step_deg: float = 3.0) -> np.ndarray:
    """The angle labels of a .prsp: 0 to 180 degrees inclusive in the
    given step (61 labels at the POLAR step of 3 degrees)."""
    n = int(round(180.0 / float(step_deg)))
    if abs(n * float(step_deg) - 180.0) > _ANGLE_TOL:
        raise AdapterError(f"The step {step_deg} degrees does not divide 180.")
    return np.linspace(0.0, 180.0, n + 1)


def _ensure_parent(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _stamp(header, telescope, instrument):
    from .. import __version__

    header["TELESCOP"] = (telescope, "mission/telescope name")
    header["INSTRUME"] = (instrument, "instrument/detector name")
    header["HDUCLASS"] = ("OGIP", "format conforms to OGIP standard")
    header["HDUCLAS1"] = ("RESPONSE", "dataset relates to polarimetric inputs")
    header["ORIGIN"] = ("MENHAG", "generating package")
    header["CREATOR"] = (f"MENHAG {__version__}", "generating program")
    header["MENHAGV"] = (__version__, "menhag version")
    header["DATE"] = (datetime.datetime.now().isoformat(), "generating date")


def write_prsp(
    out_path: str,
    energ_lo,
    energ_hi,
    pa_vals,
    pol_sim,
    unp,
    sa_edges=None,
    layout: str = DEFAULT_LAYOUT,
    telescope: str = "Tiangong-2",
    instrument: str = "POLAR",
    sa_norm: str = "cm2",
) -> str:
    """Write a .prsp from arrays in simulation order.

    Parameters
    ----------
    energ_lo, energ_hi : arrays of length nE
        The Monte Carlo energy bin edges in keV.
    pa_vals : array of length nPA
        The polarization angle labels in degrees (INPAVALS).
    pol_sim : array (nE, nPA, nSA)
        ``pol_sim[i, k]`` is the fully polarized ASAD simulated at energy
        i and at the angle ``pa_vals[k]``: simulation order, whatever the
        layout written.
    unp : array (nE, nSA)
        The unpolarized ASADs.
    sa_edges : array of length nSA + 1, optional
        Scattering-angle bin edges in degrees; default one-degree bins.
    layout : str
        ``"polpy-1.2.1"`` (default) or ``"polarpy"``; see the module
        docstring. Recorded in the PALAYOUT keyword.
    telescope, instrument : str
        The TELESCOP and INSTRUME keywords. The defaults are the POLAR
        values of the accompanying paper.
    sa_norm : {"cm2", "one"}
        The SANORM convention of the matrices (effective area or unity).
    """
    fits = _import_fits()
    layout = check_layout(layout)
    energ_lo = np.asarray(energ_lo, dtype=np.float32)
    energ_hi = np.asarray(energ_hi, dtype=np.float32)
    pa_vals = np.asarray(pa_vals, dtype=np.float32)
    pol_sim = np.asarray(pol_sim, dtype=np.float32)
    unp = np.asarray(unp, dtype=np.float32)
    n_e, n_pa, n_sa = pol_sim.shape
    if energ_lo.size != n_e or energ_hi.size != n_e:
        raise AdapterError("energ_lo and energ_hi must have one entry per energy.")
    if pa_vals.size != n_pa:
        raise AdapterError("pa_vals must have one entry per polarized angle.")
    if unp.shape != (n_e, n_sa):
        raise AdapterError("unp must have shape (nE, nSA).")
    if sa_edges is None:
        sa_edges = np.linspace(0.0, 360.0, n_sa + 1)
    sa_edges = np.asarray(sa_edges, dtype=float)
    if sa_edges.size != n_sa + 1:
        raise AdapterError("sa_edges must have nSA + 1 entries.")
    sa_min = sa_edges[:-1].astype(np.float32)
    sa_max = sa_edges[1:].astype(np.float32)
    sa_bin = np.arange(n_sa, dtype=np.int16)

    src = layout_index_map(pa_vals, layout)
    pol_file = pol_sim[:, src, :]

    hdu_e = fits.BinTableHDU.from_columns(fits.ColDefs([
        fits.Column(name="ENERG_LO", format="E", unit="keV", array=energ_lo),
        fits.Column(name="ENERG_HI", format="E", unit="keV", array=energ_hi),
    ]))
    hdu_e.header["EXTNAME"] = "INEBOUNDS"
    hdu_e.header["CHANTYPE"] = ("PI", "Channel type")
    hdu_e.header["DETCHANS"] = n_e
    hdu_e.header["EBINS"] = n_e
    hdu_e.header["HDUCLAS2"] = "INEBOUNDS"
    _stamp(hdu_e.header, telescope, instrument)

    hdu_pa = fits.BinTableHDU.from_columns(fits.ColDefs([
        fits.Column(name="PA_IN", format="E", unit="deg", array=pa_vals),
    ]))
    hdu_pa.header["EXTNAME"] = "INPAVALS"
    hdu_pa.header["PABINS"] = n_pa
    hdu_pa.header["HDUCLAS2"] = "INPAVALS"
    hdu_pa.header[LAYOUT_KEYWORD] = (layout, _LAYOUT_COMMENT[layout])
    _stamp(hdu_pa.header, telescope, instrument)

    hdu_sa = fits.BinTableHDU.from_columns(fits.ColDefs([
        fits.Column(name="SA", format="I", unit="   ", array=sa_bin),
        fits.Column(name="SA_MIN", format="E", unit="deg", array=sa_min),
        fits.Column(name="SA_MAX", format="E", unit="deg", array=sa_max),
    ]))
    hdu_sa.header["EXTNAME"] = "SABOUNDS"
    hdu_sa.header["SABINS"] = n_sa
    hdu_sa.header["HDUCLAS2"] = "INSAVALS"
    _stamp(hdu_sa.header, telescope, instrument)

    def _matrix(data, name, dim):
        hdu = fits.ImageHDU(data=data)
        hdu.header["EXTNAME"] = name
        hdu.header["IMAGEDIM"] = (dim, "axis order of the stored matrix")
        hdu.header["EMIN"] = float(np.min(energ_lo))
        hdu.header["EMAX"] = float(np.max(energ_hi))
        hdu.header["EBINS"] = n_e
        hdu.header["SABINS"] = n_sa
        if sa_norm == "cm2":
            hdu.header["SANORM"] = ("True", "The integration is total ARF.")
            hdu.header["SAUNIT"] = "cm2"
        elif sa_norm == "one":
            hdu.header["SANORM"] = ("True", "The integration is ONE.")
            hdu.header["SAUNIT"] = "1"
        else:
            raise AdapterError("sa_norm must be 'cm2' or 'one'.")
        hdu.header["HDUCLAS2"] = name
        _stamp(hdu.header, telescope, instrument)
        return hdu

    hdu_pol = _matrix(pol_file.transpose(), "SPECRESP POLMATRIX", "3D")
    hdu_pol.header["PABINS"] = n_pa
    hdu_pol.header[LAYOUT_KEYWORD] = (layout, _LAYOUT_COMMENT[layout])
    hdu_unp = _matrix(unp.transpose(), "SPECRESP UNPOLMATRIX", "2D")

    _ensure_parent(out_path)
    fits.HDUList([fits.PrimaryHDU(), hdu_e, hdu_pa, hdu_sa, hdu_pol, hdu_unp]).writeto(
        out_path, overwrite=True
    )
    logger.info("Wrote .prsp response (%s layout): %s", layout, out_path)
    return out_path


def read_prsp(path: str, layout: Optional[str] = None) -> dict:
    """Read a .prsp into simulation order.

    Returns a dict with ``energ_lo``, ``energ_hi`` (keV), ``pa_vals``
    (degrees), ``sa_edges`` (degrees, nSA + 1), ``pol`` of shape
    (nE, nPA, nSA) with ``pol[i, k]`` the ASAD simulated at ``pa_vals[k]``
    whatever order the file stored, ``unp`` of shape (nE, nSA), and
    ``layout``, the layout the file was read in.

    The layout comes from the PALAYOUT keyword when the file carries one.
    Otherwise it is the ``layout`` argument, and when that is not given
    either, the default layout with a logged warning. A ``layout`` argument
    that contradicts the keyword raises.
    """
    fits = _import_fits()
    if not os.path.exists(path):
        raise AdapterError(f".prsp file not found: {path}")
    with fits.open(path) as hdul:
        names = [h.name for h in hdul]
        for needed in ("INEBOUNDS", "INPAVALS", "SABOUNDS",
                       "SPECRESP POLMATRIX", "SPECRESP UNPOLMATRIX"):
            if needed not in names:
                raise AdapterError(
                    f"{path} has no extension {needed!r}; it is not a "
                    ".prsp polarization response."
                )
        energ_lo = np.asarray(hdul["INEBOUNDS"].data["ENERG_LO"], dtype=float)
        energ_hi = np.asarray(hdul["INEBOUNDS"].data["ENERG_HI"], dtype=float)
        pa_vals = np.asarray(hdul["INPAVALS"].data["PA_IN"], dtype=float)
        sa_min = np.asarray(hdul["SABOUNDS"].data["SA_MIN"], dtype=float)
        sa_max = np.asarray(hdul["SABOUNDS"].data["SA_MAX"], dtype=float)
        pol_file = np.asarray(hdul["SPECRESP POLMATRIX"].data, dtype=float).transpose()
        unp = np.asarray(hdul["SPECRESP UNPOLMATRIX"].data, dtype=float).transpose()
        stored = hdul["INPAVALS"].header.get(LAYOUT_KEYWORD)
        if stored is None:
            stored = hdul["SPECRESP POLMATRIX"].header.get(LAYOUT_KEYWORD)
    if stored is not None:
        stored = check_layout(str(stored).strip())
        if layout is not None and check_layout(layout) != stored:
            raise AdapterError(
                f"{path} records the layout '{stored}' in its {LAYOUT_KEYWORD} "
                f"keyword, but layout='{layout}' was requested."
            )
        layout = stored
    elif layout is None:
        layout = DEFAULT_LAYOUT
        logger.warning(
            "%s carries no %s keyword; reading it in the '%s' layout. Pass "
            "layout='%s' if it was written for the earlier plug-in.",
            path, LAYOUT_KEYWORD, layout, LAYOUT_PAPER,
        )
    else:
        layout = check_layout(layout)
    if pol_file.shape[:2] != (energ_lo.size, pa_vals.size):
        raise AdapterError(
            f"{path}: SPECRESP POLMATRIX has shape {pol_file.shape} but the "
            f"file lists {energ_lo.size} energies and {pa_vals.size} angles."
        )
    src = layout_index_map(pa_vals, layout)
    pol_sim = np.empty_like(pol_file)
    pol_sim[:, src, :] = pol_file
    sa_edges = np.concatenate([sa_min, sa_max[-1:]])
    return {
        "energ_lo": energ_lo,
        "energ_hi": energ_hi,
        "pa_vals": pa_vals,
        "sa_edges": sa_edges,
        "pol": pol_sim,
        "unp": unp,
        "layout": layout,
    }


def convert_prsp(events_root, rmfarf_root, out_path, layout: str = DEFAULT_LAYOUT,
                 telescope: str = "Tiangong-2", instrument: str = "POLAR"):
    """Export a ROOT response to a .prsp.

    ``events_root`` is a flat ROOT file in the sim_* key layout (the
    package's default ROOT format; see :mod:`menhag.adapters.polar_root`)
    holding the unpolarized ASAD and the fully polarized ASADs at 3 degree
    steps; ``rmfarf_root`` supplies the Monte Carlo energy bin edges (the
    ER_low and ER_high histograms of the instrument's RMF/ARF file).
    ``layout`` selects the angle order written (see the module docstring);
    the default is the current polpy layout. The 180 degree entry is
    written as a copy of the 0 degree ASAD; an energy or angle missing
    from the file leaves zeros, as the response files of the paper do.
    """
    uproot = _import_uproot()
    layout = check_layout(layout)
    eloMC = uproot_th1_to_nparray(rmfarf_root, "ER_low")
    ehiMC = uproot_th1_to_nparray(rmfarf_root, "ER_high")
    pa_vals = closed_angle_grid(3.0)
    n_e, n_pa = eloMC.size, pa_vals.size
    e_center = (eloMC + ehiMC) / 2.0

    with uproot.open(events_root) as handle:
        n_sa = None
        for i in range(n_e):
            name = "sim_%03d_000_000" % int(e_center[i])
            if name in handle:
                values, edges = handle[name].to_numpy()
                n_sa = values.size
                sa_edges = np.asarray(edges, dtype=float)
                break
        if n_sa is None:
            raise AdapterError(
                f"{events_root} holds no unpolarized histogram at any of the "
                "energies of the RMF/ARF file."
            )
        pol_sim = np.zeros((n_e, n_pa, n_sa), dtype=np.float32)
        unp = np.zeros((n_e, n_sa), dtype=np.float32)
        for i in range(n_e):
            for j in range(n_pa):
                angle = int(round(pa_vals[j]))
                if angle == 180:
                    pol_sim[i, j, :] = pol_sim[i, 0, :]
                    continue
                name = "sim_%03d_100_%03d" % (int(e_center[i]), angle)
                if name in handle:
                    values, _ = handle[name].to_numpy()
                    pol_sim[i, j, :] = values.astype(np.float32)
            name = "sim_%03d_000_000" % int(e_center[i])
            if name in handle:
                values, _ = handle[name].to_numpy()
                unp[i, :] = values.astype(np.float32)

    return write_prsp(out_path, eloMC, ehiMC, pa_vals, pol_sim, unp,
                      sa_edges=sa_edges, layout=layout,
                      telescope=telescope, instrument=instrument)


class PrspAdapter(InstrumentAdapter):
    """Adapter reading a .prsp in either layout and writing one.

    Parameters
    ----------
    input_path : str
        The .prsp file holding the unpolarized ASADs (and, for mode 2,
        the fully polarized ones).
    output_path : str, optional
        Destination for :meth:`save_response`. The output takes its
        energy bin edges from the input file.
    layout : str, optional
        The angle order of the input file, needed only when the file
        carries no PALAYOUT keyword; see :func:`read_prsp`.
    output_layout : str
        The angle order written by :meth:`save_response` (default the
        current polpy layout).
    norm_factor : float, optional
        Count normalization f of the stored matrices: when given, the
        per-bin errors of every ASAD read are the Poisson errors
        sqrt(n f) / f; otherwise no errors are attached and the build
        uses the configuration's ``norm_factor``.
    telescope, instrument : str
        The TELESCOP and INSTRUME keywords of the output.
    """

    def __init__(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        layout: Optional[str] = None,
        output_layout: str = DEFAULT_LAYOUT,
        norm_factor: Optional[float] = None,
        telescope: str = "Tiangong-2",
        instrument: str = "POLAR",
    ) -> None:
        self.input_path = input_path
        self.output_path = output_path
        self.layout = None if layout is None else check_layout(layout)
        self.output_layout = check_layout(output_layout)
        self.norm_factor = None if norm_factor is None else float(norm_factor)
        self.telescope = telescope
        self.instrument = instrument
        self._data = None

    def _load(self) -> dict:
        if self._data is None:
            self._data = read_prsp(self.input_path, self.layout)
            self.layout = self._data["layout"]
        return self._data

    def _energy_index(self, energy: float) -> int:
        d = self._load()
        centers = 0.5 * (d["energ_lo"] + d["energ_hi"])
        half = 0.5 * (d["energ_hi"] - d["energ_lo"])
        hits = np.flatnonzero(np.abs(centers - float(energy)) <= np.maximum(half, 1e-6))
        if hits.size == 0:
            raise MissingEnergyError(
                f"No energy bin of {self.input_path} contains {energy} keV."
            )
        return int(hits[np.argmin(np.abs(centers[hits] - float(energy)))])

    def _asad(self, counts, energy: float, meta: dict) -> ASAD:
        d = self._load()
        errors = None
        if self.norm_factor is not None:
            errors = poisson_errors(counts, self.norm_factor)
        return ASAD(counts=np.asarray(counts, dtype=float), bin_edges=d["sa_edges"],
                    errors=errors, energy=float(energy),
                    meta=dict(meta, source=self.input_path, layout=d["layout"]))

    def load_unpolarized(self, energy: float) -> ASAD:
        d = self._load()
        i = self._energy_index(energy)
        return self._asad(d["unp"][i], energy, {"energy_index": i})

    def load_polarized(self, energy: float, phi0: float) -> ASAD:
        d = self._load()
        i = self._energy_index(energy)
        hits = np.flatnonzero(np.abs(d["pa_vals"] - float(phi0)) < _ANGLE_TOL)
        if hits.size == 0:
            raise MissingEnergyError(
                f"{self.input_path} holds no polarized entry at {phi0} degrees; "
                f"its angles are {d['pa_vals'].tolist()}."
            )
        k = int(hits[0])
        return self._asad(d["pol"][i, k], energy, {"energy_index": i, "angle_index": k})

    def save_response(self, response: ResponseSet) -> None:
        if self.output_path is None:
            raise AdapterError("PrspAdapter has no output_path; pass one to save.")
        d = self._load()
        write_response_prsp(self.output_path, response, d["energ_lo"], d["energ_hi"],
                            layout=self.output_layout, telescope=self.telescope,
                            instrument=self.instrument)


def write_response_prsp(
    out_path: str,
    response: ResponseSet,
    energ_lo,
    energ_hi,
    layout: str = DEFAULT_LAYOUT,
    telescope: str = "Tiangong-2",
    instrument: str = "POLAR",
) -> str:
    """Write a :class:`ResponseSet` as a .prsp.

    The energy grid of the file is ``energ_lo``, ``energ_hi``; the
    response's energies are placed in the bins that contain them, and bins
    the response does not cover are left at zero with a logged warning.
    The angle labels are the response's polarization angles; when the
    response holds 0 degrees and no 180 degree entry, the 180 degree entry
    is added as a copy of the 0 degree ASAD. Only a fully polarized
    response (``pol_fraction`` 1) can be written, since the polarized
    matrix of the format is the 100 percent polarized one.
    """
    if abs(float(response.pol_fraction) - 1.0) > 1e-9:
        raise AdapterError(
            "A .prsp stores fully polarized ASADs; the response has "
            f"pol_fraction = {response.pol_fraction}."
        )
    energ_lo = np.asarray(energ_lo, dtype=float)
    energ_hi = np.asarray(energ_hi, dtype=float)
    centers = 0.5 * (energ_lo + energ_hi)
    half = 0.5 * (energ_hi - energ_lo)
    angles = sorted({float(a) for (_, a) in response.polarized})
    if not angles:
        raise AdapterError("The response holds no polarized ASAD.")
    duplicate_180 = (
        any(abs(a) < _ANGLE_TOL for a in angles)
        and not any(abs(a - 180.0) < _ANGLE_TOL for a in angles)
    )
    pa_vals = np.array(angles + ([180.0] if duplicate_180 else []), dtype=float)
    first = next(iter(response.polarized.values()))
    n_sa = first.counts.size
    sa_edges = first.bin_edges
    n_e, n_pa = centers.size, pa_vals.size
    pol_sim = np.zeros((n_e, n_pa, n_sa), dtype=np.float32)
    unp = np.zeros((n_e, n_sa), dtype=np.float32)
    filled = np.zeros(n_e, dtype=bool)

    def _bin(energy):
        hits = np.flatnonzero(np.abs(centers - float(energy)) <= np.maximum(half, 1e-6))
        if hits.size == 0:
            raise AdapterError(
                f"The response energy {energy} keV falls in no energy bin "
                "of the output grid."
            )
        return int(hits[np.argmin(np.abs(centers[hits] - float(energy)))])

    for energy, asad in response.unpolarized.items():
        if energy is None:
            continue
        i = _bin(energy)
        unp[i] = asad.counts
        filled[i] = True
    for (energy, phi0), asad in response.polarized.items():
        if energy is None:
            continue
        if asad.counts.size != n_sa or not np.allclose(asad.bin_edges, sa_edges):
            raise AdapterError("Every ASAD of a .prsp must share one binning.")
        i = _bin(energy)
        k = int(np.flatnonzero(np.abs(pa_vals - float(phi0)) < _ANGLE_TOL)[0])
        pol_sim[i, k] = asad.counts
        filled[i] = True
    if duplicate_180:
        k0 = int(np.flatnonzero(np.abs(pa_vals) < _ANGLE_TOL)[0])
        pol_sim[:, -1, :] = pol_sim[:, k0, :]
    if not filled.all():
        logger.warning(
            "%d of the %d energy bins of %s have no ASAD in the response "
            "and are written as zeros.", int((~filled).sum()), n_e, out_path,
        )
    return write_prsp(out_path, energ_lo, energ_hi, pa_vals, pol_sim, unp,
                      sa_edges=sa_edges, layout=layout,
                      telescope=telescope, instrument=instrument)
