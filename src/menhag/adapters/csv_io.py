# MENHAG, Mohamed Gamal Zaid, University of New Hampshire. MIT license, see LICENSE.
"""CSV I/O with an explicit schema (requires pandas: pip install menhag[csv]).

Input schema (one file, long format)
------------------------------------
Required columns::

    energy, bin_low, bin_high, counts

Optional columns::

    error       per-bin uncertainty (Poisson assumed when absent)
    pol, angle  polarization degree in percent and phi0 in degrees for
                polarized input rows (mode 2); unpolarized rows may
                omit them or use pol = 0, angle = 0

Rows for one ASAD must appear in increasing bin_low order and tile the
azimuth without gaps. Energies are matched within 1e-6 keV.

Output schema
-------------
One file with columns::

    energy, pol_fraction, pol_angle, bin_low, bin_high, counts, error

Unpolarized inputs are included with pol_fraction = 0 and pol_angle = 0.
Spectrum-integrated rows carry an empty energy field.
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

_ENERGY_TOLERANCE_KEV = 1e-6


def _import_pandas():
    try:
        import pandas
    except ImportError as exc:
        raise AdapterError(
            "The CSV adapter needs the optional dependency 'pandas'. "
            "Install it with: pip install menhag[csv]"
        ) from exc
    return pandas


def _rows_to_asad(rows, energy: float, path: str) -> ASAD:
    rows = rows.sort_values("bin_low")
    lo = rows["bin_low"].to_numpy(dtype=float)
    hi = rows["bin_high"].to_numpy(dtype=float)
    if not np.allclose(lo[1:], hi[:-1]):
        raise AdapterError(
            f"CSV bins for {energy} keV in {path} do not tile contiguously; "
            "check bin_low and bin_high."
        )
    edges = np.concatenate([lo, hi[-1:]])
    counts = rows["counts"].to_numpy(dtype=float)
    errors = (
        rows["error"].to_numpy(dtype=float) if "error" in rows.columns else None
    )
    return ASAD(counts=counts, bin_edges=edges, errors=errors, energy=energy,
                meta={"source": path})


class CsvAdapter(InstrumentAdapter):
    """Adapter for the CSV schema documented in this module.

    Parameters
    ----------
    input_path : str
        Input CSV file.
    output_path : str, optional
        Destination CSV for :meth:`save_response`.

    Examples
    --------
    >>> adapter = CsvAdapter("unpolarized.csv", "response.csv")  # doctest: +SKIP
    """

    def __init__(self, input_path: str, output_path: Optional[str] = None) -> None:
        self.input_path = input_path
        self.output_path = output_path
        self._table = None

    def _load_table(self):
        pandas = _import_pandas()
        if self._table is None:
            if not os.path.exists(self.input_path):
                raise AdapterError(f"CSV file not found: {self.input_path}")
            table = pandas.read_csv(self.input_path)
            required = {"energy", "bin_low", "bin_high", "counts"}
            missing = required - set(table.columns)
            if missing:
                raise AdapterError(
                    f"CSV {self.input_path} is missing required columns "
                    f"{sorted(missing)}; the schema is documented in "
                    "menhag.adapters.csv_io."
                )
            self._table = table
        return self._table

    def _select(self, energy: float, pol: Optional[int], angle: Optional[float]):
        table = self._load_table()
        mask = (table["energy"] - float(energy)).abs() < _ENERGY_TOLERANCE_KEV
        if pol is not None and "pol" in table.columns:
            mask &= table["pol"].fillna(0).astype(int) == int(pol)
        if angle is not None and "angle" in table.columns:
            mask &= (table["angle"].fillna(0) - float(angle)).abs() < 1e-6
        return table[mask]

    def load_unpolarized(self, energy: float) -> ASAD:
        rows = self._select(energy, pol=0, angle=0.0)
        if len(rows) == 0:
            raise MissingEnergyError(
                f"No unpolarized rows at {energy} keV in {self.input_path}."
            )
        return _rows_to_asad(rows, float(energy), self.input_path)

    def load_polarized(self, energy: float, phi0: float) -> ASAD:
        rows = self._select(energy, pol=100, angle=phi0)
        if len(rows) == 0:
            raise MissingEnergyError(
                f"No polarized rows at ({energy} keV, phi0 = {phi0} deg) in "
                f"{self.input_path}; mode 2 expects pol = 100 rows with "
                "an 'angle' column."
            )
        return _rows_to_asad(rows, float(energy), self.input_path)

    def save_response(self, response: ResponseSet) -> None:
        pandas = _import_pandas()
        if self.output_path is None:
            raise AdapterError("CsvAdapter has no output_path; pass one to save.")
        records = []

        def _emit(asad: ASAD, energy, pol_fraction: float, phi0: float) -> None:
            errors = (
                asad.errors
                if asad.errors is not None
                else np.zeros_like(asad.counts)
            )
            for lo, hi, c, s in zip(
                asad.bin_edges[:-1], asad.bin_edges[1:], asad.counts, errors
            ):
                records.append(
                    {
                        "energy": energy,
                        "pol_fraction": pol_fraction,
                        "pol_angle": phi0,
                        "bin_low": lo,
                        "bin_high": hi,
                        "counts": c,
                        "error": s,
                    }
                )

        for energy, asad in response.unpolarized.items():
            _emit(asad, energy, 0.0, 0.0)
        for (energy, phi0), asad in response.polarized.items():
            _emit(asad, energy, response.pol_fraction, phi0)
        pandas.DataFrame.from_records(records).to_csv(self.output_path, index=False)
        logger.info("Wrote CSV response: %s", self.output_path)
