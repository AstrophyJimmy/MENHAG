# MENHAG, Mohamed Gamal Zaid, University of New Hampshire. MIT license, see LICENSE.
"""Adapter interface: how menhag reads your ASADs and writes your response.

All file I/O is isolated behind :class:`InstrumentAdapter`. Three concrete
adapters ship with the package (ROOT, FITS, CSV) plus an in-memory
passthrough and a heavily commented template to copy and edit for any other
format (:mod:`menhag.adapters.template`).

An adapter needs two methods:

``load_unpolarized(energy) -> ASAD``
    Return the unpolarized ASAD at that energy (counts, bin edges in
    degrees, optionally errors). Raise
    :class:`menhag.exceptions.MissingEnergyError` if unavailable; menhag
    never fabricates an unpolarized ASAD.

``save_response(response_set) -> None``
    Persist a :class:`menhag.asad.ResponseSet` in your format of choice.

One optional method enables mode 2:

``load_polarized(energy, phi0) -> ASAD``
    Return a measured or simulated fully polarized ASAD at the given
    polarization angle. The default raises ``NotImplementedError``.

Extension point: an OGIP-compliant output adapter can be added later by
subclassing :class:`InstrumentAdapter` and implementing ``save_response``
against the OGIP RSP conventions; nothing in the engine depends on the
output format.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping, Optional, Sequence, Union

import numpy as np

from ..asad import ASAD, ResponseSet
from ..exceptions import MissingEnergyError


class InstrumentAdapter(ABC):
    """Abstract base for instrument I/O adapters (see module docstring)."""

    @abstractmethod
    def load_unpolarized(self, energy: float) -> ASAD:
        """Return the unpolarized ASAD at ``energy`` keV."""

    @abstractmethod
    def save_response(self, response: ResponseSet) -> None:
        """Persist a generated response set."""

    def load_polarized(self, energy: float, phi0: float) -> ASAD:
        """Return a fully polarized input ASAD (mode 2 only)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement load_polarized; "
            "mode 2 needs an adapter that can read polarized ASADs."
        )


class ArrayAdapter(InstrumentAdapter):
    """In-memory passthrough for users who already hold their ASADs.

    Parameters
    ----------
    unpolarized : mapping
        ``{energy_kev: ASAD}`` or ``{energy_kev: (counts, bin_edges)}`` or
        ``{energy_kev: (counts, bin_edges, errors)}``.
    polarized : mapping, optional
        ``{(energy_kev, phi0_deg): same forms}`` for mode 2.

    Saved responses are collected on the ``saved`` attribute instead of
    being written anywhere.

    Examples
    --------
    >>> import numpy as np
    >>> edges = np.linspace(0, 360, 61)
    >>> adapter = ArrayAdapter({200.0: (np.full(60, 1000.0), edges)})
    >>> adapter.load_unpolarized(200.0).total_counts
    60000.0
    """

    def __init__(
        self,
        unpolarized: Mapping[float, Union[ASAD, Sequence]],
        polarized: Optional[Mapping[tuple, Union[ASAD, Sequence]]] = None,
    ) -> None:
        self._unpolarized = {float(e): _coerce(v, e) for e, v in unpolarized.items()}
        self._polarized = {
            (float(e), float(a)): _coerce(v, e)
            for (e, a), v in (polarized or {}).items()
        }
        self.saved: list[ResponseSet] = []

    def load_unpolarized(self, energy: float) -> ASAD:
        try:
            return self._unpolarized[float(energy)]
        except KeyError:
            available = sorted(self._unpolarized)
            raise MissingEnergyError(
                f"No unpolarized ASAD at {energy} keV in the ArrayAdapter; "
                f"available energies: {available}."
            ) from None

    def load_polarized(self, energy: float, phi0: float) -> ASAD:
        try:
            return self._polarized[(float(energy), float(phi0))]
        except KeyError:
            raise MissingEnergyError(
                f"No polarized ASAD at ({energy} keV, phi0 = {phi0} deg) in "
                "the ArrayAdapter."
            ) from None

    def save_response(self, response: ResponseSet) -> None:
        self.saved.append(response)


def _coerce(value: Union[ASAD, Sequence], energy: float) -> ASAD:
    """Accept an ASAD or a (counts, edges[, errors]) tuple."""
    if isinstance(value, ASAD):
        if value.energy is None:
            value.energy = float(energy)
        return value
    parts = tuple(value)
    if len(parts) == 2:
        counts, edges = parts
        errors = None
    elif len(parts) == 3:
        counts, edges, errors = parts
    else:
        raise ValueError(
            "ArrayAdapter values must be an ASAD or a (counts, bin_edges) "
            "or (counts, bin_edges, errors) tuple."
        )
    return ASAD(
        counts=np.asarray(counts, dtype=float),
        bin_edges=np.asarray(edges, dtype=float),
        errors=None if errors is None else np.asarray(errors, dtype=float),
        energy=float(energy),
    )
