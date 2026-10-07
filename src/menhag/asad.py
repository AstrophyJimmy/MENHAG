# MENHAG, Mohamed Gamal Zaid, University of New Hampshire. MIT license, see LICENSE.
"""Containers for azimuthal scattering angle distributions and responses.

An ASAD (azimuthal scattering angle distribution, also called a modulation
curve) is a histogram of measured azimuthal scattering angles eta. The
:class:`ASAD` dataclass carries the counts, the bin edges in degrees, the
per-bin errors, and optional metadata. Output ASADs always inherit the bin
edges of the input ASAD they were derived from; menhag never re-bins.

Error convention (paper Eq. 31): a histogram whose raw contents were divided
by a normalization factor f carries per-bin Poisson uncertainties

    sigma_i = sqrt(n_i * f) / f

where n_i are the stored (already normalized) bin contents. With f = 1 this
reduces to the familiar sqrt(n_i).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def poisson_errors(counts: np.ndarray, norm_factor: float = 1.0) -> np.ndarray:
    """Per-bin Poisson uncertainty of normalized histogram contents (Eq. 31).

    Parameters
    ----------
    counts : array_like
        Stored bin contents, i.e. raw counts divided by ``norm_factor``.
    norm_factor : float
        The normalization factor f that was applied to the raw counts.

    Examples
    --------
    >>> import numpy as np
    >>> poisson_errors(np.array([100.0]), norm_factor=1.0)
    array([10.])
    """
    counts = np.asarray(counts, dtype=float)
    if norm_factor <= 0:
        raise ValueError(f"norm_factor must be positive, got {norm_factor}.")
    return np.sqrt(np.clip(counts, 0.0, None) * norm_factor) / norm_factor


@dataclass
class ASAD:
    """One azimuthal scattering angle distribution.

    Attributes
    ----------
    counts : ndarray
        Bin contents.
    bin_edges : ndarray
        Bin edges in degrees, length ``len(counts) + 1``, monotonically
        increasing.
    errors : ndarray, optional
        Per-bin uncertainties. If not supplied, use
        :meth:`with_poisson_errors` to attach the Eq. 31 convention.
    energy : float, optional
        Photon energy in keV for monoenergetic ASADs.
    meta : dict
        Free-form metadata (source angles, threshold, provenance).
    """

    counts: np.ndarray
    bin_edges: np.ndarray
    errors: Optional[np.ndarray] = None
    energy: Optional[float] = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.counts = np.asarray(self.counts, dtype=float)
        self.bin_edges = np.asarray(self.bin_edges, dtype=float)
        if self.counts.ndim != 1:
            raise ValueError("ASAD counts must be one-dimensional.")
        if self.bin_edges.shape != (self.counts.size + 1,):
            raise ValueError(
                f"bin_edges must have length len(counts) + 1 = "
                f"{self.counts.size + 1}, got {self.bin_edges.size}."
            )
        if np.any(np.diff(self.bin_edges) <= 0):
            raise ValueError("bin_edges must be strictly increasing.")
        if self.errors is not None:
            self.errors = np.asarray(self.errors, dtype=float)
            if self.errors.shape != self.counts.shape:
                raise ValueError("errors must have the same shape as counts.")

    @property
    def bin_centers(self) -> np.ndarray:
        """Bin centers in degrees."""
        return 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])

    @property
    def bin_widths(self) -> np.ndarray:
        """Bin widths in degrees."""
        return np.diff(self.bin_edges)

    @property
    def total_counts(self) -> float:
        """Sum of the bin contents."""
        return float(self.counts.sum())

    def with_poisson_errors(self, norm_factor: float = 1.0) -> "ASAD":
        """Return a copy carrying Eq. 31 Poisson errors (see module docs)."""
        return ASAD(
            counts=self.counts.copy(),
            bin_edges=self.bin_edges.copy(),
            errors=poisson_errors(self.counts, norm_factor),
            energy=self.energy,
            meta=dict(self.meta),
        )


@dataclass
class ResponseSet:
    """A polarization response: ASADs keyed by energy and polarization angle.

    Attributes
    ----------
    unpolarized : dict
        ``{energy_kev: ASAD}`` for the unpolarized inputs (or the
        spectrum-integrated unpolarized ASAD under the key ``None``).
    polarized : dict
        ``{(energy_kev, phi0_deg): ASAD}`` for the generated polarized
        ASADs. Spectrum-integrated outputs use ``(None, phi0_deg)``.
    pol_fraction : float
        Polarization fraction of the generated polarized ASADs (1.0 means
        fully polarized).
    meta : dict
        Provenance: source angles, threshold gate, engine settings.
    """

    unpolarized: dict = field(default_factory=dict)
    polarized: dict = field(default_factory=dict)
    pol_fraction: float = 1.0
    meta: dict = field(default_factory=dict)

    @property
    def energies(self) -> list:
        """Sorted list of monoenergetic energies present (excludes None)."""
        return sorted(e for e in self.unpolarized if e is not None)

    @property
    def pol_angles(self) -> list:
        """Sorted list of polarization angles present in the output."""
        return sorted({phi0 for (_, phi0) in self.polarized})
