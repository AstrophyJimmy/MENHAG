# MENHAG, Mohamed Gamal Zaid, University of New Hampshire. MIT license, see LICENSE.
"""Compton-coincidence threshold gate models and the gate JSON schema.

A polarimetry (two-hit) event requires the recoil electron and the
scattered photon to each deposit energy above the instrument trigger
threshold. Inside the Klein-Nishina integrals this appears as a gate
weight applied to the integrand, where the two deposits are fixed by
Compton kinematics:

    E_prime = E_gamma / (1 + eps * (1 - cos theta)),  eps = E_gamma / 511 keV
    E_e     = E_gamma - E_prime

Gate families (evaluated by :mod:`menhag.physics`):

``ErfGate(mu, sigma)``  (recommended)
    The continuous error-function gate on the SHARED per-event threshold:
    w = 0.5 * (1 + erf((min(E_prime, E_e) - mu) / (sqrt(2) * sigma))).
    ``mu`` is the average detection threshold of the instrument (keV) and
    ``sigma`` the effective width of the turn-on (keV). It is the exact
    continuum limit of averaging sharp-threshold gates over a Gaussian
    effective-threshold distribution with physical acceptance weighting,
    and is free of the window-closure steps of the sampled mixture
    below.

    How to choose mu and sigma: use the average detection threshold of
    the instrument for mu, and for sigma a width representative of
    everything that smears the turn-on. For a clean instrument sigma is
    simply the energy resolution near threshold (the Gaussian broadening
    of a peak there). For an instrument like POLAR, where optical
    crosstalk between neighboring channels, channel-to-channel gain
    spread, and the event selection all widen the effective turn-on,
    sigma is an effective parameter to be fitted. The POLAR example of
    the paper is erf(mu = 25 keV, sigma = 75 keV), with mu from the
    in-orbit threshold calibration (Li et al. 2018, NIM A 900, 8-24); it
    ships in examples/polar_gate.json and is an instrument-specific
    EXAMPLE, never a default.

``SharpGate(value)``
    An ideal step threshold at ``value`` keV, evaluated on the analytic
    Compton window. Equals ``ErfGate`` in the sigma -> 0 limit.

``GaussianSampledGate(mean, sigma, points, eth_grid=None)``
    A Gaussian mixture of sharp gates, sampled: ``points`` sharp
    thresholds at equal spacing over mean +- 3 sigma of the Gaussian
    effective-threshold distribution, each gated ASAD unit-normalized and
    combined with the Gaussian weights. With ``eth_grid`` the samples are
    nearest-neighbour clamped onto that grid; without it the exact sample
    values are used. The erf gate is the continuum limit of this
    construction and is the default; the sampled form is available as an
    option.

``TwoDepositErfGate(mu, sigma)``
    Independent erf thresholds on the two deposits:
    w = Phi((E_e - mu)/sigma) * Phi((E_prime - mu)/sigma).

``EnergyDependentErfGate(mu, sigma)``
    The erf gate with energy-dependent parameters: ``mu`` and ``sigma``
    may be callables ``f(E_keV) -> keV`` (or constants). Documented hook
    for instruments whose effective gate varies with energy; resolved
    per energy and evaluated on the smooth erf path.

Dictionaries are accepted everywhere a gate is expected, e.g.::

    {"type": "erf", "mu": 25.0, "sigma": 75.0}
    {"type": "sharp", "value": 50.0}
    {"type": "gaussian", "mean": 25.0, "sigma": 75.0, "points": 11}
    {"type": "erf2", "mu": 25.0, "sigma": 75.0}

and a gate described by a small JSON file (family/mu/sigma plus
free-form documentation fields) loads through :func:`load_gate_json`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Union

import numpy as np

from .exceptions import ThresholdError

# Half-width, in units of sigma, of the sampling window of the sampled
# Gaussian mixture of sharp gates (mean +- 3 sigma).
_GAUSSIAN_SAMPLING_HALF_WIDTH = 3.0


@dataclass(frozen=True)
class SharpGate:
    """Ideal step threshold at ``value`` keV."""

    value: float

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ThresholdError(
                f"Sharp threshold must be non-negative, got {self.value} keV."
            )


@dataclass(frozen=True)
class ErfGate:
    """Continuous error-function gate on the shared per-event threshold.

    Parameters
    ----------
    mu : float
        Average detection threshold of the instrument in keV (center of
        the error-function turn-on).
    sigma : float
        Effective width of the turn-on in keV. For a clean instrument this
        is the Gaussian broadening of the pulse near threshold (the
        resolution); for instruments with crosstalk or selection effects
        it is an effective parameter. ``sigma = 0`` reduces exactly to
        ``SharpGate(mu)``.
    """

    mu: float
    sigma: float

    def __post_init__(self) -> None:
        if self.sigma < 0:
            raise ThresholdError(
                f"Erf gate sigma must be non-negative, got {self.sigma} keV."
            )


@dataclass(frozen=True)
class TwoDepositErfGate:
    """Independent erf thresholds on both deposits."""

    mu: float
    sigma: float

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ThresholdError(
                f"Two-deposit erf gate sigma must be positive, got "
                f"{self.sigma} keV."
            )


@dataclass(frozen=True)
class EnergyDependentErfGate:
    """Erf gate with energy-dependent mu(E), sigma(E).

    ``mu`` and ``sigma`` are either constants (keV) or callables mapping
    the photon energy in keV to keV. Resolved per energy and evaluated
    on the smooth erf path.
    """

    mu: Union[float, Callable[[float], float]]
    sigma: Union[float, Callable[[float], float]]

    def resolve(self, energy_kev: float) -> tuple[float, float]:
        mu = self.mu(energy_kev) if callable(self.mu) else float(self.mu)
        sigma = (self.sigma(energy_kev) if callable(self.sigma)
                 else float(self.sigma))
        if sigma < 0:
            raise ThresholdError(
                f"EnergyDependentErfGate resolved sigma({energy_kev}) = "
                f"{sigma} keV; sigma must be non-negative."
            )
        return mu, sigma


@dataclass(frozen=True)
class GaussianSampledGate:
    """A Gaussian mixture of sharp gates, sampled.

    Parameters
    ----------
    mean, sigma : float
        Mean and width (keV) of the Gaussian effective-threshold
        distribution.
    points : int
        Number of sharp-threshold samples, at equal spacing over
        mean +- 3 sigma.
    eth_grid : tuple, optional
        Nearest-neighbour clamp grid for the samples. ``None`` uses the
        exact sample values.
    """

    mean: float
    sigma: float
    points: int = 11
    eth_grid: Optional[tuple] = field(default=None)

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ThresholdError(
                f"Gaussian gate sigma must be positive, got {self.sigma} keV; "
                "for a step threshold use SharpGate instead."
            )
        if self.points < 1:
            raise ThresholdError(
                f"Gaussian gate needs at least one sample point, got "
                f"{self.points}."
            )
        if self.eth_grid is not None:
            object.__setattr__(self, "eth_grid",
                               tuple(float(v) for v in self.eth_grid))

    def samples(self) -> tuple[np.ndarray, np.ndarray]:
        """Sharp-threshold sample values and normalized weights: uniform
        samples over mean +- 3 sigma with Gaussian-density weights; with
        ``eth_grid`` the samples are nearest-neighbour clamped onto the
        grid. Non-positive samples act as a fully open window."""
        from scipy.stats import norm as _norm

        if self.sigma > 1e-6 and self.points > 1:
            t = np.linspace(self.mean - _GAUSSIAN_SAMPLING_HALF_WIDTH * self.sigma,
                            self.mean + _GAUSSIAN_SAMPLING_HALF_WIDTH * self.sigma,
                            self.points)
            w = _norm.pdf(t, loc=self.mean, scale=self.sigma)
            w = w / w.sum()
        else:
            t = np.array([self.mean])
            w = np.array([1.0])
        if self.eth_grid is not None:
            grid = np.asarray(self.eth_grid, dtype=float)
            t = np.array(
                [grid[np.argmin(np.abs(grid - np.clip(s, grid[0], grid[-1])))]
                 for s in t]
            )
        return t, w


ThresholdGate = Union[SharpGate, ErfGate, TwoDepositErfGate,
                      EnergyDependentErfGate, GaussianSampledGate]
ThresholdLike = Union[ThresholdGate, Mapping[str, object]]

_GATE_TYPES = (SharpGate, ErfGate, TwoDepositErfGate, EnergyDependentErfGate,
               GaussianSampledGate)


def parse_threshold(spec: ThresholdLike) -> ThresholdGate:
    """Coerce a gate object or a specification dictionary into a gate object.

    Examples
    --------
    >>> parse_threshold({"type": "erf", "mu": 25.0, "sigma": 75.0})
    ErfGate(mu=25.0, sigma=75.0)
    >>> parse_threshold({"type": "sharp", "value": 50.0})
    SharpGate(value=50.0)
    """
    if isinstance(spec, _GATE_TYPES):
        return spec
    if not isinstance(spec, Mapping):
        raise ThresholdError(
            "Threshold must be a gate object or a dict such as "
            "{'type': 'erf', 'mu': 25.0, 'sigma': 75.0}; got "
            f"{spec!r}."
        )
    kind = str(spec.get("type", "")).lower()
    try:
        if kind == "erf":
            return ErfGate(mu=float(spec["mu"]), sigma=float(spec["sigma"]))
        if kind in ("erf2", "two_deposit_erf"):
            return TwoDepositErfGate(mu=float(spec["mu"]),
                                     sigma=float(spec["sigma"]))
        if kind == "sharp":
            return SharpGate(value=float(spec["value"]))
        if kind == "gaussian":
            return GaussianSampledGate(
                mean=float(spec["mean"]),
                sigma=float(spec["sigma"]),
                points=int(spec.get("points", 11)),
                eth_grid=spec.get("eth_grid"),
            )
    except KeyError as missing:
        raise ThresholdError(
            f"Threshold dict of type '{kind}' is missing the key {missing}."
        ) from None
    raise ThresholdError(
        f"Unknown threshold type '{kind}'. Supported types: 'erf' "
        "(the default model), 'sharp', 'gaussian' (the sampled Gaussian "
        "mixture of sharp gates), and 'erf2' (independent two-deposit "
        "thresholds)."
    )


def load_gate_json(path: str) -> tuple[ThresholdGate, dict]:
    """Load a gate from a small JSON description file.

    Required keys: ``family`` ("M2" for the erf gate, "M1" for the
    sampled Gaussian mixture), ``mu``, ``sigma``; optional ``points`` (M1),
    ``tag``, and free-form documentation fields (``mu_basis``,
    ``sigma_basis``, ``prescription``, ...), which are returned
    untouched in the metadata dict. See examples/polar_gate.json.

    Returns
    -------
    (gate, metadata) : tuple
        The parsed gate object and the full JSON contents as metadata.

    Examples
    --------
    >>> gate, meta = load_gate_json("examples/polar_gate.json")  # doctest: +SKIP
    """
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ThresholdError(f"{path} did not parse to a JSON object.")
    family = str(data.get("family", "")).upper()
    try:
        if family == "M2":
            gate: ThresholdGate = ErfGate(mu=float(data["mu"]),
                                          sigma=float(data["sigma"]))
        elif family == "M1":
            gate = GaussianSampledGate(
                mean=float(data["mu"]),
                sigma=float(data["sigma"]),
                points=int(data.get("points") or 11),
                eth_grid=data.get("eth_grid"),
            )
        else:
            raise ThresholdError(
                f"Gate JSON {path} has family '{data.get('family')}'; "
                "supported families: 'M2' (erf) and 'M1' (sampled "
                "Gaussian mixture)."
            )
    except KeyError as missing:
        raise ThresholdError(
            f"Gate JSON {path} is missing the key {missing}."
        ) from None
    return gate, data
