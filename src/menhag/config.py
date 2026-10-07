# MENHAG, Mohamed Gamal Zaid, University of New Hampshire. MIT license, see LICENSE.
"""Typed run configuration for response generation.

:class:`ResponseConfig` carries everything a build needs: the source
geometry, the energy grid, the polarization-angle grid, the threshold gate,
the normalization, the parallelism, the run mode, and, for mode 2, the list
of input polarization angles.

Polarization-angle grid
-----------------------
Generated grids are endpoint-inclusive over [0, 180]: the default grid is
``range(0, 181, 3)``, 61 angles 0, 3, ..., 180, the grid of the POLAR
responses. The 180 degree entry repeats the configuration of 0 degrees
(polarization angle is defined modulo 180 degrees); it is kept because the
threeML and polpy readers expect a closed interval, and
``include_endpoint=False`` drops it from generated grids. All of these are
accepted::

    pol_angles=range(0, 181, 3)                       # the default grid
    pol_angles=[0, 30, 60, 90, 120, 150]              # explicit angles
    pol_angles={"start": 0, "stop": 180, "step": 5}   # generated, inclusive
    pol_step=1.0                                      # regular-grid shorthand

Lists and tuples are always taken literally as angle sets; use ``range``,
the start/stop/step dict, or ``pol_step`` for generated grids, so that an
explicit set such as ``(0, 45)`` can never be misread as a step
specification. ``pol_step`` sets the step of the regular [0, 180] grid
(3 degrees is the POLAR convention, giving 61 angles; 1 degree gives 181;
sub-degree steps are accepted).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np

from .exceptions import ConfigError
from .gates import ThresholdGate, ThresholdLike, parse_threshold

_VALID_MODES = ("full", "spectrum", "polarized")
_DEFAULT_POL_RANGE = range(0, 181, 3)
DEFAULT_BASIS_ANGLES = (0.0, 45.0)


def _generated_grid(start: float, stop: float, step: float,
                    include_endpoint: bool) -> tuple[float, ...]:
    """Regular grid from start to stop; endpoint-inclusive when it lands
    on the lattice (the convention of the POLAR responses)."""
    if step <= 0:
        raise ConfigError(f"Grid step must be positive, got {step}.")
    values = np.arange(start, stop, step)
    if include_endpoint:
        n_steps = round((stop - start) / step)
        if np.isclose(start + n_steps * step, stop, rtol=1e-9, atol=1e-9):
            values = np.concatenate([values, [stop]])
    return tuple(float(v) for v in values)


def parse_pol_angles(spec: Union[Sequence, range, dict],
                     include_endpoint: bool = True) -> tuple[float, ...]:
    """Coerce any supported polarization-grid specification to a tuple.

    Lists and tuples are always literal angle sets; generated grids come
    from ``range`` (exact Python semantics) or a start/stop/step dict
    (endpoint-inclusive by default; see the module docstring).

    Examples
    --------
    >>> parse_pol_angles([0, 45, 90])
    (0.0, 45.0, 90.0)
    >>> parse_pol_angles({"start": 0, "stop": 90, "step": 30})
    (0.0, 30.0, 60.0, 90.0)
    """
    if isinstance(spec, range):
        return tuple(float(a) for a in spec)
    if isinstance(spec, dict):
        try:
            start, stop = float(spec["start"]), float(spec["stop"])
            step = float(spec["step"])
        except KeyError as missing:
            raise ConfigError(
                f"pol_angles dict is missing the key {missing}; expected "
                "{'start': ..., 'stop': ..., 'step': ...}."
            ) from None
        return _generated_grid(start, stop, step, include_endpoint)
    try:
        angles = tuple(float(a) for a in spec)
    except TypeError:
        raise ConfigError(
            "pol_angles must be a range, a list of angles, or a "
            f"start/stop/step dict; got {spec!r}."
        ) from None
    if not angles:
        raise ConfigError("pol_angles is empty; supply at least one angle.")
    return angles


@dataclass
class ResponseConfig:
    """Everything needed to generate a polarization response.

    Attributes
    ----------
    energies : sequence of float
        Monoenergetic energy grid in keV (strictly increasing). Free; the
        worked POLAR example uses range(5, 751, 5), the grid of the POLAR
        responses.
    delta, lambda_ : float
        Source zenith and azimuth incidence angles in degrees. The
        trailing underscore avoids the Python keyword.
    threshold : gate or dict
        Threshold gate model (see :mod:`menhag.gates`). Always a
        user-controlled, instrument-specific input; the erf gate with the
        instrument's average threshold mu and effective turn-on width
        sigma is the default model.
    pol_angles : see :func:`parse_pol_angles`
        Polarization-angle grid in degrees. Defaults to
        ``range(0, 181, 3)`` (61 angles, endpoint-inclusive, the grid of
        the POLAR responses).
    pol_step : float, optional
        Convenience alternative to ``pol_angles``: the step of the
        regular endpoint-inclusive [0, 180] grid. POLAR convention is 3;
        1 degree and sub-degree steps are supported. Shorthand for
        ``pol_angles={"start": 0, "stop": 180, "step": pol_step}``;
        giving both raises.
    include_endpoint : bool
        Whether generated grids include the 180 degree endpoint (default
        True; the entry repeats the configuration of 0 degrees and is
        kept for the closed interval the POLAR readers expect).
    pol_fraction : float
        Polarization fraction of the generated ASADs, in [0, 1].
    norm_factor : float
        Normalization factor f of the input unpolarized histograms, used
        for Poisson errors sigma = sqrt(n * f) / f when the input carries
        no errors. Enters the errors only.
    preserve_counts : bool
        The kernel always renormalizes each output to the total counts
        of its unpolarized input; only True is accepted (the field exists
        for configuration-file compatibility).
    n_nodes : int, optional
        Gauss-Legendre node-count override. ``None`` uses the defaults
        of the paper (32 nodes per window interval on the sharp path,
        64 nodes x 4 panels on the smooth path).
    n_jobs : int
        Parallelism over energies; -1 uses all cores when the optional
        joblib extra is installed, otherwise energies run serially.
    mode : str
        'full' (mode 1, the unpolarized-only construction over the energy
        x pol_angle grid), 'spectrum' (mode 1, spectrum-integrated), or
        'polarized' (mode 2, the construction from the unpolarized ASAD
        and the polarized ASADs at ``basis_angles``; see
        :mod:`menhag.basis`).
    basis_angles : sequence of float
        Mode 2 input polarization angles in degrees. Any number from two
        upward; the default is (0, 45). With two angles the two sinusoid
        coefficients are solved directly, with more they are the
        least-squares solution per bin and the fit residual is reported.
    spectrum_weights : sequence of float or callable, optional
        Per-energy spectral weights for 'spectrum' mode; either an array
        aligned with ``energies`` or a callable ``weight(E_keV)``.
    """

    energies: Sequence[float]
    delta: float
    lambda_: float
    threshold: ThresholdLike
    pol_angles: Union[Sequence, range, dict] = field(
        default_factory=lambda: _DEFAULT_POL_RANGE
    )
    pol_step: Optional[float] = None
    include_endpoint: bool = True
    pol_fraction: float = 1.0
    norm_factor: float = 1.0
    preserve_counts: bool = True
    n_nodes: Optional[int] = None
    n_jobs: int = -1
    mode: str = "full"
    basis_angles: Sequence[float] = DEFAULT_BASIS_ANGLES
    spectrum_weights: Optional[object] = None

    def __post_init__(self) -> None:
        self.energies = tuple(float(e) for e in self.energies)
        if not self.energies:
            raise ConfigError("energies is empty; supply at least one energy.")
        if any(e <= 0 for e in self.energies):
            raise ConfigError("All energies must be positive (keV).")
        if any(b <= a for a, b in zip(self.energies, self.energies[1:])):
            raise ConfigError("energies must be strictly increasing.")
        if not -90.0 <= float(self.delta) <= 90.0:
            raise ConfigError(
                f"delta = {self.delta} deg is outside [-90, 90]; delta is the "
                "zenith incidence angle from the detector normal."
            )
        if self.pol_step is not None:
            step = float(self.pol_step)
            if not 0.0 < step <= 180.0:
                raise ConfigError(
                    f"pol_step = {self.pol_step} deg must lie in (0, 180]."
                )
            is_default_grid = (
                isinstance(self.pol_angles, range)
                and self.pol_angles == _DEFAULT_POL_RANGE
            )
            if not is_default_grid:
                raise ConfigError(
                    "Give either pol_step or pol_angles, not both. pol_step "
                    "is shorthand for pol_angles = {'start': 0, 'stop': 180, "
                    "'step': pol_step}."
                )
            self.pol_angles = {"start": 0.0, "stop": 180.0, "step": step}
        self.pol_angles = parse_pol_angles(self.pol_angles,
                                           bool(self.include_endpoint))
        if not 0.0 <= float(self.pol_fraction) <= 1.0:
            raise ConfigError(
                f"pol_fraction = {self.pol_fraction} must lie in [0, 1]; for "
                "100 percent polarization use 1.0."
            )
        if self.norm_factor <= 0:
            raise ConfigError(f"norm_factor must be positive, got {self.norm_factor}.")
        if not self.preserve_counts:
            raise ConfigError(
                "preserve_counts=False is not available: the kernel "
                "always renormalizes each output to the counts of its "
                "unpolarized input."
            )
        if self.n_nodes is not None and self.n_nodes < 8:
            raise ConfigError("n_nodes below 8 is too coarse to be meaningful.")
        if self.mode not in _VALID_MODES:
            raise ConfigError(
                f"mode must be one of {_VALID_MODES}, got '{self.mode}'."
            )
        from .basis import check_basis_angles

        try:
            self.basis_angles = check_basis_angles(self.basis_angles)
        except Exception as exc:
            raise ConfigError(str(exc)) from None
        # Parse eagerly so bad threshold dicts fail at configuration time.
        self.gate: ThresholdGate = parse_threshold(self.threshold)

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, data: dict) -> "ResponseConfig":
        """Build a config from a plain dictionary (e.g. parsed YAML).

        Unknown keys raise, so typos in configuration files fail loudly.
        The key ``lambda`` is accepted as an alias of ``lambda_``.
        """
        data = dict(data)
        if "lambda" in data:
            data["lambda_"] = data.pop("lambda")
        known = set(cls.__dataclass_fields__)
        unknown = set(data) - known
        if unknown:
            raise ConfigError(
                f"Unknown configuration keys: {sorted(unknown)}. "
                f"Known keys: {sorted(known)}."
            )
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str) -> "ResponseConfig":
        """Load a configuration from a YAML file (see examples/).

        Needs the optional dependency pyyaml (``pip install menhag[yaml]``).
        """
        try:
            import yaml
        except ImportError as exc:
            raise ConfigError(
                "Loading YAML configurations needs the optional dependency "
                "'pyyaml'. Install it with: pip install menhag[yaml]"
            ) from exc

        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise ConfigError(f"{path} did not parse to a mapping.")
        return cls.from_dict(data)
