# MENHAG, Mohamed Gamal Zaid, University of New Hampshire. MIT license, see LICENSE.
"""Response generation engine (the rapid analytic method of the paper).

The heart of every run mode is one small function,
:func:`generate_expected_polarized_response`: it takes a measured or
simulated unpolarized ASAD together with the physics curves (F_100, F_U)
and returns the predicted polarized ASAD with propagated errors. This is
the exact function used to generate the paper's POLAR responses, kept
with all of its conventions:

- bins where the physics curves fall below a small floor keep the
  unpolarized input unchanged (a guard against dividing by numerical
  zeros near closed coincidence windows);
- the prediction is scaled by the analytic normalization between the two
  physics curves and then renormalized so its total counts equal the
  input's total counts;
- the error array is carried through with the identical multipliers, so
  fractional errors are preserved bin by bin;
- if the coincidence window is fully closed at an energy, the input is
  returned unchanged.

The measured ASAD factorizes as M(eta) = I(eta) * Phi(eta), where I
collects all instrumental and geometric effects and Phi is the physics
term. Under the assumption that instrumental effects are polarization
independent (valid for well-shielded, low-passive-mass grid
instruments), the polarized ASAD is obtained from a supplied unpolarized
ASAD by the kernel above. Geometry is inherited, not an input: no pixel
pitch or grid size is ever accepted; outputs always inherit the input
eta binning.

The (F_100, F_U) curves come from :mod:`menhag.physics` (the
single-angle float64 path) or, for full erf-gate response grids, from
:mod:`menhag.m2` (all angles in one pass with a float32 geometry cache).
The two agree at the documented cache precision, about 1e-5 relative.

Partial polarization enters through the physics curve only: the beam
curve p * F_100 + (1 - p) * F_U (both normalized) replaces F_100 in the
kernel, so p = 0 returns the input identically.

Entry points of mode 1: :func:`build_single`, :func:`build_ideal`,
:func:`build_response`, :func:`build_spectrum_response`. Mode 2 lives in
:mod:`menhag.basis`.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Sequence, Union

import numpy as np

from .asad import ASAD, ResponseSet, poisson_errors
from .config import ResponseConfig
from .exceptions import ConfigError, MissingEnergyError
from .gates import ErfGate, ThresholdLike
from .m2 import M2GRB
from .physics import (
    FF,
    N_ETA_DEFAULT,
    harmonic_amplitudes,
    mix_polarization,
    normalize_response,
)

logger = logging.getLogger("menhag")

_SCOPE_NOTE = (
    "menhag assumes polarization-independent instrumental effects "
    "(I_pol approximately equal to I_unp); this holds for well-shielded, "
    "low-passive-mass grid instruments such as POLAR, POLAR-2, and LEAP."
)
_scope_note_emitted = False


def _emit_scope_note() -> None:
    """Log the validity note once per session (kept short on purpose)."""
    global _scope_note_emitted
    if not _scope_note_emitted:
        logger.info(_SCOPE_NOTE)
        _scope_note_emitted = True


# --------------------------------------------------------------------- #
# EXPECTED RESPONSE GENERATOR (the kernel of the paper's method)
# --------------------------------------------------------------------- #
def generate_expected_polarized_response(sim_unpolarized, sim_unpolarized_errors, F_100, F_U):
    if np.all(F_100 < 1e-9) or np.all(F_U < 1e-9):
        return sim_unpolarized, sim_unpolarized_errors

    U, U_err = sim_unpolarized, sim_unpolarized_errors
    F_U_safe = np.where(F_U < 1e-9, 1e-9, F_U)
    ModulationKernel = F_100 / F_U_safe
    theoretical_norm_factor = np.sum(F_100) / np.sum(F_U) if np.sum(F_U) > 1e-12 else 1.0

    P_raw     = U     * ModulationKernel * theoretical_norm_factor
    P_raw_err = U_err * ModulationKernel * theoretical_norm_factor

    zero_mask = (F_100 < 1e-9) | (F_U < 1e-9)
    P_raw[zero_mask]     = U[zero_mask]
    P_raw_err[zero_mask] = U_err[zero_mask]

    sum_U, sum_P = np.sum(U), np.sum(P_raw)
    if sum_P < 1e-12:
        return U, U_err
    final_norm = sum_U / sum_P
    return P_raw * final_norm, P_raw_err * final_norm


# --------------------------------------------------------------------- #
# Core kernel application
# --------------------------------------------------------------------- #
def _beam_curve(F_100: np.ndarray, F_U: np.ndarray, pol_fraction: float) -> np.ndarray:
    """The beam physics curve entering the kernel: F_100 for a fully
    polarized beam, the normalized mixture otherwise."""
    if pol_fraction == 1.0:
        return F_100
    return mix_polarization(F_100, F_U, pol_fraction)


def predict_polarized_asad(
    unpolarized: ASAD,
    phi0: float,
    *,
    delta: float,
    lambda_: float,
    threshold: Optional[ThresholdLike],
    energy: Optional[float] = None,
    pol_fraction: float = 1.0,
    norm_factor: float = 1.0,
    n_nodes: Optional[int] = None,
    ff_cache: Optional[tuple] = None,
) -> ASAD:
    """Predict the polarized ASAD equivalent to an unpolarized input.

    Parameters
    ----------
    unpolarized : ASAD
        The measured or simulated unpolarized ASAD. If it carries errors
        they are propagated; otherwise Poisson errors with ``norm_factor``
        are attached first (sigma = sqrt(n * f) / f).
    phi0 : float
        Polarization angle in degrees.
    delta, lambda_ : float
        Source zenith and azimuth incidence angles in degrees.
    threshold : gate, dict, or None
        Threshold gate model; see :mod:`menhag.gates`.
    energy : float, optional
        Photon energy in keV; taken from ``unpolarized.energy`` if omitted.
    pol_fraction : float
        Polarization fraction of the predicted beam, in [0, 1].
    n_nodes : int, optional
        Gauss-Legendre order override (defaults to the values used for
        the paper).
    ff_cache : tuple, optional
        Previously computed (F_100, F_U) for this energy and phi0,
        e.g. from the one-pass grid evaluator; bypasses the single-angle
        physics computation.

    Returns
    -------
    ASAD
        The predicted polarized ASAD on the input binning, with propagated
        per-bin errors and provenance metadata. If the threshold gate
        closes the coincidence window at this energy the input is returned
        unchanged and a warning is logged.
    """
    _emit_scope_note()
    if energy is None:
        energy = unpolarized.energy
    if energy is None:
        raise ConfigError(
            "No energy given: pass energy=... or set unpolarized.energy."
        )
    centers = unpolarized.bin_centers

    if ff_cache is not None:
        F_100, F_U = ff_cache
    else:
        F_100, F_U = FF(centers, float(energy), phi0, delta, lambda_,
                        threshold, n_nodes=n_nodes)

    if np.all(F_100 < 1e-9) or np.all(F_U < 1e-9):
        logger.warning(
            "The coincidence window is closed at E = %s keV for the given "
            "threshold; returning the unpolarized input unchanged.", energy,
        )

    if unpolarized.errors is not None:
        sigma_unp = unpolarized.errors
    else:
        sigma_unp = poisson_errors(unpolarized.counts, norm_factor)

    F_beam = _beam_curve(F_100, F_U, float(pol_fraction))
    predicted, sigma_pred = generate_expected_polarized_response(
        unpolarized.counts.astype(float).copy(),
        np.asarray(sigma_unp, dtype=float).copy(),
        np.asarray(F_beam, dtype=float),
        np.asarray(F_U, dtype=float),
    )

    meta = dict(unpolarized.meta)
    meta.update(
        {
            "phi0_deg": float(phi0),
            "pol_fraction": float(pol_fraction),
            "delta_deg": float(delta),
            "lambda_deg": float(lambda_),
            "threshold": threshold,
            "method": "analytic_kernel",
        }
    )
    return ASAD(
        counts=predicted,
        bin_edges=unpolarized.bin_edges.copy(),
        errors=sigma_pred,
        energy=float(energy),
        meta=meta,
    )


# --------------------------------------------------------------------- #
# Single ASAD in, single polarized ASAD out
# --------------------------------------------------------------------- #
def build_single(
    counts: Sequence[float],
    bin_edges: Sequence[float],
    energy: float,
    delta: float,
    lambda_: float,
    phi0: float,
    threshold: Optional[ThresholdLike],
    *,
    errors: Optional[Sequence[float]] = None,
    pol_fraction: float = 1.0,
    norm_factor: float = 1.0,
    n_nodes: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """One-line entry point: one unpolarized ASAD to one polarized ASAD.

    Parameters
    ----------
    counts, bin_edges : array_like
        The unpolarized ASAD: bin contents and bin edges in degrees.
    energy : float
        Photon energy in keV.
    delta, lambda_, phi0 : float
        Source zenith, source azimuth, and polarization angle in degrees.
    threshold : gate, dict, or None
        Threshold gate model, e.g. ``{"type": "erf", "mu": 25, "sigma": 75}``
        (the POLAR example values; instrument-specific).
    errors : array_like, optional
        Per-bin errors of the input; Poisson errors with ``norm_factor``
        are used when omitted.

    Returns
    -------
    (pol_counts, pol_errors) : tuple of ndarray
        Predicted polarized bin contents and propagated per-bin errors on
        the input binning.

    Examples
    --------
    >>> import numpy as np
    >>> edges = np.linspace(0.0, 360.0, 61)
    >>> unpol = np.full(60, 1000.0)
    >>> pol, err = build_single(unpol, edges, 200.0, 10.0, 0.0, 45.0,
    ...                         {"type": "erf", "mu": 25.0, "sigma": 75.0})
    >>> np.isclose(pol.sum(), unpol.sum())
    True
    """
    unpol = ASAD(
        counts=np.asarray(counts, dtype=float),
        bin_edges=np.asarray(bin_edges, dtype=float),
        errors=None if errors is None else np.asarray(errors, dtype=float),
        energy=float(energy),
    )
    result = predict_polarized_asad(
        unpol,
        phi0,
        delta=delta,
        lambda_=lambda_,
        threshold=threshold,
        pol_fraction=pol_fraction,
        norm_factor=norm_factor,
        n_nodes=n_nodes,
    )
    return result.counts, result.errors


# --------------------------------------------------------------------- #
# Ideal polarimeter (no ASAD input)
# --------------------------------------------------------------------- #
def build_ideal(
    energy: float,
    delta: float,
    lambda_: float,
    phi0: float,
    threshold: Optional[ThresholdLike] = None,
    *,
    n_bins: int = N_ETA_DEFAULT,
    pol_fraction: float = 1.0,
    n_nodes: Optional[int] = None,
) -> dict:
    """Theoretical curves for an ideal detector (I(eta) = 1); no input.

    Returns the engine-normalized (F_100, F_U) curves themselves (the
    normalized physics densities, mean(F) * 2 pi = 1), the corrected
    modulation curve F_beam / F_U, its theoretical modulation
    mu = (max - min) / (max + min), and the relative harmonic amplitudes.

    Parameters
    ----------
    energy : float
        Photon energy in keV.
    delta, lambda_, phi0 : float
        Source zenith, source azimuth, and polarization angle in degrees.
    threshold : gate, dict, or None
        Threshold gate model; ``None`` gives the ungated Klein-Nishina
        curves.
    n_bins : int
        Number of azimuthal bins over [0, 360) degrees (default 72). Used
        only because there is no input ASAD whose binning could be
        inherited.
    pol_fraction : float
        Polarization fraction of the polarized curve, in [0, 1].

    Examples
    --------
    >>> ideal = build_ideal(200.0, 0.0, 0.0, 0.0)
    >>> 0.0 < ideal["mu"] < 1.0
    True
    """
    _emit_scope_note()
    edges = np.linspace(0.0, 360.0, int(n_bins) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    F_100, F_U = FF(centers, float(energy), phi0, delta, lambda_, threshold,
                    n_nodes=n_nodes)
    phi_beam = _beam_curve(F_100, F_U, float(pol_fraction))
    with np.errstate(divide="ignore", invalid="ignore"):
        corrected = np.divide(phi_beam, F_U,
                              out=np.zeros_like(phi_beam), where=F_U > 0)
    mu = float(
        (corrected.max() - corrected.min()) / (corrected.max() + corrected.min())
    )
    return {
        "eta_deg": centers,
        "bin_edges_deg": edges,
        "phi_pol": phi_beam,
        "phi_unp": F_U,
        "corrected": corrected,
        "mu": mu,
        "harmonics": harmonic_amplitudes(corrected),
        "energy": float(energy),
        "delta_deg": float(delta),
        "lambda_deg": float(lambda_),
        "phi0_deg": float(phi0),
        "pol_fraction": float(pol_fraction),
        "threshold": threshold,
    }


# --------------------------------------------------------------------- #
# Full response grid
# --------------------------------------------------------------------- #
def _load_unpolarized(adapter, energy: float, norm_factor: float) -> ASAD:
    """Fetch one unpolarized ASAD and make sure it carries errors."""
    try:
        unpol = adapter.load_unpolarized(energy)
    except MissingEnergyError:
        raise
    except FileNotFoundError as exc:
        raise MissingEnergyError(
            f"Adapter could not provide the unpolarized ASAD at {energy} keV: "
            f"{exc}. menhag never fabricates an unpolarized ASAD; supply one "
            "for every requested energy."
        ) from exc
    if unpol.errors is None:
        unpol = unpol.with_poisson_errors(norm_factor)
    if unpol.energy is None:
        unpol.energy = float(energy)
    return unpol


def _process_energy(
    config: ResponseConfig, unpol: ASAD, energy: float
) -> list[tuple[float, float, ASAD]]:
    """Worker: all polarization angles for one energy (single-angle FF path).

    F_U depends on (energy, delta, lambda) but not on phi0, so it is
    computed once per energy and reused across the whole polarization
    grid.
    """
    centers = unpol.bin_centers
    first_phi0 = config.pol_angles[0]
    _, F_U = FF(centers, float(energy), first_phi0, config.delta,
                config.lambda_, config.gate, n_nodes=config.n_nodes)
    out = []
    for phi0 in config.pol_angles:
        F_100, _ = FF(centers, float(energy), phi0, config.delta,
                      config.lambda_, config.gate, n_nodes=config.n_nodes)
        pred = predict_polarized_asad(
            unpol,
            phi0,
            delta=config.delta,
            lambda_=config.lambda_,
            threshold=config.gate,
            energy=energy,
            pol_fraction=config.pol_fraction,
            norm_factor=config.norm_factor,
            ff_cache=(F_100, F_U),
        )
        out.append((energy, phi0, pred))
    return out


def _process_energy_m2(
    config: ResponseConfig, unpol: ASAD, energy: float, m2: M2GRB
) -> list[tuple[float, float, ASAD]]:
    """Worker: all polarization angles for one energy through the
    one-pass evaluator (a single array pass per energy, with the geometry
    cache shared across all energies)."""
    gate = config.gate
    F1all, FUall = m2.FF(float(energy), gate.mu, gate.sigma)
    out = []
    for k, phi0 in enumerate(config.pol_angles):
        pred = predict_polarized_asad(
            unpol,
            phi0,
            delta=config.delta,
            lambda_=config.lambda_,
            threshold=gate,
            energy=energy,
            pol_fraction=config.pol_fraction,
            norm_factor=config.norm_factor,
            ff_cache=(F1all[k].astype(float), FUall[k].astype(float)),
        )
        out.append((energy, phi0, pred))
    return out


def _map_energies(config: ResponseConfig, work: Callable, energies) -> list:
    """Run the per-energy workers, in parallel when joblib is available
    and requested; serially otherwise (joblib is an optional extra)."""
    if config.n_jobs not in (None, 1):
        try:
            from joblib import Parallel, delayed
        except ImportError:
            logger.info(
                "joblib is not installed; running energies serially. "
                "Install with: pip install menhag[parallel]"
            )
        else:
            return Parallel(n_jobs=config.n_jobs, verbose=0)(
                delayed(work)(e) for e in energies
            )
    return [work(e) for e in energies]


def build_response(
    config: ResponseConfig,
    input_adapter,
    output_adapter=None,
) -> ResponseSet:
    """Generate the full polarized response grid (energy x pol angle).

    For each energy in ``config.energies`` the unpolarized ASAD is loaded
    through ``input_adapter`` and the kernel is applied for every
    polarization angle. For the erf gate (the recommended model) the
    (F_100, F_U) curves come from the one-pass evaluator of
    :mod:`menhag.m2` with its geometry cache built once and reused
    across energies; other gates use the single-angle physics path. If
    an ``output_adapter`` is supplied the set is also saved through it.

    Returns
    -------
    ResponseSet
        The unpolarized inputs plus the generated polarized grid.
    """
    _emit_scope_note()
    unpol_by_energy = {
        e: _load_unpolarized(input_adapter, e, config.norm_factor)
        for e in config.energies
    }
    logger.info(
        "Building response: %d energies x %d polarization angles "
        "(pol_fraction = %.2f).",
        len(config.energies),
        len(config.pol_angles),
        config.pol_fraction,
    )

    gate = config.gate
    edges0 = unpol_by_energy[config.energies[0]].bin_edges
    same_binning = all(
        np.array_equal(unpol_by_energy[e].bin_edges, edges0)
        for e in config.energies
    )
    use_m2 = (
        isinstance(gate, ErfGate)
        and gate.sigma > 0
        and same_binning
    )
    if use_m2:
        centers = unpol_by_energy[config.energies[0]].bin_centers
        m2 = M2GRB(np.asarray(config.pol_angles, dtype=float), centers,
                   config.delta, config.lambda_)

        def work(e):
            return _process_energy_m2(config, unpol_by_energy[e], e, m2)
    else:
        def work(e):
            return _process_energy(config, unpol_by_energy[e], e)

    chunks = _map_energies(config, work, config.energies)
    response = ResponseSet(
        unpolarized=unpol_by_energy,
        polarized={},
        pol_fraction=config.pol_fraction,
        meta={
            "delta_deg": config.delta,
            "lambda_deg": config.lambda_,
            "threshold": config.gate,
            "mode": "full",
            "evaluator": "m2_pass" if use_m2 else "single_angle",
        },
    )
    for chunk in chunks:
        for energy, phi0, pred in chunk:
            response.polarized[(energy, phi0)] = pred
    if output_adapter is not None:
        output_adapter.save_response(response)
    return response


# --------------------------------------------------------------------- #
# Spectrum-integrated response
# --------------------------------------------------------------------- #
def _resolve_weights(
    weights: Union[Sequence[float], Callable[[float], float]],
    energies: Sequence[float],
) -> np.ndarray:
    """Normalize weight input to an array aligned with the energy grid."""
    if callable(weights):
        w = np.array([float(weights(e)) for e in energies])
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (len(energies),):
            raise ConfigError(
                f"spectrum_weights has shape {w.shape}; expected one weight "
                f"per energy, i.e. shape ({len(energies)},)."
            )
    if np.any(w < 0):
        raise ConfigError("Spectral weights must be non-negative.")
    if w.sum() <= 0:
        raise ConfigError("Spectral weights sum to zero.")
    return w


def build_spectrum_response(
    config: ResponseConfig,
    input_adapter,
    output_adapter=None,
    weights: Optional[Union[Sequence[float], Callable[[float], float]]] = None,
) -> ResponseSet:
    """Generate a spectrum-integrated polarized ASAD per polarization angle.

    The integration order is: weight each energy's unpolarized counts by
    its spectral weight, apply that energy's kernel, then sum over
    energies. Because the kernel is multiplicative per energy, weighting
    before or after the kernel is mathematically identical; the order is
    fixed here for clarity and reproducibility. The spectrum-integrated
    unpolarized ASAD (the weighted sum of the inputs) is returned
    alongside. Per-bin errors of each weighted term scale with the weight
    and are combined in quadrature across energies.

    All ASADs must share identical binning; a mismatch raises. menhag
    never fabricates an unpolarized ASAD from a spectrum alone: every
    energy in ``config.energies`` must be loadable through the adapter.
    """
    _emit_scope_note()
    if weights is None:
        weights = config.spectrum_weights
    if weights is None:
        raise ConfigError(
            "Spectrum mode needs per-energy weights: pass weights=... or set "
            "config.spectrum_weights."
        )
    w = _resolve_weights(weights, config.energies)

    unpol_by_energy = {
        e: _load_unpolarized(input_adapter, e, config.norm_factor)
        for e in config.energies
    }
    edges = unpol_by_energy[config.energies[0]].bin_edges
    for e, asad in unpol_by_energy.items():
        if asad.bin_edges.shape != edges.shape or not np.allclose(
            asad.bin_edges, edges
        ):
            raise ConfigError(
                f"The unpolarized ASAD at {e} keV has different binning; all "
                "energies must share identical bin edges for spectrum mode."
            )

    n_bins = edges.size - 1
    int_unp = np.zeros(n_bins)
    int_unp_var = np.zeros(n_bins)
    int_pol = {phi0: np.zeros(n_bins) for phi0 in config.pol_angles}
    int_pol_var = {phi0: np.zeros(n_bins) for phi0 in config.pol_angles}

    for energy, weight in zip(config.energies, w):
        if weight == 0.0:
            continue
        unpol = unpol_by_energy[energy]
        weighted = ASAD(
            counts=unpol.counts * weight,
            bin_edges=unpol.bin_edges,
            errors=unpol.errors * weight,
            energy=energy,
            meta=dict(unpol.meta),
        )
        int_unp += weighted.counts
        int_unp_var += weighted.errors**2
        for _, phi0, pred in _process_energy(config, weighted, energy):
            int_pol[phi0] += pred.counts
            int_pol_var[phi0] += pred.errors**2

    meta = {
        "delta_deg": config.delta,
        "lambda_deg": config.lambda_,
        "threshold": config.gate,
        "mode": "spectrum",
        "energies": config.energies,
        "weights": w.tolist(),
    }
    response = ResponseSet(
        unpolarized={
            None: ASAD(
                counts=int_unp,
                bin_edges=edges.copy(),
                errors=np.sqrt(int_unp_var),
                meta=dict(meta),
            )
        },
        polarized={
            (None, phi0): ASAD(
                counts=int_pol[phi0],
                bin_edges=edges.copy(),
                errors=np.sqrt(int_pol_var[phi0]),
                meta={**meta, "phi0_deg": phi0},
            )
            for phi0 in config.pol_angles
        },
        pol_fraction=config.pol_fraction,
        meta=meta,
    )
    if output_adapter is not None:
        output_adapter.save_response(response)
    return response
