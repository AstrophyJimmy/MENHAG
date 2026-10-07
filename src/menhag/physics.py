# MENHAG, Mohamed Gamal Zaid, University of New Hampshire. MIT license, see LICENSE.
"""Numerical core: gated, transformed Klein-Nishina azimuthal distributions.

This module computes the physics part of an azimuthal scattering angle
distribution (ASAD): how likely a photon is to Compton scatter toward
each azimuthal direction, for any source position and polarization
state, with the detection-threshold gate folded in. It is the same code
used to produce the results of the accompanying paper.

The idea in plain words: a photon arrives from a direction described by
two angles (delta from the detector normal, lambda around it), with a
polarization angle phi0. The Klein-Nishina cross-section says how it
scatters in its own frame; a rotation carries that into the detector
frame, and integrating over the polar scattering angle at each azimuth
eta gives the two curves everything else is built from:

    F_100(eta)  the fully polarized curve, and
    F_U(eta)    the unpolarized curve.

The instrument's detection threshold enters as a gate weight inside the
integral: an event counts only if both energy deposits (the recoil
electron and the scattered photon) are detectable. The recommended
error-function gate and its alternatives live in :mod:`menhag.gates`.

Two integration paths are used. For a sharp (step) threshold the region
of scattering angles that passes the gate is solved in closed form and
Gauss-Legendre nodes are placed inside it, which is fast and exact even
when the allowed window is razor thin. For a smooth gate the weight is
applied continuously inside the integrand over the full range. Both
paths were validated against each other and against an independent
adaptive-quadrature implementation for the paper.

Conventions used everywhere in this package:

delta   source zenith incidence angle from the detector +z, degrees.
lambda  source azimuthal incidence angle, degrees.
phi0    polarization angle, degrees (local, from the detector +x axis).
eta     measured azimuthal scattering angle, degrees (the ASAD x axis).
E       photon energy in keV; eps = E / 511 keV.

Every F curve is normalized so that mean(F) * 2 pi = 1 on the uniform
full-circle eta grid. Outputs inherit the input eta binning; nothing is
ever re-binned or interpolated.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
from scipy.stats import norm as _norm
from scipy.special import erf

from .exceptions import EmptyComptonWindowError
from .gates import (
    ErfGate,
    EnergyDependentErfGate,
    GaussianSampledGate,
    SharpGate,
    ThresholdLike,
    TwoDepositErfGate,
    parse_threshold,
)

# Electron rest energy in keV.
ELECTRON_REST_KEV = 511.0
MEC2 = ELECTRON_REST_KEV

# Default numbers of Gauss-Legendre nodes: 64 nodes x 4 panels for the
# smooth-gate path and 32 nodes per window interval for the sharp path,
# the configurations used for the paper's POLAR responses.
N_NODES_SMOOTH = 64
N_PANELS_SMOOTH = 4
N_NODES_SHARP = 32

# Default number of azimuthal bins for curves generated without an input
# ASAD to inherit binning from (ideal mode only): 5 degree bins.
N_ETA_DEFAULT = 72

# --------------------------------------------------------------------- #
# Gauss-Legendre node cache
# --------------------------------------------------------------------- #
_GL_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _gl(n):
    if n not in _GL_CACHE:
        _GL_CACHE[n] = np.polynomial.legendre.leggauss(n)
    return _GL_CACHE[n]


# --------------------------------------------------------------------- #
# Rotation matrix between the photon frame and the detector frame
# --------------------------------------------------------------------- #
def trans_matrix(delta_rad, phi0_rad=0.0, lambda_rad=0.0):
    c_lambda, s_lambda = np.cos(lambda_rad), np.sin(lambda_rad)
    c_delta, s_delta = np.cos(delta_rad), np.sin(delta_rad)
    c_phi0, s_phi0 = np.cos(phi0_rad), np.sin(phi0_rad)
    a31 = s_delta * c_lambda
    a32 = s_delta * s_lambda
    a33 = c_delta
    a13 = -c_phi0 * s_delta
    a23 = s_phi0 * s_delta
    a11 = c_phi0 * a33 * c_lambda - s_phi0 * s_lambda
    a12 = c_phi0 * a33 * s_lambda + s_phi0 * c_lambda
    a21 = -s_phi0 * a33 * c_lambda - c_phi0 * s_lambda
    a22 = -s_phi0 * a33 * s_lambda + c_phi0 * c_lambda
    return a11, a12, a13, a21, a22, a23, a31, a32, a33


def rotation_matrix(delta_deg: float, lambda_deg: float, phi0_deg: float) -> np.ndarray:
    """The rotation packaged as a (3, 3) array.

    Rows are (x_gamma, y_gamma, z_gamma) in terms of detector-frame
    components; identical to Rz(phi0) Ry(delta) Rz(lambda).
    """
    a11, a12, a13, a21, a22, a23, a31, a32, a33 = trans_matrix(
        math.radians(delta_deg), math.radians(phi0_deg), math.radians(lambda_deg)
    )
    return np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])


# --------------------------------------------------------------------- #
# Compton coincidence window for a sharp threshold, in cos(theta)
# --------------------------------------------------------------------- #
def compton_window_cos_bounds(E_gamma, E_th):
    if E_th <= 0:
        return -1.0, 1.0
    if E_gamma < 2 * E_th:
        return None, None
    epsilon = E_gamma / MEC2
    cos_upper = 1 - (E_th / (epsilon * (E_gamma - E_th)))
    cos_lower = 1 - (1 / epsilon) * (E_gamma / E_th - 1)
    cos_lower = max(-1.0, min(1.0, cos_lower))
    cos_upper = max(-1.0, min(1.0, cos_upper))
    if cos_lower > cos_upper:
        return None, None
    return cos_lower, cos_upper


def compton_window_bounds(energy_kev: float, threshold_kev: float) -> tuple[float, float]:
    """Sharp-threshold coincidence window in Compton-angle degrees.

    Unit-conversion wrapper around :func:`compton_window_cos_bounds`.
    Returns
    ``(theta_min_deg, theta_max_deg)`` and raises
    :class:`EmptyComptonWindowError` when the window is closed.
    """
    lo, hi = compton_window_cos_bounds(float(energy_kev), float(threshold_kev))
    if lo is None:
        raise EmptyComptonWindowError(
            f"The coincidence window is closed at E = {energy_kev} keV for a "
            f"sharp threshold of {threshold_kev} keV. The window opens only "
            "above a crossover energy set by the threshold; lower the "
            "threshold or raise the photon energy."
        )
    # cos is decreasing in theta: the upper cos bound is the smaller angle.
    return math.degrees(math.acos(hi)), math.degrees(math.acos(lo))


def scattered_photon_energy(energy_kev: float, cos_theta) -> np.ndarray:
    """Scattered photon energy from Compton kinematics, in keV."""
    eps = energy_kev / ELECTRON_REST_KEV
    return energy_kev / (1.0 + eps * (1.0 - np.asarray(cos_theta, dtype=float)))


# --------------------------------------------------------------------- #
# Vectorized integrand pieces
# --------------------------------------------------------------------- #
def _eta_consts(bins_mean_deg, R):
    a11, a12, a13, a21, a22, a23, a31, a32, a33 = R
    phi = np.deg2rad(np.asarray(bins_mean_deg, dtype=float))
    cphi, sphi = np.cos(phi), np.sin(phi)
    C1 = cphi * a31 + sphi * a32          # cosTheta = -(sin(t)*C1 - cos(t)*a33)
    X1 = cphi * a11 + sphi * a12
    Y1 = cphi * a21 + sphi * a22
    return C1, X1, Y1


def _integrands(theta, C1, X1, Y1, a13, a23, a33, epsilon, gate=None):
    """theta: (..., n_nodes) broadcast against per-eta constants (..., 1).
    Returns (pol, unp) integrand values (no Heaviside; support/gate handled
    by the caller)."""
    st, ct = np.sin(theta), np.cos(theta)
    cosTheta = -(st * C1 - ct * a33)
    x_sq = (st * X1 - ct * a13) ** 2
    y_sq = (st * Y1 - ct * a23) ** 2
    frac = 1.0 / (1.0 + epsilon * (1.0 - cosTheta))
    common = frac + frac ** 3
    pol = (common - 2.0 * frac ** 2 * x_sq) * st
    unp = (common - frac ** 2 * (x_sq + y_sq)) * st
    if gate is not None:
        g = gate(cosTheta, frac)
        pol = pol * g
        unp = unp * g
    return pol, unp


# --------------------------------------------------------------------- #
# Closed-form theta support of the sharp coincidence window
# --------------------------------------------------------------------- #
def _window_intervals(C1, a33, cos_l, cos_u):
    """For each eta bin, the theta in [0, pi] intervals where
    cos_l <= rho cos(theta + beta) <= cos_u, with rho = sqrt(a33^2 + C1^2),
    beta = atan2(C1, a33). Returns (lo, hi) arrays of shape (n_eta, 4) with
    lo >= hi marking empty."""
    rho = np.sqrt(a33 ** 2 + C1 ** 2)
    beta = np.arctan2(C1, a33)
    hi_c = np.clip(cos_u / rho, -1.0, 1.0)
    lo_c = np.clip(cos_l / rho, -1.0, 1.0)
    # empty when the band misses [-1, 1] entirely
    dead = (cos_l / rho > 1.0) | (cos_u / rho < -1.0)
    a1 = np.arccos(hi_c)          # start of |x| band
    b1 = np.arccos(lo_c)          # end of |x| band
    # candidate x-intervals: [a1,b1], [-b1,-a1], [a1-2pi, b1-2pi],
    # [2pi-b1, 2pi-a1]
    los = np.stack([a1, -b1, a1 - 2 * np.pi, 2 * np.pi - b1], axis=-1)
    his = np.stack([b1, -a1, b1 - 2 * np.pi, 2 * np.pi - a1], axis=-1)
    # intersect with x in [beta, beta + pi]
    lo_x = np.maximum(los, beta[:, None])
    hi_x = np.minimum(his, (beta + np.pi)[:, None])
    lo_t = lo_x - beta[:, None]
    hi_t = hi_x - beta[:, None]
    lo_t = np.clip(lo_t, 0.0, np.pi)
    hi_t = np.clip(hi_t, 0.0, np.pi)
    bad = dead[:, None] | (hi_t <= lo_t)
    lo_t[bad] = 0.0
    hi_t[bad] = 0.0
    return lo_t, hi_t


def sharp_numerators(bins_mean_deg, energy, eth, phi0_deg, delta_deg,
                     lambda_deg, n_nodes=N_NODES_SHARP):
    """Unnormalized integral d(theta) over the analytic window, per eta bin.
    Returns (num_pol, num_unp) arrays of shape (n_eta,). Zeros if the window
    is closed (no scattering angle passes the gate at this energy)."""
    bounds = compton_window_cos_bounds(energy, eth)
    nb = len(np.atleast_1d(bins_mean_deg))
    if bounds[0] is None:
        return np.zeros(nb), np.zeros(nb)
    cos_l, cos_u = bounds
    R = trans_matrix(np.deg2rad(delta_deg), np.deg2rad(phi0_deg),
                     np.deg2rad(lambda_deg))
    a13, a23, a33 = R[2], R[5], R[8]
    C1, X1, Y1 = _eta_consts(bins_mean_deg, R)
    lo_t, hi_t = _window_intervals(C1, a33, cos_l, cos_u)   # (n_eta, 4)
    gx, gw = _gl(n_nodes)
    mid = 0.5 * (lo_t + hi_t)[..., None]
    half = 0.5 * (hi_t - lo_t)[..., None]
    theta = mid + half * gx                                  # (n_eta, 4, n)
    pol, unp = _integrands(theta, C1[:, None, None], X1[:, None, None],
                           Y1[:, None, None], a13, a23, a33, energy / MEC2)
    w = half * gw                                            # (n_eta, 4, n)
    return (pol * w).sum(axis=(1, 2)), (unp * w).sum(axis=(1, 2))


def smooth_numerators(bins_mean_deg, energy, gate, phi0_deg, delta_deg,
                      lambda_deg, n_nodes=N_NODES_SMOOTH,
                      n_panels=N_PANELS_SMOOTH):
    """Gate applied continuously inside the theta integrand; GL over [0, pi]
    subdivided into n_panels panels. gate(cosTheta, frac) -> weight in [0, 1].
    """
    R = trans_matrix(np.deg2rad(delta_deg), np.deg2rad(phi0_deg),
                     np.deg2rad(lambda_deg))
    a13, a23, a33 = R[2], R[5], R[8]
    C1, X1, Y1 = _eta_consts(bins_mean_deg, R)
    edges = np.linspace(0.0, np.pi, n_panels + 1)
    gx, gw = _gl(n_nodes)
    num_pol = np.zeros(len(C1))
    num_unp = np.zeros(len(C1))
    for i in range(n_panels):
        mid = 0.5 * (edges[i] + edges[i + 1])
        half = 0.5 * (edges[i + 1] - edges[i])
        theta = mid + half * gx                              # (n,)
        pol, unp = _integrands(theta[None, :], C1[:, None], X1[:, None],
                               Y1[:, None], a13, a23, a33, energy / MEC2,
                               gate=gate)
        num_pol += half * (pol * gw).sum(axis=1)
        num_unp += half * (unp * gw).sum(axis=1)
    return num_pol, num_unp


def normalize_response(numerators):
    """The engine periodic norm: F / (mean(F) * 2 pi) on the uniform
    full-circle eta grid, so that mean(F) * 2 pi = 1. Returns zeros when
    the integral vanishes (closed coincidence window)."""
    numerators = np.asarray(numerators, dtype=float)
    norm = numerators.mean() * 2.0 * np.pi     # periodic normalization
    if norm > 1e-12:
        return numerators / norm
    return np.zeros_like(numerators)


# --------------------------------------------------------------------- #
# Gate weight factories for the smooth-gate integration path
# --------------------------------------------------------------------- #
def gate_shared_erf(energy, mu, sigma):
    """The error-function gate: w = Phi((min(T, E') - mu) / sigma), a
    single threshold acting on the smaller of the two deposits. This is
    the exact limit of averaging sharp thresholds over a Gaussian spread
    with each threshold's physical acceptance kept."""
    def gate(cosTheta, frac):
        E_prime = energy * frac
        T = energy - E_prime
        m = np.minimum(T, E_prime)
        return 0.5 * (1.0 + erf((m - mu) / (sigma * np.sqrt(2.0))))
    return gate


def gate_two_deposit_erf(energy, mu, sigma):
    """Independent thresholds on the two deposits,
    w = Phi((T - mu) / sigma) * Phi((E' - mu) / sigma)."""
    def gate(cosTheta, frac):
        E_prime = energy * frac
        T = energy - E_prime
        wT = 0.5 * (1.0 + erf((T - mu) / (sigma * np.sqrt(2.0))))
        wE = 0.5 * (1.0 + erf((E_prime - mu) / (sigma * np.sqrt(2.0))))
        return wT * wE
    return gate


# --------------------------------------------------------------------- #
# Gate-model evaluators: normalized (F_100, F_U) per (energy, angle)
# --------------------------------------------------------------------- #
def FF_sharp(bins_mean, energy, eth, pa, delta, lam, n_nodes=N_NODES_SHARP):
    np_, nu_ = sharp_numerators(bins_mean, energy, eth, pa, delta, lam, n_nodes)
    return normalize_response(np_), normalize_response(nu_)


def FF_discrete_gaussian(bins_mean, energy, pa, delta, lam,
                         mu, sigma, points, eth_grid=None,
                         n_nodes=N_NODES_SHARP, _sharp_cache=None):
    """The sampled Gaussian-threshold gate: sharp-threshold components at
    the sample values, each unit-normalized, combined with Gaussian
    weights. With eth_grid the samples are clamped to that grid (the
    construction with nearest-neighbour clamped samples); without it the
    exact sample values are used."""
    if sigma > 1e-6 and points > 1:
        samples = np.linspace(mu - 3 * sigma, mu + 3 * sigma, points)
        weights = _norm.pdf(samples, loc=mu, scale=sigma)
        weights /= weights.sum()
    else:
        samples = np.array([mu])
        weights = np.array([1.0])
    if eth_grid is not None:
        eth_grid = np.asarray(eth_grid, dtype=float)
        samples = np.array(
            [eth_grid[np.argmin(np.abs(
                eth_grid - np.clip(s, eth_grid[0], eth_grid[-1])))]
             for s in samples]
        )
    # all t <= 0 give the identical full-window shape; collapse them
    eff = {}
    for s, w in zip(samples, weights):
        key = 0.0 if s <= 0 else float(s)
        eff[key] = eff.get(key, 0.0) + w
    F100 = np.zeros(len(bins_mean))
    FU = np.zeros(len(bins_mean))
    for eth, w in eff.items():
        if w < 1e-9:
            continue
        if _sharp_cache is not None and (energy, pa, eth) in _sharp_cache:
            f1, fu = _sharp_cache[(energy, pa, eth)]
        else:
            f1, fu = FF_sharp(bins_mean, energy, eth, pa, delta, lam, n_nodes)
            if _sharp_cache is not None:
                _sharp_cache[(energy, pa, eth)] = (f1, fu)
        F100 += w * f1
        FU += w * fu
    return F100, FU


def FF_smooth(bins_mean, energy, pa, delta, lam, mu, sigma,
              two_deposit=False, n_nodes=N_NODES_SMOOTH,
              n_panels=N_PANELS_SMOOTH):
    """The smooth-gate path: erf gate (or its two-deposit variant), with
    mu and sigma already resolved to numbers for this energy."""
    gf = gate_two_deposit_erf(energy, mu, sigma) if two_deposit \
        else gate_shared_erf(energy, mu, sigma)
    np_, nu_ = smooth_numerators(bins_mean, energy, gf, pa, delta, lam,
                                 n_nodes, n_panels)
    return normalize_response(np_), normalize_response(nu_)


# --------------------------------------------------------------------- #
# FF dispatcher (no new physics: routes each gate to its transcribed path)
# --------------------------------------------------------------------- #
def FF(bins_mean, energy, pa_deg, delta_deg, lambda_deg,
       threshold: Union[ThresholdLike, None],
       n_nodes: Optional[int] = None,
       n_panels: int = N_PANELS_SMOOTH) -> tuple[np.ndarray, np.ndarray]:
    """Normalized (F_100, F_U) for one energy and polarization angle.

    Dispatches the parsed threshold gate to the transcribed evaluator:

    - ``None``                  -> ungated (sharp path with eth = 0)
    - :class:`SharpGate`        -> closed-form sharp-window path
    - :class:`ErfGate`          -> smooth erf path; ``sigma == 0``
      degenerates to the sharp path at ``mu`` exactly
    - :class:`TwoDepositErfGate` -> smooth two-deposit path
    - :class:`EnergyDependentErfGate` -> ``mu(E)``, ``sigma(E)`` resolved
      at this energy, then the smooth path
    - :class:`GaussianSampledGate` -> sampled-threshold mixture (pass
      ``eth_grid`` on the gate for nearest-neighbour clamped samples)

    ``n_nodes`` defaults to the values used for the paper (32 sharp,
    64 smooth).
    """
    gate = None if threshold is None else parse_threshold(threshold)
    if gate is None:
        return FF_sharp(bins_mean, energy, 0.0, pa_deg, delta_deg, lambda_deg,
                        n_nodes or N_NODES_SHARP)
    if isinstance(gate, SharpGate):
        return FF_sharp(bins_mean, energy, gate.value, pa_deg, delta_deg,
                        lambda_deg, n_nodes or N_NODES_SHARP)
    if isinstance(gate, EnergyDependentErfGate):
        mu, sigma = gate.resolve(energy)
        if sigma == 0.0:
            return FF_sharp(bins_mean, energy, max(mu, 0.0), pa_deg,
                            delta_deg, lambda_deg, n_nodes or N_NODES_SHARP)
        return FF_smooth(bins_mean, energy, pa_deg, delta_deg, lambda_deg,
                         mu, sigma, two_deposit=False,
                         n_nodes=n_nodes or N_NODES_SMOOTH, n_panels=n_panels)
    if isinstance(gate, ErfGate):
        if gate.sigma == 0.0:
            return FF_sharp(bins_mean, energy, max(gate.mu, 0.0), pa_deg,
                            delta_deg, lambda_deg, n_nodes or N_NODES_SHARP)
        return FF_smooth(bins_mean, energy, pa_deg, delta_deg, lambda_deg,
                         gate.mu, gate.sigma, two_deposit=False,
                         n_nodes=n_nodes or N_NODES_SMOOTH, n_panels=n_panels)
    if isinstance(gate, TwoDepositErfGate):
        return FF_smooth(bins_mean, energy, pa_deg, delta_deg, lambda_deg,
                         gate.mu, gate.sigma, two_deposit=True,
                         n_nodes=n_nodes or N_NODES_SMOOTH, n_panels=n_panels)
    if isinstance(gate, GaussianSampledGate):
        return FF_discrete_gaussian(bins_mean, energy, pa_deg, delta_deg,
                                    lambda_deg, gate.mean, gate.sigma,
                                    gate.points, eth_grid=gate.eth_grid,
                                    n_nodes=n_nodes or N_NODES_SHARP)
    raise TypeError(f"Unsupported gate type: {type(gate).__name__}")


def compute_raw_curves(
    eta_deg: np.ndarray,
    energy_kev: float,
    delta_deg: float,
    lambda_deg: float,
    phi0_deg: float,
    threshold: Union[ThresholdLike, None],
    n_nodes: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Raw (unnormalized) gated numerators at the given eta points.

    Convenience wrapper over the numerator functions. Returns ``(f_pol, f_unp)``; normalize with
    :func:`normalize_response` or take ratios. Raises
    :class:`EmptyComptonWindowError` when the coincidence window is closed
    (both numerators identically zero).
    """
    eta = np.asarray(eta_deg, dtype=float)
    gate = None if threshold is None else parse_threshold(threshold)
    if gate is None:
        f_pol, f_unp = sharp_numerators(eta, energy_kev, 0.0, phi0_deg,
                                        delta_deg, lambda_deg,
                                        n_nodes or N_NODES_SHARP)
    elif isinstance(gate, SharpGate):
        f_pol, f_unp = sharp_numerators(eta, energy_kev, gate.value, phi0_deg,
                                        delta_deg, lambda_deg,
                                        n_nodes or N_NODES_SHARP)
    elif isinstance(gate, ErfGate) and gate.sigma == 0.0:
        f_pol, f_unp = sharp_numerators(eta, energy_kev, max(gate.mu, 0.0),
                                        phi0_deg, delta_deg, lambda_deg,
                                        n_nodes or N_NODES_SHARP)
    elif isinstance(gate, ErfGate):
        gf = gate_shared_erf(energy_kev, gate.mu, gate.sigma)
        f_pol, f_unp = smooth_numerators(eta, energy_kev, gf, phi0_deg,
                                         delta_deg, lambda_deg,
                                         n_nodes or N_NODES_SMOOTH)
    elif isinstance(gate, TwoDepositErfGate):
        gf = gate_two_deposit_erf(energy_kev, gate.mu, gate.sigma)
        f_pol, f_unp = smooth_numerators(eta, energy_kev, gf, phi0_deg,
                                         delta_deg, lambda_deg,
                                         n_nodes or N_NODES_SMOOTH)
    elif isinstance(gate, EnergyDependentErfGate):
        mu, sigma = gate.resolve(energy_kev)
        gf = gate_shared_erf(energy_kev, mu, sigma)
        f_pol, f_unp = smooth_numerators(eta, energy_kev, gf, phi0_deg,
                                         delta_deg, lambda_deg,
                                         n_nodes or N_NODES_SMOOTH)
    elif isinstance(gate, GaussianSampledGate):
        # Weighted sum of raw sharp numerators over the threshold samples
        # (per-component normalization happens only at the FF level).
        values, sample_weights = gate.samples()
        f_pol = np.zeros(eta.size)
        f_unp = np.zeros(eta.size)
        for v, w in zip(values, sample_weights):
            p, u = sharp_numerators(eta, energy_kev, float(v), phi0_deg,
                                    delta_deg, lambda_deg,
                                    n_nodes or N_NODES_SHARP)
            f_pol += w * p
            f_unp += w * u
    else:
        raise TypeError(f"Unsupported gate type: {type(gate).__name__}")

    if threshold is not None and f_unp.sum() <= 0.0 and f_pol.sum() <= 0.0:
        raise EmptyComptonWindowError(
            f"The threshold gate closes the Compton coincidence window at "
            f"E = {energy_kev} keV. Lower the threshold or restrict the "
            "energy grid."
        )
    return f_pol, f_unp


# --------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------- #
def periodic_integral(values: np.ndarray, eta_deg: np.ndarray) -> float:
    """Integral over eta (degrees) of a curve sampled at bin centers.

    For a uniform grid covering the full circle this equals the periodic
    mean times 360, exact for uniform periodic sampling. For a non-uniform
    or partial grid it falls back to the midpoint rule.
    """
    values = np.asarray(values, dtype=float)
    eta = np.asarray(eta_deg, dtype=float)
    if eta.size < 2:
        raise ValueError("Need at least two azimuthal points to integrate.")
    spacing = np.diff(eta)
    uniform = np.allclose(spacing, spacing[0], rtol=1e-6, atol=1e-9)
    full_circle = uniform and math.isclose(
        spacing[0] * eta.size, 360.0, rel_tol=1e-6
    )
    if full_circle:
        return float(values.mean() * 360.0)
    widths = np.empty_like(eta)
    widths[1:-1] = 0.5 * (eta[2:] - eta[:-2])
    widths[0] = eta[1] - eta[0]
    widths[-1] = eta[-1] - eta[-2]
    return float(np.sum(values * widths))


def mix_polarization(
    phi_pol: np.ndarray,
    phi_unp: np.ndarray,
    pol_fraction: float,
) -> np.ndarray:
    """Engine-normalized density of a partially polarized beam.

    A beam of polarization fraction p is the incoherent mixture
    p * F_100 + (1 - p) * F_U of engine-normalized curves (both satisfy
    mean(F) * 2 pi = 1, so the mixture does too).
    """
    if not 0.0 <= pol_fraction <= 1.0:
        raise ValueError(
            f"pol_fraction must lie in [0, 1]; got {pol_fraction}. For 100 "
            "percent polarization use 1.0."
        )
    p_pol = normalize_response(phi_pol)
    p_unp = normalize_response(phi_unp)
    return pol_fraction * p_pol + (1.0 - pol_fraction) * p_unp


def harmonic_amplitudes(values: np.ndarray, k_max: int = 4) -> dict[int, float]:
    """Relative Fourier amplitudes a_k = 2 |c_k| / |c_0| of a periodic curve.

    The curve must be sampled on a uniform grid covering the full circle.
    An on-axis fully polarized ASAD is dominated by k = 2 (the 180 degree
    component); off-axis incidence introduces k = 1, 3, 4, ...
    """
    values = np.asarray(values, dtype=float)
    coeffs = np.fft.rfft(values) / values.size
    c0 = abs(coeffs[0])
    if c0 == 0.0:
        raise ValueError("Cannot form relative amplitudes of a zero-mean curve.")
    return {k: 2.0 * abs(coeffs[k]) / c0 for k in range(1, k_max + 1)}
