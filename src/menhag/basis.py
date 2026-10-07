# MENHAG, Mohamed Gamal Zaid, University of New Hampshire. MIT license, see LICENSE.
"""Mode 2: the response from the unpolarized ASAD and polarized ASADs at a few angles.

Per azimuthal bin, the fully polarized ASAD as a function of the polarization
angle phi0 is described by a constant plus one sinusoid in twice the angle:

    P(eta; phi0) = U(eta) + cos(2 phi0) M_c(eta) + sin(2 phi0) M_s(eta).

Three numbers per bin. The constant U is the unpolarized ASAD; the two
coefficients M_c and M_s follow from polarized ASADs at a chosen list of
angles. With two angles (the default, 0 and 45 degrees) the two coefficients
are solved directly, since those angles isolate the cosine and sine terms.
With more than two angles the coefficients are the least-squares solution per
bin, and the fit residual is reported. Any number of angles from two upward is
accepted; the list is a configuration option.

The reconstruction ingests only the ASADs at the chosen angles. Every other
polarized ASAD of a simulation set stays on the validation side
(:func:`holdout_validate`), and the blind test (:func:`blind_test`) verifies
mechanically that removing every non-input ASAD from the inputs leaves the
reconstruction unchanged, bit for bit.

Errors: the input ASADs carry independent statistical noise, and the
reconstruction at any angle is a fixed linear combination of them,

    P(phi0) = w_u U + sum_k w_k P_k,

so its variance is w_u^2 var[U] + sum_k w_k^2 var[P_k]. For the default
angles 0 and 45 degrees the weights are w_0 = cos(2 phi0), w_45 = sin(2 phi0)
and w_u = 1 - w_0 - w_45.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

from .asad import ASAD, ResponseSet
from .exceptions import BasisError

DEFAULT_BASIS_ANGLES = (0.0, 45.0)


# Below this magnitude a cosine or sine of twice an angle is set to zero.
# Without this, cos(2 * 45 deg) evaluates to 6e-17 and the
# reconstruction at 90 or 135 degrees carries a weight of order 1e-16 on an
# input that should not contribute; in a bin where every contributing input
# is empty that weight produces a prediction of order 1e-19 with a variance
# of order 1e-38, which a positive-variance rule then admits as a valid
# bin with an unbounded chi-squared.
TRIG_ZERO = 1e-12


def _cos_sin_twice(angles_deg) -> np.ndarray:
    """Columns (cos 2a, sin 2a), zeroed at multiples of 90 degrees."""
    a = np.deg2rad(np.asarray(angles_deg, dtype=float))
    cs = np.column_stack([np.cos(2.0 * a), np.sin(2.0 * a)])
    cs[np.abs(cs) < TRIG_ZERO] = 0.0
    return cs


def _design_matrix(angles_deg: Sequence[float]) -> np.ndarray:
    return _cos_sin_twice(angles_deg)


def check_basis_angles(angles_deg: Sequence[float]) -> tuple[float, ...]:
    """Validate a list of input polarization angles for mode 2.

    At least two angles are needed, and their doubled angles must span
    both the cosine and the sine direction (for example 0 and 90 degrees
    do not, since both measure only the cosine term). Returns the angles
    as a tuple of floats in the order given.
    """
    try:
        angles = tuple(float(a) for a in angles_deg)
    except TypeError:
        raise BasisError(
            f"basis_angles must be a sequence of angles in degrees, got "
            f"{angles_deg!r}."
        ) from None
    if len(angles) < 2:
        raise BasisError(
            "mode 2 needs polarized ASADs at two or more polarization angles; "
            f"got {len(angles)}. The default is (0, 45)."
        )
    if len(set(round(a % 180.0, 9) for a in angles)) < len(angles):
        raise BasisError(
            f"basis_angles {angles} contain a repeated angle (modulo 180 "
            "degrees); each input angle must be distinct."
        )
    A = _design_matrix(angles)
    if np.linalg.matrix_rank(A, tol=1e-9) < 2:
        raise BasisError(
            f"basis_angles {angles} cannot separate the cosine and sine "
            "terms of the polarization dependence (their doubled angles are "
            "collinear); choose angles such as (0, 45)."
        )
    return angles


class PolarizedBasis:
    """The mode 2 inputs for one energy, or for a stack of energies.

    Parameters
    ----------
    unpolarized : array
        The unpolarized ASAD, shape ``(..., n_bins)``.
    polarized : mapping
        ``{phi0_deg: array}`` with the fully polarized ASAD at each input
        angle, each of the same shape as ``unpolarized``. Two or more
        angles.
    var_unpolarized, var_polarized : array and mapping, optional
        Per-bin variances of the inputs, same shapes, for error
        propagation.
    """

    def __init__(
        self,
        unpolarized,
        polarized: Mapping[float, np.ndarray],
        var_unpolarized=None,
        var_polarized: Optional[Mapping[float, np.ndarray]] = None,
    ) -> None:
        self.u = np.asarray(unpolarized, dtype=float)
        self.angles = check_basis_angles(list(polarized.keys()))
        stack = []
        for a in self.angles:
            p = np.asarray(polarized[a], dtype=float)
            if p.shape != self.u.shape:
                raise BasisError(
                    f"polarized ASAD at {a} deg has shape {p.shape}, the "
                    f"unpolarized ASAD has shape {self.u.shape}; all inputs "
                    "must share one binning."
                )
            stack.append(p)
        self.p = np.stack(stack)
        self.var_u = (None if var_unpolarized is None
                      else np.asarray(var_unpolarized, dtype=float))
        if var_polarized is None:
            self.var_p = None
        else:
            self.var_p = np.stack([np.asarray(var_polarized[a], dtype=float)
                                   for a in self.angles])
        self._A = _design_matrix(self.angles)
        self._Apinv = np.linalg.pinv(self._A)          # (2, n_angles)
        diff = self.p - self.u[None]
        self._coeff = np.einsum("jk,k...->j...", self._Apinv, diff)

    @property
    def n_angles(self) -> int:
        return len(self.angles)

    def templates(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(U, M_c, M_s): the constant and the two sinusoid coefficients."""
        return self.u, self._coeff[0], self._coeff[1]

    def fit_residual(self) -> np.ndarray:
        """Per-input-angle residual P_k - (U + cos(2 phi0_k) M_c + sin(2 phi0_k) M_s),
        shape ``(n_angles, ...)``. Identically zero (to round-off) with two
        angles; with more, the least-squares misfit per bin."""
        model = self.u[None] + np.einsum("kj,j...->k...", self._A, self._coeff)
        return self.p - model

    def residual_rms(self) -> float:
        """Root mean square of :meth:`fit_residual` over all bins and angles."""
        r = self.fit_residual()
        return float(np.sqrt(np.mean(r * r)))

    def weights(self, pa_deg) -> tuple[np.ndarray, np.ndarray]:
        """Linear weights (w_u, w_k) such that the reconstruction at
        ``pa_deg`` equals w_u U + sum_k w_k P_k. ``pa_deg`` may be an
        array; the weights gain a leading axis."""
        cs = _cos_sin_twice(np.atleast_1d(pa_deg))                 # (n_pa, 2)
        w_k = cs @ self._Apinv                                     # (n_pa, n_angles)
        w_k[np.abs(w_k) < TRIG_ZERO] = 0.0
        w_u = 1.0 - w_k.sum(axis=1)
        return w_u, w_k


def reconstruct(basis: PolarizedBasis, pa_deg):
    """Fully polarized ASAD(s) at polarization angle(s) ``pa_deg``.

    ``pa_deg`` may be a scalar or an array (the result then gains a
    leading angle axis). Only the inputs held by ``basis`` are used.
    """
    scalar = np.ndim(pa_deg) == 0
    w_u, w_k = basis.weights(pa_deg)
    shape = (w_u.size,) + (1,) * basis.u.ndim
    out = (w_u.reshape(shape) * basis.u[None]
           + np.einsum("pk,k...->p...", w_k, basis.p))
    return out[0] if scalar else out


def reconstruct_var(basis: PolarizedBasis, pa_deg):
    """Variance of :func:`reconstruct` from the input variances:
    w_u^2 var[U] + sum_k w_k^2 var[P_k]."""
    if basis.var_u is None or basis.var_p is None:
        raise ValueError("input variances are required for reconstruct_var")
    scalar = np.ndim(pa_deg) == 0
    w_u, w_k = basis.weights(pa_deg)
    shape = (w_u.size,) + (1,) * basis.u.ndim
    out = ((w_u.reshape(shape) ** 2) * basis.var_u[None]
           + np.einsum("pk,k...->p...", w_k ** 2, basis.var_p))
    return out[0] if scalar else out


def holdout_validate(
    basis: PolarizedBasis,
    held_out: Sequence[np.ndarray],
    held_out_angles_deg: Sequence[float],
    norm: Optional[float] = None,
    variance: str = "model",
    held_out_vars: Optional[Sequence[np.ndarray]] = None,
) -> dict:
    """Compare the reconstruction against ASADs that were not ingested.

    For each held-out angle the reconstruction is evaluated and the
    reduced chi-squared of :func:`menhag.validation.calculate_reduced_chi_squared`
    is computed with the reconstruction's propagated variance as the
    prediction variance. ``variance="model"`` (the default) takes the data
    variance from the model expectation and needs ``norm``, the count
    normalization of the histograms; ``variance="observed"`` uses
    ``held_out_vars`` instead. Returns ``{angle: {"resid", "chi2nu",
    "chi2nu_err", "nu"}}``.
    """
    from .validation import calculate_reduced_chi_squared

    out = {}
    angles = np.atleast_1d(np.asarray(held_out_angles_deg, dtype=float))
    for i, pa in enumerate(angles):
        pred = reconstruct(basis, float(pa))
        data = np.asarray(held_out[i], dtype=float)
        entry = {"resid": data - pred}
        if basis.var_u is not None:
            sig_pred = np.sqrt(reconstruct_var(basis, float(pa)))
            if variance == "observed":
                if held_out_vars is None:
                    raise ValueError("variance='observed' needs held_out_vars")
                sig_data = np.sqrt(np.asarray(held_out_vars[i], dtype=float))
            else:
                sig_data = np.zeros_like(data)
            chi2, err, nu = calculate_reduced_chi_squared(
                data, sig_data, pred, sig_pred, variance=variance, norm=norm)
            entry.update({"chi2nu": chi2, "chi2nu_err": err, "nu": nu})
        out[float(pa)] = entry
    return out


def blind_test(
    basis_angles: Sequence[float],
    unpolarized,
    polarized_all: Mapping[float, np.ndarray],
    var_unpolarized=None,
    var_polarized_all: Optional[Mapping[float, np.ndarray]] = None,
    pa_eval: Sequence[float] = (30.0, 75.0),
) -> tuple[bool, dict]:
    """The blind test: removing every non-input ASAD must change nothing.

    The reconstruction is built once from the full mapping
    ``polarized_all`` (which may hold ASADs at many angles) and once from
    a copy with every angle other than ``basis_angles`` physically
    removed; the two must agree bit for bit at every angle in
    ``pa_eval``. Returns ``(passed, report)`` with the removed angles
    listed in ``report["removed_angles"]``.
    """
    angles = check_basis_angles(basis_angles)
    keys = {float(k): k for k in polarized_all}
    missing = [a for a in angles if a not in keys]
    if missing:
        raise BasisError(f"polarized ASADs missing at input angles {missing}")

    def _basis(mapping, var_mapping):
        pol = {a: mapping[keys[a]] for a in angles}
        var = (None if var_mapping is None
               else {a: var_mapping[keys[a]] for a in angles})
        return PolarizedBasis(unpolarized, pol, var_unpolarized, var)

    full = _basis(polarized_all, var_polarized_all)
    stripped_map = {keys[a]: polarized_all[keys[a]] for a in angles}
    stripped_var = (None if var_polarized_all is None
                    else {keys[a]: var_polarized_all[keys[a]] for a in angles})
    stripped = _basis(stripped_map, stripped_var)
    report = {}
    passed = True
    for pa in pa_eval:
        identical = bool(np.array_equal(reconstruct(full, float(pa)),
                                        reconstruct(stripped, float(pa))))
        report[float(pa)] = identical
        passed = passed and identical
    report["removed_angles"] = sorted(a for a in keys if a not in angles)
    return passed, report


def build_basis_response(config, input_adapter, output_adapter=None) -> ResponseSet:
    """Build a full response grid in mode 2 through an adapter.

    For each energy in ``config.energies`` the adapter supplies the
    unpolarized ASAD (``load_unpolarized``) and the fully polarized ASADs
    at each angle of ``config.basis_angles`` (``load_polarized``); the
    reconstruction then produces every angle in ``config.pol_angles``.
    With more than two input angles the per-energy fit residual is stored
    in ``response.meta["fit_residual_rms"]``.
    """
    angles = check_basis_angles(config.basis_angles)
    response = ResponseSet(
        unpolarized={},
        polarized={},
        pol_fraction=1.0,
        meta={
            "delta_deg": config.delta,
            "lambda_deg": config.lambda_,
            "mode": "polarized",
            "basis_angles": angles,
            "fit_residual_rms": {},
        },
    )
    for energy in config.energies:
        unpol = input_adapter.load_unpolarized(energy)
        if unpol.errors is None:
            unpol = unpol.with_poisson_errors(config.norm_factor)
        pol_asads = {a: input_adapter.load_polarized(energy, a) for a in angles}
        for a, asad in pol_asads.items():
            if asad.bin_edges.shape != unpol.bin_edges.shape or not np.allclose(
                asad.bin_edges, unpol.bin_edges
            ):
                raise BasisError(
                    f"The polarized ASAD at ({energy} keV, {a} deg) has a "
                    "different binning from the unpolarized ASAD; all inputs "
                    "must share identical bin edges."
                )
        have_vars = all(asad.errors is not None for asad in pol_asads.values())
        basis = PolarizedBasis(
            unpol.counts,
            {a: asad.counts for a, asad in pol_asads.items()},
            var_unpolarized=unpol.errors ** 2 if have_vars else None,
            var_polarized=({a: asad.errors ** 2 for a, asad in pol_asads.items()}
                           if have_vars else None),
        )
        response.unpolarized[energy] = unpol
        response.meta["fit_residual_rms"][energy] = basis.residual_rms()
        for pa in config.pol_angles:
            counts = reconstruct(basis, float(pa))
            errors = (np.sqrt(reconstruct_var(basis, float(pa)))
                      if have_vars else None)
            response.polarized[(energy, float(pa))] = ASAD(
                counts=counts,
                bin_edges=unpol.bin_edges.copy(),
                errors=errors,
                energy=float(energy),
                meta={"phi0_deg": float(pa), "method": "polarized_basis",
                      "basis_angles": angles},
            )
    if output_adapter is not None:
        output_adapter.save_response(response)
    return response
