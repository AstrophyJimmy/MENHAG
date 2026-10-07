# MENHAG, Mohamed Gamal Zaid, University of New Hampshire. MIT license, see LICENSE.
"""Validation against reference responses: reduced chi squared and plots.

Given a known polarized reference (for example a full GEANT4 polarized
simulation), this module quantifies the agreement with the menhag prediction
using the reduced chi-squared of the paper's verification section:

    chi2_i   = (y_true_i - y_pred_i)^2 / (y_pred_i / f + sigma_pred_i^2)
    chi2_nu  = (1 / nu) * sum_i chi2_i        over the nu non-empty bins

where the first term of the denominator is the Poisson variance of the
reference evaluated at the model expectation (f the count normalization
of the histograms) and sigma_pred is the propagated uncertainty of the
prediction.

No parameters are fitted, so chi2_nu = 1 does not carry the usual
goodness-of-fit interpretation; it indicates that the residuals between the
two fixed distributions are consistent, on average, with their combined
uncertainties. The 1 sigma uncertainty on chi2_nu from standard propagation
is 2 * sqrt(chi2_nu / nu).

Plotting needs matplotlib (``pip install menhag[viz]``); imported lazily.
The plots take their colours, linestyles and markers from the roles of
:mod:`menhag.palette`: the reference in the ``reference`` role, the
prediction in ``mode1`` (or ``mode2``), the unpolarized input in
``auxiliary2``, and the chi-squared map in the diverging map centred at
one.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import palette
from .asad import ASAD


def calculate_reduced_chi_squared(
    y_true: np.ndarray,
    sigma_true: np.ndarray,
    y_pred: np.ndarray,
    sigma_pred: np.ndarray,
    *,
    variance: str = "model",
    norm: float | None = None,
) -> tuple[float, float, int]:
    """Modified reduced chi squared between two fixed distributions.

    Parameters
    ----------
    variance : {"model", "observed"}
        Where the variance of the DATA comes from. "model" (the default)
        takes it from the prediction, ``y_pred / norm``, which is the
        unbiased choice at finite counts. "observed" squares
        ``sigma_true`` instead, rebuilding the variance from the measured
        counts themselves; it is available for comparison and is biased
        low, because a bin that fluctuates low is given a smaller variance
        and therefore a larger weight. ``sigma_pred`` is used in both cases: it
        propagates the uncertainty of the inputs the prediction was built
        from, which is a genuine uncertainty rather than a weighting choice.
    norm : float or None
        Count normalization of the histograms, so that raw counts are
        ``value * norm``. Required by the "model" branch and deliberately
        without a default: a normalized histogram passed with an assumed
        norm of 1.0 returns a statistic wrong by exactly that factor, and
        wrong in the quiet direction, toward zero. Pass 1.0 explicitly when
        the inputs are raw counts.

    Returns
    -------
    (chi2_nu, chi2_nu_error, nu) : tuple
        The statistic, its propagated 1 sigma uncertainty, and the number
        of contributing (non-empty, finite-variance) bins.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([100.0, 110.0, 90.0])
    >>> s = np.sqrt(y)
    >>> chi2, err, nu = calculate_reduced_chi_squared(y, s, y, s)
    >>> chi2 == 0.0 and nu == 3
    True
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if variance == "model":
        if norm is None:
            raise ValueError(
                "variance='model' needs `norm`, the count normalization of the "
                "histograms, so that raw counts are value * norm. Pass norm=1.0 "
                "if the inputs are already raw counts. There is deliberately no "
                "default: guessing it silently returns a meaningless statistic "
                "when the inputs are normalized, since the variance would be "
                "wrong by exactly that factor."
            )
        data_var = np.clip(y_pred, 0.0, None) / float(norm)
    elif variance == "observed":
        data_var = np.asarray(sigma_true, dtype=float) ** 2
    else:
        raise ValueError(
            f"variance must be 'model' or 'observed', not {variance!r}"
        )
    variance = data_var + np.asarray(sigma_pred, dtype=float) ** 2
    mask = (variance > 0) & (y_true != 0)
    nu = int(mask.sum())
    if nu == 0:
        raise ValueError("No non-empty bins with positive variance to compare.")
    chi2_bins = (y_true[mask] - y_pred[mask]) ** 2 / variance[mask]
    chi2_nu = float(chi2_bins.sum() / nu)
    chi2_nu_error = float(2.0 * np.sqrt(chi2_nu / nu))
    return chi2_nu, chi2_nu_error, nu


def compare_asads(
    reference: ASAD,
    predicted: ASAD,
    *,
    variance: str = "model",
    norm: float | None = None,
) -> tuple[float, float, int]:
    """Reduced chi squared between two ASAD objects (must share binning).

    ``variance`` and ``norm`` are passed through to
    :func:`calculate_reduced_chi_squared`; see there for their meaning.
    """
    if reference.bin_edges.shape != predicted.bin_edges.shape or not np.allclose(
        reference.bin_edges, predicted.bin_edges
    ):
        raise ValueError("ASADs must share identical bin edges to compare.")
    if reference.errors is None or predicted.errors is None:
        raise ValueError("Both ASADs must carry per-bin errors.")
    return calculate_reduced_chi_squared(
        reference.counts,
        reference.errors,
        predicted.counts,
        predicted.errors,
        variance=variance,
        norm=norm,
    )


def _import_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Plotting needs the optional dependency 'matplotlib'. "
            "Install it with: pip install menhag[viz]"
        ) from exc
    return plt


def plot_overlay(
    unpolarized: ASAD,
    predicted: ASAD,
    reference: Optional[ASAD] = None,
    ax=None,
    title: Optional[str] = None,
    role: str = "mode1",
    norm: float | None = None,
):
    """Overlay the unpolarized input, the prediction and, when given, a
    reference, each in its palette role.

    ``role`` is the role of the prediction (``"mode1"`` or ``"mode2"``).
    With a reference and ``norm`` (the count normalization of the
    histograms, see :func:`calculate_reduced_chi_squared`) the reduced
    chi-squared between reference and prediction is the default title.
    Returns the matplotlib axes.
    """
    plt = _import_pyplot()
    palette.apply(plt)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.6, 3.5))
    ax.errorbar(
        unpolarized.bin_centers, unpolarized.counts, yerr=unpolarized.errors,
        fmt=palette.mk("auxiliary2"), ms=3, lw=0.8, color=palette.c("auxiliary2"),
        label="unpolarized input",
    )
    ax.errorbar(
        predicted.bin_centers, predicted.counts, yerr=predicted.errors,
        fmt=palette.mk(role), ms=3, lw=0.8, color=palette.c(role),
        label=f"{role} prediction",
    )
    if reference is not None:
        ax.errorbar(
            reference.bin_centers, reference.counts, yerr=reference.errors,
            fmt=palette.mk("reference"), ms=3, lw=0.8, color=palette.c("reference"),
            label="reference",
        )
        if title is None and norm is not None:
            chi2, err, nu = compare_asads(reference, predicted, norm=norm)
            title = f"reduced chi2 = {chi2:.2f} +/- {err:.2f} ({nu} bins)"
    if title:
        ax.set_title(title)
    ax.set_xlabel("azimuthal scattering angle eta [deg]")
    ax.set_ylabel("counts")
    ax.legend(frameon=False)
    return ax


def plot_chi2_heatmap(chi2_grid: dict, ax=None, title: Optional[str] = None):
    """Map of the reduced chi-squared over the (energy x polarization
    angle) grid, in the diverging map of the paper: blue below one, white
    at one, vermillion above.

    Parameters
    ----------
    chi2_grid : dict
        ``{(energy_kev, phi0_deg): chi2_nu}``, e.g. accumulated by calling
        :func:`compare_asads` over a response grid.

    Returns the matplotlib axes.
    """
    plt = _import_pyplot()
    palette.apply(plt)
    energies = sorted({e for e, _ in chi2_grid})
    angles = sorted({a for _, a in chi2_grid})
    grid = np.full((len(energies), len(angles)), np.nan)
    for (e, a), value in chi2_grid.items():
        grid[energies.index(e), angles.index(a)] = value
    if ax is None:
        _, ax = plt.subplots(figsize=(5.6, 3.5))
    image = ax.imshow(
        grid,
        aspect="auto",
        origin="lower",
        extent=(min(angles), max(angles), min(energies), max(energies)),
        cmap=palette.diverging_cmap(),
        norm=palette.diverging_norm(grid, center=1.0),
    )
    ax.figure.colorbar(image, ax=ax, label="reduced chi2")
    ax.set_xlabel("polarization angle phi0 [deg]")
    ax.set_ylabel("energy [keV]")
    if title:
        ax.set_title(title)
    return ax
