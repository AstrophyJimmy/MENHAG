# MENHAG, Mohamed Gamal Zaid, University of New Hampshire. MIT license, see LICENSE.
"""The palette of the accompanying paper's figures, by role.

Every plot the package draws, in :mod:`menhag.validation` and in the
notebooks, takes its colour, linestyle and marker from a fixed role, so
that the same thing has the same appearance wherever it appears and no
figure relies on colour alone. The colours are the Okabe-Ito set without
the reddish purple, plus a neutral grey.

Roles:

- ``reference``: a simulation or the data (the GEANT4 response, the
  delivered ASADs, the on-axis reference curve): blue, solid, circle.
- ``mode1``: this work's analytic construction (mode 1, the projected
  Klein-Nishina ASAD, the closed-form gate): vermillion, dashed, square.
- ``mode2``: the two-angle construction: bluish green, dash-dot, triangle.
- ``comparison``: a second curve of this work drawn for comparison:
  orange, dotted, diamond.
- ``literature``: a curve from the literature: black, dashed, cross.
- ``fit``: a fit to data: black, solid, thin, no marker.
- ``auxiliary``: a third element (a per-bin chi-squared, sampled
  weights, a prior): grey, dash-dot-dot, inverted triangle.
- ``auxiliary2``: a faint or fourth element (unpolarized data or bars):
  sky blue, solid, plus.
- ``truth``: a simulated input value or an exact reference: black, thin
  dotted, no marker.

``HARMONICS`` styles the k = 1 to 5 harmonic components, ``NEUTRAL`` a
categorical series that carries no role, ``DIVERGING`` every diverging
map (blue below the centre, white at the centre, vermillion above it),
and ``SEQUENTIAL`` a single-signed map (white to blue). ``RC`` is the
matplotlib rc block of the paper's figures. Matplotlib is imported only
by :func:`apply`, :func:`diverging_cmap` and :func:`sequential_cmap`.
"""

from __future__ import annotations

OI = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "black": "#000000",
    "grey": "#7F7F7F",
}

ROLE = {
    "reference": dict(color=OI["blue"], ls="-", marker="o"),
    "mode1": dict(color=OI["vermillion"], ls="--", marker="s"),
    "mode2": dict(color=OI["green"], ls="-.", marker="^"),
    "comparison": dict(color=OI["orange"], ls=":", marker="D"),
    "auxiliary": dict(color=OI["grey"], ls=(0, (3, 1, 1, 1)), marker="v"),
    "auxiliary2": dict(color=OI["skyblue"], ls="-", marker="P"),
    "truth": dict(color=OI["black"], ls=(0, (1, 1)), marker="none"),
    "literature": dict(color=OI["black"], ls="--", marker="x"),
    "fit": dict(color=OI["black"], ls="-", marker="none"),
}

HARMONICS = [
    dict(color=OI["blue"], ls="-"),
    dict(color=OI["orange"], ls="--"),
    dict(color=OI["green"], ls="-."),
    dict(color=OI["skyblue"], ls=":"),
    dict(color=OI["grey"], ls=(0, (5, 1, 1, 1))),
]

NEUTRAL = [
    dict(color=OI["grey"], ls="-", marker="o"),
    dict(color=OI["skyblue"], ls="--", marker="s"),
    dict(color=OI["orange"], ls="-.", marker="^"),
    dict(color=OI["black"], ls=":", marker="D"),
]

DIVERGING = [OI["blue"], "#FFFFFF", OI["vermillion"]]
SEQUENTIAL = ["#FFFFFF", OI["blue"]]

RC = {
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.3,
    "savefig.dpi": 300,
    "mathtext.fontset": "dejavuserif",
}


def c(role: str) -> str:
    """The colour of a role."""
    return ROLE[role]["color"]


def ls(role: str):
    """The linestyle of a role."""
    return ROLE[role]["ls"]


def mk(role: str) -> str:
    """The marker of a role."""
    return ROLE[role]["marker"]


def style(role: str, **overrides) -> dict:
    """Keyword arguments for a matplotlib line in the given role
    (``color``, ``ls``, ``marker``), with any overrides applied."""
    out = dict(ROLE[role])
    out.update(overrides)
    return out


def apply(plt=None):
    """Install the paper's rc block on matplotlib (imported here)."""
    if plt is None:
        import matplotlib.pyplot as plt
    plt.rcParams.update(RC)
    return plt


def diverging_cmap():
    """The diverging map of the paper: blue, white, vermillion."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("menhag_diverging", DIVERGING)


def sequential_cmap():
    """A single-signed map from white to the reference blue."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("menhag_sequential", SEQUENTIAL)


def diverging_norm(values, center: float = 1.0):
    """A two-slope normalization with white at ``center`` and the ends at
    the extremes of ``values``; the low end is at most ``center`` and the
    high end at least ``center``, so the centre is always inside."""
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    lo = float(min(finite.min(), center)) if finite.size else 0.0
    hi = float(max(finite.max(), center)) if finite.size else 2.0 * center
    if lo == center:
        lo = center - 1e-6
    if hi == center:
        hi = center + 1e-6
    return TwoSlopeNorm(vmin=lo, vcenter=center, vmax=hi)
