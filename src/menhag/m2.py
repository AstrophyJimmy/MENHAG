# MENHAG, Mohamed Gamal Zaid, University of New Hampshire. MIT license, see LICENSE.
"""Fast erf-gate evaluator: all polarization angles in one pass.

For building a full response grid, evaluating each polarization angle
separately repeats the same geometry work many times over. The
:class:`M2GRB` class in this module precomputes everything that depends
only on the source position and the eta binning (a compact float32
geometry cache covering every requested polarization angle at once), so
that each energy then needs just a single vectorized pass to produce the
(F_100, F_U) curves of every angle. This is the evaluator used to
generate the paper's full POLAR responses.

Two numerical details matter and are kept deliberately:

- The gate weight depends only on the smaller of the two energy
  deposits, m. It is tabulated on a uniform m-grid in float64 using the
  complementary error function, erfc((mu - m) / (sigma sqrt(2))) / 2,
  rather than 0.5 * (1 + erf(...)). The two forms are mathematically
  identical, but for deposits far below threshold the erf form loses all
  precision in float32 (1 + erf cancels catastrophically), while erfc
  remains accurate deep into the suppressed tail.
- The table is then gathered onto the node array with a hand-written
  float32 linear interpolation instead of np.interp, which would force
  the whole array through float64 and dominate the evaluation cost.

The default quadrature is 64 Gauss-Legendre nodes on each of 4 panels,
the configuration used for the paper. The single-angle float64 path
(:func:`menhag.physics.FF_smooth`) agrees with this evaluator at the
float32 cache precision (about 1e-5 relative); use that path for single
angles and this one for grids.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erfc

from . import physics as _physics

MEC2 = 511.0
SQRT2 = np.sqrt(2.0)


class M2GRB:
    """Precomputed (mu, sigma)-independent geometry for one source position:
    cosT/x2/y2/st*w arrays of shape (nPA, n_eta, n_nodes_total)."""

    def __init__(self, pols_deg, bins_deg, delta_deg, lambda_deg,
                 n_nodes=64, n_panels=4):
        pols = np.asarray(pols_deg, dtype=float)
        edges = np.linspace(0.0, np.pi, n_panels + 1)
        gx, gw = np.polynomial.legendre.leggauss(n_nodes)
        theta = np.concatenate([0.5 * (edges[i] + edges[i + 1])
                                + 0.5 * (edges[i + 1] - edges[i]) * gx
                                for i in range(n_panels)])
        wts = np.concatenate([0.5 * (edges[i + 1] - edges[i]) * gw
                              for i in range(n_panels)])
        st, ct = np.sin(theta), np.cos(theta)

        nPA, nB = len(pols), len(np.atleast_1d(bins_deg))
        cosT = np.empty((nPA, nB, len(theta)), dtype=np.float32)
        x2 = np.empty_like(cosT)
        y2 = np.empty_like(cosT)
        for k, pa in enumerate(pols):
            R = _physics.trans_matrix(np.deg2rad(delta_deg), np.deg2rad(pa),
                                      np.deg2rad(lambda_deg))
            a13, a23, a33 = R[2], R[5], R[8]
            C1, X1, Y1 = _physics._eta_consts(bins_deg, R)
            cosT[k] = (-(st[None, :] * C1[:, None] - ct[None, :] * a33)
                       ).astype(np.float32)
            x2[k] = ((st[None, :] * X1[:, None] - ct[None, :] * a13) ** 2
                     ).astype(np.float32)
            y2[k] = ((st[None, :] * Y1[:, None] - ct[None, :] * a23) ** 2
                     ).astype(np.float32)
        self.cosT = cosT
        self.x2 = x2
        self.y2 = y2
        self.stw = (st * wts).astype(np.float32)     # (n_nodes_total,)

    def FF(self, energy, mu, sigma, n_tab=16384):
        """Normalized (F_100, F_U) for all polarization angles at one energy.
        Returns arrays (nPA, n_eta).

        The gate weight is a function of the smaller deposit m only:
        g(m) = Phi((m - mu)/sigma) = erfc((mu - m)/(sigma sqrt(2)))/2. It
        is tabulated on an m-grid in float64 (erfc stays accurate deep in
        the suppressed tail, where 0.5 * (1 + erf) cancels in float32)
        and linearly interpolated onto the node array (relative error
        below about 5e-5 for sigma of a few keV and up, scaling as
        (dm/sigma)^2 for open gates)."""
        eps = np.float32(energy / MEC2)
        frac = 1.0 / (1.0 + eps * (1.0 - self.cosT))
        Ep = np.float32(energy) * frac
        m = np.minimum(np.float32(energy) - Ep, Ep)
        # float32 linear table lookup on the uniform m-grid (np.interp forces
        # float64 round-trips that dominated the eval cost)
        mgrid = np.linspace(0.0, energy / 2.0, n_tab + 1)
        gtab = (0.5 * erfc((mu - mgrid) / (sigma * SQRT2))).astype(np.float32)
        gtab = np.append(gtab, gtab[-1])            # pad for idx+1 gather
        inv_dm = np.float32(n_tab / (energy / 2.0))
        pos = m * inv_dm
        idx = pos.astype(np.int32)
        np.clip(idx, 0, n_tab, out=idx)
        fr = pos - idx.astype(np.float32)
        g = gtab[idx] * (np.float32(1.0) - fr) + gtab[idx + 1] * fr
        frac2 = frac * frac
        common = frac + frac * frac2
        gw = g * self.stw
        num_pol = ((common - 2.0 * frac2 * self.x2) * gw).sum(axis=2)
        num_unp = ((common - frac2 * (self.x2 + self.y2)) * gw).sum(axis=2)
        npm = num_pol.mean(axis=1) * 2.0 * np.pi
        num = num_unp.mean(axis=1) * 2.0 * np.pi
        F1 = np.where(npm[:, None] > 1e-12,
                      num_pol / np.where(npm[:, None] > 1e-12,
                                         npm[:, None], 1.0), 0.0)
        FU = np.where(num[:, None] > 1e-12,
                      num_unp / np.where(num[:, None] > 1e-12,
                                         num[:, None], 1.0), 0.0)
        return F1, FU
