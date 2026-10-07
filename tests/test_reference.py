"""The ideal-detector reference: the package integrator against the stored
off-axis curves of an independent adaptive-quadrature evaluation of the
transformed cross-section, without a gate and with the error-function
gate (tests/data/offaxis_reference.npz, the author's own product)."""

import os

import numpy as np

from menhag.gates import ErfGate
from menhag.physics import FF

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "offaxis_reference.npz")
TOL = 1e-6


def _worst(gated):
    d = np.load(FIXTURE)
    eta = d["eta_deg"]
    worst = 0.0
    rows = zip(d["gated_params"], d["gated_pol"], d["gated_unp"]) if gated else \
        zip(d["ungated_params"], d["ungated_pol"], d["ungated_unp"])
    for params, pol, unp in rows:
        if gated:
            E, delta, lam, phi0, mu, sigma = params
            gate = ErfGate(float(mu), float(sigma))
        else:
            E, delta, lam, phi0 = params
            gate = None
        F_100, F_U = FF(eta, float(E), float(phi0), float(delta), float(lam), gate)
        worst = max(worst, np.max(np.abs(F_100 - pol)), np.max(np.abs(F_U - unp)))
    return worst


def test_ungated_off_axis_curves_match_the_reference():
    """If this failed, the integrator would disagree at 1e-6 with the independent evaluation of the transformed cross-section over 54 off-axis combinations, so the off-axis physics of the paper would not be what the package computes."""
    assert _worst(gated=False) < TOL


def test_gated_off_axis_curves_match_the_reference():
    """If this failed, the same disagreement would hold with the error-function gate over 24 combinations, so the gated curves of Section 4 would be wrong."""
    assert _worst(gated=True) < TOL
