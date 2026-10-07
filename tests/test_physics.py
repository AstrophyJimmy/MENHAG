"""The identities and invariances of the gated, transformed Klein-Nishina
azimuthal distribution (Sections 2 to 4 of the paper)."""

import numpy as np

from menhag.physics import (
    compute_raw_curves,
    harmonic_amplitudes,
    normalize_response,
    periodic_integral,
    rotation_matrix,
)

ETA = np.arange(0.5, 360.0, 1.0)
GATE = {"type": "erf", "mu": 25.0, "sigma": 75.0}


def corrected_curve(energy, delta, lam, phi0, threshold=None):
    f_pol, f_unp = compute_raw_curves(ETA, energy, delta, lam, phi0, threshold)
    return normalize_response(f_pol) / normalize_response(f_unp)


def test_on_axis_unpolarized_is_flat():
    """If this failed, the transformed cross-section would carry a geometric modulation on axis, where the paper's unpolarized ASAD is flat."""
    _, f_unp = compute_raw_curves(ETA, 200.0, 0.0, 0.0, 0.0, None)
    assert f_unp.std() / f_unp.mean() < 1e-12


def test_on_axis_polarized_is_pure_second_harmonic():
    """If this failed, the on-axis polarized ASAD would carry harmonics other than the cos 2 eta term of the classical analysis."""
    amps = harmonic_amplitudes(corrected_curve(200.0, 0.0, 0.0, 0.0))
    assert amps[2] > 0.3
    assert amps[1] < 1e-10 and amps[3] < 1e-10 and amps[4] < 1e-10


def test_on_axis_modulation_matches_the_klein_nishina_value():
    """If this failed, the on-axis modulation at 200 keV would differ from the Klein-Nishina value 0.444 the paper quotes, so the integrand or its normalization would be wrong."""
    assert np.isclose(harmonic_amplitudes(corrected_curve(200.0, 0.0, 0.0, 0.0))[2], 0.444, atol=5e-4)


def test_on_axis_maxima_perpendicular_to_polarization():
    """If this failed, the sign convention of the polarization angle would be reversed against the paper's definitions (photons scatter preferentially perpendicular to the electric vector)."""
    for phi0 in (0.0, 30.0, 45.0):
        f_pol, _ = compute_raw_curves(ETA, 200.0, 0.0, 0.0, phi0, None)
        eta_max = ETA[np.argmax(f_pol)]
        distance = min(
            abs(((eta_max - (phi0 + 90.0)) + 180.0) % 360.0 - 180.0),
            abs(((eta_max - (phi0 + 270.0)) + 180.0) % 360.0 - 180.0),
        )
        assert distance <= 1.0


def test_off_axis_introduces_higher_harmonics():
    """If this failed, the off-axis projection of Figure 4, the 360 degree component of the unpolarized ASAD and the higher harmonics of the corrected one, would be lost."""
    _, f_unp = compute_raw_curves(ETA, 200.0, 40.0, 0.0, 0.0, None)
    assert harmonic_amplitudes(normalize_response(f_unp))[1] > 0.1
    assert harmonic_amplitudes(corrected_curve(200.0, 40.0, 0.0, 0.0))[1] > 0.01


def test_lambda_is_a_pure_phase_shift():
    """If this failed, the source azimuth would change the shape of the ASAD instead of shifting it, so the frame transformation would be wrong."""
    shift = 30.0
    base, base_u = compute_raw_curves(ETA, 200.0, 40.0, 0.0, 20.0, None)
    moved, moved_u = compute_raw_curves((ETA + shift) % 360.0, 200.0, 40.0, shift, 20.0, None)
    assert np.allclose(base, moved, atol=1e-12)
    assert np.allclose(base_u, moved_u, atol=1e-12)


def test_rotation_matches_the_transformed_terms_for_pure_zenith():
    """If this failed, the rotation matrix would depart from the paper's transformed terms for the worked case lambda = phi0 = 0."""
    delta = 33.0
    rot = rotation_matrix(delta, 0.0, 0.0)
    rng = np.random.default_rng(1)
    d = np.radians(delta)
    for vartheta, varphi in rng.uniform(0, np.pi, size=(20, 2)):
        v = np.array([np.sin(vartheta) * np.cos(varphi), np.sin(vartheta) * np.sin(varphi), -np.cos(vartheta)])
        xg, yg, zg = rot @ v
        expected_xg = np.sin(vartheta) * np.cos(varphi) * np.cos(d) + np.cos(vartheta) * np.sin(d)
        expected_cos_theta = -(np.sin(vartheta) * np.cos(varphi) * np.sin(d) - np.cos(vartheta) * np.cos(d))
        assert np.isclose(xg, expected_xg)
        assert np.isclose(-zg, expected_cos_theta)


def test_gate_removes_counts_and_reshapes_the_curves():
    """If this failed, the threshold gate would not act inside the integral, so the gated ASADs of Section 4 would be the ungated ones."""
    f_pol, f_unp = compute_raw_curves(ETA, 200.0, 30.0, 0.0, 0.0, None)
    g_pol, g_unp = compute_raw_curves(ETA, 200.0, 30.0, 0.0, 0.0, GATE)
    assert periodic_integral(g_unp, ETA) < periodic_integral(f_unp, ETA)
    assert periodic_integral(g_pol, ETA) < periodic_integral(f_pol, ETA)
    shape_free = normalize_response(f_unp)
    shape_gated = normalize_response(g_unp)
    assert np.max(np.abs(shape_free - shape_gated)) > 1e-3
