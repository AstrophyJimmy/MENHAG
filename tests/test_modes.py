"""Mode 1 (the unpolarized-only construction), mode 2 (the two-angle
construction), the error propagation and the reduced chi-squared of the
paper."""

import numpy as np
import pytest

from menhag import ASAD, ResponseConfig, build_response, build_single, build_spectrum_response, predict_polarized_asad
from menhag.adapters import ArrayAdapter
from menhag.basis import PolarizedBasis, blind_test, reconstruct, reconstruct_var
from menhag.gates import ErfGate, GaussianSampledGate, SharpGate
from menhag.physics import compute_raw_curves, normalize_response
from menhag.validation import calculate_reduced_chi_squared

EDGES = np.linspace(0.0, 360.0, 61)
CENTERS = 0.5 * (EDGES[:-1] + EDGES[1:])
ETA = np.arange(0.5, 360.0, 1.0)
GATE = {"type": "erf", "mu": 25.0, "sigma": 75.0}


def synthetic_unpolarized(energy, delta, lam, instrumental, total=6.0e5):
    """A synthetic measured ASAD: the gated physics shape times a known instrumental imprint."""
    _, f_unp = compute_raw_curves(CENTERS, energy, delta, lam, 0.0, GATE)
    shape = normalize_response(f_unp) * instrumental
    counts = total * shape / shape.sum()
    return ASAD(counts=counts, bin_edges=EDGES, energy=energy)


def make_truth():
    """A synthetic instrument whose polarized ASADs follow the constant-plus-sinusoid form exactly."""
    u = 1000.0 * (1.0 + 0.2 * np.cos(np.radians(CENTERS - 30.0)))
    mc = 250.0 * np.cos(np.radians(2.0 * CENTERS)) + 40.0 * np.cos(np.radians(CENTERS))
    ms = -250.0 * np.sin(np.radians(2.0 * CENTERS)) + 25.0
    return u, mc, ms


def asad_at(u, mc, ms, pa):
    return u + np.cos(np.radians(2.0 * pa)) * mc + np.sin(np.radians(2.0 * pa)) * ms


# ----------------------------------------------------------------- mode 1
def test_kernel_round_trip_recovers_the_polarized_shape():
    """If this failed, the mode 1 kernel would not return the instrumental imprint times the polarized physics curve from the unpolarized ASAD (the kernel of Appendix B), the construction the paper's mode 1 rests on."""
    energy, delta, lam, phi0 = 200.0, 26.0, 10.0, 40.0
    instrumental = 1.0 + 0.25 * np.cos(np.radians(4.0 * CENTERS))
    unpol = synthetic_unpolarized(energy, delta, lam, instrumental)
    pred = predict_polarized_asad(unpol, phi0, delta=delta, lambda_=lam, threshold=GATE)
    f_pol, _ = compute_raw_curves(CENTERS, energy, delta, lam, phi0, GATE)
    expected = instrumental * normalize_response(f_pol)
    expected *= pred.counts.sum() / expected.sum()
    assert np.allclose(pred.counts, expected, rtol=1e-10)


def test_kernel_preserves_counts_and_fractional_errors():
    """If this failed, a mode 1 output would not keep the total counts of its input, or its per-bin errors would not follow the same multipliers as the counts (the error propagation of Section 5.3.1)."""
    unpol = synthetic_unpolarized(150.0, 10.0, 0.0, np.ones_like(CENTERS)).with_poisson_errors()
    pred = predict_polarized_asad(unpol, 70.0, delta=10.0, lambda_=0.0, threshold=GATE)
    assert np.isclose(pred.total_counts, unpol.total_counts)
    mask = unpol.counts > 0
    assert np.allclose(pred.errors[mask] / pred.counts[mask], unpol.errors[mask] / unpol.counts[mask])


def test_zero_polarization_returns_the_input():
    """If this failed, a polarization fraction of zero would not return the unpolarized input identically, so partial polarization would not enter through the physics curve alone."""
    unpol = synthetic_unpolarized(150.0, 25.0, 5.0, np.ones_like(CENTERS))
    pred = predict_polarized_asad(unpol, 30.0, delta=25.0, lambda_=5.0, threshold=GATE, pol_fraction=0.0)
    assert np.allclose(pred.counts, unpol.counts, rtol=1e-10)


def test_one_pass_grid_agrees_with_the_single_angle_path():
    """If this failed, the one-pass evaluator behind full response grids would differ from the single-angle path beyond the documented cache precision, so grids and single ASADs would disagree."""
    energy, delta, lam = 200.0, 26.4, 4.9
    unpol = synthetic_unpolarized(energy, delta, lam, np.ones_like(CENTERS))
    config = ResponseConfig(energies=(energy,), delta=delta, lambda_=lam, threshold=GATE,
                            pol_angles=(0.0, 30.0, 75.0), n_jobs=1)
    response = build_response(config, ArrayAdapter({energy: unpol}))
    assert response.meta["evaluator"] == "m2_pass"
    for phi0 in (0.0, 30.0, 75.0):
        direct = predict_polarized_asad(unpol.with_poisson_errors(), phi0, delta=delta, lambda_=lam,
                                        threshold=GATE, energy=energy)
        grid = response.polarized[(energy, phi0)]
        assert np.max(np.abs(grid.counts - direct.counts) / direct.counts.max()) < 5e-4


def test_norm_factor_sets_the_poisson_errors():
    """If this failed, the Poisson convention sigma = sqrt(n f) / f for normalized histograms (Equation 31 of the paper) would be wrong."""
    from menhag import poisson_errors

    unpol_counts = np.full(60, 400.0)
    assert np.allclose(poisson_errors(unpol_counts, 1.0), 20.0)
    assert np.allclose(poisson_errors(unpol_counts, 4.0), 10.0)
    _, err_f1 = build_single(unpol_counts, EDGES, 200.0, 0.0, 0.0, 0.0, GATE, norm_factor=1.0)
    _, err_f4 = build_single(unpol_counts, EDGES, 200.0, 0.0, 0.0, 0.0, GATE, norm_factor=4.0)
    assert np.allclose(err_f4, err_f1 / 2.0)


def test_spectrum_integrated_response_totals_and_errors():
    """If this failed, the spectrum-integrated mode 1 response the README documents would not weight the energies as given, or its errors would not combine in quadrature across energies."""
    energies = (100.0, 200.0, 300.0)
    unpol = {e: synthetic_unpolarized(e, 10.0, 0.0, np.ones_like(CENTERS), total=1e5) for e in energies}
    weights = np.array([0.5, 0.3, 0.2])
    config = ResponseConfig(energies=energies, delta=10.0, lambda_=0.0, threshold=GATE,
                            pol_angles=(0.0, 60.0), n_jobs=1, mode="spectrum")
    response = build_spectrum_response(config, ArrayAdapter(unpol), weights=weights)
    expected_total = sum(w * unpol[e].total_counts for w, e in zip(weights, energies))
    assert np.isclose(response.unpolarized[None].total_counts, expected_total)
    for phi0 in (0.0, 60.0):
        assert np.isclose(response.polarized[(None, phi0)].total_counts, expected_total)
    per_energy_err = [w * unpol[e].with_poisson_errors().errors for w, e in zip(weights, energies)]
    assert np.allclose(response.unpolarized[None].errors, np.sqrt(np.sum(np.square(per_energy_err), axis=0)))


# ----------------------------------------------------------------- mode 2
def test_two_angle_reconstruction_for_the_default_pair_and_a_general_pair():
    """If this failed, mode 2 would not recover the constant-plus-sinusoid form from the polarized ASADs at 0 and 45 degrees, or the two-by-two system for another pair would be wrong (Section 5.3.2)."""
    truth_u, mc, ms = make_truth()
    for angles, checks, rtol in (((0.0, 45.0), (30.0, 72.0, 120.0, 155.0), 1e-12), ((20.0, 95.0), (137.0,), 1e-10)):
        basis = PolarizedBasis(truth_u, {a: asad_at(truth_u, mc, ms, a) for a in angles})
        for pa in checks:
            assert np.allclose(reconstruct(basis, pa), asad_at(truth_u, mc, ms, pa), rtol=rtol)
    assert np.allclose(PolarizedBasis(truth_u, {a: asad_at(truth_u, mc, ms, a) for a in (0.0, 45.0)}).fit_residual(), 0.0, atol=1e-9)


def test_more_than_two_angles_combine_by_least_squares():
    """If this failed, the least-squares combination of more than two input angles would not recover the form, or a departure from the form at one input would go unreported in the residual."""
    truth_u, mc, ms = make_truth()
    pol = {a: asad_at(truth_u, mc, ms, a) for a in (0.0, 60.0, 120.0, 150.0)}
    basis = PolarizedBasis(truth_u, pol)
    assert basis.n_angles == 4
    for pa in (30.0, 100.0):
        assert np.allclose(reconstruct(basis, pa), asad_at(truth_u, mc, ms, pa), rtol=1e-12)
    assert basis.residual_rms() < 1e-9
    pol_bad = dict(pol)
    pol_bad[150.0] = pol[150.0] + 5.0 * np.cos(np.radians(4.0 * CENTERS))
    assert PolarizedBasis(truth_u, pol_bad).residual_rms() > 1.0


def test_reconstruction_variance_matches_the_hand_propagation():
    """If this failed, the propagated variance of a mode 2 reconstruction would differ from the hand derivation from the three input variances."""
    truth_u, mc, ms = make_truth()
    pol = {a: asad_at(truth_u, mc, ms, a) for a in (0.0, 45.0)}
    var_u = np.abs(truth_u)
    var_p = {a: np.abs(v) for a, v in pol.items()}
    basis = PolarizedBasis(truth_u, pol, var_u, var_p)
    pa = 30.0
    c2, s2 = np.cos(np.radians(2 * pa)), np.sin(np.radians(2 * pa))
    expected = (1 - c2 - s2) ** 2 * var_u + c2 ** 2 * var_p[0.0] + s2 ** 2 * var_p[45.0]
    assert np.allclose(reconstruct_var(basis, pa), expected, rtol=1e-12)


def test_blind_test_removes_every_non_input_angle():
    """If this failed, removing every simulated angle other than the two inputs would change the mode 2 result, against the paper's blind test."""
    truth_u, mc, ms = make_truth()
    pol_all = {a: asad_at(truth_u, mc, ms, a) for a in (0.0, 45.0, 10.0, 20.0, 30.0, 60.0, 120.0, 150.0)}
    var_all = {a: np.abs(v) for a, v in pol_all.items()}
    passed, report = blind_test((0.0, 45.0), truth_u, pol_all, np.abs(truth_u), var_all)
    assert passed
    assert report[30.0] and report[75.0]
    assert report["removed_angles"] == [10.0, 20.0, 30.0, 60.0, 120.0, 150.0]


def test_weights_vanish_at_multiples_of_ninety_degrees():
    """If this failed, the reconstruction at an angle 90 degrees from an input would carry a weight of order 1e-16 on the other input instead of zero, so an empty bin could acquire a prediction and a variance (the paper says such a pair carries one number)."""
    rng = np.random.default_rng(3)
    u = rng.uniform(0.5, 1.5, 360)
    p0 = rng.uniform(0.5, 1.5, 360)
    p45 = rng.uniform(0.5, 1.5, 360)
    u[10] = p0[10] = p45[10] = 0.0
    u[20] = 0.0
    p45[20] = 0.0
    b = PolarizedBasis(u, {0.0: p0, 45.0: p45}, var_unpolarized=u, var_polarized={0.0: p0, 45.0: p45})
    w_u, w_k = b.weights([90.0, 135.0, 180.0])
    assert np.array_equal(w_u, [2.0, 2.0, 0.0])
    assert np.array_equal(w_k, [[-1.0, 0.0], [0.0, -1.0], [1.0, 0.0]])
    assert np.array_equal(reconstruct(b, 90.0), 2.0 * u - p0)
    assert np.array_equal(reconstruct(b, 135.0), 2.0 * u - p45)
    assert reconstruct(b, 135.0)[20] == 0.0 and reconstruct_var(b, 135.0)[20] == 0.0
    assert reconstruct(b, 90.0)[10] == 0.0 and reconstruct_var(b, 90.0)[10] == 0.0


# ------------------------------------------------------------------ gates
def test_erf_gate_is_the_continuum_limit_of_the_sampled_gaussian():
    """If this failed, the error-function gate would not be the limit of the sampled Gaussian mixture of sharp gates, or would not reduce to the sharp gate at zero width (the gate appendix of the paper)."""
    erf_pol, erf_unp = compute_raw_curves(ETA, 200.0, 20.0, 0.0, 30.0, ErfGate(mu=0.0, sigma=30.0))
    smp_pol, smp_unp = compute_raw_curves(ETA, 200.0, 20.0, 0.0, 30.0, GaussianSampledGate(mean=0.0, sigma=30.0, points=401))
    for a, b in ((erf_pol, smp_pol), (erf_unp, smp_unp)):
        a_n, b_n = normalize_response(a), normalize_response(b)
        assert np.max(np.abs(a_n - b_n)) / a_n.max() < 5e-3
    sharp = compute_raw_curves(ETA, 200.0, 20.0, 0.0, 30.0, SharpGate(30.0))
    erf0 = compute_raw_curves(ETA, 200.0, 20.0, 0.0, 30.0, ErfGate(30.0, 0.0))
    assert np.allclose(sharp[0], erf0[0]) and np.allclose(sharp[1], erf0[1])


# ------------------------------------------------------- the chi-squared
def test_reduced_chi_squared_convention():
    """If this failed, the reduced chi-squared of Section 6 would not give zero for identical distributions, one for a one-sigma shift under the observed variance, or the model-variance value the paper's convention implies."""
    y = np.full(60, 400.0)
    s = np.sqrt(y)
    chi2, err, nu = calculate_reduced_chi_squared(y, s, y, s, norm=1.0)
    assert chi2 == 0.0 and nu == 60
    shifted = y + np.sqrt(2.0) * s
    chi2, err, nu = calculate_reduced_chi_squared(y, s, shifted, s, variance="observed")
    assert np.isclose(chi2, 1.0)
    assert np.isclose(err, 2.0 * np.sqrt(chi2 / nu))
    chi2_m, _, _ = calculate_reduced_chi_squared(y, s, shifted, s, norm=1.0)
    assert chi2_m < 1.0
    assert np.isclose(chi2_m, 2.0 * y[0] / (shifted[0] + y[0]))
    with pytest.raises(ValueError):
        calculate_reduced_chi_squared(y, s, shifted, s)
