"""The quickstart against its recorded output, and both constructions
against the held-out ASADs of the shipped sample data."""

import filecmp
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(ROOT, "examples")
SAMPLE = os.path.join(EXAMPLES, "sample_data", "ideal_detector_asads.csv")

RECORDED = """sample data: 3 energies, 60 bins, polarized angles [0.0, 45.0, 90.0, 135.0]
[1/5] build_single: one ASAD in, one polarized ASAD out
  wrote quickstart_output/single_200keV_phi90.csv
  against the sampled polarized ASAD at 90 deg: reduced chi2 = 1.237 +/- 0.287 (60 bins)
[2/5] build_ideal: physics curves and the ideal modulation
  ideal mu at 200 keV, delta = 30 deg: 0.4229
[3/5] build_response: full energy x angle grid
  generated 39 polarized ASADs (3 energies x 13 angles)
[4/5] build_spectrum_response: spectrum-weighted ASADs
  wrote quickstart_output/spectrum_phi0.csv
[5/5] mode 2: full grid from the polarized ASADs at 0 and 45 deg
  generated 39 ASADs from 2 polarized angles per energy
Validation against the held-out sampled ASADs (reduced chi2, mode 1 | mode 2)
  E =   100 keV:  phi0 =  90 deg: 1.31 | 0.93  phi0 = 135 deg: 1.11 | 0.93
  E =   200 keV:  phi0 =  90 deg: 1.24 | 0.94  phi0 = 135 deg: 0.89 | 0.86
  E =   300 keV:  phi0 =  90 deg: 1.01 | 1.00  phi0 = 135 deg: 1.10 | 1.15
Done. Outputs are in ./quickstart_output/
"""


def test_quickstart_prints_its_recorded_output(tmp_path):
    """If this failed, the quickstart would not run to completion on the shipped sample data, its printed block would differ from the one recorded here and in the README, or the sample file would no longer regenerate byte for byte from the package's own builder (the two are reported apart)."""
    sys.path.insert(0, EXAMPLES)
    try:
        import make_sample_data
    finally:
        sys.path.pop(0)
    regenerated = tmp_path / "regenerated.csv"
    make_sample_data.write(str(regenerated))
    assert filecmp.cmp(str(regenerated), SAMPLE, shallow=False), "the sample file does not regenerate byte for byte"

    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(ROOT, "src") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run([sys.executable, os.path.join(EXAMPLES, "quickstart.py")],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-2000:]
    lines = result.stdout.splitlines()
    start = lines.index(RECORDED.splitlines()[0])
    printed = "\n".join(lines[start:start + len(RECORDED.splitlines())]) + "\n"
    assert printed == RECORDED, "the printed block differs from the recorded output"
    assert os.path.exists(tmp_path / "quickstart_output" / "single_200keV_phi90.csv")


def test_both_modes_reproduce_the_held_out_sample_asads():
    """If this failed, mode 1 from the unpolarized sample ASAD, or mode 2 from the two sampled polarized ASADs at 0 and 45 degrees, would not reproduce the held-out sampled ASADs at 90 and 135 degrees at the statistical level (an ideal detector, for which both constructions are exact up to the Poisson noise)."""
    from menhag import PolarizedBasis, build_single, calculate_reduced_chi_squared, reconstruct, reconstruct_var

    table = np.genfromtxt(SAMPLE, delimiter=",", names=True)
    assert set(table.dtype.names) == {"energy", "bin_low", "bin_high", "counts", "error", "pol", "angle"}
    gate = {"type": "erf", "mu": 25.0, "sigma": 75.0}
    for energy in (100.0, 200.0, 300.0):
        unp = table[(table["energy"] == energy) & (table["pol"] == 0)]
        edges = np.append(unp["bin_low"], unp["bin_high"][-1])

        def polarized(angle):
            return table[(table["energy"] == energy) & (table["pol"] == 100) & (table["angle"] == angle)]

        basis = PolarizedBasis(
            unp["counts"], {a: polarized(a)["counts"] for a in (0.0, 45.0)},
            var_unpolarized=unp["error"] ** 2,
            var_polarized={a: polarized(a)["error"] ** 2 for a in (0.0, 45.0)},
        )
        for angle in (90.0, 135.0):
            ref = polarized(angle)
            pred1, err1 = build_single(unp["counts"], edges, energy, 30.0, 0.0, angle, gate, errors=unp["error"])
            chi2_1, _, nu = calculate_reduced_chi_squared(ref["counts"], ref["error"], pred1, err1, norm=1.0)
            pred2 = reconstruct(basis, angle)
            err2 = np.sqrt(reconstruct_var(basis, angle))
            chi2_2, _, _ = calculate_reduced_chi_squared(ref["counts"], ref["error"], pred2, err2, norm=1.0)
            assert nu == 60
            assert 0.5 < chi2_1 < 1.8, (energy, angle, chi2_1)
            assert 0.5 < chi2_2 < 1.8, (energy, angle, chi2_2)
