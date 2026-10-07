# MENHAG, Mohamed Gamal Zaid, University of New Hampshire. MIT license, see LICENSE.
"""MENHAG quickstart: every entry point on the shipped sample data.

Run from the examples directory after ``pip install .``:

    python quickstart.py

It needs nothing but the package and the sample file
``sample_data/ideal_detector_asads.csv``, which the package wrote itself
with ``make_sample_data.py`` (ideal-detector ASADs, 30 degrees off axis,
the POLAR example gate, Poisson-sampled). It writes CSV outputs to
``./quickstart_output/`` and prints a validation summary. Angles are
degrees, energies keV throughout. The numbers printed are deterministic.
"""

from __future__ import annotations

import os

import numpy as np

from menhag import (
    ArrayAdapter,
    ResponseConfig,
    build_basis_response,
    build_ideal,
    build_response,
    build_single,
    build_spectrum_response,
    calculate_reduced_chi_squared,
)

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(HERE, "sample_data", "ideal_detector_asads.csv")
OUT = "quickstart_output"
os.makedirs(OUT, exist_ok=True)

# The geometry and gate the sample data were made with (see
# make_sample_data.py). The gate values are the POLAR example of the
# paper, instrument-specific inputs.
DELTA, LAMBDA = 30.0, 0.0
GATE = {"type": "erf", "mu": 25.0, "sigma": 75.0}
NORM = 1.0  # the sample ASADs are raw counts

# ------------------------------------------------------------------ #
# Read the sample data (core dependencies only: numpy)
# ------------------------------------------------------------------ #
table = np.genfromtxt(SAMPLE, delimiter=",", names=True)
energies = tuple(sorted(set(table["energy"])))
unpol, polar = {}, {}
for e in energies:
    sel = table[(table["energy"] == e) & (table["pol"] == 0)]
    edges = np.append(sel["bin_low"], sel["bin_high"][-1])
    unpol[e] = (sel["counts"], edges, sel["error"])
    for a in sorted(set(table["angle"][(table["energy"] == e) & (table["pol"] == 100)])):
        sel = table[(table["energy"] == e) & (table["pol"] == 100) & (table["angle"] == a)]
        polar[(e, a)] = (sel["counts"], edges, sel["error"])
EDGES = unpol[energies[0]][1]
print(f"sample data: {len(energies)} energies, {len(EDGES) - 1} bins, "
      f"polarized angles {[float(a) for a in sorted({a for _, a in polar})]}")


def save_csv(name: str, counts: np.ndarray, errors: np.ndarray) -> None:
    path = os.path.join(OUT, name)
    rows = np.column_stack([EDGES[:-1], EDGES[1:], counts, errors])
    np.savetxt(path, rows, delimiter=",", header="bin_low,bin_high,counts,error",
               comments="", fmt="%.6f")
    print(f"  wrote {path}")


# ------------------------------------------------------------------ #
# Mode 1, one ASAD: one unpolarized ASAD to one polarized ASAD
# ------------------------------------------------------------------ #
print("[1/5] build_single: one ASAD in, one polarized ASAD out")
counts, edges, errors = unpol[200.0]
pol_counts, pol_errors = build_single(
    counts, edges, energy=200.0, delta=DELTA, lambda_=LAMBDA, phi0=90.0,
    threshold=GATE, errors=errors,
)
save_csv("single_200keV_phi90.csv", pol_counts, pol_errors)
ref, _, ref_err = polar[(200.0, 90.0)]
chi2, chi2_err, nu = calculate_reduced_chi_squared(ref, ref_err, pol_counts, pol_errors, norm=NORM)
print(f"  against the sampled polarized ASAD at 90 deg: "
      f"reduced chi2 = {chi2:.3f} +/- {chi2_err:.3f} ({nu} bins)")

# ------------------------------------------------------------------ #
# Ideal detector curves, no input ASAD at all
# ------------------------------------------------------------------ #
print("[2/5] build_ideal: physics curves and the ideal modulation")
ideal = build_ideal(200.0, DELTA, LAMBDA, phi0=90.0, threshold=GATE)
print(f"  ideal mu at 200 keV, delta = {DELTA:.0f} deg: {ideal['mu']:.4f}")

# ------------------------------------------------------------------ #
# Mode 1, full grid: energy x polarization angle through adapters
# ------------------------------------------------------------------ #
print("[3/5] build_response: full energy x angle grid")
adapter = ArrayAdapter(unpol)
config = ResponseConfig(
    energies=energies, delta=DELTA, lambda_=LAMBDA, threshold=GATE,
    pol_angles=range(0, 181, 15), n_jobs=1,
)
response = build_response(config, adapter, adapter)
print(f"  generated {len(response.polarized)} polarized ASADs "
      f"({len(energies)} energies x {len(config.pol_angles)} angles)")

# ------------------------------------------------------------------ #
# Mode 1, spectrum-integrated response (power-law weights)
# ------------------------------------------------------------------ #
print("[4/5] build_spectrum_response: spectrum-weighted ASADs")
spec_config = ResponseConfig(
    energies=energies, delta=DELTA, lambda_=LAMBDA, threshold=GATE,
    pol_angles=range(0, 181, 15), mode="spectrum",
    spectrum_weights=lambda e_kev: e_kev ** -2.1, n_jobs=1,
)
spectrum = build_spectrum_response(spec_config, adapter)
first = next(iter(spectrum.polarized.values()))
save_csv("spectrum_phi0.csv", first.counts, first.errors)

# ------------------------------------------------------------------ #
# Mode 2: the response from the sampled polarized ASADs at 0 and 45 deg
# ------------------------------------------------------------------ #
print("[5/5] mode 2: full grid from the polarized ASADs at 0 and 45 deg")
basis_angles = (0.0, 45.0)
basis_adapter = ArrayAdapter(
    unpol, polarized={k: v for k, v in polar.items() if k[1] in basis_angles},
)
basis_config = ResponseConfig(
    energies=energies, delta=DELTA, lambda_=LAMBDA, threshold=GATE,
    pol_angles=range(0, 181, 15), mode="polarized", basis_angles=basis_angles, n_jobs=1,
)
mode2 = build_basis_response(basis_config, basis_adapter)
print(f"  generated {len(mode2.polarized)} ASADs from "
      f"{len(basis_angles)} polarized angles per energy")

# ------------------------------------------------------------------ #
# Validation: both modes against the sampled polarized ASADs at the
# held-out angles 90 and 135 degrees, which entered neither construction
# ------------------------------------------------------------------ #
print("Validation against the held-out sampled ASADs (reduced chi2, mode 1 | mode 2)")
for e in energies:
    line = f"  E = {e:5.0f} keV:"
    for a in (90.0, 135.0):
        ref, _, ref_err = polar[(e, a)]
        p1 = response.polarized[(e, a)]
        p2 = mode2.polarized[(e, a)]
        c1, _, _ = calculate_reduced_chi_squared(ref, ref_err, p1.counts, p1.errors, norm=NORM)
        c2, _, _ = calculate_reduced_chi_squared(ref, ref_err, p2.counts, p2.errors, norm=NORM)
        line += f"  phi0 = {a:3.0f} deg: {c1:.2f} | {c2:.2f}"
    print(line)

print(f"Done. Outputs are in ./{OUT}/")
