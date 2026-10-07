# MENHAG, Mohamed Gamal Zaid, University of New Hampshire. MIT license, see LICENSE.
"""Write the synthetic sample data of the package.

The sample ASADs are made by the package itself, from the ideal-detector
builder, so that the quickstart and the tests run after ``pip install``
with no external file. They are the ASADs of an ideal detector (no
instrumental imprint) at one geometry, 30 degrees off the detector axis
at azimuth 0, with the error-function threshold gate of the accompanying
paper's POLAR example (mu = 25 keV, sigma = 75 keV), at three energies:
the unpolarized ASAD and the fully polarized ASADs at 0, 45, 90 and 135
degrees, each scaled to 200,000 counts over 60 bins of 6 degrees and
Poisson-sampled with a fixed seed. The gate is an instrument-like example,
never a default: the two numbers are specific to POLAR.

The file is ``sample_data/ideal_detector_asads.csv`` in the CSV schema of
:mod:`menhag.adapters.csv_io` (columns energy, bin_low, bin_high, counts,
error, pol, angle), so the CSV adapter reads it directly. Run from the
examples directory:

    python make_sample_data.py

The output is deterministic: the test suite regenerates it and compares
byte for byte.
"""

from __future__ import annotations

import os
import sys

import numpy as np

from menhag import build_ideal

DELTA, LAMBDA = 30.0, 0.0
GATE = {"type": "erf", "mu": 25.0, "sigma": 75.0}
ENERGIES = (100.0, 200.0, 300.0)
ANGLES = (0.0, 45.0, 90.0, 135.0)
N_BINS = 60
TOTAL = 200000.0
SEED = 20260922
HEADER = "energy,bin_low,bin_high,counts,error,pol,angle"


def make_rows(rng: np.random.Generator) -> list[str]:
    rows = []
    for energy in ENERGIES:
        ideal = build_ideal(energy, DELTA, LAMBDA, phi0=0.0, threshold=GATE, n_bins=N_BINS)
        edges = ideal["bin_edges_deg"]
        curves = [(0, 0.0, ideal["phi_unp"])]
        for angle in ANGLES:
            pol = build_ideal(energy, DELTA, LAMBDA, phi0=angle, threshold=GATE, n_bins=N_BINS)
            curves.append((100, angle, pol["phi_pol"]))
        for pol_code, angle, curve in curves:
            expected = TOTAL * curve / curve.sum()
            counts = rng.poisson(expected).astype(float)
            errors = np.sqrt(counts)
            for lo, hi, n, s in zip(edges[:-1], edges[1:], counts, errors):
                rows.append(f"{energy:.0f},{lo:.1f},{hi:.1f},{n:.0f},{s:.6f},{pol_code},{angle:.0f}")
    return rows


def write(path: str) -> str:
    rng = np.random.default_rng(SEED)
    rows = make_rows(rng)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="ascii", newline="\n") as handle:
        handle.write(HEADER + "\n" + "\n".join(rows) + "\n")
    return path


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "sample_data", "ideal_detector_asads.csv")
    print("wrote", write(out))
