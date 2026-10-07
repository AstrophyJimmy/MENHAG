# Sample data

`ideal_detector_asads.csv` is made by the package itself with `../make_sample_data.py`. It holds the ASADs of an ideal detector (no instrumental imprint) at one geometry, 30 degrees off the detector axis at azimuth 0, with the error-function threshold gate of the accompanying paper's POLAR example (mu = 25 keV, sigma = 75 keV, an instrument-specific example), at 100, 200 and 300 keV: the unpolarized ASAD and the fully polarized ASADs at 0, 45, 90 and 135 degrees, each scaled to 200,000 counts over 60 bins of 6 degrees and Poisson-sampled with the seed 20260922.

The columns follow the CSV schema of `menhag.adapters.csv_io`: `energy` (keV), `bin_low` and `bin_high` (degrees of the azimuthal scattering angle), `counts`, `error` (the square root of the counts), `pol` (0 for the unpolarized rows, 100 for the fully polarized ones) and `angle` (the polarization angle in degrees). `CsvAdapter("sample_data/ideal_detector_asads.csv")` reads it directly, and `quickstart.py` reads it with NumPy.

The file is deterministic; the test suite regenerates it and compares byte for byte.
