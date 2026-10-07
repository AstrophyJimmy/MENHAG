# Changelog

Versions follow semantic versioning. The version string lives in `menhag.__version__`; the build system reads it from there.

## 1.0.0

The first public release, the version behind the results of the accompanying paper (Zaid et al., The Astrophysical Journal, 2026, submitted). It contains:

- Mode 1, the unpolarized-only mode: the polarized ASAD at any polarization angle and polarization fraction from the unpolarized ASAD alone, through the analytic modulation kernel of the gated, frame-transformed Klein-Nishina cross-section (`menhag.physics`, `menhag.engine`). Full energy by angle grids run through a one-pass evaluator with a shared geometry cache (`menhag.m2`); a spectrum-integrated variant weights the energies before applying the kernel.
- Mode 2, the two-angle mode: the ASAD at any polarization angle from the unpolarized ASAD and fully polarized ASADs at a user-chosen list of angles, two or more, default 0 and 45 degrees, solved directly with two angles and by least squares per bin with more (`menhag.basis`, configuration key `basis_angles`, command-line flag `--basis-angles`). Weights vanish at multiples of 90 degrees (`TRIG_ZERO`), and a blind test and a holdout validation ship with it.
- Threshold gates as instrument-specific inputs with no defaults: the error-function gate (the recommended model), the sharp gate, the two-deposit gate, the energy-dependent gate, and the sampled Gaussian mixture of sharp gates as an option (`menhag.gates`); a JSON gate description file with `examples/polar_gate.json` as the POLAR example.
- Statistical uncertainties of the input histograms propagated to every output bin in both modes, with the Poisson convention for inputs that carry no error bars.
- Adapters for the POLAR-style ROOT histogram layout, FITS tables and a documented CSV schema, an in-memory adapter, a template adapter for any other format adapted by editing two functions, and the `.prsp` polarization-response format read by the polpy plug-in for threeML, read and written in either of its two angle orders, `polpy-1.2.1` (the current release, the default) and `polarpy` (the earlier plug-in behind the fits of the paper), with the order recorded in the file (`menhag.adapters`).
- A YAML configuration file listing the energies, the incidence angles, the gate, the polarization grid, the mode and its input angles, and the file paths, consumed by `menhag build`, which also exports a `.prsp` in either layout; `menhag ideal` for the ideal-detector curves.
- Validation utilities: the reduced chi-squared of the paper between a prediction and a reference, overlay plots and an energy by angle chi-squared map (`menhag.validation`), drawn in the palette roles of the paper's figures (`menhag.palette`).
- The test suite, including an off-axis reference test against an independent adaptive-quadrature evaluation of the transformed cross-section with and without the gate.
- Three executed notebooks in reading order, a quickstart script and its sample data made by the package itself from the ideal-detector builder, a minimal configuration, and two conda environment files, one for production and tests and an optional one that adds threeML and polpy for fitting.
- A citation file naming the paper as the primary reference and the archived release through its DOI.
