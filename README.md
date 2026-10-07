# MENHAG

Modulation ENgine for Harmonic ASAD Generation.

MENHAG builds the polarization response of a wide field-of-view Compton polarimeter from a small set of simulated inputs. An ASAD, the azimuthal scattering angle distribution, is the histogram of the azimuthal direction into which photons Compton scatter in the detector; a polarized beam modulates it and an unpolarized beam does not. The polarization response of an instrument is the set of ASADs over photon energy and polarization angle, and the usual way to produce it is a large grid of Monte Carlo simulations, one per energy and polarization angle. MENHAG replaces that grid by one of two constructions, both derived and validated in the accompanying paper (see How to cite), for instruments such as POLAR, POLAR-2 and LEAP. The package is the code behind the paper's results, released so that anyone can generate responses for their own instrument. It is written by Mohamed Gamal Zaid (University of New Hampshire); the POLAR data and responses it was developed and validated against were provided by Hancheng Li, and the simulations and the guidance that shaped the two response modes came from Merlin Kole.

## The two response modes

Mode 1, the unpolarized-only mode, needs one unpolarized simulation per energy and nothing else. A measured ASAD factorizes as M(eta) = I(eta) Phi(eta), where I(eta) collects the instrumental and geometric effects and Phi is the physics term: the Klein-Nishina cross-section transformed into the detector frame for the source incidence (delta, lambda) and the polarization angle phi0, integrated over the polar scattering angle with the detection-threshold gate folded in. Assuming the instrumental effects do not depend on the polarization state of the beam, the polarized ASAD follows from the unpolarized one through the kernel

```
M_pol(eta_i) = M_unp(eta_i) * [F_100(eta_i) / F_U(eta_i)] * C * N
```

with F_100 and F_U the gated physics curves of the fully polarized and the unpolarized beam (normalized so that mean(F) times 2 pi is 1 on the uniform full-circle eta grid), C the analytic normalization between them and N a factor that preserves the total counts of the input. Per-bin errors are propagated with the same multipliers; when the input carries no error bars, Poisson errors are built from the count normalization f of the input histograms as sigma = sqrt(n f) / f. Outputs always inherit the binning of the input.

Mode 2, the two-angle mode, needs in addition fully polarized simulations at two polarization angles per energy, 0 and 45 degrees by default. Per azimuthal bin the fully polarized ASAD as a function of the polarization angle is a constant plus one sinusoid in twice the angle; the unpolarized simulation supplies the constant and the two polarized simulations supply the two sinusoid coefficients, so the ASAD at any polarization angle and any polarization fraction follows. Because the polarized inputs are simulated rather than predicted, this mode carries the polarization-dependent instrumental effects that mode 1 assumes away. Any number of input angles from two upward is accepted; with more than two the coefficients are the least-squares solution per bin and the fit residual is reported. Only the ASADs at the input angles are ever ingested; a blind test verifies that removing every other simulated ASAD from the inputs leaves the result unchanged, bit for bit.

## Required inputs

Both modes need, for every energy of the response, the unpolarized ASAD of the instrument (measured or simulated), the source incidence angles delta (zenith, from the detector normal) and lambda (azimuth around it) in degrees, and the threshold gate parameters. Mode 2 additionally needs the fully polarized ASADs at the chosen input angles for every energy. Geometry is inherited, not an input: no pixel pitch or grid size is ever accepted, and everything the detector does enters through the supplied ASADs. MENHAG never fabricates an unpolarized ASAD from a spectrum alone; a missing energy is an error, never an interpolation.

## The threshold gate

A polarimetry event needs both energy deposits, the recoil electron's and the scattered photon's, above the trigger threshold. Inside the physics integrals this is a gate on the smaller of the two deposits. The default model is the error-function gate

```python
threshold = {"type": "erf", "mu": 25.0, "sigma": 75.0}
```

where mu is the average detection threshold of the instrument in keV and sigma the effective width of the turn-on in keV. It is the continuum limit of a Gaussian mixture of sharp gates and is continuous in photon energy. Both numbers are instrument-specific inputs, never defaults. Use the average detection threshold of your instrument for mu. For sigma use a width representative of everything that smears the turn-on: for a clean instrument that is the energy resolution near the threshold, the Gaussian broadening of a spectral peak there; for an instrument such as POLAR, where optical crosstalk between neighbouring channels, the channel-to-channel gain spread and the event selection all widen the effective turn-on, sigma is an effective parameter and is fitted. The values above are the POLAR values of the paper (mu from the in-orbit threshold calibration of Li et al. 2018, sigma fitted); they ship in `examples/polar_gate.json` with their basis spelled out, and a gate file in that schema loads with

```python
from menhag import load_gate_json
gate, meta = load_gate_json("examples/polar_gate.json")
```

Also available: `{"type": "sharp", "value": 50.0}` for a step threshold; `{"type": "gaussian", "mean": 25.0, "sigma": 75.0, "points": 11}` for the sampled Gaussian mixture of sharp gates (each sampled sharp gate evaluated and combined with its Gaussian weight; `eth_grid` clamps the samples to a grid); `{"type": "erf2", ...}` for independent thresholds on the two deposits; and `EnergyDependentErfGate` for instruments whose effective gate varies with energy.

## Install

The package runs on Python 3.10 or later (tested on 3.10 to 3.13). There are two ways in.

The editable install from the tree: clone the repository and, from the package directory, create the shipped environment (Python 3.11, every optional dependency, and the tools that run the notebooks); creating it installs the package itself, editable, from that directory, so the code you read is the code that runs:

```
git clone https://github.com/AstrophyJimmy/menhag.git
cd menhag
conda env create -f environment.yml
conda activate menhag
```

`environment-fit.yml` is the optional variant for fitting(getting polarization measurements with the created responses): it adds threeML and polpy (the plug-in that reads the `.prsp` responses this package writes) to the same stack, on Python 3.12, the floor polpy sets, and installs the package the same way.

The install from the release tag, into an environment of your own, without the tree:

```
pip install "git+https://github.com/AstrophyJimmy/menhag.git@v1.0.0"
```

installs the core (NumPy and SciPy, arrays in and out); the optional extras name the dependencies of each path and are chosen as `pip install "menhag[root] @ git+https://github.com/AstrophyJimmy/menhag.git@v1.0.0"`: `root` for ROOT I/O (uproot, hist), the POLAR-style format; `fits` for FITS I/O and the `.prsp` response layout (astropy); `csv` for CSV I/O (pandas); `yaml` for YAML configurations and the `menhag build` command; `parallel` for parallel response builds (joblib); `viz` for the validation plots (matplotlib); `dev` for everything plus pytest.

## Quick start

The quickstart runs every entry point on sample data the package made itself, so it needs no external file: `examples/sample_data/ideal_detector_asads.csv` holds the ASADs of an ideal detector 30 degrees off axis with the POLAR example gate at three energies, the unpolarized ASAD and fully polarized ASADs at four angles, Poisson-sampled with a fixed seed (`examples/make_sample_data.py` writes it; `examples/sample_data/README.md` describes it). From the `examples` directory,

```
python quickstart.py
```

prints

```
sample data: 3 energies, 60 bins, polarized angles [0.0, 45.0, 90.0, 135.0]
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
```

The reduced chi-squared values sit near one because the sample detector is ideal and both constructions are exact for it up to the Poisson noise of the sampled ASADs; the numbers are deterministic. The same entry points, in code:

One unpolarized ASAD in, one polarized ASAD out (mode 1, NumPy arrays, no files):

```python
import numpy as np
from menhag import build_single

counts = ...                            # your unpolarized ASAD, shape (60,)
edges = np.linspace(0.0, 360.0, 61)     # bin edges in degrees

pol_counts, pol_errors = build_single(
    counts, edges,
    energy=200.0,                       # keV
    delta=27.3, lambda_=6.1,            # source position [deg]
    phi0=45.0,                          # polarization angle [deg]
    threshold={"type": "erf", "mu": 25.0, "sigma": 75.0},
)
```

Ideal polarimeter curves, no input at all:

```python
from menhag import build_ideal

ideal = build_ideal(200.0, delta=40.0, lambda_=0.0, phi0=30.0,
                    threshold={"type": "erf", "mu": 25.0, "sigma": 75.0})
ideal["phi_pol"], ideal["phi_unp"]      # normalized physics curves
ideal["mu"]                             # modulation of the corrected curve
ideal["harmonics"]                      # relative amplitudes a_k, k = 1..4
```

A full mode 1 response grid from a ROOT file in the POLAR layout:

```python
from menhag import ResponseConfig, build_response
from menhag.adapters import PolarRootAdapter

config = ResponseConfig(
    energies=range(5, 751, 5),          # the POLAR production energy grid
    delta=27.3, lambda_=6.1,
    threshold={"type": "erf", "mu": 25.0, "sigma": 75.0},
    pol_step=3,                         # 61 angles, 0 to 180 inclusive
    pol_fraction=1.0,                   # polarization degree, adjustable
)
adapter = PolarRootAdapter("unpolarized_sims.root", "analytic_response.root")
response = build_response(config, adapter, adapter)
```

The polarization-angle grid is endpoint-inclusive over [0, 180]: `pol_step=3` gives the 61 angles 0, 3, ..., 180 of the POLAR responses (the 180 degree entry repeats the configuration of 0 degrees and is kept because the threeML and polpy readers expect a closed interval; `include_endpoint=False` drops it), `pol_step=1` gives 181, and sub-degree steps are accepted. Explicit angle sets go through `pol_angles`. Input histograms are read by the key `sim_{energy:03d}_000_000` and the output grid is written as `sim_{energy:03d}_100_{angle:03d}` (energy in keV, polarization degree in percent, phi0 in degrees); pass `key_template=...` to match your own naming. For erf gates full grids run through a one-pass evaluator with a shared geometry cache.

Mode 2 from the unpolarized ASAD and two polarized ASADs per energy:

```python
from menhag import PolarizedBasis, reconstruct, build_basis_response

basis = PolarizedBasis(unpol_counts, {0.0: pol_at_0, 45.0: pol_at_45})
prediction = reconstruct(basis, 72.0)   # the ASAD at any polarization angle
```

or, over a whole energy grid through any adapter that can read polarized inputs, `build_basis_response(config, adapter, adapter)` with `mode="polarized"` in the configuration. The input angles are the configuration key `basis_angles` (default `(0, 45)`; any number of angles from two upward, combined by least squares when more than two) and the command-line flag `--basis-angles`.

Spectrum-integrated mode 1 response:

```python
from menhag import build_spectrum_response

response = build_spectrum_response(config, adapter,
                                   weights=lambda e_kev: e_kev ** -1.5)
```

## Formats

ROOT (the POLAR-style flat histogram layout, the default), FITS (one binary table per ASAD with the geometry and gate in the header) and CSV (a documented long-format schema) adapters ship with the package; each imports its dependency lazily. The `.prsp` polarization response read by the polpy plug-in for threeML is both an input and an output: `PrspAdapter` reads one (and `FitsAdapter` detects one), `menhag.adapters.convert_prsp` writes one from a ROOT response, and `menhag build` exports one. Two plug-ins have read that layout and they disagree on which polarized entry belongs to which angle, so the package names the angle order by the plug-in that reads it: `polpy-1.2.1`, the order of the current polpy release and the default, and `polarpy`, the order of the earlier plug-in behind the fits of the accompanying paper. Every writer records its order in the file's header and every reader honours it. The layouts and the option are documented in `docs/formats.md`.

## A minimal configuration

The command line consumes a YAML file naming the energy grid, the geometry, the gate, the polarization-angle grid, the mode and the file paths:

```
energies: [100.0, 200.0, 300.0]
delta: 30.0
lambda: 0.0
threshold: {type: erf, mu: 25.0, sigma: 75.0}
pol_step: 3
pol_fraction: 1.0
mode: full
io:
  format: root
  input_path: data/unpolarized_sims.root
  output_path: outputs/polarized_response.root
  key_template: "sim_{energy:03d}_{pol:03d}_{angle:03d}"
```

Then `menhag build --config config.yaml` produces the response; `mode: polarized` selects mode 2 with `basis_angles: [0, 45]` (or any list of two or more angles on `--basis-angles`, combined by least squares beyond two). The io keys `prsp_output` and `rmfarf_path` export the ROOT response to a `.prsp` as well, in the order named by `prsp_layout` or the flag `--prsp-layout`. `menhag ideal -E 200 --delta 30 --phi0 45 --mu 25 --sigma 75 -o ideal.csv` writes the ideal curves. The full annotated example is `examples/example_config.yaml`.

## Worked notebooks, in reading order

1. `examples/01_why_this_model.ipynb`: what mode 1 does and why it works, one worked example built up cell by cell on the GRB 170114A response of POLAR.
2. `examples/02_sample_data_and_first_run.ipynb`: the core algorithm on the shipped sample data, which needs no external file, and then on a POLAR file with a few lines.
3. `examples/03_full_demo_170114A.ipynb`: the production-scale demonstration on GRB 170114A, both modes over the full energy grid, compared against the GEANT4 simulations.

The notebooks ship executed, with their stored outputs, and their figures follow the palette and style of the paper's figures (`menhag.palette`). The POLAR file they read is a POLAR collaboration product not distributed with the package; a user who holds a copy places it in `examples/data/`, which is untracked.

## Validation

Given a reference polarized response (for example a full GEANT4 simulation), `menhag.validation` computes the reduced chi-squared of the paper between prediction and reference, with the variance of the reference taken from the model expectation, and provides overlay plots and an energy by angle chi-squared map in the palette roles of the paper:

```python
from menhag.validation import compare_asads, plot_overlay
chi2, chi2_err, n_bins = compare_asads(reference_asad, predicted_asad, norm=f)
```

## Adapting a new instrument

Any file structure can be adapted by copying `src/menhag/adapters/template.py` and editing its two marked functions: `load_unpolarized(energy)` returns the unpolarized ASAD (counts, bin edges in degrees, optional errors) and `save_response(response)` writes the generated set; implementing `load_polarized(energy, phi0)` the same way enables mode 2. The physics never changes. The walkthrough is `docs/adapting.md`.

## Testing

The suite is 32 tests in five files under `tests/`, each guarding a claim of the paper or a format promise of the package, with a docstring saying what would break if it failed: `test_physics.py` the identities and invariances of the transformed cross-section, `test_reference.py` the off-axis curves against an independent adaptive-quadrature evaluation at 1e-6, `test_modes.py` the two constructions, the error propagation and the reduced chi-squared of the paper, `test_formats.py` the two `.prsp` layouts and the adapters, `test_quickstart.py` the quickstart against its recorded output and both modes against the held-out sample ASADs. In the shipped environment, from the package directory,

```
python -m pytest -q
```

runs it in under a minute; one test reads a written `.prsp` back through polpy's own reader and skips when polpy is absent.

## How to cite

to cite the paper, the primary reference for the method:

> Zaid, M. G., Kole, M., McConnell, M., Muleri, F., and Li, H., "A Generalized Framework for Rapid Polarization Response Generation in Wide Field-of-View Compton Polarimeters", The Astrophysical Journal, 2026, submitted.

and the archived software release, MENHAG version 1.0.0, DOI 10.5281/zenodo.23204599. The source is hosted at https://github.com/AstrophyJimmy/menhag; GitHub's "Cite this repository" button uses the `CITATION.cff` file.

## License

MIT; see [LICENSE](LICENSE).
