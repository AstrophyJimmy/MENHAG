# Adapting MENHAG to another instrument

MENHAG reads and writes through an adapter, and the physics never sees a file. To use your own file structure, copy `src/menhag/adapters/template.py` next to your analysis code and fill in the two blocks marked `EDIT HERE`.

`load_unpolarized(energy)` returns a `menhag.asad.ASAD` for that energy: `counts` (a one-dimensional array of bin contents), `bin_edges` (bin edges of the measured azimuthal scattering angle eta in degrees, one more entry than counts) and `errors` (per-bin uncertainties, or `None` for the Poisson convention sigma = sqrt(n f) / f with the run's `norm_factor`). The generated polarized ASADs inherit exactly this binning. Raise `MissingEnergyError` for an energy you cannot supply; MENHAG never interpolates one.

`save_response(response)` receives a `menhag.asad.ResponseSet`: `response.unpolarized` is `{energy_kev: ASAD}`, `response.polarized` is `{(energy_kev, phi0_deg): ASAD}` and `response.pol_fraction` the polarization fraction. Write them in whatever layout your fitting code reads.

For mode 2, implement `load_polarized(energy, phi0)` the same way as `load_unpolarized`, returning the fully polarized ASAD at that polarization angle; the build then reads the ASADs at the configured `basis_angles` and nothing else.

Run the build exactly as with the shipped adapters:

```python
from my_adapter import MyInstrumentAdapter
from menhag import ResponseConfig, build_response

adapter = MyInstrumentAdapter("my_inputs/", "my_response.npz")
build_response(config, adapter, adapter)
```

The gate parameters mu and sigma of the configuration are instrument-specific: measure the average detection threshold of your instrument for mu, and for sigma use the energy resolution near threshold if the turn-on is clean, or fit an effective width against an unpolarized simulation otherwise (see the README).
