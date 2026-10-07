# MENHAG, Mohamed Gamal Zaid, University of New Hampshire. MIT license, see LICENSE.
"""Instrument I/O adapters.

- :class:`~menhag.adapters.base.InstrumentAdapter`: the interface.
- :class:`~menhag.adapters.base.ArrayAdapter`: in-memory passthrough.
- :class:`~menhag.adapters.polar_root.PolarRootAdapter`: the POLAR-style
  ROOT layout (configurable key template), the default.
- :class:`~menhag.adapters.fits_io.FitsAdapter`: the package's FITS table
  layout; a .prsp given as input is detected and read through the prsp
  adapter.
- :class:`~menhag.adapters.csv_io.CsvAdapter`: the documented CSV schema.
- :class:`~menhag.adapters.prsp.PrspAdapter` and
  :func:`~menhag.adapters.prsp.convert_prsp`: the .prsp polarization
  response layout of polpy, read and written in either of its two angle
  orders (``"polpy-1.2.1"``, the default, and ``"polarpy"``).
- :class:`~menhag.adapters.template.TemplateAdapter`: copy-and-edit stub
  for any other format.

The ROOT, FITS, CSV and prsp adapters import their dependencies lazily,
so the core install never blocks the other paths.
"""

from .base import ArrayAdapter, InstrumentAdapter
from .csv_io import CsvAdapter
from .fits_io import FitsAdapter, read_asad_fits, write_response_fits
from .polar_root import PolarRootAdapter, read_asad_root, write_response_root
from .prsp import (
    DEFAULT_LAYOUT,
    LAYOUTS,
    PrspAdapter,
    check_layout,
    convert_prsp,
    read_prsp,
    write_prsp,
    write_response_prsp,
)
from .template import TemplateAdapter

__all__ = [
    "InstrumentAdapter",
    "ArrayAdapter",
    "PolarRootAdapter",
    "read_asad_root",
    "write_response_root",
    "FitsAdapter",
    "read_asad_fits",
    "write_response_fits",
    "CsvAdapter",
    "PrspAdapter",
    "convert_prsp",
    "read_prsp",
    "write_prsp",
    "write_response_prsp",
    "check_layout",
    "LAYOUTS",
    "DEFAULT_LAYOUT",
    "TemplateAdapter",
]
