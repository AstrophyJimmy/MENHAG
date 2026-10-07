# MENHAG, Mohamed Gamal Zaid, University of New Hampshire. MIT license, see LICENSE.
"""Exception types raised by menhag.

All package errors derive from :class:`MenhagError` so callers can catch a
single base class. Each message is written to tell the user what to fix, not
just what went wrong.
"""

from __future__ import annotations


class MenhagError(Exception):
    """Base class for all menhag errors."""


class ThresholdError(MenhagError):
    """Invalid threshold gate specification."""


class EmptyComptonWindowError(MenhagError):
    """The Compton coincidence window is closed for the requested settings.

    Below a threshold-dependent crossover energy no scattering angle lets
    both the recoil electron and the scattered photon deposit energy above
    the trigger threshold, so the gated Klein-Nishina integrals vanish.
    Lower the threshold or raise the photon energy.
    """


class MissingEnergyError(MenhagError):
    """A requested unpolarized ASAD is not available from the adapter.

    menhag never fabricates an unpolarized ASAD from a spectrum alone, so a
    missing energy is always an error rather than a silent interpolation.
    """


class BasisError(MenhagError):
    """The mode 2 inputs cannot be solved into a response (missing, degenerate or mismatched polarized ASADs)."""


class AdapterError(MenhagError):
    """File I/O adapter failure (missing file, missing key, bad schema)."""


class ConfigError(MenhagError):
    """Invalid :class:`menhag.config.ResponseConfig` contents."""
