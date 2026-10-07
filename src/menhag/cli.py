# MENHAG, Mohamed Gamal Zaid, University of New Hampshire. MIT license, see LICENSE.
"""Command line interface.

Two subcommands::

    menhag build --config config.yaml [--basis-angles 0 45 ...]
                 [--prsp-layout polpy-1.2.1|polarpy]
        Generate a response from a YAML configuration (see
        examples/example_config.yaml). The io section of the YAML selects
        the adapter and paths; the rest maps onto
        :class:`menhag.config.ResponseConfig`. A ``gate_json`` key at the
        top level loads the threshold from a gate description file
        instead of an inline threshold block. ``--basis-angles`` overrides
        the mode 2 input angles of the configuration (any number of
        angles, two or more). The .prsp export: with ``format: prsp`` the
        input and output are .prsp files; with ``format: root`` the io keys
        ``prsp_output`` and ``rmfarf_path`` export the written ROOT
        response to a .prsp as well. The io key ``prsp_layout`` and the
        flag ``--prsp-layout`` select the angle order written (see
        :mod:`menhag.adapters.prsp`); the default is the current polpy
        layout.

    menhag ideal --energy 200 --delta 30 --phi0 45 --mu 25 --sigma 75
        Print (or save as CSV) the ideal-detector curves.
"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import ResponseConfig
from .exceptions import ConfigError, MenhagError

logger = logging.getLogger("menhag")

_FORMATS = ("root", "fits", "csv", "prsp")
_IO_KEYS = ("format", "input_path", "output_path", "key_template", "layout",
            "prsp_layout", "prsp_output", "rmfarf_path", "norm_factor")


def _make_adapter(io: dict, prsp_layout: str):
    """Build the adapter selected by the YAML io section."""
    unknown = set(io) - set(_IO_KEYS)
    if unknown:
        raise ConfigError(
            f"Unknown io keys: {sorted(unknown)}. Known keys: {sorted(_IO_KEYS)}."
        )
    fmt = str(io.get("format", "root")).lower()
    input_path = io.get("input_path")
    output_path = io.get("output_path")
    if input_path is None:
        raise ConfigError("The io section needs input_path.")
    if fmt == "root":
        from .adapters import PolarRootAdapter

        return PolarRootAdapter(
            input_path,
            output_path,
            key_template=io.get("key_template", "sim_{energy:03d}_{pol:03d}_{angle:03d}"),
            norm_factor=io.get("norm_factor"),
        )
    if fmt == "fits":
        from .adapters import FitsAdapter

        return FitsAdapter(input_path, output_path, layout=io.get("layout"),
                           norm_factor=io.get("norm_factor"))
    if fmt == "csv":
        from .adapters import CsvAdapter

        return CsvAdapter(input_path, output_path)
    if fmt == "prsp":
        from .adapters import PrspAdapter

        return PrspAdapter(input_path, output_path, layout=io.get("layout"),
                           output_layout=prsp_layout,
                           norm_factor=io.get("norm_factor"))
    raise ConfigError(
        f"Unknown io format '{fmt}'; supported formats: {_FORMATS}. For any "
        "other format, copy menhag/adapters/template.py and drive the build "
        "from Python."
    )


def _cmd_build(args: argparse.Namespace) -> int:
    try:
        import yaml
    except ImportError as exc:
        raise ConfigError(
            "The build subcommand reads YAML configurations and needs the "
            "optional dependency 'pyyaml'. Install it with: "
            "pip install menhag[yaml]"
        ) from exc

    with open(args.config, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ConfigError(f"{args.config} did not parse to a mapping.")
    io = raw.pop("io", {})
    gate_json = raw.pop("gate_json", None)
    if gate_json is not None:
        if "threshold" in raw:
            raise ConfigError(
                "Give either an inline threshold block or gate_json, not both."
            )
        from .gates import load_gate_json

        gate, meta = load_gate_json(gate_json)
        raw["threshold"] = gate
        logger.info("Loaded gate %s from %s.", meta.get("tag", gate), gate_json)
    if args.basis_angles is not None:
        raw["basis_angles"] = [float(a) for a in args.basis_angles]
    config = ResponseConfig.from_dict(raw)
    from .adapters import DEFAULT_LAYOUT, check_layout

    prsp_layout = args.prsp_layout or io.get("prsp_layout") or DEFAULT_LAYOUT
    try:
        check_layout(prsp_layout)
    except MenhagError as exc:
        raise ConfigError(str(exc)) from None
    adapter = _make_adapter(io, prsp_layout)

    if config.mode == "full":
        from .engine import build_response

        response = build_response(config, adapter, adapter)
    elif config.mode == "spectrum":
        from .engine import build_spectrum_response

        response = build_spectrum_response(config, adapter, adapter)
    else:  # polarized (mode 2)
        from .basis import build_basis_response

        response = build_basis_response(config, adapter, adapter)
    logger.info(
        "Done: %d polarized ASADs generated.", len(response.polarized)
    )
    prsp_output = io.get("prsp_output")
    if prsp_output is not None:
        fmt = str(io.get("format", "root")).lower()
        rmfarf = io.get("rmfarf_path")
        if fmt != "root" or rmfarf is None or io.get("output_path") is None:
            raise ConfigError(
                "prsp_output needs format: root with an output_path (the ROOT "
                "response the export reads) and rmfarf_path (the ROOT file "
                "with the ER_low and ER_high energy edges)."
            )
        from .adapters import convert_prsp

        convert_prsp(io["output_path"], rmfarf, prsp_output, layout=prsp_layout)
        logger.info("Exported %s (%s layout).", prsp_output, prsp_layout)
    return 0


def _cmd_ideal(args: argparse.Namespace) -> int:
    from .engine import build_ideal

    threshold = None
    if args.mu is not None:
        threshold = {"type": "erf", "mu": args.mu, "sigma": args.sigma or 0.0}
    ideal = build_ideal(
        args.energy,
        args.delta,
        args.lam,
        args.phi0,
        threshold,
        n_bins=args.bins,
        pol_fraction=args.pol_fraction,
    )
    header = "eta_deg,phi_pol,phi_unp,corrected"
    lines = [header] + [
        f"{eta:.3f},{p:.8e},{u:.8e},{c:.8e}"
        for eta, p, u, c in zip(
            ideal["eta_deg"], ideal["phi_pol"], ideal["phi_unp"], ideal["corrected"]
        )
    ]
    text = "\n".join(lines)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        logger.info("Wrote %s (mu = %.4f).", args.output, ideal["mu"])
    else:
        print(text)
        print(f"# mu = {ideal['mu']:.4f}")
    return 0


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        prog="menhag",
        description=(
            "MENHAG: Modulation ENgine for Harmonic ASAD Generation. "
            "Polarization responses of Compton polarimeters from a small "
            "set of simulated inputs."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="generate a response from a YAML config")
    p_build.add_argument("--config", "-c", required=True, help="YAML config file")
    p_build.add_argument(
        "--basis-angles", nargs="+", type=float, default=None,
        dest="basis_angles",
        help=("mode 2 input polarization angles in degrees, two or more "
              "(overrides basis_angles in the configuration; default 0 45)"),
    )
    p_build.add_argument(
        "--prsp-layout", default=None, dest="prsp_layout",
        help=("angle order of a .prsp written by this build: polpy-1.2.1 "
              "(the current polpy release, the default) or polarpy (the "
              "earlier plug-in); overrides prsp_layout in the io section"),
    )
    p_build.set_defaults(func=_cmd_build)

    p_ideal = sub.add_parser("ideal", help="ideal-detector curves")
    p_ideal.add_argument("--energy", "-E", type=float, required=True, help="keV")
    p_ideal.add_argument("--delta", type=float, default=0.0, help="zenith [deg]")
    p_ideal.add_argument("--lam", type=float, default=0.0, help="azimuth [deg]")
    p_ideal.add_argument("--phi0", type=float, default=0.0,
                         help="polarization angle [deg]")
    p_ideal.add_argument("--mu", type=float, default=None,
                         help="erf gate threshold mu [keV] (omit for no gate)")
    p_ideal.add_argument("--sigma", type=float, default=None,
                         help="erf gate turn-on width sigma [keV]")
    p_ideal.add_argument("--bins", type=int, default=72, help="azimuthal bins")
    p_ideal.add_argument("--pol-fraction", type=float, default=1.0,
                         dest="pol_fraction", help="polarization fraction")
    p_ideal.add_argument("--output", "-o", default=None, help="CSV output path")
    p_ideal.set_defaults(func=_cmd_ideal)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except MenhagError as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
