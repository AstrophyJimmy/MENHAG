"""The format promises: the .prsp polarization response in its two angle
orders, the FITS, ROOT and CSV adapters, and the build command's export.

Simulation order means: the ASAD at angle a is stored under a. In the
polpy-1.2.1 layout file entry j holds the ASAD at PA_IN[j]; in the
polarpy layout it holds the ASAD at 180 - PA_IN[j].
"""

import os

import numpy as np
import pytest

from menhag import ASAD, ResponseSet

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(os.path.dirname(HERE), "examples", "sample_data", "ideal_detector_asads.csv")
PA = np.linspace(0.0, 180.0, 61)
N_SA = 24
SA_EDGES = np.linspace(0.0, 360.0, N_SA + 1)
ETA = 0.5 * (SA_EDGES[:-1] + SA_EDGES[1:])
E_LO = np.array([97.5, 197.5])
E_HI = np.array([102.5, 202.5])
EDGES60 = np.linspace(0.0, 360.0, 61)


def sim_cube():
    """pol[i, k, :] is an exact second-harmonic ASAD whose phase encodes the angle PA[k]; unp[i, :] a flat ASAD with the energy index as its level."""
    pol = np.zeros((E_LO.size, PA.size, N_SA))
    unp = np.zeros((E_LO.size, N_SA))
    for i in range(E_LO.size):
        unp[i] = 10.0 + i
        for k, pa in enumerate(PA):
            pol[i, k] = 10.0 + i + 3.0 * np.cos(np.radians(2.0 * (ETA - pa)))
    return pol, unp


def make_response():
    rng = np.random.default_rng(3)
    unpol, polarized = {}, {}
    for energy in (100.0, 200.0):
        counts = rng.poisson(1000, size=60).astype(float)
        unpol[energy] = ASAD(counts=counts, bin_edges=EDGES60, errors=np.sqrt(counts), energy=energy)
        for phi0 in (0.0, 45.0):
            p = rng.poisson(1000, size=60).astype(float)
            polarized[(energy, phi0)] = ASAD(counts=p, bin_edges=EDGES60, errors=0.7 * np.sqrt(p), energy=energy)
    return ResponseSet(unpolarized=unpol, polarized=polarized, pol_fraction=1.0,
                       meta={"delta_deg": 26.0, "lambda_deg": 0.0})


def test_prsp_layouts_round_trip_through_read_and_write(tmp_path):
    """If this failed, a .prsp written in one layout, read back through the adapter and written in the other would not hold the same simulation-order cube, the PALAYOUT keyword would be missing or wrong, a keyword-less file would not be read in the layout named, or a response export would lack the 180 degree copy of the 0 degree ASAD."""
    pytest.importorskip("astropy", reason="optional [fits] extra")
    from astropy.io import fits as pyfits

    from menhag.adapters import DEFAULT_LAYOUT, LAYOUTS, PrspAdapter, read_prsp, write_prsp, write_response_prsp
    from menhag.adapters.prsp import LAYOUT_KEYWORD, layout_index_map

    assert DEFAULT_LAYOUT == "polpy-1.2.1" and set(LAYOUTS) == {"polpy-1.2.1", "polarpy"}
    pol, unp = sim_cube()
    for write_layout in LAYOUTS:
        for read_layout in LAYOUTS:
            first = tmp_path / f"{write_layout}_{read_layout}_first.prsp"
            write_prsp(str(first), E_LO, E_HI, PA, pol, unp, sa_edges=SA_EDGES, layout=write_layout)
            with pyfits.open(str(first)) as h:
                assert h["INPAVALS"].header[LAYOUT_KEYWORD] == write_layout
                assert h["SPECRESP POLMATRIX"].header[LAYOUT_KEYWORD] == write_layout
                stored = np.asarray(h["SPECRESP POLMATRIX"].data).transpose()
            src = layout_index_map(PA, write_layout)
            assert np.allclose(stored[:, 10, :], pol[:, src[10], :], atol=1e-5)
            d = read_prsp(str(first))
            assert d["layout"] == write_layout and np.allclose(d["pol"], pol, atol=1e-5) and np.allclose(d["unp"], unp, atol=1e-5)
            reader = PrspAdapter(str(first), output_path=str(tmp_path / f"{write_layout}_{read_layout}_second.prsp"),
                                 output_layout=read_layout)
            assert np.allclose(reader.load_polarized(200.0, 45.0).counts, pol[1, 15], atol=1e-5)
            response = ResponseSet(
                unpolarized={e: reader.load_unpolarized(e) for e in (100.0, 200.0)},
                polarized={(e, a): reader.load_polarized(e, a) for e in (100.0, 200.0) for a in PA},
                pol_fraction=1.0)
            reader.save_response(response)
            d2 = read_prsp(str(tmp_path / f"{write_layout}_{read_layout}_second.prsp"))
            assert d2["layout"] == read_layout and np.allclose(d2["pol"], pol, atol=1e-5)
    # A file without the keyword is read in the layout the caller names.
    bare = tmp_path / "bare.prsp"
    write_prsp(str(bare), E_LO, E_HI, PA, pol, unp, sa_edges=SA_EDGES, layout="polarpy")
    with pyfits.open(str(bare), mode="update") as h:
        del h["INPAVALS"].header[LAYOUT_KEYWORD]
        del h["SPECRESP POLMATRIX"].header[LAYOUT_KEYWORD]
    assert np.allclose(read_prsp(str(bare), layout="polarpy")["pol"], pol, atol=1e-5)
    assert np.allclose(read_prsp(str(bare))["pol"][:, 10, :], pol[:, 50, :], atol=1e-5)
    # A response without a 180 degree entry gets one, the copy of its 0 degree ASAD.
    for layout in LAYOUTS:
        polarized = {(100.0, a): ASAD(counts=np.full(N_SA, 1.0 + a), bin_edges=SA_EDGES) for a in (0.0, 45.0, 90.0, 135.0)}
        response = ResponseSet(unpolarized={100.0: ASAD(counts=np.full(N_SA, 7.0), bin_edges=SA_EDGES)},
                               polarized=polarized, pol_fraction=1.0)
        path = tmp_path / f"dup_{layout}.prsp"
        write_response_prsp(str(path), response, [97.5], [102.5], layout=layout)
        d = read_prsp(str(path))
        assert np.allclose(d["pa_vals"], [0.0, 45.0, 90.0, 135.0, 180.0])
        assert np.allclose(d["pol"][0, -1], d["pol"][0, 0]) and np.allclose(d["pol"][0, 0], 1.0)


def test_prsp_export_from_root_writes_both_layouts_and_the_180_degree_copy(tmp_path):
    """If this failed, the ROOT export would write the wrong angle order for a layout, default to the paper layout instead of the current polpy layout, or write the file's own 180 degree histogram instead of the copy of the 0 degree ASAD."""
    uproot = pytest.importorskip("uproot", reason="optional [root] extra")
    pytest.importorskip("astropy", reason="optional [fits] extra")
    from astropy.io import fits as pyfits

    from menhag.adapters import DEFAULT_LAYOUT, convert_prsp

    n_sa = 360
    edges = np.linspace(0.0, 360.0, n_sa + 1)
    events = tmp_path / "response.root"
    with uproot.recreate(str(events)) as out:
        for e in (100, 200):
            out[f"sim_{e:03d}_000_000"] = (np.full(n_sa, 7.0), edges)
            for angle in range(0, 181, 3):
                # The value encodes the angle plus one, so the 0 degree ASAD
                # (value 1) differs from an empty entry (0); the file's own
                # 180 degree histogram is a decoy (999).
                value = 999.0 if angle == 180 else float(angle) + 1.0
                out[f"sim_{e:03d}_100_{angle:03d}"] = (np.full(n_sa, value), edges)
    rmfarf = tmp_path / "rmfarf.root"
    with uproot.recreate(str(rmfarf)) as out:
        out["ER_low"] = (E_LO, np.array([0.0, 1.0, 2.0]))
        out["ER_high"] = (E_HI, np.array([0.0, 1.0, 2.0]))

    default = tmp_path / "default.prsp"
    convert_prsp(str(events), str(rmfarf), str(default))
    with pyfits.open(str(default)) as h:
        assert h["INPAVALS"].header["PALAYOUT"] == DEFAULT_LAYOUT == "polpy-1.2.1"
        assert np.asarray(h["INPAVALS"].data["PA_IN"]).size == 61
        assert np.allclose(np.asarray(h["SPECRESP UNPOLMATRIX"].data), 7.0)
        stored = np.asarray(h["SPECRESP POLMATRIX"].data).transpose()
    for j in (0, 10, 30, 59):
        assert np.allclose(stored[:, j, :], 3.0 * j + 1.0)
    assert np.allclose(stored[:, 60, :], 1.0)

    paper = tmp_path / "paper.prsp"
    convert_prsp(str(events), str(rmfarf), str(paper), layout="polarpy")
    with pyfits.open(str(paper)) as h:
        assert h["INPAVALS"].header["PALAYOUT"] == "polarpy"
        stored = np.asarray(h["SPECRESP POLMATRIX"].data).transpose()
    for j in (1, 10, 30, 60):
        assert np.allclose(stored[:, j, :], 180.0 - 3.0 * j + 1.0)
    assert np.allclose(stored[:, 0, :], 1.0)


def test_polpy_reads_the_export_back(tmp_path):
    """If this failed, polpy's own reader would not see the simulation at every label in the current layout and the mirror image in the paper layout, so a fit against a MENHAG export would return the wrong polarization angle (skips without polpy)."""
    pytest.importorskip("astropy", reason="optional [fits] extra")
    pytest.importorskip("polpy", reason="polpy is not installed")
    from polpy.polresponse import PolResponse

    from menhag.adapters import write_prsp

    pol, unp = sim_cube()
    for layout, reflected in (("polpy-1.2.1", False), ("polarpy", True)):
        path = tmp_path / f"{layout}.prsp"
        write_prsp(str(path), E_LO, E_HI, PA, pol, unp, sa_edges=SA_EDGES, layout=layout)
        rsp = PolResponse(str(path), pa_offset=0.0)
        # polpy maps the labels through (180 + PA_IN) % 180 and sorts them,
        # so 0 and 180 both sit at label 0 and the rest follow in order.
        assert np.allclose(rsp.pol_ang[2:], PA[1:60])
        for k in (1, 15, 30, 59):
            expect = pol[:, 60 - k, :] if reflected else pol[:, k, :]
            assert np.allclose(rsp.pol_matrix[:, k + 1, :], expect, atol=1e-5)
        model = rsp.evaluate_grid_point(45.0, 100.0)
        target = pol[:, 45, :] if reflected else pol[:, 15, :]
        assert np.allclose(model, target, atol=0.05)


def test_fits_adapter_reads_its_tables_and_detects_a_prsp(tmp_path):
    """If this failed, the FITS table layout would not round-trip with its errors, or a .prsp given as FITS input would not be detected and read in its recorded layout."""
    pytest.importorskip("astropy", reason="optional [fits] extra")
    from menhag.adapters import FitsAdapter, write_prsp, write_response_fits

    response = make_response()
    path = tmp_path / "response.fits"
    write_response_fits(str(path), response)
    adapter = FitsAdapter(str(path))
    unpol = adapter.load_unpolarized(100.0)
    assert np.allclose(unpol.counts, response.unpolarized[100.0].counts)
    assert np.allclose(unpol.errors, response.unpolarized[100.0].errors)
    pol = adapter.load_polarized(200.0, 0.0)
    assert np.allclose(pol.counts, response.polarized[(200.0, 0.0)].counts)
    assert np.allclose(pol.errors, response.polarized[(200.0, 0.0)].errors)

    cube, unp = sim_cube()
    prsp = tmp_path / "detect.prsp"
    write_prsp(str(prsp), E_LO, E_HI, PA, cube, unp, sa_edges=SA_EDGES, layout="polarpy")
    detected = FitsAdapter(str(prsp))
    assert np.allclose(detected.load_polarized(100.0, 30.0).counts, cube[0, 10], atol=1e-5)
    assert np.allclose(detected.load_unpolarized(200.0).counts, unp[1], atol=1e-5)


def test_root_adapter_round_trip_preserves_the_propagated_errors(tmp_path):
    """If this failed, the POLAR-style ROOT layout would not round-trip, or the weighted storage would lose the propagated per-bin errors, so a written response would carry the wrong uncertainties."""
    pytest.importorskip("uproot", reason="optional [root] extra")
    pytest.importorskip("hist", reason="optional [root] extra")
    from menhag.adapters import PolarRootAdapter, write_response_root

    response = make_response()
    path = tmp_path / "response.root"
    write_response_root(str(path), response)
    adapter = PolarRootAdapter(str(path))
    unpol = adapter.load_unpolarized(200.0)
    assert np.allclose(unpol.counts, response.unpolarized[200.0].counts)
    assert np.allclose(unpol.errors, response.unpolarized[200.0].errors)
    assert np.allclose(unpol.bin_edges, EDGES60)
    pol = adapter.load_polarized(100.0, 45.0)
    assert np.allclose(pol.counts, response.polarized[(100.0, 45.0)].counts)
    assert np.allclose(pol.errors, response.polarized[(100.0, 45.0)].errors)


def test_csv_adapter_reads_the_documented_schema():
    """If this failed, the documented CSV schema would not read the shipped sample file, so the CSV path of the package would be broken."""
    pytest.importorskip("pandas", reason="optional [csv] extra")
    from menhag.adapters import CsvAdapter

    adapter = CsvAdapter(SAMPLE)
    unp = adapter.load_unpolarized(100.0)
    assert unp.counts.size == 60 and unp.errors is not None
    pol = adapter.load_polarized(300.0, 45.0)
    assert pol.counts.size == 60 and np.allclose(pol.bin_edges, unp.bin_edges)


def test_build_command_exports_a_prsp_in_the_requested_layout(tmp_path):
    """If this failed, menhag build would not write the ROOT response from a configuration, or its .prsp export would come out in the wrong layout for the flag given, or in the paper layout by default."""
    uproot = pytest.importorskip("uproot", reason="optional [root] extra")
    pytest.importorskip("hist", reason="optional [root] extra")
    pytest.importorskip("astropy", reason="optional [fits] extra")
    yaml = pytest.importorskip("yaml", reason="optional [yaml] extra")
    from astropy.io import fits as pyfits

    from menhag.adapters import DEFAULT_LAYOUT, read_prsp
    from menhag.cli import main

    n_sa = 60
    edges = np.linspace(0.0, 360.0, n_sa + 1)
    events = tmp_path / "unpolarized.root"
    with uproot.recreate(str(events)) as out:
        for e in (100, 200):
            out[f"sim_{e:03d}_000_000"] = (np.full(n_sa, 1000.0), edges)
    rmfarf = tmp_path / "rmfarf.root"
    with uproot.recreate(str(rmfarf)) as out:
        out["ER_low"] = (E_LO, np.array([0.0, 1.0, 2.0]))
        out["ER_high"] = (E_HI, np.array([0.0, 1.0, 2.0]))
    for layout in ("polpy-1.2.1", "polarpy", None):
        out_root = tmp_path / f"response_{layout}.root"
        out_prsp = tmp_path / f"response_{layout}.prsp"
        cfg = {"energies": [100.0, 200.0], "delta": 30.0, "lambda": 0.0,
               "threshold": {"type": "erf", "mu": 25.0, "sigma": 75.0}, "pol_step": 3, "n_jobs": 1,
               "io": {"format": "root", "input_path": str(events), "output_path": str(out_root),
                      "prsp_output": str(out_prsp), "rmfarf_path": str(rmfarf)}}
        cfg_path = tmp_path / f"config_{layout}.yaml"
        with open(cfg_path, "w", encoding="ascii") as handle:
            yaml.safe_dump(cfg, handle)
        argv = ["build", "-c", str(cfg_path)] + ([] if layout is None else ["--prsp-layout", layout])
        assert main(argv) == 0
        assert out_root.exists() and out_prsp.exists()
        expected = layout or DEFAULT_LAYOUT
        with pyfits.open(str(out_prsp)) as h:
            assert h["INPAVALS"].header["PALAYOUT"] == expected
        d = read_prsp(str(out_prsp))
        assert d["layout"] == expected and d["pol"].shape == (2, 61, n_sa)
        assert np.allclose(d["pol"][:, 0, :], d["pol"][:, 60, :]) and np.all(d["unp"] > 0)
