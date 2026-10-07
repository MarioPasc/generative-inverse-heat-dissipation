"""Tests of F3 (T8.5): expectations, recomputed I_w and ρ, duplicated runs, byte stability, CLI."""

from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from ihdm.analysis import paper_f3 as f3
from ihdm.analysis.exploratory import regeneration_ratio
from ihdm.analysis.inherited import load_constants, within_seed_share
from ihdm.analysis.paper_f2 import IndexRun, read_index
from ihdm.cli import paper_f3 as cli

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "docs" / "RESULTS"
T75_JSON = RES / "heldout_fidelity" / "heldout_fidelity.json"
A2_JSON = RES / "paper" / "f2_a2_heldout" / "heldout_fidelity.json"
TABLES = RES / "tables" / "tables.json"
EXPLORATORY = RES / "exploratory" / "exploratory.json"
CONSTANTS = RES / "inherited_band_constants.json"
COMMITTED = (T75_JSON, A2_JSON, TABLES, EXPLORATORY, CONSTANTS)
DATA_ROOT = Path(os.environ.get("IHDM_DATA_ROOT",
                                "/media/mpascual/MeningD2/spectral_allocation_heat_diffusion_project"))
REAL_RESULTS = DATA_ROOT / "_results"

needs_committed = pytest.mark.skipif(not all(p.is_file() for p in COMMITTED),
                                     reason="committed result JSONs absent")
needs_real = pytest.mark.skipif(not (REAL_RESULTS / "index.csv").is_file(),
                                reason="IHDM_DATA_ROOT/_results not mounted")

#: The ticket's table, computed by hand from the per-run JSON values (precision F, recall F,
#: precision H, recall H); 4 decimals where the 3-decimal value sits on the ±0.0005 bound.
TICKET_PR = {
    ("ixi", "A0"): (0.655, 0.343, 0.661, 0.2375),
    ("ixi", "A2"): (0.668, 0.395, 0.683, 0.2375),
    ("ixi", "A1"): (0.700, 0.4375, 0.715, 0.094),
    ("ixi", "A3"): (0.757, 0.4625, 0.7585, 0.0825),
    ("oasis1", "A0"): (0.705, 0.456, 0.708, 0.361),
    ("oasis1", "A3"): (0.834, 0.525, 0.842, 0.141),
}
TICKET_WITHIN = {
    ("ixi", "A0"): (0.9966, 0.342, 0.343, 0.665),
    ("ixi", "A2"): (0.9966, 0.347, 0.348, 0.671),
    ("ixi", "A1"): (0.9205, 0.262, 0.285, 0.848),
    ("ixi", "A3"): (0.9205, 0.262, 0.285, 0.885),
    ("oasis1", "A0"): (0.9983, 0.448, 0.449, 0.663),
    ("oasis1", "A3"): (0.9574, 0.323, 0.337, 0.818),
}
SIGMA = {"A0": 96.0, "A2": 96.0, "A1": 24.0, "A3": 24.0}


def _index_from_ids(run_ids: list[str]) -> list[IndexRun]:
    out = []
    for rid in run_ids:
        dataset, arm, seed = rid.rsplit("_", 2)
        out.append(IndexRun(rid, dataset, arm, int(seed[1:]), {}))
    return out


def _mri_ids(heldout: dict) -> list[str]:
    return [r for r in heldout if r.startswith(("ixi_", "oasis1_"))]


# --------------------------------------------------------------------------------------------
# A synthetic _results tree reconstructed from the committed tables (no data drive needed)
# --------------------------------------------------------------------------------------------


def _write_results(root: Path) -> Path:
    """Write index.csv, runs/<id>/summary.json and final.json consistent with the tables."""
    t1c = f3.read_table_1c(TABLES)
    expl = f3.read_exploratory(EXPLORATORY)
    constants = json.loads(CONSTANTS.read_text())["datasets"]
    results = root / "_results"
    rows = []
    for dataset, arm in f3.CELLS:
        for rid in sorted(r for r in t1c if r.startswith(f"{dataset}_{arm}_s")):
            block = constants[dataset]
            band = block["sigmas"][f"{SIGMA[arm]:g}"]
            m, n_pix, total = 50.0, int(block["n_pix"]), float(block["sum_power"])
            i_w = float(t1c[rid]["inherited_within"])
            d_pix = (1.0 - i_w) * (m - 1.0) / m * total / (n_pix ** 2 - 1)
            seed = int(rid.rsplit("_s", 1)[1])
            rows.append({"run_id": rid, "dataset": dataset, "arm": arm, "seed": seed,
                         "D_pix": repr(d_pix),
                         "inherited_predicted": repr(float(band["share_predicted"]))})
            folder = results / "runs" / rid
            folder.mkdir(parents=True)
            (folder / "summary.json").write_text(json.dumps(
                {"dataset_sha256": block["sha256"]}))
            (folder / "final.json").write_text(json.dumps({
                "sigma_max": SIGMA[arm], "n_per_seed": 50,
                "variance_ratio": 10.0 ** float(expl[rid]["log_vr"]),
                "n_reference": int(block["n_ref"]),
                "inherited_predicted": float(band["share_predicted"]),
            }))
    with (results / "index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return results


@pytest.fixture()
def tree(tmp_path: Path) -> dict[str, Path]:
    if not all(p.is_file() for p in COMMITTED):
        pytest.skip("committed result JSONs absent")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    copies = {}
    for name, src in (("t75", T75_JSON), ("a2", A2_JSON), ("tables", TABLES),
                      ("exploratory", EXPLORATORY), ("constants", CONSTANTS)):
        copies[name] = inputs / f"{name}.json"
        shutil.copyfile(src, copies[name])
    copies["results"] = _write_results(tmp_path)
    return copies


def _argv(tree: dict[str, Path], out: Path) -> list[str]:
    return ["--results", str(tree["results"]), "--heldout", str(tree["t75"]), str(tree["a2"]),
            "--tables", str(tree["tables"]), "--exploratory", str(tree["exploratory"]),
            "--constants", str(tree["constants"]), "--out", str(out)]


def _paths(tree: dict[str, Path], out: Path) -> f3.F3Paths:
    return f3.F3Paths(tree["results"], (tree["t75"], tree["a2"]), tree["tables"],
                      tree["exploratory"], tree["constants"], out)


# --------------------------------------------------------------------------------------------
# 1. Panel (a) means equal the ticket's table
# --------------------------------------------------------------------------------------------


def test_default_expectations_are_the_tickets() -> None:
    expect = f3.F3Expectations()
    assert expect.pr == TICKET_PR
    assert expect.within == TICKET_WITHIN
    assert (expect.pr_tol, expect.within_tol) == (0.0005, 0.005)


@needs_committed
def test_panel_a_means_from_the_committed_jsons() -> None:
    heldout, _ = f3.load_heldout([T75_JSON, A2_JSON])
    pr = f3.load_pr(_index_from_ids(_mri_ids(heldout)), heldout)
    assert [c.n for c in pr.values()] == [3, 2, 2, 3, 2, 2]
    for cell, expected in TICKET_PR.items():
        np.testing.assert_allclose(pr[cell].mean(), expected, rtol=0, atol=0.0005)
    lines = f3.assert_pr_means(pr, f3.F3Expectations())
    assert len(lines) == 24 and all(line.endswith("ok") for line in lines)


@needs_committed
def test_panel_a_mismatch_raises() -> None:
    heldout, _ = f3.load_heldout([T75_JSON, A2_JSON])
    pr = f3.load_pr(_index_from_ids(_mri_ids(heldout)), heldout)
    wrong = dict(TICKET_PR)
    wrong[("oasis1", "A3")] = (0.835, 0.526, 0.842, 0.141)  # the ticket's uncorrected values
    with pytest.raises(f3.F3ExpectationError, match="OASIS-1 matched precision F"):
        f3.assert_pr_means(pr, f3.F3Expectations(pr=wrong))


@pytest.mark.parametrize("value, places, expected", [
    (0.2375, 3, "0.238"), (0.0825, 3, "0.083"), (0.23749999999999, 3, "0.238"),
    (0.75849999999999, 3, "0.759"), (0.6646, 2, "0.66"), (28.49, 0, "28"),
])
def test_half_up_rounding(value: float, places: int, expected: str) -> None:
    assert f3._r(value, places) == expected


# --------------------------------------------------------------------------------------------
# 2. I_w and ρ recomputed
# --------------------------------------------------------------------------------------------


@needs_real
@needs_committed
def test_iw_and_rho_recomputed_from_index() -> None:
    index = read_index(REAL_RESULTS)
    t1c = f3.read_table_1c(TABLES)
    expl = f3.read_exploratory(EXPLORATORY)
    constants = load_constants(CONSTANTS)
    n_runs = 0
    for dataset, arm in f3.CELLS:
        for run in (r for r in index if (r.dataset, r.arm) == (dataset, arm)):
            record = f3.read_run_record(REAL_RESULTS, run.run_id)
            entry, reason = constants.lookup(dataset, record.dataset_sha256, record.sigma_max)
            assert entry is not None, reason
            i_w = within_seed_share(run.value("D_pix"), record.n_per_seed, entry.n_pix,
                                    entry.sum_power)
            np.testing.assert_allclose(i_w, t1c[run.run_id]["inherited_within"], rtol=0,
                                       atol=1e-6)
            np.testing.assert_allclose(regeneration_ratio(i_w, entry.share_predicted),
                                       expl[run.run_id]["regen_ratio"], rtol=0, atol=1e-6)
            n_runs += 1
    assert n_runs == 14


@needs_real
@needs_committed
def test_panel_b_means_on_the_real_results() -> None:
    within, lines = f3.load_within(read_index(REAL_RESULTS), REAL_RESULTS,
                                   f3.read_table_1c(TABLES), f3.read_exploratory(EXPLORATORY),
                                   load_constants(CONSTANTS))
    assert len(lines) == 4
    for cell, expected in TICKET_WITHIN.items():
        mean = within[cell].mean()
        np.testing.assert_allclose(mean[0], expected[0], rtol=0, atol=0.0005)
        np.testing.assert_allclose(mean[1:], expected[1:], rtol=0, atol=0.005)


def test_within_cross_check_catches_a_changed_rho(tree: dict[str, Path]) -> None:
    document = json.loads(tree["exploratory"].read_text())
    for record in document["per_run"]:
        if record["run_id"] == "ixi_A3_s2":
            record["regen_ratio"] += 1e-4
    tree["exploratory"].write_text(json.dumps(document))
    with pytest.raises(f3.F3ConsistencyError, match="ixi_A3_s2: rho"):
        f3.load_within(read_index(tree["results"]), tree["results"],
                       f3.read_table_1c(tree["tables"]), f3.read_exploratory(tree["exploratory"]),
                       load_constants(tree["constants"]))


# --------------------------------------------------------------------------------------------
# 3. Duplicated A0 runs
# --------------------------------------------------------------------------------------------


@needs_committed
def test_duplicated_a0_runs_are_identical() -> None:
    _, record = f3.load_heldout([T75_JSON, A2_JSON])
    assert record["identical_in_several_files"] == ["ixi_A0_s1", "ixi_A0_s2", "ixi_A0_s3"]


@needs_committed
def test_perturbed_duplicate_raises(tmp_path: Path) -> None:
    document = json.loads(A2_JSON.read_text())
    document["runs"]["ixi_A0_s2"]["metrics"]["H_rminus"]["recall"] += 0.001
    perturbed = tmp_path / "heldout_fidelity.json"
    perturbed.write_text(json.dumps(document))
    with pytest.raises(f3.PaperF3Error, match="ixi_A0_s2"):
        f3.load_heldout([T75_JSON, perturbed])


# --------------------------------------------------------------------------------------------
# 4. Build, byte stability, layout
# --------------------------------------------------------------------------------------------


def test_build_end_to_end_and_byte_stable(tree: dict[str, Path], tmp_path: Path) -> None:
    first = f3.build(_paths(tree, tmp_path / "one"))
    second = f3.build(_paths(tree, tmp_path / "two"))
    assert sorted(first.files) == [f"{f3.FIGURE_NAME}.{e}" for e in ("pdf", "png", "svg")]
    assert first.sha256() == second.sha256()
    a = first.audit
    assert a.width_in == pytest.approx(5.5) and 2.0 <= a.height_in <= 2.4
    assert a.min_font_pt >= 7.0 and not a.overlaps
    assert first.denominator["same"]
    svg = (tmp_path / "one" / f"{f3.FIGURE_NAME}.svg").read_text()
    for gid in ("a_mean_F_ixi_A0", "a_mean_H_oasis1_A3", "a_arrow_F_ixi", "a_arrow_H_ixi",
                "b_ideal_ixi_A3", "b_measured_oasis1_A0", "b_total_ixi_A1", "b_rho_ixi_A0",
                "b_real_line"):
        assert f'id="{gid}"' in svg
    assert "<dc:date>" not in svg
    notes = (tmp_path / "one" / "F3.md").read_text()
    assert "R5 sensitivity" in notes and "**Verdict: the same.**" in notes


@needs_real
@needs_committed
def test_real_build_is_byte_stable(tmp_path: Path) -> None:
    paths = [f3.F3Paths(REAL_RESULTS, (T75_JSON, A2_JSON), TABLES, EXPLORATORY, CONSTANTS,
                        tmp_path / name) for name in ("one", "two")]
    assert f3.build(paths[0]).sha256() == f3.build(paths[1]).sha256()


def test_denominator_difference_is_reported(tree: dict[str, Path]) -> None:
    final = tree["results"] / "runs" / "ixi_A0_s1" / "final.json"
    record = json.loads(final.read_text())
    record["n_reference"] = 400
    final.write_text(json.dumps(record))
    data, *_ = f3.load_data(_paths(tree, tree["results"]), f3.F3Expectations())
    check = f3.denominator_check(tree["results"], tree["constants"], data.within)
    assert not check["same"] and check["n_reference"]["ixi"] == [400, 800]


def test_text_overlaps_finds_a_collision() -> None:
    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(2, 1))
    fig.text(0.1, 0.5, "first label", gid="one")
    fig.text(0.15, 0.5, "second label", gid="two")
    fig.text(0.1, 0.1, "far", gid="three")
    try:
        assert f3.text_overlaps(fig) == ["one / two"]
    finally:
        plt.close(fig)


# --------------------------------------------------------------------------------------------
# 5. CLI exit codes
# --------------------------------------------------------------------------------------------


def test_cli_writes_and_exits_0(tree: dict[str, Path], tmp_path: Path) -> None:
    assert cli.main(_argv(tree, tmp_path / "out")) == cli.EXIT_OK
    assert (tmp_path / "out" / "F3.md").is_file()


def test_cli_missing_input_exits_2(tree: dict[str, Path], tmp_path: Path) -> None:
    argv = _argv(tree, tmp_path / "out")
    argv[argv.index("--results") + 1] = str(tmp_path / "nowhere")
    assert cli.main(argv) == cli.EXIT_MISSING


def test_cli_missing_run_file_exits_2(tree: dict[str, Path], tmp_path: Path) -> None:
    (tree["results"] / "runs" / "oasis1_A3_s2" / "final.json").unlink()
    assert cli.main(_argv(tree, tmp_path / "out")) == cli.EXIT_MISSING


def test_cli_mismatched_expectation_exits_1(tree: dict[str, Path], tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch,
                                            capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setitem(f3._WITHIN_EXPECTED, ("ixi", "A3"), (0.9205, 0.262, 0.285, 0.95))
    assert cli.main(_argv(tree, tmp_path / "out")) == cli.EXIT_FAIL
    assert "total variance ratio" in capsys.readouterr().err
    assert not (tmp_path / "out" / "F3.md").exists()


def test_cli_inconsistent_duplicate_exits_1(tree: dict[str, Path], tmp_path: Path) -> None:
    document = json.loads(tree["a2"].read_text())
    document["runs"]["ixi_A0_s1"]["metrics"]["F_rminus"]["precision"] = 0.5
    tree["a2"].write_text(json.dumps(document))
    assert cli.main(_argv(tree, tmp_path / "out")) == cli.EXIT_FAIL
