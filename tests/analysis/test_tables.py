"""Tests of ``ihdm.analysis.tables`` (T6.1) on synthetic 30-run results with planted effects.

The results folders are real collections: the T5.2 builders write a healthy campaign, the real
``collect`` publishes it, and :func:`plant` overwrites the endpoints of ``index.csv`` and of each
run's ``summary.json`` together (so the loader's cross-check still passes) with a design whose
effects are known:

* IXI: A3 −0.10, A1 −0.06, A2 −0.03 against A0 on every float endpoint (planted effect);
* Churches: A3, A1, A2 null (per-seed noise of mixed sign), A2′ −0.05;
* OASIS-1 A3 −0.08, Bedrooms A3 +0.03 (the transfer signs);
* endpoint ``M`` alone: A3 −0.05 on both IXI and Churches (a null interaction).
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from ihdm.analysis.collect import collect, strict_json
from ihdm.analysis.tables import (
    _SUMMARY_KEYS,
    DETECTABLE,
    ENDPOINTS,
    EXPLORATORY_LABEL,
    NOT_DETECTABLE,
    AnalysisError,
    ResultsNotFound,
    analyse,
    build_tables,
    compute_contrast,
    compute_interaction,
    design_runs,
    fmt_num,
    fmt_step,
    load_results,
    t_settle,
    tex_escape,
    write_tables,
)
from ihdm.metrics.io import write_json
from ihdm.metrics.spectral import t_tau
from tests.analysis.synthetic_eval import Campaign, copy_campaign

STEPS = tuple(range(5000, 60001, 5000))
FLOAT_ENDPOINTS = ("lsd_final", "kid", "fid", "recall", "coverage", "M", "M_lp",
                   "seed_nn_fraction", "D_pix", "D_lp", "inherited_measured")
#: Planted mean effect of an arm against A0, per dataset (every float endpoint except ``M``).
EFFECT: dict[tuple[str, str], float] = {
    ("ixi", "A3"): -0.10, ("ixi", "A1"): -0.06, ("ixi", "A2"): -0.03,
    ("lsun_church", "A3"): 0.0, ("lsun_church", "A1"): 0.0, ("lsun_church", "A2"): 0.0,
    ("lsun_church", "A2p"): -0.05, ("oasis1", "A3"): -0.08, ("lsun_bedroom", "A3"): 0.03,
}
#: ``M`` only: the same A3 effect on both development datasets, so no interaction.
EFFECT_M: dict[tuple[str, str], float] = {("ixi", "A3"): -0.05, ("lsun_church", "A3"): -0.05}
#: Per-seed deviation of a non-A0 arm: small and mixed on IXI, larger and mixed on Churches.
NOISE: dict[str, dict[int, float]] = {
    "ixi": {1: 0.005, 2: -0.005, 3: 0.002},
    "lsun_church": {1: 0.02, 2: -0.02, 3: 0.006},
    "oasis1": {1: 0.005, 2: -0.005},
    "lsun_bedroom": {1: 0.005, 2: -0.005},
}
TIER_3 = ("lsun_bedroom_A0_s1", "lsun_bedroom_A0_s2", "oasis1_A3_s1", "oasis1_A3_s2",
          "lsun_bedroom_A3_s1", "lsun_bedroom_A3_s2")


def _round(value: float) -> float:
    """What ``write_json`` keeps, so that index.csv and summary.json hold the same number."""
    return float(f"{value:.6g}")


def planted_value(dataset: str, arm: str, seed: int, endpoint: str) -> float:
    """The planted value of one endpoint of one run."""
    a0 = 1.0 + 0.01 * seed
    if arm == "A0":
        return a0
    effects = EFFECT_M if endpoint == "M" else EFFECT
    return a0 + effects.get((dataset, arm), 0.0) + NOISE[dataset][seed]


def planted_curve(dataset: str, arm: str, seed: int) -> dict[int, float]:
    """A strictly decreasing LSD curve ending at the planted final LSD − 0.001."""
    end = planted_value(dataset, arm, seed, "lsd_final") - 0.001
    return {s: end + 0.3 * (math.exp(-s / 15000) - math.exp(-STEPS[-1] / 15000)) for s in STEPS}


def planted_design() -> dict[str, dict[str, float]]:
    """``{run_id: {index column: value}}`` for the 30 runs."""
    values: dict[str, dict[str, float]] = {}
    for rid, dataset, arm, seed in design_runs():
        record = {e: planted_value(dataset, arm, seed, e) for e in FLOAT_ENDPOINTS}
        curve = planted_curve(dataset, arm, seed)
        record.update({f"lsd_{s:06d}": v for s, v in curve.items()})
        record["lsd_final_ckpt"] = curve[STEPS[-1]]
        values[rid] = {k: _round(v) for k, v in record.items()}
    return values


def _set_path(record: dict, dotted: str, value: float) -> None:
    *parents, leaf = dotted.split(".")
    for key in parents:
        record = record[key]
    record[leaf] = value


def plant(results: Path, values: Mapping[str, Mapping[str, float]]) -> None:
    """Overwrite endpoints in ``index.csv`` and in each evaluated run's ``summary.json``."""
    index = results / "index.csv"
    with index.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns, rows = list(reader.fieldnames or []), list(reader)
    for row in rows:
        summary_path = results / "runs" / row["run_id"] / "summary.json"
        if not summary_path.is_file() or row["run_id"] not in values:
            continue
        summary = json.loads(summary_path.read_text())
        for column, value in values[row["run_id"]].items():
            row[column] = str(value)
            if column.startswith("lsd_0"):
                summary["lsd_by_step"][str(int(column[4:]))] = value
            else:
                _set_path(summary, _SUMMARY_KEYS[column], value)
        write_json(summary_path, summary)
    with index.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def collect_planted(campaign: Campaign, out: Path, allow_missing: bool = False) -> Path:
    """Collect a campaign with the real ``collect`` and plant the design into the folder."""
    report = collect(campaign.config(out, allow_missing=allow_missing))
    assert report.verdict in {"COMPLETE", "INCOMPLETE"}
    plant(out, planted_design())
    return out


def drop_runs(campaign: Campaign, run_ids: tuple[str, ...]) -> None:
    """Remove the evaluation outputs of some runs (their training directories stay)."""
    for rid in run_ids:
        campaign.summary_path(rid).unlink()
        campaign.tar_path(rid).unlink()


@pytest.fixture(scope="module")
def planted(pristine_campaign, tmp_path_factory) -> Path:
    """A complete 30-run results folder with the planted design."""
    return collect_planted(pristine_campaign, tmp_path_factory.mktemp("t61") / "results")


@pytest.fixture(scope="module")
def results(planted):
    return load_results(planted)


@pytest.fixture(scope="module")
def partial(pristine_campaign, tmp_path_factory) -> Path:
    """The 24-run shape of 2026-09-29: tier-3 cells 24-29 not evaluated."""
    root = tmp_path_factory.mktemp("t61partial")
    campaign = copy_campaign(pristine_campaign, root / "campaign")
    drop_runs(campaign, TIER_3)
    return collect_planted(campaign, root / "results", allow_missing=True)


def _tables(res) -> dict:
    return {t.name: t for t in build_tables(res)}


# ---- loading ------------------------------------------------------------------------------------


def test_the_complete_folder_loads_every_run_of_the_design(results):
    assert len(results.frame) == 30 and len(results.present) == 30
    assert results.missing == ()
    assert results.steps == STEPS
    assert results.value("ixi_A3_s2", "lsd_final") == pytest.approx(1.02 - 0.10 - 0.005)
    # the low-band shares come from the summaries through ihdm.stats.cell_table
    assert np.isfinite(results.frame["inherited_measured_low"]).all()


def test_the_partial_folder_marks_the_tier_3_runs_missing(partial):
    res = load_results(partial)
    assert set(res.missing) == set(TIER_3) and len(res.present) == 24
    assert res.frame.loc[list(TIER_3), "lsd_final"].isna().all()


def test_a_path_without_a_collection_is_not_a_results_folder(tmp_path):
    with pytest.raises(ResultsNotFound, match="no collection.json"):
        load_results(tmp_path)


def test_a_failed_collection_is_refused(planted, tmp_path):
    copy = tmp_path / "results"
    _copy_tree(planted, copy)
    collection = json.loads((copy / "collection.json").read_text())
    collection["verdict"] = "FAIL"
    (copy / "collection.json").write_text(json.dumps(collection))
    with pytest.raises(AnalysisError, match="verdict is FAIL"):
        load_results(copy)


def test_an_index_that_disagrees_with_a_summary_is_refused(planted, tmp_path):
    copy = tmp_path / "results"
    _copy_tree(planted, copy)
    text = (copy / "index.csv").read_text()
    row = next(line for line in text.splitlines() if ",ixi_A3_s1," in line)
    cells = row.split(",")
    header = text.splitlines()[0].split(",")
    cells[header.index("kid")] = "0.5"
    (copy / "index.csv").write_text(text.replace(row, ",".join(cells)))
    with pytest.raises(AnalysisError, match="ixi_A3_s1.kid"):
        load_results(copy)


def test_an_evaluated_row_without_its_summary_is_refused(planted, tmp_path):
    copy = tmp_path / "results"
    _copy_tree(planted, copy)
    (copy / "runs" / "ixi_A2_s1" / "summary.json").unlink()
    with pytest.raises(AnalysisError, match="ixi_A2_s1"):
        load_results(copy)


def _copy_tree(source: Path, dest: Path) -> None:
    import shutil

    shutil.copytree(source, dest)


# ---- planted effect, null effect, p floor -------------------------------------------------------


def test_a_planted_effect_is_recovered_with_its_sign_and_a_ci_excluding_zero(results):
    for endpoint in FLOAT_ENDPOINTS:
        if endpoint == "M":
            continue
        c = compute_contrast(results, "ixi", "A3", endpoint)
        assert c.ok and c.seeds == (1, 2, 3)
        assert c.mean == pytest.approx(-0.10 + np.mean([0.005, -0.005, 0.002]), abs=1e-5)
        assert c.ci_high < 0.0 and c.excludes_zero and c.verdict == DETECTABLE


def test_a_null_effect_gives_a_ci_containing_zero(results):
    for endpoint in FLOAT_ENDPOINTS:
        if endpoint == "M":
            continue
        c = compute_contrast(results, "lsun_church", "A3", endpoint)
        assert c.ci_low < 0.0 < c.ci_high
        assert c.excludes_zero is False and c.verdict == NOT_DETECTABLE
        assert c.p_value > c.p_min


def test_with_three_seeds_the_interval_is_the_range_of_the_paired_differences(results):
    for dataset in ("ixi", "lsun_church"):
        c = compute_contrast(results, dataset, "A3", "kid")
        assert (c.ci_low, c.ci_high) == pytest.approx((min(c.deltas), max(c.deltas)), abs=1e-12)


def test_the_permutation_floor_is_0_1_with_three_seeds_and_a_third_with_two(results):
    a3 = compute_contrast(results, "ixi", "A3", "lsd_final")
    assert (a3.n_assignments, a3.p_min, a3.p_value) == (20, pytest.approx(0.1),
                                                        pytest.approx(0.1))
    a1 = compute_contrast(results, "ixi", "A1", "lsd_final")
    assert a1.seeds == (1, 2) and a1.n_assignments == 6
    assert a1.p_min == pytest.approx(1 / 3) and a1.p_value >= a1.p_min


def test_every_p_value_in_every_table_respects_its_floor(results):
    for table in build_tables(results):
        for row in table.rows:
            for key, value in row.items():
                if key.endswith("p_value") or key.endswith("_p"):
                    floor = row.get(key.replace("p_value", "p_min") if key.endswith("p_value")
                                    else f"{key}_min")
                    if value is not None and not math.isnan(value):
                        assert value >= floor - 1e-12


def test_the_interaction_recovers_the_planted_effect_and_its_floor(results):
    result = compute_interaction(results, "lsd_final")
    assert result.point == pytest.approx((-0.10 + 0.002 / 3) - 0.002, abs=1e-5)
    assert result.ci_high < 0.0 and result.verdict == DETECTABLE
    assert result.n_assignments == 20 and result.p_value == pytest.approx(0.1)
    assert result.p_min == pytest.approx(0.1)


def test_an_equal_effect_on_both_datasets_gives_an_interaction_ci_containing_zero(results):
    result = compute_interaction(results, "M")
    assert result.mri.excludes_zero and result.photo.excludes_zero
    assert result.ci_low < 0.0 < result.ci_high and result.verdict == NOT_DETECTABLE


# ---- T_tau --------------------------------------------------------------------------------------


def test_t_tau_follows_05_section_2_with_the_curve_threshold(results):
    frame = results.frame
    for rid, dataset, _arm, seed in design_runs():
        a0 = f"{dataset}_A0_s{seed}"
        curve = {s: frame.at[rid, f"lsd_{s:06d}"] for s in STEPS}
        threshold = frame.at[a0, "lsd_060000"]
        expected = t_tau(curve, threshold)
        assert frame.at[rid, "t_tau_threshold"] == threshold
        assert frame.at[rid, "t_tau_threshold_2k"] == frame.at[a0, "lsd_final"]
        got = frame.at[rid, "t_tau"]
        assert (math.isinf(got) if expected is None else got == expected), rid
    # a monotone A0 curve reaches its own last value exactly at the last step
    assert frame.at["ixi_A0_s1", "t_tau"] == 60000
    assert frame.at["ixi_A3_s1", "t_tau"] < 60000


def test_a_curve_that_never_reaches_the_threshold_makes_the_contrast_not_computable(results):
    # Churches A2 seed 1 sits +0.02 above A0 at every step
    assert math.isinf(results.frame.at["lsun_church_A2_s1", "t_tau"])
    c = compute_contrast(results, "lsun_church", "A2", "t_tau")
    assert not c.ok and "not reached" in c.status and "lsun_church_A2_s1" in c.status
    assert math.isnan(c.ci_low)
    inter = compute_interaction(results, "t_tau")
    assert inter.status.startswith("not computable")


def test_the_settling_step_differs_from_the_first_crossing_on_a_non_monotone_curve():
    curve = {5000: 0.20, 10000: 0.30, 15000: 0.25, 20000: 0.22}
    assert t_tau(curve, 0.25) == 5000
    assert t_settle(curve, 0.25) == 15000
    assert t_settle(curve, 0.21) is None
    assert t_settle({5000: 0.3, 10000: 0.2}, 0.2) == 10000


# ---- the tables ---------------------------------------------------------------------------------


def test_every_table_is_built_and_complete_on_the_full_folder(results):
    tables = build_tables(results)
    names = [t.name for t in tables]
    assert names == ["t1a_cells_fidelity", "t1b_cells_mechanism", "t2a_contrasts_ixi",
                     "t2b_contrasts_lsun_church", "t3_interaction", "t4_decomposition",
                     "t5_a2p_control", "t6_transfer", "t7_t_tau", "t8_gates",
                     "t9_settle_exploratory"]
    for table in tables:
        assert table.complete, table.name
        assert table.status.startswith("complete ("), table.status


def test_the_cell_table_has_one_row_per_run_and_every_ticket_column(results):
    t = _tables(results)
    for name in ("t1a_cells_fidelity", "t1b_cells_mechanism"):
        assert len(t[name].rows) == 30
    keys = set(t["t1a_cells_fidelity"].rows[0]) | set(t["t1b_cells_mechanism"].rows[0])
    for key in ("lsd_final", "t_tau", "kid", "kid_ci_low", "kid_ci_high", "fid", "fid_ci_low",
                "fid_ci_high", "fid_n_reference", "recall", "coverage", "M", "M_lp", "D_pix",
                "D_lp", "inherited_measured", "inherited_predicted", "n_skipped"):
        assert key in keys, key
    run_11 = next(r for r in t["t1b_cells_mechanism"].rows if r["run_id"] == "lsun_church_A3_s3")
    assert run_11["n_skipped"] == 6


def test_the_contrast_tables_cover_each_arm_and_endpoint(results):
    t = _tables(results)
    assert len(t["t2a_contrasts_ixi"].rows) == 3 * len(ENDPOINTS)
    assert len(t["t2b_contrasts_lsun_church"].rows) == 4 * len(ENDPOINTS)
    assert {r["contrast"] for r in t["t2b_contrasts_lsun_church"].rows} == {
        "A3−A0", "A1−A0", "A2−A0", "A2′−A0"}


def test_the_decomposition_recovers_the_planted_shares(results):
    rows = {(r["dataset"], r["endpoint"]): r for r in _tables(results)["t4_decomposition"].rows}
    ixi = rows[("ixi", "kid")]
    assert ixi["seeds"] == [1, 2]
    assert ixi["share_a1"] == pytest.approx(0.6, abs=1e-4)
    assert ixi["share_a2"] == pytest.approx(0.3, abs=1e-4)
    assert ixi["status"] == "ok"
    church = rows[("lsun_church", "kid")]
    assert "not interpretable" in church["status"]


def test_the_a2p_control_contrasts_a2p_with_a0_and_with_a2(results):
    row = next(r for r in _tables(results)["t5_a2p_control"].rows if r["endpoint"] == "kid")
    assert row["a2p_a2_mean"] == pytest.approx(-0.05, abs=1e-5)
    assert row["a2p_a2_verdict"] == DETECTABLE and row["a2p_a2_p_min"] == pytest.approx(1 / 3)
    assert row["a2_a0_verdict"] == NOT_DETECTABLE


def test_the_transfer_table_reports_signs_and_magnitudes_without_p_values(results):
    table = _tables(results)["t6_transfer"]
    rows = {(r["pair"], r["endpoint"]): r for r in table.rows}
    mri = rows[("IXI → OASIS-1", "kid")]
    assert (mri["dev_sign"], mri["transfer_sign"], mri["agree"]) == ("−", "−", True)
    assert mri["transfer_mean"] == pytest.approx(-0.08, abs=1e-5)
    assert mri["transfer_relative"] == pytest.approx(-0.08 / 1.015, abs=1e-4)
    photo = rows[("Churches → Bedrooms", "kid")]
    assert photo["transfer_sign"] == "+"
    assert not any("p" == key or key.startswith("p_") or key.endswith("_p")
                   for row in table.rows for key in row)


def test_the_exploratory_table_is_labelled_and_carries_no_inference(results):
    table = _tables(results)["t9_settle_exploratory"]
    assert EXPLORATORY_LABEL in table.title
    assert not any(key.startswith(("p_", "ci_")) for row in table.rows for key in row)
    assert table.rows[-1]["contrast"].startswith("(A3−A0) IXI")


def test_the_gate_table_holds_both_pairs_of_both_gate_runs(results):
    table = _tables(results)["t8_gates"]
    assert table.complete and len(table.rows) == 4
    assert {(r["run_id"], r["early"]) for r in table.rows} == {
        ("ixi_A0_s1", 35000), ("ixi_A0_s1", 55000), ("lsun_church_A0_s1", 35000),
        ("lsun_church_A0_s1", 55000)}


def test_the_partial_folder_marks_tables_incomplete_and_drops_no_row(partial):
    res = load_results(partial)
    t = _tables(res)
    expected = {
        "t1a_cells_fidelity": "incomplete: 24/30 runs",
        "t1b_cells_mechanism": "incomplete: 24/30 runs",
        "t6_transfer": "incomplete: 14/20 runs",
        "t7_t_tau": "incomplete: 24/30 runs",
        "t9_settle_exploratory": "incomplete: 24/30 runs",
    }
    for name, table in t.items():
        assert table.status == expected.get(name, table.status), name
        assert table.complete == (name not in expected), name
    assert len(t["t1a_cells_fidelity"].rows) == 30
    missing_rows = [r for r in t["t1a_cells_fidelity"].rows if r["run_id"] in TIER_3]
    assert len(missing_rows) == 6
    assert all(r["status"] == "missing: not evaluated" for r in missing_rows)
    transfer = [r for r in t["t6_transfer"].rows if r["pair"] == "IXI → OASIS-1"]
    assert all(r["status"] == "incomplete: 2/4 runs" and r["agree"] is None for r in transfer)
    # the headline needs no tier-3 run and is computed in full
    assert t["t3_interaction"].complete
    assert t["t3_interaction"].rows[0]["verdict"] == DETECTABLE


# ---- writing ------------------------------------------------------------------------------------


def test_write_tables_writes_markdown_latex_and_one_strict_json(results, tmp_path):
    out = tmp_path / "tables"
    out.mkdir()
    (out / "README.md").write_text("hand-written\n")
    tables = build_tables(results)
    written = write_tables(results, tables, out)
    assert (out / "README.md").read_text() == "hand-written\n"
    for table in tables:
        md = (out / f"{table.name}.md").read_text()
        tex = (out / f"{table.name}.tex").read_text()
        assert md.startswith(f"## Table {table.number} — ") and "**Status: complete" in md
        assert r"\toprule" in tex and r"\midrule" in tex and r"\bottomrule" in tex
        assert rf"\label{{tab:{table.name.replace('_', '-')}}}" in tex
        body = tex.split(r"\begin{table}")[1]
        assert "$$" not in body
        assert all(ch not in body for ch in "−′Δτ—→")
    document = strict_json((out / "tables.json").read_bytes())
    assert document["complete"] is True and set(document["tables"]) == {t.name for t in tables}
    assert len(written) == 2 * len(tables) + 1
    row = document["tables"]["t3_interaction"]["rows"][0]
    assert row["endpoint"] == "lsd_final" and row["p_value"] == pytest.approx(0.1)


def test_a_partial_folder_is_bannered_in_every_file(partial, tmp_path):
    report = analyse(partial, tmp_path / "tables")
    assert not report.complete
    for table in report.tables:
        md = (tmp_path / "tables" / f"{table.name}.md").read_text()
        assert "PARTIAL COLLECTION (verdict INCOMPLETE, 24/30 runs evaluated)" in md
    six = (tmp_path / "tables" / "t6_transfer.md").read_text()
    assert "**Status: incomplete: 14/20 runs** (missing: " in six
    document = json.loads((tmp_path / "tables" / "tables.json").read_text())
    assert document["complete"] is False and sorted(document["missing_runs"]) == sorted(TIER_3)
    assert document["tables"]["t7_t_tau"]["rows"][-1]["t_tau"] is None


# ---- formatting ---------------------------------------------------------------------------------


@pytest.mark.parametrize(("value", "text"), [
    (57.2033, "57.20"), (0.0463751, "0.04638"), (0.000273773, "2.738e-04"), (-0.767507, "-0.7675"),
    (0.0, "0"), (float("nan"), "—"), (None, "—"), (float("inf"), "not reached"),
])
def test_fmt_num(value, text):
    assert fmt_num(value) == text


@pytest.mark.parametrize(("value", "signed", "text"), [
    (5000.0, False, "5k"), (60000, False, "60k"), (-30000.0, True, "-30k"), (10000.0, True, "+10k"),
    (-8333.333, True, "-8.33k"), (float("inf"), False, "not reached"), (float("nan"), False, "—"),
])
def test_fmt_step(value, signed, text):
    assert fmt_step(value, signed) == text


def test_tex_escape_typesets_minus_signs_scientific_notation_and_underscores():
    assert tex_escape("lsun_church_A3_s3") == r"lsun\_church\_A3\_s3"
    assert tex_escape("[-0.12, +0.30]") == r"[\ensuremath{-}0.12, +0.30]"
    assert tex_escape("2.738e-04") == r"\ensuremath{2.738\times10^{-4}}"
    assert tex_escape("A2′−A0") == r"A2\ensuremath{'}\ensuremath{-}A0"
    assert tex_escape("5%") == r"5\%"
