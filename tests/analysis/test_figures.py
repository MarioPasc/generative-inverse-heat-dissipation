"""Tests of the report figures (T6.2) on synthetic 30-run results trees.

The trees are real ``collect()`` outputs of T5.2's synthetic campaign
(``tests/analysis/synthetic_eval.py``): one complete, and one with the six tier-3 evaluations
absent, collected under ``allow_missing`` exactly like the real 24-run partial collection.
"""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest
from matplotlib import pyplot as plt
from PIL import Image

from ihdm.analysis import figures
from ihdm.analysis.collect import collect
from ihdm.analysis.figures import (
    FIGURES,
    FigureError,
    draw_all,
    load_results,
    write_readme,
)
from ihdm.analysis.style import ARM_ORDER, FULL_WIDTH_IN, SINGLE_WIDTH_IN, arm_style
from tests.analysis.synthetic_eval import Campaign, build_campaign, copy_campaign
from tests.analysis.test_inherited import SYNTHETIC_CORRECTED, synthetic_constants

#: The six cells whose evaluation was still running when the partial collection was made.
TIER3 = ("lsun_bedroom_A0_s1", "lsun_bedroom_A0_s2", "oasis1_A3_s1", "oasis1_A3_s2",
         "lsun_bedroom_A3_s1", "lsun_bedroom_A3_s2")
DPI = 60


def collected_results(campaign: Campaign, root: Path, absent: tuple[str, ...] = ()) -> Path:
    """Collect ``campaign`` into ``root/results`` after removing the evaluations of ``absent``."""
    if absent:
        campaign = copy_campaign(campaign, root / "campaign")
        for run_id in absent:
            campaign.summary_path(run_id).unlink()
            campaign.tar_path(run_id).unlink()
    out = root / "results"
    report = collect(campaign.config(out, allow_missing=bool(absent)))
    assert report.written is not None, "the synthetic collection was not published"
    return out


@pytest.fixture(scope="module")
def pristine(tmp_path_factory) -> Campaign:
    return build_campaign(tmp_path_factory.mktemp("t62") / "campaign")


@pytest.fixture(scope="module")
def complete_results(pristine, tmp_path_factory) -> Path:
    return collected_results(pristine, tmp_path_factory.mktemp("t62_complete"))


@pytest.fixture(scope="module")
def partial_results(pristine, tmp_path_factory) -> Path:
    return collected_results(pristine, tmp_path_factory.mktemp("t62_partial"), TIER3)


@pytest.fixture(scope="module")
def complete_drawn(complete_results, tmp_path_factory):
    out = tmp_path_factory.mktemp("t62_fig_complete")
    return out, {r.name: r for r in draw_all(load_results(complete_results), out, png_dpi=DPI)}


@pytest.fixture(scope="module")
def partial_drawn(partial_results, tmp_path_factory):
    out = tmp_path_factory.mktemp("t62_fig_partial")
    return out, {r.name: r for r in draw_all(load_results(partial_results), out, png_dpi=DPI)}


NAMES = [name for name, _ in FIGURES]


@pytest.mark.parametrize("name", NAMES)
def test_every_figure_writes_a_non_empty_pdf_and_png(complete_drawn, name):
    out, records = complete_drawn
    record = records[name]
    assert [p.name for p in record.files] == [f"{name}.pdf", f"{name}.png"]
    pdf, png = record.files
    assert pdf.read_bytes().startswith(b"%PDF") and pdf.stat().st_size > 1000
    with Image.open(png) as image:
        assert image.width == round(record.width_in * DPI)
        assert np.asarray(image.convert("L")).std() > 0  # not a blank canvas


@pytest.mark.parametrize("name", NAMES)
def test_a_complete_collection_uses_every_run_of_the_design(complete_drawn, name):
    record = complete_drawn[1][name]
    assert record.missing == [] and record.n_used == record.n_expected > 0
    assert f"n = {record.n_expected}/{record.n_expected} runs." in record.caption


def test_the_expected_run_counts_follow_the_design(complete_drawn):
    counts = {name: r.n_expected for name, r in complete_drawn[1].items()}
    assert counts == {"fig1_lsd_vs_iteration": 30, "fig2_lsd_octaves": 30,
                      "fig3_diversity_memorisation": 30, "fig4_pca_seed": 4,
                      "fig5_inherited_band": 20, "fig6_grids": 8, "fig7_training_sanity": 30}


def test_the_report_widths_are_used(complete_drawn):
    widths = {r.width_in for r in complete_drawn[1].values()}
    assert widths == {FULL_WIDTH_IN, SINGLE_WIDTH_IN}


@pytest.mark.parametrize("name", NAMES)
def test_missing_runs_are_drawn_around_and_named(partial_drawn, name):
    out, records = partial_drawn
    record = records[name]
    assert all(p.stat().st_size > 0 for p in record.files)
    assert record.n_used + len(record.missing) == record.n_expected
    assert set(record.missing) <= set(TIER3)
    assert f"n = {record.n_used}/{record.n_expected} runs." in record.caption
    for run_id in record.missing:
        assert f"`{run_id}`" in record.caption


def test_the_partial_collection_counts_what_is_absent(partial_drawn):
    coverage = {name: r.coverage for name, r in partial_drawn[1].items()}
    # The evaluation-based figures lose the six tier-3 runs; grids and histories are training
    # outputs the collection copies even for an unevaluated run.
    assert coverage == {"fig1_lsd_vs_iteration": "24/30 runs", "fig2_lsd_octaves": "24/30 runs",
                        "fig3_diversity_memorisation": "24/30 runs",
                        "fig4_pca_seed": "4/4 runs", "fig5_inherited_band": "14/20 runs",
                        "fig6_grids": "8/8 runs", "fig7_training_sanity": "30/30 runs"}


def test_a_run_without_its_history_or_grid_is_named(partial_results, tmp_path):
    root = tmp_path / "results"
    shutil.copytree(partial_results, root)
    (root / "runs" / "ixi_A1_s2" / "metrics.canonical.jsonl").unlink()
    (root / "runs" / "ixi_A0_s1" / "grid_final.png").unlink()
    results = load_results(root)
    record = figures.fig_training_sanity(results, tmp_path / "out", png_dpi=DPI)
    assert record.missing == ["ixi_A1_s2"] and record.coverage == "29/30 runs"
    record = figures.fig_grids(results, tmp_path / "out", png_dpi=DPI)
    assert record.missing == ["ixi_A0_s1"] and record.coverage == "7/8 runs"


def test_a_rerun_is_byte_stable(complete_results, tmp_path):
    results = load_results(complete_results)
    first = draw_all(results, tmp_path / "a", png_dpi=DPI)
    second = draw_all(load_results(complete_results), tmp_path / "b", png_dpi=DPI)
    for a, b in zip(first, second, strict=True):
        for fa, fb in zip(a.files, b.files, strict=True):
            assert fa.read_bytes() == fb.read_bytes(), fa.name


def test_the_pdfs_carry_no_creation_date(complete_drawn):
    for record in complete_drawn[1].values():
        data = record.files[0].read_bytes()
        assert b"/CreationDate" not in data and b"/ModDate" not in data


def test_no_pdf_writes_only_the_previews(complete_results, tmp_path):
    records = draw_all(load_results(complete_results), tmp_path, pdf=False, png_dpi=DPI,
                       only=["fig2_lsd_octaves"])
    assert [p.name for r in records for p in r.files] == ["fig2_lsd_octaves.png"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["fig2_lsd_octaves.png"]


def test_the_readme_holds_a_caption_and_the_coverage_per_figure(partial_drawn, partial_results):
    out, records = partial_drawn
    readme = write_readme(list(records.values()), load_results(partial_results),
                          out / "README.md", "python -m ihdm.cli.figures --results x --out y")
    text = readme.read_text()
    assert "> **Partial collection.**" in text
    for number, record in enumerate(records.values(), start=1):
        assert f"## Figure {number}. {record.title}" in text
        assert f"Coverage: **{record.coverage}**" in text
        assert f"({record.name}.png)" in text and f"({record.name}.pdf)" in text
    assert "lsun_church_A3_s1" in text and "stripe" in text


def test_a_complete_readme_has_no_partial_banner(complete_drawn, complete_results):
    out, records = complete_drawn
    text = write_readme(list(records.values()), load_results(complete_results),
                        out / "README.md", "cmd").read_text()
    assert "Partial collection" not in text and "complete | True" in text


def test_a_folder_without_index_is_refused(tmp_path):
    with pytest.raises(FigureError, match="index.csv"):
        load_results(tmp_path)


def test_an_index_row_without_identity_is_refused(tmp_path):
    (tmp_path / "index.csv").write_text("index,run_id\n0,x\n")
    with pytest.raises(FigureError, match="unreadable row"):
        load_results(tmp_path)


def test_every_arm_of_the_experiment_has_a_distinct_style():
    from configs.spectral.arms import ARMS

    assert set(ARM_ORDER) == set(ARMS)
    styles = [arm_style(a) for a in ARM_ORDER]
    assert len({s.color for s in styles}) == len({s.marker for s in styles}) == len(ARM_ORDER)


@pytest.mark.parametrize("values, window, expected", [
    (np.array([]), 3, np.array([])),
    (np.array([1.0, 2.0]), 3, np.array([1.0, 2.0])),
    (np.array([1.0, 2.0, 3.0, 4.0]), 2, np.array([1.0, 1.5, 2.5, 3.5])),
    (np.array([5.0]), 1, np.array([5.0])),
])
def test_the_running_mean_keeps_the_length(values, window, expected):
    np.testing.assert_allclose(figures._running_mean(values, window), expected)


def test_an_all_nan_band_gives_nan_without_a_warning():
    table = np.array([[1.0, np.nan], [3.0, np.nan]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = figures._nanmean_columns(table)
    np.testing.assert_allclose(result[0], 2.0)
    assert np.isnan(result[1])


def test_the_octave_loss_uses_only_the_last_steps():
    history = [
        {"kind": "train", "step": 1000, "loss_per_octave": {"a": 9.0, "b": None}},
        {"kind": "train", "step": 58000, "loss_per_octave": {"a": 1.0, "b": None}},
        {"kind": "train", "step": 60000, "loss_per_octave": {"a": 3.0, "b": None}},
        {"kind": "eval", "step": 60000, "loss": 0.3},
    ]
    result = figures._octave_loss(history, 60000, ("a", "b"))
    np.testing.assert_allclose(result[0], 2.0)
    assert np.isnan(result[1])


# ---- figure 5 under D23 -------------------------------------------------------------------------


def _dashed(ax) -> list:
    return [line for line in ax.get_lines() if line.get_linestyle() == "--"]


def test_figure_5_draws_the_d23_prediction_and_keeps_the_reference(complete_results, tmp_path):
    path = synthetic_constants(complete_results, tmp_path / "constants.json")
    results = load_results(complete_results, path)
    fig, ax = plt.subplots()
    notes: list[str] = []
    assert figures._band_panel(ax, results, "ixi", notes)
    assert notes == []
    dashed = _dashed(ax)
    assert len(dashed) == len(figures.BAND_ARMS)
    for line in dashed:
        np.testing.assert_allclose(line.get_ydata(), SYNTHETIC_CORRECTED)
    dotted = [line for line in ax.get_lines() if line.get_linestyle() == ":"]
    assert len(dotted) == len(figures.BAND_ARMS)
    assert all(line.get_linewidth() < 1.0 for line in dotted)
    plt.close(fig)

    record = figures.fig_inherited_band(results, tmp_path / "out", png_dpi=DPI)
    assert "below the line, the chain adds less than the variance the blur removed" in (
        record.caption)
    assert "inherit more of the seed" not in record.caption
    assert "without the D23 prediction" not in record.caption
    with Image.open(record.files[1]) as image:
        assert np.asarray(image.convert("L")).std() > 0


def test_figure_5_without_matching_constants_says_so(complete_results, tmp_path):
    results = load_results(complete_results, None)
    fig, ax = plt.subplots()
    notes: list[str] = []
    assert figures._band_panel(ax, results, "ixi", notes)
    assert _dashed(ax) == [] and len(notes) == len(figures.BAND_ARMS)
    plt.close(fig)
    record = figures.fig_inherited_band(results, tmp_path, pdf=False, png_dpi=DPI)
    assert "Panels drawn without the D23 prediction" in record.caption


def test_the_readme_explains_the_d23_correction_with_numbers(complete_drawn, complete_results,
                                                             tmp_path):
    out, records = complete_drawn
    path = synthetic_constants(complete_results, tmp_path / "constants.json")
    text = write_readme(list(records.values()), load_results(complete_results, path),
                        out / "README.md", "cmd").read_text()
    section = text.split("## Inherited band: estimator correction (D23)", 1)[1]
    assert "inherited_band_audit.md" in section
    assert "| dataset | σ_B,max | ΣP_ref | I | T | I − T |" in section
    assert "| IXI T1 | A0 | 3 |" in section
    unmatched = write_readme(list(records.values()), load_results(complete_results, None),
                             tmp_path / "README.md", "cmd").read_text()
    assert "No run of this folder matches the constants" in unmatched
