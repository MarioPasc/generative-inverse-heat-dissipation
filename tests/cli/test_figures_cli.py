"""Smoke tests of ``python -m ihdm.cli.figures`` (T6.2): exit codes, outputs, the README."""

from __future__ import annotations

from pathlib import Path

import pytest

from ihdm.cli.figures import EXIT_FAIL, EXIT_OK, EXIT_PARTIAL, EXIT_USAGE, main
from tests.analysis.synthetic_eval import Campaign, build_campaign
from tests.analysis.test_figures import TIER3, collected_results

FIGURE_NAMES = ("fig1_lsd_vs_iteration", "fig2_lsd_octaves", "fig3_diversity_memorisation",
                "fig4_pca_seed", "fig5_inherited_band", "fig6_grids", "fig7_training_sanity")


@pytest.fixture(scope="module")
def campaign(tmp_path_factory) -> Campaign:
    return build_campaign(tmp_path_factory.mktemp("t62_cli") / "campaign")


@pytest.fixture(scope="module")
def complete(campaign, tmp_path_factory) -> Path:
    return collected_results(campaign, tmp_path_factory.mktemp("t62_cli_complete"))


@pytest.fixture(scope="module")
def partial(campaign, tmp_path_factory) -> Path:
    return collected_results(campaign, tmp_path_factory.mktemp("t62_cli_partial"), TIER3)


def test_a_complete_collection_draws_every_figure_and_exits_0(complete, tmp_path, capsys):
    out = tmp_path / "figures"
    assert main(["--results", str(complete), "--out", str(out), "--png-dpi", "50"]) == EXIT_OK
    names = sorted(p.name for p in out.iterdir())
    assert names == sorted([f"{n}.{ext}" for n in FIGURE_NAMES for ext in ("pdf", "png")]
                           + ["README.md"])
    text = capsys.readouterr().out
    assert "VERDICT: COMPLETE" in text and text.count("30/30 runs") == 4


def test_a_partial_collection_exits_3_and_names_the_absent_runs(partial, tmp_path, capsys):
    out = tmp_path / "figures"
    readme = tmp_path / "elsewhere" / "README.md"
    code = main(["--results", str(partial), "--out", str(out), "--no-pdf", "--png-dpi", "50",
                 "--readme", str(readme)])
    assert code == EXIT_PARTIAL
    assert sorted(p.name for p in out.iterdir()) == sorted(f"{n}.png" for n in FIGURE_NAMES)
    text = readme.read_text()
    assert "Partial collection" in text and "--no-pdf" in text
    assert "(../figures/fig1_lsd_vs_iteration.png)" in text
    assert all(run_id in text for run_id in TIER3)
    assert "VERDICT: PARTIAL" in capsys.readouterr().out


def test_only_draws_the_named_figures(complete, tmp_path):
    out = tmp_path / "figures"
    code = main(["--results", str(complete), "--out", str(out), "--no-pdf", "--png-dpi", "50",
                 "--only", "fig4_pca_seed"])
    assert code == EXIT_OK
    assert sorted(p.name for p in out.iterdir()) == ["README.md", "fig4_pca_seed.png"]


def test_a_folder_that_is_not_a_results_folder_exits_1(tmp_path, capsys):
    assert main(["--results", str(tmp_path), "--out", str(tmp_path / "o")]) == EXIT_FAIL
    assert "index.csv" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [
    [],
    ["--results", "x"],
    ["--results", "x", "--out", "y", "--only", "fig9_nothing"],
    ["--results", "x", "--out", "y", "--png-dpi", "5"],
])
def test_unusable_arguments_exit_2(argv):
    assert main(argv) == EXIT_USAGE
