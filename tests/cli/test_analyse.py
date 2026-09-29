"""Tests of the ``ihdm.cli.analyse`` shell (T6.1): arguments, output files and exit codes."""

from __future__ import annotations

import json
import shutil

import pytest

from ihdm.cli.analyse import EXIT_FAIL, EXIT_OK, EXIT_PARTIAL, EXIT_USAGE, main
from tests.analysis.synthetic_eval import build_campaign, copy_campaign
from tests.analysis.test_tables import TIER_3, collect_planted, drop_runs


@pytest.fixture(scope="module")
def pristine(tmp_path_factory):
    return build_campaign(tmp_path_factory.mktemp("t61cli") / "campaign")


@pytest.fixture(scope="module")
def complete(pristine, tmp_path_factory):
    return collect_planted(pristine, tmp_path_factory.mktemp("t61cli_full") / "results")


@pytest.fixture(scope="module")
def partial(pristine, tmp_path_factory):
    root = tmp_path_factory.mktemp("t61cli_partial")
    campaign = copy_campaign(pristine, root / "campaign")
    drop_runs(campaign, TIER_3)
    return collect_planted(campaign, root / "results", allow_missing=True)


def test_a_complete_collection_exits_zero_and_writes_every_table(complete, tmp_path, capsys):
    out = tmp_path / "tables"
    assert main(["--results", str(complete), "--out", str(out)]) == EXIT_OK
    text = capsys.readouterr().out
    assert "VERDICT: COMPLETE" in text and "Table 3   t3_interaction" in text
    names = json.loads((out / "tables.json").read_text())["tables"]
    for name in names:
        assert (out / f"{name}.md").is_file() and (out / f"{name}.tex").is_file()


def test_a_partial_collection_exits_three_and_says_which_tables_are_incomplete(partial, tmp_path,
                                                                                capsys):
    out = tmp_path / "tables"
    assert main(["--results", str(partial), "--out", str(out)]) == EXIT_PARTIAL
    text = capsys.readouterr().out
    assert "t6_transfer                incomplete: 14/20 runs" in text
    assert "t3_interaction             complete (12/12 runs)" in text
    assert "VERDICT: INCOMPLETE" in text


def test_a_rerun_overwrites_the_tables_and_keeps_the_readme(complete, partial, tmp_path):
    out = tmp_path / "tables"
    out.mkdir()
    (out / "README.md").write_text("kept\n")
    assert main(["--results", str(partial), "--out", str(out)]) == EXIT_PARTIAL
    assert main(["--results", str(complete), "--out", str(out)]) == EXIT_OK
    assert (out / "README.md").read_text() == "kept\n"
    assert "PARTIAL" not in (out / "t1a_cells_fidelity.md").read_text()


def test_a_folder_that_is_not_a_collection_exits_two(tmp_path, capsys):
    assert main(["--results", str(tmp_path), "--out", str(tmp_path / "t")]) == EXIT_USAGE
    assert "not a results folder" in capsys.readouterr().err
    assert not (tmp_path / "t").exists()


def test_a_failed_collection_exits_one_and_writes_nothing(complete, tmp_path, capsys):
    copy = tmp_path / "results"
    shutil.copytree(complete, copy)
    collection = json.loads((copy / "collection.json").read_text())
    collection["verdict"] = "FAIL"
    (copy / "collection.json").write_text(json.dumps(collection))
    assert main(["--results", str(copy), "--out", str(tmp_path / "t")]) == EXIT_FAIL
    assert "verdict is FAIL" in capsys.readouterr().err
    assert not (tmp_path / "t").exists()
