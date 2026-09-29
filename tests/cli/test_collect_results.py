"""Tests of the ``ihdm.cli.collect_results`` shell (T5.2): arguments, output and exit codes."""

from __future__ import annotations

import json
import shutil

import pytest

from ihdm.cli.collect_results import EXIT_FAIL, EXIT_OK, EXIT_PARTIAL, EXIT_USAGE, main
from tests.analysis.synthetic_eval import SUFFIX, build_campaign, copy_campaign


@pytest.fixture(scope="module")
def pristine(tmp_path_factory):
    return build_campaign(tmp_path_factory.mktemp("t52cli") / "campaign")


@pytest.fixture
def campaign(pristine, tmp_path):
    return copy_campaign(pristine, tmp_path / "campaign")


def test_a_healthy_campaign_exits_zero_and_prints_the_verdict(campaign, tmp_path, capsys):
    out = tmp_path / "results"
    assert main(campaign.cli_args(out)) == EXIT_OK
    text = capsys.readouterr().out
    assert "30 cells" in text and "n_iters 60000 (12 steps 5000..60000); amp fp16" in text
    assert "VERDICT: COMPLETE (0 problems, 0 missing); written (complete)" in text
    assert json.loads((out / "collection.json").read_text())["complete"] is True


def test_the_gate_dir_defaults_to_the_eval_dir_gate_subdir(campaign, tmp_path, capsys):
    assert main(campaign.cli_args(tmp_path / "results")) == EXIT_OK
    assert f"gate dir {campaign.gate_dir};" in capsys.readouterr().out


def test_a_failed_check_exits_one_and_writes_nothing(campaign, tmp_path, capsys):
    campaign.edit_json("ixi_A0_s2", "summary.json", lambda s: s["sampling"].update(amp="bf16"))
    out = tmp_path / "results"
    assert main(campaign.cli_args(out)) == EXIT_FAIL
    assert "C3 FAIL" in capsys.readouterr().out
    assert not out.exists()
    assert main(campaign.cli_args(out, "--allow-missing")) == EXIT_FAIL
    assert json.loads((out / "collection.json").read_text())["verdict"] == "FAIL"


def test_absent_inputs_exit_one_without_and_three_with_allow_missing(campaign, tmp_path,
                                                                     capsys):
    for path in campaign.eval_dir.glob(f"*{SUFFIX}*"):
        path.unlink()
    shutil.rmtree(campaign.gate_dir)
    out = tmp_path / "results"
    assert main(campaign.cli_args(out)) == EXIT_FAIL
    assert not out.exists()
    assert main(campaign.cli_args(out, "--allow-missing")) == EXIT_PARTIAL
    text = capsys.readouterr().out
    assert "C1 MISSING" in text and "C7 PASS" in text
    assert "written (PARTIAL (--allow-missing))" in text
    assert json.loads((out / "collection.json").read_text())["complete"] is False


def test_an_existing_output_or_an_unusable_table_exits_two(campaign, tmp_path, capsys):
    out = tmp_path / "results"
    out.mkdir()
    assert main(campaign.cli_args(out)) == EXIT_USAGE
    assert "exists" in capsys.readouterr().err
    bad = tmp_path / "bad.csv"
    bad.write_text("index,run_id,dataset_id,arm,seed,tier\n0,ixi_A0_s9,ixi,A0,1,1\n")
    args = campaign.cli_args(tmp_path / "other")
    args[args.index("--cells") + 1] = str(bad)
    assert main(args) == EXIT_USAGE
    assert "disagrees with its row" in capsys.readouterr().err


def test_a_run_length_that_is_not_a_multiple_of_the_stride_exits_two(campaign, tmp_path,
                                                                      capsys):
    assert main(campaign.cli_args(tmp_path / "results", "--n-iters", "62500")) == EXIT_USAGE
    assert "multiple of 5000" in capsys.readouterr().err
