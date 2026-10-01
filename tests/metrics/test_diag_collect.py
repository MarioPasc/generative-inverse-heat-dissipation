"""Tests of ``slurm/diag_eval/collect.py`` (T7.2) on a synthetic two-level delta sweep."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ihdm.metrics.run_eval import EvalRequest, evaluate_run
from slurm.diag_eval import collect
from tests.metrics.test_run_eval import _build_dataset, _config, _write_run

DELTAS = (0.0125, 0.02)
SMALL: dict = {
    "ckpts": "750", "n_lsd": 8, "n_seeds": 2, "n_per_seed": 3, "sample_batch": 8,
    "skip_inception": True, "device": "cpu", "final_from_lsd": True,
}


@pytest.fixture(scope="module")
def swept(tmp_path_factory) -> Path:
    base = tmp_path_factory.mktemp("t72_collect")
    dataset = _build_dataset(base / "data")
    config = _config(dataset)
    workdir = _write_run(base / "runs" / config.run_id, config)
    for delta in DELTAS:
        evaluate_run(EvalRequest(run=workdir, delta=delta, **SMALL))
    return workdir


def test_collect_task_returns_one_row_per_delta(swept):
    record = collect.collect_task(swept, "ixi_A0_s1", "off", list(DELTAS), 750, 0.01, {"x": 1})
    assert [row["delta"] for row in record["rows"]] == list(DELTAS)
    assert [row["delta_over_sigma"] for row in record["rows"]] == pytest.approx([1.25, 2.0])
    for row in record["rows"]:
        assert row["final_set"] == "lsd"
        assert row["n_lsd_samples"] == 8
        assert row["inception"] is None
        assert {"lsd", "variance_ratio", "M", "diversity_pix", "inherited_measured"} <= set(row)
    assert record["meta"] == {"x": 1}


def test_collect_task_refuses_a_missing_delta(swept):
    with pytest.raises(collect.CollectError):
        collect.collect_task(swept, "ixi_A0_s1", "off", [0.03], 750, 0.01, {})


def test_merge_checks_the_anchor(swept, tmp_path):
    record = collect.collect_task(swept, "ixi_A0_s1", "off", list(DELTAS), 750, 0.01, {})
    lsd = record["rows"][0]["lsd"]
    path = tmp_path / "task.json"
    path.write_text(json.dumps(record))
    merged = collect.merge([path], {"job": "x"})
    anchor = merged["anchor"]["runs"]["ixi_A0_s1"]
    assert anchor["reproduced"] == lsd
    assert anchor["ok"] is (abs(lsd - collect.ANCHORS["ixi_A0_s1"]) <= 0.003)
    assert merged["job"] == "x"


def test_the_cli_writes_the_task_json(swept, tmp_path):
    out = tmp_path / "t.json"
    assert collect.main([
        "task", "--shadow", str(swept), "--run-id", "r", "--amp", "off",
        "--deltas", "0.0125:0.02", "--step", "750", "--out", str(out),
    ]) == 0
    assert len(json.loads(out.read_text())["rows"]) == 2
