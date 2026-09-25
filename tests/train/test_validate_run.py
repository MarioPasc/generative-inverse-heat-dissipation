"""Tests of ``ihdm.train.validate_run`` on (copies of) the CPU smoke run."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from ihdm.train.validate_run import (
    check_metrics,
    check_run_dir,
    expectation_from_config,
    read_metrics_strict,
)


@pytest.fixture(scope="module")
def pristine_run(tmp_path_factory, smoke_dataset, train_runner) -> Path:
    """A smoke run of its own (the session's ``smoke_run`` is extended by other modules)."""
    workdir = tmp_path_factory.mktemp("validate_run") / "synthetic_smoke_s1"
    result = train_runner(smoke_dataset.parent, workdir, n_iters=6)
    assert result.returncode == 0, result.stdout + result.stderr
    return workdir


@pytest.fixture
def run_copy(tmp_path, pristine_run) -> Path:
    """A private copy of the pristine run that a test may corrupt."""
    target = tmp_path / "run"
    shutil.copytree(pristine_run, target)
    return target


def _expectation(run: Path, **overrides):
    return expectation_from_config(json.loads((run / "config.json").read_text()), **overrides)


def test_the_smoke_run_passes(run_copy, smoke_dataset):
    assert check_run_dir(run_copy, data_root=smoke_dataset.parent) == []


def test_a_nan_token_is_caught(run_copy):
    path = run_copy / "metrics.jsonl"
    lines = path.read_text().splitlines()
    number = next(i for i, line in enumerate(lines) if '"wall_s": ' in line)
    lines[number] = lines[number].replace('"wall_s": ', '"wall_s": NaN, "x": ', 1)
    path.write_text("\n".join(lines) + "\n")
    _, problems = read_metrics_strict(path)
    assert problems == [f"metrics.jsonl:{number + 1}: non-strict JSON token NaN"]


def test_a_missing_key_and_a_wrong_lr_are_caught(run_copy):
    records, _ = read_metrics_strict(run_copy / "metrics.jsonl")
    train = next(r for r in records if r["kind"] == "train" and r["step"] == 3)
    del train["amp_scale"]
    problems = check_metrics(records, _expectation(run_copy))
    assert any("train 3: keys ['amp_scale']" in p for p in problems)
    problems = check_metrics(read_metrics_strict(run_copy / "metrics.jsonl")[0],
                             _expectation(run_copy, lr=1e-4))
    assert any("lr" in p for p in problems)


def test_a_post_clip_grad_norm_is_caught(run_copy):
    records, _ = read_metrics_strict(run_copy / "metrics.jsonl")
    for r in records:
        if r["kind"] == "train":
            r["grad_norm"] = 1.0
    assert any("does not vary" in p for p in check_metrics(records, _expectation(run_copy)))


def test_cadence_gaps_skips_and_aborts_are_caught(run_copy):
    records, _ = read_metrics_strict(run_copy / "metrics.jsonl")
    records = [r for r in records if not (r["kind"] == "eval" and r["step"] == 3)]
    records.append({"step": 4, "kind": "skip", "loss": None, "n_skipped": 1, "consecutive": 1})
    records.append({"step": 5, "kind": "abort", "reason": "x", "loss": "nan", "n_skipped": 1,
                    "consecutive": 1, "resume_saved": True})
    problems = check_metrics(records, _expectation(run_copy))
    assert any(p.startswith("eval cadence: missing [3]") for p in problems)
    assert any(p.startswith("skip at step 4") for p in problems)
    assert any(p.startswith("abort at step 5") for p in problems)
    allowed = check_metrics([r for r in records if r["kind"] != "abort"],
                            _expectation(run_copy, allow_skips=True))
    assert not any("skip" in p for p in allowed)


def test_missing_artefacts_are_caught(run_copy, smoke_dataset):
    (run_copy / "checkpoints" / "ema_iter_000003.pt").unlink()
    (run_copy / "grids" / "iter_000006.png").unlink()
    problems = check_run_dir(run_copy, data_root=smoke_dataset.parent)
    assert "missing ema_iter_000003.pt" in problems
    assert "missing grids/iter_000006.png" in problems
    (run_copy / "DONE").unlink()
    assert check_run_dir(run_copy)[0].startswith("missing top-level entries ['DONE']")


def test_a_wrong_data_hash_is_caught(run_copy, smoke_dataset, tmp_path):
    other = tmp_path / "other_root"
    shutil.copytree(smoke_dataset, other / smoke_dataset.name)
    meta_path = other / smoke_dataset.name / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["sha256_images"] = "0" * 64
    meta_path.write_text(json.dumps(meta))
    problems = check_run_dir(run_copy, data_root=other)
    assert any("images_sha256" in p for p in problems)
