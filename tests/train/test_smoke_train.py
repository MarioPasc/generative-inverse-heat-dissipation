"""End-to-end CPU smoke run of the trainer, then a resume to a larger ``n_iters``.

Covers the acceptance items of T2.1: the run directory of ``04-run-artifacts.md`` §3 is written
exactly, the ``metrics.jsonl`` events of §3.2 appear at the configured cadences, and resubmitting
the same workdir with a larger ``n_iters`` continues the run instead of restarting it (D10).

This module replaces T0.1's ``tests/train/test_released_smoke.py``: after this ticket ``train.py``
is no longer the released file, so a separate "does the released trainer run" test would exercise
exactly this code path with weaker assertions. Its subprocess pattern is kept in
``tests/train/conftest.py: run_training`` together with the reason for it (the ``DataParallel``
caveat of ``04-run-artifacts.md`` §6).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

EXPECTED_TOP_LEVEL = {
    "manifest.json",
    "config.json",
    "metrics.jsonl",
    "tensorboard",
    "checkpoints",
    "checkpoints-meta",
    "grids",
    "DONE",
}


def _records(workdir: Path) -> list[dict]:
    with (workdir / "metrics.jsonl").open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _steps(records: list[dict], kind: str) -> list[int]:
    return [record["step"] for record in records if record["kind"] == kind]


def test_run_directory_matches_the_contract(smoke_run):
    assert {entry.name for entry in smoke_run.iterdir()} == EXPECTED_TOP_LEVEL
    assert {entry.name for entry in (smoke_run / "checkpoints").iterdir()} == {
        "ema_iter_000003.pt",
        "ema_iter_000006.pt",
        "full_final.pt",
    }
    assert (smoke_run / "checkpoints-meta" / "checkpoint.pth").is_file()
    assert {entry.name for entry in (smoke_run / "grids").iterdir()} == {
        "seeds.npy",
        "iter_000003.png",
        "iter_000006.png",
    }
    assert any((smoke_run / "tensorboard").iterdir())
    assert (smoke_run / "DONE").is_file()


def test_manifest_has_the_contract_keys(smoke_run):
    manifest = json.loads((smoke_run / "manifest.json").read_text())
    expected = {
        "run_id", "dataset_id", "arm", "seed", "git_sha", "git_dirty", "hostname", "gpu",
        "python", "torch", "cuda", "started", "slurm_job_id", "data", "schedule",
        "config_sha256", "n_params", "batch_size", "n_iters",
    }
    assert expected <= manifest.keys()
    assert manifest["run_id"] == "synthetic_smoke_s1"
    assert manifest["seed"] == 1
    assert manifest["n_params"] > 0
    assert manifest["batch_size"] == 4
    assert manifest["n_iters"] == 6
    assert set(manifest["schedule"]) == {"name", "file", "sha256", "values"}
    assert len(manifest["schedule"]["values"]) == 9  # K = 8
    assert manifest["data"]["images_sha256"]
    assert manifest["data"]["meta"]["dataset_id"] == "synthetic"


def test_config_json_is_serialisable_and_complete(smoke_run):
    config = json.loads((smoke_run / "config.json").read_text())
    assert isinstance(config["model"]["blur_schedule"], list)
    assert config["device"] == "cpu"
    assert config["model"]["train_level_max_inclusive"] is True
    assert config["training"]["n_iters"] == 6


def test_metrics_events_and_cadences(smoke_run):
    records = _records(smoke_run)
    assert _steps(records, "train") == [0, 1, 2, 3, 4, 5, 6]  # log_every = 1
    assert _steps(records, "eval") == [0, 3, 6]  # eval_every = 3
    assert _steps(records, "ckpt") == [3, 6]  # ckpt_every = 3, and n_iters
    assert _steps(records, "grid") == [3, 6]  # grid_every = 3, and n_iters
    assert _steps(records, "resume") == []  # a fresh run does not resume
    assert _steps(records, "abort") == []
    assert _steps(records, "skip") == []
    done = [r for r in records if r["kind"] == "done"]
    assert [(r["step"], r["n_skipped"]) for r in done][:1] == [(6, 0)]  # D19: the final event


def test_train_lines_have_the_contract_fields(smoke_run):
    line = next(r for r in _records(smoke_run) if r["kind"] == "train" and r["step"] == 6)
    expected = {
        "step", "kind", "loss", "lr", "it_per_s", "img_per_s", "gpu_mem_peak_gb",
        "grad_norm", "amp_scale", "wall_s", "loss_per_octave",
    }
    assert expected == line.keys()
    assert line["loss"] > 0
    assert line["lr"] > 0
    assert line["it_per_s"] > 0
    assert line["img_per_s"] == line["it_per_s"] * 4
    assert line["gpu_mem_peak_gb"] == 0.0  # CUDA is masked in the subprocess
    assert line["amp_scale"] is None  # the smoke config trains without AMP


def test_grad_norm_is_the_pre_clip_norm(smoke_run):
    # D19: array 1 logged the post-clip norm, identically optim.grad_clip = 1.0. The pre-clip
    # norm varies from line to line and is not capped at the clip value.
    norms = [r["grad_norm"] for r in _records(smoke_run) if r["kind"] == "train"]
    assert all(n is not None and n > 0 for n in norms)
    assert len(set(norms)) == len(norms)
    assert max(norms) > 1.0, f"a 32x32 model at init has gradient norms far above 1: {norms}"


def test_loss_per_octave_is_binned_by_blur_scale(smoke_run):
    octaves = [r["loss_per_octave"] for r in _records(smoke_run) if r["kind"] == "train"]
    names = ["0.5-1", "1-2", "2-4", "4-8", "8-16", "16-32", "32-64", "64-96"]
    for record in octaves:
        assert list(record) == names
        # The smoke schedule stops at sigma_B = 8, so nothing can land above the 8-16 bin.
        assert all(record[name] is None for name in names[5:])
    populated = {name for record in octaves for name, value in record.items() if value is not None}
    assert populated, "no training sample was assigned to an octave bin"
    assert populated <= set(names[:5])


def test_seed_images_are_stored_once(smoke_run):
    seeds = np.load(smoke_run / "grids" / "seeds.npy")
    assert seeds.shape == (8, 32, 32)
    assert seeds.dtype == np.uint8


def test_resume_extends_the_run(smoke_run, smoke_dataset, train_runner):
    saved_step = int(torch.load(
        smoke_run / "checkpoints-meta" / "checkpoint.pth", map_location="cpu")["step"])
    # The loop is range(initial_step, n_iters + 1), so a finished n_iters=6 run has taken 7
    # optimiser steps and the rolling checkpoint stores 7: the next run starts there.
    assert saved_step == 7
    before = len(_records(smoke_run))

    result = train_runner(smoke_dataset.parent, smoke_run, n_iters=9)
    assert result.returncode == 0, result.stdout + result.stderr

    new_records = _records(smoke_run)[before:]
    assert _steps(new_records, "resume") == [saved_step], "exactly one resume event"
    assert min(_steps(new_records, "train")) == saved_step
    assert (smoke_run / "checkpoints" / "ema_iter_000009.pt").is_file()
    assert (smoke_run / "checkpoints" / "ema_iter_000006.pt").is_file()
    assert (smoke_run / "DONE").is_file()


def test_rerun_with_a_smaller_n_iters_exits_cleanly(smoke_run, smoke_dataset, train_runner):
    before = len(_records(smoke_run))
    result = train_runner(smoke_dataset.parent, smoke_run, n_iters=6)
    assert result.returncode == 0, result.stdout + result.stderr
    new_records = _records(smoke_run)[before:]
    assert _steps(new_records, "train") == [], "a finished run must not train again"
    assert (smoke_run / "DONE").is_file()
