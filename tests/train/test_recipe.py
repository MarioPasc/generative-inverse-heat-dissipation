"""Unit tests of ``ihdm.train.recipe``: what a resume may and may not change (D19)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from configs.spectral import smoke
from ihdm.train.manifest import config_to_json, write_config_json, write_manifest
from ihdm.train.recipe import (
    RECIPE_EXCLUDED_KEYS,
    RECIPE_EXIT_CODE,
    RecipeMismatchError,
    check_resume_recipe,
    flatten,
    recipe_diff,
    recipe_of,
    recipe_sha256,
)


def _config():
    config = smoke.get_config()
    config.data.root = "/data/root"
    return config


def _started_run(tmp_path):
    """A run directory as the trainer leaves it after its first start."""
    config = _config()
    write_manifest(tmp_path, config, torch.nn.Linear(2, 2), None)
    write_config_json(tmp_path, config)
    return tmp_path


def test_exit_code_is_distinct_from_the_guard():
    assert RECIPE_EXIT_CODE == 4


def test_flatten_keeps_lists_as_leaves():
    assert flatten({"a": {"b": 1, "c": [1, 2]}, "d": "x"}) == {"a.b": 1, "a.c": [1, 2], "d": "x"}


def test_the_recipe_drops_only_the_iteration_count_the_cadences_and_run_id():
    full = flatten(config_to_json(_config()))
    recipe = recipe_of(config_to_json(_config()))
    assert set(full) - set(recipe) == RECIPE_EXCLUDED_KEYS & set(full)
    for key in ("optim.lr", "training.batch_size", "model.blur_schedule",
                "model.blur_schedule_sha256", "data.dataset", "data.root", "sampling.prior_noise",
                "seed", "arm", "device"):
        assert key in recipe


def test_the_hash_is_stable_and_equals_the_manifest_field(tmp_path):
    run = _started_run(tmp_path)
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["recipe_sha256"] == recipe_sha256(_config())


@pytest.mark.parametrize(
    "change",
    [
        {"training.n_iters": 60000},
        {"training.resume_every": 50},
        {"training.log_every": 10},
        {"training.eval_every": 50},
        {"training.ckpt_every": 100},
        {"training.grid_every": 100},
        {"training.snapshot_freq_for_preemption": 50},
        {"training.n_iters": 400, "training.resume_every": 50, "training.log_every": 10},
    ],
)
def test_extension_and_cadence_changes_are_allowed(tmp_path, change):
    run = _started_run(tmp_path)
    config = _config()
    for key, value in change.items():
        section, leaf = key.split(".")
        setattr(getattr(config, section), leaf, value)
    check_resume_recipe(run, config)  # does not raise


def _set(config, key, value):
    *parents, leaf = key.split(".")
    node = config
    for part in parents:
        node = getattr(node, part)
    setattr(node, leaf, value)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("optim.lr", 1e-4),
        ("optim.warmup", 1000),
        ("optim.grad_clip", 0.5),
        ("optim.automatic_mp", True),
        ("training.batch_size", 8),
        ("training.max_skips", 5),
        ("model.K", 9),
        ("model.blur_schedule_sha256", "0" * 64),
        ("model.blur_sigma_max", 24.0),
        ("model.ema_rate", 0.9999),
        ("data.dataset", "other"),
        ("data.root", "/elsewhere"),
        ("sampling.prior_noise", True),
        ("seed", 2),
        ("arm", "A3"),
    ],
)
def test_recipe_changes_are_refused_with_the_key_named(tmp_path, key, value):
    run = _started_run(tmp_path)
    config = _config()
    _set(config, key, value)
    with pytest.raises(RecipeMismatchError, match=key.replace(".", r"\.")):
        check_resume_recipe(run, config)


def test_a_changed_schedule_array_is_refused(tmp_path):
    run = _started_run(tmp_path)
    config = _config()
    config.model.blur_schedule = np.asarray(config.model.blur_schedule) * 1.01
    with pytest.raises(RecipeMismatchError, match=r"model\.blur_schedule"):
        check_resume_recipe(run, config)


def test_check_reads_and_writes_nothing_else(tmp_path):
    run = _started_run(tmp_path)
    before = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in run.iterdir()}
    config = _config()
    config.optim.lr = 1e-4
    with pytest.raises(RecipeMismatchError):
        check_resume_recipe(run, config)
    after = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in run.iterdir()}
    assert before == after


def test_a_pre_d19_manifest_is_checked_against_config_json(tmp_path):
    run = _started_run(tmp_path)
    manifest = json.loads((run / "manifest.json").read_text())
    del manifest["recipe_sha256"]
    (run / "manifest.json").write_text(json.dumps(manifest))
    check_resume_recipe(run, _config())
    config = _config()
    config.optim.lr = 1e-4
    with pytest.raises(RecipeMismatchError, match=r"optim\.lr: 0\.0002 -> 0\.0001"):
        check_resume_recipe(run, config)


def test_a_checkpoint_without_manifest_or_config_is_refused(tmp_path):
    with pytest.raises(RecipeMismatchError, match="cannot be established"):
        check_resume_recipe(tmp_path, _config())


def test_the_diff_names_absent_keys_and_truncates_long_values():
    lines = recipe_diff({"a": 1, "b": list(range(100))}, {"b": list(range(99)), "c": 2})
    assert lines[0] == "a: 1 -> <absent>"
    assert lines[1].startswith("b: [0, 1, 2") and "..." in lines[1]
    assert lines[2] == "c: <absent> -> 2"
    assert recipe_diff({"a": [1.0, 2.0]}, {"a": [1.0, 2.0]}) == []
