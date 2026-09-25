"""Unit tests of ``ihdm.train.recipe``: what a resume may and may not change (D19)."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest
import torch

from configs.spectral import smoke
from ihdm.train.manifest import config_to_json, load_data_meta, write_config_json, write_manifest
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


@pytest.fixture
def data_root(synthetic_dataset):
    """The directory holding the synthetic standard-format dataset (``synthetic/meta.json``)."""
    return synthetic_dataset.parent


@pytest.fixture
def make_config(data_root):
    def build():
        config = smoke.get_config()
        config.data.root = str(data_root)
        return config
    return build


@pytest.fixture
def started_run(tmp_path, make_config):
    """A run directory as the trainer leaves it after its first start."""
    config = make_config()
    run = tmp_path / "run"
    write_manifest(run, config, torch.nn.Linear(2, 2), load_data_meta(config))
    write_config_json(run, config)
    return run


def _set(config, key, value):
    *parents, leaf = key.split(".")
    node = config
    for part in parents:
        node = getattr(node, part)
    setattr(node, leaf, value)


def test_exit_code_is_distinct_from_the_guard():
    assert RECIPE_EXIT_CODE == 4


def test_flatten_keeps_lists_as_leaves():
    assert flatten({"a": {"b": 1, "c": [1, 2]}, "d": "x"}) == {"a.b": 1, "a.c": [1, 2], "d": "x"}


def test_the_recipe_drops_cadences_paths_and_hosts_only(make_config):
    full = flatten(config_to_json(make_config()))
    recipe = recipe_of(config_to_json(make_config()))
    assert set(full) - set(recipe) == RECIPE_EXCLUDED_KEYS & set(full)
    for key in ("model.blur_schedule_file", "data.root", "device", "training.n_iters"):
        assert key not in recipe
    for key in ("optim.lr", "training.batch_size", "model.blur_schedule",
                "model.blur_schedule_sha256", "data.dataset", "sampling.prior_noise",
                "seed", "arm"):
        assert key in recipe


def test_the_hash_is_stable_and_equals_the_manifest_field(started_run, make_config):
    manifest = json.loads((started_run / "manifest.json").read_text())
    assert manifest["recipe_sha256"] == recipe_sha256(make_config())
    assert manifest["data"]["images_sha256"]  # the synthetic dataset records its hash


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
        {"model.blur_schedule_file": "/a/moved/clone/schedules/smoke_log_K8.npy"},
        {"device": torch.device("cuda:1")},
    ],
)
def test_extension_cadence_and_location_changes_are_allowed(started_run, make_config, change):
    config = make_config()
    for key, value in change.items():
        _set(config, key, value)
    check_resume_recipe(started_run, config)  # does not raise


def test_a_moved_data_root_with_the_same_images_is_allowed(started_run, make_config, data_root,
                                                           tmp_path):
    moved = tmp_path / "moved_root"
    shutil.copytree(data_root / "synthetic", moved / "synthetic")
    config = make_config()
    config.data.root = str(moved)
    check_resume_recipe(started_run, config)  # does not raise


def test_a_data_root_with_other_images_is_refused(started_run, make_config, data_root, tmp_path):
    other = tmp_path / "other_root"
    shutil.copytree(data_root / "synthetic", other / "synthetic")
    meta_path = other / "synthetic" / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["sha256_images"] = "f" * 64
    meta_path.write_text(json.dumps(meta))
    config = make_config()
    config.data.root = str(other)
    with pytest.raises(RecipeMismatchError, match="images_sha256"):
        check_resume_recipe(started_run, config)


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
        ("data.num_workers", 3),
        ("sampling.prior_noise", True),
        ("seed", 2),
        ("arm", "A3"),
    ],
)
def test_recipe_changes_are_refused_with_the_key_named(started_run, make_config, key, value):
    config = make_config()
    _set(config, key, value)
    with pytest.raises(RecipeMismatchError, match=key.replace(".", r"\.")):
        check_resume_recipe(started_run, config)


def test_a_changed_schedule_array_is_refused(started_run, make_config):
    config = make_config()
    config.model.blur_schedule = np.asarray(config.model.blur_schedule) * 1.01
    with pytest.raises(RecipeMismatchError, match=r"model\.blur_schedule"):
        check_resume_recipe(started_run, config)


def test_check_reads_and_writes_nothing_else(started_run, make_config):
    before = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in started_run.iterdir()}
    config = make_config()
    config.optim.lr = 1e-4
    with pytest.raises(RecipeMismatchError):
        check_resume_recipe(started_run, config)
    after = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in started_run.iterdir()}
    assert before == after


def test_a_pre_d19_manifest_is_checked_against_config_json(started_run, make_config):
    manifest = json.loads((started_run / "manifest.json").read_text())
    del manifest["recipe_sha256"]
    (started_run / "manifest.json").write_text(json.dumps(manifest))
    check_resume_recipe(started_run, make_config())
    config = make_config()
    config.optim.lr = 1e-4
    with pytest.raises(RecipeMismatchError, match=r"optim\.lr: 0\.0002 -> 0\.0001"):
        check_resume_recipe(started_run, config)


def test_a_stale_manifest_hash_defers_to_config_json(started_run, make_config):
    # A hash written by an older definition of the recipe, with config.json showing no differing
    # recipe key, must not block the resume; a real change is still refused.
    manifest = json.loads((started_run / "manifest.json").read_text())
    manifest["recipe_sha256"] = "0" * 64
    (started_run / "manifest.json").write_text(json.dumps(manifest))
    check_resume_recipe(started_run, make_config())
    config = make_config()
    config.training.batch_size = 8
    with pytest.raises(RecipeMismatchError, match=r"training\.batch_size"):
        check_resume_recipe(started_run, config)
    (started_run / "config.json").unlink()
    with pytest.raises(RecipeMismatchError, match="no config.json"):
        check_resume_recipe(started_run, make_config())


def test_a_checkpoint_without_manifest_or_config_is_refused(tmp_path, make_config):
    with pytest.raises(RecipeMismatchError, match="cannot be established"):
        check_resume_recipe(tmp_path, make_config())


def test_the_diff_names_absent_keys_and_truncates_long_values():
    lines = recipe_diff({"a": 1, "b": list(range(100))}, {"b": list(range(99)), "c": 2})
    assert lines[0] == "a: 1 -> <absent>"
    assert lines[1].startswith("b: [0, 1, 2") and "..." in lines[1]
    assert lines[2] == "c: <absent> -> 2"
    assert recipe_diff({"a": [1.0, 2.0]}, {"a": [1.0, 2.0]}) == []
