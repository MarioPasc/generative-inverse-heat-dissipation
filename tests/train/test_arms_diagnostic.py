"""The M7 diagnostic cells of the config factory (T7.1) and the 30-cell regression against the base.

The regression is the contract that matters most: adding the two diagnostic datasets must not
move a single byte of the configs of the 30 production runs. It compares, for every
``EXPERIMENT_CELLS`` run, the factory of this checkout with the factory of the base commit
``48363e2`` loaded from git, through ``ihdm.train.manifest.config_sha256`` (the hash the trainer
records in ``manifest.json`` and in every EMA checkpoint). Both factories read the real frozen
``schedules/`` directory, so the hash also covers the schedule arrays they load.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from configs.spectral.arms import (
    ARMS,
    DATASETS,
    DIAGNOSTIC_CELLS,
    DIAGNOSTIC_DATASETS,
    EXPERIMENT_CELLS,
    PHOTOGRAPH_DATASETS,
    get_config,
)
from ihdm.data.dataset import NPY_DATASETS
from ihdm.paths import data_root, repo_root
from ihdm.train.manifest import config_sha256, config_to_json

BASE_COMMIT = "48363e27d180eb4cad5b3f83f18a58f18f9bcdba"
SCHEDULES = repo_root() / "schedules"

EXPERIMENT_RUNS = [
    (dataset, arm, seed) for dataset, arm, seeds in EXPERIMENT_CELLS for seed in seeds
]

# The only config paths in which a diagnostic cell may differ from lsun_church,A0 (one factor).
_IDENTITY = {"dataset_id", "run_id", "data.dataset"}
_SCHEDULE = {
    "model.blur_sigma_max",
    "model.blur_schedule_name",
    "model.blur_schedule_file",
    "model.blur_schedule_sha256",
    "model.blur_schedule",
}
ALLOWED_DIFF = {
    "lsun_church_r128": _IDENTITY | {"data.image_size"} | _SCHEDULE,
    "lsun_church_n32k": _IDENTITY,
}


@pytest.fixture
def real_schedules(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point both factories at the committed ``schedules/`` and a fixed data root."""
    monkeypatch.delenv("IHDM_SCHEDULES_DIR", raising=False)
    monkeypatch.setenv("IHDM_DATA_ROOT", str(tmp_path / "data"))
    return SCHEDULES


@pytest.fixture(scope="module")
def base_arms(tmp_path_factory: pytest.TempPathFactory) -> ModuleType:
    """``configs/spectral/arms.py`` as it was at the base commit, imported under its own name."""
    try:
        out = subprocess.run(
            ["git", "show", f"{BASE_COMMIT}:configs/spectral/arms.py"], cwd=repo_root(),
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        pytest.skip("git is not available")
    if out.returncode != 0:
        pytest.skip("the base commit is not reachable from this checkout")
    path = tmp_path_factory.mktemp("base_arms") / "arms_base_48363e2.py"
    path.write_text(out.stdout)
    spec = importlib.util.spec_from_file_location("arms_base_48363e2", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve the module of ArmSpec through here
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def _flatten(tree: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested JSON dict into ``{"a.b": value}``."""
    flat: dict[str, Any] = {}
    for key, value in tree.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{path}."))
        else:
            flat[path] = value
    return flat


def test_experiment_table_is_the_base_table(base_arms: ModuleType) -> None:
    assert not hasattr(base_arms, "DIAGNOSTIC_CELLS")  # really the pre-T7.1 factory
    assert EXPERIMENT_CELLS == base_arms.EXPERIMENT_CELLS
    assert DATASETS == base_arms.DATASETS
    assert ARMS == base_arms.ARMS
    assert len(EXPERIMENT_RUNS) == 30


@pytest.mark.parametrize(("dataset", "arm", "seed"), EXPERIMENT_RUNS)
def test_every_production_config_hashes_like_the_base_commit(
    real_schedules: Path, base_arms: ModuleType, dataset: str, arm: str, seed: int
) -> None:
    spec = f"{dataset},{arm},seed={seed}"
    new, old = get_config(spec), base_arms.get_config(spec)
    assert config_to_json(new) == config_to_json(old)
    assert config_sha256(new) == config_sha256(old)


@pytest.mark.parametrize(("dataset", "arm", "seed"), EXPERIMENT_RUNS)
def test_production_configs_survive_the_trainer_overrides(
    real_schedules: Path, base_arms: ModuleType, dataset: str, arm: str, seed: int
) -> None:
    """The array passes ``--config.seed`` and ``--config.training.n_iters`` after the factory."""
    new, old = get_config(f"{dataset},{arm}"), base_arms.get_config(f"{dataset},{arm}")
    for config in (new, old):
        config.seed = seed
        config.training.n_iters = 60000
        config.run_id = f"{config.data.dataset}_{config.arm}_s{seed}"
    assert config_sha256(new) == config_sha256(old)


def test_diagnostic_tables() -> None:
    assert DIAGNOSTIC_DATASETS == ("lsun_church_r128", "lsun_church_n32k")
    assert DIAGNOSTIC_CELLS == (("lsun_church_r128", "A0", (1,)), ("lsun_church_n32k", "A0", (1,)))
    assert not set(DIAGNOSTIC_DATASETS) & set(DATASETS)
    assert not {cell[0] for cell in EXPERIMENT_CELLS} & set(DIAGNOSTIC_DATASETS)
    assert set(DIAGNOSTIC_DATASETS) <= set(PHOTOGRAPH_DATASETS)
    assert set(DIAGNOSTIC_DATASETS) <= NPY_DATASETS


@pytest.mark.parametrize(
    ("dataset", "image_size", "sigma_max", "schedule"),
    [("lsun_church_r128", 128, 64.0, "log_W2_128"), ("lsun_church_n32k", 192, 96.0, "log_W2")],
)
def test_diagnostic_cell_builds(
    real_schedules: Path, dataset: str, image_size: int, sigma_max: float, schedule: str
) -> None:
    config = get_config(f"{dataset},A0")
    assert config.run_id == f"{dataset}_A0_s1"
    assert config.data.dataset == dataset and config.dataset_id == dataset
    assert config.data.image_size == image_size and isinstance(config.data.image_size, int)
    assert config.model.blur_sigma_max == sigma_max
    assert config.model.blur_schedule_name == schedule
    assert Path(config.model.blur_schedule_file) == (real_schedules / f"{schedule}.npy").resolve()
    values = np.load(real_schedules / f"{schedule}.npy")
    np.testing.assert_array_equal(config.model.blur_schedule, values)
    assert config.model.blur_schedule[-1] == sigma_max and config.model.blur_schedule[1] == 0.5


@pytest.mark.parametrize("dataset", DIAGNOSTIC_DATASETS)
def test_diagnostic_cell_differs_from_the_a0_baseline_in_one_factor_only(
    real_schedules: Path, dataset: str
) -> None:
    base = _flatten(config_to_json(get_config("lsun_church,A0")))
    diag = _flatten(config_to_json(get_config(f"{dataset},A0")))
    assert set(base) == set(diag)
    differing = {key for key in base if base[key] != diag[key]}
    assert differing <= ALLOWED_DIFF[dataset]
    assert {"dataset_id", "run_id", "data.dataset"} <= differing
    # The A0 recipe, spelled out (T7.1 frozen contracts).
    assert diag["training.batch_size"] == 16 and diag["optim.lr"] == 1e-4
    assert diag["optim.warmup"] == 1000 and diag["optim.grad_clip"] == 1.0
    assert diag["optim.automatic_mp"] is True and diag["model.ema_rate"] == 0.999
    assert diag["model.K"] == 200 and diag["model.sigma"] == 0.01
    assert diag["sampling.delta_factor"] == 1.25 and diag["model.blur_sigma_min"] == 0.5
    assert diag["training.ckpt_every"] == 2500 and diag["training.grid_every"] == 2500


@pytest.mark.parametrize("dataset", DIAGNOSTIC_DATASETS)
@pytest.mark.parametrize("arm", [arm for arm in ARMS if arm != "A0"])
def test_diagnostic_datasets_refuse_every_other_arm(
    real_schedules: Path, dataset: str, arm: str
) -> None:
    with pytest.raises(ValueError, match="accepts only the arms"):
        get_config(f"{dataset},{arm}")


def test_seed_override_names_the_diagnostic_run(real_schedules: Path) -> None:
    assert get_config("lsun_church_r128,A0,seed=2").run_id == "lsun_church_r128_A0_s2"


def test_unknown_dataset_still_raises(real_schedules: Path) -> None:
    with pytest.raises(ValueError, match="unknown dataset_id"):
        get_config("lsun_church_r64,A0")


def test_diagnostic_ids_route_to_the_npy_backend() -> None:
    source = (repo_root() / "scripts" / "datasets.py").read_text()
    assert source.index("config.data.dataset in NPY_DATASETS") < source.index(
        'config.data.dataset == "lsun_church"'
    )


@pytest.mark.integration
@pytest.mark.parametrize("dataset", DIAGNOSTIC_DATASETS)
def test_config_image_size_matches_the_dataset_on_disk(
    monkeypatch: pytest.MonkeyPatch, dataset: str
) -> None:
    """Needs the built dataset under ``IHDM_DATA_ROOT``; skipped elsewhere."""
    root = data_root() / dataset
    if not (root / "meta.json").is_file():
        pytest.skip(f"{root} not present")
    monkeypatch.delenv("IHDM_SCHEDULES_DIR", raising=False)
    meta = json.loads((root / "meta.json").read_text())
    images = np.load(root / "images.npy", mmap_mode="r")
    side = get_config(f"{dataset},A0").data.image_size
    assert side == meta["image_size"] == images.shape[1] == images.shape[2]
