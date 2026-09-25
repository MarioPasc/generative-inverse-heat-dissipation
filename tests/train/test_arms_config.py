"""Contract of the arm config factory (``docs/SPECIFICATIONS/04-run-artifacts.md`` §2)."""

from __future__ import annotations

import numpy as np
import pytest

from configs.spectral.arms import ARMS, DATASETS, EXPERIMENT_CELLS, ArmSpec, get_config

# The transfer pair runs only the paper arm and the brain arm (00-overview.md §1).
TRANSFER_DATASETS = ("oasis1", "lsun_bedroom")
PHOTOGRAPHS = ("lsun_church", "lsun_bedroom")

VALID_COMBINATIONS = [
    (dataset, arm)
    for dataset in DATASETS
    for arm in ARMS
    if not (arm == "A2p" and dataset not in PHOTOGRAPHS)
    and not (dataset in TRANSFER_DATASETS and arm not in ("A0", "A3"))
]

INVALID_COMBINATIONS = [
    ("ixi", "A2p"),
    ("oasis1", "A2p"),
    ("oasis1", "A1"),
    ("oasis1", "A2"),
    ("lsun_bedroom", "A1"),
    ("lsun_bedroom", "A2"),
    ("lsun_bedroom", "A2p"),
    ("cifar10", "A0"),
    ("ixi", "A4"),
]



def _log_schedule(sigma_max: float, k: int = 200) -> np.ndarray:
    """The released log schedule, as ``tests/train/conftest.py`` writes it."""
    levels = np.exp(np.linspace(np.log(0.5), np.log(sigma_max), k))
    return np.concatenate([[0.0], levels]).astype(np.float64)


ARM_EXPECTATIONS = {
    "A0": ("log_W2", 96.0),
    "A1": ("log_W8", 24.0),
    "A2": ("ixi_W2", 96.0),
    "A3": ("ixi_W8", 24.0),
    "A2p": ("lsun_church_W2", 96.0),
}


@pytest.mark.parametrize(("dataset", "arm"), VALID_COMBINATIONS)
def test_every_valid_combination_builds(schedules_dir, dataset, arm):
    config = get_config(f"{dataset},{arm}")
    assert config.data.dataset == dataset
    assert config.arm == arm
    assert config.run_id == f"{dataset}_{arm}_s1"


@pytest.mark.parametrize(("dataset", "arm"), INVALID_COMBINATIONS)
def test_invalid_combinations_raise(schedules_dir, dataset, arm):
    with pytest.raises(ValueError):
        get_config(f"{dataset},{arm}")


@pytest.mark.parametrize("spec", ["ixi", "", "ixi,A3,batch_size"])
def test_malformed_specs_raise(schedules_dir, spec):
    with pytest.raises(ValueError):
        get_config(spec)


@pytest.mark.parametrize(("dataset", "arm", "seeds"), EXPERIMENT_CELLS)
def test_every_experiment_cell_builds(schedules_dir, dataset, arm, seeds):
    for seed in seeds:
        config = get_config(f"{dataset},{arm},seed={seed}")
        assert config.seed == seed
        assert config.run_id == f"{dataset}_{arm}_s{seed}"


def test_experiment_cells_cover_thirty_runs():
    assert sum(len(seeds) for _, _, seeds in EXPERIMENT_CELLS) == 30


@pytest.mark.parametrize("arm", list(ARM_EXPECTATIONS))
def test_schedule_name_and_endpoints_match_the_arm(schedules_dir, arm):
    name, sigma_max = ARM_EXPECTATIONS[arm]
    config = get_config(f"lsun_church,{arm}")
    schedule = np.asarray(config.model.blur_schedule)

    assert config.model.blur_schedule_name == name
    assert config.model.blur_schedule_file == str((schedules_dir / f"{name}.npy").resolve())
    assert config.model.blur_sigma_max == sigma_max
    assert schedule.dtype == np.float64
    assert schedule.shape == (config.model.K + 1,)
    assert schedule[0] == 0.0
    np.testing.assert_allclose(schedule[1], 0.5)
    np.testing.assert_allclose(schedule[config.model.K], sigma_max)
    assert np.all(np.diff(schedule) > 0)
    assert len(config.model.blur_schedule_sha256) == 64


def test_both_paper_versus_code_fixes_are_on(schedules_dir):
    config = get_config("ixi,A3")
    assert config.model.train_level_max_inclusive is True
    assert config.sampling.prior_noise is True
    assert config.sampling.delta_factor == 1.25


def test_frozen_recipe_values(schedules_dir):
    config = get_config("ixi,A3")
    assert config.training.batch_size == 16
    assert config.training.n_iters == 40000
    assert config.training.ckpt_every == 2500
    assert config.training.resume_every == 500
    assert config.training.log_every == 50
    assert config.training.eval_every == 500
    assert config.training.grid_every == 2500
    assert config.eval.batch_size == config.training.batch_size
    assert config.optim.lr == pytest.approx(1e-4)  # D19, recipe v2 (v1 was 2e-4)
    assert config.training.max_consecutive_skips == 10  # D19 skip policy
    assert config.training.max_skips == 100
    assert config.optim.warmup == 1000
    assert config.optim.grad_clip == 1.0
    assert config.optim.automatic_mp is True
    assert config.model.ema_rate == 0.999
    assert config.model.sigma == 0.01
    assert config.model.K == 200
    assert config.model.blur_sigma_min == 0.5
    assert config.model.model_channels == 128
    assert tuple(config.model.channel_mult) == (1, 2, 2, 2)
    assert config.model.num_res_blocks == 4
    assert tuple(config.model.attention_levels) == (2, 3)
    assert config.model.dropout == 0.1
    assert config.data.image_size == 192
    assert config.data.num_channels == 1
    assert config.data.split_train == "train"
    assert config.data.split_eval == "ref"


def test_released_cadence_keys_kept_under_the_old_names(schedules_dir):
    config = get_config("ixi,A0")
    assert config.training.snapshot_freq == config.training.ckpt_every
    assert config.training.snapshot_freq_for_preemption == config.training.resume_every
    assert config.training.log_freq == config.training.log_every
    assert config.training.eval_freq == config.training.eval_every
    assert config.training.sampling_freq == config.training.grid_every


def test_overrides_are_applied_and_typed(schedules_dir):
    config = get_config("lsun_church,A2p,seed=3,training.batch_size=32,optim.automatic_mp=false")
    assert config.seed == 3
    assert config.training.batch_size == 32
    assert config.eval.batch_size == 32
    assert config.optim.automatic_mp is False
    assert config.run_id == "lsun_church_A2p_s3"


def test_missing_schedule_names_the_path_and_the_ticket(tmp_path, monkeypatch):
    monkeypatch.setenv("IHDM_SCHEDULES_DIR", str(tmp_path / "empty"))
    with pytest.raises(FileNotFoundError) as excinfo:
        get_config("ixi,A3")
    message = str(excinfo.value)
    assert "ixi_W8.npy" in message
    assert "T1.3" in message


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (lambda s: s.astype(np.float32), "float64"),
        (lambda s: s[:-1], "shape"),
        (lambda s: np.concatenate([[0.1], s[1:]]), "s[0]"),
        (lambda s: np.concatenate([s[:1], [0.25], s[2:]]), "s[1]"),
        (lambda s: np.concatenate([s[:-1], [s[-2]]]), "increasing"),
        (lambda s: np.concatenate([s[:-1], [s[-1] * 2]]), "s[K]"),
    ],
)
def test_schedule_validation_rejects_broken_arrays(tmp_path, monkeypatch, mutate, needle):
    target = tmp_path / "schedules"
    target.mkdir()
    np.save(target / "ixi_W8.npy", mutate(_log_schedule(24.0)))
    monkeypatch.setenv("IHDM_SCHEDULES_DIR", str(target))
    with pytest.raises(ValueError) as excinfo:
        get_config("ixi,A3")
    assert needle in str(excinfo.value)


def test_armspec_parses_the_spec_string():
    spec = ArmSpec.parse("lsun_church,A2p,seed=2,training.n_iters=30000")
    assert spec == ArmSpec(
        dataset_id="lsun_church",
        arm="A2p",
        seed=2,
        overrides=(("training.n_iters", "30000"),),
    )
