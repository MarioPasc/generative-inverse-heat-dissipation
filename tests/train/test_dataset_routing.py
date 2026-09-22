"""Every arm config reaches the standard-format loader, not a released torchvision branch.

Regression test for a defect found while running the acceptance demo of T2.1: the released
``scripts/datasets.get_dataset`` has an ``elif config.data.dataset == "lsun_church":`` branch that
opens an LSUN lmdb store under ``data/lsun``. ``lsun_church`` is also one of our four dataset ids,
so that branch shadowed T0.1's ``NPY_DATASETS`` hook and every churches run died with
``ModuleNotFoundError: No module named 'lmdb'`` before the first step. Twelve of the thirty runs
depend on this routing.
"""

from __future__ import annotations

import pytest
import torch

from configs.spectral.arms import get_config
from ihdm.data.dataset import NPY_DATASETS, NpyImageDataset
from scripts.datasets import get_dataset

ARM_OF_DATASET = {
    "ixi": "A3",
    "oasis1": "A0",
    "lsun_church": "A0",
    "lsun_bedroom": "A3",
}


def _loaders(tmp_path, monkeypatch, dataset_factory, dataset_id, arm):
    dataset_factory(tmp_path, dataset_id)
    monkeypatch.setenv("IHDM_DATA_ROOT", str(tmp_path))
    config = get_config(f"{dataset_id},{arm}")
    config.training.batch_size = 4
    config.eval.batch_size = 4
    config.data.num_workers = 0
    return config, get_dataset(config, uniform_dequantization=False)


@pytest.mark.parametrize(("dataset_id", "arm"), sorted(ARM_OF_DATASET.items()))
def test_arm_configs_route_to_the_standard_format_backend(
    tmp_path, monkeypatch, schedules_dir, dataset_factory, dataset_id, arm
):
    config, (train_loader, eval_loader) = _loaders(
        tmp_path, monkeypatch, dataset_factory, dataset_id, arm)

    assert config.data.dataset == dataset_id
    assert isinstance(train_loader.dataset, NpyImageDataset)
    assert isinstance(eval_loader.dataset, NpyImageDataset)
    assert train_loader.dataset.split == "train"
    assert eval_loader.dataset.split == "ref"


def test_lsun_church_yields_real_batches(tmp_path, monkeypatch, schedules_dir, dataset_factory):
    _, (train_loader, _) = _loaders(
        tmp_path, monkeypatch, dataset_factory, "lsun_church", "A0")
    batch = next(iter(train_loader))[0]
    assert batch.shape == (4, 1, 32, 32)  # the fixture is 32x32; the real dataset is 192x192
    assert batch.dtype == torch.float32
    assert 0.0 <= float(batch.min()) and float(batch.max()) <= 1.0


def test_the_npy_branch_is_checked_before_the_released_lsun_church_branch():
    import inspect

    import scripts.datasets as released

    source = inspect.getsource(released.get_dataset)
    assert "lsun_church" in NPY_DATASETS
    assert source.index("config.data.dataset in NPY_DATASETS") < source.index(
        'config.data.dataset == "lsun_church"'
    ), "the released lsun_church branch would shadow the standard-format backend again"
