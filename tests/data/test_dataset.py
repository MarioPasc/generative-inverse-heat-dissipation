"""Tests for ihdm.data.dataset: NpyImageDataset and make_loaders."""

import ml_collections
import pytest
import torch

from ihdm.data.dataset import NPY_DATASETS, NpyImageDataset, make_loaders
from ihdm.data.errors import DataFormatError


def test_npy_datasets_includes_real_ids_and_synthetic():
    assert NPY_DATASETS == frozenset(
        {"ixi", "oasis1", "lsun_church", "lsun_bedroom", "synthetic"}
    )


def test_npy_image_dataset_item_shape_and_range(synthetic_dataset):
    ds = NpyImageDataset(synthetic_dataset, "train")
    assert len(ds) == 48  # 6 train subjects x 8 slices (train_frac=0.8 of 8 subjects)

    img, extra = ds[0]
    assert img.shape == (1, 32, 32)
    assert img.dtype == torch.float32
    assert extra == {}
    assert float(img.min()) >= 0.0
    assert float(img.max()) <= 1.0


def test_npy_image_dataset_split_sizes(synthetic_dataset):
    train_ds = NpyImageDataset(synthetic_dataset, "train")
    ref_ds = NpyImageDataset(synthetic_dataset, "ref")
    seed_ds = NpyImageDataset(synthetic_dataset, "seed")

    assert len(train_ds) + len(ref_ds) == 64
    assert len(seed_ds) <= len(ref_ds)


def test_npy_image_dataset_rejects_invalid_split(synthetic_dataset):
    with pytest.raises(DataFormatError):
        NpyImageDataset(synthetic_dataset, "bogus")


def _make_config(root) -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.data = data = ml_collections.ConfigDict()
    data.dataset = "synthetic"
    data.root = str(root.parent)
    data.split_train = "train"
    data.split_eval = "ref"
    data.random_flip = False
    data.num_workers = 0
    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 4
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.batch_size = 4
    config.seed = 1
    return config


def test_make_loaders_batch_shape(synthetic_dataset):
    config = _make_config(synthetic_dataset)
    train_loader, eval_loader = make_loaders(config)

    train_batch, _ = next(iter(train_loader))
    assert train_batch.shape == (4, 1, 32, 32)

    eval_batch, _ = next(iter(eval_loader))
    assert eval_batch.shape[1:] == (1, 32, 32)


def test_eval_loader_is_deterministic(synthetic_dataset):
    config = _make_config(synthetic_dataset)
    _, eval_loader_a = make_loaders(config)
    _, eval_loader_b = make_loaders(config)

    a = torch.cat([batch for batch, _ in eval_loader_a])
    b = torch.cat([batch for batch, _ in eval_loader_b])
    torch.testing.assert_close(a, b)
