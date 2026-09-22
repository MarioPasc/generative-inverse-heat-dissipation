"""The two seed sources of ``04-run-artifacts.md`` §4 (D11)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ihdm.sampling.errors import SamplingError
from ihdm.sampling.seeds import SEED_SLICE, load_seed_images


def test_seed_source_picks_slice_five_per_subject(tiny_dataset):
    """One image per seed subject, the Fig. 1 plane, sorted by subject id."""
    index = pd.read_csv(tiny_dataset / "index.csv")
    splits = json.loads((tiny_dataset / "splits.json").read_text())
    expected_subjects = sorted(index.iloc[splits["seed"]]["subject"].unique().tolist())

    images, idx = load_seed_images(tiny_dataset, "seed", n=40, rng_seed=0)

    assert images.shape == (len(expected_subjects), 32, 32)
    assert images.dtype == np.uint8
    rows = index.iloc[idx]
    assert rows["subject"].tolist() == expected_subjects
    assert rows["slice"].tolist() == [SEED_SLICE] * len(expected_subjects)
    stored = np.load(tiny_dataset / "images.npy")
    np.testing.assert_array_equal(images, stored[idx])


def test_seed_source_caps_at_n(tiny_dataset):
    """Fewer subjects than the dataset holds are returned in subject order."""
    images, idx = load_seed_images(tiny_dataset, "seed", n=1, rng_seed=0)
    all_images, all_idx = load_seed_images(tiny_dataset, "seed", n=40, rng_seed=0)

    assert images.shape[0] == 1
    assert idx[0] == all_idx[0]
    np.testing.assert_array_equal(images[0], all_images[0])


def test_seed_source_is_deterministic(tiny_dataset):
    """The seed source ignores ``rng_seed``: it is a fixed set."""
    _, first = load_seed_images(tiny_dataset, "seed", n=40, rng_seed=0)
    _, second = load_seed_images(tiny_dataset, "seed", n=40, rng_seed=123)
    np.testing.assert_array_equal(first, second)


def test_train_source_draws_distinct_indices(tiny_dataset):
    """``train`` draws distinct training indices, reproducibly from ``rng_seed``."""
    splits = json.loads((tiny_dataset / "splits.json").read_text())

    images, idx = load_seed_images(tiny_dataset, "train", n=8, rng_seed=0)
    _, again = load_seed_images(tiny_dataset, "train", n=8, rng_seed=0)
    _, other = load_seed_images(tiny_dataset, "train", n=8, rng_seed=1)

    assert images.shape == (8, 32, 32)
    assert len(set(idx.tolist())) == 8
    assert set(idx.tolist()) <= set(splits["train"])
    np.testing.assert_array_equal(idx, again)
    assert not np.array_equal(idx, other)


def test_train_source_falls_back_to_replacement(tiny_dataset, caplog):
    """More seeds than the split holds are drawn with replacement (05-metrics.md §4)."""
    splits = json.loads((tiny_dataset / "splits.json").read_text())
    n = len(splits["train"]) + 5

    with caplog.at_level("WARNING"):
        images, idx = load_seed_images(tiny_dataset, "train", n=n, rng_seed=0)

    assert images.shape[0] == n
    assert idx.shape == (n,)
    assert "with replacement" in caplog.text


@pytest.mark.parametrize("n", [0, -3])
def test_non_positive_n_raises(tiny_dataset, n):
    """A non-positive request is a programming error, not an empty result."""
    with pytest.raises(SamplingError, match="must be positive"):
        load_seed_images(tiny_dataset, "train", n=n, rng_seed=0)


def test_unknown_source_raises(tiny_dataset):
    """Only the two frozen sources exist."""
    with pytest.raises(SamplingError, match="unknown seed source"):
        load_seed_images(tiny_dataset, "ref", n=4, rng_seed=0)


def test_missing_dataset_raises(tmp_path):
    """A directory that is not a standard-format dataset is reported as such."""
    with pytest.raises(SamplingError, match="not a standard-format dataset"):
        load_seed_images(tmp_path, "train", n=4, rng_seed=0)


def test_seed_subject_without_slice_five_raises(tmp_path, tiny_dataset):
    """A multi-row seed subject must expose exactly one ``slice == 5`` row."""
    import shutil

    broken = tmp_path / "broken"
    shutil.copytree(tiny_dataset, broken)
    index = pd.read_csv(broken / "index.csv")
    index.loc[index["slice"] == SEED_SLICE, "slice"] = 6
    index.to_csv(broken / "index.csv", index=False)

    with pytest.raises(SamplingError, match=f"slice == {SEED_SLICE}"):
        load_seed_images(broken, "seed", n=40, rng_seed=0)


def test_single_row_subject_is_taken_as_is(tmp_path, tiny_dataset):
    """Photograph datasets have one row per subject and no slice 5; that row is the seed."""
    import shutil

    photos = tmp_path / "photos"
    shutil.copytree(tiny_dataset, photos)
    index = pd.read_csv(photos / "index.csv")
    index["subject"] = [f"img_{i:04d}" for i in range(len(index))]
    index["slice"] = 0
    index.to_csv(photos / "index.csv", index=False)

    images, idx = load_seed_images(photos, "seed", n=40, rng_seed=0)

    splits = json.loads((photos / "splits.json").read_text())
    assert set(idx.tolist()) <= set(splits["seed"])
    assert images.shape[0] == len(splits["seed"])
