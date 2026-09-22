"""Tests for ihdm.data.format: write_dataset, read_dataset, validate_dataset, split_by_subject."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from ihdm.data.format import (
    INDEX_COLUMNS,
    read_dataset,
    split_by_subject,
    validate_dataset,
)


def test_valid_fixture_has_no_violations(synthetic_dataset):
    assert validate_dataset(synthetic_dataset) == []


def test_read_dataset_roundtrip(synthetic_dataset):
    images, index, splits, meta = read_dataset(synthetic_dataset)

    assert images.dtype == np.uint8
    assert images.shape == (64, 32, 32)
    assert len(index) == 64
    assert list(index.columns) == list(INDEX_COLUMNS)
    assert set(splits.keys()) >= {"train", "ref", "seed"}
    assert meta.dataset_id == "synthetic"
    assert meta.n_images == 64
    assert meta.sha256_images  # stamped by write_dataset, non-empty


def test_read_dataset_missing_file_raises(tmp_path):
    from ihdm.data.errors import DataFormatError

    with pytest.raises(DataFormatError):
        read_dataset(tmp_path / "does_not_exist")


def test_split_by_subject_partitions_and_seed_subset():
    subjects = [f"S{i:02d}" for i in range(10) for _ in range(4)]  # 10 subjects x 4 rows
    splits = split_by_subject(subjects, rng_seed=2026, train_frac=0.8, n_seed=2)
    n = len(subjects)

    assert sorted(splits["train"] + splits["ref"]) == list(range(n))
    assert set(splits["train"]).isdisjoint(splits["ref"])
    assert set(splits["seed"]).issubset(set(splits["ref"]))
    assert len(splits["train_subjects"]) == 8
    assert len(splits["ref_subjects"]) == 2
    assert len(splits["seed_subjects"]) == 2
    assert splits["rng_seed"] == 2026


def test_split_by_subject_is_reproducible():
    subjects = [f"S{i:02d}" for i in range(10) for _ in range(4)]
    a = split_by_subject(subjects, rng_seed=2026)
    b = split_by_subject(subjects, rng_seed=2026)
    assert a == b


@pytest.mark.parametrize(
    "corrupt, expected_substring",
    [
        ("wrong_dtype", "dtype"),
        ("subject_straddles_split", "both 'train' and 'ref'"),
        ("missing_column", "columns"),
        ("bad_hash", "sha256_images"),
        ("constant_image", "entirely constant"),
        ("seed_not_subset_of_ref", "not a subset of 'ref'"),
    ],
)
def test_validate_dataset_catches_corruption(synthetic_dataset, corrupt, expected_substring):
    root = synthetic_dataset

    if corrupt == "wrong_dtype":
        images = np.load(root / "images.npy")
        np.save(root / "images.npy", images.astype(np.float32))

    elif corrupt == "subject_straddles_split":
        index = pd.read_csv(root / "index.csv")
        splits = json.loads((root / "splits.json").read_text())
        train_subject = index.loc[index["split"] == "train", "subject"].iloc[0]
        rows = index.loc[index["subject"] == train_subject, "idx"].tolist()
        moved = int(rows[1])
        splits["train"] = sorted(set(splits["train"]) - {moved})
        splits["ref"] = sorted(set(splits["ref"]) | {moved})
        (root / "splits.json").write_text(json.dumps(splits))

    elif corrupt == "missing_column":
        index = pd.read_csv(root / "index.csv")
        index = index.drop(columns=["z_mm"])
        index.to_csv(root / "index.csv", index=False)

    elif corrupt == "bad_hash":
        meta = json.loads((root / "meta.json").read_text())
        meta["sha256_images"] = "0" * 64
        (root / "meta.json").write_text(json.dumps(meta))

    elif corrupt == "constant_image":
        images = np.load(root / "images.npy")
        images[0] = 100
        np.save(root / "images.npy", images)
        # Recompute the hash so only the "constant image" violation is isolated.
        meta = json.loads((root / "meta.json").read_text())
        meta["sha256_images"] = hashlib.sha256((root / "images.npy").read_bytes()).hexdigest()
        (root / "meta.json").write_text(json.dumps(meta))

    elif corrupt == "seed_not_subset_of_ref":
        splits = json.loads((root / "splits.json").read_text())
        splits["seed"] = sorted(set(splits["seed"]) | {splits["train"][0]})
        (root / "splits.json").write_text(json.dumps(splits))

    violations = validate_dataset(root)
    assert violations, "expected at least one violation"
    assert any(expected_substring in v for v in violations), violations
