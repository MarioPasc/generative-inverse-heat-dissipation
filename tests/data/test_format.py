"""Tests for ihdm.data.format: write_dataset, read_dataset, validate_dataset, split_by_subject."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from ihdm.data.errors import DataFormatError
from ihdm.data.format import (
    INDEX_COLUMNS,
    read_dataset,
    split_by_subject,
    validate_dataset,
)
from ihdm.paths import data_root


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


# --- T1.5: split_by_subject stratification -----------------------------------------------


def test_split_by_subject_strata_none_matches_the_synthetic_fixture(synthetic_dataset):
    """``strata=None`` must reproduce the splits.json a fixture built before stratification."""
    _, index, splits, _ = read_dataset(synthetic_dataset)
    subjects = index["subject"].astype(str).tolist()

    recomputed = split_by_subject(subjects, rng_seed=splits["rng_seed"], n_seed=2)

    for key in (
        "train", "ref", "seed",
        "train_subjects", "ref_subjects", "seed_subjects",
        "rule", "rng_seed",
    ):
        assert recomputed[key] == splits[key], key


_ARCHIVED_ROOT = data_root()
requires_archived_datasets = pytest.mark.skipif(
    not (_ARCHIVED_ROOT / "oasis1" / "splits.json").is_file()
    or not (_ARCHIVED_ROOT / "lsun_church" / "splits.json").is_file(),
    reason=f"archived oasis1/lsun_church datasets not present at {_ARCHIVED_ROOT}",
)


@requires_archived_datasets
@pytest.mark.parametrize("dataset_id", ["oasis1", "lsun_church"])
def test_split_by_subject_strata_none_matches_archived_datasets(dataset_id):
    """Regression (T1.5): ``strata=None`` still reproduces datasets built before stratification.

    OASIS-1 stays unstratified after T1.5; ``lsun_church`` never had a notion of strata at
    all. Both are re-run through today's ``split_by_subject`` and must match their archived
    ``splits.json`` byte-for-byte (field by field, since JSON key order is not semantic).
    """
    root = _ARCHIVED_ROOT / dataset_id
    archived = json.loads((root / "splits.json").read_text())
    index = pd.read_csv(root / "index.csv")
    subjects = index["subject"].astype(str).tolist()

    recomputed = split_by_subject(subjects, rng_seed=archived["rng_seed"])

    for key in (
        "train", "ref", "seed",
        "train_subjects", "ref_subjects", "seed_subjects",
        "rule", "rng_seed",
    ):
        assert recomputed[key] == archived[key], f"{dataset_id}: {key} differs from the archive"


def _site_like_subjects(
    sizes: tuple[tuple[str, int], ...] = (("Guys", 205), ("HH", 144), ("IOP", 51)),
    seed: int = 0,
) -> dict[str, str]:
    """Build a synthetic subject -> stratum-label map shaped like IXI's real site sizes."""
    rng = np.random.default_rng(seed)
    n_total = sum(n for _, n in sizes)
    subjects = rng.permutation([f"S{i:04d}" for i in range(n_total)]).tolist()
    strata: dict[str, str] = {}
    idx = 0
    for label, n in sizes:
        for subject in subjects[idx : idx + n]:
            strata[subject] = label
        idx += n
    return strata


def test_split_by_subject_stratified_counts_within_one_of_proportional():
    strata = _site_like_subjects()
    subjects = sorted(strata)
    splits = split_by_subject(subjects, rng_seed=2026, train_frac=0.8, strata=strata)

    n_by_label: dict[str, int] = {}
    for label in strata.values():
        n_by_label[label] = n_by_label.get(label, 0) + 1

    train_counts = splits["strata_mix"]["train"]
    ref_counts = splits["strata_mix"]["ref"]
    for label, n in n_by_label.items():
        assert abs(train_counts.get(label, 0) - 0.8 * n) <= 1
        assert abs(ref_counts.get(label, 0) - 0.2 * n) <= 1
    assert sum(train_counts.values()) == len(splits["train_subjects"]) == 320
    assert sum(ref_counts.values()) == len(splits["ref_subjects"]) == 80


def test_split_by_subject_stratified_seed_allocation_proportional_and_sums_to_n_seed():
    strata = _site_like_subjects()
    subjects = sorted(strata)
    splits = split_by_subject(subjects, rng_seed=2026, train_frac=0.8, strata=strata, n_seed=40)

    seed_counts = splits["strata_mix"]["seed"]
    ref_counts = splits["strata_mix"]["ref"]
    assert sum(seed_counts.values()) == 40
    n_ref_total = sum(ref_counts.values())
    for label, n_ref in ref_counts.items():
        expected = 40 * n_ref / n_ref_total
        assert abs(seed_counts.get(label, 0) - expected) <= 1
    assert set(splits["seed_subjects"]) <= set(splits["ref_subjects"])


def test_split_by_subject_stratified_disjoint_and_covering():
    strata = _site_like_subjects()
    unique_subjects = sorted(strata)
    n_slices = 3
    subjects = [s for s in unique_subjects for _ in range(n_slices)]  # MRI-like repeats
    splits = split_by_subject(subjects, rng_seed=2026, train_frac=0.8, strata=strata)

    train, ref = set(splits["train"]), set(splits["ref"])
    assert train.isdisjoint(ref)
    assert train | ref == set(range(len(subjects)))
    assert set(splits["train_subjects"]).isdisjoint(splits["ref_subjects"])
    assert set(splits["train_subjects"]) | set(splits["ref_subjects"]) == set(unique_subjects)
    assert set(splits["seed_subjects"]) <= set(splits["ref_subjects"])

    by_subject: dict[str, set[str]] = {}
    for position, subject in enumerate(subjects):
        by_subject.setdefault(subject, set()).add("train" if position in train else "ref")
    assert all(len(sides) == 1 for sides in by_subject.values())


def test_split_by_subject_stratified_is_deterministic():
    strata = _site_like_subjects()
    subjects = sorted(strata)
    a = split_by_subject(subjects, rng_seed=2026, train_frac=0.8, strata=strata)
    b = split_by_subject(subjects, rng_seed=2026, train_frac=0.8, strata=strata)
    assert a == b


def test_split_by_subject_stratified_carries_strata_and_custom_name():
    strata = _site_like_subjects()
    subjects = sorted(strata)
    splits = split_by_subject(
        subjects, rng_seed=2026, train_frac=0.8, strata=strata, strata_name="acquisition site"
    )
    assert splits["strata"] == strata
    assert "stratified by acquisition site" in splits["rule"]
    assert set(splits["strata_mix"]) == {"train", "ref", "seed"}


def test_split_by_subject_stratified_single_subject_stratum_does_not_crash():
    strata = {"A1": "solo"} | {f"B{i}": "big" for i in range(9)}
    subjects = sorted(strata)

    splits = split_by_subject(subjects, rng_seed=2026, train_frac=0.8, strata=strata, n_seed=2)

    assert set(splits["train_subjects"]) | set(splits["ref_subjects"]) == set(subjects)
    assert set(splits["train_subjects"]).isdisjoint(splits["ref_subjects"])
    solo_train = "A1" in splits["train_subjects"]
    solo_ref = "A1" in splits["ref_subjects"]
    assert solo_train != solo_ref, "the lone 'solo' subject must land on exactly one side"


def test_split_by_subject_stratified_missing_label_raises():
    strata = {"S00": "site_a"}
    with pytest.raises(DataFormatError):
        split_by_subject(["S00", "S01"], rng_seed=2026, strata=strata)
