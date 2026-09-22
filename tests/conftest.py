"""Shared pytest fixtures: the synthetic standard-format dataset used across the suite."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ihdm.data.format import DatasetMeta, split_by_subject, write_dataset

N_SUBJECTS = 8
N_SLICES = 8


def _build_synthetic_dataset(root: Path, rng_seed: int = 0) -> Path:
    """Write the 64-image, 32x32, 8-subjects-by-8-slices synthetic fixture at ``root``."""
    rng = np.random.default_rng(rng_seed)
    n = N_SUBJECTS * N_SLICES
    images = rng.integers(0, 256, size=(n, 32, 32), dtype=np.uint8)

    subjects = [f"SUBJ{s:03d}" for s in range(N_SUBJECTS) for _ in range(N_SLICES)]
    slices = [sl for _ in range(N_SUBJECTS) for sl in range(N_SLICES)]

    splits = split_by_subject(subjects, rng_seed=2026, train_frac=0.8, n_seed=2)
    train_set, seed_set = set(splits["train"]), set(splits["seed"])
    fine_split = [
        "train" if i in train_set else ("seed" if i in seed_set else "ref") for i in range(n)
    ]

    index = pd.DataFrame(
        {
            "idx": np.arange(n),
            "subject": subjects,
            "slice": slices,
            "z_mm": [float(sl) for sl in slices],
            "source": [
                f"synthetic/{subj}/{sl}" for subj, sl in zip(subjects, slices, strict=True)
            ],
            "split": fine_split,
        }
    )

    meta = DatasetMeta(
        dataset_id="synthetic",
        n_images=n,
        image_size=32,
        dtype="uint8",
        pipeline="tests.conftest",
        pipeline_version="1.0",
        git_sha="test",
        created="2026-09-22T00:00:00",
        raw_root=str(root),
        parameters={"note": "synthetic fixture for T0.1, MRI-like index"},
        counts={
            "subjects_total": N_SUBJECTS,
            "subjects_used": N_SUBJECTS,
            "slices_per_subject": N_SLICES,
        },
    )

    write_dataset(root, images, index, splits, meta)
    return root


@pytest.fixture
def synthetic_dataset(tmp_path_factory) -> Path:
    """A valid 64-image standard-format dataset at ``<tmp>/synthetic``.

    8 "subjects" x 8 "slices" (MRI-like index), split with ``n_seed=2``.

    Returns
    -------
    Path
        The dataset root directory (named ``synthetic``).
    """
    root = tmp_path_factory.mktemp("synthetic_dataset") / "synthetic"
    return _build_synthetic_dataset(root)
