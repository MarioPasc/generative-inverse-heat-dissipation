"""Seed images for the offline sampler: the two sources of ``04-run-artifacts.md`` §4 (D11).

``train``
    the fidelity/LSD endpoint of ``05-metrics.md`` §2 and the memorisation endpoint of §4 start
    from images of the training split, one sample per seed (the paper's Alg. 2);
``seed``
    the diversity (§3) and inherited-band (§5) endpoints start from the 40 held-out seed subjects,
    one image per subject, and draw 50 samples from the same prior state.

For MRI datasets a seed subject contributes several slices and the Fig. 1 plane (``slice == 5``)
is the one used; for photographs every image is its own subject and the single row is taken as is.

A third source, ``file``, reads the dataset indices from a ``.npy`` array instead of drawing them:
it is what the common-random-numbers rule of ``05-metrics.md`` §8a (D17) needs, since every
checkpoint and every arm of a dataset must be seeded from one frozen list. Every source validates
its indices against the split it claims to draw from through :func:`assert_in_split`, after T4.2
found a recorded sample set whose seed indices were not in ``train``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from ihdm.data.errors import DataFormatError
from ihdm.data.format import read_dataset
from ihdm.sampling.errors import SamplingError

__all__ = ["SEED_SLICE", "assert_in_split", "load_seed_images"]

logger = logging.getLogger(__name__)

#: The slice index that represents a seed subject in the MRI datasets (the Fig. 1 plane).
SEED_SLICE: int = 5

SeedSource = Literal["train", "seed", "file"]


def assert_in_split(idx: np.ndarray, split: np.ndarray, name: str = "split") -> np.ndarray:
    """Return ``idx`` after checking that every index belongs to ``split``.

    The guard of D17: a sample set is only reproducible if its seeds are the frozen list of the
    split it claims, and T4.2 found a recorded set of 64 "training" seeds of which 18 were rows of
    the ``ref`` split. Every seed source of this module passes through here.

    Parameters
    ----------
    idx : numpy.ndarray
        Dataset indices, any integer dtype, any shape; flattened for the check.
    split : numpy.ndarray
        The dataset indices that make up the declared split.
    name : str
        Name of the split, used in the error message.

    Returns
    -------
    numpy.ndarray
        ``idx`` as ``int64``, unchanged.

    Raises
    ------
    SamplingError
        If ``split`` is empty, or any index of ``idx`` is outside it.
    """
    indices = np.asarray(idx, dtype=np.int64)
    allowed = np.asarray(split, dtype=np.int64)
    if allowed.size == 0:
        raise SamplingError(f"the {name} split is empty; no index can belong to it")
    outside = np.setdiff1d(np.unique(indices), np.unique(allowed))
    if outside.size:
        raise SamplingError(
            f"{outside.size} of {np.unique(indices).size} distinct seed indices are outside the "
            f"{name} split (first offenders: {outside[:8].tolist()})"
        )
    return indices


def _read(dataset_root: Path) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Read a standard-format dataset, translating its errors into :class:`SamplingError`."""
    try:
        images, index, splits, _ = read_dataset(Path(dataset_root), mmap=True)
    except DataFormatError as exc:
        raise SamplingError(f"{dataset_root}: not a standard-format dataset ({exc})") from exc
    return images, index, splits


def _train_indices(splits: dict[str, Any], n: int, rng_seed: int) -> np.ndarray:
    """Draw ``n`` training indices with ``np.random.default_rng(rng_seed)``.

    Distinct indices are drawn whenever the split is large enough. ``05-metrics.md`` §4 asks for
    5 000 training-seeded samples while the photograph training splits hold 3 200 images and
    specifies "seeds drawn with replacement", so the draw falls back to replacement rather than
    failing when ``n`` exceeds the split.
    """
    pool = np.asarray(splits.get("train", []), dtype=np.int64)
    if pool.size == 0:
        raise SamplingError("the dataset has an empty train split")
    if n <= 0:
        raise SamplingError(f"n must be positive, got {n}")
    replace = n > pool.size
    if replace:
        logger.warning(
            "requested %d training seeds from a split of %d images; drawing with replacement",
            n,
            pool.size,
        )
    rng = np.random.default_rng(rng_seed)
    drawn = np.asarray(rng.choice(pool, size=n, replace=replace), dtype=np.int64)
    return assert_in_split(drawn, pool, "train")


def _file_indices(splits: dict[str, Any], idx_file: Path, n: int, split: str) -> np.ndarray:
    """Return the first ``n`` indices of ``idx_file``, checked against ``splits[split]``."""
    path = Path(idx_file)
    try:
        stored = np.asarray(np.load(path))
    except (OSError, ValueError) as exc:
        raise SamplingError(f"{path}: cannot be read as a .npy index array ({exc})") from exc
    if stored.ndim != 1 or not np.issubdtype(stored.dtype, np.integer):
        raise SamplingError(
            f"{path}: expected a 1-D integer array of dataset indices, "
            f"got {stored.ndim}-D {stored.dtype}"
        )
    if n <= 0:
        raise SamplingError(f"n must be positive, got {n}")
    if stored.size < n:
        raise SamplingError(f"{path} holds {stored.size} indices, fewer than the {n} requested")
    pool = np.asarray(splits.get(split, []), dtype=np.int64)
    return assert_in_split(stored[:n], pool, split)


def _seed_row(rows: pd.DataFrame, subject: str) -> int:
    """Return the dataset index representing one seed subject."""
    if len(rows) == 1:
        return int(rows["idx"].iloc[0])
    if "slice" not in rows.columns:
        raise SamplingError(f"seed subject {subject!r} has {len(rows)} rows and no 'slice' column")
    chosen = rows[rows["slice"] == SEED_SLICE]
    if len(chosen) != 1:
        raise SamplingError(
            f"seed subject {subject!r} has {len(chosen)} rows with slice == {SEED_SLICE}, "
            "expected exactly one"
        )
    return int(chosen["idx"].iloc[0])


def _seed_subject_indices(index: pd.DataFrame, splits: dict[str, Any], n: int) -> np.ndarray:
    """Return one index per seed subject, sorted by subject id, at most ``n`` of them."""
    seed_rows = np.asarray(splits.get("seed", []), dtype=np.int64)
    if seed_rows.size == 0:
        raise SamplingError("the dataset has an empty seed split")
    if n <= 0:
        raise SamplingError(f"n must be positive, got {n}")
    frame = index.iloc[seed_rows]
    subjects = sorted(frame["subject"].unique().tolist())
    if n > len(subjects):
        logger.warning(
            "requested %d seed subjects but the dataset holds %d; returning all of them",
            n,
            len(subjects),
        )
    chosen = [_seed_row(frame[frame["subject"] == subject], subject) for subject in subjects[:n]]
    return np.asarray(chosen, dtype=np.int64)


def load_seed_images(
    dataset_root: Path,
    source: SeedSource,
    n: int,
    rng_seed: int = 0,
    *,
    idx_file: Path | None = None,
    split: str = "train",
) -> tuple[np.ndarray, np.ndarray]:
    """Return the seed images of one source and the dataset indices they came from.

    Parameters
    ----------
    dataset_root : Path
        A standard-format dataset directory (``images.npy``, ``index.csv``, ``splits.json``,
        ``meta.json``).
    source : {"train", "seed", "file"}
        ``"train"``: ``n`` indices of the training split, drawn with
        ``np.random.default_rng(rng_seed)``, distinct unless ``n`` exceeds the split;
        ``"seed"``: one image per held-out seed subject, sorted by subject id, the ``slice == 5``
        row when the subject has several (MRI) and the single row otherwise (photographs);
        ``"file"``: the first ``n`` dataset indices of the ``.npy`` array at ``idx_file``, which
        is how the frozen evaluation seed lists of ``05-metrics.md`` §8a are passed in.
    n : int
        How many seeds to return. For ``"seed"`` the result is capped at the number of seed
        subjects (all 40 of the frozen datasets when ``n >= 40``).
    rng_seed : int
        Seed of the training draw; ignored by ``"seed"`` and ``"file"``, which are deterministic.
    idx_file : Path or None
        Required by ``source="file"``: a 1-D integer ``.npy`` of dataset indices.
    split : str
        The split ``source="file"`` declares; its indices are checked against it with
        :func:`assert_in_split`. Ignored by the other sources, which declare their own.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(images uint8 (m, H, W), idx int64 (m,))`` with ``m <= n``.

    Raises
    ------
    SamplingError
        If the directory is not a standard-format dataset, the requested split is empty, ``n`` is
        not positive, ``source`` is unknown, ``source="file"`` is given without ``idx_file`` or
        with indices outside ``split``, or a seed subject does not expose exactly one
        ``slice == 5`` row.
    """
    images, index, splits = _read(dataset_root)
    if source == "train":
        chosen = _train_indices(splits, n, rng_seed)
    elif source == "seed":
        chosen = _seed_subject_indices(index, splits, n)
    elif source == "file":
        if idx_file is None:
            raise SamplingError("source='file' requires idx_file")
        chosen = _file_indices(splits, Path(idx_file), n, split)
    else:
        raise SamplingError(
            f"unknown seed source {source!r}; expected 'train', 'seed' or 'file'"
        )
    return np.asarray(images[chosen], dtype=np.uint8), chosen
