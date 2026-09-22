"""Standard on-disk dataset format: writer, reader, validator, subject split.

Frozen contract: ``docs/SPECIFICATIONS/03-data-format.md``. A dataset directory holds
``images.npy`` (uint8, ``(N, H, W)``), ``index.csv``, ``splits.json`` and ``meta.json``.
This module is the sole owner of the contract; every producer (MRI, photograph
preprocessing) and every consumer (:mod:`ihdm.data.dataset`, the spectral profile,
the metrics harness) reads and writes through it.

Parameters
----------
Not applicable at module level; see each function below.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ihdm.data.errors import DataFormatError

__all__ = [
    "DataFormatError",
    "DatasetMeta",
    "write_dataset",
    "read_dataset",
    "validate_dataset",
    "split_by_subject",
]

logger = logging.getLogger(__name__)

INDEX_COLUMNS: tuple[str, ...] = ("idx", "subject", "slice", "z_mm", "source", "split")
SPLIT_NAMES: tuple[str, ...] = ("train", "ref", "seed")
_REQUIRED_META_KEYS: frozenset[str] = frozenset(
    {
        "dataset_id",
        "n_images",
        "image_size",
        "dtype",
        "pipeline",
        "pipeline_version",
        "git_sha",
        "created",
        "raw_root",
        "parameters",
        "counts",
        "sha256_images",
    }
)
_REQUIRED_SPLIT_KEYS: frozenset[str] = frozenset(
    {
        "train",
        "ref",
        "seed",
        "train_subjects",
        "ref_subjects",
        "seed_subjects",
        "rule",
        "rng_seed",
    }
)


@dataclass(frozen=True)
class DatasetMeta:
    """Provenance and parameters of a standard-format dataset (mirrors ``meta.json``).

    Parameters
    ----------
    dataset_id : str
        One of ``ixi``, ``oasis1``, ``lsun_church``, ``lsun_bedroom``, or any other
        folder name the backend is pointed at (e.g. ``synthetic`` for tests).
    n_images : int
        Number of rows in ``images.npy`` / ``index.csv``.
    image_size : int
        Side length of the square images, in pixels.
    dtype : str
        NumPy dtype name of ``images.npy`` (``"uint8"``).
    pipeline : str
        Dotted path of the producing pipeline module.
    pipeline_version : str
        Version string of the producing pipeline.
    git_sha : str
        Git commit of the repository at production time.
    created : str
        ISO-8601 creation timestamp.
    raw_root : str
        Root directory of the raw source data.
    parameters : dict[str, Any]
        Every parameter of the pipeline (registration settings, z band, window
        rule, intensity rule, ...).
    counts : dict[str, Any]
        Subject and slice counts, failures, padding fraction, etc.
    sha256_images : str
        Hex SHA-256 of the bytes of ``images.npy``. Computed and overwritten by
        :func:`write_dataset`; an empty string before the dataset is written.
    """

    dataset_id: str
    n_images: int
    image_size: int
    dtype: str
    pipeline: str
    pipeline_version: str
    git_sha: str
    created: str
    raw_root: str
    parameters: dict[str, Any]
    counts: dict[str, Any]
    sha256_images: str = ""

    def to_json(self) -> dict[str, Any]:
        """Return the ``meta.json``-shaped dict for this metadata.

        Returns
        -------
        dict[str, Any]
            A plain, JSON-serialisable dict with exactly the required keys.
        """
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> DatasetMeta:
        """Build a :class:`DatasetMeta` from a parsed ``meta.json`` dict.

        Parameters
        ----------
        data : dict[str, Any]
            Parsed JSON object; must contain every required key.

        Returns
        -------
        DatasetMeta
            The parsed metadata.

        Raises
        ------
        DataFormatError
            If a required key is missing.
        """
        missing = _REQUIRED_META_KEYS - data.keys()
        if missing:
            raise DataFormatError(f"meta.json: missing key(s) {sorted(missing)}")
        return cls(**{k: data[k] for k in _REQUIRED_META_KEYS})


def split_by_subject(
    subjects: list[str],
    rng_seed: int = 2026,
    train_frac: float = 0.8,
    n_seed: int = 40,
) -> dict[str, Any]:
    """Partition images into train/ref/seed splits by subject.

    Every image of a subject inherits that subject's split (a subject never
    straddles ``train`` and ``ref``). For photographs, each image is its own
    subject, so this reduces to a per-image split.

    Parameters
    ----------
    subjects : list[str]
        One entry per image, in ``index.csv`` row order (length ``N``); repeats
        for images that share a subject (e.g. MRI slices).
    rng_seed : int
        Seed of ``np.random.default_rng`` driving both the train/ref partition
        and the seed-subject draw, so the result is reproducible from the
        subject list alone.
    train_frac : float
        Fraction of unique subjects assigned to ``train``.
    n_seed : int
        Number of ``ref`` subjects drawn as ``seed`` subjects (clipped to the
        number of ``ref`` subjects if fewer are available).

    Returns
    -------
    dict[str, Any]
        The ``splits.json``-shaped dict: ``train``, ``ref``, ``seed`` (sorted
        image-index lists), ``train_subjects``, ``ref_subjects``, ``seed_subjects``
        (sorted subject-id lists), ``rule`` and ``rng_seed``.
    """
    unique_subjects = sorted(set(subjects))
    n_subjects = len(unique_subjects)
    rng = np.random.default_rng(rng_seed)
    permuted = rng.permutation(unique_subjects)

    n_train = round(n_subjects * train_frac)
    train_subjects = set(permuted[:n_train].tolist())
    ref_subjects_list = permuted[n_train:].tolist()
    ref_subjects = set(ref_subjects_list)

    n_seed_eff = min(n_seed, len(ref_subjects_list))
    seed_draw = rng.choice(len(ref_subjects_list), size=n_seed_eff, replace=False)
    seed_subjects = {ref_subjects_list[i] for i in seed_draw.tolist()}

    train_idx = [i for i, s in enumerate(subjects) if s in train_subjects]
    ref_idx = [i for i, s in enumerate(subjects) if s in ref_subjects]
    seed_idx = [i for i, s in enumerate(subjects) if s in seed_subjects]

    return {
        "train": train_idx,
        "ref": ref_idx,
        "seed": seed_idx,
        "train_subjects": sorted(train_subjects),
        "ref_subjects": sorted(ref_subjects),
        "seed_subjects": sorted(seed_subjects),
        "rule": (
            f"{train_frac * 100:.0f}/{(1 - train_frac) * 100:.0f} by subject, "
            f"seed = {n_seed} subjects drawn from ref with rng seed {rng_seed}"
        ),
        "rng_seed": rng_seed,
    }


def write_dataset(
    root: Path,
    images: np.ndarray,
    index: pd.DataFrame,
    splits: dict[str, Any],
    meta: DatasetMeta,
) -> None:
    """Write a dataset in the standard format.

    ``images.npy`` is written atomically (temp file, then rename) and its
    SHA-256 is computed after the write and stamped into ``meta.json`` as
    ``sha256_images``, overriding whatever ``meta.sha256_images`` held.

    Parameters
    ----------
    root : Path
        Destination directory; created if missing.
    images : np.ndarray
        ``uint8`` array of shape ``(N, H, W)``.
    index : pd.DataFrame
        ``N`` rows with exactly the columns in :data:`INDEX_COLUMNS`, in order.
    splits : dict[str, Any]
        The ``splits.json``-shaped dict, e.g. as returned by :func:`split_by_subject`.
    meta : DatasetMeta
        Provenance and parameters; ``sha256_images`` is recomputed on write.

    Raises
    ------
    DataFormatError
        If ``images``, ``index`` or ``meta.n_images`` are mutually inconsistent.
    """
    if images.dtype != np.uint8:
        raise DataFormatError(f"images: expected dtype uint8, got {images.dtype}")
    if images.ndim != 3:
        raise DataFormatError(f"images: expected shape (N, H, W), got {images.shape}")
    n = images.shape[0]
    if len(index) != n:
        raise DataFormatError(f"index has {len(index)} rows, images has N={n}")
    if list(index.columns) != list(INDEX_COLUMNS):
        raise DataFormatError(
            f"index columns {list(index.columns)} do not match {list(INDEX_COLUMNS)}"
        )
    if meta.n_images != n:
        raise DataFormatError(f"meta.n_images={meta.n_images} does not match N={n}")

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "qc").mkdir(exist_ok=True)

    images_path = root / "images.npy"
    # np.save appends ".npy" unless the given path already ends with it, so the
    # temp file must itself end in ".npy" or the atomic rename below targets a
    # file that was never written.
    tmp_path = root / "images.tmp.npy"
    np.save(tmp_path, images)
    tmp_path.replace(images_path)
    sha256_images = hashlib.sha256(images_path.read_bytes()).hexdigest()

    index.to_csv(root / "index.csv", index=False)

    (root / "splits.json").write_text(json.dumps(_to_jsonable(splits), indent=2))

    meta_final = dataclasses.replace(meta, sha256_images=sha256_images)
    (root / "meta.json").write_text(json.dumps(meta_final.to_json(), indent=2))


def read_dataset(
    root: Path, mmap: bool = True
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any], DatasetMeta]:
    """Read a dataset written by :func:`write_dataset`.

    Parameters
    ----------
    root : Path
        Dataset directory.
    mmap : bool
        Memory-map ``images.npy`` instead of loading it fully into RAM.

    Returns
    -------
    tuple[np.ndarray, pd.DataFrame, dict[str, Any], DatasetMeta]
        ``(images, index, splits, meta)``.

    Raises
    ------
    DataFormatError
        If any of the four files is missing.
    """
    root = Path(root)
    for name in ("images.npy", "index.csv", "splits.json", "meta.json"):
        if not (root / name).is_file():
            raise DataFormatError(f"missing file: {root / name}")

    images = np.load(root / "images.npy", mmap_mode="r" if mmap else None)
    index = pd.read_csv(root / "index.csv")
    splits = json.loads((root / "splits.json").read_text())
    meta = DatasetMeta.from_json(json.loads((root / "meta.json").read_text()))
    return images, index, splits, meta


def validate_dataset(root: Path) -> list[str]:
    """Check a dataset directory against the standard-format contract.

    Checks, at least: file presence; ``images.npy`` dtype/shape/``N``;
    ``index.csv`` columns, row count and ``idx == row``; ``splits.json``
    partition and subset properties and subject-level consistency (every
    subject's images in one split); ``meta.json`` required keys; the
    ``sha256_images`` hash; no image entirely constant; the global mean
    intensity in ``(5, 200)``; ``z_mm`` finite for every row or ``nan`` for
    every row.

    Parameters
    ----------
    root : Path
        Dataset directory to validate.

    Returns
    -------
    list[str]
        Violation messages, each naming the offending file/key/idx. Empty
        means the dataset is valid.
    """
    root = Path(root)
    violations: list[str] = []

    required_files = ["images.npy", "index.csv", "splits.json", "meta.json"]
    missing = [f for f in required_files if not (root / f).is_file()]
    violations.extend(f"missing file: {f}" for f in missing)
    if missing:
        return violations

    meta_raw = _load_json_safely(root / "meta.json", "meta.json", violations)
    if meta_raw is not None:
        missing_meta = _REQUIRED_META_KEYS - meta_raw.keys()
        violations.extend(f"meta.json: missing key '{k}'" for k in sorted(missing_meta))
    meta_raw = meta_raw or {}

    images = _load_images_safely(root / "images.npy", violations)
    n_meta = meta_raw.get("n_images")
    if images is not None:
        violations.extend(_validate_images(images, root / "images.npy", n_meta, meta_raw))

    index = _load_index_safely(root / "index.csv", violations)
    if index is not None:
        violations.extend(_validate_index(index, images))

    splits = _load_json_safely(root / "splits.json", "splits.json", violations)
    if splits is not None:
        n = n_meta if n_meta is not None else (images.shape[0] if images is not None else None)
        violations.extend(_validate_splits(splits, index, n))

    return violations


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to native Python for JSON dumping."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    return obj


def _load_json_safely(path: Path, label: str, violations: list[str]) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        violations.append(f"{label}: could not parse ({exc})")
        return None


def _load_images_safely(path: Path, violations: list[str]) -> np.ndarray | None:
    try:
        return np.load(path)
    except (OSError, ValueError) as exc:
        violations.append(f"images.npy: could not load ({exc})")
        return None


def _load_index_safely(path: Path, violations: list[str]) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        violations.append(f"index.csv: could not parse ({exc})")
        return None


def _validate_images(
    images: np.ndarray, path: Path, n_meta: int | None, meta_raw: dict[str, Any]
) -> list[str]:
    violations: list[str] = []
    if images.dtype != np.uint8:
        violations.append(f"images.npy: dtype is {images.dtype}, expected uint8")
    if images.ndim != 3:
        violations.append(f"images.npy: expected 3 dims (N, H, W), got shape {images.shape}")
        return violations
    if n_meta is not None and images.shape[0] != n_meta:
        violations.append(
            f"images.npy: N={images.shape[0]} does not match meta.json.n_images={n_meta}"
        )

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = meta_raw.get("sha256_images")
    if expected is not None and digest != expected:
        violations.append("meta.json.sha256_images: does not match the hash of images.npy")

    flat = images.reshape(images.shape[0], -1)
    constant_idx = np.flatnonzero(flat.max(axis=1) == flat.min(axis=1))
    violations.extend(f"images.npy: image idx={i} is entirely constant" for i in constant_idx.tolist())

    mean_val = float(images.mean())
    if not (5.0 < mean_val < 200.0):
        violations.append(f"images.npy: mean intensity {mean_val:.3f} outside (5, 200)")
    return violations


def _validate_index(index: pd.DataFrame, images: np.ndarray | None) -> list[str]:
    violations: list[str] = []
    if list(index.columns) != list(INDEX_COLUMNS):
        violations.append(
            f"index.csv: columns {list(index.columns)} do not match expected {list(INDEX_COLUMNS)}"
        )
        return violations

    if images is not None and len(index) != images.shape[0]:
        violations.append(f"index.csv: {len(index)} rows does not match images.npy N={images.shape[0]}")

    if not (index["idx"].to_numpy() == np.arange(len(index))).all():
        violations.append("index.csv: 'idx' column does not equal the row position")

    bad_split = sorted(set(index["split"]) - set(SPLIT_NAMES))
    if bad_split:
        violations.append(f"index.csv: 'split' has values outside {SPLIT_NAMES}: {bad_split}")

    z = pd.to_numeric(index["z_mm"], errors="coerce").to_numpy(dtype=float)
    n_nan = int(np.isnan(z).sum())
    if n_nan not in (0, len(z)):
        violations.append("index.csv: 'z_mm' mixes NaN and non-NaN values across rows")
    elif n_nan == 0 and not np.all(np.isfinite(z)):
        violations.append("index.csv: 'z_mm' contains non-finite (inf) values")

    return violations


def _validate_splits(
    splits: dict[str, Any], index: pd.DataFrame | None, n: int | None
) -> list[str]:
    violations: list[str] = []
    missing_keys = _REQUIRED_SPLIT_KEYS - splits.keys()
    violations.extend(f"splits.json: missing key '{k}'" for k in sorted(missing_keys))
    if missing_keys:
        return violations

    train_idx, ref_idx, seed_idx = splits["train"], splits["ref"], splits["seed"]
    if train_idx != sorted(train_idx):
        violations.append("splits.json: 'train' is not sorted ascending")
    if ref_idx != sorted(ref_idx):
        violations.append("splits.json: 'ref' is not sorted ascending")
    if seed_idx != sorted(seed_idx):
        violations.append("splits.json: 'seed' is not sorted ascending")

    train_set, ref_set, seed_set = set(train_idx), set(ref_idx), set(seed_idx)
    overlap = train_set & ref_set
    if overlap:
        violations.append(
            f"splits.json: 'train' and 'ref' overlap at idx {sorted(overlap)[:5]}"
        )
    if n is not None and (train_set | ref_set) != set(range(n)):
        violations.append("splits.json: 'train' + 'ref' do not partition range(N)")
    if not seed_set.issubset(ref_set):
        violations.append("splits.json: 'seed' is not a subset of 'ref'")

    if index is not None and "subject" in index.columns and "idx" in index.columns:
        for subject, group in index.groupby("subject"):
            idxs = set(group["idx"].tolist())
            if (idxs & train_set) and (idxs & ref_set):
                violations.append(
                    f"splits.json: subject '{subject}' has rows in both 'train' and 'ref'"
                )

    return violations
