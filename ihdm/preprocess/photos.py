"""Photograph preprocessing: grayscale 192x192 centre crops, index, splits, QC sheets.

Functional core of the photograph pipeline
(`docs/SPECIFICATIONS/M1-data/T1.2-photograph-pipeline.md`).
Every function here is a pure transformation of arrays or records except the three QC
writers at the end, which render a figure to a given path. The imperative shell (shard
streaming, raw PNG appending, dataset writing) lives in :mod:`ihdm.cli.preprocess_photos`.

The processing rule is frozen by `docs/SPECIFICATIONS/03-data-format.md` §5: RGB to
luminance with ``PIL.Image.convert("L")``, then the centre 192x192 crop of the native
frame. Only a short side above 256 px is resampled (LANCZOS down to a 256-px short side)
before the crop; LSUN ships with a 256-px short side, so no LSUN row is resampled.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from ihdm.preprocess.errors import PreprocessError
from ihdm.preprocess.fetch_hf import RawRow

__all__ = [
    "CROP_SIZE",
    "TARGET_SHORT_SIDE",
    "PhotoRecord",
    "DuplicateRecord",
    "CollectResult",
    "to_gray_crop",
    "needs_resize",
    "array_sha1",
    "image_id",
    "collect_photos",
    "build_index",
    "fine_split_labels",
    "write_contact_sheet",
    "write_intensity_hist",
    "write_duplicates_md",
]

logger = logging.getLogger(__name__)

CROP_SIZE = 192
TARGET_SHORT_SIDE = 256


@dataclass(frozen=True)
class PhotoRecord:
    """One accepted image and its provenance.

    Parameters
    ----------
    idx : int
        Position in ``images.npy`` (equals the accepted-row counter).
    image_id : str
        Per-image subject id, e.g. ``church_00000``.
    shard : str
        Parquet file the row came from.
    row : int
        Row number inside that shard.
    source : str
        The ``index.csv`` source string ``"<repo_id>/<shard>/<row>"``.
    sha1 : str
        SHA-1 of the cropped uint8 array, the de-duplication key.
    resized : bool
        Whether the image was resampled to a 256-px short side before cropping.
    """

    idx: int
    image_id: str
    shard: str
    row: int
    source: str
    sha1: str
    resized: bool


@dataclass(frozen=True)
class DuplicateRecord:
    """One row dropped because its cropped array repeats an earlier accepted one.

    Parameters
    ----------
    shard : str
        Parquet file the dropped row came from.
    row : int
        Row number inside that shard.
    sha1 : str
        The repeated SHA-1.
    first_idx : int
        ``idx`` of the accepted image that first carried this SHA-1.
    """

    shard: str
    row: int
    sha1: str
    first_idx: int


@dataclass
class CollectResult:
    """Outcome of one acceptance pass over a row stream.

    Parameters
    ----------
    images : np.ndarray
        ``uint8`` array of shape ``(n_accepted, 192, 192)``.
    records : list[PhotoRecord]
        One record per accepted image, in ``idx`` order.
    duplicates : list[DuplicateRecord]
        Rows dropped as duplicates, in the order they were met.
    counts : dict[str, Any]
        Scan statistics: ``rows_scanned``, ``rows_rejected_size``,
        ``rows_rejected_unreadable``, ``duplicates_dropped``, ``resized``,
        ``n_accepted``, and ``per_shard`` (scanned/accepted per parquet file).
    """

    images: np.ndarray
    records: list[PhotoRecord] = field(default_factory=list)
    duplicates: list[DuplicateRecord] = field(default_factory=list)
    counts: dict[str, Any] = field(default_factory=dict)


def to_gray_crop(img: Image.Image) -> np.ndarray:
    """Convert an image to the standard-format grayscale 192x192 centre crop.

    Parameters
    ----------
    img : PIL.Image.Image
        Source image in any mode.

    Returns
    -------
    np.ndarray
        ``uint8`` array of shape ``(192, 192)``.

    Raises
    ------
    PreprocessError
        If the short side is below 192 px, or the crop does not come out square.
    """
    short = min(img.size)
    if short < CROP_SIZE:
        raise PreprocessError(f"short side is {short} px, below the {CROP_SIZE} px minimum")

    gray = img.convert("L")
    if short > TARGET_SHORT_SIDE:
        gray = _resize_short_side(gray, TARGET_SHORT_SIDE)

    array = _centre_crop(np.asarray(gray, dtype=np.uint8))
    if array.shape != (CROP_SIZE, CROP_SIZE):
        raise PreprocessError(f"crop has shape {array.shape}, expected {(CROP_SIZE, CROP_SIZE)}")
    return array


def needs_resize(img: Image.Image) -> bool:
    """Return whether :func:`to_gray_crop` would resample this image.

    Parameters
    ----------
    img : PIL.Image.Image
        Source image.

    Returns
    -------
    bool
        ``True`` if the short side exceeds 256 px.
    """
    return min(img.size) > TARGET_SHORT_SIDE


def array_sha1(array: np.ndarray) -> str:
    """Return the hex SHA-1 of an array's bytes, the de-duplication key.

    Parameters
    ----------
    array : np.ndarray
        Any array; made C-contiguous before hashing so the digest depends on
        values and shape order only.

    Returns
    -------
    str
        Hex digest.
    """
    return hashlib.sha1(np.ascontiguousarray(array).tobytes()).hexdigest()


def image_id(prefix: str, idx: int) -> str:
    """Return the per-image subject id of an accepted image.

    Parameters
    ----------
    prefix : str
        Dataset prefix, e.g. ``church``.
    idx : int
        Accepted-row index.

    Returns
    -------
    str
        ``"<prefix>_<idx:05d>"``.
    """
    return f"{prefix}_{idx:05d}"


def collect_photos(
    rows: Iterable[RawRow],
    target: int,
    id_prefix: str,
    repo_id: str,
    on_accept: Callable[[int, RawRow], None] | None = None,
) -> CollectResult:
    """Accept the first ``target`` usable, distinct images of a row stream.

    Rows are taken in stream (shard) order. A row is rejected if its short side is
    below 192 px or it cannot be decoded; it is dropped as a duplicate if the SHA-1
    of its cropped array repeats an earlier accepted one. Dropped rows are replaced
    by the next rows, so the accepted count reaches ``target`` whenever the stream is
    long enough.

    Parameters
    ----------
    rows : Iterable[RawRow]
        Row stream, e.g. :func:`ihdm.preprocess.fetch_hf.iter_source_rows`.
    target : int
        Number of images to accept.
    id_prefix : str
        Prefix of the per-image subject ids.
    repo_id : str
        Hugging Face repository id, used to build the ``source`` string.
    on_accept : Callable[[int, RawRow], None] | None
        Called with ``(idx, row)`` for every accepted image, before the next row is
        read. Used by the CLI to append the raw grayscale PNG.

    Returns
    -------
    CollectResult
        Accepted images, records, duplicates and scan counts.

    Raises
    ------
    PreprocessError
        If the stream ends before ``target`` images are accepted.
    """
    images: list[np.ndarray] = []
    records: list[PhotoRecord] = []
    duplicates: list[DuplicateRecord] = []
    seen: dict[str, int] = {}
    counts = _new_counts()

    for raw in rows:
        counts["rows_scanned"] += 1
        per_shard = counts["per_shard"].setdefault(raw.shard, {"scanned": 0, "accepted": 0})
        per_shard["scanned"] += 1

        resized = needs_resize(raw.image)
        try:
            array = to_gray_crop(raw.image)
        except PreprocessError as exc:
            _count_rejection(counts, exc)
            continue

        digest = array_sha1(array)
        if digest in seen:
            counts["duplicates_dropped"] += 1
            duplicates.append(
                DuplicateRecord(raw.shard, raw.row, digest, first_idx=seen[digest])
            )
            continue

        idx = len(images)
        seen[digest] = idx
        images.append(array)
        counts["resized"] += int(resized)
        per_shard["accepted"] += 1
        records.append(
            PhotoRecord(
                idx=idx,
                image_id=image_id(id_prefix, idx),
                shard=raw.shard,
                row=raw.row,
                source=f"{repo_id}/{raw.shard}/{raw.row}",
                sha1=digest,
                resized=resized,
            )
        )
        if on_accept is not None:
            on_accept(idx, raw)
        if len(images) >= target:
            break

    if len(images) < target:
        raise PreprocessError(
            f"{repo_id}: stream exhausted after {counts['rows_scanned']} rows with "
            f"{len(images)} accepted images, target was {target}"
        )

    counts["n_accepted"] = len(images)
    return CollectResult(
        images=np.stack(images).astype(np.uint8),
        records=records,
        duplicates=duplicates,
        counts=counts,
    )


def fine_split_labels(splits: dict[str, Any], n: int) -> list[str]:
    """Return the per-row finest split label (``train``, ``ref`` or ``seed``).

    Parameters
    ----------
    splits : dict[str, Any]
        The ``splits.json``-shaped dict from
        :func:`ihdm.data.format.split_by_subject`.
    n : int
        Number of images.

    Returns
    -------
    list[str]
        One label per image index, ``seed`` taking precedence over ``ref``.
    """
    train, seed = set(splits["train"]), set(splits["seed"])
    return ["train" if i in train else ("seed" if i in seed else "ref") for i in range(n)]


def build_index(records: Sequence[PhotoRecord], splits: dict[str, Any]) -> pd.DataFrame:
    """Build the ``index.csv`` frame of a photograph dataset.

    Each photograph is its own subject, so ``slice`` is 0 and ``z_mm`` is ``nan``
    for every row (`03-data-format.md` §2).

    Parameters
    ----------
    records : Sequence[PhotoRecord]
        Accepted records in ``idx`` order.
    splits : dict[str, Any]
        The ``splits.json``-shaped dict used to label each row.

    Returns
    -------
    pd.DataFrame
        Frame with exactly the contract's columns, in order.
    """
    n = len(records)
    return pd.DataFrame(
        {
            "idx": list(range(n)),
            "subject": [r.image_id for r in records],
            "slice": [0] * n,
            "z_mm": [float("nan")] * n,
            "source": [r.source for r in records],
            "split": fine_split_labels(splits, n),
        }
    )


def write_contact_sheet(
    path: Path,
    images: np.ndarray,
    labels: Sequence[str],
    n_tiles: int = 64,
    rng_seed: int = 2026,
    title: str | None = None,
) -> None:
    """Render a contact sheet of randomly drawn images annotated with their split.

    Parameters
    ----------
    path : Path
        Destination PNG.
    images : np.ndarray
        ``uint8`` array of shape ``(N, H, W)``.
    labels : Sequence[str]
        Per-image split label, same length as ``images``.
    n_tiles : int
        Number of tiles; laid out on the nearest square grid.
    rng_seed : int
        Seed of the tile draw, so the sheet is reproducible.
    title : str | None
        Figure title.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(rng_seed)
    n_tiles = min(n_tiles, len(images))
    picks = np.sort(rng.choice(len(images), size=n_tiles, replace=False))
    side = int(np.ceil(np.sqrt(n_tiles)))

    fig, axes = plt.subplots(side, side, figsize=(1.5 * side, 1.62 * side))
    for ax, pick in zip(np.ravel(axes), picks, strict=False):
        ax.imshow(images[pick], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(f"{pick} {labels[pick]}", fontsize=5, pad=1.5)
    for ax in np.ravel(axes):
        ax.set_axis_off()
    if title is not None:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_intensity_hist(path: Path, images: np.ndarray, title: str | None = None) -> None:
    """Render the pooled intensity histogram of a dataset.

    Parameters
    ----------
    path : Path
        Destination PNG.
    images : np.ndarray
        ``uint8`` array of shape ``(N, H, W)``.
    title : str | None
        Figure title.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = np.bincount(np.asarray(images).reshape(-1), minlength=256).astype(float)
    counts /= counts.sum()

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.bar(np.arange(256), counts, width=1.0, color="0.25")
    ax.set_xlabel("intensity (uint8)")
    ax.set_ylabel("fraction of pixels")
    ax.set_xlim(0, 255)
    if title is not None:
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_duplicates_md(
    path: Path,
    dataset_id: str,
    duplicates: Sequence[DuplicateRecord],
    counts: dict[str, Any],
) -> None:
    """Write the duplicate-handling report of a dataset.

    Parameters
    ----------
    path : Path
        Destination Markdown file.
    dataset_id : str
        Dataset id, used in the heading.
    duplicates : Sequence[DuplicateRecord]
        Rows dropped as duplicates.
    counts : dict[str, Any]
        The scan counts of :class:`CollectResult`.
    """
    lines = [
        f"# Duplicate report — {dataset_id}",
        "",
        "Rule: SHA-1 of the cropped uint8 (192, 192) array. The first row carrying a digest is",
        "accepted; every later row with the same digest is dropped and replaced by the next row,",
        "so the accepted count still reaches the target.",
        "",
        f"- rows scanned: {counts.get('rows_scanned', 0)}",
        f"- rows rejected (short side < {CROP_SIZE} px): {counts.get('rows_rejected_size', 0)}",
        f"- rows rejected (unreadable): {counts.get('rows_rejected_unreadable', 0)}",
        f"- duplicates dropped: {counts.get('duplicates_dropped', 0)}",
        f"- images accepted: {counts.get('n_accepted', 0)}",
        "",
    ]
    if duplicates:
        lines += ["| shard | row | sha1 | duplicate of idx |", "|---|---|---|---|"]
        lines += [f"| `{d.shard}` | {d.row} | `{d.sha1}` | {d.first_idx} |" for d in duplicates]
    else:
        lines.append("No duplicate was found among the scanned rows.")
    lines.append("")
    path.write_text("\n".join(lines))


def _new_counts() -> dict[str, Any]:
    """Return a zeroed scan-count dict."""
    return {
        "rows_scanned": 0,
        "rows_rejected_size": 0,
        "rows_rejected_unreadable": 0,
        "duplicates_dropped": 0,
        "resized": 0,
        "n_accepted": 0,
        "per_shard": {},
    }


def _count_rejection(counts: dict[str, Any], exc: PreprocessError) -> None:
    """Attribute a rejected row to the size or the unreadable bucket."""
    key = "rows_rejected_size" if "short side" in str(exc) else "rows_rejected_unreadable"
    counts[key] += 1


def _resize_short_side(img: Image.Image, target: int) -> Image.Image:
    """Resample so the short side is exactly ``target`` px, keeping the aspect ratio."""
    width, height = img.size
    if width <= height:
        size = (target, max(target, round(height * target / width)))
    else:
        size = (max(target, round(width * target / height)), target)
    return img.resize(size, Image.Resampling.LANCZOS)


def _centre_crop(array: np.ndarray) -> np.ndarray:
    """Return the centre ``CROP_SIZE`` square of a 2-D array (same rule as the CPU analyses)."""
    top = (array.shape[0] - CROP_SIZE) // 2
    left = (array.shape[1] - CROP_SIZE) // 2
    return np.ascontiguousarray(array[top : top + CROP_SIZE, left : left + CROP_SIZE])
