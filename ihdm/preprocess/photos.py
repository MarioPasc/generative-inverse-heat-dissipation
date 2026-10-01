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

The M7 diagnostic (T7.1, ``docs/SPECIFICATIONS/M7-diagnostics/``) adds the whole-scene rule of
the paper's loader, :func:`to_gray_resize_crop` (short side resized to 128, centre crop), and the
helpers that rebuild a dataset from the shard rows another one recorded: :func:`parse_source`,
:func:`select_rows`, :func:`sha1_collisions`, :func:`centre_correlation`, and the ``seen`` /
``start_idx`` arguments of :func:`collect_photos`, which extend a dataset without re-admitting
any image it already holds. Its ``screen`` argument, with :class:`NearDuplicateScreen`, also
keeps out re-encoded copies of reference images that SHA-1 cannot see (:func:`thumbnails`,
:func:`near_duplicate_pairs`). Their shell is :mod:`ihdm.cli.build_diagnostic_photos`.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Iterable, Iterator, Sequence
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
    "ScreenedRecord",
    "CollectResult",
    "NearDuplicateScreen",
    "thumbnails",
    "near_duplicate_pairs",
    "to_gray_crop",
    "to_gray_resize_crop",
    "needs_resize",
    "array_sha1",
    "image_id",
    "collect_photos",
    "parse_source",
    "select_rows",
    "sha1_collisions",
    "centre_correlation",
    "build_index",
    "fine_split_labels",
    "write_contact_sheet",
    "write_pair_sheet",
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


@dataclass(frozen=True)
class ScreenedRecord:
    """One distinct row rejected by the ``screen`` of :func:`collect_photos`.

    Parameters
    ----------
    shard : str
        Parquet file the rejected row came from.
    row : int
        Row number inside that shard.
    sha1 : str
        SHA-1 of its cropped array.
    verdict : dict[str, Any]
        What the screen returned, e.g. ``{"max_r": 0.998, "ref_idx": 1645}``.
    """

    shard: str
    row: int
    sha1: str
    verdict: dict[str, Any]


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
        ``n_accepted``, and ``per_shard`` (scanned/accepted per parquet file); plus
        ``rows_screened_out`` when a screen was given.
    screened : list[ScreenedRecord]
        Rows rejected by the screen, in the order they were met.
    """

    images: np.ndarray
    records: list[PhotoRecord] = field(default_factory=list)
    duplicates: list[DuplicateRecord] = field(default_factory=list)
    counts: dict[str, Any] = field(default_factory=dict)
    screened: list[ScreenedRecord] = field(default_factory=list)


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


def to_gray_resize_crop(img: Image.Image, size: int) -> np.ndarray:
    """Convert an image to grayscale, resize its short side to ``size``, take the centre square.

    The whole-scene framing of the paper's loader, ``transforms.Resize(size)`` followed by
    ``transforms.CenterCrop(size)`` (``scripts/datasets.py`` lines 36-37), with its exact
    geometry: the long side becomes ``int(size * long / short)`` and the crop offset is
    ``int(round((side - size) / 2))``. Two deliberate differences from that loader, both fixed by
    the T7.1 ticket: luminance is taken first (``convert("L")``), and the resampling filter is
    LANCZOS instead of torchvision's default bilinear.

    Parameters
    ----------
    img : PIL.Image.Image
        Source image in any mode.
    size : int
        Output side in pixels (128 for ``lsun_church_r128``).

    Returns
    -------
    np.ndarray
        ``uint8`` array of shape ``(size, size)``.

    Raises
    ------
    PreprocessError
        If the short side is below ``size`` (the rule only ever downsamples) or the crop does
        not come out square.
    """
    width, height = img.size
    short, long = min(width, height), max(width, height)
    if short < size:
        raise PreprocessError(f"short side is {short} px, below the {size} px minimum")

    new_long = int(size * long / short)
    new_size = (size, new_long) if width <= height else (new_long, size)
    array = np.asarray(img.convert("L").resize(new_size, Image.Resampling.LANCZOS), np.uint8)

    top = int(round((array.shape[0] - size) / 2.0))
    left = int(round((array.shape[1] - size) / 2.0))
    array = np.ascontiguousarray(array[top : top + size, left : left + size])
    if array.shape != (size, size):
        raise PreprocessError(f"crop has shape {array.shape}, expected {(size, size)}")
    return array


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
    seen: dict[str, int] | None = None,
    start_idx: int = 0,
    screen: Callable[[np.ndarray], dict[str, Any] | None] | None = None,
) -> CollectResult:
    """Accept the first ``target`` usable, distinct images of a row stream.

    Rows are taken in stream (shard) order. A row is rejected if its short side is
    below 192 px or it cannot be decoded; it is dropped as a duplicate if the SHA-1
    of its cropped array repeats an earlier accepted one (or one of ``seen``); it is
    screened out if ``screen`` returns a verdict for its crop. Dropped rows are replaced
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
    seen : dict[str, int] | None
        SHA-1 digests of the images a dataset already holds, mapped to their ``idx``, so an
        extension never re-admits one of them (T7.1's ``lsun_church_n32k``). Not mutated.
    start_idx : int
        ``idx`` of the first accepted image, so the records and image ids of an extension
        continue the existing numbering. 0 for a dataset built from scratch.
    screen : Callable[[np.ndarray], dict[str, Any] | None] | None
        Called on the crop of every row that passed the SHA-1 check; a returned dict
        rejects the row and is kept as its :class:`ScreenedRecord` verdict (T7.1's
        :class:`NearDuplicateScreen`). ``None`` screens nothing and adds no count key.

    Returns
    -------
    CollectResult
        Accepted images, records, duplicates, screened rows and scan counts.

    Raises
    ------
    PreprocessError
        If the stream ends before ``target`` images are accepted.
    """
    images: list[np.ndarray] = []
    records: list[PhotoRecord] = []
    duplicates: list[DuplicateRecord] = []
    screened: list[ScreenedRecord] = []
    seen = dict(seen or {})
    counts = _new_counts()
    if screen is not None:
        counts["rows_screened_out"] = 0

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

        verdict = screen(array) if screen is not None else None
        if verdict is not None:
            counts["rows_screened_out"] += 1
            screened.append(ScreenedRecord(raw.shard, raw.row, digest, verdict))
            continue

        idx = start_idx + len(images)
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
        screened=screened,
    )


def thumbnails(images: np.ndarray, side: int = 32) -> np.ndarray:
    """Mean-pool images to ``side x side`` thumbnails, centred and scaled to unit norm.

    The dot product of two rows is the Pearson correlation of the two thumbnails, which is
    insensitive to re-encoding, mild re-processing and global brightness or contrast changes,
    so it finds copies of a photograph that SHA-1 cannot.

    Parameters
    ----------
    images : np.ndarray
        ``(N, H, W)`` array with ``H`` and ``W`` multiples of ``side``.
    side : int
        Thumbnail side in pixels.

    Returns
    -------
    np.ndarray
        ``float64`` array of shape ``(N, side * side)``; a constant image maps to zeros.

    Raises
    ------
    PreprocessError
        If ``H`` or ``W`` is not a multiple of ``side``.
    """
    n, height, width = images.shape
    if height % side or width % side:
        raise PreprocessError(f"image shape {(height, width)} is not a multiple of {side}")
    # Pool the stored dtype with a float64 accumulator: no float64 copy of the whole array.
    pooled = np.asarray(images).reshape(n, side, height // side, side, width // side).mean(
        axis=(2, 4), dtype=np.float64
    ).reshape(n, -1)
    pooled -= pooled.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(pooled, axis=1, keepdims=True)
    return pooled / np.where(norm > 0, norm, 1.0)


class NearDuplicateScreen:
    """Reject a crop whose thumbnail correlates above a threshold with a reference image.

    Used as the ``screen`` of :func:`collect_photos` when ``lsun_church_n32k`` appends train
    rows (T7.1): a re-encoded copy of a ``ref`` image would otherwise enter ``train``, which the
    baseline ``lsun_church`` does not have, and the dataset would differ in more than its size.

    Parameters
    ----------
    reference : np.ndarray
        ``(M, H, W)`` reference images.
    reference_idx : Sequence[int]
        Their ``idx``, reported in the verdicts.
    threshold : float
        Pearson r of the 32x32 thumbnails above which a crop is rejected.
    """

    def __init__(
        self, reference: np.ndarray, reference_idx: Sequence[int], threshold: float = 0.95
    ) -> None:
        self.reference = thumbnails(reference)
        self.reference_idx = [int(i) for i in reference_idx]
        self.threshold = threshold
        self.rejected_crops: list[np.ndarray] = []

    def __call__(self, crop: np.ndarray) -> dict[str, Any] | None:
        """Return ``{"max_r", "ref_idx"}`` if ``crop`` copies a reference image, else ``None``."""
        r = self.reference @ thumbnails(crop[None])[0]
        best = int(np.argmax(r))
        if r[best] <= self.threshold:
            return None
        self.rejected_crops.append(np.array(crop))
        return {"max_r": round(float(r[best]), 6), "ref_idx": self.reference_idx[best]}


def near_duplicate_pairs(
    images: np.ndarray, threshold: float = 0.95, block: int = 1024
) -> list[tuple[int, int, float]]:
    """All pairs of images whose 32x32 thumbnails correlate above ``threshold`` (record only).

    Parameters
    ----------
    images : np.ndarray
        ``(N, H, W)`` array, ``H`` and ``W`` multiples of 32.
    threshold : float
        Pearson r of the thumbnails above which a pair is reported.
    block : int
        Rows per block of the pairwise product, bounding memory at ``block * N`` floats.

    Returns
    -------
    list[tuple[int, int, float]]
        ``(i, j, r)`` with ``i < j``, sorted by ``i`` then ``j``.
    """
    thumbs = thumbnails(images)
    pairs: list[tuple[int, int, float]] = []
    for start in range(0, len(thumbs), block):
        corr = thumbs[start : start + block] @ thumbs.T
        for i, j in zip(*np.nonzero(corr > threshold), strict=True):
            if start + i < j:
                pairs.append((int(start + i), int(j), round(float(corr[i, j]), 6)))
    return sorted(pairs)


def parse_source(source: str, repo_id: str) -> tuple[str, int]:
    """Split an ``index.csv`` source string ``"<repo_id>/<shard>/<row>"`` into shard and row.

    Parameters
    ----------
    source : str
        The ``source`` value of one photograph row, as :func:`collect_photos` writes it.
    repo_id : str
        The Hugging Face repository the dataset was built from.

    Returns
    -------
    tuple[str, int]
        ``(shard, row)``.

    Raises
    ------
    PreprocessError
        If the string does not start with ``repo_id`` or does not end in a row number.
    """
    prefix = f"{repo_id}/"
    shard, _, row = source[len(prefix) :].rpartition("/")
    if not source.startswith(prefix) or not shard or not row.isdigit():
        raise PreprocessError(f"source {source!r} is not '{prefix}<shard>/<row>'")
    return shard, int(row)


def select_rows(rows: Iterable[RawRow], wanted: Sequence[tuple[str, int]]) -> Iterator[RawRow]:
    """Yield the rows of a stream named by ``(shard, row)``, in the order they are named.

    Used to rebuild a dataset from the exact rows another one accepted: those rows are recorded
    in shard order, so one pass over the stream finds them all, and the stream is not read past
    the last one.

    Parameters
    ----------
    rows : Iterable[RawRow]
        The row stream, in shard order.
    wanted : Sequence[tuple[str, int]]
        The ``(shard, row)`` pairs to yield; they must follow the stream order.

    Yields
    ------
    RawRow
        One row per entry of ``wanted``, in that order.

    Raises
    ------
    PreprocessError
        If the stream ends before every wanted row was met, which is also what happens when
        ``wanted`` is out of stream order.
    """
    pending = iter(wanted)
    target = next(pending, None)
    if target is None:
        return
    for raw in rows:
        if (raw.shard, raw.row) != target:
            continue
        yield raw
        target = next(pending, None)
        if target is None:
            return
    raise PreprocessError(
        f"stream ended before shard row {target}; the wanted rows must exist and follow the "
        "stream order"
    )


def sha1_collisions(images: np.ndarray) -> list[tuple[int, int, str]]:
    """Report the images whose SHA-1 repeats an earlier image's, without dropping anything.

    Parameters
    ----------
    images : np.ndarray
        ``uint8`` array of shape ``(N, H, W)``.

    Returns
    -------
    list[tuple[int, int, str]]
        ``(idx, first_idx, digest)`` for every repeat, in ``idx`` order; empty when every
        image is distinct.
    """
    first: dict[str, int] = {}
    repeats: list[tuple[int, int, str]] = []
    for idx, image in enumerate(images):
        digest = array_sha1(image)
        if digest in first:
            repeats.append((idx, first[digest], digest))
        else:
            first[digest] = idx
    return repeats


def centre_correlation(crops192: np.ndarray, crops128: np.ndarray) -> np.ndarray:
    """Per-row Pearson correlation of a native 192 crop with the same region of a 128 crop.

    For a photograph with a 256-px short side, the 192² native crop covers the central 192
    native pixels, and the 128² whole-scene crop shows the central 256 native pixels at half
    scale; the 192 crop's footprint is therefore the central 96x96 of the 128 crop, to within
    the sub-pixel rounding of the two crop offsets. The 192 crop is 2x2 mean-pooled to 96x96 and
    correlated with that footprint. The same photograph gives r close to 1; two different
    photographs give the correlation of unrelated scenes.

    Parameters
    ----------
    crops192 : np.ndarray
        ``(N, 192, 192)`` native crops.
    crops128 : np.ndarray
        ``(N, 128, 128)`` whole-scene crops, row ``i`` paired with ``crops192[i]``.

    Returns
    -------
    np.ndarray
        ``float64`` array of shape ``(N,)``.

    Raises
    ------
    PreprocessError
        If the shapes are not ``(N, 192, 192)`` and ``(N, 128, 128)`` with the same ``N``.
    """
    n = crops192.shape[0]
    if crops192.shape != (n, 192, 192) or crops128.shape != (n, 128, 128):
        raise PreprocessError(f"expected (N,192,192) and (N,128,128), got {crops192.shape} "
                              f"and {crops128.shape}")
    pooled = np.asarray(crops192, np.float64).reshape(n, 96, 2, 96, 2).mean(axis=(2, 4))
    footprint = np.asarray(crops128, np.float64)[:, 16:112, 16:112]
    a = pooled.reshape(n, -1) - pooled.reshape(n, -1).mean(axis=1, keepdims=True)
    b = footprint.reshape(n, -1) - footprint.reshape(n, -1).mean(axis=1, keepdims=True)
    return (a * b).sum(axis=1) / np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))


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


def write_pair_sheet(
    path: Path,
    left: np.ndarray,
    right: np.ndarray,
    picks: Sequence[int],
    title: str | None = None,
) -> None:
    """Render image pairs side by side (``left[i]`` next to ``right[i]``), four pairs per row.

    Parameters
    ----------
    path : Path
        Destination PNG.
    left, right : np.ndarray
        ``uint8`` arrays of shape ``(N, H, W)``; the two sides may differ in ``H, W``.
    picks : Sequence[int]
        Row indices to show.
    title : str | None
        Figure title.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = int(np.ceil(len(picks) / 4))
    fig, axes = plt.subplots(n_rows, 8, figsize=(12.0, 1.62 * n_rows), squeeze=False)
    for k, pick in enumerate(picks):
        for side, (images, tag) in enumerate(((left, "L"), (right, "R"))):
            ax = axes[k // 4][2 * (k % 4) + side]
            ax.imshow(images[pick], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_title(f"{pick} {tag} {images.shape[1]}", fontsize=5, pad=1.5)
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
