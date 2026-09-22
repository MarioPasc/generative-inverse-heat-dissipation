"""Hugging Face shard access for the photograph pipeline.

The two photograph datasets of the experiment (`lsun_church`, `lsun_bedroom`) come from
parquet shards published on the Hugging Face Hub. The `datasets` package is not part of
the `ihdm` environment, so the shards are downloaded with
:func:`huggingface_hub.hf_hub_download` and streamed with :mod:`pyarrow.parquet`, which
yields the file's physical row order: the deterministic "shard order" of the ticket.

Shards are downloaded lazily, one at a time, so a fallback shard listed in
:data:`PHOTO_SOURCES` is fetched only if the preceding shards do not supply enough
accepted rows.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError

from ihdm.preprocess.errors import PreprocessError

__all__ = [
    "PhotoSource",
    "RawRow",
    "PHOTO_SOURCES",
    "download_shard",
    "shard_num_rows",
    "iter_shard_rows",
    "iter_source_rows",
]

logger = logging.getLogger(__name__)

IMAGE_COLUMN = "image"


@dataclass(frozen=True)
class PhotoSource:
    """Provenance of one photograph dataset on the Hugging Face Hub.

    Parameters
    ----------
    dataset_id : str
        Standard-format dataset id (the output folder name).
    repo_id : str
        Hugging Face dataset repository id.
    shards : tuple[str, ...]
        Parquet files inside the repository, in the order they are consumed. The
        first entry is the shard that produced the existing 2000-image controls;
        later entries are fallbacks used only if the earlier ones run out of rows.
    id_prefix : str
        Prefix of the per-image subject id (``church`` gives ``church_00000``).
    """

    dataset_id: str
    repo_id: str
    shards: tuple[str, ...]
    id_prefix: str


@dataclass(frozen=True)
class RawRow:
    """One undecoded-to-array image row, tagged with its shard provenance.

    Parameters
    ----------
    shard : str
        Parquet file path inside the repository.
    row : int
        Zero-based row number within that shard, in physical file order.
    image : PIL.Image.Image
        The decoded-on-access image; its pixel data is read lazily by PIL.
    """

    shard: str
    row: int
    image: Image.Image


PHOTO_SOURCES: dict[str, PhotoSource] = {
    "lsun_church": PhotoSource(
        dataset_id="lsun_church",
        repo_id="tglcourse/lsun_church_train",
        shards=(
            "data/test-00000-of-00001-b3f98c62a94e5d25.parquet",
            "data/train-00000-of-00001-6cbaf0200a9bf96f.parquet",
        ),
        id_prefix="church",
    ),
    "lsun_bedroom": PhotoSource(
        dataset_id="lsun_bedroom",
        repo_id="pcuenq/lsun-bedrooms",
        shards=(
            "data/test-00000-of-00001-7c2280a6897e2462.parquet",
            "data/train-00000-of-00009-5beebd96eb33b02b.parquet",
        ),
        id_prefix="bedroom",
    ),
}


def download_shard(repo_id: str, filename: str, cache_dir: str | None = None) -> Path:
    """Download one parquet shard from the Hub (or return the cached copy).

    Parameters
    ----------
    repo_id : str
        Hugging Face dataset repository id.
    filename : str
        Path of the parquet file inside the repository.
    cache_dir : str | None
        Override of the Hugging Face cache directory.

    Returns
    -------
    Path
        Local path of the parquet file.

    Raises
    ------
    PreprocessError
        If the download fails for any reason.
    """
    try:
        local = hf_hub_download(repo_id, filename, repo_type="dataset", cache_dir=cache_dir)
    except Exception as exc:  # hub errors are many and every one of them is fatal here
        raise PreprocessError(f"could not download {repo_id}/{filename}: {exc}") from exc
    return Path(local)


def shard_num_rows(repo_id: str, filename: str, cache_dir: str | None = None) -> int:
    """Return the number of rows of one parquet shard.

    Parameters
    ----------
    repo_id : str
        Hugging Face dataset repository id.
    filename : str
        Path of the parquet file inside the repository.
    cache_dir : str | None
        Override of the Hugging Face cache directory.

    Returns
    -------
    int
        Row count reported by the parquet footer.
    """
    return pq.ParquetFile(download_shard(repo_id, filename, cache_dir)).metadata.num_rows


def iter_shard_rows(
    repo_id: str,
    filename: str,
    batch_size: int = 64,
    cache_dir: str | None = None,
) -> Iterator[RawRow]:
    """Stream the image rows of one parquet shard in physical file order.

    Parameters
    ----------
    repo_id : str
        Hugging Face dataset repository id.
    filename : str
        Path of the parquet file inside the repository.
    batch_size : int
        Number of rows pulled from parquet per batch; bounds peak memory.
    cache_dir : str | None
        Override of the Hugging Face cache directory.

    Yields
    ------
    RawRow
        One row per image, in shard order.

    Raises
    ------
    PreprocessError
        If the shard has no ``image`` column or a row cannot be opened by PIL.
    """
    path = download_shard(repo_id, filename, cache_dir)
    parquet = pq.ParquetFile(path)
    if IMAGE_COLUMN not in parquet.schema_arrow.names:
        raise PreprocessError(
            f"{repo_id}/{filename}: no '{IMAGE_COLUMN}' column "
            f"(columns: {parquet.schema_arrow.names})"
        )

    row = 0
    for batch in parquet.iter_batches(batch_size=batch_size, columns=[IMAGE_COLUMN]):
        for value in batch.column(IMAGE_COLUMN).to_pylist():
            yield RawRow(shard=filename, row=row, image=_open_image(value, repo_id, filename, row))
            row += 1


def iter_source_rows(
    source: PhotoSource,
    batch_size: int = 64,
    cache_dir: str | None = None,
) -> Iterator[RawRow]:
    """Stream every shard of one photograph source, in the declared shard order.

    Shards are downloaded lazily, so a consumer that stops early never fetches the
    later (much larger) fallback shards.

    Parameters
    ----------
    source : PhotoSource
        The dataset whose shards are streamed.
    batch_size : int
        Number of rows pulled from parquet per batch.
    cache_dir : str | None
        Override of the Hugging Face cache directory.

    Yields
    ------
    RawRow
        One row per image, shard by shard in order.
    """
    for shard in source.shards:
        logger.info("streaming %s/%s", source.repo_id, shard)
        yield from iter_shard_rows(source.repo_id, shard, batch_size, cache_dir)


def _open_image(value: object, repo_id: str, filename: str, row: int) -> Image.Image:
    """Open one parquet image cell (a ``{bytes, path}`` struct or raw bytes) with PIL."""
    payload = value.get("bytes") if isinstance(value, dict) else value
    if not isinstance(payload, (bytes, bytearray)):
        raise PreprocessError(f"{repo_id}/{filename}/{row}: image cell holds no bytes")
    try:
        return Image.open(io.BytesIO(payload))
    except UnidentifiedImageError as exc:
        raise PreprocessError(f"{repo_id}/{filename}/{row}: unreadable image") from exc
