"""Build the two M7 diagnostic photograph datasets, each one factor away from ``lsun_church``.

Ticket T7.1 (``docs/SPECIFICATIONS/M7-diagnostics/``). Both datasets are derived from the
standard-format ``lsun_church`` and the Hugging Face shards it was built from:

* ``lsun_church_r128`` changes the resolution and framing only: the same 4,000 shard rows, in the
  same order, read back from the ``source`` column of ``lsun_church/index.csv`` (not "the first
  4,000 rows": T1.2 dropped two duplicates), each converted to luminance, resized to a 128-px
  short side and centre-cropped (the paper's ``Resize(128)`` + ``CenterCrop(128)``). The index
  and the splits are ``lsun_church``'s. A SHA-1 collision among the 128² crops is recorded and
  never dropped, because row identity with ``lsun_church`` comes first. Row identity is proven
  on every row: the 192² crop of the row read back must equal ``lsun_church``'s image pixel for
  pixel, and the per-row centre correlation is reported against a mismatched pairing.
* ``lsun_church_n32k`` changes the training-set size only: idx 0-3999 are ``lsun_church`` byte
  for byte (index rows and splits included, so ``ref`` and ``seed`` are pixel-identical), and
  idx 4000-32799 are the first 28,800 rows of the ``train`` shard accepted by T1.2's rule (192²
  native centre crop, short side >= 192 px), not repeating any image already in the dataset
  (SHA-1), and not a near-copy of a ``ref`` image (32x32 thumbnail Pearson r > 0.95; decided by
  ``main`` on 2026-10-01, because the HF train shard holds re-encoded copies of 6 ``ref``
  images while ``lsun_church``'s own train split holds none).

Run as ``python -m ihdm.cli.build_diagnostic_photos --set lsun_church_r128|lsun_church_n32k``.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import subprocess
import sys
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ihdm.data.format import DatasetMeta, read_dataset, validate_dataset, write_dataset
from ihdm.paths import data_root
from ihdm.preprocess.errors import PreprocessError
from ihdm.preprocess.fetch_hf import (
    PHOTO_SOURCES,
    RawRow,
    download_shard,
    iter_shard_rows,
    shard_num_rows,
)
from ihdm.preprocess.photos import (
    CROP_SIZE,
    DuplicateRecord,
    NearDuplicateScreen,
    PhotoRecord,
    ScreenedRecord,
    array_sha1,
    centre_correlation,
    collect_photos,
    near_duplicate_pairs,
    parse_source,
    select_rows,
    sha1_collisions,
    thumbnails,
    to_gray_crop,
    to_gray_resize_crop,
    write_contact_sheet,
    write_duplicates_md,
    write_intensity_hist,
    write_pair_sheet,
)

logger = logging.getLogger(__name__)

PIPELINE = "ihdm.cli.build_diagnostic_photos"
PIPELINE_VERSION = "1.0"
BASE_ID = "lsun_church"
R128_ID = "lsun_church_r128"
N32K_ID = "lsun_church_n32k"
R128_SIZE = 128
N32K_NEW_TRAIN = 28_800
# The second declared shard of lsun_church, never read by T1.2 (its first shard sufficed).
N32K_SHARD = PHOTO_SOURCES[BASE_ID].shards[1]
EXPECTED_N = {R128_ID: 4_000, N32K_ID: 4_000 + N32K_NEW_TRAIN}
# Pearson r of 32x32 thumbnails above which two images are the same photograph (main, D of
# 2026-10-01): re-encoded copies score 0.99+, unrelated church photographs at most ~0.72.
NEAR_DUP_THRESHOLD = 0.95


@dataclass(frozen=True)
class BaseDataset:
    """The ``lsun_church`` dataset every diagnostic dataset is derived from.

    Parameters
    ----------
    images : np.ndarray
        ``uint8`` array of shape ``(N, 192, 192)``, loaded into memory.
    index : pd.DataFrame
        Its ``index.csv``.
    splits : dict[str, Any]
        Its ``splits.json``.
    meta : DatasetMeta
        Its ``meta.json``.
    """

    images: np.ndarray
    index: pd.DataFrame
    splits: dict[str, Any]
    meta: DatasetMeta


@dataclass
class Built:
    """A diagnostic dataset ready for :func:`ihdm.data.format.write_dataset`, plus its evidence.

    Parameters
    ----------
    images, index, splits, meta
        The four standard-format parts.
    evidence : dict[str, Any]
        Identity checks, collisions and scan counts, also copied into ``meta.json``.
    duplicates : list[DuplicateRecord]
        Rows dropped as duplicates (``n32k`` only), for the QC report.
    screened_pairs : list[tuple[np.ndarray, np.ndarray, float]]
        ``(ref image, rejected crop, r)`` of every near-copy kept out (``n32k`` only), for QC.
    """

    images: np.ndarray
    index: pd.DataFrame
    splits: dict[str, Any]
    meta: DatasetMeta
    evidence: dict[str, Any] = field(default_factory=dict)
    duplicates: list[DuplicateRecord] = field(default_factory=list)
    screened_pairs: list[tuple[np.ndarray, np.ndarray, float]] = field(default_factory=list)


def base_rows(base: BaseDataset) -> list[tuple[str, int]]:
    """Return the ``(shard, row)`` of every image of the base dataset, in ``idx`` order.

    Parameters
    ----------
    base : BaseDataset
        The base dataset.

    Returns
    -------
    list[tuple[str, int]]
        One pair per row of ``index.csv``.
    """
    repo_id = base.meta.parameters["hf_dataset"]
    return [parse_source(source, repo_id) for source in base.index["source"]]


def build_r128(base: BaseDataset, rows: Iterable[RawRow], git_sha: str, raw_root: str) -> Built:
    """Build ``lsun_church_r128`` from the base dataset's own shard rows.

    Parameters
    ----------
    base : BaseDataset
        ``lsun_church``.
    rows : Iterable[RawRow]
        The stream of the shards the base was built from, in shard order.
    git_sha : str
        Commit recorded in ``meta.json``.
    raw_root : str
        Local directory of the parquet shards, recorded in ``meta.json``.

    Returns
    -------
    Built
        The dataset and its identity evidence.

    Raises
    ------
    PreprocessError
        If a row cannot be found, or the 192² crop of a row read back differs from the base
        image of the same ``idx`` (the rows would not be the same photographs).
    """
    wanted = base_rows(base)
    crops: list[np.ndarray] = []
    mismatched: list[int] = []
    for idx, raw in enumerate(select_rows(rows, wanted)):
        crops.append(to_gray_resize_crop(raw.image, R128_SIZE))
        if not np.array_equal(to_gray_crop(raw.image), base.images[idx]):
            mismatched.append(idx)
    if len(crops) != len(wanted) or mismatched:
        raise PreprocessError(f"row identity broken: {len(crops)} of {len(wanted)} rows read, "
                              f"192² crop differs from {BASE_ID} at idx {mismatched[:10]}")
    images = np.stack(crops).astype(np.uint8)
    evidence = _r128_evidence(base, images)
    meta = DatasetMeta(
        dataset_id=R128_ID,
        n_images=int(images.shape[0]),
        image_size=R128_SIZE,
        dtype="uint8",
        pipeline=PIPELINE,
        pipeline_version=PIPELINE_VERSION,
        git_sha=git_sha,
        created=datetime.now(UTC).isoformat(timespec="seconds"),
        raw_root=raw_root,
        parameters=_r128_parameters(base, evidence),
        counts={"n_images": int(images.shape[0]), "rows_selected": len(wanted),
                "sha1_collisions": len(evidence["sha1_collisions"]), "padding_fraction": 0.0},
    )
    return Built(images, base.index.copy(), dict(base.splits), meta, evidence)


def _r128_evidence(base: BaseDataset, images: np.ndarray) -> dict[str, Any]:
    """Row-identity evidence and the record-only SHA-1 collision list of ``lsun_church_r128``."""
    n = images.shape[0]
    matched = centre_correlation(base.images, images)
    # Null distribution: every 192² crop against the 128² crop of another row, half the
    # dataset away (a fixed derangement, so the number is reproducible).
    null = centre_correlation(base.images, images[np.roll(np.arange(n), n // 2)])
    return {
        "rows_192_reproduced": n,
        "rows_192_rule": "to_gray_crop of the row read back == lsun_church images[idx], "
                         "pixel for pixel, for every idx",
        "centre_correlation": {
            "rule": "Pearson r of the 2x2-mean-pooled lsun_church 192² crop (96x96) and the "
                    "central 96x96 of the 128² crop (the same native footprint)",
            "matched": _summary(matched),
            "null_shifted_by_half": _summary(null),
            "n_matched_below_null_max": int((matched <= null.max()).sum()),
        },
        "sha1_collisions": [
            {"idx": i, "first_idx": j, "sha1": d} for i, j, d in sha1_collisions(images)
        ],
        # Record only (main, 2026-10-01): near-copies inside the dataset are kept, and the same
        # scan of lsun_church's 192² crops says whether r128 inherited them or created them.
        "near_duplicate_pairs_within": {
            "rule": f"32x32 thumbnail Pearson r > {NEAR_DUP_THRESHOLD}; recorded, never dropped",
            "pairs": [list(p) for p in near_duplicate_pairs(images, NEAR_DUP_THRESHOLD)],
            "pairs_in_lsun_church_192": [
                list(p) for p in near_duplicate_pairs(base.images, NEAR_DUP_THRESHOLD)
            ],
        },
    }


def _summary(values: np.ndarray) -> dict[str, float]:
    """Min, 1st percentile, median and max of a vector, rounded for ``meta.json``."""
    return {"min": round(float(values.min()), 6), "p01": round(float(np.percentile(values, 1)), 6),
            "median": round(float(np.median(values)), 6), "max": round(float(values.max()), 6)}


def _r128_parameters(base: BaseDataset, evidence: dict[str, Any]) -> dict[str, Any]:
    """The ``meta.json.parameters`` of ``lsun_church_r128``."""
    params = base.meta.parameters
    return {
        "derived_from": BASE_ID,
        "derived_from_sha256_images": base.meta.sha256_images,
        "hf_dataset": params["hf_dataset"],
        "shards_used": params["shards_used"],
        "row_selection": (f"exactly the {base.images.shape[0]} (shard, row) of "
                          f"{BASE_ID}/index.csv source, in idx order (T1.2 dropped "
                          f"{base.meta.counts.get('duplicates_dropped', '?')} duplicates)"),
        "resize_rule": (f"PIL convert('L'), LANCZOS resize to a {R128_SIZE} px short side with "
                        f"the long side int({R128_SIZE} * long / short) (torchvision Resize)"),
        "crop_rule": (f"centre {R128_SIZE}x{R128_SIZE} crop at offset int(round((side - "
                      f"{R128_SIZE}) / 2)) (torchvision CenterCrop)"),
        "framing": "whole scene, the paper's Resize(128) + CenterCrop(128); no 192 native crop",
        "dedup_rule": ("SHA-1 of the cropped uint8 (128, 128) array; collisions are recorded in "
                       "sha1_collisions and never dropped (row identity with lsun_church first)"),
        "split_rule": f"identical to {BASE_ID}/splits.json (same index lists)",
        "orientation": "photograph as published; no flip",
        "identity_check": evidence,
    }


def build_n32k(
    base: BaseDataset,
    rows: Iterable[RawRow],
    git_sha: str,
    raw_root: str,
    n_new: int = N32K_NEW_TRAIN,
) -> Built:
    """Build ``lsun_church_n32k``: the base dataset plus ``n_new`` train images from a new shard.

    Parameters
    ----------
    base : BaseDataset
        ``lsun_church``.
    rows : Iterable[RawRow]
        The stream of the new shard, in shard order.
    git_sha : str
        Commit recorded in ``meta.json``.
    raw_root : str
        Local directory of the parquet shards, recorded in ``meta.json``.
    n_new : int
        Number of new train images (28,800 in production).

    Returns
    -------
    Built
        The dataset, its scan counts and the duplicate rows.

    Raises
    ------
    PreprocessError
        If the stream runs out before ``n_new`` images are accepted, or the copied ``ref`` /
        ``seed`` images differ from the base's.
    """
    repo_id = base.meta.parameters["hf_dataset"]
    n_base = int(base.images.shape[0])
    seen: dict[str, int] = {}
    for idx, image in enumerate(base.images):
        seen.setdefault(array_sha1(image), idx)
    ref = list(base.splits["ref"])
    screen = NearDuplicateScreen(base.images[ref], ref, threshold=NEAR_DUP_THRESHOLD)
    result = collect_photos(rows, target=n_new, id_prefix=PHOTO_SOURCES[BASE_ID].id_prefix,
                            repo_id=repo_id, seen=seen, start_idx=n_base, screen=screen)

    images = np.concatenate([base.images, result.images]).astype(np.uint8)
    new_idx = [r.idx for r in result.records]
    index = pd.concat([base.index, _new_index_rows(result.records)], ignore_index=True)
    splits = _n32k_splits(base.splits, new_idx, [r.image_id for r in result.records])
    evidence = _n32k_evidence(base, images, splits, result.counts)
    evidence["near_duplicate_filter"] = _near_dup_evidence(base, images, splits, result.screened)
    screened_pairs = [
        (base.images[s.verdict["ref_idx"]], crop, float(s.verdict["max_r"]))
        for s, crop in zip(result.screened, screen.rejected_crops, strict=True)
    ]

    meta = DatasetMeta(
        dataset_id=N32K_ID,
        n_images=int(images.shape[0]),
        image_size=CROP_SIZE,
        dtype="uint8",
        pipeline=PIPELINE,
        pipeline_version=PIPELINE_VERSION,
        git_sha=git_sha,
        created=datetime.now(UTC).isoformat(timespec="seconds"),
        raw_root=raw_root,
        parameters=_n32k_parameters(base, n_new, evidence),
        counts=evidence["counts"],
    )
    return Built(images, index, splits, meta, evidence, list(result.duplicates), screened_pairs)


def _near_dup_evidence(
    base: BaseDataset, images: np.ndarray, splits: dict[str, Any],
    screened: list[ScreenedRecord],
) -> dict[str, Any]:
    """The near-copy rule, the rows it kept out, and the property it restores."""
    ref_thumbs = thumbnails(base.images[base.splits["ref"]])
    rejected = [
        {"shard": s.shard, "row": s.row, **s.verdict}
        for s in sorted(screened, key=lambda s: -s.verdict["max_r"])
    ]
    return {
        "rule": (f"a new row is rejected if the Pearson r of its 32x32 mean-pooled thumbnail "
                 f"with any of the {len(base.splits['ref'])} ref images (seed within ref) "
                 f"exceeds {NEAR_DUP_THRESHOLD}; applied after the SHA-1 dedup, to new rows "
                 "only; the next shard row replaces it"),
        "threshold": NEAR_DUP_THRESHOLD,
        "n_rejected": len(rejected),
        "rejected": rejected,
        "ref_with_near_copy_in_train": {
            BASE_ID: _count_ref_with_copy(ref_thumbs, base.images, base.splits["train"]),
            N32K_ID: _count_ref_with_copy(ref_thumbs, images, splits["train"]),
        },
    }


def _count_ref_with_copy(ref_thumbs: np.ndarray, images: np.ndarray, train: list[int],
                         block: int = 2048) -> int:
    """Number of reference images with a train image above the near-copy threshold."""
    best = np.full(len(ref_thumbs), -1.0)
    for start in range(0, len(train), block):
        corr = thumbnails(images[train[start : start + block]]) @ ref_thumbs.T
        best = np.maximum(best, corr.max(axis=0))
    return int((best > NEAR_DUP_THRESHOLD).sum())


def _new_index_rows(records: Sequence[PhotoRecord]) -> pd.DataFrame:
    """``index.csv`` rows of the appended images (each its own subject, all ``train``)."""
    n = len(records)
    return pd.DataFrame({
        "idx": [r.idx for r in records],
        "subject": [r.image_id for r in records],
        "slice": [0] * n,
        "z_mm": [float("nan")] * n,
        "source": [r.source for r in records],
        "split": ["train"] * n,
    })


def _n32k_splits(base: dict[str, Any], new_idx: list[int], new_ids: list[str]) -> dict[str, Any]:
    """The base splits with every appended image in ``train``; ``ref`` and ``seed`` unchanged."""
    n_base = len(base["train"]) + len(base["ref"])
    return {
        "train": sorted([*base["train"], *new_idx]),
        "ref": list(base["ref"]),
        "seed": list(base["seed"]),
        "train_subjects": sorted([*base["train_subjects"], *new_ids]),
        "ref_subjects": list(base["ref_subjects"]),
        "seed_subjects": list(base["seed_subjects"]),
        "rule": (f"{BASE_ID}'s splits ({base['rule']}) for idx 0-{n_base - 1}; every image "
                 "appended from the train shard is in train"),
        "rng_seed": base["rng_seed"],
    }


def _n32k_evidence(
    base: BaseDataset, images: np.ndarray, splits: dict[str, Any], scan: dict[str, Any]
) -> dict[str, Any]:
    """Check the ref/seed identity and the base prefix, and assemble the counts."""
    n_base = int(base.images.shape[0])
    for name in ("ref", "seed"):
        if splits[name] != base.splits[name] or not np.array_equal(
            images[splits[name]], base.images[base.splits[name]]
        ):
            raise PreprocessError(f"{name} images differ from {BASE_ID}'s")
    if not np.array_equal(images[:n_base], base.images):
        raise PreprocessError(f"idx 0-{n_base - 1} differ from {BASE_ID}")
    counts = {
        "n_images": int(images.shape[0]),
        "composition": {
            "train_from_base": len(base.splits["train"]),
            "train_new": int(images.shape[0]) - n_base,
            "train_total": len(splits["train"]),
            "ref": len(splits["ref"]),
            "seed_within_ref": len(splits["seed"]),
        },
        "new_rows_scan": scan,
        "padding_fraction": 0.0,
    }
    return {
        "ref_seed_pixel_identical": True,
        "base_prefix_identical": f"images[0:{n_base}] == {BASE_ID}/images.npy",
        "counts": counts,
    }


def _n32k_parameters(base: BaseDataset, n_new: int, evidence: dict[str, Any]) -> dict[str, Any]:
    """The ``meta.json.parameters`` of ``lsun_church_n32k``."""
    params = base.meta.parameters
    n_base = int(base.images.shape[0])
    return {
        "derived_from": BASE_ID,
        "derived_from_sha256_images": base.meta.sha256_images,
        "hf_dataset": params["hf_dataset"],
        "shards_used": [*params["shards_used"], N32K_SHARD],
        "row_selection": (f"idx 0-{n_base - 1}: {BASE_ID} copied byte for byte (images, index "
                          f"rows, splits); idx {n_base}-{n_base + n_new - 1}: the first {n_new} "
                          f"accepted rows of {N32K_SHARD} in shard order"),
        "min_short_side_px": CROP_SIZE,
        "crop_rule": params["crop_rule"],
        "resize_rule": params["resize_rule"],
        "dedup_rule": ("SHA-1 of the cropped uint8 (192, 192) array against every image already "
                       f"in the dataset (the {n_base} of {BASE_ID}, ref and seed included) and "
                       "every earlier new row; a repeated digest is dropped and replaced by the "
                       "next row"),
        "near_duplicate_rule": evidence["near_duplicate_filter"]["rule"],
        "split_rule": f"{BASE_ID}'s splits; every appended image is in train",
        "orientation": params["orientation"],
        "identity_check": {k: v for k, v in evidence.items() if k != "counts"},
    }


# ------------------------------------------------------------------------------------------
# Imperative shell
# ------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Entry point of ``python -m ihdm.cli.build_diagnostic_photos``.

    Parameters
    ----------
    argv : list[str] | None
        Argument vector; ``None`` reads ``sys.argv[1:]``.

    Returns
    -------
    int
        0 on success, 1 if the written dataset does not validate.
    """
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = Path(args.data_root)
    out_dir = root / args.set
    if not args.force and _already_built(out_dir, EXPECTED_N[args.set]):
        print(f"OK {args.set}: already built at {out_dir} (use --force to rebuild)")
        return 0

    started = time.perf_counter()
    images, index, splits, meta = read_dataset(root / BASE_ID, mmap=False)
    base = BaseDataset(np.asarray(images), index, splits, meta)
    built = _build(args.set, base, args.batch_size)

    write_dataset(out_dir, built.images, built.index, built.splits, built.meta)
    violations = validate_dataset(out_dir)
    _write_qc(out_dir, args.set, base, built)
    _report(args.set, out_dir, built, violations, time.perf_counter() - started)
    return 0 if not violations else 1


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", required=True, choices=sorted(EXPECTED_N))
    parser.add_argument("--force", action="store_true", help="rebuild even if the output validates")
    parser.add_argument("--data-root", default=str(data_root()), help="standard-format data root")
    parser.add_argument("--batch-size", type=int, default=64, help="parquet rows read per batch")
    return parser.parse_args(argv)


def _already_built(out_dir: Path, expected_n: int) -> bool:
    """Return whether ``out_dir`` already holds a valid dataset of ``expected_n`` images."""
    if not (out_dir / "meta.json").is_file() or validate_dataset(out_dir):
        return False
    return int(np.load(out_dir / "images.npy", mmap_mode="r").shape[0]) == expected_n


def _build(dataset_id: str, base: BaseDataset, batch_size: int) -> Built:
    """Stream the right shards and run the builder of ``dataset_id``."""
    repo_id = base.meta.parameters["hf_dataset"]
    if dataset_id == R128_ID:
        shards = list(dict.fromkeys(shard for shard, _ in base_rows(base)))
        rows: Iterator[RawRow] = itertools.chain.from_iterable(
            iter_shard_rows(repo_id, shard, batch_size) for shard in shards
        )
        return build_r128(base, rows, _git_sha(), _raw_root(repo_id, shards[0]))
    rows = iter_shard_rows(repo_id, N32K_SHARD, batch_size)
    built = build_n32k(base, rows, _git_sha(), _raw_root(repo_id, N32K_SHARD),
                       n_new=N32K_NEW_TRAIN)
    built.meta.counts["shard_num_rows"] = {N32K_SHARD: shard_num_rows(repo_id, N32K_SHARD)}
    return built


def _raw_root(repo_id: str, shard: str) -> str:
    """The local directory of a downloaded shard (the HF cache snapshot)."""
    return str(download_shard(repo_id, shard).parent)


def _write_qc(out_dir: Path, dataset_id: str, base: BaseDataset, built: Built) -> None:
    """Render the QC artefacts into ``<out_dir>/qc/``."""
    qc = out_dir / "qc"
    qc.mkdir(parents=True, exist_ok=True)
    labels = built.index["split"].tolist()
    side = built.meta.image_size
    write_contact_sheet(qc / "contact_sheet.png", built.images, labels,
                        title=f"{dataset_id}: 64 random {side}x{side} images (idx, split)")
    write_intensity_hist(qc / "intensity_hist.png", built.images,
                         title=f"{dataset_id}: pooled intensity histogram, "
                               f"{len(built.images)} images")
    if dataset_id == R128_ID:
        picks = np.linspace(0, len(built.images) - 1, 16).astype(int).tolist()
        write_pair_sheet(qc / "pairs_vs_lsun_church.png", base.images, built.images, picks,
                         title=f"left: {BASE_ID} 192² native crop; right: {R128_ID} 128² "
                               "whole scene (same idx)")
        _write_collisions_md(qc / "duplicates.md", built.evidence["sha1_collisions"])
        return
    new = np.arange(base.images.shape[0], len(built.images))
    write_contact_sheet(qc / "contact_sheet_new_rows.png", built.images[new],
                        [f"{i}" for i in new], rng_seed=2027,
                        title=f"{dataset_id}: 64 random appended train images (idx)")
    write_duplicates_md(qc / "duplicates.md", dataset_id, built.duplicates,
                        built.meta.counts["new_rows_scan"])
    if built.screened_pairs:
        pairs = sorted(built.screened_pairs, key=lambda pair: -pair[2])
        write_pair_sheet(qc / "near_duplicates_rejected.png", np.stack([p[0] for p in pairs]),
                         np.stack([p[1] for p in pairs]), list(range(len(pairs))),
                         title=f"left: {BASE_ID} ref image; right: rejected train-shard row "
                               f"(thumbnail r > {NEAR_DUP_THRESHOLD}); {len(pairs)} rows, "
                               "highest r first")


def _write_collisions_md(path: Path, collisions: list[dict[str, Any]]) -> None:
    """The record-only SHA-1 collision report of ``lsun_church_r128``."""
    lines = [f"# Duplicate report — {R128_ID}", "",
             "Rule: SHA-1 of the cropped uint8 (128, 128) array, computed on every row. A",
             "collision is recorded here and in meta.json and never dropped: row identity",
             f"with {BASE_ID} comes first (T7.1).", "",
             f"- collisions: {len(collisions)}", ""]
    if collisions:
        lines += ["| idx | same SHA-1 as idx | sha1 |", "|---|---|---|"]
        lines += [f"| {c['idx']} | {c['first_idx']} | `{c['sha1']}` |" for c in collisions]
    else:
        lines.append("No two 128² crops share a SHA-1.")
    path.write_text("\n".join(lines) + "\n")


def _git_sha() -> str:
    """Return the git commit of the repository, or ``"unknown"`` outside a checkout."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
                             capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def _report(dataset_id: str, out_dir: Path, built: Built, violations: list[str],
            elapsed: float) -> None:
    """Print the violations and the final one-line summary."""
    for violation in violations:
        print(f"  VIOLATION {violation}", file=sys.stderr)
    status = "OK" if not violations else "FAIL"
    s = built.splits
    head = (f"{status} {dataset_id}: {len(built.images)} images "
            f"{built.images.shape[1]}x{built.images.shape[2]}, "
            f"{len(s['train'])}/{len(s['ref'])}/{len(s['seed'])} train/ref/seed")
    if dataset_id == R128_ID:
        corr = built.evidence["centre_correlation"]
        tail = (f"192² rows reproduced {built.evidence['rows_192_reproduced']}, centre r "
                f"min {corr['matched']['min']:.3f} median {corr['matched']['median']:.3f} "
                f"(null max {corr['null_shifted_by_half']['max']:.3f}), "
                f"{len(built.evidence['sha1_collisions'])} SHA-1 collisions")
    else:
        scan = built.meta.counts["new_rows_scan"]
        near = built.evidence["near_duplicate_filter"]
        tail = (f"new rows: scanned {scan['rows_scanned']}, rejected "
                f"{scan['rows_rejected_size']} by size and {scan['rows_rejected_unreadable']} "
                f"unreadable, {scan['duplicates_dropped']} SHA-1 duplicates dropped, "
                f"{near['n_rejected']} ref near-copies kept out (ref with a train near-copy: "
                f"{near['ref_with_near_copy_in_train']}), ref/seed pixel-identical")
    print(f"{head}, {tail}, {elapsed:.1f} s, {out_dir}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PreprocessError as error:
        print(f"FAIL {error}", file=sys.stderr)
        raise SystemExit(2) from error
