"""Build a standard-format photograph dataset from its Hugging Face shards.

Imperative shell of the photograph pipeline: stream the shards in order, accept the
first ``--target`` usable and distinct rows, append the missing raw grayscale PNGs to
``NATURAL_CONTROLS/<set>/``, write the dataset through :func:`ihdm.data.format.write_dataset`,
validate it, render the QC sheets, and check that the leading images still equal the
existing 2000-image controls.

Run as ``python -m ihdm.cli.preprocess_photos --set lsun_church|lsun_bedroom``.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ihdm.data.format import DatasetMeta, split_by_subject, validate_dataset, write_dataset
from ihdm.paths import data_root, raw_root
from ihdm.preprocess.errors import PreprocessError
from ihdm.preprocess.fetch_hf import (
    PHOTO_SOURCES,
    PhotoSource,
    RawRow,
    iter_source_rows,
    shard_num_rows,
)
from ihdm.preprocess.photos import (
    CROP_SIZE,
    TARGET_SHORT_SIDE,
    CollectResult,
    PhotoRecord,
    build_index,
    collect_photos,
    to_gray_crop,
    write_contact_sheet,
    write_duplicates_md,
    write_intensity_hist,
)

logger = logging.getLogger(__name__)

PIPELINE = "ihdm.preprocess.photos"
PIPELINE_VERSION = "1.0"
RAW_SUBDIR = "NATURAL_CONTROLS"
N_VERIFY = 20


def main(argv: list[str] | None = None) -> int:
    """Entry point of ``python -m ihdm.cli.preprocess_photos``.

    Parameters
    ----------
    argv : list[str] | None
        Argument vector; ``None`` reads ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code: 0 on success, 1 if the written dataset does not validate.
    """
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    source = PHOTO_SOURCES[args.set]
    out_dir = Path(args.data_root) / source.dataset_id
    raw_dir = Path(args.raw_root) / RAW_SUBDIR / source.dataset_id

    if not args.force and _already_built(out_dir, args.target):
        print(f"OK {source.dataset_id}: already built at {out_dir} (use --force to rebuild)")
        return 0

    started = time.perf_counter()
    raw_dir.mkdir(parents=True, exist_ok=True)
    n_existing = len(list(raw_dir.glob("[0-9]*.png")))
    logger.info("%s: %d existing raw PNGs in %s", source.dataset_id, n_existing, raw_dir)

    result = _collect(source, args.target, raw_dir, n_existing, args.batch_size)
    match = _verify_against_controls(result.images, result.records, raw_dir, n_existing)
    shard_rows = {
        shard: shard_num_rows(source.repo_id, shard) for shard in result.counts["per_shard"]
    }

    splits = split_by_subject([r.image_id for r in result.records], rng_seed=2026)
    index = build_index(result.records, splits)
    meta = _build_meta(source, result, raw_dir, shard_rows, match, args.target)

    write_dataset(out_dir, result.images, index, splits, meta)
    violations = validate_dataset(out_dir)
    _write_qc(out_dir, source.dataset_id, result, index["split"].tolist())

    elapsed = time.perf_counter() - started
    _report(source, out_dir, result, splits, match, violations, elapsed, shard_rows)
    return 0 if not violations else 1


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", required=True, choices=sorted(PHOTO_SOURCES))
    parser.add_argument("--target", type=int, default=4000, help="number of images to accept")
    parser.add_argument("--force", action="store_true", help="rebuild even if the output validates")
    parser.add_argument("--data-root", default=str(data_root()), help="standard-format data root")
    parser.add_argument("--raw-root", default=str(raw_root()), help="root holding NATURAL_CONTROLS")
    parser.add_argument("--batch-size", type=int, default=64, help="parquet rows read per batch")
    return parser.parse_args(argv)


def _already_built(out_dir: Path, target: int) -> bool:
    """Return whether ``out_dir`` already holds a valid dataset of ``target`` images."""
    if not (out_dir / "meta.json").is_file():
        return False
    if validate_dataset(out_dir):
        return False
    return int(np.load(out_dir / "images.npy", mmap_mode="r").shape[0]) == target


def _collect(
    source: PhotoSource, target: int, raw_dir: Path, n_existing: int, batch_size: int
) -> CollectResult:
    """Run the acceptance pass, appending raw PNGs beyond the existing numbering."""

    def on_accept(idx: int, raw: RawRow) -> None:
        # The raw folder is numbered by SHARD ROW, the convention the pre-existing 2000
        # controls already follow (PNG `r` is shard row `r`, verified by pixel equality).
        # Accepted index and shard row diverge as soon as a duplicate is dropped, so
        # numbering the new files by index would give the folder two meanings.
        png = raw_dir / f"{raw.row:05d}.png"
        if png.exists():
            return  # never rewrite an existing control
        raw.image.convert("L").save(png)

    return collect_photos(
        iter_source_rows(source, batch_size=batch_size),
        target=target,
        id_prefix=source.id_prefix,
        repo_id=source.repo_id,
        on_accept=on_accept,
    )


def _verify_against_controls(
    images: np.ndarray, records: list[PhotoRecord], raw_dir: Path, n_existing: int
) -> dict[str, Any]:
    """Check the accepted images against the pre-existing control PNGs.

    Two separate facts are reported. ``all_match`` says whether every sampled
    accepted image equals the crop of the control PNG of **its own shard row**: that
    is the real claim, "the same shards, the same rows, the same conversion". Index
    alignment is reported separately, because de-duplication shifts the accepted
    index away from the shard row: the controls were fetched without a duplicate
    filter, so a dropped row makes every later accepted image sit one position
    earlier than the identically numbered control PNG.

    Parameters
    ----------
    images : np.ndarray
        The accepted ``uint8`` images.
    records : list[PhotoRecord]
        Accepted records, carrying the shard row of each image.
    raw_dir : Path
        Raw folder holding the control PNGs, numbered by shard row.
    n_existing : int
        Number of PNGs that existed before this run.

    Returns
    -------
    dict[str, Any]
        ``n_existing_before``, ``n_checked``, ``checked_idx``, ``mismatched_idx``,
        ``all_match``, ``n_index_shifted``, ``first_shifted_idx`` and ``note``.
    """
    covered = [r for r in records if r.row < n_existing]
    shifted = [r.idx for r in covered if r.row != r.idx]
    result: dict[str, Any] = {
        "n_existing_before": n_existing,
        "n_covered_by_controls": len(covered),
        "n_index_shifted": len(shifted),
        "first_shifted_idx": min(shifted) if shifted else None,
        "n_checked": 0,
        "checked_idx": [],
        "mismatched_idx": [],
        "all_match": None,
        "note": (
            "accepted image idx is compared against the control PNG of its own shard row; "
            "n_index_shifted counts the images whose accepted idx differs from that row "
            "because an earlier duplicate row was dropped"
        ),
    }
    if not covered:
        return result

    picks = np.unique(np.linspace(0, len(covered) - 1, min(N_VERIFY, len(covered))).astype(int))
    chosen = [covered[int(p)] for p in picks]
    mismatched = [
        rec.idx
        for rec in chosen
        if not np.array_equal(
            to_gray_crop(Image.open(raw_dir / f"{rec.row:05d}.png")), images[rec.idx]
        )
    ]
    result["n_checked"] = len(chosen)
    result["checked_idx"] = [rec.idx for rec in chosen]
    result["mismatched_idx"] = mismatched
    result["all_match"] = not mismatched
    return result


def _build_meta(
    source: PhotoSource,
    result: CollectResult,
    raw_dir: Path,
    shard_rows: dict[str, int],
    match: dict[str, Any],
    target: int,
) -> DatasetMeta:
    """Assemble the ``meta.json`` payload of a photograph dataset."""
    counts = dict(result.counts)
    counts["shard_num_rows"] = shard_rows
    counts["padding_fraction"] = 0.0
    counts["n_target"] = target

    parameters = {
        "hf_dataset": source.repo_id,
        "shards_used": sorted(result.counts["per_shard"]),
        "shards_declared": list(source.shards),
        "row_selection": f"shard order, first {target} accepted rows",
        "min_short_side_px": CROP_SIZE,
        "crop_rule": (
            f"PIL convert('L'), then the centre {CROP_SIZE}x{CROP_SIZE} crop of the native frame"
        ),
        "resize_rule": (
            f"short side > {TARGET_SHORT_SIDE} px: LANCZOS resize to a {TARGET_SHORT_SIDE} px "
            "short side, then crop; short side in [192, 256]: no resampling"
        ),
        "dedup_rule": (
            "SHA-1 of the cropped uint8 (192, 192) array; a repeated digest is dropped and "
            "replaced by the next accepted row"
        ),
        "split_rule": "image-level split_by_subject, rng 2026, 80/20, 40 seed images",
        "raw_png_rule": (
            "the raw folder holds the grayscale (convert('L')) source frame at native size, "
            "named by SHARD ROW (<row>.png), the convention of the pre-existing 2000 controls; "
            "an existing file is never rewritten; index.csv.source carries the row, so the "
            "raw file of image idx is the row at the end of its source string"
        ),
        "orientation": "photograph as published; native pixel grid, no flip",
        "first_2000_check": match,
    }

    return DatasetMeta(
        dataset_id=source.dataset_id,
        n_images=int(result.images.shape[0]),
        image_size=CROP_SIZE,
        dtype="uint8",
        pipeline=PIPELINE,
        pipeline_version=PIPELINE_VERSION,
        git_sha=_git_sha(),
        created=datetime.now(UTC).isoformat(timespec="seconds"),
        raw_root=str(raw_dir),
        parameters=parameters,
        counts=counts,
    )


def _write_qc(out_dir: Path, dataset_id: str, result: CollectResult, labels: list[str]) -> None:
    """Render the three QC artefacts into ``<out_dir>/qc/``."""
    qc = out_dir / "qc"
    qc.mkdir(parents=True, exist_ok=True)
    write_contact_sheet(
        qc / "contact_sheet.png",
        result.images,
        labels,
        title=f"{dataset_id}: 64 random {CROP_SIZE}x{CROP_SIZE} crops (idx, split)",
    )
    write_intensity_hist(
        qc / "intensity_hist.png",
        result.images,
        title=f"{dataset_id}: pooled intensity histogram, {len(result.images)} images",
    )
    write_duplicates_md(qc / "duplicates.md", dataset_id, result.duplicates, result.counts)


def _git_sha() -> str:
    """Return the git commit of the repository, or ``"unknown"`` outside a checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def _report(
    source: PhotoSource,
    out_dir: Path,
    result: CollectResult,
    splits: dict[str, Any],
    match: dict[str, Any],
    violations: list[str],
    elapsed: float,
    shard_rows: dict[str, int],
) -> None:
    """Print the per-shard detail and the final one-line summary."""
    for shard, per in result.counts["per_shard"].items():
        print(
            f"  {shard}: {shard_rows.get(shard, -1)} rows in shard, "
            f"{per['scanned']} scanned, {per['accepted']} accepted"
        )
    for violation in violations:
        print(f"  VIOLATION {violation}", file=sys.stderr)

    status = "OK" if not violations else "FAIL"
    counts = result.counts
    print(
        f"{status} {source.dataset_id}: {counts['n_accepted']} images, "
        f"{len(splits['train'])}/{len(splits['ref'])}/{len(splits['seed'])} train/ref/seed, "
        f"scanned {counts['rows_scanned']}, rejected {counts['rows_rejected_size']} by size, "
        f"{counts['duplicates_dropped']} duplicates dropped, {counts['resized']} resized, "
        f"control match={match['all_match']} (checked {match['n_checked']} of "
        f"{match['n_covered_by_controls']} covered by the first {match['n_existing_before']} "
        f"PNGs, {match['n_index_shifted']} index-shifted from "
        f"{match['first_shifted_idx']}), {elapsed:.1f} s, {out_dir}"
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PreprocessError as error:
        print(f"FAIL {error}", file=sys.stderr)
        raise SystemExit(2) from error
