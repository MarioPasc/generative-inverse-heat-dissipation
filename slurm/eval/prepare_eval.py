"""One writer of the evaluation inputs (T5.1): seed lists and reference Inception features.

Run by ``slurm/eval/prepare_eval.sbatch`` on one A100 before any evaluation task. For every
dataset it

1. checks ``splits.json`` against the digest the lists were derived from locally;
2. draws the two D17 seed lists into a *shadow* copy of the dataset on local disk and compares
   their sha256 with ``expected_seed_lists.csv`` -- nothing is written to the real dataset unless
   every dataset matches;
3. writes (or reads back, when present) the lists and ``_features_inception_ref.npy`` in the real
   dataset directory, checks the lists again and that the reference matrix is finite
   ``(800, 2048)``;
4. checks that no file other than the six cache files appeared in the dataset directory.

Idempotent: a second run reads every file back and writes nothing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from ihdm.metrics.inception import (
    INCEPTION_FEATURE_DIM,
    inception_weights_path,
    reference_features,
)
from ihdm.metrics.run_eval import ensure_seed_lists, load_views

CACHE_FILES: tuple[str, ...] = (
    "eval_seeds_500.npy",
    "eval_seeds_500.json",
    "eval_seeds_final_2000.npy",
    "eval_seeds_final_2000.json",
    "_features_inception_ref.npy",
    "_features_inception_ref.json",
)
MAX_FILES_WRITTEN: int = 24
DATASET_INPUTS: tuple[str, ...] = ("images.npy", "index.csv", "splits.json", "meta.json")

logger = logging.getLogger("prepare_eval")


class PrepareError(Exception):
    """A dataset or a digest does not match what the evaluation expects."""


def _sha256_file(path: Path) -> str:
    """Return the sha256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_expected(path: Path) -> dict[str, dict[str, str]]:
    """Return ``{dataset: row}`` of ``expected_seed_lists.csv``."""
    with path.open() as handle:
        return {row["dataset"]: row for row in csv.DictReader(handle)}


def _shadow_dataset(source: Path, shadow: Path) -> Path:
    """Link the four inputs of a dataset into ``shadow`` and return it."""
    shadow.mkdir(parents=True, exist_ok=True)
    for name in DATASET_INPUTS:
        link = shadow / name
        if not link.exists():
            link.symlink_to(source / name)
    return shadow


def _check_lists(dataset: str, root: Path, expected: dict[str, str]) -> dict[str, str]:
    """Draw or read the lists under ``root`` and compare their digests with ``expected``."""
    lists = ensure_seed_lists(root)
    got = {"intermediate": lists.intermediate.sha256, "final": lists.final.sha256}
    want = {"intermediate": expected["intermediate_sha256"], "final": expected["final_sha256"]}
    if got != want:
        raise PrepareError(f"{dataset}: seed-list digests {got} under {root} differ from {want}")
    return got


def prepare(
    data_root: Path,
    datasets: list[str],
    expected_csv: Path,
    shadow_root: Path,
    device: str,
    fid_batch: int,
) -> list[dict[str, object]]:
    """Verify, then write, the one-writer caches of every dataset.

    Parameters
    ----------
    data_root : Path
        The directory holding the dataset folders.
    datasets : list[str]
        Dataset folder names.
    expected_csv : Path
        ``expected_seed_lists.csv`` with the locally computed digests.
    shadow_root : Path
        Local scratch directory for the dry draw of the lists.
    device : str
        Torch device of the Inception extractor.
    fid_batch : int
        Images per Inception forward pass.

    Returns
    -------
    list[dict[str, object]]
        One record per dataset with digests, timings and the files written.

    Raises
    ------
    PrepareError
        On any digest mismatch, a malformed reference matrix, or an unexpected file.
    """
    expected = _read_expected(expected_csv)
    missing = [name for name in datasets if name not in expected]
    if missing:
        raise PrepareError(f"no expected digests for {missing} in {expected_csv}")

    # Phase 1: nothing touches the real dataset until every dataset's inputs and lists check out.
    for name in datasets:
        source = data_root / name
        splits_sha = _sha256_file(source / "splits.json")
        if splits_sha != expected[name]["splits_json_sha256"]:
            raise PrepareError(
                f"{name}: splits.json sha256 {splits_sha} differs from the local "
                f"{expected[name]['splits_json_sha256']}"
            )
        got = _check_lists(name, _shadow_dataset(source, shadow_root / name), expected[name])
        logger.info("%s: shadow lists match (%s)", name, got)

    # Phase 2: the real writes.
    records: list[dict[str, object]] = []
    for name in datasets:
        root = data_root / name
        before = {path.name for path in root.iterdir()}
        got = _check_lists(name, root, expected[name])
        views = load_views(root)
        start = time.perf_counter()
        features = reference_features(
            root, views.reference, views.dataset_sha256, device=device, batch=fid_batch
        )
        elapsed = time.perf_counter() - start
        n_ref = int(views.reference.shape[0])
        if features.shape != (n_ref, INCEPTION_FEATURE_DIM) or not np.isfinite(features).all():
            raise PrepareError(f"{name}: reference features {features.shape} are not finite")
        after = {path.name for path in root.iterdir()}
        new = sorted(after - before)
        unexpected = [item for item in new if item not in CACHE_FILES]
        if unexpected:
            raise PrepareError(f"{name}: unexpected files appeared in {root}: {unexpected}")
        absent = [item for item in CACHE_FILES if item not in after]
        if absent:
            raise PrepareError(f"{name}: cache files still missing in {root}: {absent}")
        records.append(
            {
                "dataset": name,
                "root": str(root),
                "intermediate_sha256": got["intermediate"],
                "final_sha256": got["final"],
                "dataset_sha256": views.dataset_sha256,
                "n_reference": n_ref,
                "reference_features_sha256": _sha256_file(root / CACHE_FILES[4]),
                "reference_seconds": round(elapsed, 2),
                "files_written": new,
            }
        )
        logger.info("%s: %d new files, reference features in %.1f s", name, len(new), elapsed)

    total = sum(len(record["files_written"]) for record in records)  # type: ignore[arg-type]
    if total > MAX_FILES_WRITTEN:
        raise PrepareError(f"{total} files written, above the budget of {MAX_FILES_WRITTEN}")
    return records


def main(argv: list[str] | None = None) -> int:
    """Parse the command line, run :func:`prepare` and print one line per dataset."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--datasets", default="ixi,lsun_bedroom,lsun_church,oasis1")
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--shadow-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fid-batch", type=int, default=64)
    parser.add_argument("--out", type=Path, default=None, help="JSON record of the run")
    args = parser.parse_args(argv)

    print(f"inception weights path: {inception_weights_path()}")
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    try:
        records = prepare(
            args.data_root, datasets, args.expected, args.shadow_root, args.device, args.fid_batch
        )
    except PrepareError as error:
        print(f"FAIL {error}", file=sys.stderr)
        return 3
    for record in records:
        print(
            f"PREPARED {record['dataset']} intermediate={record['intermediate_sha256']} "
            f"final={record['final_sha256']} ref_features={record['reference_features_sha256']} "
            f"n_ref={record['n_reference']} ref_s={record['reference_seconds']} "
            f"new_files={len(record['files_written'])}"  # type: ignore[arg-type]
        )
    if args.out is not None:
        args.out.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
