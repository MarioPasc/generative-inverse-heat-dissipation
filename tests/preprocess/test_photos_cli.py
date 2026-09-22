"""Unit tests of the photograph CLI helpers that do not need the network.

The load-bearing one is `_verify_against_controls`: it must compare an accepted image
against the control PNG of its own **shard row**, and report index drift separately,
because de-duplication moves accepted indices away from shard rows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ihdm.cli.preprocess_photos import _already_built, _verify_against_controls
from ihdm.preprocess.photos import CROP_SIZE, PhotoRecord

RNG_SEED = 20260922


def _write_controls(raw_dir: Path, arrays: list[np.ndarray]) -> None:
    """Save one grayscale control PNG per shard row, named by row."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for row, array in enumerate(arrays):
        Image.fromarray(array, "L").save(raw_dir / f"{row:05d}.png")


def _record(idx: int, row: int) -> PhotoRecord:
    """Build a record whose accepted index and shard row can differ."""
    return PhotoRecord(
        idx=idx,
        image_id=f"church_{idx:05d}",
        shard="data/test.parquet",
        row=row,
        source=f"owner/set/data/test.parquet/{row}",
        sha1=f"{idx:040d}",
        resized=False,
    )


def test_verify_matches_by_shard_row_and_reports_the_index_shift(tmp_path: Path) -> None:
    """A dropped duplicate shifts the index but not the row-wise equality."""
    rng = np.random.default_rng(RNG_SEED)
    rows = [
        rng.integers(0, 256, size=(CROP_SIZE, CROP_SIZE), dtype=np.uint8) for _ in range(5)
    ]
    _write_controls(tmp_path, rows)

    # Row 2 was dropped as a duplicate, so accepted idx 2 holds shard row 3, and so on.
    records = [_record(0, 0), _record(1, 1), _record(2, 3), _record(3, 4)]
    images = np.stack([rows[0], rows[1], rows[3], rows[4]])

    out = _verify_against_controls(images, records, tmp_path, n_existing=5)

    assert out["all_match"] is True
    assert out["mismatched_idx"] == []
    assert out["n_checked"] == 4
    assert out["n_covered_by_controls"] == 4
    assert out["n_index_shifted"] == 2
    assert out["first_shifted_idx"] == 2


def test_verify_flags_a_real_content_mismatch(tmp_path: Path) -> None:
    """An image that is not the crop of its own control PNG is reported by index."""
    rng = np.random.default_rng(RNG_SEED)
    rows = [
        rng.integers(0, 256, size=(CROP_SIZE, CROP_SIZE), dtype=np.uint8) for _ in range(3)
    ]
    _write_controls(tmp_path, rows)

    images = np.stack(rows)
    images[1] = 255 - images[1]
    records = [_record(i, i) for i in range(3)]

    out = _verify_against_controls(images, records, tmp_path, n_existing=3)

    assert out["all_match"] is False
    assert out["mismatched_idx"] == [1]
    assert out["n_index_shifted"] == 0


def test_verify_without_any_control_png(tmp_path: Path) -> None:
    """A first-ever build reports no verdict rather than a false pass."""
    images = np.zeros((2, CROP_SIZE, CROP_SIZE), dtype=np.uint8)
    out = _verify_against_controls(images, [_record(0, 0), _record(1, 1)], tmp_path, n_existing=0)

    assert out["all_match"] is None
    assert out["n_checked"] == 0
    assert out["n_covered_by_controls"] == 0


def test_already_built_is_false_without_an_output(tmp_path: Path) -> None:
    """A missing or incomplete output directory never counts as built."""
    assert _already_built(tmp_path / "lsun_church", target=4000) is False
