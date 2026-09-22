"""Unit tests of the photograph pipeline: crop rule, acceptance pass, index, splits, QC.

No network access: every test builds synthetic PIL images and synthetic shard rows.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from ihdm.data.format import DatasetMeta, split_by_subject, validate_dataset, write_dataset
from ihdm.preprocess.errors import PreprocessError
from ihdm.preprocess.fetch_hf import PHOTO_SOURCES, RawRow
from ihdm.preprocess.photos import (
    CROP_SIZE,
    TARGET_SHORT_SIDE,
    array_sha1,
    build_index,
    collect_photos,
    fine_split_labels,
    image_id,
    needs_resize,
    to_gray_crop,
    write_contact_sheet,
    write_duplicates_md,
    write_intensity_hist,
)

RNG_SEED = 20260922


def _rgb_image(width: int, height: int, seed: int = 0) -> Image.Image:
    """Return a deterministic, non-constant RGB image of the given size."""
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8), "RGB")


def _raw_rows(sizes: list[tuple[int, int]], shard: str = "data/test.parquet") -> list[RawRow]:
    """Wrap synthetic images of the given ``(width, height)`` sizes as shard rows."""
    return [
        RawRow(shard=shard, row=i, image=_rgb_image(w, h, seed=i))
        for i, (w, h) in enumerate(sizes)
    ]


@pytest.mark.parametrize(
    ("width", "height"),
    [(256, 256), (256, 384), (384, 256), (192, 192), (200, 260)],
)
def test_to_gray_crop_no_resize_path(width: int, height: int) -> None:
    """A short side in [192, 256] is cropped from the native frame, never resampled."""
    img = _rgb_image(width, height, seed=1)
    assert not needs_resize(img)

    out = to_gray_crop(img)
    assert out.shape == (CROP_SIZE, CROP_SIZE)
    assert out.dtype == np.uint8

    gray = np.asarray(img.convert("L"), dtype=np.uint8)
    top, left = (height - CROP_SIZE) // 2, (width - CROP_SIZE) // 2
    np.testing.assert_array_equal(out, gray[top : top + CROP_SIZE, left : left + CROP_SIZE])


@pytest.mark.parametrize(("width", "height"), [(300, 300), (512, 384), (300, 400)])
def test_to_gray_crop_resize_path(width: int, height: int) -> None:
    """A short side above 256 px is LANCZOS-resized to 256 px, then cropped."""
    img = _rgb_image(width, height, seed=2)
    assert needs_resize(img)

    out = to_gray_crop(img)
    assert out.shape == (CROP_SIZE, CROP_SIZE)
    assert out.dtype == np.uint8

    scale = TARGET_SHORT_SIDE / min(width, height)
    if width <= height:
        expected_size = (TARGET_SHORT_SIDE, max(TARGET_SHORT_SIDE, round(height * scale)))
    else:
        expected_size = (max(TARGET_SHORT_SIDE, round(width * scale)), TARGET_SHORT_SIDE)
    resized = img.convert("L").resize(expected_size, Image.Resampling.LANCZOS)
    assert min(resized.size) == TARGET_SHORT_SIDE

    gray = np.asarray(resized, dtype=np.uint8)
    top = (gray.shape[0] - CROP_SIZE) // 2
    left = (gray.shape[1] - CROP_SIZE) // 2
    np.testing.assert_array_equal(out, gray[top : top + CROP_SIZE, left : left + CROP_SIZE])


@pytest.mark.parametrize(("width", "height"), [(100, 100), (191, 400), (400, 191), (1, 1)])
def test_to_gray_crop_rejects_small_images(width: int, height: int) -> None:
    """A short side below 192 px raises `PreprocessError`."""
    with pytest.raises(PreprocessError, match="short side"):
        to_gray_crop(_rgb_image(width, height, seed=3))


def test_to_gray_crop_is_idempotent_on_its_own_output() -> None:
    """Cropping an already-192 grayscale frame returns it unchanged."""
    out = to_gray_crop(_rgb_image(256, 256, seed=4))
    again = to_gray_crop(Image.fromarray(out, "L"))
    np.testing.assert_array_equal(out, again)


def test_array_sha1_depends_on_values_only() -> None:
    """Equal arrays hash equally; a single changed pixel changes the digest."""
    base = np.zeros((4, 4), dtype=np.uint8)
    assert array_sha1(base) == array_sha1(base.copy())
    changed = base.copy()
    changed[2, 2] = 1
    assert array_sha1(base) != array_sha1(changed)


def test_image_id_format() -> None:
    """Image ids are zero-padded to five digits."""
    assert image_id("church", 0) == "church_00000"
    assert image_id("bedroom", 3999) == "bedroom_03999"


def test_collect_photos_accepts_in_shard_order() -> None:
    """The accepted images are the first `target` usable rows, in stream order."""
    rows = _raw_rows([(256, 256)] * 5)
    result = collect_photos(rows, target=3, id_prefix="church", repo_id="owner/set")

    assert result.images.shape == (3, CROP_SIZE, CROP_SIZE)
    assert result.images.dtype == np.uint8
    assert [r.idx for r in result.records] == [0, 1, 2]
    assert [r.image_id for r in result.records] == ["church_00000", "church_00001", "church_00002"]
    assert [r.row for r in result.records] == [0, 1, 2]
    assert result.records[0].source == "owner/set/data/test.parquet/0"
    assert result.counts["rows_scanned"] == 3
    assert result.counts["n_accepted"] == 3
    assert result.counts["per_shard"]["data/test.parquet"] == {"scanned": 3, "accepted": 3}


def test_collect_photos_skips_small_rows_and_counts_them() -> None:
    """Rows below the minimum short side are rejected and replaced by later rows."""
    rows = _raw_rows([(256, 256), (100, 100), (256, 256), (150, 300), (256, 256)])
    result = collect_photos(rows, target=3, id_prefix="church", repo_id="owner/set")

    assert [r.row for r in result.records] == [0, 2, 4]
    assert result.counts["rows_rejected_size"] == 2
    assert result.counts["rows_rejected_unreadable"] == 0
    assert result.counts["rows_scanned"] == 5


def test_collect_photos_drops_duplicates_and_keeps_the_count() -> None:
    """A repeated crop is dropped, recorded, and replaced by the next distinct row."""
    rows = _raw_rows([(256, 256), (256, 256), (256, 256)])
    rows[1] = RawRow(shard=rows[1].shard, row=1, image=rows[0].image.copy())

    result = collect_photos(rows, target=2, id_prefix="church", repo_id="owner/set")

    assert result.counts["duplicates_dropped"] == 1
    assert result.counts["n_accepted"] == 2
    assert [r.row for r in result.records] == [0, 2]
    assert len(result.duplicates) == 1
    assert result.duplicates[0].row == 1
    assert result.duplicates[0].first_idx == 0
    assert len({r.sha1 for r in result.records}) == 2


def test_collect_photos_counts_resized_rows() -> None:
    """Rows whose short side exceeds 256 px are counted as resampled."""
    result = collect_photos(
        _raw_rows([(300, 300), (256, 256)]),
        target=2,
        id_prefix="church",
        repo_id="owner/set",
    )
    assert result.counts["resized"] == 1
    assert [r.resized for r in result.records] == [True, False]


def test_collect_photos_calls_on_accept_once_per_image() -> None:
    """The acceptance callback sees every accepted image exactly once, in order."""
    seen: list[tuple[int, int]] = []
    collect_photos(
        _raw_rows([(256, 256), (100, 100), (256, 256)]),
        target=2,
        id_prefix="church",
        repo_id="owner/set",
        on_accept=lambda idx, raw: seen.append((idx, raw.row)),
    )
    assert seen == [(0, 0), (1, 2)]


def test_collect_photos_raises_when_the_stream_is_short() -> None:
    """An exhausted stream is a contract violation, not a silently smaller dataset."""
    with pytest.raises(PreprocessError, match="stream exhausted"):
        collect_photos(_raw_rows([(256, 256)] * 2), target=5, id_prefix="c", repo_id="owner/set")


def test_fine_split_labels_prefers_seed_over_ref() -> None:
    """The per-row label is the finest one: `seed` wins over `ref`."""
    splits = {"train": [0, 1], "ref": [2, 3], "seed": [3]}
    assert fine_split_labels(splits, 4) == ["train", "train", "ref", "seed"]


def test_build_index_matches_the_frozen_contract() -> None:
    """`index.csv` columns, ids, slice, z_mm and split labels follow 03-data-format.md."""
    n = 50
    rows = _raw_rows([(256, 256)] * n)
    result = collect_photos(rows, target=n, id_prefix="church", repo_id="owner/set")

    splits = split_by_subject([r.image_id for r in result.records], rng_seed=2026)
    index = build_index(result.records, splits)

    assert list(index.columns) == ["idx", "subject", "slice", "z_mm", "source", "split"]
    assert index["idx"].tolist() == list(range(n))
    assert index["subject"].tolist() == [image_id("church", i) for i in range(n)]
    assert set(index["slice"]) == {0}
    assert index["z_mm"].isna().all()
    assert set(index["split"]) <= {"train", "ref", "seed"}
    assert index["source"].iloc[7] == "owner/set/data/test.parquet/7"


def test_split_by_subject_on_photograph_ids_partitions_by_image() -> None:
    """Each photograph is its own subject, so the split is an image-level 80/20 partition."""
    ids = [image_id("church", i) for i in range(50)]
    splits = split_by_subject(ids, rng_seed=2026)

    assert len(splits["train"]) == 40
    assert len(splits["ref"]) == 10
    assert set(splits["train"]) | set(splits["ref"]) == set(range(50))
    assert not set(splits["train"]) & set(splits["ref"])
    assert set(splits["seed"]).issubset(set(splits["ref"]))
    assert splits["train"] == sorted(splits["train"])
    assert splits["rng_seed"] == 2026


def test_split_by_subject_at_the_real_size() -> None:
    """At 4000 images the split is exactly 3200 train / 800 ref / 40 seed."""
    ids = [image_id("church", i) for i in range(4000)]
    splits = split_by_subject(ids, rng_seed=2026)
    assert (len(splits["train"]), len(splits["ref"]), len(splits["seed"])) == (3200, 800, 40)


def test_end_to_end_write_and_validate(tmp_path: Path) -> None:
    """A dataset assembled from the pipeline's own pieces passes `validate_dataset`."""
    n = 50
    rng = np.random.default_rng(RNG_SEED)
    images = rng.integers(0, 256, size=(n, CROP_SIZE, CROP_SIZE), dtype=np.uint8)
    records_ids = [image_id("church", i) for i in range(n)]

    splits = split_by_subject(records_ids, rng_seed=2026)
    index = pd.DataFrame(
        {
            "idx": list(range(n)),
            "subject": records_ids,
            "slice": [0] * n,
            "z_mm": [float("nan")] * n,
            "source": [f"owner/set/data/test.parquet/{i}" for i in range(n)],
            "split": fine_split_labels(splits, n),
        }
    )
    meta = DatasetMeta(
        dataset_id="lsun_church",
        n_images=n,
        image_size=CROP_SIZE,
        dtype="uint8",
        pipeline="ihdm.preprocess.photos",
        pipeline_version="1.0",
        git_sha="0" * 40,
        created=datetime.now(UTC).isoformat(timespec="seconds"),
        raw_root=str(tmp_path / "raw"),
        parameters={"crop_rule": "centre 192", "dedup_rule": "sha1"},
        counts={"padding_fraction": 0.0, "duplicates_dropped": 0, "resized": 0},
    )

    write_dataset(tmp_path / "lsun_church", images, index, splits, meta)
    assert validate_dataset(tmp_path / "lsun_church") == []


def test_qc_writers_produce_their_files(tmp_path: Path) -> None:
    """The three QC artefacts are written and are non-empty."""
    rng = np.random.default_rng(RNG_SEED)
    images = rng.integers(0, 256, size=(12, CROP_SIZE, CROP_SIZE), dtype=np.uint8)
    labels = ["train"] * 10 + ["ref", "seed"]

    write_contact_sheet(tmp_path / "contact_sheet.png", images, labels, n_tiles=9, title="t")
    write_intensity_hist(tmp_path / "intensity_hist.png", images, title="t")
    write_duplicates_md(
        tmp_path / "duplicates.md",
        "lsun_church",
        [],
        {"rows_scanned": 12, "n_accepted": 12, "duplicates_dropped": 0},
    )

    for name in ("contact_sheet.png", "intensity_hist.png", "duplicates.md"):
        assert (tmp_path / name).stat().st_size > 0
    assert "No duplicate" in (tmp_path / "duplicates.md").read_text()


def test_photo_sources_declare_the_control_shards() -> None:
    """The declared first shard of each set is the test shard that produced the controls."""
    assert set(PHOTO_SOURCES) == {"lsun_church", "lsun_bedroom"}
    church, bedroom = PHOTO_SOURCES["lsun_church"], PHOTO_SOURCES["lsun_bedroom"]
    assert church.repo_id == "tglcourse/lsun_church_train"
    assert bedroom.repo_id == "pcuenq/lsun-bedrooms"
    assert church.shards[0].startswith("data/test-")
    assert bedroom.shards[0].startswith("data/test-")
    assert (church.id_prefix, bedroom.id_prefix) == ("church", "bedroom")
