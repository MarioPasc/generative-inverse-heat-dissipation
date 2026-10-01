"""Unit tests of the M7 diagnostic photograph datasets (T7.1): r128 and n32k builders.

No network access: the base dataset and the shard rows are synthetic. The images are smooth
(low-resolution noise upsampled), so the centre correlation behaves as on photographs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from ihdm.cli import build_diagnostic_photos as cli
from ihdm.cli.build_diagnostic_photos import (
    BASE_ID,
    N32K_ID,
    N32K_SHARD,
    R128_ID,
    BaseDataset,
    build_n32k,
    build_r128,
)
from ihdm.cli.validate_dataset import main as validate_main
from ihdm.data.format import (
    DatasetMeta,
    read_dataset,
    split_by_subject,
    validate_dataset,
    write_dataset,
)
from ihdm.preprocess.errors import PreprocessError
from ihdm.preprocess.fetch_hf import PHOTO_SOURCES, RawRow
from ihdm.preprocess.photos import (
    NearDuplicateScreen,
    array_sha1,
    build_index,
    centre_correlation,
    collect_photos,
    near_duplicate_pairs,
    parse_source,
    select_rows,
    sha1_collisions,
    thumbnails,
    to_gray_crop,
    to_gray_resize_crop,
)

REPO = PHOTO_SOURCES[BASE_ID].repo_id
BASE_SHARD = PHOTO_SOURCES[BASE_ID].shards[0]
N_BASE = 20


def _smooth(width: int, height: int, seed: int) -> Image.Image:
    """A smooth, non-constant RGB image: 8x6 noise upsampled bicubically."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, size=(6, 8, 3), dtype=np.uint8)
    return Image.fromarray(small, "RGB").resize((width, height), Image.Resampling.BICUBIC)


def _base_stream() -> list[RawRow]:
    """22 rows of the base shard; rows 3 and 9 repeat rows 1 and 5, so 20 are accepted."""
    sizes = [(341, 256) if i % 3 else (256, 341) for i in range(22)]
    rows = [RawRow(BASE_SHARD, i, _smooth(*sizes[i], seed=i)) for i in range(22)]
    rows[3] = RawRow(BASE_SHARD, 3, rows[1].image)
    rows[9] = RawRow(BASE_SHARD, 9, rows[5].image)
    return rows


def _write_base(root: Path) -> BaseDataset:
    """Build and write a synthetic ``lsun_church`` (20 images, 16/4/2) with T1.2's code path."""
    result = collect_photos(_base_stream(), target=N_BASE, id_prefix="church", repo_id=REPO)
    splits = split_by_subject([r.image_id for r in result.records], rng_seed=2026, n_seed=2)
    meta = DatasetMeta(
        dataset_id=BASE_ID, n_images=N_BASE, image_size=192, dtype="uint8",
        pipeline="ihdm.preprocess.photos", pipeline_version="1.0", git_sha="test",
        created="2026-09-22T00:00:00+00:00", raw_root=str(root),
        parameters={"hf_dataset": REPO, "shards_used": [BASE_SHARD],
                    "crop_rule": "centre 192", "resize_rule": "none",
                    "orientation": "as published"},
        counts={"duplicates_dropped": result.counts["duplicates_dropped"]},
    )
    write_dataset(root / BASE_ID, result.images, build_index(result.records, splits), splits, meta)
    images, index, splits, meta = read_dataset(root / BASE_ID, mmap=False)
    return BaseDataset(np.asarray(images), index, splits, meta)


@pytest.fixture
def base(tmp_path: Path) -> BaseDataset:
    return _write_base(tmp_path)


# ------------------------------------------------------------------ pure functions


@pytest.mark.parametrize(
    ("width", "height"),
    [(341, 256), (256, 341), (256, 256), (300, 257), (257, 300), (129, 128), (128, 500)],
)
def test_resize_crop_equals_the_papers_torchvision_loader(width: int, height: int) -> None:
    """Resize(128, LANCZOS) + CenterCrop(128) on the luminance image, pixel for pixel."""
    transforms = pytest.importorskip("torchvision.transforms")
    loader = transforms.Compose([
        transforms.Resize(128, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(128),
    ])
    img = _smooth(width, height, seed=width * height)
    out = to_gray_resize_crop(img, 128)
    assert out.shape == (128, 128) and out.dtype == np.uint8
    np.testing.assert_array_equal(out, np.asarray(loader(img.convert("L"))))


def test_resize_crop_never_upsamples() -> None:
    with pytest.raises(PreprocessError, match="short side"):
        to_gray_resize_crop(_smooth(127, 300, seed=0), 128)


def test_parse_source_round_trip_and_errors() -> None:
    assert parse_source(f"{REPO}/{BASE_SHARD}/4001", REPO) == (BASE_SHARD, 4001)
    for bad in (f"other/{BASE_SHARD}/1", f"{REPO}/{BASE_SHARD}/x", f"{REPO}/7"):
        with pytest.raises(PreprocessError):
            parse_source(bad, REPO)


def test_select_rows_yields_in_order_and_stops_at_the_last() -> None:
    pulled: list[int] = []

    def stream():
        for i in range(10):
            pulled.append(i)
            yield RawRow("s", i, _smooth(8, 8, seed=i))

    got = [raw.row for raw in select_rows(stream(), [("s", 1), ("s", 4), ("s", 5)])]
    assert got == [1, 4, 5]
    assert pulled == [0, 1, 2, 3, 4, 5]
    assert list(select_rows(stream(), [])) == []


@pytest.mark.parametrize("wanted", [[("s", 2), ("s", 12)], [("s", 4), ("s", 2)], [("t", 0)]])
def test_select_rows_raises_when_a_row_is_missing_or_out_of_order(wanted) -> None:
    rows = [RawRow("s", i, _smooth(8, 8, seed=i)) for i in range(10)]
    with pytest.raises(PreprocessError, match="stream ended"):
        list(select_rows(rows, wanted))


def test_sha1_collisions_reports_and_keeps_every_row() -> None:
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, size=(6, 4, 4), dtype=np.uint8)
    images[4] = images[1]
    images[5] = images[1]
    found = sha1_collisions(images)
    assert [(i, j) for i, j, _ in found] == [(4, 1), (5, 1)]
    assert found[0][2] == array_sha1(images[1])
    assert sha1_collisions(images[:4]) == []


def test_centre_correlation_separates_same_and_different_photos() -> None:
    imgs = [_smooth(341, 256, seed=s) for s in range(6)]
    c192 = np.stack([to_gray_crop(im) for im in imgs])
    c128 = np.stack([to_gray_resize_crop(im, 128) for im in imgs])
    matched = centre_correlation(c192, c128)
    shuffled = centre_correlation(c192, c128[np.roll(np.arange(6), 3)])
    assert matched.min() > 0.98
    assert shuffled.max() < matched.min()
    with pytest.raises(PreprocessError):
        centre_correlation(c192, c192)


def test_collect_photos_extends_a_dataset_without_readmitting_its_images() -> None:
    rows = [RawRow("n", i, _smooth(300, 256, seed=100 + i)) for i in range(5)]
    already = to_gray_crop(rows[1].image)
    rows.insert(3, RawRow("n", 99, rows[0].image))  # repeats a new row
    result = collect_photos(rows, target=3, id_prefix="church", repo_id=REPO,
                            seen={array_sha1(already): 7}, start_idx=50)
    assert [r.idx for r in result.records] == [50, 51, 52]
    assert [r.image_id for r in result.records] == ["church_00050", "church_00051",
                                                    "church_00052"]
    assert [r.row for r in result.records] == [0, 2, 3]
    assert [(d.row, d.first_idx) for d in result.duplicates] == [(1, 7), (99, 50)]


def _reencoded(img: Image.Image, quality: int = 60, shift: int = 6) -> Image.Image:
    """A re-encoded, slightly brightened copy: the same photograph, different bytes."""
    import io

    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=quality)
    pixels = np.asarray(Image.open(io.BytesIO(buffer.getvalue())), np.int16) + shift
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), "RGB")


def test_thumbnails_dot_product_is_the_pearson_correlation() -> None:
    rng = np.random.default_rng(3)
    images = rng.integers(0, 256, size=(3, 64, 96), dtype=np.uint8)
    images[2] = 7
    thumbs = thumbnails(images)
    assert thumbs.shape == (3, 1024)
    pooled = images.astype(float).reshape(3, 32, 2, 32, 3).mean(axis=(2, 4)).reshape(3, -1)
    np.testing.assert_allclose(thumbs[0] @ thumbs[1], np.corrcoef(pooled[0], pooled[1])[0, 1],
                               rtol=1e-12)
    np.testing.assert_array_equal(thumbs[2], 0.0)
    with pytest.raises(PreprocessError, match="multiple"):
        thumbnails(images[:, :50])


def test_near_duplicate_screen_flags_a_reencoded_copy_only() -> None:
    refs = [_smooth(341, 256, seed=s) for s in range(4)]
    screen = NearDuplicateScreen(np.stack([to_gray_crop(im) for im in refs]), [10, 11, 12, 13])
    copy = to_gray_crop(_reencoded(refs[2]))
    assert not np.array_equal(copy, to_gray_crop(refs[2]))  # SHA-1 would not see it
    verdict = screen(copy)
    assert verdict is not None and verdict["ref_idx"] == 12 and verdict["max_r"] > 0.99
    assert screen(to_gray_crop(_smooth(341, 256, seed=77))) is None
    assert len(screen.rejected_crops) == 1


def test_collect_photos_replaces_screened_rows_and_keeps_t12_counts() -> None:
    rows = [RawRow("n", i, _smooth(300, 256, seed=200 + i)) for i in range(4)]
    plain = collect_photos(rows, target=3, id_prefix="church", repo_id=REPO)
    assert "rows_screened_out" not in plain.counts and plain.screened == []

    banned = to_gray_crop(rows[1].image)
    screen = NearDuplicateScreen(banned[None], [5])
    result = collect_photos(rows, target=3, id_prefix="church", repo_id=REPO, screen=screen)
    assert [r.row for r in result.records] == [0, 2, 3]
    assert result.counts["rows_screened_out"] == 1
    assert [(s.row, s.verdict["ref_idx"]) for s in result.screened] == [(1, 5)]


def test_near_duplicate_pairs_finds_planted_pairs_across_blocks() -> None:
    photos = [_smooth(341, 256, seed=s) for s in range(6)]
    images = np.stack([to_gray_crop(im) for im in photos])
    images[5] = to_gray_crop(_reencoded(photos[1]))
    images[4] = images[0]
    pairs = near_duplicate_pairs(images, threshold=0.95, block=2)
    assert [(i, j) for i, j, _ in pairs] == [(0, 4), (1, 5)]
    assert pairs[0][2] == pytest.approx(1.0)


# ------------------------------------------------------------------ r128


def test_build_r128_keeps_rows_index_and_splits(base: BaseDataset, tmp_path: Path) -> None:
    built = build_r128(base, iter(_base_stream()), git_sha="test", raw_root="raw")
    assert built.images.shape == (N_BASE, 128, 128) and built.images.dtype == np.uint8
    pd.testing.assert_frame_equal(built.index, base.index)
    assert built.splits == base.splits
    assert built.meta.image_size == 128 and built.meta.n_images == N_BASE
    assert built.meta.dataset_id == R128_ID

    # Row identity: the rows read back are T1.2's accepted rows, not the first 20 rows.
    rows = _base_stream()
    accepted = [r for r in rows if r.row not in (3, 9)]
    for idx, raw in enumerate(accepted):
        np.testing.assert_array_equal(built.images[idx], to_gray_resize_crop(raw.image, 128))
    evidence = built.evidence
    assert evidence["rows_192_reproduced"] == N_BASE
    assert evidence["sha1_collisions"] == []
    assert evidence["near_duplicate_pairs_within"]["pairs"] == []
    assert evidence["near_duplicate_pairs_within"]["pairs_in_lsun_church_192"] == []
    corr = evidence["centre_correlation"]
    assert corr["matched"]["min"] > corr["null_shifted_by_half"]["max"]
    assert corr["n_matched_below_null_max"] == 0

    write_dataset(tmp_path / R128_ID, built.images, built.index, built.splits, built.meta)
    assert validate_dataset(tmp_path / R128_ID) == []
    assert validate_main([str(tmp_path / R128_ID)]) == 0


def test_build_r128_refuses_a_substituted_row(base: BaseDataset) -> None:
    rows = _base_stream()
    rows[6] = RawRow(BASE_SHARD, 6, _smooth(341, 256, seed=999))
    with pytest.raises(PreprocessError, match="row identity broken"):
        build_r128(base, iter(rows), git_sha="test", raw_root="raw")


# ------------------------------------------------------------------ n32k


def _new_stream(base_stream: list[RawRow]) -> list[RawRow]:
    """The new shard: 10 rows with a base ref image, a base train image, an internal repeat,
    and a row below 192 px, so 6 of them are accepted."""
    rows = [RawRow(N32K_SHARD, i, _smooth(341, 256, seed=500 + i)) for i in range(10)]
    rows[1] = RawRow(N32K_SHARD, 1, base_stream[0].image)   # base idx 0
    rows[4] = RawRow(N32K_SHARD, 4, base_stream[2].image)   # base idx 2
    rows[6] = RawRow(N32K_SHARD, 6, rows[5].image)          # repeats a new row
    rows[8] = RawRow(N32K_SHARD, 8, _smooth(180, 256, seed=1))  # too small
    return rows


def test_build_n32k_appends_new_train_rows_only(base: BaseDataset, tmp_path: Path) -> None:
    built = build_n32k(base, iter(_new_stream(_base_stream())), git_sha="test",
                       raw_root="raw", n_new=6)
    n = N_BASE + 6
    assert built.images.shape == (n, 192, 192)
    np.testing.assert_array_equal(built.images[:N_BASE], base.images)
    for name in ("ref", "seed"):
        assert built.splits[name] == base.splits[name]
        np.testing.assert_array_equal(built.images[built.splits[name]],
                                      base.images[base.splits[name]])
    assert built.splits["train"] == sorted(base.splits["train"] + list(range(N_BASE, n)))
    pd.testing.assert_frame_equal(built.index.iloc[:N_BASE], base.index)
    new = built.index.iloc[N_BASE:]
    assert new["split"].eq("train").all()
    assert [parse_source(s, REPO) for s in new["source"]] == [
        (N32K_SHARD, r) for r in (0, 2, 3, 5, 7, 9)
    ]
    assert list(new["subject"]) == [f"church_{i:05d}" for i in range(N_BASE, n)]
    assert [(d.row, d.first_idx) for d in built.duplicates] == [(1, 0), (4, 2), (6, N_BASE + 3)]

    counts = built.meta.counts
    assert counts["composition"] == {"train_from_base": 16, "train_new": 6, "train_total": 22,
                                     "ref": 4, "seed_within_ref": 2}
    assert counts["new_rows_scan"]["duplicates_dropped"] == 3
    assert counts["new_rows_scan"]["rows_rejected_size"] == 1
    assert counts["new_rows_scan"]["rows_scanned"] == 10
    assert built.meta.parameters["shards_used"] == [BASE_SHARD, N32K_SHARD]

    write_dataset(tmp_path / N32K_ID, built.images, built.index, built.splits, built.meta)
    assert validate_dataset(tmp_path / N32K_ID) == []


def test_build_n32k_keeps_out_a_near_copy_of_a_ref_image(base: BaseDataset) -> None:
    accepted = [r for r in _base_stream() if r.row not in (3, 9)]
    ref_idx = base.splits["ref"][1]
    rows = _new_stream(_base_stream())
    rows.insert(2, RawRow(N32K_SHARD, 50, _reencoded(accepted[ref_idx].image)))
    built = build_n32k(base, iter(rows), git_sha="test", raw_root="raw", n_new=6)

    near = built.evidence["near_duplicate_filter"]
    assert near["n_rejected"] == 1 and near["threshold"] == 0.95
    assert near["rejected"][0]["row"] == 50 and near["rejected"][0]["ref_idx"] == ref_idx
    assert near["rejected"][0]["max_r"] > 0.99
    assert near["ref_with_near_copy_in_train"] == {BASE_ID: 0, N32K_ID: 0}
    assert built.meta.counts["new_rows_scan"]["rows_screened_out"] == 1
    assert len(built.images) == N_BASE + 6
    assert 50 not in [parse_source(s, REPO)[1] for s in built.index["source"].iloc[N_BASE:]]
    (ref_image, crop, r), = built.screened_pairs
    np.testing.assert_array_equal(ref_image, base.images[ref_idx])
    assert r == near["rejected"][0]["max_r"]


def test_build_n32k_fails_when_the_shard_runs_out(base: BaseDataset) -> None:
    with pytest.raises(PreprocessError, match="stream exhausted"):
        build_n32k(base, iter(_new_stream(_base_stream())), git_sha="test", raw_root="raw",
                   n_new=7)


# ------------------------------------------------------------------ CLI


def test_cli_builds_validates_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_base(tmp_path)
    streams = {BASE_SHARD: _base_stream()}
    streams[N32K_SHARD] = _new_stream(streams[BASE_SHARD])
    monkeypatch.setattr(cli, "iter_shard_rows", lambda repo, shard, bs=64: iter(streams[shard]))
    monkeypatch.setattr(cli, "shard_num_rows", lambda repo, shard: len(streams[shard]))
    monkeypatch.setattr(cli, "download_shard", lambda repo, shard: tmp_path / "hf" / shard)
    monkeypatch.setattr(cli, "N32K_NEW_TRAIN", 6)
    monkeypatch.setitem(cli.EXPECTED_N, R128_ID, N_BASE)
    monkeypatch.setitem(cli.EXPECTED_N, N32K_ID, N_BASE + 6)

    for dataset_id in (R128_ID, N32K_ID):
        assert cli.main(["--set", dataset_id, "--data-root", str(tmp_path)]) == 0
        assert capsys.readouterr().out.startswith(f"OK {dataset_id}:")
        assert validate_dataset(tmp_path / dataset_id) == []
        assert (tmp_path / dataset_id / "qc" / "contact_sheet.png").is_file()
        assert not list((tmp_path / dataset_id).glob("eval_seeds_*"))
        assert cli.main(["--set", dataset_id, "--data-root", str(tmp_path)]) == 0
        assert "already built" in capsys.readouterr().out

    assert (tmp_path / R128_ID / "qc" / "pairs_vs_lsun_church.png").is_file()
    assert (tmp_path / N32K_ID / "qc" / "contact_sheet_new_rows.png").is_file()
    meta = read_dataset(tmp_path / N32K_ID)[3]
    assert meta.counts["shard_num_rows"] == {N32K_SHARD: 10}
    assert (tmp_path / BASE_ID / "images.npy").stat().st_size > 0  # base untouched
