"""Tests of F1, the visual abstract (T8.1): mapping, level counts, numbers, rules, CLI, bytes."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ihdm.analysis import paper_f1 as f1
from ihdm.cli import paper_f1 as cli
from ihdm.data.format import DatasetMeta, write_dataset

REPO = Path(__file__).resolve().parents[2]
WIDTH = 192
DATA_ROOT = Path("/media/mpascual/MeningD2/spectral_allocation_heat_diffusion_project")

#: The ticket's numbers, typed here (and only here) to check the figure's strings against them.
TICKET_STRINGS = {"ixi_coarse": "4.1%", "ixi_peak": "29.8%", "church_coarse": "23.8%",
                  "inherited_default": "0.3%", "inherited_matched": "8.5%", "kid": "−63%",
                  "seed_nn": ("6%", "72%")}

#: Rules of the tiny synthetic campaign: 6 subjects x 10 slices, train = the first three.
TINY_RULES = replace(f1.DEFAULT_RULES, n_train=30, teaser_dataset_index=35)


def _smooth_stack(n: int, rng: np.random.Generator) -> np.ndarray:
    """Images with variance in every octave: white noise plus a coarse random field."""
    coarse = f1.heat_blur(rng.normal(size=(n, WIDTH, WIDTH)), 6.0) * 20.0
    fine = rng.normal(size=(n, WIDTH, WIDTH)) * 0.1
    return 0.5 + 0.2 * (coarse + fine)


# --------------------------------------------------------------------------------------------
# 1. The frequency mapping
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("sigma", f1.VERIFY_SIGMAS)
def test_heat_blur_equals_released_dctblur(sigma: float) -> None:
    images = np.random.default_rng(0).random((3, WIDTH, WIDTH))
    np.testing.assert_allclose(f1.heat_blur(images, sigma), f1.released_dct_blur(images, sigma),
                               rtol=0.0, atol=1e-12)


def test_heat_blur_identity_at_zero_and_single_image() -> None:
    image = np.random.default_rng(1).random((WIDTH, WIDTH))
    np.testing.assert_array_equal(f1.heat_blur(image, 0.0), image)
    np.testing.assert_allclose(f1.heat_blur(image, 8.0), f1.heat_blur(image[None], 8.0)[0],
                               rtol=0.0, atol=1e-15)


def test_kept_variance_matches_prediction() -> None:
    stack = _smooth_stack(12, np.random.default_rng(2))
    checks = f1.verify_mapping(stack, f1.VERIFY_SIGMAS, chunk=5)
    assert [c.sigma for c in checks] == list(f1.VERIFY_SIGMAS)
    for check in checks:
        assert check.passed, check
        np.testing.assert_allclose(check.measured, check.predicted, rtol=f1.MAPPING_RTOL,
                                   atol=f1.MAPPING_ATOL)


def test_mapping_check_detects_a_wrong_blur(monkeypatch: pytest.MonkeyPatch) -> None:
    released = f1.released_dct_blur
    monkeypatch.setattr(f1, "released_dct_blur", lambda x, s: released(x, s * 1.01))
    checks = f1.verify_mapping(_smooth_stack(6, np.random.default_rng(3)), (8.0, 24.0))
    assert not any(c.passed for c in checks)


def test_mapping_needs_two_images() -> None:
    with pytest.raises(f1.PaperFigureError):
        f1.kept_variance_per_octave(np.zeros((1, WIDTH, WIDTH)), (8.0,))


@pytest.mark.parametrize("sigma", f1.VERIFY_SIGMAS)
def test_half_power_and_mode_scale(sigma: float) -> None:
    assert round(f1.mode_scale(WIDTH), 1) == 43.2
    assert round(f1.half_power_cycles(1.0, WIDTH), 1) == 25.4
    numeric = f1.half_power_cycles_numeric(sigma, WIDTH)
    assert abs(25.4 / sigma - numeric) / numeric < 0.01
    np.testing.assert_allclose(numeric, f1.half_power_cycles(sigma, WIDTH), rtol=1e-10)
    # At c = 43.2/sigma the mode's length-scale equals sigma, so d^2 = exp(-2).
    c = f1.cycles_of_sigma(sigma, WIDTH)
    np.testing.assert_allclose(f1.kept_power(np.array([c]), sigma, WIDTH), np.exp(-2.0),
                               rtol=1e-12)


def test_kept_power_matches_dct_multiplier() -> None:
    d = f1.heat_multiplier(WIDTH, 24.0)
    n = np.hypot(*np.meshgrid(np.arange(WIDTH), np.arange(WIDTH), indexing="ij"))
    np.testing.assert_allclose(d**2, f1.kept_power(n / 2.0, 24.0, WIDTH), rtol=1e-12)


@pytest.mark.integration
def test_mapping_on_ixi_train() -> None:
    if not (DATA_ROOT / "ixi" / "images.npy").is_file():
        pytest.skip("IXI data root not mounted")
    from ihdm.data.format import read_dataset

    images, _, splits, _ = read_dataset(DATA_ROOT / "ixi")
    train = images[np.asarray(sorted(splits["train"]))]
    assert all(c.passed for c in f1.verify_mapping(train))


# --------------------------------------------------------------------------------------------
# 2. Level counts
# --------------------------------------------------------------------------------------------


def _repo_schedules() -> dict[str, np.ndarray]:
    return {name: np.load(REPO / "schedules" / f"{name}.npy") for name in f1.SCHEDULES.values()}


def test_level_counts_equal_data_profile_and_ticket() -> None:
    table = f1.parse_level_table(REPO / "docs" / "RESULTS" / "data_profile.md")
    assert table["log_W2"] == (27, 26, 26, 26, 27, 26, 26, 16)
    assert table["ixi_W8"] == (24, 37, 46, 43, 32, 18, 0, 0)
    counts = f1.verify_level_counts(_repo_schedules(), table)
    assert counts == f1.EXPECTED_LEVEL_COUNTS
    assert all(sum(c) == 200 for c in counts.values())


def test_level_count_mismatch_raises() -> None:
    table = f1.parse_level_table(REPO / "docs" / "RESULTS" / "data_profile.md")
    table["ixi_W8"] = (25, 36, 46, 43, 32, 18, 0, 0)
    with pytest.raises(f1.PaperFigureError, match="ixi_W8"):
        f1.verify_level_counts(_repo_schedules(), table)


def _direct_frequency_histogram(levels: np.ndarray) -> list[int]:
    """Independent count: each level's c = 43.2/sigma, placed by a linear scan of the edges."""
    edges = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 96.0]
    counts = [0] * 8
    for sigma in levels:
        c = WIDTH / (np.sqrt(2.0) * np.pi * sigma)
        k = 0 if c < edges[1] else 7 if c >= edges[7] else next(
            j for j in range(8) if edges[j] <= c < edges[j + 1])
        counts[k] += 1
    return counts


@pytest.mark.parametrize(("name", "expected", "below"), [
    ("log_W2", (31, 26, 26, 26, 27, 26, 26, 12), 4),
    ("ixi_W8", (0, 5, 30, 38, 47, 42, 29, 9), 0),
])
def test_frequency_octave_counts(name: str, expected: tuple[int, ...], below: int) -> None:
    levels = np.load(REPO / "schedules" / f"{name}.npy")[1:]
    counts, folded = f1.levels_per_frequency_octave(levels, WIDTH)
    assert sum(counts) == 200
    assert list(counts) == _direct_frequency_histogram(levels)
    assert counts == expected and folded == below


def test_frequency_octave_counts_reject_level_zero() -> None:
    with pytest.raises(f1.PaperFigureError):
        f1.levels_per_frequency_octave(np.array([0.0, 1.0]), WIDTH)


def test_parse_level_table_without_section(tmp_path: Path) -> None:
    path = tmp_path / "profile.md"
    path.write_text("# nothing\n\n## 4. other\n\n## 6. other\n", encoding="utf-8")
    with pytest.raises(f1.PaperFigureError):
        f1.parse_level_table(path)


# --------------------------------------------------------------------------------------------
# 3. Annotated numbers
# --------------------------------------------------------------------------------------------


def test_annotations_equal_their_sources(tmp_path: Path) -> None:
    inputs = f1.F1Inputs(tmp_path, tmp_path, REPO, tmp_path)
    ann = f1.load_annotations(inputs)
    assert ann.share_text("ixi", ann.coarsest) == TICKET_STRINGS["ixi_coarse"]
    assert ann.ixi_peak == "8-16"
    assert ann.share_text("ixi", ann.ixi_peak) == TICKET_STRINGS["ixi_peak"]
    assert ann.share_text("church", ann.coarsest) == TICKET_STRINGS["church_coarse"]
    assert ann.inherited_default_text == TICKET_STRINGS["inherited_default"]
    assert ann.inherited_matched_text == TICKET_STRINGS["inherited_matched"]
    assert ann.kid_text == TICKET_STRINGS["kid"]
    assert ann.seed_nn_texts == TICKET_STRINGS["seed_nn"]
    assert ann.kid_n_seeds == 3 and ann.seed_nn_n_seeds == (3, 3)
    np.testing.assert_allclose(sum(ann.ixi_shares), 1.0, rtol=1e-12)
    assert all(ok for *_, ok in f1.verify_annotations(ann, inputs))


# --------------------------------------------------------------------------------------------
# 4. Selection rules, the CLI and byte stability on a tiny synthetic campaign
# --------------------------------------------------------------------------------------------


def _write_tiny_dataset(root: Path) -> np.ndarray:
    rng = np.random.default_rng(7)
    n_subjects, n_slices = 6, 10
    n = n_subjects * n_slices
    images = np.clip(_smooth_stack(n, rng) * 255.0, 0, 255).astype(np.uint8)
    subjects = [f"S{s:02d}" for s in range(n_subjects) for _ in range(n_slices)]
    slices = [k for _ in range(n_subjects) for k in range(n_slices)]
    train = list(range(0, 30))
    ref = list(range(30, 60))
    index = pd.DataFrame({"idx": np.arange(n), "subject": subjects, "slice": slices,
                          "z_mm": [float(k) for k in slices],
                          "source": [f"tiny/{s}/{k}"
                                     for s, k in zip(subjects, slices, strict=True)],
                          "split": ["train"] * 30 + ["ref"] * 30})
    splits = {"train": train, "ref": ref, "seed": list(range(30, 50)),
              "train_subjects": ["S00", "S01", "S02"], "ref_subjects": ["S03", "S04", "S05"],
              "seed_subjects": ["S03", "S04"], "rng_seed": 2026}
    meta = DatasetMeta(dataset_id="ixi", n_images=n, image_size=WIDTH, dtype="uint8",
                       pipeline="tests.analysis.test_paper_f1", pipeline_version="1.0",
                       git_sha="test", created="2026-10-06T00:00:00", raw_root=str(root),
                       parameters={}, counts={})
    write_dataset(root / "ixi", images, index, splits, meta)
    return images


def _add(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mtime = 0
    archive.addfile(info, io.BytesIO(payload))


def _npy(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


def _write_tar(eval_dir: Path, run: str, images: np.ndarray, seed_idx: np.ndarray,
               rng_seed: int = 2026, batch_size: int = 32) -> None:
    rng = np.random.default_rng(zlib.crc32(run.encode()))
    shape = [len(seed_idx), 2, WIDTH, WIDTH]
    samples = rng.integers(0, 256, size=shape, dtype=np.uint8)
    request = {"shape": shape, "set": "heldout",
               "signature": {"rng_seed": rng_seed, "batch_size": batch_size, "set": "heldout"}}
    prefix = f"{run}/samples_amp-fp16/060000/heldout/"
    eval_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(eval_dir / f"{run}_amp-fp16.tar", "w") as archive:
        _add(archive, f"{run}/metrics_amp-fp16/summary.json", b"{}")
        _add(archive, prefix + "samples.npy", _npy(samples))
        _add(archive, prefix + "seed_idx.npy", _npy(seed_idx.astype(np.int64)))
        _add(archive, prefix + "seeds.npy", _npy(np.asarray(images[seed_idx])))
        _add(archive, prefix + "request.json", json.dumps(request).encode())


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("paper_f1")
    images = _write_tiny_dataset(base / "data")
    seed_idx = np.array([45, 35])
    for run in TINY_RULES.runs:
        _write_tar(base / "eval", run, images, seed_idx)
    return {"data": base / "data", "eval": base / "eval", "base": base}


@pytest.fixture
def tiny_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "RULES", TINY_RULES)


def _run_cli(tiny: dict[str, Path], out: Path, eval_dir: Path | None = None,
             work: Path | None = None) -> int:
    argv = ["--data-root", str(tiny["data"]), "--eval-dir", str(eval_dir or tiny["eval"]),
            "--out", str(out)]
    if work is not None:
        argv += ["--work", str(work)]
    return cli.main(argv)


def _digests(folder: Path) -> dict[str, str]:
    return {ext: hashlib.sha256((folder / f"{f1.F1_NAME}.{ext}").read_bytes()).hexdigest()
            for ext in ("pdf", "svg", "png")}


def test_cli_writes_byte_stable_outputs(tiny: dict[str, Path], tiny_cli: None,
                                        tmp_path: Path) -> None:
    out = tmp_path / "out"
    assert _run_cli(tiny, out, work=tmp_path / "work") == cli.EXIT_OK
    first = _digests(out)
    assert _run_cli(tiny, out) == cli.EXIT_OK  # fresh temporary extraction this time
    assert _digests(out) == first
    record = (out / "F1.md").read_text(encoding="utf-8")
    assert record.count("| yes |") >= 3  # the three files are identical to the previous run
    assert "dataset index 35 (expected 35)" in record
    svg = (out / f"{f1.F1_NAME}.svg").read_text(encoding="utf-8")
    assert "<text" in svg and "data:image/png;base64" in svg and "<dc:date>" not in svg
    assert b"CreationDate" not in (out / f"{f1.F1_NAME}.pdf").read_bytes()


def test_load_and_layout_on_tiny_campaign(tiny: dict[str, Path], tmp_path: Path) -> None:
    inputs = f1.F1Inputs(tiny["data"], tiny["eval"], REPO, tmp_path / "work")
    data = f1.load_f1_data(inputs, TINY_RULES)
    assert data.subjects == [("S03", 35), ("S04", 45), ("S05", 55)]
    assert data.teaser_index == 35 and data.n_train == 30
    assert data.example_index in {35, 45, 55}
    np.testing.assert_allclose(data.teaser_seed * 255.0,
                               np.load(tiny["data"] / "ixi" / "images.npy")[35], atol=1e-9)
    assert len(data.schedules["default"]) == 200 and data.schedules["default"].min() > 0
    report = f1.layout_report(f1.draw_f1(data))
    assert report.size_in[0] == f1.PAPER_WIDTH_IN
    assert report.size_in[1] <= f1.PAPER_MAX_HEIGHT_IN
    assert report.min_font_pt >= f1.MIN_FONT_PT
    assert report.min_thumb_in >= f1.MIN_THUMB_IN
    assert report.overlaps == [] and report.outside == []
    caption = f1.caption_text(data)
    words = len(caption.replace("*", " ").split())
    assert 120 <= words <= 180, words
    for name, code in (("default", "A0"), ("+prior", "A1"), ("+spacing", "A2"),
                       ("matched", "A3")):
        assert caption.count(f"{name} ({code})") == 1
        assert f1.ARM_LABELS[code] == name


@pytest.mark.parametrize("damage", ["seed_idx", "rng_seed", "batch_size", "teaser_index"])
def test_cli_rule_failure_returns_1(tiny: dict[str, Path], tiny_cli: None, tmp_path: Path,
                                    damage: str) -> None:
    images = np.load(tiny["data"] / "ixi" / "images.npy")
    eval_dir = tmp_path / "eval"
    good = np.array([45, 35])
    a3 = {"seed_idx": dict(seed_idx=np.array([45, 55])),
          "rng_seed": dict(seed_idx=good, rng_seed=0),
          "batch_size": dict(seed_idx=good, batch_size=16),
          "teaser_index": None}[damage]
    if damage == "teaser_index":
        for run in TINY_RULES.runs:
            _write_tar(eval_dir, run, images, np.array([35, 45]))
    else:
        _write_tar(eval_dir, TINY_RULES.runs[0], images, good)
        _write_tar(eval_dir, TINY_RULES.runs[1], images, **a3)
    out = tmp_path / "out"
    assert _run_cli(tiny, out, eval_dir=eval_dir) == cli.EXIT_FAIL
    assert not (out / "F1.md").exists()


def test_cli_missing_tar_returns_2(tiny: dict[str, Path], tiny_cli: None, tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    images = np.load(tiny["data"] / "ixi" / "images.npy")
    _write_tar(eval_dir, TINY_RULES.runs[0], images, np.array([45, 35]))
    assert _run_cli(tiny, tmp_path / "out", eval_dir=eval_dir) == cli.EXIT_MISSING


def test_cli_missing_member_returns_2(tiny: dict[str, Path], tiny_cli: None,
                                      tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    for run in TINY_RULES.runs:
        with tarfile.open(eval_dir / f"{run}_amp-fp16.tar", "w") as archive:
            _add(archive, f"{run}/metrics_amp-fp16/summary.json", b"{}")
    assert _run_cli(tiny, tmp_path / "out", eval_dir=eval_dir) == cli.EXIT_MISSING


def test_cli_missing_data_root_returns_2(tiny: dict[str, Path], tiny_cli: None,
                                         tmp_path: Path) -> None:
    argv = ["--data-root", str(tmp_path / "nowhere"), "--eval-dir", str(tiny["eval"]),
            "--out", str(tmp_path / "out")]
    assert cli.main(argv) == cli.EXIT_MISSING


def test_cli_usage_error_returns_2() -> None:
    assert cli.main(["--out", "x"]) == cli.EXIT_MISSING


def test_extract_reuses_existing_members(tiny: dict[str, Path], tmp_path: Path) -> None:
    run = TINY_RULES.runs[0]
    folder = f1.extract_heldout(tiny["eval"] / f"{run}_amp-fp16.tar", run, "060000", tmp_path)
    assert sorted(p.name for p in folder.iterdir()) == sorted(f1.HELDOUT_MEMBERS)
    again = f1.extract_heldout(tmp_path / "absent.tar", run, "060000", tmp_path)
    assert again == folder
