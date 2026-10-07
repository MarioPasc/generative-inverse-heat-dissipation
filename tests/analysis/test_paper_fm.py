"""Tests of FM, the method figure (T8.4): DCT basis, octaves, macro-steps, levels, CLI, bytes."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.fft import idctn

from ihdm.analysis import paper_f1 as f1
from ihdm.analysis import paper_fm as fm
from ihdm.cli import paper_fm as cli
from ihdm.data.format import DatasetMeta, write_dataset
from ihdm.spectral import power

REPO = Path(__file__).resolve().parents[2]
WIDTH = 192
DATA_ROOT = Path("/media/mpascual/MeningD2/spectral_allocation_heat_diffusion_project")

#: The ticket's numbers, typed here (and only here) to check the computed ones against them.
TICKET_LEVELS = {"default": (31, 26, 26, 26, 27, 26, 26, 12),
                 "matched": (0, 5, 30, 38, 47, 42, 29, 9)}
TICKET_SHARES = ("4.1", "11.3", "11.0", "16.8", "29.8", "18.3", "7.8", "1.0")
TICKET_EXAMPLE_INDEX = 3705


# --------------------------------------------------------------------------------------------
# 1. The DCT decomposition
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 7, 16])
def test_1d_basis_is_orthonormal(n: int) -> None:
    basis = fm.dct_basis_1d(n)
    np.testing.assert_allclose(basis @ basis.T, np.eye(n), rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("width", [4, 8])
def test_2d_basis_images_are_orthonormal(width: int) -> None:
    phis = np.stack([fm.dct_mode(width, i, j).ravel()
                     for i in range(width) for j in range(width)])
    np.testing.assert_allclose(phis @ phis.T, np.eye(width * width), rtol=0.0, atol=1e-12)


@pytest.mark.parametrize(("width", "i", "j"), [(12, 0, 0), (12, 3, 7), (WIDTH, 0, 1),
                                               (WIDTH, 16, 16), (WIDTH, 191, 5)])
def test_basis_image_equals_scipy_inverse_dct(width: int, i: int, j: int) -> None:
    impulse = np.zeros((width, width))
    impulse[i, j] = 1.0
    np.testing.assert_allclose(fm.dct_mode(width, i, j), idctn(impulse, norm="ortho"),
                               rtol=0.0, atol=1e-12)


def test_explicit_sum_reconstructs_a_small_image() -> None:
    width = 8
    image = np.random.default_rng(3).random((width, width))
    weights = fm.dct_weights(image)
    total = sum(weights[i, j] * fm.dct_mode(width, i, j)
                for i in range(width) for j in range(width))
    np.testing.assert_allclose(total, image, rtol=0.0, atol=1e-10)


def test_synthesis_reconstructs_a_192_image() -> None:
    image = np.random.default_rng(4).random((WIDTH, WIDTH))
    np.testing.assert_allclose(fm.synthesise(fm.dct_weights(image)), image, rtol=0.0,
                               atol=1e-10)


@pytest.mark.parametrize("call", [lambda: fm.dct_basis_1d(0),
                                  lambda: fm.dct_mode(8, 8, 0),
                                  lambda: fm.dct_mode(8, 0, -1),
                                  lambda: fm.synthesise(np.zeros((3, 4))),
                                  lambda: fm.synthesise(np.zeros(5))])
def test_decomposition_rejects_bad_arguments(call) -> None:
    with pytest.raises(fm.MethodFigureError):
        call()


# --------------------------------------------------------------------------------------------
# 2. The octave masks
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("width", [WIDTH, 64, 2])
def test_octave_masks_partition_the_non_dc_modes_up_to_96(width: int) -> None:
    masks = np.stack(fm.octave_masks(width))
    cycles = fm.mode_cycles(width)
    assert masks.sum(axis=0).max() <= 1  # disjoint
    np.testing.assert_array_equal(masks.any(axis=0), (cycles > 0) & (cycles <= 96.0))
    assert not masks[:, 0, 0].any()  # the DC mode is in no octave
    for ours, theirs in zip(fm.octave_masks(width), power._octave_masks(width), strict=True):
        np.testing.assert_array_equal(ours, theirs)


def test_octave_mode_counts() -> None:
    counts = fm.octave_mode_counts(WIDTH)
    assert len(counts) == 8
    assert counts[0] == 3  # (0, 1), (1, 0), (1, 1): data_profile.md §2
    cycles = fm.mode_cycles(WIDTH)
    assert sum(counts) == int(((cycles > 0) & (cycles <= 96.0)).sum())
    assert WIDTH**2 - 1 - sum(counts) == int((cycles > 96.0).sum())


@pytest.mark.parametrize("label_lo", [lo for _, lo, _ in power.octave_bins()])
def test_representative_modes_lie_in_their_octave(label_lo: float) -> None:
    k = [lo for _, lo, _ in power.octave_bins()].index(label_lo)
    mask = fm.octave_masks(WIDTH)[k]
    edge, diagonal = fm.representative_modes(label_lo)
    assert mask[edge] and mask[diagonal]
    cycles = fm.mode_cycles(WIDTH)
    np.testing.assert_allclose(cycles[edge], label_lo, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(cycles[diagonal], np.sqrt(2.0) * label_lo, rtol=1e-12)


@pytest.mark.parametrize("low", [0.25, 0.75, 0.0])
def test_representative_modes_reject_off_grid_edges(low: float) -> None:
    with pytest.raises(fm.MethodFigureError):
        fm.representative_modes(low)


# --------------------------------------------------------------------------------------------
# 3. The macro-steps
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def macro_checks() -> list[fm.MacroStepCheck]:
    return fm.verify_macro_steps(WIDTH)


def test_macro_steps_pass(macro_checks: list[fm.MacroStepCheck]) -> None:
    assert [c.label for c in macro_checks] == list(power.OCTAVE_LABELS)
    assert all(c.passed for c in macro_checks)


def test_macro_step_edge_keeps_e_minus_1(macro_checks: list[fm.MacroStepCheck]) -> None:
    for c in macro_checks:
        assert c.n_edge >= 2  # (0, 2c_b) and (2c_b, 0) at least
        np.testing.assert_allclose(c.d_edge, np.exp(-1.0), rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(c.d_edge_released, np.exp(-1.0), rtol=0.0, atol=1e-9)
        assert c.ring_min <= np.exp(-1.0) <= c.ring_max  # within the radial binning
        # sigma_b * c_b = W / (sqrt(2) pi), the 43.2 of the ticket
        np.testing.assert_allclose(c.sigma * c.low, WIDTH / (np.sqrt(2.0) * np.pi), rtol=1e-12)


def test_macro_step_finer_modes_keep_at_most_e_minus_4(
        macro_checks: list[fm.MacroStepCheck]) -> None:
    for c in macro_checks:
        assert c.n_beyond > 0
        assert c.beyond_max <= np.exp(-4.0) + 1e-12
    # The bound is attained wherever a mode lies exactly at 2 c_b (all but the last octave).
    for c in macro_checks[:-1]:
        np.testing.assert_allclose(c.beyond_max, np.exp(-4.0), rtol=1e-12)


def test_macro_step_matches_the_dctblur_formula() -> None:
    # d = exp(-lambda t), lambda = (pi n / W)^2, t = sigma^2 / 2, written out independently.
    for _, low, _ in power.octave_bins():
        sigma = fm.macro_step_sigma(low, WIDTH)
        for n, expected in ((2.0 * low, np.exp(-1.0)), (4.0 * low, np.exp(-4.0))):
            d = np.exp(-((np.pi * n / WIDTH) ** 2) * sigma**2 / 2.0)
            np.testing.assert_allclose(d, expected, rtol=1e-12)


# --------------------------------------------------------------------------------------------
# 4. The level counts
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["default", "matched"])
def test_level_counts_equal_paper_f1(key: str) -> None:
    schedule = np.load(REPO / "schedules" / f"{f1.SCHEDULES[key]}.npy")
    counts, folded = fm.frequency_level_counts(schedule, WIDTH)
    assert (counts, folded) == f1.levels_per_frequency_octave(schedule[1:], WIDTH)
    assert counts == TICKET_LEVELS[key]
    assert sum(counts) == schedule.size - 1 == 200


@pytest.mark.parametrize("bad", [np.array([0.5, 1.0, 2.0]), np.array([0.0]),
                                 np.zeros((2, 3))])
def test_level_counts_reject_a_bad_schedule(bad: np.ndarray) -> None:
    with pytest.raises(fm.MethodFigureError):
        fm.frequency_level_counts(bad, WIDTH)


# --------------------------------------------------------------------------------------------
# 5. Drawing, the CLI and byte stability on a tiny synthetic dataset
# --------------------------------------------------------------------------------------------


def _write_tiny_dataset(root: Path) -> None:
    rng = np.random.default_rng(7)
    n_subjects, n_slices = 4, 10
    n = n_subjects * n_slices
    coarse = f1.heat_blur(rng.normal(size=(n, WIDTH, WIDTH)), 6.0) * 20.0
    images = np.clip((0.5 + 0.2 * (coarse + 0.1 * rng.normal(size=coarse.shape))) * 255.0,
                     0, 255).astype(np.uint8)
    subjects = [f"S{s:02d}" for s in range(n_subjects) for _ in range(n_slices)]
    slices = [k for _ in range(n_subjects) for k in range(n_slices)]
    index = pd.DataFrame({"idx": np.arange(n), "subject": subjects, "slice": slices,
                          "z_mm": [float(k) for k in slices],
                          "source": [f"tiny/{s}/{k}"
                                     for s, k in zip(subjects, slices, strict=True)],
                          "split": ["train"] * 20 + ["ref"] * 20})
    splits = {"train": list(range(20)), "ref": list(range(20, 40)), "seed": [],
              "train_subjects": ["S00", "S01"], "ref_subjects": ["S02", "S03"],
              "seed_subjects": [], "rng_seed": 2026}
    meta = DatasetMeta(dataset_id="ixi", n_images=n, image_size=WIDTH, dtype="uint8",
                       pipeline="tests.analysis.test_paper_fm", pipeline_version="1.0",
                       git_sha="test", created="2026-10-07T00:00:00", raw_root=str(root),
                       parameters={}, counts={})
    write_dataset(root / "ixi", images, index, splits, meta)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("paper_fm") / "data"
    _write_tiny_dataset(root)
    return root


def _repo_copy(dest: Path) -> Path:
    """The four repository files FM reads, copied so that a test may damage them."""
    inputs = fm.FMInputs(DATA_ROOT, REPO)
    for src in [inputs.profile, inputs.data_profile_md]:
        target = dest / src.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, target)
    for name in f1.SCHEDULES.values():
        target = dest / "schedules" / f"{name}.npy"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(inputs.schedule(name), target)
    return dest


def _digests(folder: Path) -> dict[str, str]:
    return {ext: hashlib.sha256((folder / f"{fm.FM_NAME}.{ext}").read_bytes()).hexdigest()
            for ext in ("pdf", "svg", "png")}


def test_layout_and_caption_on_tiny_dataset(tiny: Path) -> None:
    data = fm.load_fm_data(fm.FMInputs(tiny, REPO))
    assert data.example_index in set(range(20, 40)) and data.example_index % 10 == 5
    assert data.level_counts == TICKET_LEVELS
    assert tuple(f"{100.0 * s:.1f}" for s in data.shares) == TICKET_SHARES
    report = fm.fm_layout_report(fm.draw_fm(data))
    assert report.base.size_in[0] == f1.PAPER_WIDTH_IN
    assert report.base.size_in[1] <= fm.FM_MAX_HEIGHT_IN
    assert report.base.min_font_pt >= f1.MIN_FONT_PT
    assert report.min_mode_in >= fm.MIN_MODE_THUMB_IN
    assert report.min_state_in >= fm.MIN_STATE_THUMB_IN
    assert report.base.overlaps == [] and report.base.outside == []
    caption = fm.caption_text(data)
    assert 150 <= len(caption.replace("*", " ").split()) <= 200
    for needle in ("macro-step", "ideal path", "/".join(map(str, TICKET_LEVELS["default"])),
                   "/".join(map(str, TICKET_LEVELS["matched"])), "q(u_k | u_0)",
                   "p_θ(u_{k−1} | u_k)"):
        assert needle in caption, needle


def test_cli_writes_byte_stable_outputs(tiny: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    assert cli.main(["--data-root", str(tiny), "--out", str(out)]) == cli.EXIT_OK
    first = _digests(out)
    assert cli.main(["--data-root", str(tiny), "--out", str(out),
                     "--eval-dir", str(tmp_path / "unused"),
                     "--work", str(tmp_path / "work")]) == cli.EXIT_OK
    assert _digests(out) == first
    record = (out / "FM.md").read_text(encoding="utf-8")
    assert record.count("| yes |") >= 3 + 8  # three identical files, eight macro-steps passed
    assert fm.check_markdown_numbers(record, fm.load_fm_data(fm.FMInputs(tiny, REPO))) == []
    svg = (out / f"{fm.FM_NAME}.svg").read_text(encoding="utf-8")
    assert "<text" in svg and "data:image/png;base64" in svg and "<dc:date>" not in svg
    assert 'id="top-box"' in svg and 'id="index-plane"' in svg and 'id="side-panel"' in svg
    assert b"CreationDate" not in (out / f"{fm.FM_NAME}.pdf").read_bytes()


def test_cli_missing_data_root_returns_2(tmp_path: Path) -> None:
    argv = ["--data-root", str(tmp_path / "nowhere"), "--out", str(tmp_path / "out")]
    assert cli.main(argv) == cli.EXIT_MISSING
    assert not (tmp_path / "out" / "FM.md").exists()


def test_cli_missing_repo_file_returns_2(tiny: Path, tmp_path: Path,
                                         monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _repo_copy(tmp_path / "repo")
    (repo / "schedules" / f"{f1.SCHEDULES['matched']}.npy").unlink()
    monkeypatch.setattr(cli, "REPO_ROOT", repo)
    assert cli.main(["--data-root", str(tiny), "--out", str(tmp_path / "out")]) == \
        cli.EXIT_MISSING


def test_cli_usage_error_returns_2() -> None:
    assert cli.main(["--out", "x"]) == cli.EXIT_MISSING


@pytest.mark.parametrize("damage", ["schedule", "range", "labels", "shares"])
def test_cli_analysis_error_returns_1(tiny: Path, tmp_path: Path,
                                      monkeypatch: pytest.MonkeyPatch, damage: str) -> None:
    repo = _repo_copy(tmp_path / "repo")
    inputs = fm.FMInputs(tiny, repo)
    if damage in ("schedule", "range"):
        path = inputs.schedule(f1.SCHEDULES["default"])
        schedule = np.load(path)
        if damage == "schedule":  # in range, but its counts differ from data_profile.md §5
            np.save(path, np.concatenate([[0.0], np.linspace(0.5, 96.0, schedule.size - 1)]))
        else:  # levels beyond 96 px
            np.save(path, schedule * 1.1)
    else:
        with np.load(inputs.profile) as profile:
            arrays = {k: profile[k] for k in profile.files}
        if damage == "labels":
            arrays["octave_labels"] = arrays["octave_labels"][::-1]
        else:
            arrays["octave_shares_train"] = arrays["octave_shares_train"][::-1]
        np.savez(inputs.profile, **arrays)
    monkeypatch.setattr(cli, "REPO_ROOT", repo)
    out = tmp_path / "out"
    assert cli.main(["--data-root", str(tiny), "--out", str(out)]) == cli.EXIT_FAIL
    assert not (out / "FM.md").exists()


def test_check_markdown_numbers_reports_what_is_missing(tiny: Path) -> None:
    data = fm.load_fm_data(fm.FMInputs(tiny, REPO))
    missing = fm.check_markdown_numbers("nothing here", data)
    assert "/".join(map(str, TICKET_LEVELS["default"])) in missing


@pytest.mark.integration
@pytest.mark.skipif(not DATA_ROOT.is_dir(), reason="IHDM data root not mounted")
def test_real_slice_is_f1s_and_reconstructs() -> None:
    data = fm.load_fm_data(fm.FMInputs(DATA_ROOT, REPO))
    assert data.example_index == TICKET_EXAMPLE_INDEX
    np.testing.assert_allclose(fm.synthesise(fm.dct_weights(data.example)), data.example,
                               rtol=0.0, atol=1e-10)
