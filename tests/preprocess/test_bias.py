"""Tests of the N4 bias-field stage, the site labels and the sensitivity move (T1.4)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from ihdm.cli.preprocess_mri import move_to_sensitivity
from ihdm.paths import template_dir
from ihdm.preprocess.bias import (
    N4Config,
    binary_mask,
    n4_correct,
    read_n4_sidecar,
    simpleitk_n4_defaults,
    write_corrected,
)
from ihdm.preprocess.errors import PreprocessError
from ihdm.preprocess.mri import IXI_SITES, OASIS1_SITE, site_from_source
from ihdm.preprocess.registration import TEMPLATE_STEM, dilate_mask_mm

RAMP_LOW, RAMP_HIGH = 0.8, 1.2
FLATNESS_TOLERANCE = 0.03


def _ellipsoid_mask(shape: tuple[int, int, int], radii: tuple[float, float, float]) -> np.ndarray:
    """Return a centred ellipsoidal boolean mask on an ``(i, j, k)`` grid."""
    grids = np.meshgrid(
        *[np.arange(n, dtype=float) - (n - 1) / 2.0 for n in shape], indexing="ij"
    )
    radius = sum((g / r) ** 2 for g, r in zip(grids, radii, strict=True))
    return radius <= 1.0


def _phantom(
    shape: tuple[int, int, int], ramp_axis: int = 1, seed: int = 2026
) -> tuple[np.ndarray, np.ndarray]:
    """Return a two-tissue head phantom and its brain mask, both on an ``(i, j, k)`` grid.

    A bright "white matter" core sits inside a darker "grey matter" shell, so N4's histogram
    sharpening has two modes to work with. The core is a cylinder along ``ramp_axis``: the
    tissue composition of every plane perpendicular to the ramp is then the same, so any
    profile the test measures along that axis comes from intensity, not from geometry.
    """
    rng = np.random.default_rng(seed)
    outer = _ellipsoid_mask(shape, tuple(0.38 * n for n in shape))
    core_radii = [0.20 * n for n in shape]
    core_radii[ramp_axis] = float(max(shape)) * 1e3
    inner = _ellipsoid_mask(shape, tuple(core_radii)) & outer
    volume = np.zeros(shape, dtype=np.float32)
    volume[outer] = 100.0
    volume[inner] = 160.0
    volume[outer] += rng.normal(0.0, 2.0, size=volume.shape).astype(np.float32)[outer]
    return volume, outer


def _linear_ramp(shape: tuple[int, int, int], axis: int) -> np.ndarray:
    """Return a multiplicative field rising linearly from ``RAMP_LOW`` to ``RAMP_HIGH``."""
    line = np.linspace(RAMP_LOW, RAMP_HIGH, shape[axis], dtype=np.float32)
    return np.moveaxis(np.broadcast_to(line, shape[:axis] + shape[axis + 1:] + (shape[axis],)),
                       -1, axis).copy()


def _as_sitk(array: np.ndarray) -> sitk.Image:
    """Wrap an ``(i, j, k)`` array as a 1 mm isotropic SimpleITK image."""
    image = sitk.GetImageFromArray(np.ascontiguousarray(array.transpose(2, 1, 0)))
    image.SetSpacing((1.0, 1.0, 1.0))
    return image


def _residual_ramp(
    measured: np.ndarray, truth: np.ndarray, mask: np.ndarray, axis: int
) -> float:
    """Relative peak-to-peak spread of the masked mean of ``measured / truth`` along ``axis``.

    This is the quantity the acceptance criterion names: with no bias field the ratio is a
    constant, so the spread is zero; with the phantom's 0.8 -> 1.2 ramp it is the part of
    that ramp the ellipsoid covers, about 30 %.
    """
    inside = mask & (truth > 10.0)
    ratio = np.where(inside, measured / np.maximum(truth, 1e-6), 0.0)
    other = tuple(a for a in range(3) if a != axis)
    weight = inside.sum(axis=other)
    keep = weight > 0.05 * weight.max()
    profile = ratio.sum(axis=other)[keep] / weight[keep]
    return float((profile.max() - profile.min()) / profile.mean())


def test_n4config_matches_simpleitk_defaults() -> None:
    """Every N4 parameter this pipeline sets is SimpleITK's own default."""
    cfg = N4Config()
    defaults = simpleitk_n4_defaults()
    for key, value in defaults.items():
        assert getattr(cfg, key) == value, key


def test_binary_mask_does_not_invert_a_uint8_mask() -> None:
    """The T1.1 ``BinaryThreshold`` hazard: a uint8 mask must not come back inverted."""
    array = np.zeros((12, 12, 12), dtype=np.uint8)
    array[3:9, 3:9, 3:9] = 1
    mask = binary_mask(_as_sitk(array))
    out = sitk.GetArrayFromImage(mask)
    assert int(out.sum()) == int(array.sum()) == 216
    assert out.dtype == np.uint8


@pytest.mark.parametrize("axis", [0, 1])
def test_n4_recovers_a_flat_intensity_on_a_synthetic_ramp(axis: int) -> None:
    """A 0.8 -> 1.2 multiplicative ramp is removed to within 3 % inside the mask."""
    shape = (64, 64, 64)
    clean, brain = _phantom(shape, ramp_axis=axis)
    field = _linear_ramp(shape, axis=axis)
    biased = clean * field

    before = _residual_ramp(biased, clean, brain, axis)
    result = n4_correct(_as_sitk(biased), _as_sitk(brain.astype(np.uint8)), N4Config())
    corrected = sitk.GetArrayFromImage(result.corrected).transpose(2, 1, 0)
    after = _residual_ramp(corrected, clean, brain, axis)

    assert before > 0.25, f"the phantom's ramp should be visible before N4, got {before:.3f}"
    assert after < FLATNESS_TOLERANCE, f"residual ramp {after:.3%} exceeds 3 % (was {before:.3%})"
    assert result.diagnostics["log_field_max"] > result.diagnostics["log_field_min"]
    assert result.diagnostics["seconds"] > 0.0
    assert np.all(corrected >= 0.0)


def test_n4_rejects_a_mismatched_or_empty_mask() -> None:
    """A mask on another grid, or an empty one, is a hard error."""
    volume = _as_sitk(np.ones((16, 16, 16), dtype=np.float32))
    with pytest.raises(PreprocessError, match="mask size"):
        n4_correct(volume, _as_sitk(np.ones((8, 8, 8), dtype=np.uint8)))
    with pytest.raises(PreprocessError, match="empty"):
        n4_correct(volume, _as_sitk(np.zeros((16, 16, 16), dtype=np.uint8)))


def test_write_and_read_the_n4_sidecar(tmp_path: Path) -> None:
    """The cache sidecar carries the parameters, the log-field range and the extra fields."""
    shape = (32, 32, 32)
    clean, brain = _phantom(shape)
    result = n4_correct(
        _as_sitk(clean * _linear_ramp(shape, axis=0)), _as_sitk(brain.astype(np.uint8))
    )
    destination = tmp_path / "IXI012.nii.gz"
    write_corrected(
        destination, result, N4Config(), "IXI012", "12_HH/IXI012-HH-1211-T1.nii.gz",
        extra={"site": "HH"},
    )
    sidecar = read_n4_sidecar(destination)
    assert destination.is_file()
    assert sidecar["subject"] == "IXI012"
    assert sidecar["site"] == "HH"
    assert sidecar["n4"]["max_iterations"] == [50, 50, 50, 50]
    assert sidecar["n4"]["shrink_factor"] == 4
    assert sidecar["log_field_min"] <= sidecar["log_field_mean"] <= sidecar["log_field_max"]
    # The sidecar must be plain JSON, not a repr of numpy scalars.
    json.loads(destination.with_suffix("").with_suffix(".json").read_text())


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("12_HH/IXI012-HH-1211-T1.nii.gz", "HH"),
        ("100_Guys/IXI100-Guys-0747-T1.nii.gz", "Guys"),
        ("581_IOP/IXI581-IOP-1145-T1.nii.gz", "IOP"),
        ("IXI002-Guys-0828-T1.nii.gz", "Guys"),
    ],
)
def test_site_from_ixi_filename(source: str, expected: str) -> None:
    """The site is the second dash-separated token of the IXI file name."""
    assert site_from_source("ixi", source) == expected
    assert expected in IXI_SITES


def test_site_of_oasis1_is_the_single_scanner() -> None:
    """Every OASIS-1 subject carries the same site label."""
    assert site_from_source("oasis1", "disc1/OAS1_0001_MR1/RAW/OAS1_0001_MR1_mpr-1_anon.hdr") == (
        OASIS1_SITE
    )


@pytest.mark.parametrize(
    ("cohort", "source"),
    [("ixi", "12_XX/IXI012-XX-1211-T1.nii.gz"), ("ixi", "nonsense.nii.gz"), ("lsun", "x.png")],
)
def test_site_parsing_rejects_the_unknown(cohort: str, source: str) -> None:
    """An unknown cohort, an unknown site or an unparsable name all fail loudly."""
    with pytest.raises(PreprocessError):
        site_from_source(cohort, source)


def test_move_to_sensitivity_is_idempotent(tmp_path: Path) -> None:
    """The first call moves the dataset aside; later calls leave both directories alone."""
    dataset = tmp_path / "ixi"
    sensitivity = tmp_path / "_sensitivity" / "ixi_no_n4"
    dataset.mkdir()
    (dataset / "meta.json").write_text('{"pipeline_version": "1.0"}')

    assert move_to_sensitivity(dataset, sensitivity) == sensitivity
    assert not dataset.exists()
    assert json.loads((sensitivity / "meta.json").read_text())["pipeline_version"] == "1.0"

    # A rebuild writes a new dataset; the archived copy must survive a second call untouched.
    dataset.mkdir()
    (dataset / "meta.json").write_text('{"pipeline_version": "1.1"}')
    assert move_to_sensitivity(dataset, sensitivity) is None
    assert json.loads((sensitivity / "meta.json").read_text())["pipeline_version"] == "1.0"
    assert json.loads((dataset / "meta.json").read_text())["pipeline_version"] == "1.1"


def test_move_to_sensitivity_without_a_dataset(tmp_path: Path) -> None:
    """Nothing to move is not an error."""
    assert move_to_sensitivity(tmp_path / "missing", tmp_path / "_sensitivity" / "x_no_n4") is None


def test_move_to_sensitivity_rejects_a_file_destination(tmp_path: Path) -> None:
    """A destination that is not a directory is a hard error."""
    dataset = tmp_path / "ixi"
    dataset.mkdir()
    destination = tmp_path / "taken"
    destination.write_text("not a directory")
    with pytest.raises(PreprocessError, match="not a directory"):
        move_to_sensitivity(dataset, destination)


@pytest.mark.integration
def test_n4_on_the_real_template_mask() -> None:
    """The same ramp recovery with the mask and the grid the pipeline actually uses.

    The phantom stays synthetic (two tissues: the brain mask bright, the 10 mm dilation ring
    darker) but it lives on the 193 x 229 x 193 MNI152 lattice and is corrected with the same
    dilated mask the production stage passes, so the 10 mm dilation, the grid match and the
    shrink / full-resolution-field round trip are exercised on the real geometry.
    """
    mask_path = template_dir() / f"{TEMPLATE_STEM}_desc-brain_mask.nii.gz"
    image_path = template_dir() / f"{TEMPLATE_STEM}_T1w.nii.gz"
    if not mask_path.is_file() or not image_path.is_file():
        pytest.skip(f"MNI152 template not available under {template_dir()}")

    template = sitk.ReadImage(str(image_path), sitk.sitkFloat32)
    brain_mask = sitk.ReadImage(str(mask_path))
    dilated = dilate_mask_mm(brain_mask, 10.0)
    brain = sitk.GetArrayFromImage(sitk.Cast(brain_mask, sitk.sitkUInt8)).transpose(2, 1, 0) > 0
    head = sitk.GetArrayFromImage(dilated).transpose(2, 1, 0) > 0

    rng = np.random.default_rng(2026)
    clean = np.zeros(brain.shape, dtype=np.float32)
    clean[head] = 100.0
    clean[brain] = 160.0
    clean[head] += rng.normal(0.0, 2.0, size=clean.shape).astype(np.float32)[head]

    field = _linear_ramp(clean.shape, axis=1)
    biased = _as_sitk(clean * field)
    biased.CopyInformation(template)

    before = _residual_ramp(clean * field, clean, head, axis=1)
    result = n4_correct(biased, dilated)
    corrected = sitk.GetArrayFromImage(result.corrected).transpose(2, 1, 0)
    after = _residual_ramp(corrected, clean, head, axis=1)

    assert before > 0.25
    assert after < FLATNESS_TOLERANCE, (
        f"residual ramp {after:.3%} exceeds 3 % on the real template grid (was {before:.3%})"
    )
