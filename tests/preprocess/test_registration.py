"""Unit tests of the rigid registration: accuracy on a phantom, mask dilation, resampling."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import SimpleITK as sitk

from ihdm.preprocess.errors import PreprocessError
from ihdm.preprocess.raw_mri import ras_array_to_sitk
from ihdm.preprocess.registration import (
    RegistrationConfig,
    dilate_mask_mm,
    homogeneous_matrix,
    matrix_discrepancy,
    register,
    resample_to_template,
)

PHANTOM_SHAPE = (64, 72, 64)
PHANTOM_SPACING = (2.0, 2.0, 2.0)


def make_head_phantom(
    shape: tuple[int, int, int] = PHANTOM_SHAPE,
    spacing: tuple[float, float, float] = PHANTOM_SPACING,
) -> sitk.Image:
    """Build a synthetic head: an ellipsoid with a brighter, off-centre inner ellipsoid.

    Three distinct semi-axes fix the orientation up to the 180-degree symmetries of an
    ellipsoid, and the off-centre inner body breaks those, so all six rigid degrees of
    freedom are identifiable.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Grid shape in ``(i, j, k)`` order.
    spacing : tuple[float, float, float]
        Voxel size in millimetres.

    Returns
    -------
    sitk.Image
        A smoothed ``float32`` phantom in LPS physical space.
    """
    indices = np.indices(shape).astype(float)
    centre = (np.array(shape, dtype=float) - 1.0) / 2.0
    x, y, z = ((indices[a] - centre[a]) * spacing[a] for a in range(3))

    outer = (x / 45.0) ** 2 + (y / 55.0) ** 2 + (z / 40.0) ** 2 <= 1.0
    inner = ((x - 8.0) / 22.0) ** 2 + ((y + 10.0) / 26.0) ** 2 + ((z - 6.0) / 18.0) ** 2 <= 1.0
    volume = 60.0 * outer.astype(np.float32) + 90.0 * inner.astype(np.float32)

    image = ras_array_to_sitk(volume, spacing)
    return sitk.SmoothingRecursiveGaussian(image, 2.0)


def make_ground_truth(image: sitk.Image) -> sitk.Euler3DTransform:
    """Return a rigid transform of 8 degrees and 6 mm about the image centre."""
    size = np.array(image.GetSize(), dtype=float)
    centre = image.TransformContinuousIndexToPhysicalPoint(((size - 1.0) / 2.0).tolist())
    transform = sitk.Euler3DTransform()
    transform.SetCenter(centre)
    transform.SetRotation(math.radians(3.0), math.radians(-5.0), math.radians(8.0))
    transform.SetTranslation((6.0, -4.0, 3.0))
    return transform


def test_phantom_rigid_recovery_within_half_mm_and_half_degree() -> None:
    """A phantom rotated 8 deg and translated 6 mm registers back within 0.5 mm / 0.5 deg."""
    phantom = make_head_phantom()
    ground_truth = make_ground_truth(phantom)
    moved = sitk.Resample(phantom, phantom, ground_truth, sitk.sitkLinear, 0.0, sitk.sitkFloat32)

    result = register(moved, phantom, None, RegistrationConfig())

    # `moved(p) = phantom(T_gt(p))`, so the transform that maps fixed to moving points is
    # the inverse of the ground truth; composing the two must give the identity.
    composite = np.linalg.inv(homogeneous_matrix(ground_truth))
    angle, shift = matrix_discrepancy(composite, homogeneous_matrix(result.transform))
    assert angle < 0.5, f"rotation error {angle:.3f} deg"
    assert shift < 0.5, f"translation error {shift:.3f} mm"
    assert result.final_metric < 0.0
    assert result.iterations > 0


def test_registration_is_deterministic() -> None:
    """The fixed metric-sampling seed makes two runs agree to floating-point noise.

    The samples drawn are identical; the residual 1e-12 spread is the non-deterministic
    reduction order of the multithreaded metric, well below any physical tolerance.
    """
    phantom = make_head_phantom()
    moved = sitk.Resample(
        phantom, phantom, make_ground_truth(phantom), sitk.sitkLinear, 0.0, sitk.sitkFloat32
    )
    cfg = dataclasses.replace(RegistrationConfig(), max_iterations=40)
    first = register(moved, phantom, None, cfg)
    second = register(moved, phantom, None, cfg)
    np.testing.assert_allclose(first.parameters(), second.parameters(), rtol=1e-8, atol=1e-8)
    angle, shift = matrix_discrepancy(
        homogeneous_matrix(first.transform), homogeneous_matrix(second.transform)
    )
    assert angle < 1e-4 and shift < 1e-4


def test_dilate_mask_mm_is_an_exact_physical_ball() -> None:
    """Dilating a single voxel by r mm marks exactly the voxels within r mm of it."""
    shape = (31, 31, 31)
    spacing = (1.0, 2.0, 1.0)
    volume = np.zeros(shape, dtype=np.float32)
    volume[15, 15, 15] = 1.0
    mask = ras_array_to_sitk(volume, spacing)

    dilated = dilate_mask_mm(sitk.Cast(mask, sitk.sitkUInt8), 5.0)
    array = sitk.GetArrayFromImage(dilated).transpose(2, 1, 0).astype(bool)

    indices = np.indices(shape).astype(float)
    distance = np.sqrt(sum(((indices[a] - 15.0) * spacing[a]) ** 2 for a in range(3)))
    np.testing.assert_array_equal(array, distance <= 5.0)


def test_dilate_mask_mm_does_not_invert_a_uint8_mask() -> None:
    """A zero-millimetre dilation returns the mask itself, not its complement."""
    volume = np.zeros((16, 16, 16), dtype=np.float32)
    volume[4:8, 4:8, 4:8] = 1.0
    mask = sitk.Cast(ras_array_to_sitk(volume, (1.0, 1.0, 1.0)), sitk.sitkUInt8)
    kept = sitk.GetArrayFromImage(dilate_mask_mm(mask, 0.0)).sum()
    assert kept == 64


def test_resample_to_template_uses_the_template_grid_and_clips_negatives() -> None:
    """The single resampling lands on the template grid with no negative overshoot."""
    phantom = make_head_phantom()
    template = ras_array_to_sitk(
        np.zeros((40, 44, 40), dtype=np.float32), (3.0, 3.0, 3.0)
    )
    identity = sitk.Euler3DTransform()

    resampled = resample_to_template(phantom, template, identity, RegistrationConfig())

    assert resampled.GetSize() == template.GetSize()
    assert resampled.GetSpacing() == template.GetSpacing()
    assert resampled.GetOrigin() == template.GetOrigin()
    assert float(sitk.GetArrayFromImage(resampled).min()) >= 0.0


def test_resample_keeps_negatives_when_clipping_is_off() -> None:
    """The clip is a deliberate choice, so turning it off really does keep the overshoot."""
    phantom = make_head_phantom()
    cfg = dataclasses.replace(RegistrationConfig(), clip_negative=False)
    shifted = sitk.Euler3DTransform()
    shifted.SetTranslation((1.3, 0.7, 0.4))
    resampled = resample_to_template(phantom, phantom, shifted, cfg)
    assert float(sitk.GetArrayFromImage(resampled).min()) < 0.0


def test_registration_config_round_trips_to_json() -> None:
    """Every configuration value reaches ``meta.json`` as a JSON-native type."""
    payload = RegistrationConfig().to_json()
    assert payload["shrink_factors"] == [4, 2, 1]
    assert payload["smoothing_sigmas_mm"] == [2.0, 1.0, 0.0]
    assert payload["histogram_bins"] == 50
    assert payload["sampling_percentage"] == pytest.approx(0.20)
    assert payload["interpolator_resample"] == "bspline3"


def test_register_rejects_a_degenerate_moving_image() -> None:
    """A constant image has no mutual information to maximise."""
    flat = ras_array_to_sitk(np.zeros((24, 24, 24), dtype=np.float32), (2.0, 2.0, 2.0))
    with pytest.raises(PreprocessError):
        register(flat, flat, None, dataclasses.replace(RegistrationConfig(), max_iterations=5))
