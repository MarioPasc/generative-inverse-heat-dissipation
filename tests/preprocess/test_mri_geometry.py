"""Tests of the template geometry, the 192-px window and the stored slice orientation."""

from __future__ import annotations

import numpy as np
import pytest

from ihdm.paths import template_dir
from ihdm.preprocess.errors import PreprocessError
from ihdm.preprocess.mri import (
    IMAGE_SIZE,
    SLICE_Z_INDICES,
    extract_slices,
    geometry_from_affine,
    template_geometry,
    to_display_orientation,
)
from ihdm.preprocess.registration import TEMPLATE_STEM

TEMPLATE_PATH = template_dir() / f"{TEMPLATE_STEM}_T1w.nii.gz"
requires_template = pytest.mark.skipif(
    not TEMPLATE_PATH.is_file(), reason=f"MNI152 template not present at {TEMPLATE_PATH}"
)


def mni_affine() -> np.ndarray:
    """Return the MNI152NLin2009cAsym 1 mm affine (RAS, unit diagonal)."""
    affine = np.eye(4)
    affine[:3, 3] = (-96.0, -132.0, -78.0)
    return affine


def test_geometry_derives_the_window_from_the_affine() -> None:
    """The window is the 192 block centred on the voxel of the MNI in-plane origin."""
    geometry = geometry_from_affine(mni_affine(), (193, 229, 193))

    assert geometry.origin_voxel == (96.0, 132.0)
    assert geometry.x_slice == (0, 192)
    assert geometry.y_slice == (36, 228)
    assert geometry.size == IMAGE_SIZE
    # The window really is centred: the MNI origin voxel sits at offset size // 2.
    assert geometry.x_slice[0] + geometry.size // 2 == int(geometry.origin_voxel[0])
    assert geometry.y_slice[0] + geometry.size // 2 == int(geometry.origin_voxel[1])


def test_geometry_z_coordinates_come_from_the_affine() -> None:
    """``z_mm`` is the affine applied to the fixed voxel z indices, not a hard-coded list."""
    geometry = geometry_from_affine(mni_affine(), (193, 229, 193))
    expected = tuple(float(k - 78.0) for k in SLICE_Z_INDICES)
    assert geometry.z_indices == SLICE_Z_INDICES
    np.testing.assert_allclose(geometry.z_mm, expected)
    assert geometry.z_mm[5] == pytest.approx(18.0)


def test_geometry_scales_with_a_different_voxel_size() -> None:
    """A 2 mm grid puts the origin at half the index and doubles the millimetre step."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = (-96.0, -132.0, -78.0)
    geometry = geometry_from_affine(affine, (97, 115, 97), z_indices=(10, 20), size=32)
    assert geometry.origin_voxel == (48.0, 66.0)
    assert geometry.x_slice == (32, 64)
    np.testing.assert_allclose(geometry.z_mm, (-58.0, -38.0))


@pytest.mark.parametrize(
    ("affine", "shape", "z_indices", "size"),
    [
        (np.array([[0, 1, 0, -96], [1, 0, 0, -132], [0, 0, 1, -78], [0, 0, 0, 1]], float),
         (193, 229, 193), SLICE_Z_INDICES, 192),          # not axis-aligned
        (np.diag([-1.0, 1.0, 1.0, 1.0]), (193, 229, 193), SLICE_Z_INDICES, 192),  # not RAS-positive
        (mni_affine(), (193, 229, 193), (55, 400), 192),   # plane outside the grid
        (mni_affine(), (193, 229, 193), SLICE_Z_INDICES, 260),  # window leaves the grid
    ],
)
def test_geometry_rejects_grids_it_cannot_honour(
    affine: np.ndarray, shape: tuple[int, int, int], z_indices: tuple[int, ...], size: int
) -> None:
    """Every geometric precondition raises instead of silently producing wrong windows."""
    with pytest.raises(PreprocessError):
        geometry_from_affine(affine, shape, z_indices=z_indices, size=size)


def test_geometry_rejects_a_non_integral_origin_voxel() -> None:
    """A grid whose MNI origin falls between voxels is refused rather than rounded."""
    affine = np.eye(4)
    affine[:3, 3] = (-96.5, -132.0, -78.0)
    with pytest.raises(PreprocessError):
        geometry_from_affine(affine, (193, 229, 193))


def test_display_orientation_puts_anterior_at_the_top_and_left_on_the_left() -> None:
    """A marker at anterior-left lands in the top-left corner of the stored array."""
    # Axes of the input plane: axis 0 increases towards R, axis 1 towards A.
    plane = np.zeros((8, 8), dtype=np.float32)
    plane[0, 7] = 1.0  # smallest i (subject-left), largest j (most anterior)
    stored = to_display_orientation(plane)
    assert stored.shape == (8, 8)
    assert stored[0, 0] == 1.0
    assert stored.sum() == 1.0


def test_display_orientation_is_not_a_mirror() -> None:
    """A marker at anterior-right lands top-right, so left and right are not swapped."""
    plane = np.zeros((8, 8), dtype=np.float32)
    plane[7, 7] = 1.0  # largest i (subject-right), largest j (most anterior)
    assert to_display_orientation(plane)[0, 7] == 1.0


def test_display_orientation_rejects_a_non_plane() -> None:
    """Only two-dimensional input is accepted."""
    with pytest.raises(PreprocessError):
        to_display_orientation(np.zeros((4, 4, 4)))


def test_extract_slices_shape_and_row_order() -> None:
    """The stack has one row per plane and row 0 carries the largest y coordinate."""
    geometry = geometry_from_affine(mni_affine(), (193, 229, 193))
    # Encode the world y coordinate in the voxel value, so the row order is checkable.
    volume = np.broadcast_to(
        (np.arange(229, dtype=np.float32) - 132.0)[None, :, None], (193, 229, 193)
    ).copy()

    stack = extract_slices(volume, geometry)

    assert stack.shape == (len(SLICE_Z_INDICES), IMAGE_SIZE, IMAGE_SIZE)
    first_row_y, last_row_y = stack[0][0, 0], stack[0][-1, 0]
    assert first_row_y > last_row_y, "row 0 must be the most anterior row"
    assert first_row_y == pytest.approx(227 - 132.0)
    assert last_row_y == pytest.approx(36 - 132.0)


def test_extract_slices_column_order() -> None:
    """Column 0 carries the smallest x coordinate, i.e. the subject's left."""
    geometry = geometry_from_affine(mni_affine(), (193, 229, 193))
    volume = np.broadcast_to(
        (np.arange(193, dtype=np.float32) - 96.0)[:, None, None], (193, 229, 193)
    ).copy()
    stack = extract_slices(volume, geometry)
    assert stack[0][0, 0] == pytest.approx(-96.0)
    assert stack[0][0, -1] == pytest.approx(95.0)


def test_extract_slices_rejects_a_mismatched_volume() -> None:
    """A volume that is not on the template grid is refused."""
    geometry = geometry_from_affine(mni_affine(), (193, 229, 193))
    with pytest.raises(PreprocessError):
        extract_slices(np.zeros((64, 64, 64), dtype=np.float32), geometry)


@requires_template
@pytest.mark.integration
def test_real_template_geometry_matches_the_derivation() -> None:
    """The shipped MNI152 template reproduces the derived window and plane coordinates."""
    geometry = template_geometry(TEMPLATE_PATH)
    assert geometry.shape == (193, 229, 193)
    assert geometry.origin_voxel == (96.0, 132.0)
    assert geometry.x_slice == (0, 192)
    assert geometry.y_slice == (36, 228)
    np.testing.assert_allclose(geometry.z_mm, [float(k - 78) for k in SLICE_Z_INDICES])


@requires_template
@pytest.mark.integration
def test_real_template_slices_are_192_square_and_anterior_first() -> None:
    """Slicing the template itself gives ten 192-px planes with anterior in row 0."""
    import nibabel as nib

    geometry = template_geometry(TEMPLATE_PATH)
    volume = np.asarray(nib.load(str(TEMPLATE_PATH)).dataobj, dtype=np.float32)
    stack = extract_slices(volume, geometry)

    assert stack.shape == (10, 192, 192)
    # The affine is the authority: row 0 must be the row of largest world y.
    y_of_row = geometry.affine[1, 3] + geometry.affine[1, 1] * np.arange(
        geometry.y_slice[1] - 1, geometry.y_slice[0] - 1, -1
    )
    assert y_of_row[0] == max(y_of_row)
    assert y_of_row[0] > y_of_row[-1]
    x_of_col = geometry.affine[0, 3] + geometry.affine[0, 0] * np.arange(*geometry.x_slice)
    assert x_of_col[0] == min(x_of_col), "column 0 must be the subject's left"
    assert stack.max() > stack.min()
