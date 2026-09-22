"""Tests of raw-volume discovery and of the RAS/LPS orientation round trip.

The OASIS-1 axis map is the one piece of geometry no header can confirm, so it is pinned
here against the empirically established map of
``projects/GenAI/analysis/unprocessed_loaders.py::load_volume_ras``: a change to
:func:`ihdm.preprocess.raw_mri.load_sitk` that silently mirrors the cohort fails this file.
"""

from __future__ import annotations

import numpy as np
import pytest
import SimpleITK as sitk

from ihdm.paths import raw_root
from ihdm.preprocess.errors import PreprocessError
from ihdm.preprocess.mri import affine_from_sitk
from ihdm.preprocess.raw_mri import (
    COHORTS,
    RawVolumeRef,
    cohort_raw_root,
    list_subjects,
    load_sitk,
    ras_array_to_sitk,
    sitk_to_ras_array,
)

RAW_ROOT = raw_root()
requires_raw = pytest.mark.skipif(
    not (RAW_ROOT / "UNPROCESSED_MRI" / "HEALTHY").is_dir(),
    reason=f"raw MRI tree not present under {RAW_ROOT}",
)


def marked_volume() -> np.ndarray:
    """Return a volume with a unique bright voxel at right-anterior-superior."""
    volume = np.zeros((6, 7, 8), dtype=np.float32)
    volume[5, 6, 7] = 1.0  # largest i (R), largest j (A), largest k (S)
    volume[0, 0, 0] = 0.5  # left-posterior-inferior, to break the symmetry
    return volume


def test_ras_array_to_sitk_sets_the_lps_direction() -> None:
    """Index axes increasing towards R, A and S become ``diag(-1, -1, 1)`` in LPS."""
    image = ras_array_to_sitk(marked_volume(), (1.0, 2.0, 3.0))
    np.testing.assert_allclose(
        np.array(image.GetDirection()).reshape(3, 3), np.diag([-1.0, -1.0, 1.0])
    )
    assert image.GetSize() == (6, 7, 8)
    np.testing.assert_allclose(image.GetSpacing(), (1.0, 2.0, 3.0))


def test_ras_array_to_sitk_keeps_the_index_order() -> None:
    """The bright voxel keeps its ``(i, j, k)`` position through the SimpleITK wrapping."""
    volume = marked_volume()
    image = ras_array_to_sitk(volume, (1.0, 1.0, 1.0))
    recovered = sitk.GetArrayFromImage(image).transpose(2, 1, 0)
    np.testing.assert_array_equal(recovered, volume)


def test_ras_array_to_sitk_affine_is_ras_positive() -> None:
    """The derived voxel-to-RAS affine has a positive diagonal, as the slicing assumes."""
    image = ras_array_to_sitk(marked_volume(), (1.0, 2.0, 3.0))
    affine = affine_from_sitk(image)
    np.testing.assert_allclose(np.diag(affine[:3, :3]), (1.0, 2.0, 3.0))
    assert np.allclose(affine[:3, :3], np.diag(np.diag(affine[:3, :3])))


def test_sitk_to_ras_array_round_trips() -> None:
    """Wrapping an array and reading it back is the identity."""
    volume = marked_volume()
    recovered, spacing = sitk_to_ras_array(ras_array_to_sitk(volume, (1.0, 2.0, 3.0)))
    np.testing.assert_array_equal(recovered, volume)
    assert spacing == (1.0, 2.0, 3.0)


def test_sitk_to_ras_array_undoes_a_permuted_and_flipped_grid() -> None:
    """A volume stored in an arbitrary axis order comes back in R, A, S order."""
    volume = marked_volume()
    # Store the same anatomy as (S, R->L reversed, A) with matching direction cosines.
    stored = np.transpose(volume, (2, 0, 1))[:, ::-1, :]
    image = sitk.GetImageFromArray(np.ascontiguousarray(stored.transpose(2, 1, 0)))
    image.SetSpacing((3.0, 1.0, 2.0))
    # Columns: index 0 -> +S(+z), index 1 -> L(+x in LPS), index 2 -> A(-y in LPS).
    image.SetDirection((0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0))

    recovered, spacing = sitk_to_ras_array(image)

    np.testing.assert_array_equal(recovered, volume)
    assert spacing == (1.0, 2.0, 3.0)


def test_degenerate_direction_cosines_never_reach_the_reorientation() -> None:
    """Two index axes on one world axis is refused, by SimpleITK or by the reorientation.

    SimpleITK validates orthonormality in ``SetDirection``, so the guard inside
    :func:`sitk_to_ras_array` is defensive; this test pins that one of the two layers
    always rejects the case instead of silently guessing a permutation.
    """
    image = sitk.GetImageFromArray(np.zeros((4, 4, 4), dtype=np.float32))
    with pytest.raises((RuntimeError, PreprocessError)):
        image.SetDirection((1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0))
        sitk_to_ras_array(image)


def test_cohort_raw_root_rejects_an_unknown_cohort() -> None:
    """Only the two MRI cohorts of this ticket are accepted."""
    with pytest.raises(PreprocessError):
        cohort_raw_root("lsun_church", RAW_ROOT)
    assert set(COHORTS) == {"ixi", "oasis1"}


def test_list_subjects_rejects_a_missing_tree(tmp_path) -> None:
    """A wrong raw root fails loudly instead of returning an empty cohort."""
    with pytest.raises(PreprocessError):
        list_subjects("ixi", tmp_path)


def test_load_sitk_rejects_an_unknown_cohort(tmp_path) -> None:
    """``load_sitk`` dispatches on the cohort and refuses anything else."""
    ref = RawVolumeRef(cohort="other", subject="X", path=tmp_path / "x.nii.gz", source="x")
    with pytest.raises(PreprocessError):
        load_sitk(ref)


def test_load_sitk_reports_an_unreadable_file(tmp_path) -> None:
    """A truncated NIfTI raises ``PreprocessError``, not a bare SimpleITK RuntimeError."""
    broken = tmp_path / "broken.nii.gz"
    broken.write_bytes(b"not a nifti")
    ref = RawVolumeRef(cohort="ixi", subject="X", path=broken, source="broken.nii.gz")
    with pytest.raises(PreprocessError):
        load_sitk(ref)


@requires_raw
@pytest.mark.integration
def test_real_cohorts_have_one_volume_per_subject() -> None:
    """IXI has 581 subjects and OASIS-1 416 first-session subjects, none duplicated."""
    ixi = list_subjects("ixi", RAW_ROOT)
    oasis = list_subjects("oasis1", RAW_ROOT)

    assert len(ixi) == 581
    assert len(oasis) == 416
    for refs, prefix in ((ixi, "IXI"), (oasis, "OAS1_")):
        assert len({r.subject for r in refs}) == len(refs)
        assert all(r.subject.startswith(prefix) for r in refs)
        assert [r.subject for r in refs] == sorted(r.subject for r in refs)
    assert all("_MR1/" in r.source and "mpr-1" in r.source for r in oasis)


@requires_raw
@pytest.mark.integration
def test_oasis_axis_map_matches_the_ported_reference() -> None:
    """``load_sitk`` reproduces ``unprocessed_loaders.load_volume_ras`` voxel for voxel.

    The reference map is ``transpose(raw, (2, 0, 1))[::-1]`` with the zooms permuted the
    same way; it was established empirically, not read from the Analyze header.
    """
    import nibabel as nib

    ref = list_subjects("oasis1", RAW_ROOT, limit=1)[0]
    raw = np.squeeze(np.asarray(nib.load(str(ref.path)).dataobj, dtype=np.float32))
    zooms = tuple(float(z) for z in nib.load(str(ref.path)).header.get_zooms()[:3])
    expected = np.ascontiguousarray(np.transpose(raw, (2, 0, 1))[::-1])

    image = load_sitk(ref)
    actual, spacing = sitk_to_ras_array(image)

    np.testing.assert_array_equal(actual, expected)
    assert spacing == (zooms[2], zooms[0], zooms[1]) == (1.25, 1.0, 1.0)
    assert image.GetSize() == (128, 256, 256)


@requires_raw
@pytest.mark.integration
def test_oasis_flip_lr_really_mirrors_the_volume() -> None:
    """The polarity switch used by the QC check is an exact left-right mirror."""
    ref = list_subjects("oasis1", RAW_ROOT, limit=1)[0]
    ported, _ = sitk_to_ras_array(load_sitk(ref))
    flipped, _ = sitk_to_ras_array(load_sitk(ref, flip_lr=True))
    np.testing.assert_array_equal(flipped, ported[::-1])


@requires_raw
@pytest.mark.integration
def test_ixi_keeps_its_oblique_direction_cosines() -> None:
    """IXI is read through its own affine; the obliquity is not snapped away."""
    ref = list_subjects("ixi", RAW_ROOT, limit=1)[0]
    image = load_sitk(ref)
    direction = np.array(image.GetDirection()).reshape(3, 3)
    assert image.GetSize() == (256, 256, 150)
    # At least one column is off-axis: snapping it would be an uncorrected rotation.
    assert np.abs(direction).max(axis=0).min() < 0.999
