"""Geometry, intensity and slicing of the registered MRI volumes.

Everything here is a pure function of arrays and of the template's affine. The template
grid is the only source of geometric truth: the ten axial planes are fixed template voxel
z indices (decision D1'), the in-plane window is the 192-pixel block centred on the voxel
that carries the MNI origin, and the display orientation is derived from the affine rather
than assumed. No resampling happens in this module; the registered volumes already live on
the template lattice.

Stored orientation (``03-data-format.md`` §5): axis 0 runs anterior to posterior (row 0 is
the most anterior row) and axis 1 runs subject-left to subject-right (column 0 is the
subject's left, the neurological convention).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from ihdm.preprocess.errors import PreprocessError

__all__ = [
    "IMAGE_SIZE",
    "affine_from_sitk",
    "assert_grid_matches_affine",
    "ORIENTATION",
    "PIPELINE_VERSION",
    "SLICE_Z_INDICES",
    "TemplateGeometry",
    "extract_slices",
    "scale_intensity",
    "slices_to_uint8",
    "template_geometry",
    "to_display_orientation",
]

logger = logging.getLogger(__name__)

PIPELINE_VERSION = "1.0"
IMAGE_SIZE = 192

#: Template voxel z indices of the ten axial planes (decision D1' of 00-overview.md).
SLICE_Z_INDICES: tuple[int, ...] = (55, 63, 71, 79, 87, 96, 104, 112, 120, 128)

ORIENTATION = (
    "axis0 = anterior->posterior (row 0 is the most anterior row); "
    "axis1 = subject-left->subject-right (column 0 is the subject's left, "
    "neurological convention); derived from the MNI152NLin2009cAsym affine"
)


@dataclass(frozen=True)
class TemplateGeometry:
    """In-plane window and axial planes of the template grid, derived from its affine.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Template array shape in ``(i, j, k)`` index order.
    affine : np.ndarray
        The 4x4 voxel-to-RAS affine of the template.
    z_indices : tuple[int, ...]
        Template voxel z indices of the axial planes.
    z_mm : tuple[float, ...]
        MNI z coordinate of each plane, in millimetres.
    origin_voxel : tuple[float, float]
        Voxel indices ``(i0, j0)`` of the MNI in-plane origin (x = 0, y = 0 mm).
    x_slice : tuple[int, int]
        Half-open range of ``i`` covered by the window.
    y_slice : tuple[int, int]
        Half-open range of ``j`` covered by the window.
    size : int
        Side of the square window, in pixels.
    """

    shape: tuple[int, int, int]
    affine: np.ndarray
    z_indices: tuple[int, ...]
    z_mm: tuple[float, ...]
    origin_voxel: tuple[float, float]
    x_slice: tuple[int, int]
    y_slice: tuple[int, int]
    size: int

    def to_json(self) -> dict[str, object]:
        """Return the JSON-serialisable description stored in ``meta.json``.

        Returns
        -------
        dict[str, object]
            Window, planes and the affine used to derive them.
        """
        return {
            "template_shape": list(self.shape),
            "template_affine": [[float(v) for v in row] for row in self.affine],
            "slice_z_indices": list(self.z_indices),
            "slice_z_mm": list(self.z_mm),
            "mni_origin_voxel": list(self.origin_voxel),
            "window_x_voxels": list(self.x_slice),
            "window_y_voxels": list(self.y_slice),
            "window_size": self.size,
            "window_rule": (
                "the size x size block of the template grid centred on the voxel of the "
                "MNI in-plane origin (x = 0 mm, y = 0 mm), derived from the affine"
            ),
        }


def template_geometry(
    template_path: Path,
    z_indices: tuple[int, ...] = SLICE_Z_INDICES,
    size: int = IMAGE_SIZE,
) -> TemplateGeometry:
    """Derive the slice planes and the in-plane window from the template affine.

    Parameters
    ----------
    template_path : Path
        The template NIfTI whose affine defines the grid.
    z_indices : tuple[int, ...]
        Template voxel z indices of the axial planes.
    size : int
        Side of the square window, in pixels.

    Returns
    -------
    TemplateGeometry
        The derived geometry.

    Raises
    ------
    PreprocessError
        If the template is not an axis-aligned RAS grid, if the MNI origin does not fall
        on a voxel centre, if the window leaves the grid, or if a plane is out of range.
    """
    handle = nib.load(str(template_path))
    affine = np.asarray(handle.affine, dtype=float)
    shape = tuple(int(s) for s in handle.shape[:3])
    return geometry_from_affine(affine, shape, z_indices=z_indices, size=size)


def geometry_from_affine(
    affine: np.ndarray,
    shape: tuple[int, int, int],
    z_indices: tuple[int, ...] = SLICE_Z_INDICES,
    size: int = IMAGE_SIZE,
) -> TemplateGeometry:
    """Derive the geometry from an affine and a shape (the testable core).

    Parameters
    ----------
    affine : np.ndarray
        4x4 voxel-to-RAS affine; must be diagonal with positive entries, so that index
        axes 0, 1 and 2 increase towards R, A and S.
    shape : tuple[int, int, int]
        Grid shape in ``(i, j, k)`` order.
    z_indices : tuple[int, ...]
        Voxel z indices of the axial planes.
    size : int
        Side of the square window.

    Returns
    -------
    TemplateGeometry
        The derived geometry.

    Raises
    ------
    PreprocessError
        On a non-RAS-diagonal affine, a non-integral MNI origin voxel, a window that
        leaves the grid, or a plane outside the grid.
    """
    affine = np.asarray(affine, dtype=float)
    linear = affine[:3, :3]
    if not np.allclose(linear, np.diag(np.diag(linear)), atol=1e-6):
        raise PreprocessError(f"template affine is not axis-aligned:\n{linear}")
    if not np.all(np.diag(linear) > 0):
        raise PreprocessError(
            f"template affine is not RAS-positive (diag {np.diag(linear)}); the stored "
            "orientation rule assumes index axes increasing towards R, A and S"
        )

    origin_i = -affine[0, 3] / linear[0, 0]
    origin_j = -affine[1, 3] / linear[1, 1]
    for name, value in (("i", origin_i), ("j", origin_j)):
        if abs(value - round(value)) > 1e-6:
            raise PreprocessError(
                f"the MNI in-plane origin does not fall on a voxel centre along {name} "
                f"(index {value:.6f})"
            )

    half = size // 2
    x_start, y_start = int(round(origin_i)) - half, int(round(origin_j)) - half
    x_slice, y_slice = (x_start, x_start + size), (y_start, y_start + size)
    if x_slice[0] < 0 or x_slice[1] > shape[0]:
        raise PreprocessError(f"window {x_slice} leaves the grid along i (extent {shape[0]})")
    if y_slice[0] < 0 or y_slice[1] > shape[1]:
        raise PreprocessError(f"window {y_slice} leaves the grid along j (extent {shape[1]})")

    bad = [k for k in z_indices if not 0 <= k < shape[2]]
    if bad:
        raise PreprocessError(f"slice z indices {bad} leave the grid (extent {shape[2]})")

    z_mm = tuple(
        float(affine[2, 3] + linear[2, 2] * k + linear[2, 0] * origin_i + linear[2, 1] * origin_j)
        for k in z_indices
    )
    return TemplateGeometry(
        shape=shape,
        affine=affine,
        z_indices=tuple(int(k) for k in z_indices),
        z_mm=z_mm,
        origin_voxel=(float(origin_i), float(origin_j)),
        x_slice=x_slice,
        y_slice=y_slice,
        size=size,
    )


def affine_from_sitk(image: object) -> np.ndarray:
    """Return the voxel-to-RAS affine of a SimpleITK image.

    SimpleITK works in LPS, nibabel in RAS, and the two libraries index a NIfTI array in
    the same order, so the affine is ``diag(-1, -1, 1)`` applied to the LPS mapping.

    Parameters
    ----------
    image : sitk.Image
        Any three-dimensional SimpleITK image.

    Returns
    -------
    np.ndarray
        The 4x4 voxel-to-RAS affine.
    """
    direction = np.array(image.GetDirection(), dtype=float).reshape(3, 3)
    spacing = np.array(image.GetSpacing(), dtype=float)
    lps_to_ras = np.diag([-1.0, -1.0, 1.0])
    affine = np.eye(4)
    affine[:3, :3] = lps_to_ras @ direction @ np.diag(spacing)
    affine[:3, 3] = lps_to_ras @ np.array(image.GetOrigin(), dtype=float)
    return affine


def assert_grid_matches_affine(image: object, geometry: TemplateGeometry) -> None:
    """Check that a SimpleITK image sits on exactly the grid the geometry was derived from.

    The whole orientation contract rests on SimpleITK and nibabel indexing the template
    array identically. This turns that assumption into a checked precondition, so a future
    template with a different storage order fails loudly instead of producing mirrored
    slices.

    Parameters
    ----------
    image : sitk.Image
        The image to check (the template, or a volume resampled onto it).
    geometry : TemplateGeometry
        The geometry derived from the template's nibabel affine.

    Raises
    ------
    PreprocessError
        If the size or the affine differ.
    """
    size = tuple(int(s) for s in image.GetSize())
    if size != geometry.shape:
        raise PreprocessError(
            f"image size {size} does not match the template geometry {geometry.shape}"
        )
    actual = affine_from_sitk(image)
    if not np.allclose(actual, geometry.affine, atol=1e-4):
        raise PreprocessError(
            "the SimpleITK grid and the nibabel affine disagree; the stored orientation "
            f"cannot be trusted.\nSimpleITK:\n{actual}\nnibabel:\n{geometry.affine}"
        )


def scale_intensity(
    volume: np.ndarray, foreground: np.ndarray, percentile: float = 99.0
) -> tuple[np.ndarray, dict[str, float]]:
    """Scale a registered volume to ``[0, 1]`` by its foreground percentile.

    The foreground is the template brain mask dilated by 10 mm, used only to pick the
    percentile; the mask is never applied to the image, so the acquisition's background
    noise survives into the stored images (decision of ``learning/03`` §21.1a). There is
    no low clip: zero maps to zero.

    Parameters
    ----------
    volume : np.ndarray
        Registered volume on the template grid, any shape matching ``foreground``.
    foreground : np.ndarray
        Boolean mask of the same shape.
    percentile : float
        Percentile of the foreground voxels mapped to 1.0.

    Returns
    -------
    tuple[np.ndarray, dict[str, float]]
        The scaled ``float32`` volume and diagnostics (``p99``, ``clip_high``,
        ``foreground_voxels``).

    Raises
    ------
    PreprocessError
        If the shapes differ, the foreground is empty, or the percentile is not positive
        (a volume that is constant or empty inside the brain).
    """
    if volume.shape != foreground.shape:
        raise PreprocessError(
            f"volume shape {volume.shape} does not match foreground {foreground.shape}"
        )
    n_foreground = int(foreground.sum())
    if n_foreground == 0:
        raise PreprocessError("foreground mask is empty")

    p_high = float(np.percentile(volume[foreground], percentile))
    if not np.isfinite(p_high) or p_high <= 0.0:
        raise PreprocessError(f"foreground p{percentile:g} is {p_high}, expected a positive value")

    scaled = np.clip(volume.astype(np.float32) / p_high, 0.0, 1.0)
    info = {
        "p99": p_high,
        "clip_high": float((volume > p_high).mean()),
        "foreground_voxels": float(n_foreground),
    }
    return scaled, info


def to_display_orientation(plane_ij: np.ndarray) -> np.ndarray:
    """Turn an ``(i, j)`` = ``(R, A)`` plane into the stored ``(A->P, L->R)`` orientation.

    Parameters
    ----------
    plane_ij : np.ndarray
        Two-dimensional array whose axis 0 increases towards the subject's right and
        whose axis 1 increases towards the subject's anterior.

    Returns
    -------
    np.ndarray
        Array whose axis 0 runs anterior to posterior and whose axis 1 runs
        subject-left to subject-right.

    Raises
    ------
    PreprocessError
        If the input is not two-dimensional.
    """
    if plane_ij.ndim != 2:
        raise PreprocessError(f"expected a 2-D plane, got shape {plane_ij.shape}")
    # Transpose puts j (anterior) on the rows, then reversing the rows puts the most
    # anterior row first; the columns stay i, which increases towards the subject's right,
    # so column 0 is the subject's left.
    return np.ascontiguousarray(plane_ij.T[::-1, :])


def extract_slices(volume: np.ndarray, geometry: TemplateGeometry) -> np.ndarray:
    """Cut the ten windowed axial planes out of a volume on the template grid.

    Parameters
    ----------
    volume : np.ndarray
        Volume on the template grid in ``(i, j, k)`` index order.
    geometry : TemplateGeometry
        Window and plane definition.

    Returns
    -------
    np.ndarray
        Stack of shape ``(len(z_indices), size, size)`` in the stored orientation.

    Raises
    ------
    PreprocessError
        If the volume shape does not match the template shape.
    """
    if tuple(volume.shape) != geometry.shape:
        raise PreprocessError(
            f"volume shape {volume.shape} does not match the template {geometry.shape}"
        )
    x0, x1 = geometry.x_slice
    y0, y1 = geometry.y_slice
    block = volume[x0:x1, y0:y1, :]
    return np.stack([to_display_orientation(block[:, :, k]) for k in geometry.z_indices])


def slices_to_uint8(slices01: np.ndarray) -> np.ndarray:
    """Quantise slices in ``[0, 1]`` to ``uint8``.

    Parameters
    ----------
    slices01 : np.ndarray
        Array with values in ``[0, 1]``.

    Returns
    -------
    np.ndarray
        ``uint8`` array of the same shape, ``round(x * 255)``.

    Raises
    ------
    PreprocessError
        If any value falls outside ``[0, 1]`` or is not finite.
    """
    if not np.all(np.isfinite(slices01)):
        raise PreprocessError("slices contain non-finite values")
    if float(slices01.min()) < 0.0 or float(slices01.max()) > 1.0:
        raise PreprocessError(
            f"slices leave [0, 1]: min {float(slices01.min())}, max {float(slices01.max())}"
        )
    return np.round(np.asarray(slices01, dtype=np.float64) * 255.0).astype(np.uint8)
