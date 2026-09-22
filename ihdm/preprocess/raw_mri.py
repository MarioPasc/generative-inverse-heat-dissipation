"""Discovery and physical-space loading of the raw IXI and OASIS-1 T1 volumes.

Two cohorts, two file formats, one output type. IXI ships valid NIfTI-1 with usable
(slightly oblique) direction cosines, so SimpleITK reads it directly and the obliquity is
absorbed by the single rigid resampling later in the pipeline. OASIS-1 ships Analyze 7.5,
which carries no usable orientation at all; its axis map was established empirically in
``projects/GenAI/analysis/unprocessed_loaders.py`` (mid-plane contact sheets plus
translation-aligned cross-correlation against the FOMO60K copy of the same scan, 12 of 12
subjects) and is ported here verbatim rather than re-derived from the header.

Every function returns SimpleITK images in LPS physical space, the convention SimpleITK
itself uses, so a subject volume and the MNI152 template are directly comparable.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from ihdm.preprocess.errors import PreprocessError

__all__ = [
    "COHORTS",
    "cohort_raw_root",
    "RawVolumeRef",
    "list_subjects",
    "load_sitk",
    "ras_array_to_sitk",
    "sitk_to_ras_array",
]

logger = logging.getLogger(__name__)

COHORTS: tuple[str, ...] = ("ixi", "oasis1")

#: RAS -> LPS direction of a volume whose array axes increase towards R, A and S.
#: SimpleITK works in LPS, so index axis 0 (towards R) is -x, axis 1 (towards A) is -y and
#: axis 2 (towards S) is +z. Columns of the matrix are the index-axis directions.
_RAS_INDEX_DIRECTION_LPS: tuple[float, ...] = (
    -1.0, 0.0, 0.0,
    0.0, -1.0, 0.0,
    0.0, 0.0, 1.0,
)

_IXI_SUBJECT_RE = re.compile(r"^(IXI\d+)")
_OASIS_SUBJECT_RE = re.compile(r"^(OAS1_\d+)_MR1$")


@dataclass(frozen=True)
class RawVolumeRef:
    """One raw volume of one subject.

    Parameters
    ----------
    cohort : str
        ``"ixi"`` or ``"oasis1"``.
    subject : str
        Subject identifier (``IXI002``, ``OAS1_0001``).
    path : Path
        Absolute path of the file passed to the reader (the ``.nii.gz`` for IXI, the
        Analyze ``.hdr`` for OASIS-1).
    source : str
        Path of ``path`` relative to the cohort's raw root, stored in ``index.csv``.
    """

    cohort: str
    subject: str
    path: Path
    source: str


def cohort_raw_root(cohort: str, raw_root: Path) -> Path:
    """Return the raw-data directory of one cohort.

    Parameters
    ----------
    cohort : str
        ``"ixi"`` or ``"oasis1"``.
    raw_root : Path
        Root of the unprocessed data tree (``ihdm.paths.raw_root()``).

    Returns
    -------
    Path
        ``<raw_root>/UNPROCESSED_MRI/HEALTHY/IXI/T1`` or ``.../OASIS1``.

    Raises
    ------
    PreprocessError
        If the cohort name is unknown.
    """
    healthy = Path(raw_root) / "UNPROCESSED_MRI" / "HEALTHY"
    if cohort == "ixi":
        return healthy / "IXI" / "T1"
    if cohort == "oasis1":
        return healthy / "OASIS1"
    raise PreprocessError(f"unknown cohort {cohort!r}; expected one of {COHORTS}")


def list_subjects(cohort: str, raw_root: Path, limit: int | None = None) -> list[RawVolumeRef]:
    """List one raw volume per subject of a cohort, sorted by subject id.

    IXI keeps one T1 per subject; OASIS-1 keeps only the first MPRAGE (``mpr-1``) of the
    first session (``MR1``), so later sessions of the same participant are never mixed in.

    Parameters
    ----------
    cohort : str
        ``"ixi"`` or ``"oasis1"``.
    raw_root : Path
        Root of the unprocessed data tree.
    limit : int | None
        Keep only the first ``limit`` subjects (after sorting), for dry runs.

    Returns
    -------
    list[RawVolumeRef]
        One reference per subject, sorted ascending by subject id.

    Raises
    ------
    PreprocessError
        If the cohort is unknown, its directory is missing, no volume is found, or a
        subject id appears twice.
    """
    root = cohort_raw_root(cohort, raw_root)
    if not root.is_dir():
        raise PreprocessError(f"raw directory for cohort {cohort!r} not found: {root}")

    if cohort == "ixi":
        refs = _list_ixi(root)
    else:
        refs = _list_oasis1(root)

    if not refs:
        raise PreprocessError(f"no raw volumes found for cohort {cohort!r} under {root}")

    duplicates = _duplicate_subjects(refs)
    if duplicates:
        raise PreprocessError(f"cohort {cohort!r}: duplicate subject id(s) {duplicates}")

    refs.sort(key=lambda r: r.subject)
    logger.info("cohort %s: %d subjects under %s", cohort, len(refs), root)
    return refs[:limit] if limit is not None else refs


def _list_ixi(root: Path) -> list[RawVolumeRef]:
    """Collect the IXI T1 NIfTI files, one per ``IXInnn`` subject."""
    refs: list[RawVolumeRef] = []
    for path in root.glob("*/IXI*-T1.nii.gz"):
        match = _IXI_SUBJECT_RE.match(path.name)
        if match is None:
            raise PreprocessError(f"IXI file name does not start with a subject id: {path}")
        refs.append(
            RawVolumeRef(
                cohort="ixi",
                subject=match.group(1),
                path=path.resolve(),
                source=str(path.relative_to(root)),
            )
        )
    return refs


def _list_oasis1(root: Path) -> list[RawVolumeRef]:
    """Collect the OASIS-1 first-session, first-MPRAGE Analyze headers."""
    refs: list[RawVolumeRef] = []
    for path in root.glob("disc*/OAS1_*_MR1/RAW/*_mpr-1_anon.hdr"):
        session_dir = path.parent.parent.name
        match = _OASIS_SUBJECT_RE.match(session_dir)
        if match is None:
            raise PreprocessError(f"OASIS-1 session folder is not OAS1_xxxx_MR1: {path}")
        if not path.with_suffix(".img").is_file():
            raise PreprocessError(f"OASIS-1 header without its .img companion: {path}")
        refs.append(
            RawVolumeRef(
                cohort="oasis1",
                subject=match.group(1),
                path=path.resolve(),
                source=str(path.relative_to(root)),
            )
        )
    return refs


def _duplicate_subjects(refs: list[RawVolumeRef]) -> list[str]:
    """Return the subject ids that occur more than once in ``refs``."""
    seen: set[str] = set()
    duplicates: set[str] = set()
    for ref in refs:
        if ref.subject in seen:
            duplicates.add(ref.subject)
        seen.add(ref.subject)
    return sorted(duplicates)


def load_sitk(ref: RawVolumeRef, flip_lr: bool = False) -> sitk.Image:
    """Read one raw volume as a ``float32`` SimpleITK image in LPS physical space.

    Parameters
    ----------
    ref : RawVolumeRef
        The volume to read.
    flip_lr : bool
        OASIS-1 only: reverse the left-right polarity of the ported axis map. Used solely
        by the polarity check of the QC stage; never by the production pipeline.

    Returns
    -------
    sitk.Image
        ``float32`` image with correct spacing and direction cosines.

    Raises
    ------
    PreprocessError
        If the file cannot be read or is not three-dimensional.
    """
    if ref.cohort == "ixi":
        return _load_nifti(ref.path)
    if ref.cohort == "oasis1":
        return _load_analyze_oasis1(ref.path, flip_lr=flip_lr)
    raise PreprocessError(f"unknown cohort {ref.cohort!r}; expected one of {COHORTS}")


def _load_nifti(path: Path) -> sitk.Image:
    """Read a NIfTI volume with SimpleITK, keeping its exact direction cosines."""
    try:
        image = sitk.ReadImage(str(path), sitk.sitkFloat32)
    except RuntimeError as exc:
        raise PreprocessError(f"cannot read NIfTI volume {path}: {exc}") from exc
    if image.GetDimension() != 3:
        raise PreprocessError(f"{path} has dimension {image.GetDimension()}, expected 3")
    return image


def _load_analyze_oasis1(path: Path, flip_lr: bool = False) -> sitk.Image:
    """Read an OASIS-1 Analyze volume and orient it with the ported empirical axis map.

    The map, from ``analysis/unprocessed_loaders.py::load_volume_ras``, is
    ``np.transpose(raw, (2, 0, 1))[::-1]`` with the zooms permuted the same way, which
    turns the stored ``(P->A, I->S, R->L)`` axes into array axes increasing towards R, A
    and S. nibabel's SPM2-Analyze affine is deliberately ignored: Analyze 7.5 carries no
    orientation and that affine is a guess.
    """
    try:
        handle = nib.load(str(path))
        raw = np.squeeze(np.asarray(handle.dataobj, dtype=np.float32))
        zooms_raw = tuple(float(z) for z in handle.header.get_zooms()[:3])
    except Exception as exc:  # nibabel raises a wide variety of errors on bad files
        raise PreprocessError(f"cannot read Analyze volume {path}: {exc}") from exc

    if raw.ndim != 3:
        raise PreprocessError(f"{path} has {raw.ndim} dimensions after squeeze, expected 3")

    volume = np.transpose(raw, (2, 0, 1))
    volume = volume[::-1] if not flip_lr else volume
    spacing = (zooms_raw[2], zooms_raw[0], zooms_raw[1])
    return ras_array_to_sitk(np.ascontiguousarray(volume), spacing)


def ras_array_to_sitk(
    volume: np.ndarray, spacing: tuple[float, float, float], centre_origin: bool = True
) -> sitk.Image:
    """Wrap a RAS-index-ordered array in a SimpleITK image in LPS physical space.

    Parameters
    ----------
    volume : np.ndarray
        Array whose index axes increase towards R (axis 0), A (axis 1) and S (axis 2).
    spacing : tuple[float, float, float]
        Voxel size in millimetres along those three index axes.
    centre_origin : bool
        Place the physical origin so the volume centre sits at ``(0, 0, 0)`` in LPS.
        The registration uses a moments initialiser, so the absolute origin is
        immaterial; centring only keeps the initial overlap with the template sensible.

    Returns
    -------
    sitk.Image
        ``float32`` image with ``direction = diag(-1, -1, 1)``.
    """
    # SimpleITK's numpy view is index-reversed: GetArrayFromImage gives (k, j, i).
    image = sitk.GetImageFromArray(np.ascontiguousarray(volume.transpose(2, 1, 0), np.float32))
    image.SetSpacing(tuple(float(s) for s in spacing))
    image.SetDirection(_RAS_INDEX_DIRECTION_LPS)
    if centre_origin:
        half = (np.array(volume.shape, dtype=float) - 1.0) / 2.0 * np.array(spacing, dtype=float)
        # LPS coordinate of index 0 along each axis, so the volume centre lands on zero.
        # Axes 0 and 1 point towards R and A, i.e. -x and -y in LPS; axis 2 towards +z.
        image.SetOrigin((float(half[0]), float(half[1]), float(-half[2])))
    return image


def sitk_to_ras_array(image: sitk.Image) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Return an image as an array whose axes increase towards R, A and S.

    Permutation and axis flips only; no interpolation. Oblique direction cosines are
    snapped to their nearest axis, which is adequate for the QC display of raw volumes
    and is never used on data that enters ``images.npy`` (those are already on the
    axis-aligned template grid).

    Parameters
    ----------
    image : sitk.Image
        Any three-dimensional image in LPS physical space.

    Returns
    -------
    tuple[np.ndarray, tuple[float, float, float]]
        The reoriented array and its voxel sizes along the R, A and S axes.

    Raises
    ------
    PreprocessError
        If two index axes map onto the same world axis.
    """
    array = sitk.GetArrayFromImage(image).transpose(2, 1, 0)
    spacing = np.array(image.GetSpacing(), dtype=float)
    lps_to_ras = np.diag([-1.0, -1.0, 1.0])
    direction_ras = lps_to_ras @ np.array(image.GetDirection(), dtype=float).reshape(3, 3)

    world_axis = np.argmax(np.abs(direction_ras), axis=0)
    if len(set(world_axis.tolist())) != 3:
        raise PreprocessError(f"degenerate direction cosines: {image.GetDirection()}")

    permutation = np.argsort(world_axis)
    array = np.transpose(array, permutation)
    spacing_out = spacing[permutation]
    for axis in range(3):
        source = permutation[axis]
        if direction_ras[world_axis[source], source] < 0:
            array = np.flip(array, axis=axis)
    return np.ascontiguousarray(array), tuple(float(s) for s in spacing_out)
