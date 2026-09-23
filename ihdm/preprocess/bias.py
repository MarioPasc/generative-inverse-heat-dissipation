"""N4 bias-field correction of the registered MRI volumes (decision D15).

The stage sits between T1.1's registered cache and the foreground-p99 intensity scaling.
It exists because T1.3 measured 84 % of IXI's coarse-bin (0.5-1 cycles per image)
between-image variance in the single DCT mode ``(1, 0)``, an anterior-posterior
between-subject intensity ramp: a receive-coil / multi-site residue of the acquisition, not
coarse brain anatomy. N4 (Tustison et al., *N4ITK: improved N3 bias correction*, IEEE TMI
29(6):1310-1320, 2010, doi:10.1109/TMI.2010.2046908) is the standard estimator of that
artefact and is applied with **library defaults only**; no parameter of this module was ever
chosen by looking at a spectral number.

The estimate is computed on a shrink-factor-4 image with the dilated template brain mask —
the same foreground the intensity rule uses — and the resulting B-spline log bias field is
then evaluated on the full-resolution input, so no data is interpolated: the correction is a
voxelwise division by a smooth positive field.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from ihdm.preprocess.errors import PreprocessError

__all__ = [
    "N4Config",
    "N4Result",
    "n4_correct",
    "binary_mask",
    "read_n4_sidecar",
    "simpleitk_n4_defaults",
    "write_corrected",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class N4Config:
    """Every parameter of the N4 stage, all of them SimpleITK's own defaults.

    The values are written out rather than left implicit so that ``meta.json`` can list
    "every parameter actually used" and so that a SimpleITK release that moved a default is
    caught by :func:`simpleitk_n4_defaults` instead of silently changing the data.

    Parameters
    ----------
    shrink_factor : int
        Downsampling factor applied to the image and the mask before the estimate. The
        field is a low-order B-spline, so estimating it at 4 mm and evaluating it at 1 mm
        costs nothing in accuracy and about 60x in time.
    max_iterations : tuple[int, ...]
        Iterations per fitting level; the length sets the number of fitting levels.
    convergence_threshold : float
        Coefficient-of-variation threshold of the field update below which a level stops.
    spline_order : int
        Order of the B-spline field.
    number_of_control_points : tuple[int, int, int]
        Control points of the coarsest fitting level, per axis.
    number_of_histogram_bins : int
        Bins of the log-intensity histogram the sharpening deconvolution works on.
    wiener_filter_noise : float
        Noise level of that deconvolution.
    bias_field_fwhm : float
        Full width at half maximum, in log-intensity units, of the assumed field kernel.
    mask_label : int
        Label of the mask voxels the estimate is computed over.
    clip_negative : bool
        Clip the corrected volume at zero. The division is by a strictly positive field, so
        this only guards against a degenerate field; it is kept because the contract asks
        for it.
    """

    shrink_factor: int = 4
    max_iterations: tuple[int, ...] = (50, 50, 50, 50)
    convergence_threshold: float = 0.001
    spline_order: int = 3
    number_of_control_points: tuple[int, int, int] = (4, 4, 4)
    number_of_histogram_bins: int = 200
    wiener_filter_noise: float = 0.01
    bias_field_fwhm: float = 0.15
    mask_label: int = 1
    clip_negative: bool = True

    def to_json(self) -> dict[str, object]:
        """Return a JSON-serialisable dict of every field.

        Returns
        -------
        dict[str, object]
            Field name to value, with tuples turned into lists.
        """
        out = {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}
        out["filter"] = "SimpleITK N4BiasFieldCorrectionImageFilter"
        out["simpleitk_version"] = str(sitk.Version.VersionString())
        out["reference"] = "Tustison et al., IEEE TMI 29(6):1310-1320, 2010"
        out["mask_rule"] = (
            "the template brain mask dilated by the registration config's mask_dilation_mm, "
            "cast to uint8 from (mask > 0.5); the same foreground the p99 intensity rule uses"
        )
        out["application_rule"] = (
            "estimate on the shrunk image and mask; GetLogBiasFieldAsImage on the "
            "full-resolution input; corrected = input / exp(log field); clip at 0"
        )
        return out


@dataclass(frozen=True, eq=False)
class N4Result:
    """Outcome of one N4 correction.

    Parameters
    ----------
    corrected : sitk.Image
        ``float32`` corrected volume on the input grid.
    log_bias_field : sitk.Image
        ``float32`` log bias field on the input grid; ``exp`` of it is what was divided out.
    diagnostics : dict[str, float]
        Log-field statistics inside the mask, the mask size, the fraction of masked voxels
        that are exactly zero in the input, and the wall time of the estimate.
    """

    corrected: sitk.Image
    log_bias_field: sitk.Image
    diagnostics: dict[str, float] = field(default_factory=dict)


def simpleitk_n4_defaults() -> dict[str, object]:
    """Return the defaults of an untouched ``N4BiasFieldCorrectionImageFilter``.

    Returns
    -------
    dict[str, object]
        The getters of a freshly constructed filter, keyed by the corresponding
        :class:`N4Config` field name.
    """
    filt = sitk.N4BiasFieldCorrectionImageFilter()
    return {
        "max_iterations": tuple(int(v) for v in filt.GetMaximumNumberOfIterations()),
        "convergence_threshold": float(filt.GetConvergenceThreshold()),
        "spline_order": int(filt.GetSplineOrder()),
        "number_of_control_points": tuple(int(v) for v in filt.GetNumberOfControlPoints()),
        "number_of_histogram_bins": int(filt.GetNumberOfHistogramBins()),
        "wiener_filter_noise": float(filt.GetWienerFilterNoise()),
        "bias_field_fwhm": float(filt.GetBiasFieldFullWidthAtHalfMaximum()),
        "mask_label": int(filt.GetMaskLabel()),
    }


def binary_mask(mask: sitk.Image, label: int = 1) -> sitk.Image:
    """Cast any mask to a ``uint8`` image that is ``label`` inside and 0 outside.

    ``sitk.BinaryThreshold`` with an upper bound that the input pixel type cannot express
    silently inverts a ``uint8`` mask (T1.1 log §2 and §6 measured 7.4 M voxels instead of
    2.8 M), so the threshold is written as a comparison on a float cast instead.

    Parameters
    ----------
    mask : sitk.Image
        Any mask image, non-zero inside.
    label : int
        Value written inside the mask.

    Returns
    -------
    sitk.Image
        ``uint8`` mask carrying ``label`` inside and 0 outside.
    """
    binary = sitk.Cast(sitk.Cast(mask, sitk.sitkFloat32) > 0.5, sitk.sitkUInt8)
    return binary if label == 1 else sitk.Cast(binary * int(label), sitk.sitkUInt8)


def _shrink(image: sitk.Image, factor: int) -> sitk.Image:
    """Shrink an image isotropically by an integer factor."""
    return sitk.Shrink(image, [int(factor)] * image.GetDimension())


def _log_field_statistics(
    log_field: sitk.Image, mask: sitk.Image, volume: sitk.Image
) -> dict[str, float]:
    """Return the log-field range inside the mask plus the mask's degenerate-voxel share."""
    inside = sitk.GetArrayViewFromImage(mask) > 0
    field_values = sitk.GetArrayViewFromImage(log_field)[inside]
    volume_values = sitk.GetArrayViewFromImage(volume)[inside]
    return {
        "log_field_min": float(field_values.min()),
        "log_field_max": float(field_values.max()),
        "log_field_mean": float(field_values.mean()),
        "log_field_std": float(field_values.std()),
        "field_ratio_max_over_min": float(np.exp(field_values.max() - field_values.min())),
        "mask_voxels": float(inside.sum()),
        "mask_zero_fraction": float((volume_values <= 0.0).mean()),
    }


def n4_correct(
    volume: sitk.Image, mask: sitk.Image, cfg: N4Config | None = None
) -> N4Result:
    """Remove the multiplicative bias field of one registered volume.

    Parameters
    ----------
    volume : sitk.Image
        The registered volume, on the template grid.
    mask : sitk.Image
        Foreground mask on the same grid (the dilated template brain mask).
    cfg : N4Config | None
        Parameters; the default configuration when ``None``.

    Returns
    -------
    N4Result
        The corrected volume, the full-resolution log bias field and the diagnostics.

    Raises
    ------
    PreprocessError
        If the mask does not share the volume's grid, if the mask is empty, or if the
        filter or the field evaluation fails or produces non-finite values.
    """
    cfg = cfg or N4Config()
    image = sitk.Cast(volume, sitk.sitkFloat32)
    binary = binary_mask(mask, cfg.mask_label)
    if binary.GetSize() != image.GetSize():
        raise PreprocessError(
            f"mask size {binary.GetSize()} differs from the volume {image.GetSize()}"
        )
    if not np.allclose(binary.GetSpacing(), image.GetSpacing(), atol=1e-4):
        raise PreprocessError("mask spacing differs from the volume spacing")
    # The mask comes from the template and the volume from the registered cache, both on the
    # template lattice; a float round-trip can leave origins differing in the last bit, which
    # the filter rejects outright.
    binary.CopyInformation(image)
    if int(sitk.GetArrayViewFromImage(binary).sum()) == 0:
        raise PreprocessError("N4 mask is empty")

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(list(cfg.max_iterations))
    corrector.SetConvergenceThreshold(float(cfg.convergence_threshold))
    corrector.SetSplineOrder(int(cfg.spline_order))
    corrector.SetNumberOfControlPoints(list(cfg.number_of_control_points))
    corrector.SetNumberOfHistogramBins(int(cfg.number_of_histogram_bins))
    corrector.SetWienerFilterNoise(float(cfg.wiener_filter_noise))
    corrector.SetBiasFieldFullWidthAtHalfMaximum(float(cfg.bias_field_fwhm))
    corrector.SetMaskLabel(int(cfg.mask_label))

    started = time.time()
    try:
        corrector.Execute(_shrink(image, cfg.shrink_factor), _shrink(binary, cfg.shrink_factor))
        log_field = corrector.GetLogBiasFieldAsImage(image)
    except RuntimeError as exc:
        raise PreprocessError(f"N4 bias-field correction failed: {exc}") from exc
    elapsed = time.time() - started

    if not np.all(np.isfinite(sitk.GetArrayViewFromImage(log_field))):
        raise PreprocessError("the estimated log bias field contains non-finite values")

    corrected = sitk.Divide(image, sitk.Exp(log_field))
    if cfg.clip_negative:
        corrected = sitk.Clamp(corrected, sitk.sitkFloat32, 0.0, float(np.finfo(np.float32).max))
    corrected = sitk.Cast(corrected, sitk.sitkFloat32)
    if not np.all(np.isfinite(sitk.GetArrayViewFromImage(corrected))):
        raise PreprocessError("the corrected volume contains non-finite values")

    diagnostics = _log_field_statistics(log_field, binary, image)
    diagnostics["seconds"] = float(elapsed)
    return N4Result(corrected=corrected, log_bias_field=log_field, diagnostics=diagnostics)


def sidecar_path(path: Path) -> Path:
    """Return the sidecar JSON path of a cached corrected volume.

    Parameters
    ----------
    path : Path
        The ``<subject>.nii.gz`` cache path.

    Returns
    -------
    Path
        The ``<subject>.json`` path next to it.
    """
    path = Path(path)
    return path.with_name(path.name.replace(".nii.gz", ".json"))


def write_corrected(
    path: Path,
    result: N4Result,
    cfg: N4Config,
    subject: str,
    source: str,
    extra: dict[str, object] | None = None,
) -> None:
    """Write a corrected volume and its sidecar JSON to the cache.

    Parameters
    ----------
    path : Path
        Destination ``.nii.gz``; the sidecar takes the same stem with ``.json``.
    result : N4Result
        The correction to store.
    cfg : N4Config
        The configuration used, stored for provenance.
    subject : str
        Subject identifier.
    source : str
        Raw file path relative to the cohort's raw root.
    extra : dict[str, object] | None
        Extra fields merged into the sidecar (the site label, the registered source path).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.nii.gz")
    sitk.WriteImage(result.corrected, str(tmp), useCompression=True)
    tmp.replace(path)
    sidecar = {
        "subject": subject,
        "source": source,
        "n4": cfg.to_json(),
        **{k: float(v) for k, v in result.diagnostics.items()},
        **(extra or {}),
    }
    sidecar_path(path).write_text(json.dumps(sidecar, indent=2))


def read_n4_sidecar(path: Path) -> dict[str, object]:
    """Read the sidecar JSON of a cached corrected volume.

    Parameters
    ----------
    path : Path
        The ``<subject>.nii.gz`` cache path (not the JSON itself).

    Returns
    -------
    dict[str, object]
        The parsed sidecar.

    Raises
    ------
    PreprocessError
        If the sidecar is missing or unparsable.
    """
    json_path = sidecar_path(path)
    try:
        return json.loads(json_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PreprocessError(f"cannot read N4 sidecar {json_path}: {exc}") from exc
