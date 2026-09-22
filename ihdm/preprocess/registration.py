"""Rigid (6 dof) registration of a raw T1 volume into the MNI152NLin2009cAsym 1 mm grid.

The pipeline interpolates each raw acquisition exactly once: the Euler3D transform found by
the multi-resolution Mattes mutual-information registration is composed with the change of
grid inside a single ``sitk.Resample`` call from the native acquisition lattice onto the
template lattice. No scaling, shearing or non-linear warp is applied, so head size is
preserved and only position and orientation are standardised.

The metric is evaluated only inside the template brain mask dilated by 10 mm, which keeps
the criterion on the brain and the immediately surrounding skull and scalp instead of on
the neck and the variable amount of shoulder each acquisition includes. The mask is never
applied to the image itself.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from ihdm.preprocess.errors import PreprocessError

__all__ = [
    "homogeneous_matrix",
    "matrix_discrepancy",
    "RegistrationConfig",
    "RegistrationResult",
    "Template",
    "dilate_mask_mm",
    "load_template",
    "register",
    "resample_to_template",
    "read_sidecar",
    "write_registered",
]

logger = logging.getLogger(__name__)

TEMPLATE_STEM = "tpl-MNI152NLin2009cAsym_res-01"


@dataclass(frozen=True)
class RegistrationConfig:
    """Every parameter of the rigid registration and of the single resampling.

    Parameters
    ----------
    histogram_bins : int
        Bins of the Mattes mutual-information estimator.
    sampling_percentage : float
        Fraction of the fixed-image voxels drawn as metric samples at each level.
    sampling_seed : int
        Seed of the metric sampler. Fixed, so the objective is deterministic and the
        whole pipeline is reproducible.
    shrink_factors : tuple[int, ...]
        Downsampling factor of the fixed image per resolution level.
    smoothing_sigmas_mm : tuple[float, ...]
        Gaussian smoothing per resolution level, in millimetres.
    learning_rate : float
        Initial step of the regular-step gradient descent. With
        ``SetOptimizerScalesFromPhysicalShift`` the step is a physical displacement, so
        1.0 means one millimetre.
    min_step : float
        Step below which the optimiser declares convergence, in millimetres.
    max_iterations : int
        Iteration cap per resolution level.
    relaxation_factor : float
        Factor the step is multiplied by whenever the gradient direction reverses.
    gradient_tolerance : float
        Gradient-magnitude convergence tolerance.
    mask_dilation_mm : float
        Dilation of the template brain mask, used both as the metric region and as the
        foreground of the intensity rule.
    interpolator_metric : str
        Interpolator used *inside* the optimisation (linear: cheap and smooth).
    interpolator_resample : str
        Interpolator of the one and only resampling (B-spline order 3).
    default_value : float
        Value written outside the moving image's field of view.
    clip_negative : bool
        Clip the B-spline overshoot below zero to zero after resampling.
    optimiser : str
        Name of the optimiser, recorded in ``meta.json``.
    """

    histogram_bins: int = 50
    sampling_percentage: float = 0.20
    sampling_seed: int = 2026
    shrink_factors: tuple[int, ...] = (4, 2, 1)
    smoothing_sigmas_mm: tuple[float, ...] = (2.0, 1.0, 0.0)
    learning_rate: float = 1.0
    min_step: float = 1e-4
    max_iterations: int = 300
    relaxation_factor: float = 0.5
    gradient_tolerance: float = 1e-8
    mask_dilation_mm: float = 10.0
    interpolator_metric: str = "linear"
    interpolator_resample: str = "bspline3"
    default_value: float = 0.0
    clip_negative: bool = True
    optimiser: str = "RegularStepGradientDescent"

    def to_json(self) -> dict[str, object]:
        """Return a JSON-serialisable dict of every field.

        Returns
        -------
        dict[str, object]
            Field name to value, with tuples turned into lists.
        """
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}


@dataclass(frozen=True, eq=False)
class RegistrationResult:
    """Outcome of one rigid registration.

    Parameters
    ----------
    transform : sitk.Transform
        The Euler3D transform mapping template (fixed) points to subject (moving) points.
    final_metric : float
        Metric value at convergence. SimpleITK minimises the negative mutual information,
        so a more negative value is a better alignment.
    iterations : int
        Optimiser iterations used at the finest resolution level.
    stop_condition : str
        The optimiser's stop-condition description at the finest level.
    hit_max_iterations : bool
        True when the finest level exhausted ``RegistrationConfig.max_iterations``.
    """

    transform: sitk.Transform
    final_metric: float
    iterations: int
    stop_condition: str
    hit_max_iterations: bool = False

    def parameters(self) -> list[float]:
        """Return the transform parameters (three rotations in rad, three shifts in mm).

        Returns
        -------
        list[float]
            The six Euler3D parameters.
        """
        return [float(p) for p in self.transform.GetParameters()]

    def fixed_parameters(self) -> list[float]:
        """Return the transform's fixed parameters (the centre of rotation, in mm).

        Returns
        -------
        list[float]
            The centre of rotation in LPS millimetres.
        """
        return [float(p) for p in self.transform.GetFixedParameters()]


@dataclass(frozen=True, eq=False)
class Template:
    """The MNI152 template, its brain mask and the dilated metric/foreground mask.

    Parameters
    ----------
    image : sitk.Image
        ``float32`` T1w template on the 1 mm grid.
    brain_mask : sitk.Image
        ``uint8`` binary brain mask on the same grid.
    dilated_mask : sitk.Image
        ``uint8`` brain mask dilated by ``RegistrationConfig.mask_dilation_mm``.
    path : Path
        Path of the T1w template file, recorded in ``meta.json``.
    mask_path : Path
        Path of the brain-mask file.
    dilation_mm : float
        The dilation actually applied.
    """

    image: sitk.Image
    brain_mask: sitk.Image
    dilated_mask: sitk.Image
    path: Path
    mask_path: Path
    dilation_mm: float

    foreground: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))


def dilate_mask_mm(mask: sitk.Image, millimetres: float) -> sitk.Image:
    """Dilate a binary mask by an exact physical distance.

    Uses the signed Euclidean distance map rather than a discrete ball structuring
    element: the threshold is exact for any voxel spacing and costs O(N) instead of
    O(N r^3).

    Parameters
    ----------
    mask : sitk.Image
        Binary mask (non-zero inside).
    millimetres : float
        Dilation radius in millimetres; ``0`` returns the mask unchanged.

    Returns
    -------
    sitk.Image
        ``uint8`` mask, non-zero within ``millimetres`` of the input mask.
    """
    # `mask > 0.5` rather than BinaryThreshold: an explicit upper bound has to be
    # expressible in the input pixel type, and a uint8 mask silently inverts when it is
    # not (measured on the template mask: 6.6 M voxels instead of 1.9 M).
    binary = sitk.Cast(sitk.Cast(mask, sitk.sitkFloat32) > 0.5, sitk.sitkUInt8)
    if millimetres <= 0:
        return binary
    distance = sitk.SignedMaurerDistanceMap(
        binary, insideIsPositive=False, squaredDistance=False, useImageSpacing=True
    )
    return sitk.Cast(distance <= float(millimetres), sitk.sitkUInt8)


def load_template(template_dir: Path, cfg: RegistrationConfig | None = None) -> Template:
    """Read the MNI152NLin2009cAsym 1 mm template and build the dilated metric mask.

    Parameters
    ----------
    template_dir : Path
        Directory holding ``tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz`` and
        ``tpl-MNI152NLin2009cAsym_res-01_desc-brain_mask.nii.gz``.
    cfg : RegistrationConfig | None
        Supplies ``mask_dilation_mm``; the default configuration is used when ``None``.

    Returns
    -------
    Template
        Template image, brain mask, dilated mask and the boolean foreground array in
        ``(i, j, k)`` index order.

    Raises
    ------
    PreprocessError
        If either file is missing, or the mask grid differs from the image grid.
    """
    cfg = cfg or RegistrationConfig()
    template_dir = Path(template_dir)
    image_path = template_dir / f"{TEMPLATE_STEM}_T1w.nii.gz"
    mask_path = template_dir / f"{TEMPLATE_STEM}_desc-brain_mask.nii.gz"
    for path in (image_path, mask_path):
        if not path.is_file():
            raise PreprocessError(f"template file not found: {path}")

    image = sitk.ReadImage(str(image_path), sitk.sitkFloat32)
    brain_mask = sitk.Cast(sitk.ReadImage(str(mask_path)), sitk.sitkUInt8)
    if brain_mask.GetSize() != image.GetSize():
        raise PreprocessError(
            f"template mask size {brain_mask.GetSize()} differs from image {image.GetSize()}"
        )
    if not np.allclose(brain_mask.GetOrigin(), image.GetOrigin(), atol=1e-4):
        raise PreprocessError("template mask origin differs from the image origin")

    dilated = dilate_mask_mm(brain_mask, cfg.mask_dilation_mm)
    foreground = sitk.GetArrayFromImage(dilated).transpose(2, 1, 0).astype(bool)
    return Template(
        image=image,
        brain_mask=brain_mask,
        dilated_mask=dilated,
        path=image_path,
        mask_path=mask_path,
        dilation_mm=cfg.mask_dilation_mm,
        foreground=foreground,
    )


def register(
    moving: sitk.Image,
    template: sitk.Image,
    mask: sitk.Image | None,
    cfg: RegistrationConfig,
) -> RegistrationResult:
    """Register a subject volume rigidly onto the template.

    Parameters
    ----------
    moving : sitk.Image
        The subject volume on its native acquisition grid.
    template : sitk.Image
        The fixed image (the MNI152 template).
    mask : sitk.Image | None
        Fixed-image mask restricting the metric region; ``None`` uses the whole template.
    cfg : RegistrationConfig
        Registration parameters.

    Returns
    -------
    RegistrationResult
        The transform mapping template points to subject points, plus the diagnostics of
        the quality gate.

    Raises
    ------
    PreprocessError
        If SimpleITK fails to optimise the transform.
    """
    fixed = sitk.Cast(template, sitk.sitkFloat32)
    moving = sitk.Cast(moving, sitk.sitkFloat32)

    try:
        initial = sitk.CenteredTransformInitializer(
            fixed, moving, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS
        )
    except RuntimeError as exc:
        # The moments initialiser divides by the total image mass, so an empty or constant
        # volume (an unreadable or truncated acquisition) aborts here rather than later.
        raise PreprocessError(f"moments initialiser failed: {exc}") from exc
    # Optimise a true Euler3DTransform in place: `inPlace=False` makes Execute return a
    # CompositeTransform, which has no GetMatrix/GetCenter and cannot be written to the
    # sidecar or composed in the tests.
    optimised = sitk.Euler3DTransform(initial)

    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=cfg.histogram_bins)
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(cfg.sampling_percentage, cfg.sampling_seed)
    if mask is not None:
        method.SetMetricFixedMask(sitk.Cast(mask, sitk.sitkUInt8))
    method.SetInterpolator(_INTERPOLATORS[cfg.interpolator_metric])
    method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=cfg.learning_rate,
        minStep=cfg.min_step,
        numberOfIterations=cfg.max_iterations,
        relaxationFactor=cfg.relaxation_factor,
        gradientMagnitudeTolerance=cfg.gradient_tolerance,
    )
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel(list(cfg.shrink_factors))
    method.SetSmoothingSigmasPerLevel(list(cfg.smoothing_sigmas_mm))
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    method.SetInitialTransform(optimised, inPlace=True)

    try:
        method.Execute(fixed, moving)
    except RuntimeError as exc:
        raise PreprocessError(f"rigid registration failed: {exc}") from exc
    transform = optimised

    iterations = int(method.GetOptimizerIteration())
    stop_condition = str(method.GetOptimizerStopConditionDescription())
    return RegistrationResult(
        transform=transform,
        final_metric=float(method.GetMetricValue()),
        iterations=iterations,
        stop_condition=stop_condition,
        hit_max_iterations=iterations >= cfg.max_iterations,
    )


def resample_to_template(
    moving: sitk.Image,
    template: sitk.Image,
    transform: sitk.Transform,
    cfg: RegistrationConfig | None = None,
) -> sitk.Image:
    """Resample a subject volume onto the template grid, exactly once.

    Parameters
    ----------
    moving : sitk.Image
        The subject volume on its native acquisition grid.
    template : sitk.Image
        Defines the output grid (size, spacing, origin, direction).
    transform : sitk.Transform
        Maps template points to subject points.
    cfg : RegistrationConfig | None
        Supplies the interpolator, the default value and the negative-clipping rule.

    Returns
    -------
    sitk.Image
        ``float32`` image on the template grid.
    """
    cfg = cfg or RegistrationConfig()
    resampled = sitk.Resample(
        sitk.Cast(moving, sitk.sitkFloat32),
        template,
        transform,
        _INTERPOLATORS[cfg.interpolator_resample],
        float(cfg.default_value),
        sitk.sitkFloat32,
    )
    if cfg.clip_negative:
        # B-spline interpolation overshoots at the sharp scalp/air edge; the acquisition
        # itself has no negative intensities, so the overshoot is clipped away.
        resampled = sitk.Clamp(resampled, sitk.sitkFloat32, 0.0, float(np.finfo(np.float32).max))
    return resampled


def write_registered(
    path: Path,
    image: sitk.Image,
    result: RegistrationResult,
    cfg: RegistrationConfig,
    subject: str,
    source: str,
) -> None:
    """Write a registered volume and its sidecar JSON to the cache.

    Parameters
    ----------
    path : Path
        Destination ``.nii.gz``; the sidecar takes the same stem with ``.json``.
    image : sitk.Image
        The registered volume.
    result : RegistrationResult
        Diagnostics of the registration that produced ``image``.
    cfg : RegistrationConfig
        The configuration used, stored for provenance.
    subject : str
        Subject identifier.
    source : str
        Raw file path relative to the cohort's raw root.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.nii.gz")
    sitk.WriteImage(image, str(tmp), useCompression=True)
    tmp.replace(path)
    sidecar = {
        "subject": subject,
        "source": source,
        "final_metric": result.final_metric,
        "iterations": result.iterations,
        "stop_condition": result.stop_condition,
        "hit_max_iterations": result.hit_max_iterations,
        "transform_parameters": result.parameters(),
        "transform_fixed_parameters": result.fixed_parameters(),
        "config": cfg.to_json(),
    }
    sidecar_path(path).write_text(json.dumps(sidecar, indent=2))


def sidecar_path(path: Path) -> Path:
    """Return the sidecar JSON path of a cached registered volume.

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


def read_sidecar(path: Path) -> dict[str, object]:
    """Read the sidecar JSON of a cached registered volume.

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
        raise PreprocessError(f"cannot read registration sidecar {json_path}: {exc}") from exc


def homogeneous_matrix(transform: sitk.Euler3DTransform) -> np.ndarray:
    """Return the 4x4 homogeneous matrix of a centred rigid transform.

    A SimpleITK centred transform maps ``x -> R (x - c) + c + t``, so the equivalent
    homogeneous matrix has rotation ``R`` and translation ``c + t - R c``. Composing
    transforms through these matrices is exact and independent of the centres, which is
    what the registration-accuracy test needs.

    Parameters
    ----------
    transform : sitk.Euler3DTransform
        A centred rigid transform.

    Returns
    -------
    np.ndarray
        The 4x4 homogeneous matrix.
    """
    rotation = np.array(transform.GetMatrix(), dtype=float).reshape(3, 3)
    centre = np.array(transform.GetCenter(), dtype=float)
    translation = np.array(transform.GetTranslation(), dtype=float)
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = centre + translation - rotation @ centre
    return matrix


def matrix_discrepancy(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    """Return the rotation angle and translation norm of ``first^-1 @ second``.

    Parameters
    ----------
    first, second : np.ndarray
        Two 4x4 homogeneous rigid matrices.

    Returns
    -------
    tuple[float, float]
        Rotation discrepancy in degrees and translation discrepancy in millimetres.
    """
    delta = np.linalg.inv(np.asarray(first, dtype=float)) @ np.asarray(second, dtype=float)
    cosine = (np.trace(delta[:3, :3]) - 1.0) / 2.0
    angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    return angle, float(np.linalg.norm(delta[:3, 3]))


_INTERPOLATORS: dict[str, int] = {
    "nearest": sitk.sitkNearestNeighbor,
    "linear": sitk.sitkLinear,
    "bspline3": sitk.sitkBSpline,
}
