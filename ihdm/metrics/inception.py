"""Inception metrics of ``05-metrics.md`` §7: KID (headline), FID, recall and coverage.

Everything here works **in memory**. ``clean-fid``'s own entry points want a folder of PNGs; a
5 000-image folder per run would cost the evaluation array 150 000 files against Picasso's
FSCRATCH quota (H-PICASSO §1 leaves ≈ 30 000 files of headroom), so this module drives the same
feature extractor and the same distances over numpy stacks instead, and never writes an image.

The pipeline is clean-fid's: grayscale replicated to three channels, PIL bicubic resize to
``299²`` without an intermediate quantisation (``build_resizer("clean")``), the torchscript
Inception of Szegedy et al. used by StyleGAN's metric suite, 2048-dimensional pool features.

Two rules of ``05-metrics.md`` §7 are encoded in the result type rather than left to the caller:
**KID is the headline** (it is unbiased; FID against an 800-image reference is biased upward by
a term of order ``1/N_ref``), and **every FID carries its ``n_reference``**, so that no absolute
value is quoted against the paper's without the caveat.
"""

from __future__ import annotations

import contextlib
import json
import logging
import platform
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ihdm.metrics.errors import MetricError

__all__ = [
    "INCEPTION_FEATURE_DIM",
    "InceptionResult",
    "bootstrap_inception",
    "fid_from_features",
    "inception_features",
    "inception_weights_path",
    "kid_from_features",
    "recall_coverage",
    "reference_features",
]

logger = logging.getLogger(__name__)

#: Dimension of the pool features of the torchscript Inception clean-fid uses.
INCEPTION_FEATURE_DIM: int = 2048

#: Side of the square the extractor is fed.
INCEPTION_INPUT_SIZE: int = 299

#: File name of the weights clean-fid downloads on first use.
INCEPTION_WEIGHTS_NAME: str = "inception-2015-12-05.pt"

#: Neighbour count of the recall/coverage estimator (``05-metrics.md`` §7).
DEFAULT_K: int = 5

#: Resamples of the FID/KID bootstrap over samples.
DEFAULT_N_BOOT: int = 200

#: Subsets drawn per KID evaluation. The frozen point estimate uses clean-fid's 100; the
#: bootstrap replicates use fewer, because the resampling is already the variance estimate and
#: 100 subsets per replicate would cost 13 minutes per run for no extra information.
KID_SUBSETS: int = 100
KID_SUBSETS_BOOTSTRAP: int = 10

#: File names of the per-dataset reference feature cache.
REFERENCE_FEATURES_NAME: str = "_features_inception_ref.npy"
REFERENCE_SIDECAR_NAME: str = "_features_inception_ref.json"

_EXTRACTORS: dict[tuple[str, str], Any] = {}


# --------------------------------------------------------------------------------------------
# The extractor
# --------------------------------------------------------------------------------------------


def inception_weights_path() -> Path:
    """Return the file clean-fid downloads the Inception weights to.

    ``cleanfid.features.feature_extractor`` hard-codes this directory (``/tmp`` on Linux, ``./``
    on Windows) and passes ``download=True``. The evaluation array of T5.1 must therefore either
    pre-seed this path on every compute node or run once on a node with outbound network access;
    the CLI prints the path for that reason.

    Returns
    -------
    Path
        The absolute path of the weights file, whether or not it exists yet.
    """
    directory = Path("./") if platform.system() == "Windows" else Path("/tmp")
    return (directory / INCEPTION_WEIGHTS_NAME).resolve()


def build_extractor(device: str = "cuda", mode: str = "clean") -> Any:
    """Return clean-fid's feature extractor, built once per ``(mode, device)``.

    Parameters
    ----------
    device : str
        Torch device string.
    mode : str
        clean-fid mode; ``"clean"`` is the one ``05-metrics.md`` §7 names.

    Returns
    -------
    Callable
        A function mapping a ``(B, 3, 299, 299)`` tensor in ``[0, 255]`` to ``(B, 2048)``.

    Raises
    ------
    MetricError
        If ``clean-fid`` is not installed or the weights cannot be obtained.
    """
    key = (mode, str(device))
    if key in _EXTRACTORS:
        return _EXTRACTORS[key]
    try:
        import torch
        from cleanfid.features import build_feature_extractor
    except ImportError as error:  # pragma: no cover - environment problem, not logic
        raise MetricError(f"clean-fid is not available: {error}") from error
    try:
        extractor = build_feature_extractor(
            mode, device=torch.device(device), use_dataparallel=False
        )
    except Exception as error:
        raise MetricError(
            f"could not build the clean-fid extractor (weights at {inception_weights_path()}): "
            f"{error}"
        ) from error
    logger.info("inception weights: %s", inception_weights_path())
    _EXTRACTORS[key] = extractor
    return extractor


def _as_u8_stack(images: Any) -> np.ndarray:
    """Return ``images`` as a ``(N, H, W)`` ``uint8`` stack."""
    stack = np.asarray(images)
    if stack.ndim == 4 and stack.shape[-1] == 1:
        stack = stack[..., 0]
    if stack.ndim != 3:
        raise MetricError(f"expected a stack of shape (N, H, W), got {stack.shape}")
    if stack.shape[0] == 0:
        raise MetricError("expected at least one image")
    if np.issubdtype(stack.dtype, np.floating):
        if not np.all(np.isfinite(stack)):
            raise MetricError("the image stack holds a non-finite value")
        stack = np.rint(np.clip(stack, 0.0, 1.0) * 255.0)
    return np.ascontiguousarray(stack.astype(np.uint8))


def inception_features(
    images_u8: Any, device: str = "cuda", batch: int = 64, mode: str = "clean"
) -> np.ndarray:
    """Inception pool features of a grayscale image stack, computed in memory.

    Each image is resized once with clean-fid's ``"clean"`` resizer and the result is replicated
    to three channels. Replicating before or after the resize is identical (the resizer maps each
    channel independently with the same filter), and resizing once is three times cheaper.

    Parameters
    ----------
    images_u8 : array-like
        ``(N, H, W)`` ``uint8``, or float in ``[0, 1]``, which is quantised to ``uint8`` first
        because that is what the samples on disk are.
    device : str
        Torch device the extractor runs on.
    batch : int
        Images per forward pass.
    mode : str
        clean-fid mode.

    Returns
    -------
    numpy.ndarray
        ``(N, 2048)`` ``float64``.

    Raises
    ------
    MetricError
        If the stack is malformed or non-finite, ``batch`` is not positive, or the extractor
        cannot be built.
    """
    import torch
    from cleanfid.fid import get_batch_features
    from cleanfid.resize import build_resizer

    stack = _as_u8_stack(images_u8)
    if batch < 1:
        raise MetricError(f"batch must be positive, got {batch}")

    extractor = build_extractor(device, mode)
    resize = build_resizer(mode)
    size = INCEPTION_INPUT_SIZE
    out = np.empty((stack.shape[0], INCEPTION_FEATURE_DIM), dtype=np.float64)

    for start in range(0, stack.shape[0], batch):
        block = stack[start : start + batch]
        resized = np.empty((block.shape[0], 3, size, size), dtype=np.float32)
        for row, image in enumerate(block):
            single = resize(np.repeat(image[:, :, None], 3, axis=2))[:, :, 0]
            resized[row] = np.repeat(single[None], 3, axis=0)
        tensor = torch.from_numpy(resized)
        features = get_batch_features(tensor, extractor, device)
        out[start : start + block.shape[0]] = np.asarray(features, dtype=np.float64)
    return out


# --------------------------------------------------------------------------------------------
# Distances
# --------------------------------------------------------------------------------------------


def _as_features(features: Any, name: str, minimum: int = 2) -> np.ndarray:
    """Return a validated ``(N, D)`` ``float64`` feature matrix."""
    array = np.asarray(features, dtype=np.float64)
    if array.ndim != 2:
        raise MetricError(f"{name}: expected (N, D) features, got {array.shape}")
    if array.shape[0] < minimum:
        raise MetricError(f"{name}: expected at least {minimum} rows, got {array.shape[0]}")
    if not np.all(np.isfinite(array)):
        raise MetricError(f"{name}: holds a non-finite value")
    return array


def fid_from_features(f_samples: Any, f_reference: Any) -> float:
    """Frechet Inception distance between two feature sets (clean-fid's estimator).

    Parameters
    ----------
    f_samples, f_reference : array-like
        ``(N, D)`` and ``(M, D)`` feature matrices.

    Returns
    -------
    float
        The FID. It is biased upward by a term of order ``1/M``; the bias cancels in differences
        between arms because every arm of a dataset uses the same reference, but the absolute
        value is not comparable to a value computed against 50 000 references.

    Raises
    ------
    MetricError
        If either matrix is malformed, or the two dimensions differ.
    """
    from cleanfid.fid import frechet_distance

    samples = _as_features(f_samples, "f_samples")
    reference = _as_features(f_reference, "f_reference")
    if samples.shape[1] != reference.shape[1]:
        raise MetricError(
            f"feature dimension mismatch: {samples.shape[1]} against {reference.shape[1]}"
        )
    return float(
        frechet_distance(
            samples.mean(axis=0),
            np.cov(samples, rowvar=False),
            reference.mean(axis=0),
            np.cov(reference, rowvar=False),
        )
    )


@dataclass(frozen=True)
class _ReferenceMoments:
    """Precomputed pieces of the reference side of the FID, reused by every bootstrap replicate.

    Parameters
    ----------
    mean : numpy.ndarray
        Reference feature mean, ``(D,)``.
    trace : float
        ``tr(Sigma_ref)``.
    root : numpy.ndarray
        The symmetric positive square root of ``Sigma_ref``, ``(D, D)``.
    """

    mean: np.ndarray
    trace: float
    root: np.ndarray


def _reference_moments(reference: np.ndarray) -> _ReferenceMoments:
    """Return the mean, the trace and the matrix square root of the reference covariance."""
    covariance = np.cov(reference, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    values = np.clip(values, 0.0, None)
    root = (vectors * np.sqrt(values)) @ vectors.T
    return _ReferenceMoments(
        mean=reference.mean(axis=0), trace=float(np.trace(covariance)), root=root
    )


def _fid_fast(samples: np.ndarray, moments: _ReferenceMoments) -> float:
    """FID of ``samples`` against precomputed reference moments, without a dense ``sqrtm``.

    ``tr sqrt(Sigma_s Sigma_r)`` is the sum of the square roots of the eigenvalues of
    ``Sigma_s Sigma_r``, which are the eigenvalues of the Gram matrix of
    ``Z = X_c Sigma_r^{1/2} / sqrt(n - 1)``; the Gram matrix is ``n x n`` rather than
    ``2048 x 2048`` whenever the sample set is smaller than the feature dimension, and a
    symmetric eigenvalue problem either way. This is algebraically the same number that
    ``scipy.linalg.sqrtm`` returns, at a fraction of the cost, which is what makes a
    200-resample bootstrap affordable.

    Parameters
    ----------
    samples : numpy.ndarray
        ``(n, D)`` sample features.
    moments : _ReferenceMoments
        The reference side.

    Returns
    -------
    float
        The FID.
    """
    n_samples = samples.shape[0]
    mean = samples.mean(axis=0)
    centred = samples - mean
    trace_samples = float(np.square(centred).sum() / (n_samples - 1))
    projected = centred @ moments.root / np.sqrt(n_samples - 1)
    gram = (
        projected @ projected.T
        if n_samples <= projected.shape[1]
        else projected.T @ projected
    )
    eigenvalues = np.clip(np.linalg.eigvalsh((gram + gram.T) / 2.0), 0.0, None)
    difference = mean - moments.mean
    return float(
        np.square(difference).sum()
        + trace_samples
        + moments.trace
        - 2.0 * np.sqrt(eigenvalues).sum()
    )


@contextlib.contextmanager
def _legacy_rng(seed: int) -> Iterator[None]:
    """Seed the legacy global numpy RNG for the duration of the block, then restore it.

    ``cleanfid.fid.kernel_distance`` draws its subsets with the global ``numpy.random``, so two
    calls on the same features return two numbers. Seeding around it makes KID reproducible
    without reimplementing the estimator.
    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def kid_from_features(
    f_samples: Any,
    f_reference: Any,
    num_subsets: int = KID_SUBSETS,
    max_subset_size: int = 1000,
    rng_seed: int = 0,
) -> float:
    """Kernel Inception distance between two feature sets (clean-fid's estimator).

    KID is the headline Inception metric of ``05-metrics.md`` §7: it is an unbiased estimator of
    the polynomial-kernel MMD, so unlike FID it is not inflated by the 800-image reference.

    Parameters
    ----------
    f_samples, f_reference : array-like
        ``(N, D)`` and ``(M, D)`` feature matrices.
    num_subsets : int
        Subsets the estimator averages over.
    max_subset_size : int
        Largest subset drawn.
    rng_seed : int
        Seed applied to the legacy global RNG clean-fid draws from.

    Returns
    -------
    float
        The KID.

    Raises
    ------
    MetricError
        If either matrix is malformed, or the two dimensions differ.
    """
    from cleanfid.fid import kernel_distance

    samples = _as_features(f_samples, "f_samples")
    reference = _as_features(f_reference, "f_reference")
    if samples.shape[1] != reference.shape[1]:
        raise MetricError(
            f"feature dimension mismatch: {samples.shape[1]} against {reference.shape[1]}"
        )
    with _legacy_rng(rng_seed):
        return float(
            kernel_distance(
                samples, reference, num_subsets=num_subsets, max_subset_size=max_subset_size
            )
        )


def recall_coverage(f_samples: Any, f_reference: Any, k: int = DEFAULT_K) -> dict[str, float]:
    """Precision, recall, density and coverage on Inception features (``prdc``, ``k = 5``).

    Parameters
    ----------
    f_samples, f_reference : array-like
        ``(N, D)`` and ``(M, D)`` feature matrices.
    k : int
        Neighbour count of the manifold estimator.

    Returns
    -------
    dict[str, float]
        ``precision``, ``recall``, ``density``, ``coverage`` and ``k``.

    Raises
    ------
    MetricError
        If ``prdc`` is missing, a matrix is malformed, the dimensions differ, or either set holds
        fewer than ``k + 1`` rows.
    """
    try:
        from prdc import compute_prdc
    except ImportError as error:  # pragma: no cover - environment problem, not logic
        raise MetricError(f"prdc is not available: {error}") from error

    samples = _as_features(f_samples, "f_samples", minimum=k + 1)
    reference = _as_features(f_reference, "f_reference", minimum=k + 1)
    if samples.shape[1] != reference.shape[1]:
        raise MetricError(
            f"feature dimension mismatch: {samples.shape[1]} against {reference.shape[1]}"
        )
    values = compute_prdc(real_features=reference, fake_features=samples, nearest_k=int(k))
    out = {name: float(value) for name, value in values.items()}
    out["k"] = float(k)
    return out


# --------------------------------------------------------------------------------------------
# The bundled result
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class InceptionResult:
    """The §7 block of ``final.json``.

    Parameters
    ----------
    kid : float
        The headline metric.
    kid_ci_low, kid_ci_high : float
        Percentile bootstrap interval of KID over resampled samples.
    fid : float
        The Frechet distance, always reported beside ``n_reference``.
    fid_ci_low, fid_ci_high : float
        Percentile bootstrap interval of FID over resampled samples.
    precision, recall, density, coverage : float
        The ``prdc`` quadruple at ``k``.
    k : int
        Neighbour count.
    n_samples, n_reference : int
        Set sizes; ``n_reference`` is the number the FID bias scales with.
    n_boot : int
        Resamples behind the two intervals.
    feature_dim : int
        Dimension of the features.
    weights_path : str
        Where the Inception weights were read from, for the Picasso worker of T5.1.
    notes : dict[str, str]
        The reading rules of §7 carried with the numbers.
    """

    kid: float
    kid_ci_low: float
    kid_ci_high: float
    fid: float
    fid_ci_low: float
    fid_ci_high: float
    precision: float
    recall: float
    density: float
    coverage: float
    k: int
    n_samples: int
    n_reference: int
    n_boot: int
    feature_dim: int
    weights_path: str
    notes: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """Return the JSON record of ``05-metrics.md`` §9."""
        return {
            "kid": float(self.kid),
            "kid_ci_low": float(self.kid_ci_low),
            "kid_ci_high": float(self.kid_ci_high),
            "fid": float(self.fid),
            "fid_ci_low": float(self.fid_ci_low),
            "fid_ci_high": float(self.fid_ci_high),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "density": float(self.density),
            "coverage": float(self.coverage),
            "k": int(self.k),
            "n_samples": int(self.n_samples),
            "n_reference": int(self.n_reference),
            "n_boot": int(self.n_boot),
            "feature_dim": int(self.feature_dim),
            "weights_path": str(self.weights_path),
            "notes": dict(self.notes),
        }


def bootstrap_inception(
    f_samples: Any,
    f_reference: Any,
    k: int = DEFAULT_K,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = 0.05,
    rng_seed: int = 0,
) -> InceptionResult:
    """Compute KID, FID, recall and coverage with bootstrap intervals over samples.

    The resampling unit is the sample, not the seed: ``05-metrics.md`` §7 asks for "a bootstrap
    CI over samples". The reference set is held fixed, which is the right conditioning — every
    arm of a dataset is compared against the same 800 images, and the reference's own sampling
    error is common to all of them.

    Parameters
    ----------
    f_samples, f_reference : array-like
        ``(N, D)`` and ``(M, D)`` feature matrices.
    k : int
        Neighbour count of ``prdc``.
    n_boot : int
        Resamples.
    alpha : float
        Two-sided level.
    rng_seed : int
        Seed of the resampling RNG.

    Returns
    -------
    InceptionResult
        Every §7 quantity with its interval.

    Raises
    ------
    MetricError
        If the feature matrices are malformed or ``n_boot`` is not positive.
    """
    samples = _as_features(f_samples, "f_samples", minimum=k + 1)
    reference = _as_features(f_reference, "f_reference", minimum=k + 1)
    if n_boot < 1:
        raise MetricError(f"n_boot must be positive, got {n_boot}")

    kid = kid_from_features(samples, reference, rng_seed=rng_seed)
    fid = fid_from_features(samples, reference)
    prdc_values = recall_coverage(samples, reference, k=k)

    moments = _reference_moments(reference)
    rng = np.random.default_rng(rng_seed)
    kid_replicates = np.empty(n_boot)
    fid_replicates = np.empty(n_boot)
    for index in range(n_boot):
        rows = rng.integers(0, samples.shape[0], size=samples.shape[0])
        resample = samples[rows]
        fid_replicates[index] = _fid_fast(resample, moments)
        kid_replicates[index] = kid_from_features(
            resample,
            reference,
            num_subsets=KID_SUBSETS_BOOTSTRAP,
            rng_seed=int(rng_seed + index + 1),
        )
    percentiles = [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)]
    kid_low, kid_high = np.percentile(kid_replicates, percentiles)
    fid_low, fid_high = np.percentile(fid_replicates, percentiles)

    return InceptionResult(
        kid=kid,
        kid_ci_low=float(kid_low),
        kid_ci_high=float(kid_high),
        fid=fid,
        fid_ci_low=float(fid_low),
        fid_ci_high=float(fid_high),
        precision=prdc_values["precision"],
        recall=prdc_values["recall"],
        density=prdc_values["density"],
        coverage=prdc_values["coverage"],
        k=int(k),
        n_samples=int(samples.shape[0]),
        n_reference=int(reference.shape[0]),
        n_boot=int(n_boot),
        feature_dim=int(samples.shape[1]),
        weights_path=str(inception_weights_path()),
        notes={
            "headline": "KID is the headline of the two; it is unbiased.",
            "fid_bias": (
                f"FID against {reference.shape[0]} reference images is biased upward by a term "
                "of order 1/n_reference; the bias cancels in differences between arms of the "
                "same dataset and makes absolute values incomparable to the paper's."
            ),
            "kid_bootstrap": (
                f"the KID point estimate averages {KID_SUBSETS} subsets; each bootstrap "
                f"replicate averages {KID_SUBSETS_BOOTSTRAP}, so the interval is conservative."
            ),
            "ci_shift": (
                "the intervals are percentile intervals over resampled samples (05 section 8); "
                "a resample holds duplicates, which inflates the finite-sample bias of FID, so "
                "fid can lie below fid_ci_low. The interval measures spread, not location."
            ),
        },
    )


# --------------------------------------------------------------------------------------------
# The per-dataset reference feature cache
# --------------------------------------------------------------------------------------------


def reference_features(
    dataset_root: Path,
    images_u8: Any,
    dataset_sha256: str,
    device: str = "cuda",
    batch: int = 64,
    force: bool = False,
) -> np.ndarray:
    """Return the Inception features of a dataset's reference split, cached on disk.

    The cache is keyed by the dataset's ``sha256_images``: a dataset that is rebuilt invalidates
    it. Thirty runs of one dataset share the file, which is what makes the evaluation array's
    Inception cost a per-dataset constant rather than a per-run one.

    Parameters
    ----------
    dataset_root : Path
        The dataset directory the cache lives in.
    images_u8 : array-like
        The reference images, ``(M, H, W)`` ``uint8``.
    dataset_sha256 : str
        ``meta.json``'s ``sha256_images``, stored in the sidecar and checked on read.
    device : str
        Torch device the extractor runs on.
    batch : int
        Images per forward pass.
    force : bool
        Recompute even when a valid cache exists.

    Returns
    -------
    numpy.ndarray
        ``(M, 2048)`` ``float64``.

    Raises
    ------
    MetricError
        If the cached file does not match the request and cannot be replaced.
    """
    root = Path(dataset_root)
    cache = root / REFERENCE_FEATURES_NAME
    sidecar = root / REFERENCE_SIDECAR_NAME
    expected = int(np.asarray(images_u8).shape[0])

    if cache.exists() and sidecar.exists() and not force:
        record = json.loads(sidecar.read_text())
        stored = np.load(cache)
        matches = (
            record.get("dataset_sha256") == dataset_sha256
            and int(record.get("n_reference", -1)) == expected
            and stored.shape == (expected, INCEPTION_FEATURE_DIM)
        )
        if matches:
            logger.info("reference Inception features read from %s", cache)
            return np.asarray(stored, dtype=np.float64)
        logger.warning("%s does not match the dataset; recomputing", cache)

    features = inception_features(images_u8, device=device, batch=batch)
    np.save(cache, features)
    sidecar.write_text(
        json.dumps(
            {
                "dataset_sha256": dataset_sha256,
                "n_reference": expected,
                "feature_dim": INCEPTION_FEATURE_DIM,
                "mode": "clean",
                "weights_path": str(inception_weights_path()),
                "created": datetime.now(UTC).isoformat(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    logger.info("reference Inception features written to %s", cache)
    return features
