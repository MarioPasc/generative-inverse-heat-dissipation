"""Within-seed diversity and the PCA panel around the seed (``05-metrics.md`` §3, §6).

The two endpoints on the *held-out*-seeded sample set of ``05-metrics.md`` §8a (40 seeds, 50
samples each, the same prior state, independent sampling noise):

* :func:`within_seed_diversity` -- how much the chain moves when only the sampling noise
  changes. It is the non-DC pixel variance of the 50 samples about their own mean, per pixel,
  in pixel space (``D_pix``) and after the released heat kernel at ``sigma_lp = 16`` px
  (``D_lp``). Zero means the terminal state determines the sample: the model has nothing left
  to generate, which is the failure mode the terminal-blur hypothesis predicts for a large
  ``sigma_B,max``.
* :func:`pca_around_seed` -- the figure of §6: two components fitted on the training split, the
  seed and its samples projected onto them, so the arrow from a seed to its sample centroid can
  be read against the spread of the real data.

The torch device conventions come from :mod:`ihdm.metrics.memorisation`
(:func:`~ihdm.metrics.memorisation.resolve_device`,
:func:`~ihdm.metrics.memorisation.float32_full_precision`) so they exist once.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

from ihdm.metrics.errors import MetricError
from ihdm.metrics.lowpass import dct_lowpass, remove_dc, to_float_stack
from ihdm.metrics.memorisation import (
    DEFAULT_SIGMA_LP,
    float32_full_precision,
    resolve_device,
)

__all__ = [
    "DivResult",
    "PCA_OVERSAMPLING",
    "PCA_POWER_ITERATIONS",
    "PcaResult",
    "pca_around_seed",
    "within_seed_diversity",
]

logger = logging.getLogger(__name__)

#: Extra columns of the randomised range finder of :func:`pca_around_seed`
#: (``q = n_components + PCA_OVERSAMPLING``), and its number of power iterations. Both are the
#: values the ticket freezes; they are the standard oversampling of Halko, Martinsson & Tropp
#: (SIAM Review 53(2), 2011, doi:10.1137/090771806).
PCA_OVERSAMPLING: int = 8
PCA_POWER_ITERATIONS: int = 4


@dataclass(frozen=True)
class DivResult:
    """Result of :func:`within_seed_diversity` (``05-metrics.md`` §3).

    Parameters
    ----------
    D_pix_mean, D_lp_mean : float
        Mean over seeds of the per-seed diversity, in pixel space and after the low-pass.
        Units are squared ``[0, 1]`` intensity per pixel.
    per_seed_pix, per_seed_lp : numpy.ndarray
        ``(n_seeds,)`` ``float64`` arrays with the per-seed values.
    n_seeds, n_per_seed : int
        Shape of the sample set the result was computed on.
    """

    D_pix_mean: float
    D_lp_mean: float
    per_seed_pix: np.ndarray
    per_seed_lp: np.ndarray
    n_seeds: int
    n_per_seed: int


def _seed_variance(block: np.ndarray) -> float:
    """Non-DC pixel variance of one seed's samples about their own mean.

    ``05-metrics.md`` §3: ``D(s) = (1 / M) sum_m ||y_sm - mean_m y_sm||^2 / (H W - 1)`` with the
    DC removed from every ``y`` first. The ``1 / M`` (rather than ``1 / (M - 1)``) is the frozen
    form, so ``D`` estimates ``(1 - 1/M)`` times the true within-seed variance; at the fixed
    ``M = 50`` of §8a the 2% factor is common to every arm and cancels in every comparison.

    Parameters
    ----------
    block : numpy.ndarray
        ``float32`` stack of shape ``(M, H, W)``, DC already removed.

    Returns
    -------
    float
        The per-pixel variance, accumulated in ``float64``.
    """
    n_per_seed, height, width = block.shape
    residual = block - block.mean(axis=0, keepdims=True, dtype=np.float64)
    total = float(np.square(residual, dtype=np.float64).sum())
    return total / (n_per_seed * (height * width - 1))


def within_seed_diversity(samples: np.ndarray, sigma_lp: float = DEFAULT_SIGMA_LP) -> DivResult:
    """Within-seed diversity of a held-out-seeded sample set (``05-metrics.md`` §3).

    One seed at a time, so the peak memory is one seed's block rather than the whole
    ``(40, 50, 192, 192)`` set.

    Parameters
    ----------
    samples : numpy.ndarray
        Stack of shape ``(S, M, H, W)``, ``uint8`` or float; ``M >= 2``.
    sigma_lp : float
        Low-pass length-scale of ``D_lp``, in pixels.

    Returns
    -------
    DivResult
        The two means and the two per-seed arrays.

    Raises
    ------
    MetricError
        If the stack is not ``(S, M, H, W)`` with square images, is empty, holds fewer than two
        samples per seed, or holds a non-finite value.
    """
    block = np.asanyarray(samples)
    if block.ndim != 4:
        raise MetricError(f"samples: expected a stack of shape (S, M, H, W), got {block.shape}")
    if block.shape[0] < 1:
        raise MetricError("samples: expected at least one seed")
    if block.shape[1] < 2:
        raise MetricError(
            f"samples: within-seed diversity needs at least two samples per seed, got "
            f"{block.shape[1]}"
        )

    per_seed_pix = np.empty(block.shape[0], dtype=np.float64)
    per_seed_lp = np.empty(block.shape[0], dtype=np.float64)
    for seed in range(block.shape[0]):
        images = to_float_stack(block[seed], f"samples[{seed}]", min_images=2)
        per_seed_pix[seed] = _seed_variance(remove_dc(images))
        per_seed_lp[seed] = _seed_variance(remove_dc(dct_lowpass(images, sigma_lp)))
    return DivResult(
        D_pix_mean=float(per_seed_pix.mean()),
        D_lp_mean=float(per_seed_lp.mean()),
        per_seed_pix=per_seed_pix,
        per_seed_lp=per_seed_lp,
        n_seeds=int(block.shape[0]),
        n_per_seed=int(block.shape[1]),
    )


@dataclass(frozen=True)
class PcaResult:
    """Result of :func:`pca_around_seed` (``05-metrics.md`` §6).

    Parameters
    ----------
    components : numpy.ndarray
        ``(n_components, H, W)`` ``float32``: the principal directions as images, orthonormal
        when flattened, with a fixed sign convention (the entry of largest magnitude of each
        component is positive).
    mean : numpy.ndarray
        ``(H, W)`` ``float32``: the mean of the DC-removed training split, subtracted from every
        stack before projection.
    explained_variance : numpy.ndarray
        ``(n_components,)`` ``float64``: the variance of the training split along each
        component, ``sigma_k^2 / (C - 1)``.
    train_scores : numpy.ndarray
        ``(C, n_components)`` ``float32``.
    seed_scores : numpy.ndarray
        ``(S, n_components)`` ``float32``.
    sample_scores : numpy.ndarray
        ``(S, M, n_components)`` ``float32``.
    """

    components: np.ndarray
    mean: np.ndarray
    explained_variance: np.ndarray
    train_scores: np.ndarray
    seed_scores: np.ndarray
    sample_scores: np.ndarray


def _flat_centred(images: np.ndarray, name: str, mean: np.ndarray | None) -> np.ndarray:
    """Return a stack DC-removed, flattened and, when ``mean`` is given, mean-subtracted.

    Parameters
    ----------
    images : numpy.ndarray
        Stack of shape ``(N, H, W)``.
    name : str
        Name used in error messages.
    mean : numpy.ndarray or None
        ``(H * W,)`` training mean to subtract, or ``None``.

    Returns
    -------
    numpy.ndarray
        ``float32`` matrix of shape ``(N, H * W)``.
    """
    stack = remove_dc(to_float_stack(images, name))
    flat = stack.reshape(stack.shape[0], -1)
    if mean is not None:
        flat -= mean
    return flat


def _fit(
    matrix: np.ndarray, n_components: int, rng_seed: int, device: str
) -> tuple[np.ndarray, np.ndarray]:
    """Randomised SVD of the centred training matrix on ``device``.

    ``torch.pca_lowrank`` is the randomised range finder of Halko, Martinsson & Tropp (2011)
    with ``PCA_POWER_ITERATIONS`` power iterations; it is chosen over
    ``scipy.sparse.linalg.svds`` because the training matrix is *dense* (3 200 x 36 864,
    472 MB in ``float32``) and ARPACK would run it in ``float64`` on the CPU, while this path
    reuses the device the other pixel-space metrics already use. The global torch RNG is forked
    rather than seeded, so the function leaves no RNG state behind (``02`` §2).

    Parameters
    ----------
    matrix : numpy.ndarray
        ``(C, D)`` ``float32``, already centred.
    n_components : int
        Number of components to keep.
    rng_seed : int
        Seed of the random test matrix.
    device : str
        Device to run on.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(V[:, :k] as float32, explained_variance[k] as float64)``.
    """
    rank = min(n_components + PCA_OVERSAMPLING, *matrix.shape)
    devices = [] if device == "cpu" else [torch.device(device)]
    with torch.inference_mode(), float32_full_precision(), torch.random.fork_rng(devices=devices):
        torch.manual_seed(rng_seed)
        data = torch.from_numpy(matrix).to(device)
        _, singular, basis = torch.pca_lowrank(
            data, q=rank, center=False, niter=PCA_POWER_ITERATIONS
        )
        vectors = basis[:, :n_components].detach().cpu().numpy().copy()
        variance = (
            (singular[:n_components].double() ** 2 / (matrix.shape[0] - 1)).detach().cpu().numpy()
        )
        del data, basis, singular
    if device != "cpu":
        torch.cuda.empty_cache()
    return vectors, np.asarray(variance, dtype=np.float64).copy()


def _fix_signs(vectors: np.ndarray) -> np.ndarray:
    """Make the entry of largest magnitude of every component positive.

    ``torch.pca_lowrank`` draws its own Gaussian test matrix, so without a convention the sign
    of each component -- and therefore of every score -- depends on the draw and on the device,
    and "the PCA scores are reproducible" would be meaningless. The convention is a property of
    the subspace, not of the draw.

    Parameters
    ----------
    vectors : numpy.ndarray
        ``(D, k)`` basis, columns orthonormal.

    Returns
    -------
    numpy.ndarray
        The same basis with the sign of each column fixed.
    """
    pivot = np.abs(vectors).argmax(axis=0)
    signs = np.sign(vectors[pivot, np.arange(vectors.shape[1])])
    signs = np.where(signs == 0.0, np.float32(1.0), signs).astype(vectors.dtype)
    return vectors * signs


def pca_around_seed(
    train: np.ndarray,
    seeds: np.ndarray,
    samples: np.ndarray,
    n_components: int = 2,
    rng_seed: int = 0,
    device: str = "cuda",
) -> PcaResult:
    """Fit the PCA of the training split and project the seeds and their samples.

    ``05-metrics.md`` §6. The fit is on the DC-removed training split, mean-centred; the seeds
    ``(S, H, W)`` and their samples ``(S, M, H, W)`` are projected onto the same components, so
    the figure can draw an arrow from each seed to its sample centroid inside the cloud of the
    real data.

    Parameters
    ----------
    train : numpy.ndarray
        Training split, shape ``(C, H, W)``, ``uint8`` or float.
    seeds : numpy.ndarray
        Seed images, shape ``(S, H, W)``.
    samples : numpy.ndarray
        Samples of those seeds, shape ``(S, M, H, W)``, aligned with ``seeds``.
    n_components : int
        Number of components.
    rng_seed : int
        Seed of the randomised SVD.
    device : str
        ``"cuda"``, ``"cuda:<k>"`` or ``"cpu"``; falls back to the CPU when CUDA is absent or
        out of memory.

    Returns
    -------
    PcaResult
        Components, mean, explained variance and the three score arrays.

    Raises
    ------
    MetricError
        If a stack is malformed, empty or non-finite, the image sizes disagree, the seed counts
        disagree, ``n_components`` is not positive, or the training split holds fewer than
        ``n_components + 1`` images.
    """
    if n_components < 1:
        raise MetricError(f"n_components must be positive, got {n_components}")
    sample_block = np.asanyarray(samples)
    if sample_block.ndim != 4:
        raise MetricError(
            f"samples: expected a stack of shape (S, M, H, W), got {sample_block.shape}"
        )

    train_flat = _flat_centred(train, "train", None)
    if train_flat.shape[0] < n_components + 1:
        raise MetricError(
            f"train: need at least {n_components + 1} images to fit {n_components} components, "
            f"got {train_flat.shape[0]}"
        )
    mean = train_flat.mean(axis=0, dtype=np.float64).astype(np.float32)
    train_flat -= mean

    seed_flat = _flat_centred(seeds, "seeds", mean)
    flat_samples = sample_block.reshape(-1, *sample_block.shape[2:])
    sample_flat = _flat_centred(flat_samples, "samples", mean)
    if {seed_flat.shape[1], sample_flat.shape[1]} != {train_flat.shape[1]}:
        raise MetricError("train, seeds and samples must share the image size")
    if seed_flat.shape[0] != sample_block.shape[0]:
        raise MetricError(
            f"seeds has {seed_flat.shape[0]} images for {sample_block.shape[0]} seeds of samples"
        )

    target = resolve_device(device)
    try:
        vectors, variance = _fit(train_flat, n_components, rng_seed, target)
    except RuntimeError as error:
        if target == "cpu" or (
            not isinstance(error, torch.cuda.OutOfMemoryError) and "out of memory" not in str(error)
        ):
            raise
        logger.warning("pca_around_seed: out of memory on %s; falling back to the CPU", target)
        torch.cuda.empty_cache()
        target = "cpu"
        vectors, variance = _fit(train_flat, n_components, rng_seed, target)

    basis = _fix_signs(vectors)
    side = int(round(float(np.sqrt(train_flat.shape[1]))))
    return PcaResult(
        components=np.ascontiguousarray(basis.T).reshape(n_components, side, side),
        mean=mean.reshape(side, side),
        explained_variance=np.asarray(variance, dtype=np.float64),
        train_scores=train_flat @ basis,
        seed_scores=seed_flat @ basis,
        sample_scores=(sample_flat @ basis).reshape(*sample_block.shape[:2], n_components),
    )
