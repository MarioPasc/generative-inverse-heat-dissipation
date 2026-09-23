"""Pixel-space nearest neighbours and the memorisation ratio ``M`` (``05-metrics.md`` §4).

``M`` answers the question the terminal-blur design worries about: when the prior hands the
chain a *training* seed, does the model give back a new image or a copy of that seed? It is the
median distance from a training-seeded sample to its nearest training image, divided by the same
median for real held-out images:

* ``M = 1``: the samples sit as far from the training set as a genuinely new real image;
* ``M < 1``: they sit closer, which is copying;
* ``M_lp``: the same after the released heat kernel at ``sigma_lp = 16`` px, which asks whether
  the copying is of the coarse structure alone (the part the prior actually hands over) rather
  than of the fine detail;
* ``seed_nn_fraction``: the fraction of samples whose nearest training image belongs to the
  subject of their own seed — the sharpest form of the same question, and the one that
  distinguishes "close to the training manifold" from "close to *this* training image".

``05-metrics.md`` §4 fixes the sample set (the 2 000 training-seeded samples of §8a), the corpus
(the **full** training split, the sample's own seed included) and the reference (the ``ref``
split minus the seed subjects, which the caller filters and this module verifies by hash).

This module also owns the two torch conventions the pixel-space metrics of T4.2 share --
:func:`resolve_device` and :func:`float32_full_precision` -- which ``ihdm.metrics.diversity``
imports rather than restating; ``ihdm.metrics.lowpass`` stays pure numpy.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from ihdm.metrics.errors import MetricError
from ihdm.metrics.lowpass import dct_lowpass, to_float_stack

__all__ = [
    "DEFAULT_CHUNK",
    "DEFAULT_SIGMA_LP",
    "MemResult",
    "float32_full_precision",
    "memorisation_ratio",
    "nn_distances",
    "resolve_device",
]

logger = logging.getLogger(__name__)

#: Query images per ``torch.cdist`` call. At 192 px, 256 queries against the 3 200-image
#: training split cost a 3.3 MB distance block; the corpus itself (472 MB) is resident.
DEFAULT_CHUNK: int = 256

#: Low-pass length-scale of ``M_lp``, in pixels (``05-metrics.md`` §4).
DEFAULT_SIGMA_LP: float = 16.0


def resolve_device(device: str) -> str:
    """Return the device actually usable, falling back to the CPU when CUDA is absent.

    Parameters
    ----------
    device : str
        Requested device, ``"cuda"``, ``"cuda:<k>"`` or ``"cpu"``.

    Returns
    -------
    str
        ``device`` itself, or ``"cpu"`` when a CUDA device was asked for and none is available.

    Raises
    ------
    MetricError
        If the string names neither a CPU nor a CUDA device.
    """
    if device.startswith("cpu"):
        return "cpu"
    if not device.startswith("cuda"):
        raise MetricError(f"unknown device {device!r}; expected 'cpu' or 'cuda[:k]'")
    if torch.cuda.is_available():
        return device
    logger.warning("CUDA is not available; running %s on the CPU", device)
    return "cpu"


@contextlib.contextmanager
def float32_full_precision() -> Iterator[None]:
    """Force full ``float32`` matrix multiplication for the duration of the block.

    ``torch.cdist`` at ``p = 2`` uses the ``|a|^2 + |b|^2 - 2 a.b`` matrix-multiply form above
    25 rows, and ``torch.pca_lowrank`` is matmul-bound as well, so on an Ampere GPU the TF32
    path (a 10-bit mantissa) would put the CUDA result about 1e-2 away from the CPU result on
    distances of order 10 -- two orders of magnitude outside the 1e-4 the metric harness asks
    for. The flag is global and any caller can flip it, so the metrics set it themselves and
    restore the previous value on exit.

    Yields
    ------
    None
    """
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def _flat_centred(images: object, name: str) -> np.ndarray:
    """Return a stack as a DC-removed ``float32`` matrix of shape ``(N, H * W)``.

    The centring is done in place on the fresh array :func:`to_float_stack` returns, which is
    what ``ihdm.metrics.lowpass.remove_dc`` does out of place; the in-place form is used here
    because the corpus of the real runs is 472 MB and a second copy of it is pure waste.

    Parameters
    ----------
    images : object
        Stack of shape ``(N, H, W)``, ``uint8`` or float.
    name : str
        Name used in error messages.

    Returns
    -------
    numpy.ndarray
        ``float32`` matrix of shape ``(N, H * W)``, every row of zero mean.

    Raises
    ------
    MetricError
        If the stack is malformed, empty or non-finite.
    """
    stack = to_float_stack(images, name, min_images=1)
    flat = stack.reshape(stack.shape[0], -1)
    flat -= flat.mean(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    return flat


def _nn_on_device(
    queries: np.ndarray, corpus: np.ndarray, chunk: int, device: str
) -> tuple[np.ndarray, np.ndarray]:
    """Chunked ``torch.cdist`` nearest neighbour of every query row in the corpus.

    Parameters
    ----------
    queries, corpus : numpy.ndarray
        ``float32`` matrices of shape ``(Q, D)`` and ``(C, D)``.
    chunk : int
        Query rows per ``cdist`` call.
    device : str
        Device to run on.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(dist[Q] float32, argmin[Q] int64)``.
    """
    distances = np.empty(queries.shape[0], dtype=np.float32)
    neighbours = np.empty(queries.shape[0], dtype=np.int64)
    with torch.inference_mode(), float32_full_precision():
        corpus_t = torch.from_numpy(corpus).to(device)
        for start in range(0, queries.shape[0], chunk):
            stop = min(start + chunk, queries.shape[0])
            block = torch.from_numpy(queries[start:stop]).to(device)
            best, where = torch.min(torch.cdist(block, corpus_t), dim=1)
            distances[start:stop] = best.detach().cpu().numpy()
            neighbours[start:stop] = where.detach().cpu().numpy()
            del block, best, where
        del corpus_t
    if device != "cpu":
        torch.cuda.empty_cache()
    return distances, neighbours


def nn_distances(
    queries: np.ndarray,
    corpus: np.ndarray,
    chunk: int = DEFAULT_CHUNK,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """Distance from every query image to its nearest corpus image, in DC-removed pixel space.

    Both sides are converted to ``float32`` in ``[0, 1]`` and have their per-image mean removed
    (``05-metrics.md`` §4), then flattened; the Euclidean distance is computed with
    ``torch.cdist`` over chunks of queries, the whole corpus resident on the device. A missing
    CUDA device or a CUDA out-of-memory error falls back to the CPU with a warning.

    Parameters
    ----------
    queries : numpy.ndarray
        Stack of shape ``(Q, H, W)``, ``uint8`` or float.
    corpus : numpy.ndarray
        Stack of shape ``(C, H, W)`` with the same image size.
    chunk : int
        Query images per ``cdist`` call.
    device : str
        ``"cuda"``, ``"cuda:<k>"`` or ``"cpu"``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(dist, argmin)`` of shapes ``(Q,)`` ``float32`` and ``(Q,)`` ``int64``; ``argmin``
        indexes rows of ``corpus``.

    Raises
    ------
    MetricError
        If either stack is malformed, empty or non-finite, the image sizes disagree, ``chunk``
        is not positive, or a distance comes out non-finite.
    """
    if chunk < 1:
        raise MetricError(f"chunk must be positive, got {chunk}")
    query_flat = _flat_centred(queries, "queries")
    corpus_flat = _flat_centred(corpus, "corpus")
    if query_flat.shape[1] != corpus_flat.shape[1]:
        raise MetricError(
            f"queries and corpus must have the same image size, got "
            f"{np.asanyarray(queries).shape[1:]} and {np.asanyarray(corpus).shape[1:]}"
        )

    # The Euclidean distance is invariant to a translation applied to both sides, and torch.cdist
    # computes it as |a|^2 + |b|^2 - 2 a.b above 25 rows: the cancellation error of that form is
    # set by how much of |a|^2 the two sides share. Subtracting the corpus mean image removes
    # exactly that shared part, which on the MRI sets (every image a brain in the same atlas
    # space) is a large fraction of the energy. Measured on 1 000 ixi training images at 192 px,
    # it halves the distance of an image to its own copy (0.094 -> 0.038, against real nearest
    # neighbours at 26-48) and leaves every other distance unchanged to 1e-5.
    centre = corpus_flat.mean(axis=0, dtype=np.float64).astype(np.float32)
    query_flat -= centre
    corpus_flat -= centre

    target = resolve_device(device)
    if target != "cpu":
        try:
            return _checked(_nn_on_device(query_flat, corpus_flat, chunk, target))
        except RuntimeError as error:
            if not isinstance(error, torch.cuda.OutOfMemoryError) and "out of memory" not in str(
                error
            ):
                raise
            logger.warning(
                "nn_distances: %s ran out of memory on %s (%d x %d at %d px); "
                "falling back to the CPU",
                type(error).__name__,
                target,
                query_flat.shape[0],
                corpus_flat.shape[0],
                int(round(float(np.sqrt(query_flat.shape[1])))),
            )
            torch.cuda.empty_cache()
    return _checked(_nn_on_device(query_flat, corpus_flat, chunk, "cpu"))


def _checked(result: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Raise :class:`MetricError` if a distance is not finite.

    Parameters
    ----------
    result : tuple[numpy.ndarray, numpy.ndarray]
        ``(dist, argmin)`` as returned by :func:`_nn_on_device`.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The same tuple.

    Raises
    ------
    MetricError
        If any distance is ``nan`` or infinite.
    """
    if not np.isfinite(result[0]).all():
        raise MetricError("nearest-neighbour distances came out non-finite")
    return result


def _digests(stack: np.ndarray) -> list[bytes]:
    """128-bit content digest of every image of a ``float32`` stack.

    Parameters
    ----------
    stack : numpy.ndarray
        ``float32`` stack of shape ``(N, H, W)`` produced by
        :func:`ihdm.metrics.lowpass.to_float_stack`, so that a ``uint8`` corpus and a
        ``float32`` copy of the same images hash identically.

    Returns
    -------
    list[bytes]
        One 16-byte digest per image.
    """
    return [
        hashlib.blake2b(np.ascontiguousarray(image).tobytes(), digest_size=16).digest()
        for image in stack
    ]


def _assert_disjoint(train: np.ndarray, heldout: np.ndarray) -> None:
    """Raise if any held-out image also appears in the training corpus.

    ``05-metrics.md`` §4 divides by the median distance of *held-out* real images; a held-out
    image that is also in the corpus has distance zero and silently deflates the denominator,
    which inflates ``M`` towards "no copying". The caller filters the seed subjects out; this
    check is the guard on that contract.

    Parameters
    ----------
    train, heldout : numpy.ndarray
        ``float32`` stacks of shapes ``(C, H, W)`` and ``(R, H, W)``.

    Raises
    ------
    MetricError
        If the two sets share an image.
    """
    train_digests = set(_digests(train))
    shared = [i for i, digest in enumerate(_digests(heldout)) if digest in train_digests]
    if shared:
        raise MetricError(
            f"{len(shared)} held-out images are also in the training corpus "
            f"(first at held-out index {shared[0]}); the caller must remove the seed subjects"
        )


@dataclass(frozen=True)
class MemResult:
    """Result of :func:`memorisation_ratio` (``05-metrics.md`` §4).

    Parameters
    ----------
    M : float
        ``median d(samples) / median d(heldout)`` in DC-removed pixel space.
    M_lp : float
        The same after the heat kernel at ``sigma_lp`` on samples, corpus and held-out set.
    d_samples_median, d_heldout_median : float
        The two medians of the pixel-space ratio, in the units of the images (``[0, 1]``
        intensities, Euclidean norm over ``H * W`` pixels).
    seed_nn_fraction : float
        Fraction of samples whose nearest training image belongs to the subject of their own
        seed. ``0.0`` when ``sample_seed_idx`` is ``None``.
    n_samples, n_train, n_heldout : int
        Sizes of the three sets.
    per_sample_d : numpy.ndarray
        ``(n_samples,)`` ``float32``: the pixel-space distance of every sample.
    per_sample_nn : numpy.ndarray
        ``(n_samples,)`` ``int64``: the row of ``train`` each sample is closest to.
    """

    M: float
    M_lp: float
    d_samples_median: float
    d_heldout_median: float
    seed_nn_fraction: float
    n_samples: int
    n_train: int
    n_heldout: int
    per_sample_d: np.ndarray
    per_sample_nn: np.ndarray


def _ratio(
    samples: np.ndarray, train: np.ndarray, heldout: np.ndarray, chunk: int, device: str, tag: str
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """One memorisation ratio and the pieces it is made of.

    Parameters
    ----------
    samples, train, heldout : numpy.ndarray
        The three stacks, already low-passed when the low-pass variant is being computed.
    chunk : int
        Query images per ``cdist`` call.
    device : str
        Device passed to :func:`nn_distances`.
    tag : str
        ``"M"`` or ``"M_lp"``, used in the error message.

    Returns
    -------
    tuple
        ``(ratio, median_samples, median_heldout, per_sample_d, per_sample_nn)``.

    Raises
    ------
    MetricError
        If the held-out median distance is zero, which leaves the ratio undefined.
    """
    d_samples, nn_samples = nn_distances(samples, train, chunk=chunk, device=device)
    d_heldout, _ = nn_distances(heldout, train, chunk=chunk, device=device)
    median_samples = float(np.median(d_samples))
    median_heldout = float(np.median(d_heldout))
    if median_heldout <= 0.0:
        raise MetricError(
            f"{tag} is undefined: the median held-out nearest-neighbour distance is "
            f"{median_heldout}, so more than half of the held-out images are in the corpus"
        )
    return median_samples / median_heldout, median_samples, median_heldout, d_samples, nn_samples


def _seed_fraction(
    neighbours: np.ndarray, train_subjects: Sequence[str], sample_seed_idx: np.ndarray | None
) -> float:
    """Fraction of samples whose nearest training image shares its seed's subject.

    Parameters
    ----------
    neighbours : numpy.ndarray
        ``(Y,)`` rows of ``train`` the samples are closest to.
    train_subjects : Sequence[str]
        Subject of every training image, aligned with the rows of ``train``. For the photograph
        datasets one image is one subject, so "same subject" is "same image".
    sample_seed_idx : numpy.ndarray or None
        ``(Y,)`` **row of ``train``** each sample was seeded from, or ``None``.

    Returns
    -------
    float
        A number in ``[0, 1]``; ``0.0`` when ``sample_seed_idx`` is ``None``.

    Raises
    ------
    MetricError
        If the index array has the wrong shape or leaves the corpus.
    """
    if sample_seed_idx is None:
        logger.warning(
            "sample_seed_idx is None: seed_nn_fraction cannot be computed and is reported as 0.0"
        )
        return 0.0
    index = np.asarray(sample_seed_idx)
    if index.ndim != 1 or index.shape[0] != neighbours.shape[0]:
        raise MetricError(
            f"sample_seed_idx must have shape ({neighbours.shape[0]},), got {index.shape}"
        )
    if not np.issubdtype(index.dtype, np.integer):
        raise MetricError(f"sample_seed_idx must be an integer array, got {index.dtype}")
    if int(index.min()) < 0 or int(index.max()) >= len(train_subjects):
        raise MetricError(
            "sample_seed_idx must hold rows of the train stack, i.e. values in "
            f"[0, {len(train_subjects)}); got [{index.min()}, {index.max()}]. The sampler's "
            "seed_idx.npy holds dataset indices; the caller maps them to rows of train."
        )
    subjects = np.asarray(train_subjects, dtype=object)
    return float(np.mean(subjects[neighbours] == subjects[index]))


def memorisation_ratio(
    samples: np.ndarray,
    train: np.ndarray,
    heldout: np.ndarray,
    train_subjects: Sequence[str],
    sample_seed_idx: np.ndarray | None,
    sigma_lp: float = DEFAULT_SIGMA_LP,
    device: str = "cuda",
    chunk: int = DEFAULT_CHUNK,
) -> MemResult:
    """Memorisation ratio ``M``, its low-pass variant and the seed nearest-neighbour fraction.

    ``05-metrics.md`` §4. ``samples`` are the **training-seeded** samples of §8a (one per
    training seed); the corpus is the **full** training split, the sample's own seed included;
    ``heldout`` is the ``ref`` split with the seed subjects already removed by the caller, which
    this function verifies by hashing both sets.

    Parameters
    ----------
    samples : numpy.ndarray
        Training-seeded samples, shape ``(Y, H, W)``, ``uint8`` or float.
    train : numpy.ndarray
        The whole training split, shape ``(C, H, W)``.
    heldout : numpy.ndarray
        Held-out real images, shape ``(R, H, W)``, disjoint from ``train``.
    train_subjects : Sequence[str]
        Subject identifier of every row of ``train`` (``index.csv``'s ``subject`` column,
        selected by the training split). One image is one subject in the photograph datasets.
    sample_seed_idx : numpy.ndarray or None
        For every sample, the **row of ``train``** it was seeded from. The sampler writes
        *dataset* indices in ``seed_idx.npy`` and ``splits["train"]`` is not sorted, so the
        caller maps them through an argsort
        (``order = np.argsort(rows); order[np.searchsorted(rows, seed_idx, sorter=order)]``).
        ``None`` reports ``seed_nn_fraction = 0.0`` with a warning.
    sigma_lp : float
        Low-pass length-scale of ``M_lp``, in pixels.
    device : str
        ``"cuda"``, ``"cuda:<k>"`` or ``"cpu"``; falls back to the CPU when CUDA is absent or
        out of memory.
    chunk : int
        Query images per ``cdist`` call.

    Returns
    -------
    MemResult
        The ratios, the two medians, the seed fraction, the set sizes and the per-sample arrays.

    Raises
    ------
    MetricError
        If any stack is malformed, empty or non-finite, the image sizes disagree,
        ``train_subjects`` is not aligned with ``train``, the held-out set overlaps the corpus,
        or a ratio is undefined.
    """
    sample_stack = to_float_stack(samples, "samples")
    train_stack = to_float_stack(train, "train")
    heldout_stack = to_float_stack(heldout, "heldout")
    sizes = {stack.shape[1:] for stack in (sample_stack, train_stack, heldout_stack)}
    if len(sizes) != 1:
        raise MetricError(f"samples, train and heldout must share the image size, got {sizes}")
    if len(train_subjects) != train_stack.shape[0]:
        raise MetricError(
            f"train_subjects has {len(train_subjects)} entries for {train_stack.shape[0]} "
            "training images"
        )
    _assert_disjoint(train_stack, heldout_stack)

    ratio, median_samples, median_heldout, per_d, per_nn = _ratio(
        sample_stack, train_stack, heldout_stack, chunk, device, "M"
    )
    ratio_lp = _ratio(
        dct_lowpass(sample_stack, sigma_lp),
        dct_lowpass(train_stack, sigma_lp),
        dct_lowpass(heldout_stack, sigma_lp),
        chunk,
        device,
        "M_lp",
    )[0]
    return MemResult(
        M=ratio,
        M_lp=ratio_lp,
        d_samples_median=median_samples,
        d_heldout_median=median_heldout,
        seed_nn_fraction=_seed_fraction(per_nn, train_subjects, sample_seed_idx),
        n_samples=int(sample_stack.shape[0]),
        n_train=int(train_stack.shape[0]),
        n_heldout=int(heldout_stack.shape[0]),
        per_sample_d=per_d,
        per_sample_nn=per_nn,
    )
