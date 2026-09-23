"""Resampling statistics of ``05-metrics.md`` §8 and the paired plateau gate of §8a.

A cell of the design is a (dataset, arm) pair holding ``s`` seeds. Every contrast the analysis
makes is a **paired** difference within a dataset (seed 1 of the arm against seed 1 of A0), its
uncertainty comes from resampling seeds with replacement, and the interaction estimand is the
difference of two such paired means across the two data types.

The plateau gate of D10/D17 is a different animal: it does not resample seeds of the *design* but
the 500 evaluation seeds of one run, and it compares the same run at two checkpoints, so the two
sample stacks are paired image by image (identical seeds, identical noise streams, D17). Its
statistic is the bootstrap distribution of ``LSD(35k) - LSD(40k)`` over resampled evaluation
seeds, and the run is extended only when the 95% interval lies strictly above zero.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.fft import dctn

from ihdm.metrics.spectral import (
    N_LOG_BINS,
    log_bin_centres,
    log_bin_edges,
    radial_log_profile,
)
from ihdm.spectral.power import radial_index
from ihdm.stats.errors import StatsError

__all__ = [
    "GateResult",
    "Interaction",
    "Interval",
    "PairedDelta",
    "bootstrap_ci",
    "interaction",
    "paired_delta",
    "paired_lsd_gate",
]

logger = logging.getLogger(__name__)

#: Default number of bootstrap resamples of ``05-metrics.md`` §8.
N_BOOT: int = 10_000

#: Default number of bootstrap resamples of the plateau gate (``05-metrics.md`` §8a).
N_BOOT_GATE: int = 1_000

#: Bootstrap replicates evaluated in one matrix product by the gate's fast path.
GATE_CHUNK: int = 100

#: Images transformed at a time when the gate precomputes DCT coefficients.
DCT_CHUNK: int = 64


# --------------------------------------------------------------------------------------------
# Intervals
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Interval:
    """A point estimate with a percentile bootstrap interval.

    Parameters
    ----------
    point : float
        The statistic on the observed sample.
    low, high : float
        The ``alpha/2`` and ``1 - alpha/2`` percentiles of the bootstrap distribution.
    n : int
        Size of the observed sample the statistic was computed on.
    n_boot : int
        Number of resamples.
    alpha : float
        Two-sided level; ``0.05`` is the 95% interval of ``05-metrics.md`` §8.
    """

    point: float
    low: float
    high: float
    n: int
    n_boot: int
    alpha: float

    @property
    def excludes_zero(self) -> bool:
        """``True`` when the interval lies strictly on one side of zero."""
        return self.low > 0.0 or self.high < 0.0

    def to_json(self) -> dict[str, Any]:
        """Return the interval as the JSON record written into the result files."""
        return {
            "point": float(self.point),
            "ci_low": float(self.low),
            "ci_high": float(self.high),
            "n": int(self.n),
            "n_boot": int(self.n_boot),
            "alpha": float(self.alpha),
            "excludes_zero": bool(self.excludes_zero),
        }


def _finite_sample(values: Any, name: str) -> np.ndarray:
    """Return ``values`` as a finite 1-D ``float64`` array."""
    array = np.asarray(values, dtype=np.float64).ravel()
    if array.size == 0:
        raise StatsError(f"{name} is empty")
    if not np.all(np.isfinite(array)):
        raise StatsError(f"{name} holds a non-finite value")
    return array


def _check_boot(n_boot: int, alpha: float) -> None:
    """Validate the resample count and the level."""
    if n_boot < 1:
        raise StatsError(f"n_boot must be positive, got {n_boot}")
    if not 0.0 < alpha < 1.0:
        raise StatsError(f"alpha must lie in (0, 1), got {alpha}")


def bootstrap_ci(
    values: Any,
    n_boot: int = N_BOOT,
    alpha: float = 0.05,
    rng_seed: int = 0,
    statistic: str = "mean",
) -> Interval:
    """Percentile bootstrap interval of the mean of ``values`` (``05-metrics.md`` §8).

    The units resampled are the entries of ``values``: seeds of a cell for a design contrast,
    samples for a sample-level quantity. Resampling is with replacement at the observed size,
    the interval is the plain percentile interval (no BCa correction: with ``s = 3`` seeds per
    cell the acceleration term is not estimable and the spec asks for the percentile interval).

    Parameters
    ----------
    values : array-like
        The observed sample, 1-D after ravelling.
    n_boot : int
        Number of resamples.
    alpha : float
        Two-sided level; the interval runs from the ``100·alpha/2`` to the
        ``100·(1 - alpha/2)`` percentile.
    rng_seed : int
        Seed of ``numpy.random.default_rng``.
    statistic : {"mean", "median"}
        The functional resampled.

    Returns
    -------
    Interval
        The statistic on the observed sample and its interval.

    Raises
    ------
    StatsError
        If the sample is empty or non-finite, ``n_boot`` is not positive, ``alpha`` is outside
        ``(0, 1)``, or ``statistic`` is unknown.
    """
    sample = _finite_sample(values, "values")
    _check_boot(n_boot, alpha)
    if statistic == "mean":
        reduce = np.mean
    elif statistic == "median":
        reduce = np.median
    else:
        raise StatsError(f"unknown statistic {statistic!r}; expected 'mean' or 'median'")

    rng = np.random.default_rng(rng_seed)
    draws = rng.integers(0, sample.size, size=(n_boot, sample.size))
    replicates = reduce(sample[draws], axis=1)
    low, high = np.percentile(replicates, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return Interval(
        point=float(reduce(sample)),
        low=float(low),
        high=float(high),
        n=int(sample.size),
        n_boot=int(n_boot),
        alpha=float(alpha),
    )


# --------------------------------------------------------------------------------------------
# Paired differences and the interaction
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PairedDelta:
    """Seed-paired difference of a metric between two cells (``05-metrics.md`` §8).

    Parameters
    ----------
    seeds : tuple[int, ...]
        The seeds present in both cells, sorted.
    deltas : numpy.ndarray
        ``a[seed] - b[seed]`` in the order of ``seeds``.
    interval : Interval
        The mean difference and its bootstrap interval over seeds.
    """

    seeds: tuple[int, ...]
    deltas: np.ndarray
    interval: Interval

    @property
    def mean(self) -> float:
        """The mean paired difference."""
        return float(self.interval.point)

    def to_json(self) -> dict[str, Any]:
        """Return the record written into the analysis tables."""
        record = self.interval.to_json()
        record["seeds"] = [int(s) for s in self.seeds]
        record["deltas"] = [float(d) for d in self.deltas]
        return record


def paired_delta(
    a: Mapping[int, float],
    b: Mapping[int, float],
    n_boot: int = N_BOOT,
    alpha: float = 0.05,
    rng_seed: int = 0,
) -> PairedDelta:
    """Seed-paired difference ``a - b`` of one metric between two cells.

    ``05-metrics.md`` §8 defines the arm contrast as ``Delta = m(arm) - m(A0)`` with seeds
    paired, so ``a`` is the arm and ``b`` is A0.

    Parameters
    ----------
    a, b : Mapping[int, float]
        The metric keyed by seed in the two cells. The two mappings must hold the same seeds.
    n_boot : int
        Resamples of the seed-level bootstrap.
    alpha : float
        Two-sided level.
    rng_seed : int
        Seed of the resampling RNG.

    Returns
    -------
    PairedDelta
        The per-seed differences and their bootstrap interval.

    Raises
    ------
    StatsError
        If either mapping is empty, their key sets differ, or a value is not finite.
    """
    if not a or not b:
        raise StatsError("both cells must hold at least one seed")
    keys_a = {int(k) for k in a}
    keys_b = {int(k) for k in b}
    if keys_a != keys_b:
        raise StatsError(
            f"the two cells are not paired: seeds only in a {sorted(keys_a - keys_b)}, "
            f"only in b {sorted(keys_b - keys_a)}"
        )
    seeds = tuple(sorted(keys_a))
    values_a = _finite_sample([a[s] for s in seeds], "a")
    values_b = _finite_sample([b[s] for s in seeds], "b")
    deltas = values_a - values_b
    return PairedDelta(
        seeds=seeds,
        deltas=deltas,
        interval=bootstrap_ci(deltas, n_boot=n_boot, alpha=alpha, rng_seed=rng_seed),
    )


@dataclass(frozen=True)
class Interaction:
    """The interaction estimand of ``05-metrics.md`` §8, ``mean(MRI) - mean(photographs)``.

    Parameters
    ----------
    interval : Interval
        The difference of the two paired means and its bootstrap interval.
    mean_mri, mean_photo : float
        The two paired means.
    n_mri, n_photo : int
        How many paired differences each side holds.
    """

    interval: Interval
    mean_mri: float
    mean_photo: float
    n_mri: int
    n_photo: int

    @property
    def point(self) -> float:
        """The interaction estimate."""
        return float(self.interval.point)

    def to_json(self) -> dict[str, Any]:
        """Return the record written into the analysis tables."""
        record = self.interval.to_json()
        record.update(
            {
                "mean_mri": float(self.mean_mri),
                "mean_photo": float(self.mean_photo),
                "n_mri": int(self.n_mri),
                "n_photo": int(self.n_photo),
            }
        )
        return record


def interaction(
    delta_mri: Any,
    delta_photo: Any,
    n_boot: int = N_BOOT,
    alpha: float = 0.05,
    rng_seed: int = 0,
) -> Interaction:
    """Difference of two paired means, with its bootstrap interval.

    The two sides come from different datasets and different seeds, so they are resampled
    independently: each bootstrap replicate draws ``n_mri`` MRI differences and ``n_photo``
    photograph differences with replacement and takes the difference of the two means. A
    positive value means the arm's effect is larger on the MRI datasets, which is the direction
    the experiment predicts for the spectral-allocation arms.

    Parameters
    ----------
    delta_mri, delta_photo : array-like
        Per-seed paired differences (the ``deltas`` of :func:`paired_delta`), pooled over the
        datasets of each data type.
    n_boot : int
        Resamples.
    alpha : float
        Two-sided level.
    rng_seed : int
        Seed of the resampling RNG.

    Returns
    -------
    Interaction
        The estimate, its interval and the two means.

    Raises
    ------
    StatsError
        If either side is empty or non-finite, or the resampling arguments are invalid.
    """
    mri = _finite_sample(delta_mri, "delta_mri")
    photo = _finite_sample(delta_photo, "delta_photo")
    _check_boot(n_boot, alpha)

    rng = np.random.default_rng(rng_seed)
    draws_mri = rng.integers(0, mri.size, size=(n_boot, mri.size))
    draws_photo = rng.integers(0, photo.size, size=(n_boot, photo.size))
    replicates = mri[draws_mri].mean(axis=1) - photo[draws_photo].mean(axis=1)
    low, high = np.percentile(replicates, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    point = float(mri.mean() - photo.mean())
    return Interaction(
        interval=Interval(
            point=point,
            low=float(low),
            high=float(high),
            n=int(mri.size + photo.size),
            n_boot=int(n_boot),
            alpha=float(alpha),
        ),
        mean_mri=float(mri.mean()),
        mean_photo=float(photo.mean()),
        n_mri=int(mri.size),
        n_photo=int(photo.size),
    )


# --------------------------------------------------------------------------------------------
# The paired plateau gate
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GateResult:
    """Result of :func:`paired_lsd_gate` (the D10/D17 plateau gate).

    Parameters
    ----------
    lsd_a, lsd_b : float
        The LSD of the two sample stacks against the reference on the full seed list; ``a`` is
        the earlier checkpoint.
    difference : Interval
        ``lsd_a - lsd_b`` and its bootstrap interval over resampled evaluation seeds. Positive
        means the later checkpoint is closer to the reference spectrum.
    extend : bool
        ``True`` when the interval lies strictly above zero, i.e. the later checkpoint is still
        improving and the run is extended (D10).
    relative_change : float
        ``(lsd_a - lsd_b) / lsd_a``, the 5% quantity D10 was originally worded with, reported
        beside the test but not used to decide.
    n_seeds : int
        Number of paired seeds.
    n_reference : int
        Number of reference images.
    n_bins : int
        Number of radial bins the RMS runs over (the populated ones).
    """

    lsd_a: float
    lsd_b: float
    difference: Interval
    extend: bool
    relative_change: float
    n_seeds: int
    n_reference: int
    n_bins: int

    def to_json(self) -> dict[str, Any]:
        """Return the ``metrics/gate.json`` record."""
        return {
            "lsd_a": float(self.lsd_a),
            "lsd_b": float(self.lsd_b),
            "difference": self.difference.to_json(),
            "extend": bool(self.extend),
            "relative_change": float(self.relative_change),
            "n_seeds": int(self.n_seeds),
            "n_reference": int(self.n_reference),
            "n_bins": int(self.n_bins),
        }


def _centred_coefficients(images: Any, name: str) -> np.ndarray:
    """Return the DCT coefficients of a stack, per-image DC removed, mean-centred over the stack.

    Reproduces ``ihdm.metrics.spectral._DcRemoved`` followed by
    ``ihdm.spectral.power.mode_power``'s two-pass centring, but keeps the per-image coefficients
    instead of collapsing them, so that a bootstrap over images is a matrix product rather than
    ``n_boot`` transforms. Centring by the full-stack mean once (rather than by the resample's
    own mean) keeps the one-pass identity
    ``mean_B[(C - C_B)^2] = mean_B[D^2] - (mean_B[D])^2`` numerically harmless, because ``D`` is
    already centred and ``mean_B[D]`` is a small bootstrap fluctuation.

    Parameters
    ----------
    images : array-like
        Stack ``(N, W, W)``, ``uint8`` or float.
    name : str
        Name used in error messages.

    Returns
    -------
    numpy.ndarray
        ``(N, W*W)`` ``float64``, the stack mean removed.

    Raises
    ------
    StatsError
        If the stack is not ``(N, W, W)`` with ``N >= 2``, or holds a non-finite value.
    """
    stack = np.asarray(images)
    if stack.ndim != 3 or stack.shape[1] != stack.shape[2]:
        raise StatsError(f"{name}: expected a stack of square images (N, W, W), got {stack.shape}")
    if stack.shape[0] < 2:
        raise StatsError(f"{name}: expected at least two images, got {stack.shape[0]}")

    n_images, width = stack.shape[0], stack.shape[1]
    coefficients = np.empty((n_images, width * width), dtype=np.float64)
    for start in range(0, n_images, DCT_CHUNK):
        block = np.asarray(stack[start : start + DCT_CHUNK])
        if np.issubdtype(block.dtype, np.integer):
            block = block.astype(np.float32) / np.float32(255.0)
        block = np.asarray(block, dtype=np.float64)
        if not np.all(np.isfinite(block)):
            raise StatsError(f"{name}: holds a non-finite value")
        block = block - block.mean(axis=(-2, -1), keepdims=True)
        coefficients[start : start + block.shape[0]] = dctn(
            block, axes=(1, 2), norm="ortho"
        ).reshape(block.shape[0], -1)
    coefficients -= coefficients.mean(axis=0, keepdims=True)
    return coefficients


def _bin_matrix(width: int, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``(W*W, n_bins)`` averaging matrix of the fine radial grid and its counts."""
    cycles = (radial_index(width) / 2.0).ravel()
    edges = log_bin_edges(n_bins)
    matrix = np.zeros((cycles.size, n_bins), dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        inside = (cycles >= low) & (cycles < high)
        if index == n_bins - 1:
            inside |= cycles == high
        counts[index] = int(inside.sum())
        if counts[index]:
            matrix[inside, index] = 1.0 / counts[index]
    return matrix, counts


def _bootstrap_profiles(
    coefficients: np.ndarray,
    weights: np.ndarray,
    bin_matrix: np.ndarray,
    dc_column: int,
) -> np.ndarray:
    """Radial profiles of the per-mode variance under bootstrap weights.

    Parameters
    ----------
    coefficients : numpy.ndarray
        ``(N, W*W)`` centred DCT coefficients of one stack.
    weights : numpy.ndarray
        ``(B, N)`` resample counts; each row sums to ``N``.
    bin_matrix : numpy.ndarray
        ``(W*W, n_bins)`` averaging matrix of :func:`_bin_matrix`.
    dc_column : int
        Column index of the DC mode, zeroed before binning.

    Returns
    -------
    numpy.ndarray
        ``(B, n_bins)`` mean per-mode variance per bin.
    """
    squares = np.square(coefficients)
    n_images = coefficients.shape[0]
    out = np.empty((weights.shape[0], bin_matrix.shape[1]), dtype=np.float64)
    for start in range(0, weights.shape[0], GATE_CHUNK):
        block = weights[start : start + GATE_CHUNK]
        mean_c = (block @ coefficients) / n_images
        power = (block @ squares) / n_images - np.square(mean_c)
        power[:, dc_column] = 0.0
        out[start : start + block.shape[0]] = power @ bin_matrix
    return out


def _rms_difference(profiles: np.ndarray, reference: np.ndarray, usable: np.ndarray) -> np.ndarray:
    """RMS over the usable bins of ``log10(profile) - log10(reference)``."""
    values = profiles[:, usable]
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.where(values > 0.0, np.log10(np.where(values > 0.0, values, 1.0)), np.nan)
    difference = logs - reference[usable][None, :]
    return np.sqrt(np.nanmean(np.square(difference), axis=1))


def paired_lsd_gate(
    samples_a: Any,
    samples_b: Any,
    reference: Any,
    n_boot: int = N_BOOT_GATE,
    alpha: float = 0.05,
    rng_seed: int = 0,
    n_bins: int = N_LOG_BINS,
) -> GateResult:
    """The paired plateau gate of D10/D17: is the later checkpoint still improving?

    ``samples_a`` and ``samples_b`` are the sample sets of one run at two checkpoints, drawn from
    the **same** frozen seed list with the **same** sampling RNG and batch (D17), so row ``k`` of
    the two stacks shares its seed and its noise stream. A bootstrap replicate draws the seed
    positions with replacement once and applies that same resample to both stacks, so the
    per-replicate difference removes the seed-composition noise the two checkpoints share; that
    pairing is the whole point of the common-random-numbers rule.

    The LSD of each replicate is the RMS over the populated bins of the fine radial grid of
    ``log10 P_sample - log10 P_reference``, exactly the functional of
    :func:`ihdm.metrics.spectral.lsd`. It is evaluated through a matrix product over precomputed
    DCT coefficients rather than by calling ``lsd`` ``2·n_boot`` times, which would transform the
    two stacks and the reference thousands of times. Memory is about ``2·N·W²·8`` bytes per
    stack (≈ 300 MB for the 500-image, ``192²`` evaluation set).

    Parameters
    ----------
    samples_a : array-like
        Samples of the **earlier** checkpoint, ``(N, W, W)``.
    samples_b : array-like
        Samples of the later checkpoint, ``(N, W, W)``, same ``N`` and the same seed order.
    reference : array-like
        The ``ref`` split of the dataset, ``(M, W, W)``.
    n_boot : int
        Bootstrap resamples of the seed list.
    alpha : float
        Two-sided level of the interval.
    rng_seed : int
        Seed of the resampling RNG.
    n_bins : int
        Number of log-spaced radial bins.

    Returns
    -------
    GateResult
        The two LSDs, the paired difference with its interval, and the ``extend`` decision.

    Raises
    ------
    StatsError
        If the two sample stacks differ in shape, either stack or the reference is malformed or
        non-finite, or the resampling arguments are invalid.
    """
    _check_boot(n_boot, alpha)
    stack_a = np.asarray(samples_a)
    stack_b = np.asarray(samples_b)
    if stack_a.shape != stack_b.shape:
        raise StatsError(
            f"the two sample stacks are not paired: {stack_a.shape} against {stack_b.shape}"
        )

    coefficients_a = _centred_coefficients(stack_a, "samples_a")
    coefficients_b = _centred_coefficients(stack_b, "samples_b")
    centres, log_reference = radial_log_profile(reference, n_bins)
    del centres
    width = stack_a.shape[1]
    if log_reference.size != n_bins:
        raise StatsError("the reference profile does not match the requested bin count")

    bin_matrix, counts = _bin_matrix(width, n_bins)
    n_seeds = int(stack_a.shape[0])
    identity = np.ones((1, n_seeds), dtype=np.float64)
    profile_a = _bootstrap_profiles(coefficients_a, identity, bin_matrix, 0)
    profile_b = _bootstrap_profiles(coefficients_b, identity, bin_matrix, 0)

    usable = (
        (counts > 0)
        & np.isfinite(log_reference)
        & (profile_a[0] > 0.0)
        & (profile_b[0] > 0.0)
    )
    if int(usable.sum()) < 2:
        raise StatsError(
            f"fewer than two usable radial bins ({int(usable.sum())} of {n_bins}); "
            "the stacks hold no resolvable spectrum"
        )

    lsd_a = float(_rms_difference(profile_a, log_reference, usable)[0])
    lsd_b = float(_rms_difference(profile_b, log_reference, usable)[0])

    rng = np.random.default_rng(rng_seed)
    draws = rng.integers(0, n_seeds, size=(n_boot, n_seeds))
    weights = np.zeros((n_boot, n_seeds), dtype=np.float64)
    rows = np.repeat(np.arange(n_boot), n_seeds)
    np.add.at(weights, (rows, draws.ravel()), 1.0)

    replicates = _rms_difference(
        _bootstrap_profiles(coefficients_a, weights, bin_matrix, 0), log_reference, usable
    ) - _rms_difference(
        _bootstrap_profiles(coefficients_b, weights, bin_matrix, 0), log_reference, usable
    )
    if not np.all(np.isfinite(replicates)):
        raise StatsError("a bootstrap replicate produced a non-finite LSD difference")

    low, high = np.percentile(replicates, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    difference = Interval(
        point=lsd_a - lsd_b,
        low=float(low),
        high=float(high),
        n=n_seeds,
        n_boot=int(n_boot),
        alpha=float(alpha),
    )
    relative = (lsd_a - lsd_b) / lsd_a if lsd_a > 0.0 else math.nan
    return GateResult(
        lsd_a=lsd_a,
        lsd_b=lsd_b,
        difference=difference,
        extend=bool(difference.low > 0.0),
        relative_change=float(relative) if math.isfinite(relative) else 0.0,
        n_seeds=n_seeds,
        n_reference=int(np.asarray(reference).shape[0]),
        n_bins=int(usable.sum()),
    )


def gate_bin_centres(n_bins: int = N_LOG_BINS) -> Sequence[float]:
    """Return the centres of the fine radial grid, for reporting beside a gate result."""
    return [float(value) for value in log_bin_centres(n_bins)]
