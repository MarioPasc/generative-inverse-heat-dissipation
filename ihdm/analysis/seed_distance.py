"""Distance of a sample to its own seed and to the training set (T8.8, post hoc).

Copying is a nearest-neighbour property, so it is measured with the pixel-space nearest neighbours
of :mod:`ihdm.metrics.memorisation` (DC removed, Euclidean, :func:`~ihdm.metrics.memorisation.
nn_distances`), not with a projection. Every distance is divided by
:math:`D_{\\text{ref}}`, the median distance from a real image of R⁻ (the ``ref`` split without
the seed subjects) to its nearest training image, which is the denominator of the memorisation
ratio ``M``. On that scale 1 means "as far as a new real image is from the training set".

Per sample of a run:

* **F** (training-seeded): :math:`q_{\\text{seed}}` = d(sample, its own seed image) and
  :math:`q_{\\text{train}}` = d(sample, the nearest training image of *another* subject), so
  that copying the own seed subject cannot make the second quantity small;
* **H** (seeded from unseen subjects): :math:`q_{\\text{seed}}` = d(sample, its own unseen seed)
  and :math:`q_{\\text{train}}` = d(sample, the nearest training image).

The nearest training image *including* the own subject is kept as well; its median over F divided
by :math:`D_{\\text{ref}}` is ``M`` and the share of F samples whose nearest image belongs to the
seed subject is the seed-NN fraction, which anchors this module to ``index.csv``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ihdm.analysis.tables import AnalysisError
from ihdm.metrics.errors import MetricError
from ihdm.metrics.memorisation import DEFAULT_CHUNK, nn_distances

__all__ = [
    "CACHE_VERSION",
    "QUANTITIES",
    "ConfigDistances",
    "NearestResult",
    "RunDistances",
    "SeedDistanceError",
    "compute_run",
    "config_distances",
    "load_or_compute",
    "nearest_training",
    "own_seed_distances",
    "reference_distances",
]

logger = logging.getLogger(__name__)

#: The four per-sample quantities, in plotting order.
QUANTITIES: tuple[str, ...] = ("F_seed", "F_train", "H_seed", "H_train")
#: Bumped whenever the cached arrays change meaning.
CACHE_VERSION: int = 1
#: The distances run on the CPU (the workstation has no GPU for this ticket).
DEVICE: str = "cpu"


class SeedDistanceError(AnalysisError):
    """The distances cannot be computed as specified (wrong shapes, no other subject, ...)."""


def _nn(queries: np.ndarray, corpus: np.ndarray, chunk: int) -> tuple[np.ndarray, np.ndarray]:
    """:func:`nn_distances` on the CPU with this module's error type."""
    try:
        return nn_distances(queries, corpus, chunk=chunk, device=DEVICE)
    except MetricError as error:
        raise SeedDistanceError(str(error)) from error


def own_seed_distances(samples: np.ndarray, seeds: np.ndarray,
                       chunk: int = DEFAULT_CHUNK) -> np.ndarray:
    """Distance of every sample to its own seed image.

    Samples are seed-major: sample ``i`` belongs to seed ``i // n_per_seed``. Each seed is the
    one-image corpus of :func:`nn_distances`, so the distance is the module's DC-removed
    Euclidean distance.

    Parameters
    ----------
    samples : numpy.ndarray
        ``(n_seeds * n_per_seed, H, W)`` uint8 (or float) samples.
    seeds : numpy.ndarray
        ``(n_seeds, H, W)`` seed images.
    chunk : int
        Query images per ``cdist`` call.

    Returns
    -------
    numpy.ndarray
        ``(n_samples,)`` float32 distances.

    Raises
    ------
    SeedDistanceError
        If the sample count is not a multiple of the seed count.
    """
    n_seeds = int(seeds.shape[0])
    if n_seeds == 0 or samples.shape[0] % n_seeds:
        raise SeedDistanceError(f"{samples.shape[0]} samples do not split over {n_seeds} seeds")
    per = samples.shape[0] // n_seeds
    out = np.empty(samples.shape[0], dtype=np.float32)
    for k in range(n_seeds):
        out[k * per:(k + 1) * per] = _nn(np.asarray(samples[k * per:(k + 1) * per]),
                                         np.asarray(seeds[k:k + 1]), chunk)[0]
    return out


@dataclass(frozen=True)
class NearestResult:
    """Nearest training image of every query, with and without the query's own group.

    Attributes
    ----------
    d_all, nn_all : numpy.ndarray
        Distance to, and row of, the nearest training image (every row allowed).
    d_other, nn_other : numpy.ndarray
        The same over the training rows of *other* groups only (equal to ``d_all`` when no group
        is given).
    own : numpy.ndarray
        ``(Q,)`` bool: the nearest training image belongs to the query's own group.
    """

    d_all: np.ndarray
    nn_all: np.ndarray
    d_other: np.ndarray
    nn_other: np.ndarray
    own: np.ndarray


def nearest_training(queries: np.ndarray, train: np.ndarray,
                     query_groups: np.ndarray | None = None,
                     train_groups: np.ndarray | None = None,
                     chunk: int = DEFAULT_CHUNK) -> NearestResult:
    """Nearest training image, and nearest training image of another group (subject).

    The nearest image of another group equals the overall nearest unless the overall nearest is
    of the query's own group; only those queries are searched again, group by group, in the
    training rows of the other groups.

    Parameters
    ----------
    queries : numpy.ndarray
        ``(Q, H, W)`` images.
    train : numpy.ndarray
        ``(C, H, W)`` training images.
    query_groups, train_groups : numpy.ndarray | None
        Group (subject) of every query and of every training row; both or neither.
    chunk : int
        Query images per ``cdist`` call.

    Returns
    -------
    NearestResult
        Distances and rows with and without the own group.

    Raises
    ------
    SeedDistanceError
        If only one of the group arrays is given, they are misaligned, or a group owns every
        training row.
    """
    if (query_groups is None) != (train_groups is None):
        raise SeedDistanceError("give both query_groups and train_groups, or neither")
    d_all, nn_all = _nn(np.asarray(queries), np.asarray(train), chunk)
    if query_groups is None or train_groups is None:
        return NearestResult(d_all, nn_all, d_all.copy(), nn_all.copy(),
                             np.zeros(d_all.shape, dtype=bool))
    query_groups, train_groups = np.asarray(query_groups), np.asarray(train_groups)
    if query_groups.shape != d_all.shape or train_groups.shape != (train.shape[0],):
        raise SeedDistanceError(f"groups {query_groups.shape}/{train_groups.shape} do not match "
                                f"{d_all.shape[0]} queries and {train.shape[0]} training images")
    own = train_groups[nn_all] == query_groups
    d_other, nn_other = d_all.copy(), nn_all.copy()
    for group in sorted(set(query_groups[own].tolist())):
        rows = np.flatnonzero(own & (query_groups == group))
        keep = np.flatnonzero(train_groups != group)
        if keep.size == 0:
            raise SeedDistanceError(f"group {group!r} owns every training image")
        d, nn = _nn(np.asarray(queries[rows]), np.asarray(train[keep]), chunk)
        d_other[rows], nn_other[rows] = d, keep[nn]
    return NearestResult(d_all, nn_all, d_other, nn_other, own)


def reference_distances(reference: np.ndarray, train: np.ndarray,
                        chunk: int = DEFAULT_CHUNK) -> np.ndarray:
    """Distance from every real reference image (R⁻) to its nearest training image.

    Its median is :math:`D_{\\text{ref}}`, the denominator of ``M``.
    """
    return _nn(np.asarray(reference), np.asarray(train), chunk)[0]


@dataclass(frozen=True)
class RunDistances:
    """The per-sample distances of one run (pixel units, before the division by D_ref).

    Attributes
    ----------
    run_id : str
        Run id.
    d_ref : float
        :math:`D_{\\text{ref}}` of the run's dataset.
    f_seed, f_train, f_nearest : numpy.ndarray
        F samples: to the own seed, to the nearest training image of another subject, to the
        nearest training image of any subject (``M``'s numerator).
    f_own_nn : numpy.ndarray
        F samples: the nearest training image belongs to the seed subject (seed-NN).
    h_seed, h_train : numpy.ndarray
        H samples: to the own unseen seed, to the nearest training image.
    """

    run_id: str
    d_ref: float
    f_seed: np.ndarray
    f_train: np.ndarray
    f_nearest: np.ndarray
    f_own_nn: np.ndarray
    h_seed: np.ndarray
    h_train: np.ndarray

    def q(self, name: str) -> np.ndarray:
        """Return one of :data:`QUANTITIES` divided by ``d_ref`` (float64)."""
        attr = {"F_seed": self.f_seed, "F_train": self.f_train, "H_seed": self.h_seed,
                "H_train": self.h_train}
        if name not in attr:
            raise SeedDistanceError(f"unknown quantity {name!r}; expected one of {QUANTITIES}")
        return np.asarray(attr[name], dtype=np.float64) / self.d_ref

    def medians(self) -> np.ndarray:
        """Medians of the four quantities, in :data:`QUANTITIES` order."""
        return np.array([float(np.median(self.q(n))) for n in QUANTITIES])

    @property
    def m_ratio(self) -> float:
        """``M`` recomputed: median nearest-training distance of F over ``d_ref``."""
        return float(np.median(np.asarray(self.f_nearest, dtype=np.float64)) / self.d_ref)

    @property
    def seed_nn(self) -> float:
        """Seed-NN fraction recomputed."""
        return float(np.mean(self.f_own_nn))


def compute_run(run_id: str, final_samples: np.ndarray, final_seed_rows: np.ndarray,
                heldout_samples: np.ndarray, heldout_seeds: np.ndarray, train: np.ndarray,
                train_subjects: Sequence[str], d_ref: float,
                chunk: int = DEFAULT_CHUNK) -> RunDistances:
    """Compute the distances of one run.

    Parameters
    ----------
    run_id : str
        Run id (for messages).
    final_samples : numpy.ndarray
        F samples, one per training seed, ``(n_F, H, W)``.
    final_seed_rows : numpy.ndarray
        For every F sample, the **row of** ``train`` it was seeded from.
    heldout_samples, heldout_seeds : numpy.ndarray
        H samples (seed-major) and their unseen seed images.
    train : numpy.ndarray
        The full training split, sorted by dataset index.
    train_subjects : Sequence[str]
        Subject of every row of ``train``.
    d_ref : float
        :math:`D_{\\text{ref}}`.
    chunk : int
        Query images per ``cdist`` call.

    Returns
    -------
    RunDistances
        The raw distances.

    Raises
    ------
    SeedDistanceError
        If the seed rows leave the training split or a set is inconsistent.
    """
    rows = np.asarray(final_seed_rows, dtype=np.int64)
    if rows.shape != (final_samples.shape[0],) or rows.min() < 0 or rows.max() >= len(train):
        raise SeedDistanceError(f"{run_id}: seed rows do not index the training split")
    if not d_ref > 0:
        raise SeedDistanceError(f"{run_id}: D_ref must be positive, got {d_ref}")
    subjects = np.asarray(list(train_subjects), dtype=object)
    if subjects.shape != (len(train),):
        raise SeedDistanceError(f"{run_id}: {subjects.size} subjects for {len(train)} images")
    logger.info("%s: distances of %d F and %d H samples", run_id, final_samples.shape[0],
                heldout_samples.shape[0])
    f_seed = own_seed_distances(final_samples, np.asarray(train)[rows], chunk)
    f = nearest_training(final_samples, train, subjects[rows], subjects, chunk)
    h_seed = own_seed_distances(heldout_samples, heldout_seeds, chunk)
    h = nearest_training(heldout_samples, train, chunk=chunk)
    return RunDistances(run_id, float(d_ref), f_seed, f.d_other, f.d_all, f.own, h_seed, h.d_all)


# --------------------------------------------------------------------------------------------
# Cache (the scratch folder of the build): keyed on the inputs' content
# --------------------------------------------------------------------------------------------

_ARRAYS: tuple[str, ...] = ("f_seed", "f_train", "f_nearest", "f_own_nn", "h_seed", "h_train")


def _digest(*parts: bytes) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(hashlib.sha256(part).digest())
    return digest.hexdigest()


def cache_key(final_samples: np.ndarray, final_seed_rows: np.ndarray,
              heldout_samples: np.ndarray, heldout_seeds: np.ndarray, train_idx: np.ndarray,
              d_ref: float) -> str:
    """Return the content key of one run's distances (inputs and :data:`CACHE_VERSION`)."""
    return _digest(str(CACHE_VERSION).encode(),
                   np.ascontiguousarray(final_samples).tobytes(),
                   np.asarray(final_seed_rows, dtype="<i8").tobytes(),
                   np.ascontiguousarray(heldout_samples).tobytes(),
                   np.ascontiguousarray(heldout_seeds).tobytes(),
                   np.asarray(train_idx, dtype="<i8").tobytes(),
                   np.float64(d_ref).tobytes())


def load_or_compute(cache_dir: Path, run_id: str, final_samples: np.ndarray,
                    final_seed_rows: np.ndarray, heldout_samples: np.ndarray,
                    heldout_seeds: np.ndarray, train: np.ndarray, train_idx: np.ndarray,
                    train_subjects: Sequence[str], d_ref: float) -> RunDistances:
    """:func:`compute_run`, reusing ``<cache_dir>/<run_id>.npz`` when its key matches.

    The key hashes every input array and ``d_ref``, so a cached file is only reused for the
    same samples, seeds, training split and reference; the arrays are stored losslessly.
    """
    key = cache_key(final_samples, final_seed_rows, heldout_samples, heldout_seeds, train_idx,
                    d_ref)
    path = Path(cache_dir) / f"{run_id}.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as stored:
            if str(stored["key"]) == key:
                logger.info("%s: distances read from %s", run_id, path)
                return RunDistances(run_id, float(d_ref),
                                    *(np.array(stored[name]) for name in _ARRAYS))
    result = compute_run(run_id, final_samples, final_seed_rows, heldout_samples,
                         heldout_seeds, train, train_subjects, d_ref)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part.npz")
    np.savez(tmp, key=np.array(key), **{name: getattr(result, name) for name in _ARRAYS})
    tmp.replace(path)
    return result


# --------------------------------------------------------------------------------------------
# Per-configuration summary
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfigDistances:
    """The distances of one configuration (dataset, arm): one row per run, seed order.

    Attributes
    ----------
    dataset, arm : str
        The cell.
    run_ids : tuple[str, ...]
        Its runs.
    medians : numpy.ndarray
        ``(n_runs, 4)`` per-run medians of :data:`QUANTITIES` (units of D_ref).
    quartiles : numpy.ndarray
        ``(4, 2)`` 25th and 75th percentiles of the per-sample values pooled over the runs.
    m_ratio, seed_nn : numpy.ndarray
        ``(n_runs,)`` recomputed ``M`` and seed-NN fraction.
    """

    dataset: str
    arm: str
    run_ids: tuple[str, ...]
    medians: np.ndarray
    quartiles: np.ndarray
    m_ratio: np.ndarray
    seed_nn: np.ndarray

    @property
    def n(self) -> int:
        return len(self.run_ids)

    def mean(self) -> np.ndarray:
        """Seed means of the per-run medians, in :data:`QUANTITIES` order."""
        return self.medians.mean(axis=0)


def config_distances(dataset: str, arm: str, runs: Sequence[RunDistances]) -> ConfigDistances:
    """Summarise the runs of one configuration.

    Raises
    ------
    SeedDistanceError
        If ``runs`` is empty.
    """
    if not runs:
        raise SeedDistanceError(f"{dataset} {arm}: no run")
    pooled = [np.concatenate([r.q(name) for r in runs]) for name in QUANTITIES]
    return ConfigDistances(
        dataset, arm, tuple(r.run_id for r in runs), np.stack([r.medians() for r in runs]),
        np.array([np.percentile(p, [25.0, 75.0]) for p in pooled]),
        np.array([r.m_ratio for r in runs]), np.array([r.seed_nn for r in runs]))


def summary_json(configs: Sequence[ConfigDistances]) -> str:
    """Return the per-run medians as canonical JSON (for logs and tests)."""
    record = {f"{c.dataset}/{c.arm}": {rid: dict(zip(QUANTITIES, map(float, row), strict=True))
                                       for rid, row in zip(c.run_ids, c.medians, strict=True)}
              for c in configs}
    return json.dumps(record, indent=2, sort_keys=True)
