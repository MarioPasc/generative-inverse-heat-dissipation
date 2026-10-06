"""Fidelity from held-out seeds (T7.5, M7, post hoc): references, per-sample metrics, reading.

Every fidelity number of the experiment is computed on *training-seeded* samples, whose prior
state is a blurred training image. T7.5 reads the same Inception metrics on the 40 × 50
*held-out-seeded* samples the evaluation stored, against references from which the seed subjects
are removed, and applies a reading rule fixed before any number existed
(``docs/SPECIFICATIONS/M7-diagnostics/T7.5-heldout-seed-fidelity.md`` step 6, amended by main on
2026-10-03 before any contrast was computed: the KID fallback applies the whole rule, rule 1
included).

This module holds the pure parts: the reference selection (R⁻ and R5), the per-sample precision
and density that reproduce :func:`ihdm.metrics.inception.recall_coverage`, the within-run cluster
bootstrap, the seed-paired contrasts and the reading rule. Input/output lives in
:mod:`ihdm.cli.heldout_fidelity`.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ihdm.analysis.tables import AnalysisError
from ihdm.metrics.inception import DEFAULT_K, kid_from_features, recall_coverage
from ihdm.stats import paired_delta, permutation_test

__all__ = [
    "HeldoutFidelityError",
    "MRI_DATASETS",
    "OUTCOME_TEXT",
    "PHOTO_DATASETS",
    "PerSample",
    "ReferenceSets",
    "anchor_features",
    "anchor_metrics",
    "bootstrap_mean_ci",
    "bootstrap_replicates",
    "classify",
    "contrast",
    "expected_heldout_seeds",
    "index_sha256",
    "per_sample_prdc",
    "read_rule",
    "select_references",
    "set_metrics",
    "sign_summary",
]

logger = logging.getLogger(__name__)

#: Datasets whose seed rule works by subject (10 slices per subject) and that get R5.
MRI_DATASETS: tuple[str, ...] = ("ixi", "oasis1")

#: Datasets whose seed rule works by image (one image per "subject").
PHOTO_DATASETS: tuple[str, ...] = ("lsun_church", "lsun_bedroom")

#: Slice level of the held-out seeds on MRI, and of the R5 sensitivity reference.
SEED_SLICE: int = 5

#: Resamples, level and seed of the within-run bootstrap (ticket step 4).
N_BOOT_WITHIN: int = 1_000
ALPHA: float = 0.05
RNG_SEED: int = 0

#: Rule 2 versus rule 3 threshold: ``G_H >= RATIO * G_F`` (ticket step 6).
GAIN_RATIO: float = 0.5

#: Tolerance of the reproduction anchor (b) on precision, recall, density and coverage.
ANCHOR_TOLERANCE: float = 0.01

#: Tolerance between the per-sample aggregates and ``recall_coverage`` (ticket step 4).
PER_SAMPLE_TOLERANCE: float = 1e-12

NOT_EVALUABLE = "not_evaluable"
NOT_EVALUABLE_BOTH = "not_evaluable_precision_or_kid"
GENERALISES = "generalises"
PARTLY_TIED = "partly_tied"
NOT_SHOWN = "not_shown"

#: The literal outcome texts of the reading rule.
OUTCOME_TEXT: dict[str, str] = {
    NOT_EVALUABLE: "the comparator shows no precision gain on R⁻; not evaluable on precision",
    NOT_EVALUABLE_BOTH: "not evaluable on precision or KID",
    GENERALISES: "the fidelity gain generalises to unseen seeds",
    PARTLY_TIED: "partly tied to training seeds",
    NOT_SHOWN: "not shown on unseen seeds",
}


class HeldoutFidelityError(AnalysisError):
    """Raised when the held-out fidelity analysis cannot proceed (inputs or an invariant)."""


# --------------------------------------------------------------------------------------------
# References
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceSets:
    """The reference index sets of one dataset (ticket steps 1 and 2).

    Parameters
    ----------
    dataset : str
        Dataset id.
    ref_idx : numpy.ndarray
        The full ``ref`` split, sorted (the order of the production feature cache).
    r_minus_idx : numpy.ndarray
        ``ref`` minus every image of the seed subjects, sorted.
    r_minus_rows : numpy.ndarray
        Positions of ``r_minus_idx`` inside ``ref_idx``, so that R⁻ features are a slice of the
        full-``ref`` features.
    r5_idx : numpy.ndarray | None
        Slice-5 images of the ``train`` subjects, sorted; ``None`` on photographs.
    seed_subjects : tuple[str, ...]
        The seed subjects.
    """

    dataset: str
    ref_idx: np.ndarray
    r_minus_idx: np.ndarray
    r_minus_rows: np.ndarray
    r5_idx: np.ndarray | None
    seed_subjects: tuple[str, ...]


def index_sha256(indices: np.ndarray) -> str:
    """Return the sha256 of an index array written as little-endian ``int64``."""
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def _subjects(index: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
    """Return the subject of each row of ``rows``."""
    return index["subject"].astype(str).to_numpy()[np.asarray(rows, dtype=np.int64)]


def _check_index(index: pd.DataFrame) -> None:
    """Raise unless ``index`` has the columns and the ``idx == row`` layout the selection uses."""
    for column in ("idx", "subject", "slice"):
        if column not in index.columns:
            raise HeldoutFidelityError(f"index.csv has no column {column!r}")
    if not np.array_equal(index["idx"].to_numpy(), np.arange(len(index))):
        raise HeldoutFidelityError("index.csv: idx does not equal the row number")


def select_references(
    index: pd.DataFrame, splits: Mapping[str, Any], dataset: str
) -> ReferenceSets:
    """Select R⁻ (and R5 on MRI) from a dataset's index and splits.

    On MRI, R⁻ removes every ``ref`` row whose subject is a seed subject (40 subjects × 10
    slices); on photographs it removes the ``seed`` images themselves. Both rules are checked
    against each other: no R⁻ row may belong to a seed subject. R5 is the slice-5 image of every
    ``train`` subject, one per subject.

    Parameters
    ----------
    index : pandas.DataFrame
        ``index.csv`` (columns ``idx, subject, slice, ...``).
    splits : Mapping[str, Any]
        ``splits.json`` (``train``, ``ref``, ``seed``, ``seed_subjects`` at least).
    dataset : str
        Dataset id; decides the MRI or the photograph rule.

    Returns
    -------
    ReferenceSets
        The sorted index sets.

    Raises
    ------
    HeldoutFidelityError
        If the dataset is unknown, ``seed`` is not inside ``ref``, R⁻ is empty or holds a seed
        subject, or a ``train`` subject has no unique slice-5 image.
    """
    if dataset not in MRI_DATASETS + PHOTO_DATASETS:
        raise HeldoutFidelityError(f"unknown dataset {dataset!r}")
    _check_index(index)
    ref = np.sort(np.asarray(splits["ref"], dtype=np.int64))
    seed = np.sort(np.asarray(splits["seed"], dtype=np.int64))
    seed_subjects = tuple(str(s) for s in splits["seed_subjects"])
    if not np.all(np.isin(seed, ref)):
        raise HeldoutFidelityError(f"{dataset}: the seed split is not inside the ref split")
    if set(_subjects(index, seed)) != set(seed_subjects):
        raise HeldoutFidelityError(f"{dataset}: seed rows and seed_subjects disagree")

    if dataset in MRI_DATASETS:
        keep = ~np.isin(_subjects(index, ref), np.asarray(seed_subjects))
    else:
        keep = ~np.isin(ref, seed)
    r_minus = ref[keep]
    if r_minus.size == 0:
        raise HeldoutFidelityError(f"{dataset}: R⁻ is empty")
    if np.isin(_subjects(index, r_minus), np.asarray(seed_subjects)).any():
        raise HeldoutFidelityError(f"{dataset}: R⁻ holds an image of a seed subject")

    r5: np.ndarray | None = None
    if dataset in MRI_DATASETS:
        train = np.sort(np.asarray(splits["train"], dtype=np.int64))
        slices = index["slice"].to_numpy()[train]
        r5 = train[slices == SEED_SLICE]
        train_subjects = set(_subjects(index, train))
        r5_subjects = _subjects(index, r5)
        if len(set(r5_subjects)) != r5.size or set(r5_subjects) != train_subjects:
            raise HeldoutFidelityError(
                f"{dataset}: R5 must hold exactly one slice-{SEED_SLICE} image per train subject"
            )
        if np.isin(r5_subjects, np.asarray(seed_subjects)).any():
            raise HeldoutFidelityError(f"{dataset}: R5 holds a seed subject")

    return ReferenceSets(
        dataset=dataset,
        ref_idx=ref,
        r_minus_idx=r_minus,
        r_minus_rows=np.flatnonzero(keep),
        r5_idx=r5,
        seed_subjects=seed_subjects,
    )


def expected_heldout_seeds(
    index: pd.DataFrame, splits: Mapping[str, Any], dataset: str
) -> np.ndarray:
    """Return the sorted dataset indices the held-out set must be seeded from.

    Parameters
    ----------
    index : pandas.DataFrame
        ``index.csv``.
    splits : Mapping[str, Any]
        ``splits.json``.
    dataset : str
        Dataset id.

    Returns
    -------
    numpy.ndarray
        MRI: the slice-5 image of each seed subject; photographs: the ``seed`` split.
    """
    _check_index(index)
    seed = np.sort(np.asarray(splits["seed"], dtype=np.int64))
    if dataset in MRI_DATASETS:
        return seed[index["slice"].to_numpy()[seed] == SEED_SLICE]
    return seed


# --------------------------------------------------------------------------------------------
# Per-sample precision and density
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PerSample:
    """Per-sample manifold quantities of one sample set against one reference.

    Parameters
    ----------
    precision : numpy.ndarray
        ``(N,)`` ``float64`` 0/1: the sample lies inside some reference k-NN ball.
    density : numpy.ndarray
        ``(N,)`` ``float64``: the number of reference balls holding the sample, divided by ``k``.
    aggregates : dict[str, float]
        ``recall_coverage``'s precision, recall, density and coverage on the same features.
    """

    precision: np.ndarray
    density: np.ndarray
    aggregates: dict[str, float]


def per_sample_prdc(f_samples: Any, f_reference: Any, k: int = DEFAULT_K) -> PerSample:
    """Per-sample precision indicators and density counts with ``prdc``'s own distances.

    The radii and the cross distances come from ``prdc.prdc``'s helpers, so the per-sample
    values are the summands of ``compute_prdc``'s aggregates. Their means are checked against
    :func:`ihdm.metrics.inception.recall_coverage` to ``1e-12``.

    Parameters
    ----------
    f_samples, f_reference : array-like
        ``(N, D)`` and ``(M, D)`` feature matrices.
    k : int
        Neighbour count.

    Returns
    -------
    PerSample
        The per-sample values and ``recall_coverage``'s aggregates.

    Raises
    ------
    HeldoutFidelityError
        If a per-sample mean does not reproduce the aggregate.
    """
    from prdc.prdc import compute_nearest_neighbour_distances, compute_pairwise_distance

    samples = np.asarray(f_samples, dtype=np.float64)
    reference = np.asarray(f_reference, dtype=np.float64)
    aggregates = recall_coverage(samples, reference, k=k)
    radii = compute_nearest_neighbour_distances(reference, k)
    inside = compute_pairwise_distance(reference, samples) < np.expand_dims(radii, axis=1)
    precision = inside.any(axis=0).astype(np.float64)
    density = inside.sum(axis=0).astype(np.float64) / float(k)
    for name, values in (("precision", precision), ("density", density)):
        gap = abs(float(values.mean()) - aggregates[name])
        if gap > PER_SAMPLE_TOLERANCE:
            raise HeldoutFidelityError(
                f"per-sample {name} does not reproduce recall_coverage (|Δ| = {gap:.3e})"
            )
    return PerSample(precision=precision, density=density, aggregates=aggregates)


# --------------------------------------------------------------------------------------------
# Within-run bootstrap
# --------------------------------------------------------------------------------------------


def bootstrap_replicates(
    values: Any,
    clusters: Any | None = None,
    n_boot: int = N_BOOT_WITHIN,
    rng_seed: int = RNG_SEED,
) -> np.ndarray:
    """Bootstrap replicates of the mean of ``values``, resampling samples or whole clusters.

    Parameters
    ----------
    values : array-like
        ``(N,)`` per-sample values.
    clusters : array-like | None
        ``(N,)`` cluster labels (the held-out seed of each sample). ``None`` resamples samples.
        With clusters, each replicate draws as many clusters as there are, with replacement, and
        takes the mean over all samples of the drawn clusters.
    n_boot : int
        Replicates.
    rng_seed : int
        Seed of ``numpy.random.default_rng``.

    Returns
    -------
    numpy.ndarray
        ``(n_boot,)`` replicate means.

    Raises
    ------
    HeldoutFidelityError
        If ``values`` is empty or non-finite, the labels do not match it, or ``n_boot < 1``.
    """
    data = np.asarray(values, dtype=np.float64).ravel()
    if data.size == 0 or not np.all(np.isfinite(data)):
        raise HeldoutFidelityError("bootstrap values must be non-empty and finite")
    if n_boot < 1:
        raise HeldoutFidelityError(f"n_boot must be positive, got {n_boot}")
    rng = np.random.default_rng(rng_seed)
    if clusters is None:
        rows = rng.integers(0, data.size, size=(n_boot, data.size))
        return data[rows].mean(axis=1)
    labels = np.asarray(clusters).ravel()
    if labels.shape != data.shape:
        raise HeldoutFidelityError("cluster labels must have one entry per value")
    _, inverse = np.unique(labels, return_inverse=True)
    sums = np.bincount(inverse, weights=data)
    sizes = np.bincount(inverse).astype(np.float64)
    draws = rng.integers(0, sums.size, size=(n_boot, sums.size))
    return sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)


def bootstrap_mean_ci(
    values: Any,
    clusters: Any | None = None,
    n_boot: int = N_BOOT_WITHIN,
    alpha: float = ALPHA,
    rng_seed: int = RNG_SEED,
) -> tuple[float, float]:
    """Percentile bootstrap interval of the mean (see :func:`bootstrap_replicates`).

    Parameters
    ----------
    values, clusters, n_boot, rng_seed
        As in :func:`bootstrap_replicates`.
    alpha : float
        Two-sided level.

    Returns
    -------
    tuple[float, float]
        ``(low, high)``.
    """
    replicates = bootstrap_replicates(values, clusters, n_boot=n_boot, rng_seed=rng_seed)
    low, high = np.percentile(replicates, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(low), float(high)


def set_metrics(
    f_samples: np.ndarray, f_reference: np.ndarray, clusters: np.ndarray | None
) -> dict[str, Any]:
    """Every metric of one sample set against one reference (ticket step 4).

    Parameters
    ----------
    f_samples : numpy.ndarray
        ``(N, D)`` sample features.
    f_reference : numpy.ndarray
        ``(M, D)`` reference features.
    clusters : numpy.ndarray | None
        Seed label of each sample for the held-out set; ``None`` for the training-seeded set.

    Returns
    -------
    dict[str, Any]
        ``precision`` and ``density`` with their 95% intervals, ``recall``, ``coverage``,
        ``kid`` (production defaults), the set sizes and the bootstrap unit.
    """
    per_sample = per_sample_prdc(f_samples, f_reference)
    p_low, p_high = bootstrap_mean_ci(per_sample.precision, clusters)
    d_low, d_high = bootstrap_mean_ci(per_sample.density, clusters)
    return {
        "precision": per_sample.aggregates["precision"],
        "precision_ci": [p_low, p_high],
        "density": per_sample.aggregates["density"],
        "density_ci": [d_low, d_high],
        "recall": per_sample.aggregates["recall"],
        "coverage": per_sample.aggregates["coverage"],
        "kid": kid_from_features(f_samples, f_reference),
        "n_samples": int(np.asarray(f_samples).shape[0]),
        "n_reference": int(np.asarray(f_reference).shape[0]),
        "bootstrap_unit": "seed cluster" if clusters is not None else "sample",
    }


# --------------------------------------------------------------------------------------------
# Contrasts and the reading
# --------------------------------------------------------------------------------------------


def contrast(arm: Mapping[int, float], a0: Mapping[int, float]) -> dict[str, Any]:
    """Seed-paired contrast ``arm − A0`` as tables 2a/2b compute it.

    Parameters
    ----------
    arm, a0 : Mapping[int, float]
        The metric keyed by run seed.

    Returns
    -------
    dict[str, Any]
        Seeds, per-seed deltas, mean, 95% seed-bootstrap interval (``paired_delta`` defaults)
        and the exact permutation p-value with its floor.
    """
    delta = paired_delta(arm, a0)
    seeds = delta.seeds
    test = permutation_test([arm[s] for s in seeds], [a0[s] for s in seeds])
    return {
        "seeds": [int(s) for s in seeds],
        "deltas": [float(d) for d in delta.deltas],
        "mean": delta.mean,
        "ci_low": float(delta.interval.low),
        "ci_high": float(delta.interval.high),
        "p_value": float(test.p_value),
        "p_min": float(test.p_min),
        "n_assignments": int(test.n_assignments),
        "arm_mean": float(np.mean([arm[s] for s in seeds])),
        "a0_mean": float(np.mean([a0[s] for s in seeds])),
    }


def _comparator_positive(gains: np.ndarray) -> bool:
    """Rule 1's condition negated: ``G_F > 0`` and every ``Δ_F(s) > 0``."""
    return bool(gains.mean() > 0.0 and np.all(gains > 0.0))


def classify(gains_f: Sequence[float], gains_h: Sequence[float]) -> str:
    """Apply rules 1–4 of ticket step 6 to one metric's per-seed gains.

    A gain is positive when the arm is better than A0 (precision: ``Δ``; KID: ``−Δ``).

    Parameters
    ----------
    gains_f, gains_h : Sequence[float]
        Per-seed gains on the training-seeded (F) and held-out-seeded (H) sets, same seeds and
        order.

    Returns
    -------
    str
        One of ``not_evaluable``, ``generalises``, ``partly_tied``, ``not_shown``.

    Raises
    ------
    HeldoutFidelityError
        If the two sequences are empty, of different lengths, or non-finite.
    """
    f = np.asarray(gains_f, dtype=np.float64)
    h = np.asarray(gains_h, dtype=np.float64)
    if f.size == 0 or f.shape != h.shape or not (np.all(np.isfinite(f)) and np.all(np.isfinite(h))):
        raise HeldoutFidelityError("gains must be finite, non-empty and paired")
    if not _comparator_positive(f):
        return NOT_EVALUABLE
    if np.all(h > 0.0):
        return GENERALISES if h.mean() >= GAIN_RATIO * f.mean() else PARTLY_TIED
    return NOT_SHOWN


def _block(metric: str, gains_f: np.ndarray, gains_h: np.ndarray, outcome: str) -> dict[str, Any]:
    """Return the JSON record of one application of the rule."""
    return {
        "metric": metric,
        "gains_f": [float(g) for g in gains_f],
        "gains_h": [float(g) for g in gains_h],
        "G_F": float(gains_f.mean()),
        "G_H": float(gains_h.mean()),
        "threshold": float(GAIN_RATIO * gains_f.mean()),
        "all_gains_h_positive": bool(np.all(gains_h > 0.0)),
        "comparator_gain_positive": _comparator_positive(gains_f),
        "outcome": outcome,
        "text": OUTCOME_TEXT[outcome],
    }


def read_rule(
    precision_delta_f: Sequence[float],
    precision_delta_h: Sequence[float],
    kid_delta_f: Sequence[float],
    kid_delta_h: Sequence[float],
) -> dict[str, Any]:
    """The pre-registered reading of ticket step 6, with main's amendment of 2026-10-03.

    Precision is primary (gain ``Δ = m(A3) − m(A0)``). If rule 1 fires on precision, the whole
    rule, rule 1 included, is applied to KID with the gain ``A0 − A3`` and the result is labelled
    secondary; if the KID comparator also fails, the outcome is ``"not evaluable on precision or
    KID"``.

    Parameters
    ----------
    precision_delta_f, precision_delta_h : Sequence[float]
        Per-seed ``Δ_F`` and ``Δ_H`` of precision.
    kid_delta_f, kid_delta_h : Sequence[float]
        Per-seed ``Δ_F`` and ``Δ_H`` of KID (``m(A3) − m(A0)``; negative is better).

    Returns
    -------
    dict[str, Any]
        ``outcome``, ``text``, ``label`` (``primary (precision)`` or ``secondary (KID)``), the
        ``precision`` block, the ``kid`` block when the fallback ran, and
        ``kid_comparator_gain_positive``.
    """
    p_f = np.asarray(precision_delta_f, dtype=np.float64)
    p_h = np.asarray(precision_delta_h, dtype=np.float64)
    k_f = -np.asarray(kid_delta_f, dtype=np.float64)
    k_h = -np.asarray(kid_delta_h, dtype=np.float64)
    primary = classify(p_f, p_h)
    record: dict[str, Any] = {
        "precision": _block("precision", p_f, p_h, primary),
        "kid_comparator_gain_positive": _comparator_positive(k_f),
    }
    if primary != NOT_EVALUABLE:
        record.update(outcome=primary, label="primary (precision)")
    else:
        secondary = classify(k_f, k_h)
        outcome = NOT_EVALUABLE_BOTH if secondary == NOT_EVALUABLE else secondary
        record["kid"] = _block("kid", k_f, k_h, outcome)
        record.update(outcome=outcome, label="secondary (KID)")
    record["text"] = OUTCOME_TEXT[record["outcome"]]
    return record


def sign_summary(deltas: Sequence[float]) -> dict[str, Any]:
    """Mean and sign agreement of per-seed deltas (the R5 sensitivity rows, H only).

    Parameters
    ----------
    deltas : Sequence[float]
        Per-seed ``Δ_H``.

    Returns
    -------
    dict[str, Any]
        ``deltas``, ``mean``, ``all_positive``, ``all_negative`` and ``signs_agree``.
    """
    values = np.asarray(deltas, dtype=np.float64)
    positive = bool(np.all(values > 0.0))
    negative = bool(np.all(values < 0.0))
    return {
        "deltas": [float(v) for v in values],
        "mean": float(values.mean()),
        "all_positive": positive,
        "all_negative": negative,
        "signs_agree": positive or negative,
    }


# --------------------------------------------------------------------------------------------
# Reproduction anchors (ticket step 7)
# --------------------------------------------------------------------------------------------


def anchor_features(local: np.ndarray, cached: np.ndarray) -> dict[str, float]:
    """Anchor (a): local reference features against the production cache (reported, no gate).

    Parameters
    ----------
    local, cached : numpy.ndarray
        ``(M, D)`` feature matrices in the same row order.

    Returns
    -------
    dict[str, float]
        ``max_abs_diff`` and ``min_cosine`` (row-wise).

    Raises
    ------
    HeldoutFidelityError
        If the shapes differ.
    """
    a = np.asarray(local, dtype=np.float64)
    b = np.asarray(cached, dtype=np.float64)
    if a.shape != b.shape:
        raise HeldoutFidelityError(f"anchor (a): shapes {a.shape} and {b.shape} differ")
    cosine = (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return {
        "max_abs_diff": float(np.max(np.abs(a - b))),
        "min_cosine": float(np.min(cosine)),
        "n_rows": int(a.shape[0]),
    }


def anchor_metrics(local: Mapping[str, Any], stored: Mapping[str, Any]) -> dict[str, Any]:
    """Anchor (b): local F-versus-full-ref metrics against the stored ``final.json`` block.

    Parameters
    ----------
    local : Mapping[str, Any]
        :func:`set_metrics` of F against the full ``ref``.
    stored : Mapping[str, Any]
        The ``inception`` block of the run's ``final.json``.

    Returns
    -------
    dict[str, Any]
        Per metric the local value, the stored value, the deviation and ``ok``; KID with its
        stored interval; ``passed`` and ``worst_deviation``.
    """
    rows: dict[str, Any] = {}
    for name in ("precision", "recall", "density", "coverage"):
        deviation = abs(float(local[name]) - float(stored[name]))
        rows[name] = {
            "local": float(local[name]),
            "stored": float(stored[name]),
            "abs_diff": deviation,
            "ok": deviation <= ANCHOR_TOLERANCE,
        }
    kid = float(local["kid"])
    low, high = float(stored["kid_ci_low"]), float(stored["kid_ci_high"])
    rows["kid"] = {
        "local": kid,
        "stored": float(stored["kid"]),
        "stored_ci": [low, high],
        "abs_diff": abs(kid - float(stored["kid"])),
        "ok": low <= kid <= high,
    }
    passed = all(bool(row["ok"]) for row in rows.values())
    worst = max(rows[name]["abs_diff"] for name in ("precision", "recall", "density", "coverage"))
    return {"metrics": rows, "passed": passed, "worst_deviation": float(worst)}
