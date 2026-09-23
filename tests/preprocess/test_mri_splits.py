"""Tests of the quality gate, the seeded subject selection and the subject-level splits."""

from __future__ import annotations

import numpy as np
import pytest

from ihdm.cli.preprocess_mri import _build_index, _select_subjects
from ihdm.data.format import split_by_subject
from ihdm.preprocess.errors import PreprocessError
from ihdm.preprocess.mri import SLICE_Z_INDICES
from ihdm.preprocess.qc import SubjectRecord, apply_gate

N_SUBJECTS = 400
N_SLICES = len(SLICE_Z_INDICES)


def make_records(
    metrics: list[float], hit_max: set[str] | None = None
) -> list[SubjectRecord]:
    """Build one :class:`SubjectRecord` per metric value, named ``S000``, ``S001``, ..."""
    hit_max = hit_max or set()
    return [
        SubjectRecord(
            subject=f"S{i:03d}",
            source=f"raw/S{i:03d}.nii.gz",
            final_metric=metric,
            iterations=300 if f"S{i:03d}" in hit_max else 80,
            stop_condition="converged",
            hit_max_iterations=f"S{i:03d}" in hit_max,
        )
        for i, metric in enumerate(metrics)
    ]


def test_gate_fails_only_the_metric_outliers() -> None:
    """A subject whose metric exceeds median + 3 MAD fails; the rest pass."""
    metrics = [-0.20, -0.19, -0.21, -0.20, -0.195, -0.205, -0.02]
    gate = apply_gate(make_records(metrics))

    assert gate.median == pytest.approx(-0.20)
    assert gate.mad == pytest.approx(0.005)
    assert gate.threshold == pytest.approx(-0.185)
    assert gate.failed_metric == ("S006",)
    assert gate.failed == ("S006",)
    assert "S006" not in gate.passing
    assert len(gate.passing) == 6


def test_gate_records_the_iteration_cap_without_excluding() -> None:
    """Reaching the iteration cap is a recorded diagnostic, not a failure.

    Contract amended by the orchestrator on 2026-09-22: measured on this data the metric
    is already converged when the finest level exhausts its iterations, so excluding those
    subjects would discard good registrations (ticket log §2).
    """
    metrics = [-0.20] * 6 + [-0.21]
    gate = apply_gate(make_records(metrics, hit_max={"S002"}))

    assert gate.capped == ("S002",)
    assert gate.failed == ()
    assert "S002" in gate.passing
    payload = gate.to_json()
    assert payload["n_hit_max_iterations"] == 1
    assert payload["hit_max_iterations_rate"] == pytest.approx(1 / 7)
    assert "NOT a failure" in str(payload["rule"])


def test_gate_excludes_subjects_flagged_by_eye() -> None:
    """Visual inspection of the registration sheet can exclude a subject explicitly."""
    metrics = [-0.20] * 7
    gate = apply_gate(make_records(metrics), exclude=["S003"])

    assert gate.excluded_by_eye == ("S003",)
    assert gate.failed == ("S003",)
    assert "S003" not in gate.passing
    assert len(gate.passing) == 6


def test_gate_rejects_an_unknown_exclusion() -> None:
    """A typo in ``--exclude`` fails loudly instead of silently excluding nothing."""
    with pytest.raises(PreprocessError):
        apply_gate(make_records([-0.20] * 3), exclude=["NOT_A_SUBJECT"])


def test_gate_reports_the_scaled_mad_threshold_without_applying_it() -> None:
    """The consistency-scaled threshold is looser and is reported, never used to select."""
    metrics = [-0.20, -0.19, -0.21, -0.20, -0.195, -0.205, -0.02]
    gate = apply_gate(make_records(metrics))
    assert gate.threshold_scaled > gate.threshold
    assert gate.n_failed_scaled <= len(gate.failed_metric)
    assert "median + 3 * MAD" in str(gate.to_json()["rule"])


def test_gate_needs_at_least_one_subject() -> None:
    """An empty cohort is a programming error, not an empty result."""
    with pytest.raises(PreprocessError):
        apply_gate([])


def test_subject_selection_is_seeded_and_sorted() -> None:
    """The draw is reproducible and returns sorted ids, whatever the input order."""
    passing = tuple(f"S{i:03d}" for i in range(500))
    first = _select_subjects(passing, N_SUBJECTS)
    second = _select_subjects(tuple(reversed(passing)), N_SUBJECTS)

    assert first == second, "selection must depend only on the sorted passing set"
    assert len(first) == N_SUBJECTS
    assert first == sorted(first)
    assert set(first) <= set(passing)
    expected = sorted(np.random.default_rng(2026).permutation(sorted(passing))[:N_SUBJECTS])
    assert first == expected


def test_subject_selection_takes_everything_when_too_few_pass() -> None:
    """A short cohort yields all its passing subjects rather than failing."""
    passing = tuple(f"S{i:03d}" for i in range(11))
    assert _select_subjects(passing, N_SUBJECTS) == sorted(passing)


def test_splits_are_disjoint_by_subject() -> None:
    """No subject's slices straddle train and ref, and the counts are 320/80/40."""
    subjects = [f"SUB{s:03d}" for s in range(N_SUBJECTS) for _ in range(N_SLICES)]
    splits = split_by_subject(subjects, rng_seed=2026)

    assert len(splits["train_subjects"]) == 320
    assert len(splits["ref_subjects"]) == 80
    assert len(splits["seed_subjects"]) == 40
    assert set(splits["train_subjects"]).isdisjoint(splits["ref_subjects"])
    assert set(splits["seed_subjects"]) <= set(splits["ref_subjects"])

    train, ref = set(splits["train"]), set(splits["ref"])
    assert train.isdisjoint(ref)
    assert train | ref == set(range(N_SUBJECTS * N_SLICES))
    assert len(train) == 320 * N_SLICES
    assert len(ref) == 80 * N_SLICES
    assert set(splits["seed"]) <= ref

    by_subject: dict[str, set[str]] = {}
    for position, subject in enumerate(subjects):
        by_subject.setdefault(subject, set()).add("train" if position in train else "ref")
    assert all(len(sides) == 1 for sides in by_subject.values())


def test_index_labels_seed_subjects_separately() -> None:
    """``index.csv`` carries the finest label, so ``seed`` rows are not labelled ``ref``."""
    subjects = [f"SUB{s:03d}" for s in range(N_SUBJECTS) for _ in range(N_SLICES)]
    slices = [position for _ in range(N_SUBJECTS) for position in range(N_SLICES)]
    rows = [
        {"subject": s, "slice": sl, "z_mm": float(sl), "source": f"raw/{s}.nii.gz"}
        for s, sl in zip(subjects, slices, strict=True)
    ]
    splits = split_by_subject(subjects, rng_seed=2026)

    index = _build_index(rows, splits)

    assert list(index.columns) == ["idx", "subject", "slice", "z_mm", "source", "split"]
    counts = index["split"].value_counts().to_dict()
    assert counts == {"train": 3200, "ref": 400, "seed": 400}
    assert (index["idx"].to_numpy() == np.arange(len(index))).all()
    # Every row of a subject carries one and the same label.
    assert index.groupby("subject")["split"].nunique().max() == 1


def test_index_labels_respect_a_stratified_split() -> None:
    """T1.5: the index still carries the finest label when the split is stratified by site.

    Mirrors the call ``ihdm.cli.preprocess_mri.main`` makes for cohort ``ixi``: build a
    ``sites``-shaped ``{subject: label}`` map and pass it as ``strata`` to
    ``split_by_subject``, then check ``_build_index`` labels every row correctly and that
    every split's site mix sums back to its subject count.
    """
    sites = {
        f"SUB{s:03d}": ("Guys" if s % 3 == 0 else "HH" if s % 3 == 1 else "IOP")
        for s in range(N_SUBJECTS)
    }
    subjects = [f"SUB{s:03d}" for s in range(N_SUBJECTS) for _ in range(N_SLICES)]
    slices = [position for _ in range(N_SUBJECTS) for position in range(N_SLICES)]
    rows = [
        {"subject": s, "slice": sl, "z_mm": float(sl), "source": f"raw/{s}.nii.gz"}
        for s, sl in zip(subjects, slices, strict=True)
    ]
    splits = split_by_subject(subjects, rng_seed=2026, strata=sites)

    index = _build_index(rows, splits)

    assert list(index.columns) == ["idx", "subject", "slice", "z_mm", "source", "split"]
    counts = index["split"].value_counts().to_dict()
    assert counts == {"train": 3200, "ref": 400, "seed": 400}
    assert index.groupby("subject")["split"].nunique().max() == 1

    assert "strata_mix" in splits
    expected_totals = {"train": 320, "ref": 80, "seed": 40}
    for split_name, expected_total in expected_totals.items():
        assert sum(splits["strata_mix"][split_name].values()) == expected_total
