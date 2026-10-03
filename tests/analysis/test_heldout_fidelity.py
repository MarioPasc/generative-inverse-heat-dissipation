"""Tests of the held-out-seed fidelity analysis (T7.5): references, per-sample metrics, reading."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ihdm.analysis.heldout_fidelity import (
    GENERALISES,
    NOT_EVALUABLE,
    NOT_EVALUABLE_BOTH,
    NOT_SHOWN,
    OUTCOME_TEXT,
    PARTLY_TIED,
    HeldoutFidelityError,
    anchor_features,
    anchor_metrics,
    bootstrap_mean_ci,
    bootstrap_replicates,
    classify,
    contrast,
    expected_heldout_seeds,
    per_sample_prdc,
    read_rule,
    select_references,
    sign_summary,
)
from ihdm.cli import heldout_fidelity as cli
from ihdm.metrics.inception import recall_coverage
from ihdm.stats import paired_delta

# --------------------------------------------------------------------------------------------
# Synthetic datasets
# --------------------------------------------------------------------------------------------

N_SLICES = 10


def _mri(n_train: int = 8, n_ref: int = 4, n_seed: int = 2) -> tuple[pd.DataFrame, dict]:
    """``n_train + n_ref`` subjects × 10 slices; the last ``n_seed`` ref subjects are seeds."""
    subjects = [f"S{i:02d}" for i in range(n_train + n_ref)]
    train_s, ref_s = subjects[:n_train], subjects[n_train:]
    seed_s = ref_s[n_ref - n_seed:]
    rows = [(s, k) for s in subjects for k in range(N_SLICES)]
    index = pd.DataFrame(
        {"idx": np.arange(len(rows)), "subject": [r[0] for r in rows],
         "slice": [r[1] for r in rows]}
    )
    index["split"] = np.where(index["subject"].isin(train_s), "train",
                              np.where(index["subject"].isin(seed_s), "seed", "ref"))
    splits = {
        "train": index.index[index["subject"].isin(train_s)].tolist(),
        "ref": index.index[index["subject"].isin(ref_s)].tolist(),
        "seed": index.index[index["subject"].isin(seed_s)].tolist(),
        "seed_subjects": seed_s,
    }
    return index, splits


def _photo(n_train: int = 20, n_ref: int = 10, n_seed: int = 3) -> tuple[pd.DataFrame, dict]:
    """One image per subject; the first ``n_seed`` ref images are seeds."""
    n = n_train + n_ref
    index = pd.DataFrame({"idx": np.arange(n), "subject": [f"img_{i:03d}" for i in range(n)],
                          "slice": np.zeros(n, dtype=int)})
    ref = list(range(n_train, n))
    seed = ref[:n_seed]
    splits = {"train": list(range(n_train)), "ref": ref, "seed": seed,
              "seed_subjects": [f"img_{i:03d}" for i in seed]}
    return index, splits


# --------------------------------------------------------------------------------------------
# R⁻ and R5
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dataset", ["ixi", "oasis1"])
def test_r_minus_mri_removes_every_slice_of_the_seed_subjects(dataset: str) -> None:
    index, splits = _mri()
    refs = select_references(index, splits, dataset)
    assert refs.ref_idx.size == 40
    assert refs.r_minus_idx.size == (4 - 2) * N_SLICES
    subjects = set(index["subject"].to_numpy()[refs.r_minus_idx])
    assert subjects.isdisjoint(splits["seed_subjects"])
    assert np.all(np.isin(refs.r_minus_idx, splits["ref"]))
    np.testing.assert_array_equal(refs.ref_idx[refs.r_minus_rows], refs.r_minus_idx)


def test_r5_is_one_slice_five_image_per_train_subject() -> None:
    index, splits = _mri()
    refs = select_references(index, splits, "ixi")
    assert refs.r5_idx is not None and refs.r5_idx.size == 8
    assert np.all(index["slice"].to_numpy()[refs.r5_idx] == 5)
    assert np.all(np.isin(refs.r5_idx, splits["train"]))
    assert len(set(index["subject"].to_numpy()[refs.r5_idx])) == 8


@pytest.mark.parametrize("dataset", ["lsun_church", "lsun_bedroom"])
def test_r_minus_photographs_removes_the_seed_images(dataset: str) -> None:
    index, splits = _photo()
    refs = select_references(index, splits, dataset)
    assert refs.r_minus_idx.size == 10 - 3
    assert not np.isin(refs.r_minus_idx, splits["seed"]).any()
    assert refs.r5_idx is None


def test_expected_heldout_seeds() -> None:
    index, splits = _mri()
    seeds = expected_heldout_seeds(index, splits, "ixi")
    assert seeds.size == 2
    assert np.all(index["slice"].to_numpy()[seeds] == 5)
    assert set(index["subject"].to_numpy()[seeds]) == set(splits["seed_subjects"])
    index_p, splits_p = _photo()
    np.testing.assert_array_equal(expected_heldout_seeds(index_p, splits_p, "lsun_church"),
                                  sorted(splits_p["seed"]))


def test_seed_outside_ref_is_refused() -> None:
    index, splits = _photo()
    splits = dict(splits, seed=[0, 1, 2], seed_subjects=["img_000", "img_001", "img_002"])
    with pytest.raises(HeldoutFidelityError, match="not inside the ref"):
        select_references(index, splits, "lsun_church")


def test_unknown_dataset_and_missing_slice_five_are_refused() -> None:
    index, splits = _mri()
    with pytest.raises(HeldoutFidelityError, match="unknown dataset"):
        select_references(index, splits, "cifar")
    index = index.copy()
    index.loc[5, "slice"] = 4  # subject S00 loses its slice-5 image
    with pytest.raises(HeldoutFidelityError, match="R5"):
        select_references(index, splits, "ixi")


# --------------------------------------------------------------------------------------------
# Per-sample precision and density
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("n_fake", "n_real", "dim", "seed"),
                         [(60, 40, 16, 0), (7, 12, 3, 1), (200, 50, 32, 2)])
def test_per_sample_reproduces_recall_coverage(n_fake: int, n_real: int, dim: int,
                                               seed: int) -> None:
    rng = np.random.default_rng(seed)
    real = rng.normal(size=(n_real, dim))
    fake = rng.normal(loc=0.3, size=(n_fake, dim))
    per = per_sample_prdc(fake, real, k=5)
    expected = recall_coverage(fake, real, k=5)
    assert per.precision.shape == (n_fake,) and per.density.shape == (n_fake,)
    assert set(np.unique(per.precision)) <= {0.0, 1.0}
    np.testing.assert_allclose(per.precision.mean(), expected["precision"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(per.density.mean(), expected["density"], rtol=0, atol=1e-12)


def test_per_sample_with_duplicated_samples() -> None:
    rng = np.random.default_rng(3)
    real = rng.normal(size=(30, 8))
    fake = np.repeat(rng.normal(size=(6, 8)), 5, axis=0)
    per = per_sample_prdc(fake, real)
    expected = recall_coverage(fake, real)
    np.testing.assert_allclose(per.density.mean(), expected["density"], rtol=0, atol=1e-12)


# --------------------------------------------------------------------------------------------
# Within-run bootstrap
# --------------------------------------------------------------------------------------------


def test_cluster_bootstrap_resamples_whole_seeds() -> None:
    # two seeds of 50 samples, each seed constant: any replicate built from whole seeds is the
    # mean of a multiset of {0, 1} of size 2, i.e. 0, 0.5 or 1
    values = np.r_[np.zeros(50), np.ones(50)]
    clusters = np.repeat([0, 1], 50)
    replicates = bootstrap_replicates(values, clusters, n_boot=500)
    assert set(np.round(replicates, 12)) <= {0.0, 0.5, 1.0}
    by_sample = bootstrap_replicates(values, None, n_boot=500)
    assert not set(np.round(by_sample, 12)) <= {0.0, 0.5, 1.0}


def test_cluster_bootstrap_matches_manual_draws_with_unequal_sizes() -> None:
    values = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 5.0])
    clusters = np.array([7, 7, 7, 9, 9, 4])  # labels need not be contiguous
    replicates = bootstrap_replicates(values, clusters, n_boot=50, rng_seed=0)
    # unique labels sorted: 4 -> [5], 7 -> [1, 2, 3], 9 -> [10, 20]
    sums, sizes = np.array([5.0, 6.0, 30.0]), np.array([1.0, 3.0, 2.0])
    draws = np.random.default_rng(0).integers(0, 3, size=(50, 3))
    np.testing.assert_allclose(replicates, sums[draws].sum(1) / sizes[draws].sum(1), rtol=1e-12)


def test_bootstrap_interval_brackets_the_mean_and_rejects_bad_input() -> None:
    rng = np.random.default_rng(0)
    values = rng.binomial(1, 0.4, size=2000).astype(float)
    low, high = bootstrap_mean_ci(values, np.repeat(np.arange(40), 50))
    assert low <= values.mean() <= high
    with pytest.raises(HeldoutFidelityError):
        bootstrap_replicates(np.array([]))
    with pytest.raises(HeldoutFidelityError):
        bootstrap_replicates(np.array([1.0, np.nan]))
    with pytest.raises(HeldoutFidelityError):
        bootstrap_replicates(np.ones(4), np.ones(3))


# --------------------------------------------------------------------------------------------
# The reading rule
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("gains_f", "gains_h", "outcome"),
    [
        ([0.10, 0.08, 0.09], [0.06, 0.05, 0.04], GENERALISES),     # G_H = 0.05 >= 0.045
        ([0.10, 0.08, 0.09], [0.045, 0.045, 0.045], GENERALISES),  # boundary G_H = 0.5 G_F
        ([0.10, 0.08, 0.09], [0.02, 0.01, 0.03], PARTLY_TIED),
        ([0.10, 0.08, 0.09], [0.06, -0.01, 0.05], NOT_SHOWN),
        ([0.10, 0.08, 0.09], [0.0, 0.05, 0.05], NOT_SHOWN),        # Δ_H = 0 is not > 0
        ([0.10, -0.01, 0.09], [0.06, 0.05, 0.04], NOT_EVALUABLE),  # not all Δ_F > 0
        ([-0.10, -0.08, -0.09], [0.06, 0.05, 0.04], NOT_EVALUABLE),
        ([0.10, 0.08], [0.06, 0.05], GENERALISES),                 # OASIS-1: 2 run seeds
    ],
)
def test_classify_outcomes(gains_f: list[float], gains_h: list[float], outcome: str) -> None:
    assert classify(gains_f, gains_h) == outcome


def test_classify_rejects_unpaired_gains() -> None:
    with pytest.raises(HeldoutFidelityError):
        classify([0.1, 0.2], [0.1])


@pytest.mark.parametrize(
    ("p_f", "p_h", "outcome"),
    [
        ([0.10, 0.08, 0.09], [0.06, 0.05, 0.04], GENERALISES),
        ([0.10, 0.08, 0.09], [0.02, 0.01, 0.03], PARTLY_TIED),
        ([0.10, 0.08, 0.09], [0.06, -0.01, 0.05], NOT_SHOWN),
    ],
)
def test_read_rule_precision_outcomes_are_primary(p_f: list[float], p_h: list[float],
                                                  outcome: str) -> None:
    record = read_rule(p_f, p_h, [-0.02, -0.02, -0.02], [-0.01, -0.01, -0.01])
    assert record["outcome"] == outcome
    assert record["label"] == "primary (precision)"
    assert record["text"] == OUTCOME_TEXT[outcome]
    assert "kid" not in record
    assert record["kid_comparator_gain_positive"] is True


def test_read_rule_falls_back_to_kid_with_gain_a0_minus_a3() -> None:
    # precision not evaluable; KID falls with A3 (Δ < 0, gain > 0) on F and on H
    record = read_rule([0.01, -0.02, 0.03], [0.0, 0.0, 0.0],
                       [-0.030, -0.028, -0.032], [-0.020, -0.018, -0.016])
    assert record["precision"]["outcome"] == NOT_EVALUABLE
    assert record["label"] == "secondary (KID)"
    assert record["outcome"] == GENERALISES
    np.testing.assert_allclose(record["kid"]["G_F"], 0.030, rtol=1e-12)
    np.testing.assert_allclose(record["kid"]["G_H"], 0.018, rtol=1e-12)


def test_read_rule_kid_fallback_partly_tied() -> None:
    record = read_rule([-0.01, -0.02, -0.03], [0.0, 0.0, 0.0],
                       [-0.030, -0.028, -0.032], [-0.005, -0.004, -0.006])
    assert record["outcome"] == PARTLY_TIED and record["label"] == "secondary (KID)"


def test_read_rule_not_evaluable_on_precision_or_kid() -> None:
    # main's amendment of 2026-10-03: rule 1 is applied to KID too. KID rises with A3 on F, so
    # the comparator fails, even though every held-out KID gain is positive
    record = read_rule([0.01, -0.02, 0.03], [0.05, 0.05, 0.05],
                       [0.010, 0.012, 0.011], [-0.020, -0.018, -0.016])
    assert record["outcome"] == NOT_EVALUABLE_BOTH
    assert record["text"] == "not evaluable on precision or KID"
    assert record["kid_comparator_gain_positive"] is False


def test_sign_summary() -> None:
    assert sign_summary([0.1, 0.2, 0.05])["signs_agree"] is True
    assert sign_summary([-0.1, -0.2])["all_negative"] is True
    mixed = sign_summary([0.1, -0.2, 0.3])
    assert mixed["signs_agree"] is False
    np.testing.assert_allclose(mixed["mean"], 0.2 / 3, rtol=1e-12)


# --------------------------------------------------------------------------------------------
# Contrasts and anchors
# --------------------------------------------------------------------------------------------


def test_contrast_uses_paired_delta_and_the_exact_permutation() -> None:
    arm = {1: 0.60, 2: 0.65, 3: 0.62}
    a0 = {1: 0.55, 2: 0.50, 3: 0.57}
    record = contrast(arm, a0)
    expected = paired_delta(arm, a0)
    np.testing.assert_allclose(record["deltas"], expected.deltas, rtol=1e-12)
    np.testing.assert_allclose([record["ci_low"], record["ci_high"]],
                               [expected.interval.low, expected.interval.high], rtol=1e-12)
    np.testing.assert_allclose(record["p_min"], 0.1, rtol=1e-12)
    assert record["n_assignments"] == 20
    two = contrast({1: 0.6, 2: 0.7}, {1: 0.5, 2: 0.4})
    np.testing.assert_allclose(two["p_min"], 1 / 3, rtol=1e-12)


def test_anchor_metrics_pass_and_fail() -> None:
    stored = {"precision": 0.574, "recall": 0.2575, "density": 0.3122, "coverage": 0.42625,
              "kid": 0.0464, "kid_ci_low": 0.0440, "kid_ci_high": 0.0488}
    local = {"precision": 0.570, "recall": 0.26, "density": 0.31, "coverage": 0.43, "kid": 0.046}
    assert anchor_metrics(local, stored)["passed"] is True
    assert anchor_metrics(dict(local, precision=0.55), stored)["passed"] is False
    assert anchor_metrics(dict(local, kid=0.05), stored)["passed"] is False


def test_anchor_features_reports_difference_and_cosine() -> None:
    a = np.eye(3) + 1.0
    record = anchor_features(a, a * 2.0)
    np.testing.assert_allclose(record["min_cosine"], 1.0, rtol=1e-12)
    np.testing.assert_allclose(record["max_abs_diff"], 2.0, rtol=1e-12)
    with pytest.raises(HeldoutFidelityError):
        anchor_features(a, a[:2])


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def _data_root(tmp_path, datasets=("ixi", "oasis1", "lsun_church", "lsun_bedroom")):
    root = tmp_path / "data"
    for name in datasets:
        (root / name).mkdir(parents=True)
        for file in cli.DATASET_FILES:
            (root / name / file).write_text("")
    return root


def test_cli_exit_code_on_missing_tar(tmp_path, capsys) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    code = cli.main(["--eval-dir", str(eval_dir), "--data-root", str(_data_root(tmp_path)),
                     "--work", str(tmp_path / "work"), "--out", str(tmp_path / "out")])
    assert code == cli.EXIT_MISSING == 2
    assert "ixi_A0_s1_amp-fp16.tar" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_cli_exit_code_on_missing_dataset(tmp_path) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    for rid in ("ixi_A0_s1", "lsun_church_A3_s1"):
        (eval_dir / f"{rid}_amp-fp16.tar").write_bytes(b"")
    code = cli.main(["--eval-dir", str(eval_dir),
                     "--data-root", str(_data_root(tmp_path, ("ixi",))),
                     "--work", str(tmp_path / "work"), "--out", str(tmp_path / "out"),
                     "--runs", "ixi_A0_s1", "--anchors-only"])
    assert code == cli.EXIT_MISSING


def test_cli_unknown_run_is_an_analysis_failure(tmp_path) -> None:
    code = cli.main(["--eval-dir", str(tmp_path), "--data-root", str(tmp_path),
                     "--work", str(tmp_path / "w"), "--out", str(tmp_path / "o"),
                     "--runs", "ixi_A2_s1"])
    assert code == cli.EXIT_FAIL


def test_selected_runs_are_the_24() -> None:
    runs = cli.selected_runs(None)
    assert len(runs) == 24
    assert {r[2] for r in runs} == {"A0", "A1", "A3"}
    assert sum(r[1] == "ixi" for r in runs) == 8
    assert sum(r[1] == "oasis1" for r in runs) == 4
