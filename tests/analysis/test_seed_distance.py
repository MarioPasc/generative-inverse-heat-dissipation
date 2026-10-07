"""Tests of the seed distances of F2(b) (T8.8): q_seed, q_train, own-subject exclusion, D_ref."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ihdm.analysis import seed_distance as sd
from ihdm.analysis.seed_distance import (
    QUANTITIES,
    RunDistances,
    SeedDistanceError,
    compute_run,
    config_distances,
    load_or_compute,
    nearest_training,
    own_seed_distances,
    reference_distances,
)

SIZE = 8


def _stack(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.integers(0, 256, (n, SIZE, SIZE), dtype=np.uint8)


def _brute(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """DC-removed Euclidean distances between every image of ``a`` and every image of ``b``."""
    fa = a.reshape(a.shape[0], -1).astype(np.float64) / 255.0
    fb = b.reshape(b.shape[0], -1).astype(np.float64) / 255.0
    fa -= fa.mean(axis=1, keepdims=True)
    fb -= fb.mean(axis=1, keepdims=True)
    return np.sqrt(((fa[:, None, :] - fb[None, :, :]) ** 2).sum(axis=2))


@pytest.fixture(scope="module")
def world() -> dict[str, np.ndarray]:
    """Six training subjects of four images each, and samples near chosen training images."""
    rng = np.random.default_rng(3)
    train = _stack(rng, 24)
    subjects = np.array([f"S{k // 4}" for k in range(24)], dtype=object)
    seed_rows = np.array([0, 5, 9, 13, 18, 23, 2])
    # Samples: a small perturbation of their own seed (copies), or of another subject's image.
    near = np.array([0, 5, 10, 1, 18, 23, 7])  # rows 3 and 6 copy another subject
    noise = rng.integers(-6, 7, (near.size, SIZE, SIZE))
    samples = np.clip(train[near].astype(int) + noise, 0, 255).astype(np.uint8)
    heldout_seeds = _stack(rng, 3)
    held = np.repeat(heldout_seeds, 4, axis=0).astype(int) + rng.integers(-20, 21, (12, SIZE, SIZE))
    return {"train": train, "subjects": subjects, "seed_rows": seed_rows, "samples": samples,
            "heldout_seeds": heldout_seeds, "heldout": np.clip(held, 0, 255).astype(np.uint8),
            "reference": _stack(rng, 9)}


def test_own_seed_distances_match_brute_force(world) -> None:
    held, seeds = world["heldout"], world["heldout_seeds"]
    got = own_seed_distances(held, seeds)
    want = np.array([_brute(held[i:i + 1], seeds[i // 4:i // 4 + 1])[0, 0] for i in range(12)])
    np.testing.assert_allclose(got, want, rtol=1e-5)


def test_own_seed_distances_rejects_uneven_sets(world) -> None:
    with pytest.raises(SeedDistanceError, match="do not split"):
        own_seed_distances(world["heldout"][:11], world["heldout_seeds"])


def test_nearest_training_excludes_the_own_subject(world) -> None:
    train, subjects, rows = world["train"], world["subjects"], world["seed_rows"]
    samples = world["samples"]
    result = nearest_training(samples, train, subjects[rows], subjects)
    dist = _brute(samples, train)
    np.testing.assert_allclose(result.d_all, dist.min(axis=1), rtol=1e-5)
    np.testing.assert_array_equal(result.nn_all, dist.argmin(axis=1))
    other = np.where(subjects[None, :] == subjects[rows][:, None], np.inf, dist)
    np.testing.assert_allclose(result.d_other, other.min(axis=1), rtol=1e-5)
    np.testing.assert_array_equal(result.nn_other, other.argmin(axis=1))
    assert not np.any(subjects[result.nn_other] == subjects[rows])
    # Samples 3 and 6 copy another subject's image: their nearest image is not their own.
    np.testing.assert_array_equal(result.own, [True, True, True, False, True, True, False])
    np.testing.assert_allclose(result.d_other[[3, 6]], result.d_all[[3, 6]])
    assert np.all(result.d_other >= result.d_all - 1e-6)


def test_nearest_training_without_groups_and_bad_groups(world) -> None:
    train, subjects = world["train"], world["subjects"]
    result = nearest_training(world["heldout"], train)
    np.testing.assert_array_equal(result.d_other, result.d_all)
    assert not result.own.any()
    with pytest.raises(SeedDistanceError, match="both"):
        nearest_training(world["heldout"], train, query_groups=subjects[:12])
    with pytest.raises(SeedDistanceError, match="owns every training image"):
        nearest_training(world["samples"][:1], train[:4], subjects[:1], subjects[:4])


def test_compute_run_normalises_by_d_ref(world) -> None:
    train, subjects, rows = world["train"], world["subjects"], world["seed_rows"]
    d_real = reference_distances(world["reference"], train)
    np.testing.assert_allclose(d_real, _brute(world["reference"], train).min(axis=1), rtol=1e-5)
    d_ref = float(np.median(d_real))
    run = compute_run("ixi_A3_s1", world["samples"], rows, world["heldout"],
                      world["heldout_seeds"], train, subjects, d_ref)
    dist = _brute(world["samples"], train)
    f_seed = np.array([dist[i, r] for i, r in enumerate(rows)])
    np.testing.assert_allclose(run.q("F_seed"), f_seed / d_ref, rtol=1e-5)
    other = np.where(subjects[None, :] == subjects[rows][:, None], np.inf, dist)
    np.testing.assert_allclose(run.q("F_train"), other.min(axis=1) / d_ref, rtol=1e-5)
    h_train = _brute(world["heldout"], train).min(axis=1)
    np.testing.assert_allclose(run.q("H_train"), h_train / d_ref, rtol=1e-5)
    np.testing.assert_allclose(run.m_ratio, np.median(dist.min(axis=1)) / d_ref, rtol=1e-5)
    np.testing.assert_allclose(run.seed_nn, 5 / 7)
    np.testing.assert_allclose(run.medians(), [np.median(run.q(n)) for n in QUANTITIES])
    # The own seed is a training image of the own subject, so q_seed >= the nearest overall.
    assert np.all(run.f_seed >= run.f_nearest - 1e-5)
    with pytest.raises(SeedDistanceError, match="unknown quantity"):
        run.q("F_other")


@pytest.mark.parametrize("d_ref, rows, message", [
    (0.0, None, "D_ref must be positive"),
    (1.0, np.array([0, 1, 2, 3, 4, 5, 99]), "do not index"),
])
def test_compute_run_rejects_bad_inputs(world, d_ref, rows, message) -> None:
    rows = world["seed_rows"] if rows is None else rows
    with pytest.raises(SeedDistanceError, match=message):
        compute_run("r", world["samples"], rows, world["heldout"], world["heldout_seeds"],
                    world["train"], world["subjects"], d_ref)


def _run(run_id: str, scale: float, n: int = 5) -> RunDistances:
    base = np.arange(1, n + 1, dtype=np.float32) * scale
    return RunDistances(run_id, 2.0, base, base + 1, base - 0.5, np.array([True] * n),
                        base * 2, base + 2)


def test_config_distances_means_and_pooled_quartiles() -> None:
    runs = [_run("ixi_A0_s1", 1.0), _run("ixi_A0_s2", 2.0)]
    cfg = config_distances("ixi", "A0", runs)
    assert cfg.n == 2 and cfg.run_ids == ("ixi_A0_s1", "ixi_A0_s2")
    np.testing.assert_allclose(cfg.medians[:, 0], [1.5, 3.0])  # median of 1..5 (x1, x2) / 2
    np.testing.assert_allclose(cfg.mean()[0], 2.25)
    pooled = np.concatenate([r.q("H_seed") for r in runs])
    np.testing.assert_allclose(cfg.quartiles[2], np.percentile(pooled, [25, 75]))
    np.testing.assert_allclose(cfg.seed_nn, [1.0, 1.0])
    with pytest.raises(SeedDistanceError, match="no run"):
        config_distances("ixi", "A0", [])


def test_cache_is_reused_only_for_the_same_inputs(world, tmp_path: Path, monkeypatch) -> None:
    args = (world["samples"], world["seed_rows"], world["heldout"], world["heldout_seeds"],
            world["train"], np.arange(24), world["subjects"], 3.0)
    first = load_or_compute(tmp_path, "ixi_A0_s1", *args)
    assert (tmp_path / "ixi_A0_s1.npz").is_file()

    def boom(*_: object, **__: object) -> None:
        raise AssertionError("recomputed although the cache matches")

    monkeypatch.setattr(sd, "compute_run", boom)
    again = load_or_compute(tmp_path, "ixi_A0_s1", *args)
    for name in ("f_seed", "f_train", "f_nearest", "f_own_nn", "h_seed", "h_train"):
        np.testing.assert_array_equal(getattr(first, name), getattr(again, name))
    changed = (*args[:-1], 3.5)
    with pytest.raises(AssertionError, match="recomputed"):
        load_or_compute(tmp_path, "ixi_A0_s1", *changed)
