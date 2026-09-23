"""Tests of ``ihdm.metrics.inception``.

The unit tests exercise the distances on synthetic feature matrices, which needs no weights. The
tests that actually run Inception are marked ``integration``: they download ≈ 90 MB on first use
and are skipped when the weights cannot be fetched.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ihdm.metrics.errors import MetricError
from ihdm.metrics.inception import (
    INCEPTION_FEATURE_DIM,
    _fid_fast,
    _reference_moments,
    bootstrap_inception,
    fid_from_features,
    inception_features,
    inception_weights_path,
    kid_from_features,
    recall_coverage,
    reference_features,
)
from ihdm.paths import data_root

IXI_ROOT = data_root() / "ixi"


# --------------------------------------------------------------------------------------------
# The distances, on synthetic features
# --------------------------------------------------------------------------------------------


def test_fid_of_a_feature_set_against_itself_is_zero():
    """Identical moments give a zero Frechet distance up to the ``sqrtm`` tolerance."""
    rng = np.random.default_rng(0)
    features = rng.normal(size=(40, 32))
    assert fid_from_features(features, features) == pytest.approx(0.0, abs=1e-6)


def test_fid_increases_with_the_mean_shift():
    """Shifting one set away from the other increases the distance monotonically."""
    rng = np.random.default_rng(1)
    reference = rng.normal(size=(60, 32))
    values = [fid_from_features(reference + shift, reference) for shift in (0.0, 0.5, 1.0)]
    assert values[0] < values[1] < values[2]


def test_fast_fid_matches_the_frozen_sqrtm_estimator():
    """The bootstrap's fast path is the same number ``cleanfid.fid.frechet_distance`` returns.

    This is the pin that lets a 200-resample bootstrap run in a minute instead of half an hour:
    ``tr sqrt(S_s S_r)`` through the Gram matrix of ``X_c S_r^{1/2}`` is algebraically identical
    to the dense matrix square root.
    """
    rng = np.random.default_rng(2)
    reference = rng.normal(scale=1.3, size=(80, 48))
    for n_samples in (12, 48, 96):
        samples = rng.normal(size=(n_samples, 48)) + 0.4
        expected = fid_from_features(samples, reference)
        actual = _fid_fast(samples, _reference_moments(reference))
        assert actual == pytest.approx(expected, rel=1e-6, abs=1e-8)


def test_kid_of_two_independent_sets_of_one_distribution_is_near_zero():
    """KID estimates the MMD, which vanishes when the two sets come from one distribution.

    The two sets must be *independent* draws. ``cleanfid.fid.kernel_distance`` removes the
    diagonal from the within-set terms but not from the cross term, which is correct for
    independent sets and leaves a deterministic negative offset of about
    ``-2 (k(x, x) - k(x, y)) / m`` when the same array is passed twice; that offset is a
    property of the estimator, not of the data, and is why the integration test below uses two
    disjoint halves of the reference split.
    """
    rng = np.random.default_rng(3)
    for _ in range(4):
        value = kid_from_features(rng.normal(size=(200, 64)), rng.normal(size=(200, 64)))
        assert abs(value) < 2e-2


def test_kid_is_reproducible_across_calls():
    """The wrapper seeds the subset draw, so the same features give the same number."""
    rng = np.random.default_rng(3)
    features, other = rng.normal(size=(120, 64)), rng.normal(size=(120, 64))
    assert kid_from_features(features, other, rng_seed=0) == kid_from_features(
        features, other, rng_seed=0
    )


def test_kid_does_not_disturb_the_global_numpy_rng():
    """``kernel_distance`` draws from the legacy global RNG; the wrapper restores its state."""
    rng = np.random.default_rng(4)
    features = rng.normal(size=(60, 32))
    np.random.seed(1234)
    before = np.random.random()
    np.random.seed(1234)
    kid_from_features(features, features, num_subsets=5, rng_seed=99)
    assert np.random.random() == before


def test_kid_separates_two_distributions():
    """A shifted set has a clearly positive KID."""
    rng = np.random.default_rng(5)
    reference = rng.normal(size=(200, 32))
    assert kid_from_features(reference + 1.0, reference, rng_seed=0) > kid_from_features(
        reference, reference, rng_seed=0
    )


def test_recall_coverage_of_a_set_against_itself_is_high():
    """Samples drawn from the reference manifold are covered by it."""
    rng = np.random.default_rng(6)
    reference = rng.normal(size=(120, 16))
    values = recall_coverage(reference, reference, k=5)
    assert set(values) >= {"precision", "recall", "density", "coverage", "k"}
    assert values["coverage"] > 0.9
    assert values["precision"] > 0.9
    assert values["k"] == 5.0


def test_recall_coverage_collapses_for_a_distant_sample_set():
    """A set far from the reference manifold has near-zero precision and coverage."""
    rng = np.random.default_rng(7)
    reference = rng.normal(size=(120, 16))
    values = recall_coverage(reference + 50.0, reference, k=5)
    assert values["precision"] < 0.05
    assert values["coverage"] < 0.05


def test_bootstrap_inception_returns_every_key_and_an_interval_around_the_point():
    """The bundled result carries KID, FID, prdc, the two intervals and ``n_reference``."""
    rng = np.random.default_rng(8)
    reference = rng.normal(size=(120, 24))
    samples = rng.normal(size=(80, 24)) + 0.2
    result = bootstrap_inception(samples, reference, k=5, n_boot=25, rng_seed=0)

    record = result.to_json()
    assert set(record) >= {
        "kid", "kid_ci_low", "kid_ci_high", "fid", "fid_ci_low", "fid_ci_high",
        "precision", "recall", "density", "coverage", "k", "n_samples", "n_reference",
        "n_boot", "feature_dim", "weights_path", "notes",
    }
    assert record["n_samples"] == 80
    assert record["n_reference"] == 120
    assert record["n_boot"] == 25
    assert record["fid_ci_low"] <= record["fid_ci_high"]
    assert record["kid_ci_low"] <= record["kid_ci_high"]
    assert all(np.isfinite(record[key]) for key in ("kid", "fid", "recall", "coverage"))
    assert "unbiased" in record["notes"]["headline"]
    assert "resample" in record["notes"]["ci_shift"]


def test_the_fid_bootstrap_interval_sits_above_the_point_estimate():
    """A resampled set holds duplicates, which inflates the FID's finite-sample bias.

    The percentile interval of ``05-metrics.md`` §8 therefore does **not** bracket the point
    estimate for FID: it measures the spread, not the location. The behaviour is recorded here so
    that a reader of ``final.json`` who finds ``fid < fid_ci_low`` knows it is expected. KID,
    being the unbiased estimator, is the metric whose location the interval is about.
    """
    rng = np.random.default_rng(9)
    reference = rng.normal(size=(150, 24))
    samples = rng.normal(size=(100, 24)) + 0.2
    result = bootstrap_inception(samples, reference, k=5, n_boot=40, rng_seed=0)
    assert result.fid_ci_low > result.fid
    assert result.kid_ci_low < result.kid_ci_high


@pytest.mark.parametrize(
    ("samples", "reference", "message"),
    [
        (np.zeros((4, 3)), np.zeros((4, 5)), "dimension mismatch"),
        (np.zeros(4), np.zeros((4, 3)), "expected"),
        (np.full((4, 3), np.nan), np.zeros((4, 3)), "non-finite"),
    ],
)
def test_distances_reject_bad_features(samples, reference, message):
    """Malformed feature matrices raise :class:`MetricError`, never a numpy error."""
    with pytest.raises(MetricError, match=message):
        fid_from_features(samples, reference)


def test_inception_weights_path_is_absolute_and_named():
    """T5.1 needs the path to pre-seed the weights on the compute nodes."""
    path = inception_weights_path()
    assert path.is_absolute()
    assert path.name == "inception-2015-12-05.pt"


# --------------------------------------------------------------------------------------------
# The extractor itself
# --------------------------------------------------------------------------------------------


def _weights_available() -> bool:
    """Return whether the torchscript Inception can be built without a download."""
    return inception_weights_path().exists()


requires_weights = pytest.mark.skipif(
    not _weights_available(),
    reason=f"the Inception weights are not at {inception_weights_path()}",
)
requires_ixi = pytest.mark.skipif(
    not (IXI_ROOT / "images.npy").exists(), reason=f"no ixi dataset at {IXI_ROOT}"
)


@pytest.mark.integration
@requires_weights
def test_inception_features_shape_and_determinism():
    """Features are ``(N, 2048)`` and the extractor is deterministic on the same input."""
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, size=(8, 64, 64), dtype=np.uint8)
    first = inception_features(images, device="cpu", batch=4)
    second = inception_features(images, device="cpu", batch=8)
    assert first.shape == (8, INCEPTION_FEATURE_DIM)
    np.testing.assert_allclose(first, second, rtol=1e-5, atol=1e-5)


@pytest.mark.integration
@requires_weights
@requires_ixi
def test_kid_of_two_disjoint_halves_of_the_ixi_reference_is_near_zero():
    """Two halves of the same 64 real images are the same distribution: KID sits at zero.

    This is the licence check of H-METRICS §3 in its cheapest form. FID does not vanish at this
    sample size (it is biased upward by a term of order ``1/N``), which is exactly why
    ``05-metrics.md`` §7 makes KID the headline.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = json.loads((IXI_ROOT / "splits.json").read_text())
    rows = np.asarray(splits["ref"], dtype=np.int64)[:64]
    images = np.load(IXI_ROOT / "images.npy", mmap_mode="r")[rows]

    features = inception_features(np.asarray(images), device=device, batch=32)
    kid = kid_from_features(features[:32], features[32:], rng_seed=0)
    fid = fid_from_features(features[:32], features[32:])

    assert abs(kid) < 0.05, f"KID of two halves of the ref split is {kid}"
    assert np.isfinite(fid)

    shifted = np.asarray(images).astype(np.float32)
    shifted = np.clip(shifted + 40.0, 0, 255).astype(np.uint8)
    shifted_features = inception_features(shifted, device=device, batch=32)
    assert kid_from_features(shifted_features, features, rng_seed=0) > kid


@pytest.mark.integration
@requires_weights
@requires_ixi
def test_reference_features_cache_round_trip(tmp_path):
    """The per-dataset cache is written once, reused, and invalidated by a different dataset sha."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    images = np.load(IXI_ROOT / "images.npy", mmap_mode="r")[:8]
    first = reference_features(tmp_path, np.asarray(images), "sha-a", device=device, batch=8)
    assert (tmp_path / "_features_inception_ref.npy").exists()
    second = reference_features(tmp_path, np.asarray(images), "sha-a", device=device, batch=8)
    np.testing.assert_array_equal(first, second)

    third = reference_features(tmp_path, np.asarray(images), "sha-b", device=device, batch=8)
    sidecar = json.loads((tmp_path / "_features_inception_ref.json").read_text())
    assert sidecar["dataset_sha256"] == "sha-b"
    np.testing.assert_allclose(first, third, rtol=1e-5, atol=1e-5)
