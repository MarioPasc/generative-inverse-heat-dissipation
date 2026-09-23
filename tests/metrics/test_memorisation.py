"""Identities of the memorisation ratio and of the chunked nearest neighbour (H-METRICS §1).

The synthetic fields are ``1/f^alpha`` random images rather than white noise, because ``M_lp``
compares the sets *after* a heat-kernel low-pass and a white stack has almost nothing left
there; the natural-image spectrum is also what ``05-metrics.md`` assumes throughout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from scipy.fft import idctn

from ihdm.metrics.errors import MetricError
from ihdm.metrics.memorisation import memorisation_ratio, nn_distances
from ihdm.spectral.power import radial_index

#: Low-pass length-scale of the synthetic tests, in pixels. The frozen value is 16 px at 192 px;
#: on the 32 px test fields the scale-equivalent is 16 * 32 / 192 = 2.7 px.
TEST_SIGMA_LP: float = 2.0


def random_field(rng: np.random.Generator, n: int, side: int = 32, alpha: float = 2.0) -> np.ndarray:
    """Draw ``n`` random images with a ``P(n) ~ n^-alpha`` radial spectrum, scaled to ``[0, 1]``.

    Parameters
    ----------
    rng : numpy.random.Generator
        Source of randomness.
    n : int
        Number of images.
    side : int
        Image side length in pixels.
    alpha : float
        Spectral exponent.

    Returns
    -------
    numpy.ndarray
        ``float32`` stack of shape ``(n, side, side)``.
    """
    radius = radial_index(side)
    amplitude = np.where(radius > 0.0, np.maximum(radius, 1e-9) ** (-alpha / 2.0), 0.0)
    images = idctn(rng.normal(size=(n, side, side)) * amplitude, axes=(1, 2), norm="ortho")
    images = (images - images.min()) / (images.max() - images.min())
    return images.astype(np.float32)


@pytest.fixture
def corpus() -> np.ndarray:
    """A 64-image corpus of ``1/f^2`` fields.

    Returns
    -------
    numpy.ndarray
        ``float32`` stack of shape ``(64, 32, 32)``.
    """
    return random_field(np.random.default_rng(11), 64)


@pytest.fixture
def queries() -> np.ndarray:
    """A 24-image query set of ``1/f^2`` fields, independent of ``corpus``.

    Returns
    -------
    numpy.ndarray
        ``float32`` stack of shape ``(24, 32, 32)``.
    """
    return random_field(np.random.default_rng(12), 24)


# ------------------------------------------------------------------------------------------
# nn_distances
# ------------------------------------------------------------------------------------------


def test_chunking_matches_one_shot(queries: np.ndarray, corpus: np.ndarray) -> None:
    """Chunking is an implementation detail, not a change of the metric."""
    whole = nn_distances(queries, corpus, chunk=1024, device="cpu")
    pieces = nn_distances(queries, corpus, chunk=5, device="cpu")
    np.testing.assert_allclose(whole[0], pieces[0], atol=1e-5, rtol=0.0)
    np.testing.assert_array_equal(whole[1], pieces[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cpu_and_gpu_agree(queries: np.ndarray, corpus: np.ndarray) -> None:
    """The two device paths agree to 1e-4 and pick the same neighbour (H-METRICS §1)."""
    on_cpu = nn_distances(queries, corpus, chunk=7, device="cpu")
    on_gpu = nn_distances(queries, corpus, chunk=7, device="cuda")
    assert float(np.abs(on_cpu[0] - on_gpu[0]).max()) < 1e-4
    np.testing.assert_array_equal(on_cpu[1], on_gpu[1])


def test_matches_a_direct_computation(queries: np.ndarray, corpus: np.ndarray) -> None:
    """The chunked result equals the textbook definition on DC-removed images."""
    centred_q = (queries - queries.mean(axis=(1, 2), keepdims=True)).reshape(len(queries), -1)
    centred_c = (corpus - corpus.mean(axis=(1, 2), keepdims=True)).reshape(len(corpus), -1)
    exact = np.linalg.norm(centred_q[:, None, :] - centred_c[None, :, :], axis=2)
    distances, neighbours = nn_distances(queries, corpus, chunk=8, device="cpu")
    np.testing.assert_allclose(distances, exact.min(axis=1), atol=1e-4, rtol=0.0)
    np.testing.assert_array_equal(neighbours, exact.argmin(axis=1))


def test_dc_is_removed(queries: np.ndarray, corpus: np.ndarray) -> None:
    """A per-image intensity offset is not a distance."""
    offsets = np.linspace(-0.2, 0.2, len(queries), dtype=np.float32)[:, None, None]
    shifted = nn_distances(queries + offsets, corpus, chunk=8, device="cpu")
    plain = nn_distances(queries, corpus, chunk=8, device="cpu")
    np.testing.assert_allclose(shifted[0], plain[0], atol=1e-5, rtol=0.0)
    np.testing.assert_array_equal(shifted[1], plain[1])


def test_exact_copies_have_zero_distance(corpus: np.ndarray, queries: np.ndarray) -> None:
    """A query that *is* a corpus image is found, at a distance of numerical zero.

    "Numerical zero" is not machine zero: ``torch.cdist`` uses the matrix-multiply form
    ``|a|^2 + |b|^2 - 2 a.b``, whose cancellation leaves a floor of order
    ``sqrt(eps_32) * |a|``. The floor is asserted against the scale of a real distance, which is
    what the metric is ever compared to; on the ixi training split at 192 px it is 0.04 against
    nearest neighbours of 26 to 48.
    """
    distances, neighbours = nn_distances(corpus[[3, 17, 40]], corpus, chunk=2, device="cpu")
    scale = float(np.median(nn_distances(queries, corpus, chunk=8, device="cpu")[0]))
    np.testing.assert_array_equal(neighbours, [3, 17, 40])
    assert float(distances.max()) < 1e-3 * scale


@pytest.mark.parametrize(
    ("queries_in", "corpus_in", "chunk"),
    [
        (np.zeros((2, 8, 8), dtype=np.float32), np.zeros((2, 6, 6), dtype=np.float32), 4),
        (np.zeros((2, 8, 8), dtype=np.float32), np.zeros((0, 8, 8), dtype=np.float32), 4),
        (np.zeros((2, 8), dtype=np.float32), np.zeros((2, 8, 8), dtype=np.float32), 4),
        (np.full((2, 8, 8), np.inf, dtype=np.float32), np.zeros((2, 8, 8), dtype=np.float32), 4),
        (np.zeros((2, 8, 8), dtype=np.float32), np.zeros((2, 8, 8), dtype=np.float32), 0),
    ],
)
def test_nn_distances_rejects_bad_input(
    queries_in: np.ndarray, corpus_in: np.ndarray, chunk: int
) -> None:
    """Shape mismatch, empty sets, wrong rank, non-finite input and a bad chunk all raise."""
    with pytest.raises(MetricError):
        nn_distances(queries_in, corpus_in, chunk=chunk, device="cpu")


# ------------------------------------------------------------------------------------------
# memorisation_ratio
# ------------------------------------------------------------------------------------------


def subjects_of(n: int, per_subject: int = 1) -> list[str]:
    """Subject labels for a corpus of ``n`` images, ``per_subject`` images each.

    Parameters
    ----------
    n : int
        Number of images.
    per_subject : int
        Images per subject.

    Returns
    -------
    list[str]
        The labels, aligned with the corpus rows.
    """
    return [f"sub{i // per_subject:03d}" for i in range(n)]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_M_is_one_when_the_samples_are_held_out_images(seed: int) -> None:
    """Held-out real images as "samples" give ``M = 1`` within 5% (H-METRICS §1)."""
    rng = np.random.default_rng(seed)
    train = random_field(rng, 300)
    real = random_field(rng, 240)
    result = memorisation_ratio(
        real[:120], train, real[120:], subjects_of(300), None,
        sigma_lp=TEST_SIGMA_LP, device="cpu",
    )
    assert abs(result.M - 1.0) < 0.05
    assert abs(result.M_lp - 1.0) < 0.05
    assert result.n_samples == 120 and result.n_train == 300 and result.n_heldout == 120


def test_M_is_zero_when_the_samples_are_training_copies() -> None:
    """Exact copies of training images give ``M < 0.05`` (H-METRICS §1)."""
    rng = np.random.default_rng(5)
    train = random_field(rng, 200)
    heldout = random_field(rng, 100)
    result = memorisation_ratio(
        train[:80], train, heldout, subjects_of(200), None,
        sigma_lp=TEST_SIGMA_LP, device="cpu",
    )
    assert result.M < 0.05
    assert result.M_lp < 0.05
    assert result.d_samples_median >= 0.0


def test_seed_nn_fraction_is_one_when_every_sample_is_its_own_seed() -> None:
    """Samples that *are* their seeds give ``seed_nn_fraction = 1`` exactly."""
    rng = np.random.default_rng(6)
    train = random_field(rng, 120)
    heldout = random_field(rng, 60)
    seed_idx = np.arange(0, 120, 3, dtype=np.int64)
    result = memorisation_ratio(
        train[seed_idx], train, heldout, subjects_of(120), seed_idx,
        sigma_lp=TEST_SIGMA_LP, device="cpu",
    )
    assert result.seed_nn_fraction == 1.0
    np.testing.assert_array_equal(result.per_sample_nn, seed_idx)


def test_seed_nn_fraction_is_zero_when_every_sample_is_another_subject() -> None:
    """Samples copied from a different subject give ``seed_nn_fraction = 0`` exactly."""
    rng = np.random.default_rng(7)
    train = random_field(rng, 120)
    heldout = random_field(rng, 60)
    seed_idx = np.arange(0, 60, dtype=np.int64)
    copied_from = seed_idx + 60
    result = memorisation_ratio(
        train[copied_from], train, heldout, subjects_of(120), seed_idx,
        sigma_lp=TEST_SIGMA_LP, device="cpu",
    )
    assert result.seed_nn_fraction == 0.0


def test_seed_nn_fraction_counts_any_slice_of_the_seed_subject() -> None:
    """The MRI rule: the nearest image may be another slice of the seed's subject.

    The corpus holds two near-duplicate images per subject; every sample is copied from the
    *sibling* of its seed, so the nearest neighbour is never the seed itself and the fraction is
    still 1.
    """
    rng = np.random.default_rng(8)
    base = random_field(rng, 60)
    train = np.repeat(base, 2, axis=0) + rng.normal(0.0, 1e-3, size=(120, 32, 32)).astype(
        np.float32
    )
    heldout = random_field(rng, 60)
    seed_idx = np.arange(0, 120, 2, dtype=np.int64)
    siblings = seed_idx + 1
    result = memorisation_ratio(
        train[siblings], train, heldout, subjects_of(120, per_subject=2), seed_idx,
        sigma_lp=TEST_SIGMA_LP, device="cpu",
    )
    np.testing.assert_array_equal(result.per_sample_nn, siblings)
    assert result.seed_nn_fraction == 1.0


def test_missing_seed_index_warns_and_reports_zero(caplog: pytest.LogCaptureFixture) -> None:
    """``sample_seed_idx = None`` is a reported gap, not a silent zero."""
    rng = np.random.default_rng(9)
    train = random_field(rng, 60)
    heldout = random_field(rng, 30)
    with caplog.at_level("WARNING"):
        result = memorisation_ratio(
            train[:10], train, heldout, subjects_of(60), None,
            sigma_lp=TEST_SIGMA_LP, device="cpu",
        )
    assert result.seed_nn_fraction == 0.0
    assert "sample_seed_idx is None" in caplog.text


def test_overlapping_heldout_is_rejected() -> None:
    """A held-out image that is also in the corpus would deflate the denominator."""
    rng = np.random.default_rng(10)
    train = random_field(rng, 40)
    heldout = np.concatenate([random_field(rng, 19), train[[5]]])
    with pytest.raises(MetricError, match="also in the training corpus"):
        memorisation_ratio(
            train[:8], train, heldout, subjects_of(40), None,
            sigma_lp=TEST_SIGMA_LP, device="cpu",
        )


def test_uint8_and_float_inputs_agree() -> None:
    """The on-disk ``uint8`` path and its float twin give the same ratio."""
    rng = np.random.default_rng(13)
    train = np.rint(random_field(rng, 80) * 255.0).astype(np.uint8)
    heldout = np.rint(random_field(rng, 40) * 255.0).astype(np.uint8)
    as_uint8 = memorisation_ratio(
        heldout[:20], train, heldout[20:], subjects_of(80), None,
        sigma_lp=TEST_SIGMA_LP, device="cpu",
    )
    as_float = memorisation_ratio(
        heldout[:20].astype(np.float32) / 255.0,
        train.astype(np.float32) / 255.0,
        heldout[20:].astype(np.float32) / 255.0,
        subjects_of(80),
        None,
        sigma_lp=TEST_SIGMA_LP,
        device="cpu",
    )
    assert abs(as_uint8.M - as_float.M) < 1e-5
    assert abs(as_uint8.M_lp - as_float.M_lp) < 1e-5


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"train_subjects": subjects_of(5)}, "train_subjects"),
        ({"sample_seed_idx": np.zeros(3, dtype=np.int64)}, "sample_seed_idx must have shape"),
        ({"sample_seed_idx": np.full(8, 999, dtype=np.int64)}, "rows of the train stack"),
        ({"sample_seed_idx": np.zeros(8, dtype=np.float32)}, "integer array"),
    ],
)
def test_memorisation_ratio_rejects_bad_input(kwargs: dict[str, Any], match: str) -> None:
    """The alignment of ``train_subjects`` and ``sample_seed_idx`` is checked, not assumed."""
    rng = np.random.default_rng(14)
    train = random_field(rng, 40)
    heldout = random_field(rng, 20)
    call: dict[str, Any] = {
        "samples": train[:8],
        "train": train,
        "heldout": heldout,
        "train_subjects": subjects_of(40),
        "sample_seed_idx": None,
        "sigma_lp": TEST_SIGMA_LP,
        "device": "cpu",
    }
    call.update(kwargs)
    with pytest.raises(MetricError, match=match):
        memorisation_ratio(**call)


def test_image_size_mismatch_is_rejected() -> None:
    """The three stacks must live on the same grid."""
    rng = np.random.default_rng(15)
    with pytest.raises(MetricError, match="share the image size"):
        memorisation_ratio(
            random_field(rng, 4, side=16),
            random_field(rng, 20, side=32),
            random_field(rng, 10, side=32),
            subjects_of(20),
            None,
            device="cpu",
        )


# ------------------------------------------------------------------------------------------
# Integration: the pilot samples of T4.1 against the real ixi training split
# ------------------------------------------------------------------------------------------


@pytest.mark.integration
def test_memorisation_on_the_pilot_samples(
    pilot_sample_dirs: dict[str, Path] | None, ixi_dataset: dict[str, Any] | None
) -> None:
    """``M``, ``M_lp`` and ``seed_nn_fraction`` on the 64 training-seeded pilot samples.

    The numbers themselves are recorded in ``docs/RESULTS/metrics_bracket.md`` §5; this test
    checks that the real path runs end to end on the real shapes and returns finite numbers in
    the range the definition allows.
    """
    if pilot_sample_dirs is None or ixi_dataset is None:
        pytest.skip("the pilot sample folders or the ixi dataset are not on this machine")

    index = ixi_dataset["index"]
    train_rows = np.asarray(ixi_dataset["splits"]["train"], dtype=np.int64)
    heldout_rows = np.asarray(index.index[index["split"] == "ref"], dtype=np.int64)
    images = ixi_dataset["images"]
    train = np.asarray(images[train_rows])
    heldout = np.asarray(images[heldout_rows])

    samples = np.load(pilot_sample_dirs["lsd"] / "samples.npy")[:, 0]
    seed_dataset_idx = np.load(pilot_sample_dirs["lsd"] / "seed_idx.npy")
    # 18 of the 64 recorded pilot seeds are not rows of the current training split (T4.2 log §6);
    # M is defined on training-seeded samples, so only the rest can enter it.
    in_train = np.isin(seed_dataset_idx, train_rows)
    samples = samples[in_train]
    order = np.argsort(train_rows)
    seed_rows = order[np.searchsorted(train_rows, seed_dataset_idx[in_train], sorter=order)]
    assert np.array_equal(train_rows[seed_rows], seed_dataset_idx[in_train])

    result = memorisation_ratio(
        samples,
        train,
        heldout,
        index.loc[train_rows, "subject"].tolist(),
        seed_rows,
        device="cuda",
    )
    assert result.n_samples == samples.shape[0]
    assert result.n_train == 3200 and result.n_heldout == 400
    assert np.isfinite([result.M, result.M_lp, result.seed_nn_fraction]).all()
    assert result.M > 0.0
    assert 0.0 <= result.seed_nn_fraction <= 1.0
    assert result.per_sample_d.shape == (samples.shape[0],)
    assert result.per_sample_nn.min() >= 0 and result.per_sample_nn.max() < 3200
