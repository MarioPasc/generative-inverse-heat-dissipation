"""Identities of the within-seed diversity and of the PCA panel (H-METRICS §1).

``within_seed_diversity`` is checked against the one case where the answer is known in closed
form -- ``y_sm = x_s + eps`` with ``eps ~ N(0, v)`` -- and ``pca_around_seed`` against a planted
two-dimensional subspace, which is the only way to test a randomised SVD without re-implementing
it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ihdm.metrics.diversity import DivResult, pca_around_seed, within_seed_diversity
from ihdm.metrics.errors import MetricError
from tests.metrics.test_memorisation import TEST_SIGMA_LP, random_field


@pytest.fixture
def seed_images() -> np.ndarray:
    """Four ``1/f^2`` seed images at 32 px.

    Returns
    -------
    numpy.ndarray
        ``float32`` stack of shape ``(4, 32, 32)``.
    """
    return random_field(np.random.default_rng(31), 4)


# ------------------------------------------------------------------------------------------
# within_seed_diversity
# ------------------------------------------------------------------------------------------


def test_identical_samples_have_zero_diversity(seed_images: np.ndarray) -> None:
    """A chain that always returns the same image has no diversity (H-METRICS §1)."""
    result = within_seed_diversity(
        np.repeat(seed_images[:, None, :, :], 12, axis=1), sigma_lp=TEST_SIGMA_LP
    )
    assert result.D_pix_mean == 0.0
    assert result.D_lp_mean == 0.0
    assert result.n_seeds == 4 and result.n_per_seed == 12


@pytest.mark.parametrize("variance", [1e-4, 1e-2])
def test_gaussian_perturbation_recovers_its_variance(
    seed_images: np.ndarray, variance: float
) -> None:
    """``y_sm = x_s + eps`` gives ``D_pix = v (1 - 1/M)`` within 5% (H-METRICS §1).

    The ``(1 - 1/M)`` factor is the frozen ``1/M`` normalisation of ``05-metrics.md`` §3 (the
    within-seed variance is taken about the *sample* mean of the same ``M`` draws); at the
    ``M = 50`` of §8a it is a 2% deficit common to every arm.
    """
    rng = np.random.default_rng(32)
    n_per_seed = 50
    noise = rng.normal(0.0, np.sqrt(variance), size=(4, n_per_seed, 32, 32)).astype(np.float32)
    result = within_seed_diversity(seed_images[:, None, :, :] + noise, sigma_lp=TEST_SIGMA_LP)
    expected = variance * (1.0 - 1.0 / n_per_seed)
    assert abs(result.D_pix_mean / expected - 1.0) < 0.05
    np.testing.assert_allclose(result.per_seed_pix, expected, rtol=0.05)


def test_the_low_pass_removes_most_of_a_white_perturbation(seed_images: np.ndarray) -> None:
    """White sampling noise is fine-scale, so ``D_lp`` is far below ``D_pix``."""
    rng = np.random.default_rng(33)
    noise = rng.normal(0.0, 0.1, size=(4, 30, 32, 32)).astype(np.float32)
    result = within_seed_diversity(seed_images[:, None, :, :] + noise, sigma_lp=TEST_SIGMA_LP)
    assert 0.0 < result.D_lp_mean < 0.1 * result.D_pix_mean


def test_diversity_ignores_a_per_image_offset(seed_images: np.ndarray) -> None:
    """The DC is removed, so a per-sample intensity shift is not diversity."""
    rng = np.random.default_rng(34)
    block = seed_images[:, None, :, :] + rng.normal(0.0, 0.05, size=(4, 20, 32, 32)).astype(
        np.float32
    )
    offsets = rng.normal(0.0, 0.3, size=(4, 20, 1, 1)).astype(np.float32)
    plain = within_seed_diversity(block, sigma_lp=TEST_SIGMA_LP)
    shifted = within_seed_diversity(block + offsets, sigma_lp=TEST_SIGMA_LP)
    np.testing.assert_allclose(shifted.per_seed_pix, plain.per_seed_pix, rtol=1e-4)


def test_uint8_and_float_inputs_agree(seed_images: np.ndarray) -> None:
    """The on-disk ``uint8`` path and its float twin give the same diversity."""
    rng = np.random.default_rng(35)
    block = seed_images[:, None, :, :] + rng.normal(0.0, 0.05, size=(4, 16, 32, 32)).astype(
        np.float32
    )
    quantised = np.rint(np.clip(block, 0.0, 1.0) * 255.0).astype(np.uint8)
    as_uint8 = within_seed_diversity(quantised, sigma_lp=TEST_SIGMA_LP)
    as_float = within_seed_diversity(
        quantised.astype(np.float32) / 255.0, sigma_lp=TEST_SIGMA_LP
    )
    np.testing.assert_allclose(as_uint8.per_seed_pix, as_float.per_seed_pix, rtol=1e-6)
    assert isinstance(as_uint8, DivResult)


@pytest.mark.parametrize(
    "samples",
    [
        np.zeros((4, 32, 32), dtype=np.float32),
        np.zeros((0, 5, 8, 8), dtype=np.float32),
        np.zeros((2, 1, 8, 8), dtype=np.float32),
        np.zeros((2, 3, 8, 6), dtype=np.float32),
        np.full((2, 3, 8, 8), np.nan, dtype=np.float32),
    ],
)
def test_within_seed_diversity_rejects_bad_input(samples: np.ndarray) -> None:
    """Rank, emptiness, a single sample per seed, squareness and finiteness all raise."""
    with pytest.raises(MetricError):
        within_seed_diversity(samples, sigma_lp=TEST_SIGMA_LP)


# ------------------------------------------------------------------------------------------
# pca_around_seed
# ------------------------------------------------------------------------------------------


def planted_subspace(rng: np.random.Generator, n_train: int = 80, side: int = 16) -> np.ndarray:
    """A training stack whose variance lives almost entirely in two known directions.

    Parameters
    ----------
    rng : numpy.random.Generator
        Source of randomness.
    n_train : int
        Number of training images.
    side : int
        Image side length.

    Returns
    -------
    numpy.ndarray
        ``float32`` stack of shape ``(n_train, side, side)``.
    """
    basis = random_field(rng, 2, side=side)
    weights = rng.normal(size=(n_train, 2)) * np.array([4.0, 2.0])
    field = weights @ basis.reshape(2, -1)
    field += 0.01 * rng.normal(size=field.shape)
    return field.reshape(n_train, side, side).astype(np.float32)


@pytest.fixture
def pca_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A planted-subspace training stack with three seeds and five samples each.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        ``(train, seeds, samples)``.
    """
    rng = np.random.default_rng(41)
    train = planted_subspace(rng)
    seeds = train[:3].copy()
    samples = seeds[:, None, :, :] + rng.normal(0.0, 0.05, size=(3, 5, 16, 16)).astype(np.float32)
    return train, seeds, samples


def test_components_are_orthonormal(pca_inputs: tuple[np.ndarray, ...]) -> None:
    """The returned components are an orthonormal basis of the fitted subspace."""
    result = pca_around_seed(*pca_inputs, n_components=2, rng_seed=0, device="cpu")
    flat = result.components.reshape(2, -1)
    np.testing.assert_allclose(flat @ flat.T, np.eye(2), atol=1e-5)
    assert result.components.shape == (2, 16, 16)
    assert result.mean.shape == (16, 16)
    assert result.explained_variance.shape == (2,)
    assert result.explained_variance[0] > result.explained_variance[1] > 0.0


def test_scores_are_reproducible(pca_inputs: tuple[np.ndarray, ...]) -> None:
    """The same ``rng_seed`` on the same device gives bit-identical scores."""
    first = pca_around_seed(*pca_inputs, n_components=2, rng_seed=3, device="cpu")
    second = pca_around_seed(*pca_inputs, n_components=2, rng_seed=3, device="cpu")
    np.testing.assert_array_equal(first.train_scores, second.train_scores)
    np.testing.assert_array_equal(first.seed_scores, second.seed_scores)
    np.testing.assert_array_equal(first.sample_scores, second.sample_scores)
    np.testing.assert_array_equal(first.components, second.components)


def test_the_sign_convention_survives_a_different_draw(
    pca_inputs: tuple[np.ndarray, ...],
) -> None:
    """A different random test matrix finds the same subspace with the same signs.

    Without the sign convention the components of two draws differ by an arbitrary sign each,
    and "the scores are reproducible" would hold only per seed of the randomised SVD.
    """
    first = pca_around_seed(*pca_inputs, n_components=2, rng_seed=0, device="cpu")
    second = pca_around_seed(*pca_inputs, n_components=2, rng_seed=17, device="cpu")
    np.testing.assert_allclose(first.components, second.components, atol=1e-3)
    np.testing.assert_allclose(first.seed_scores, second.seed_scores, atol=1e-2)


def test_the_planted_subspace_is_recovered(pca_inputs: tuple[np.ndarray, ...]) -> None:
    """The two components span the plane the training variance was planted in.

    The fixture adds isotropic noise of standard deviation 0.01 per pixel on top of the plane,
    a floor of 1.3% of the total norm; the recovered subspace is required to leave no more than
    that, so the test measures the fit and not the noise.
    """
    train, _, _ = pca_inputs
    result = pca_around_seed(*pca_inputs, n_components=2, rng_seed=0, device="cpu")
    flat = result.components.reshape(2, -1)
    centred = train.reshape(len(train), -1) - train.reshape(len(train), -1).mean(axis=0)
    centred -= centred.mean(axis=1, keepdims=True)
    residual = centred - (centred @ flat.T) @ flat
    noise_floor = 0.01 * np.sqrt(centred.size) / float(np.linalg.norm(centred))
    assert float(np.linalg.norm(residual) / np.linalg.norm(centred)) < 1.1 * noise_floor


def test_projection_is_consistent_with_the_components(
    pca_inputs: tuple[np.ndarray, ...],
) -> None:
    """``seed_scores`` is exactly the documented projection of the DC-removed seeds."""
    train, seeds, samples = pca_inputs
    result = pca_around_seed(train, seeds, samples, n_components=2, rng_seed=0, device="cpu")
    flat_seeds = seeds.reshape(len(seeds), -1) - seeds.reshape(len(seeds), -1).mean(
        axis=1, keepdims=True
    )
    expected = (flat_seeds - result.mean.reshape(-1)) @ result.components.reshape(2, -1).T
    np.testing.assert_allclose(result.seed_scores, expected, atol=1e-3)
    assert result.sample_scores.shape == (3, 5, 2)
    assert result.train_scores.shape == (len(train), 2)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_components": 0}, "n_components"),
        ({"samples": np.zeros((3, 16, 16), dtype=np.float32)}, "S, M, H, W"),
        ({"seeds": np.zeros((2, 16, 16), dtype=np.float32)}, "2 images for 3 seeds"),
        ({"train": np.zeros((2, 16, 16), dtype=np.float32)}, "at least 3 images"),
    ],
)
def test_pca_rejects_bad_input(
    pca_inputs: tuple[np.ndarray, ...], kwargs: dict[str, Any], match: str
) -> None:
    """Every alignment the figure depends on is checked at the boundary."""
    train, seeds, samples = pca_inputs
    call: dict[str, Any] = {
        "train": train,
        "seeds": seeds,
        "samples": samples,
        "n_components": 2,
        "device": "cpu",
    }
    call.update(kwargs)
    with pytest.raises(MetricError, match=match):
        pca_around_seed(**call)


# ------------------------------------------------------------------------------------------
# Integration: the 4 x 10 held-out-seeded pilot set
# ------------------------------------------------------------------------------------------


@pytest.mark.integration
def test_diversity_and_pca_on_the_pilot_samples(
    pilot_sample_dirs: dict[str, Path] | None, ixi_dataset: dict[str, Any] | None
) -> None:
    """Diversity of the 4 x 10 held-out-seeded set and the PCA scores of its four seeds.

    The numbers are recorded in ``docs/RESULTS/metrics_bracket.md`` §5.
    """
    if pilot_sample_dirs is None or ixi_dataset is None:
        pytest.skip("the pilot sample folders or the ixi dataset are not on this machine")

    samples = np.load(pilot_sample_dirs["seed"] / "samples.npy")
    seeds = np.load(pilot_sample_dirs["seed"] / "seeds.npy")
    assert samples.shape == (4, 10, 192, 192)

    diversity = within_seed_diversity(samples)
    assert diversity.n_seeds == 4 and diversity.n_per_seed == 10
    assert np.isfinite(diversity.per_seed_pix).all() and (diversity.per_seed_pix > 0.0).all()
    assert np.isfinite(diversity.per_seed_lp).all()
    assert diversity.D_lp_mean < diversity.D_pix_mean

    train_rows = np.asarray(ixi_dataset["splits"]["train"], dtype=np.int64)
    train = np.asarray(ixi_dataset["images"][train_rows])
    pca = pca_around_seed(train, seeds, samples, n_components=2, rng_seed=0, device="cuda")
    assert pca.train_scores.shape == (3200, 2)
    assert pca.seed_scores.shape == (4, 2)
    assert pca.sample_scores.shape == (4, 10, 2)
    assert np.isfinite(pca.train_scores).all() and np.isfinite(pca.sample_scores).all()
    flat = pca.components.reshape(2, -1)
    np.testing.assert_allclose(flat @ flat.T, np.eye(2), atol=1e-4)
