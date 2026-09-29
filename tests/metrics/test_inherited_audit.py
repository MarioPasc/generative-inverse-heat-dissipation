"""Pin the findings of the inherited-band audit (T6.3, ``docs/RESULTS/inherited_band_audit.md``).

The metric of ``ihdm.metrics.spectral.inherited_band`` is left unchanged. These tests state what it
measures when its two tacit assumptions fail:

* a population mean image with non-DC content (brain MRI): the residual about ``d x`` then carries
  ``(1 - d)^2 mu^2`` on top of ``(1 - d^2) P``, so the "share" becomes ``I - T`` and goes negative;
* a model that regenerates less than the removed variance (Churches): the shortfall reads as
  inheritance.

They also pin the exact identity ``sum V = D_pix (W^2 - 1) + sum_s (ybar_s - d x_s)^2`` that
lets the audit split every ``final.json`` value into a within-seed part and a seed-mean part
without the samples. The synthetic worlds use antithetic ``+-1`` draws, so every identity holds to
floating-point precision rather than up to estimation noise. The last test is ``integration``: it
reads the 24-run partial collection and the ``ref`` splits, and skips when the disk is absent.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.fft import dctn, idctn

from ihdm.metrics import inherited_band, log_bin_edges
from ihdm.metrics.diversity import within_seed_diversity
from ihdm.spectral.power import eigenvalues, inherited_share, mode_power, power_law, radial_profile

#: Side length of the synthetic worlds. ``sigma_max = W / 2`` gives the same damping per radial
#: index as the A0 arm (96 px at 192), ``W / 8`` the same as A3 (24 px at 192).
SIZE = 64

#: Samples per seed; even, so that the antithetic draws cancel exactly.
N_PER_SEED = 8

N_SEEDS = 4


def _damping(sigma_max: float, n_pix: int = SIZE) -> np.ndarray:
    """The ``DCTBlur`` kernel ``d = exp(-lambda sigma_max^2 / 2)`` on the grid."""
    return np.exp(-eigenvalues(n_pix) * sigma_max**2 / 2.0)


def _brain_like_mean(power: np.ndarray) -> np.ndarray:
    """A mean image in DCT coefficients, heavy in the even low modes as the registered MRI are.

    The ratios ``mu^2 / P`` of the three modes are those measured on the IXI ``ref`` split at
    ``(0, 2)``, ``(2, 0)`` and ``(1, 0)`` (25.7, 18.7, 5.7; audit §3).
    """
    mu = np.zeros_like(power)
    for (i, j), ratio in {(0, 2): 25.7, (2, 0): 18.7, (1, 0): 5.7}.items():
        mu[i, j] = np.sqrt(ratio * power[i, j])
    return mu


def _world(
    sigma_max: float, mean: bool, gain: float, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Seeds and samples of the linear-Gaussian model, optionally with a mean image.

    ``x_s = mu + xi_s`` with ``Var(xi) = P``; ``y_sm = mu + d (x_s - mu) + gain e_sm`` in the DCT
    basis, with ``e_sm = +- sqrt((1 - d^2) P)`` drawn in antithetic pairs. ``gain = 1`` is the
    model of ``05-metrics.md`` §5 written for a population with a mean; ``gain < 1`` is a model
    that regenerates only ``gain^2`` of the variance the blur removed.
    """
    power = power_law(SIZE, 2.0, 1.0)
    power[0, 0] = 0.0
    mu = _brain_like_mean(power) if mean else np.zeros_like(power)
    d = _damping(sigma_max)
    xi = rng.standard_normal((N_SEEDS, SIZE, SIZE)) * np.sqrt(power)[None]
    x_hat = mu[None] + xi
    half = rng.choice([-1.0, 1.0], size=(N_SEEDS, N_PER_SEED // 2, SIZE, SIZE))
    signs = np.concatenate([half, -half], axis=1)
    e = signs * np.sqrt((1.0 - d**2) * power)[None, None]
    y_hat = mu[None, None] + d[None, None] * (x_hat[:, None] - mu[None, None]) + gain * e
    return {
        "power": power,
        "mu": mu,
        "d": d,
        "seeds": idctn(x_hat, axes=(-2, -1), norm="ortho"),
        "samples": idctn(y_hat, axes=(-2, -1), norm="ortho"),
    }


def _mean_term(power: np.ndarray, mu: np.ndarray, d: np.ndarray) -> float:
    """``T = sum (1 - d)^2 mu^2 / sum P`` over the non-DC modes."""
    band = np.ones_like(power, dtype=bool)
    band[0, 0] = False
    return float(((1.0 - d) ** 2 * mu**2)[band].sum() / power[band].sum())


# ---------------------------------------------------------------------------------------------
# The mean image: why the MRI share is negative and the low bins spike
# ---------------------------------------------------------------------------------------------


def test_a_mean_image_shifts_the_measured_share_by_exactly_the_mean_term() -> None:
    """A perfect linear-Gaussian model on a population with a mean reads ``I - T``, not ``I``."""
    sigma_max = SIZE / 2.0
    world = _world(sigma_max, mean=True, gain=1.0, rng=np.random.default_rng(3))
    power, mu, d = world["power"], world["mu"], world["d"]
    result = inherited_band(world["samples"], world["seeds"], power, sigma_max)

    share = float(inherited_share(power, sigma_max))
    term = _mean_term(power, mu, d)
    assert term > 1.0 > share
    np.testing.assert_allclose(result.share_predicted, share, rtol=1e-12)
    np.testing.assert_allclose(result.share_measured, share - term, rtol=1e-9, atol=1e-12)
    assert result.share_measured < 0.0


def test_the_low_bin_spikes_are_mu_squared_over_p() -> None:
    """Mode by mode the measured ratio is ``(1 - d^2) + (1 - d)^2 mu^2 / P``, not ``1 - d^2``."""
    sigma_max = SIZE / 2.0
    world = _world(sigma_max, mean=True, gain=1.0, rng=np.random.default_rng(5))
    power, mu, d = world["power"], world["mu"], world["d"]
    result = inherited_band(world["samples"], world["seeds"], power, sigma_max)

    safe = np.where(power > 0.0, power, 1.0)
    corrected = np.where(power > 0.0, (1.0 - d**2) + (1.0 - d) ** 2 * mu**2 / safe, 0.0)
    expected = radial_profile(corrected, log_bin_edges())
    finite = np.isfinite(expected)
    np.testing.assert_array_equal(finite, np.isfinite(result.radial_measured))
    np.testing.assert_allclose(result.radial_measured[finite], expected[finite], rtol=1e-9)
    assert np.nanmax(result.radial_measured) > 10.0 * np.nanmax(result.radial_predicted)


def test_the_within_seed_share_is_immune_to_the_mean_image() -> None:
    """``1 - D_pix (W^2 - 1) / sum P`` recovers ``I`` exactly; the frozen share reads ``I - T``."""
    sigma_max = SIZE / 2.0
    world = _world(sigma_max, mean=True, gain=1.0, rng=np.random.default_rng(7))
    power = world["power"]
    diversity = within_seed_diversity(world["samples"])
    within = diversity.D_pix_mean * (SIZE * SIZE - 1) / float(power.sum())
    np.testing.assert_allclose(1.0 - within, inherited_share(power, sigma_max), rtol=1e-5)


# ---------------------------------------------------------------------------------------------
# Under-regeneration: why Churches reads 0.76 against 0.018
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("gain_squared", [0.05, 0.25, 1.0])
def test_under_regeneration_reads_as_inheritance(gain_squared: float) -> None:
    """A model regenerating ``g^2`` of the removed variance reads ``1 - g^2 (1 - I)``."""
    sigma_max = SIZE / 2.0
    world = _world(
        sigma_max, mean=False, gain=float(np.sqrt(gain_squared)), rng=np.random.default_rng(11)
    )
    power = world["power"]
    result = inherited_band(world["samples"], world["seeds"], power, sigma_max)
    share = float(inherited_share(power, sigma_max))
    np.testing.assert_allclose(
        result.share_measured, 1.0 - gain_squared * (1.0 - share), rtol=1e-9
    )
    # The prior state carries the same ``I`` in every case; only the regeneration differs.
    np.testing.assert_allclose(result.share_predicted, share, rtol=1e-12)


# ---------------------------------------------------------------------------------------------
# The identity that splits a final.json value without the samples
# ---------------------------------------------------------------------------------------------


def test_residual_splits_into_within_seed_variance_and_seed_mean_bias() -> None:
    """``sum V = D_pix (W^2 - 1) + mean_s sum (ybar_s - d x_s)^2`` on arbitrary samples."""
    rng = np.random.default_rng(13)
    sigma_max = SIZE / 8.0
    world = _world(sigma_max, mean=True, gain=0.6, rng=rng)
    # Break the model on purpose: a per-seed offset and Gaussian noise the identity must survive.
    samples = world["samples"] + 0.05 * rng.standard_normal(world["samples"].shape)
    samples += 0.1 * rng.standard_normal((N_SEEDS, 1, SIZE, SIZE))
    power, d = world["power"], world["d"]
    result = inherited_band(samples, world["seeds"], power, sigma_max)
    residual_sum = (1.0 - result.share_measured) * float(power.sum())

    centred = samples - samples.mean(axis=(-2, -1), keepdims=True)
    seeds = world["seeds"] - world["seeds"].mean(axis=(-2, -1), keepdims=True)
    y_bar = dctn(centred, axes=(-2, -1), norm="ortho").mean(axis=1)
    bias = (y_bar - d[None] * dctn(seeds, axes=(-2, -1), norm="ortho")) ** 2
    bias[:, 0, 0] = 0.0
    within = within_seed_diversity(samples).D_pix_mean * (SIZE * SIZE - 1)
    np.testing.assert_allclose(residual_sum, within + bias.sum(axis=(-2, -1)).mean(), rtol=1e-5)


# ---------------------------------------------------------------------------------------------
# Integration: the 24-run partial collection
# ---------------------------------------------------------------------------------------------


_DATA_ROOT = Path(
    os.environ.get(
        "IHDM_DATA_ROOT", "/media/mpascual/MeningD2/spectral_allocation_heat_diffusion_project"
    )
)
_PARTIAL = _DATA_ROOT / "_results_partial_24" / "runs"


def _reference(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    """``P_ref`` and the mean image (DCT, DC zeroed) of a ``ref`` split, as ``run_eval`` has it."""
    images = np.load(_DATA_ROOT / dataset / "images.npy", mmap_mode="r")
    splits = json.loads((_DATA_ROOT / dataset / "splits.json").read_text())
    ref_idx = np.sort(np.asarray(splits["ref"]))
    reference = np.asarray(images[ref_idx])
    power = mode_power(reference.astype(np.float32) / np.float32(255.0))
    power[0, 0] = 0.0
    x = reference.astype(np.float64) / 255.0
    x -= x.mean(axis=(1, 2), keepdims=True)
    mu = dctn(x.mean(axis=0), norm="ortho")
    mu[0, 0] = 0.0
    return power, mu


def _finals(prefix: str) -> list[dict]:
    """The ``final.json`` of every run of the partial collection whose id starts with ``prefix``."""
    return [
        json.loads((run / "final.json").read_text())
        for run in sorted(_PARTIAL.glob(f"{prefix}_s*"))
        if (run / "final.json").is_file()
    ]


@pytest.mark.integration
def test_partial_collection_decomposes_as_the_audit_states() -> None:
    """IXI A0 is negative because of the mean image; Churches A0 is high because of dispersion."""
    needed = [_DATA_ROOT / ds / "images.npy" for ds in ("ixi", "lsun_church")]
    if not (_PARTIAL.is_dir() and all(path.is_file() for path in needed)):
        pytest.skip("the partial collection or the datasets are not mounted")
    for dataset, prefix in (("ixi", "ixi_A0"), ("lsun_church", "lsun_church_A0")):
        finals = _finals(prefix)
        if not finals:
            pytest.skip(f"no evaluated {prefix} run in the partial collection")
        power, mu = _reference(dataset)
        n_pix = power.shape[0]
        d = _damping(96.0, n_pix)
        share = float(inherited_share(power, 96.0))
        term = _mean_term(power, mu, d)
        total = float(power.sum())
        for final in finals:
            np.testing.assert_allclose(final["inherited_predicted"], share, rtol=1e-4)
            within = final["diversity_pix"] * (n_pix * n_pix - 1) / total
            bias = 1.0 - final["inherited_measured"] - within
            m = final["n_per_seed"]
            if dataset == "ixi":
                # Negative share, carried by the seed-mean part, which is the mean-image term.
                assert term > 1.0
                assert final["inherited_measured"] < 0.0
                assert abs(bias - (term + (1.0 - share) / m)) < 0.25 * term
            else:
                # Small mean term; the within-seed variance is a few percent of the population's.
                assert term < 0.1
                assert within < 0.1 < (1.0 - 1.0 / m) * (1.0 - share)
