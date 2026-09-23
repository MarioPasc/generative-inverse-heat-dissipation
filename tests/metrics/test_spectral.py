"""Identities of the spectral metrics (``docs/HARNESSES/metrics.md`` §1) and the JSON contract.

Every identity is checked against a field whose population spectrum is known analytically, by DCT
synthesis from ``ihdm.spectral.power.power_law`` plus random phases, never against a second
implementation of the same formula. The last test is the ``integration`` one: it runs the
endpoints on the real pilot samples drawn from ``pilot_ixi_A0_s1`` and on the ``ref`` split of
``ixi``, and skips when the data disk is not mounted.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.fft import dctn, idctn

from ihdm.metrics import (
    InheritedResult,
    LsdResult,
    MetricError,
    inherited_band,
    log_bin_centres,
    log_bin_edges,
    lsd,
    radial_log_profile,
    read_json,
    t_tau,
    write_json,
)
from ihdm.spectral.power import (
    OCTAVE_LABELS,
    eigenvalues,
    mode_power,
    power_law,
    radial_index,
)

#: Side length of the real grid: the 48 log bins between 0.5 and 96 cycles per image are defined
#: against it, and five of them hold no mode there (ticket log §2, D-T4.1-2).
IMAGE_SIZE = 192

#: Side length of the tests that only need a spectrum, kept small so they stay fast.
SMALL_IMAGE_SIZE = 64

#: The five bins of the 48 that hold no mode of the 192 grid.
EMPTY_BINS_AT_192 = [1, 2, 4, 5, 8]


# ---------------------------------------------------------------------------------------------
# Synthetic fields
# ---------------------------------------------------------------------------------------------


def synthesise(power: np.ndarray, n_images: int, rng: np.random.Generator) -> np.ndarray:
    """Draw a Gaussian stack whose population per-mode variance is ``power``."""
    coefficients = rng.standard_normal((n_images, *power.shape)) * np.sqrt(power)[None]
    coefficients[:, 0, 0] = 0.0
    return idctn(coefficients, axes=(1, 2), norm="ortho")


def heat_blur(images: np.ndarray, sigma: float) -> np.ndarray:
    """Blur a stack with the DCT heat kernel, the semantics of ``model_code.utils.DCTBlur``.

    ``DCTBlur`` multiplies every DCT coefficient by ``exp(-lambda_ij sigma^2 / 2)`` with
    ``lambda_ij = pi^2 (i^2 + j^2) / W^2``, which is ``ihdm.spectral.power.eigenvalues``; the same
    operation in numpy, so the tests do not depend on ``torch_dct``.
    """
    kernel = np.exp(-eigenvalues(images.shape[1]) * sigma**2 / 2.0)
    return idctn(dctn(images, axes=(1, 2), norm="ortho") * kernel[None], axes=(1, 2), norm="ortho")


def linear_gaussian_samples(
    seeds: np.ndarray,
    power: np.ndarray,
    sigma_max: float,
    n_per_seed: int,
    rng: np.random.Generator,
    gaussian: bool = True,
) -> np.ndarray:
    """Samples of the linear-Gaussian model of ``05-metrics.md`` §5.

    ``y_sm = d x_s + eps_sm`` in the DCT basis with ``d = exp(-lambda sigma_max^2 / 2)`` and
    ``Var(eps) = (1 - d^2) P``: the surviving part of the seed plus a regenerated part carrying
    exactly the variance the forward blur removed. With ``gaussian=False`` the regenerated part
    is drawn with random signs at exactly its nominal standard deviation, so the measured per-mode
    variance is ``(1 - d^2) P`` with no estimation error.
    """
    damping = np.exp(-eigenvalues(power.shape[0]) * sigma_max**2 / 2.0)
    deviation = np.sqrt(np.clip(1.0 - damping**2, 0.0, None) * power)
    out = np.empty((seeds.shape[0], n_per_seed, *power.shape))
    for s in range(seeds.shape[0]):
        prior = damping * dctn(seeds[s], norm="ortho")
        shape = (n_per_seed, *power.shape)
        draw = rng.standard_normal(shape) if gaussian else rng.choice([-1.0, 1.0], size=shape)
        out[s] = idctn(prior[None] + draw * deviation[None], axes=(1, 2), norm="ortho")
    return out


@pytest.fixture(scope="module")
def brain_like_power() -> np.ndarray:
    """A ``1/f^2`` per-mode variance on the real grid, total non-DC variance one."""
    return power_law(IMAGE_SIZE, 2.0, 1.0)


@pytest.fixture(scope="module")
def field_stack(brain_like_power: np.ndarray) -> np.ndarray:
    """64 independent ``1/f^2`` fields at 192 squared."""
    return synthesise(brain_like_power, 64, np.random.default_rng(101))


@pytest.fixture(scope="module")
def linear_model() -> dict[str, object]:
    """A linear-Gaussian world at 64 squared: seeds, samples, the spectrum, the terminal blur.

    ``sigma_max = W / 8`` is the brain configuration of the experiment; at ``W / 2`` the inherited
    share is a couple of percent and the identity would be testing the estimator's noise.
    """
    rng = np.random.default_rng(11)
    power = power_law(SMALL_IMAGE_SIZE, 2.0, 1.0)
    sigma_max = SMALL_IMAGE_SIZE / 8.0
    seeds = synthesise(power, 6, rng)
    return {
        "power": power,
        "sigma_max": sigma_max,
        "seeds": seeds,
        "gaussian": linear_gaussian_samples(seeds, power, sigma_max, 256, rng, gaussian=True),
        "exact": linear_gaussian_samples(seeds, power, sigma_max, 64, rng, gaussian=False),
    }


# ---------------------------------------------------------------------------------------------
# The fine radial grid
# ---------------------------------------------------------------------------------------------


def test_log_bins_span_the_frozen_range() -> None:
    edges = log_bin_edges(48)
    centres = log_bin_centres(48)
    assert edges.shape == (49,)
    assert centres.shape == (48,)
    np.testing.assert_allclose([edges[0], edges[-1]], [0.5, 96.0], rtol=1e-12)
    assert np.all(np.diff(edges) > 0)
    assert np.all(centres > edges[:-1]) and np.all(centres < edges[1:])
    ratios = edges[1:] / edges[:-1]
    np.testing.assert_allclose(ratios, ratios[0], rtol=1e-12)


@pytest.mark.parametrize("n_bins", [-1, 0, 1])
def test_log_bin_edges_rejects_degenerate_grids(n_bins: int) -> None:
    with pytest.raises(MetricError):
        log_bin_edges(n_bins)


def test_radial_profile_has_five_empty_bins_at_192(field_stack: np.ndarray) -> None:
    """The 192 grid leaves five of the 48 log bins with no mode at all (D-T4.1-2)."""
    _, profile = radial_log_profile(field_stack)
    assert np.flatnonzero(np.isnan(profile)).tolist() == EMPTY_BINS_AT_192


def test_radial_profile_recovers_the_spectral_exponent(field_stack: np.ndarray) -> None:
    """A ``1/f^2`` stack gives a profile of slope -2 in log-log over the populated bins."""
    centres, profile = radial_log_profile(field_stack)
    usable = np.isfinite(profile) & (centres >= 2.0)
    slope = float(np.polyfit(np.log10(centres[usable]), profile[usable], 1)[0])
    assert abs(slope + 2.0) < 0.05, slope


# ---------------------------------------------------------------------------------------------
# LSD
# ---------------------------------------------------------------------------------------------


def test_lsd_of_a_stack_against_itself_is_exactly_zero(field_stack: np.ndarray) -> None:
    result = lsd(field_stack, field_stack)
    assert isinstance(result, LsdResult)
    assert result.lsd == 0.0
    assert set(result.octaves) == set(OCTAVE_LABELS)
    assert all(value == 0.0 for value in result.octaves.values())
    assert result.variance_ratio == 1.0
    assert (result.n_samples, result.n_reference) == (64, 64)


def test_lsd_is_insensitive_to_the_uint8_conversion(field_stack: np.ndarray) -> None:
    """``uint8`` input and the same images already divided by 255 give the same number."""
    scaled = field_stack / (8.0 * float(np.abs(field_stack).max())) + 0.5
    as_u8 = np.rint(scaled * 255.0).clip(0, 255).astype(np.uint8)
    as_float = as_u8.astype(np.float32) / np.float32(255.0)
    from_u8 = lsd(as_u8[:32], as_u8[32:])
    from_float = lsd(as_float[:32], as_float[32:])
    np.testing.assert_allclose(from_u8.lsd, from_float.lsd, rtol=1e-12)
    np.testing.assert_allclose(from_u8.variance_ratio, from_float.variance_ratio, rtol=1e-12)


def test_lsd_increases_along_the_blur_ladder(field_stack: np.ndarray) -> None:
    """LSD against an unblurred half grows strictly with the blur applied to the other half."""
    reference, blurred_half = field_stack[32:], field_stack[:32]
    values = [lsd(heat_blur(blurred_half, sigma), reference).lsd for sigma in (1.0, 2.0, 4.0, 8.0)]
    assert all(np.isfinite(values))
    assert values == sorted(values) and len(set(values)) == 4, values
    # The two unblurred halves are draws from the same law: far below the first rung.
    assert lsd(blurred_half, reference).lsd < values[0]


def test_octave_rms_matches_the_fine_lsd_on_a_smooth_spectrum(
    brain_like_power: np.ndarray,
) -> None:
    """RMS of the eight octave differences within 10% of the 48-bin LSD (H-METRICS §1).

    The two stacks share their random coefficients and differ by a deterministic ``n^-0.4``
    factor, so the per-mode log difference is exactly the smooth function the identity is about
    and no estimation noise enters.
    """
    rng = np.random.default_rng(7)
    coefficients = rng.standard_normal((128, IMAGE_SIZE, IMAGE_SIZE)) * np.sqrt(brain_like_power)
    coefficients[:, 0, 0] = 0.0
    radius = radial_index(IMAGE_SIZE)
    tilt = np.where(radius > 0, np.maximum(radius, 1e-9) ** (-0.4), 0.0)
    reference = idctn(coefficients, axes=(1, 2), norm="ortho")
    samples = idctn(coefficients * np.sqrt(tilt)[None], axes=(1, 2), norm="ortho")

    result = lsd(samples, reference)
    octave_rms = float(np.sqrt(np.mean(np.square(list(result.octaves.values())))))
    assert abs(octave_rms - result.lsd) / result.lsd < 0.10
    # A softened spectrum loses variance in every octave, the high ones most.
    assert result.variance_ratio < 1.0
    assert all(value <= 0.0 for value in result.octaves.values())
    assert result.octaves["64-96"] < result.octaves["0.5-1"]


def test_lsd_rejects_a_size_mismatch(field_stack: np.ndarray) -> None:
    other = synthesise(power_law(SMALL_IMAGE_SIZE, 2.0, 1.0), 8, np.random.default_rng(1))
    with pytest.raises(MetricError, match="size mismatch"):
        lsd(field_stack, other)


@pytest.mark.parametrize(
    ("maker", "match"),
    [
        (lambda stack: stack[:1], "at least 2 images"),
        (lambda stack: stack[:, 0], r"\(N, W, W\)"),
        (lambda stack: stack[:, :, :16], "square images"),
    ],
)
def test_lsd_rejects_malformed_stacks(field_stack, maker, match) -> None:
    with pytest.raises(MetricError, match=match):
        lsd(maker(field_stack), field_stack)


def test_lsd_rejects_non_finite_input(field_stack: np.ndarray) -> None:
    broken = field_stack.copy()
    broken[3, 7, 11] = np.nan
    with pytest.raises(MetricError, match="non-finite"):
        lsd(broken, field_stack)


def test_lsd_rejects_a_constant_reference(field_stack: np.ndarray) -> None:
    with pytest.raises(MetricError):
        lsd(field_stack, np.zeros_like(field_stack))


# ---------------------------------------------------------------------------------------------
# T_tau
# ---------------------------------------------------------------------------------------------


def test_t_tau_returns_the_first_step_below_the_threshold() -> None:
    curve = {20000: 0.18, 5000: 0.62, 10000: 0.41, 15000: 0.25}
    assert t_tau(curve, 0.30) == 15000
    assert t_tau(curve, 0.70) == 5000


def test_t_tau_is_inclusive_at_equality() -> None:
    assert t_tau({5000: 0.5, 10000: 0.4}, 0.5) == 5000


def test_t_tau_returns_none_when_never_reached() -> None:
    assert t_tau({5000: 0.62, 10000: 0.41}, 0.10) is None


def test_t_tau_rejects_an_empty_curve() -> None:
    with pytest.raises(MetricError, match="empty"):
        t_tau({}, 0.3)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_t_tau_rejects_non_finite_values(bad: float) -> None:
    with pytest.raises(MetricError):
        t_tau({5000: bad}, 0.3)
    with pytest.raises(MetricError):
        t_tau({5000: 0.3}, bad)


# ---------------------------------------------------------------------------------------------
# Inherited band
# ---------------------------------------------------------------------------------------------


def test_inherited_band_is_exact_on_a_noiseless_linear_model(linear_model) -> None:
    """With the regenerated part at exactly its nominal variance the identity is exact."""
    result = inherited_band(
        linear_model["exact"],
        linear_model["seeds"],
        linear_model["power"],
        linear_model["sigma_max"],
    )
    assert isinstance(result, InheritedResult)
    np.testing.assert_allclose(result.share_measured, result.share_predicted, rtol=1e-10)
    usable = np.isfinite(result.radial_measured) & np.isfinite(result.radial_predicted)
    np.testing.assert_allclose(
        result.radial_measured[usable], result.radial_predicted[usable], rtol=1e-9
    )
    assert result.sigma_max == linear_model["sigma_max"]
    np.testing.assert_allclose(result.centres, log_bin_centres(48))


def test_inherited_band_matches_the_prediction_on_a_gaussian_linear_model(linear_model) -> None:
    """H-METRICS §1: measured share within 5% of the prediction, each bin within 10%."""
    result = inherited_band(
        linear_model["gaussian"],
        linear_model["seeds"],
        linear_model["power"],
        linear_model["sigma_max"],
    )
    assert 0.0 < result.share_predicted < 1.0
    relative = abs(result.share_measured - result.share_predicted) / result.share_predicted
    assert relative < 0.05, (result.share_measured, result.share_predicted)
    usable = (
        np.isfinite(result.radial_measured)
        & np.isfinite(result.radial_predicted)
        & (result.radial_predicted > 1e-3)
    )
    assert int(usable.sum()) > 20
    per_bin = np.abs(result.radial_measured[usable] - result.radial_predicted[usable])
    assert np.all(per_bin / result.radial_predicted[usable] < 0.10)


def test_inherited_band_low_band_restriction_is_self_consistent(linear_model) -> None:
    """Restricting both sums keeps the identity but changes the number the share reports."""
    full = inherited_band(
        linear_model["exact"],
        linear_model["seeds"],
        linear_model["power"],
        linear_model["sigma_max"],
    )
    banded = inherited_band(
        linear_model["exact"],
        linear_model["seeds"],
        linear_model["power"],
        linear_model["sigma_max"],
        low_band_sigma_px=8.0,
    )
    np.testing.assert_allclose(banded.share_measured, banded.share_predicted, rtol=1e-10)
    # The band drops modes the terminal blur has erased: it shrinks the denominator and not the
    # numerator, so the restricted share is the larger of the two (D-T4.1-3).
    assert banded.share_measured > full.share_measured


def test_inherited_band_reacts_to_the_terminal_blur(linear_model) -> None:
    """Samples built from a W/8 prior inherit far more than a W/2 prior would hand them."""
    strong = inherited_band(
        linear_model["exact"],
        linear_model["seeds"],
        linear_model["power"],
        SMALL_IMAGE_SIZE / 2.0,
    )
    assert strong.share_predicted < 0.05
    assert strong.share_measured > 2.0 * strong.share_predicted


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sigma_max": 0.0}, "sigma_max"),
        ({"sigma_max": np.inf}, "sigma_max"),
        ({"low_band_sigma_px": 0.0}, "low_band_sigma_px"),
    ],
)
def test_inherited_band_rejects_bad_scales(linear_model, kwargs, match) -> None:
    call = {
        "samples_per_seed": linear_model["exact"],
        "seeds": linear_model["seeds"],
        "reference_power": linear_model["power"],
        "sigma_max": linear_model["sigma_max"],
    }
    call.update(kwargs)
    with pytest.raises(MetricError, match=match):
        inherited_band(**call)


def test_inherited_band_rejects_shape_problems(linear_model) -> None:
    samples = linear_model["exact"]
    seeds = linear_model["seeds"]
    power = linear_model["power"]
    sigma_max = linear_model["sigma_max"]
    with pytest.raises(MetricError, match=r"\(S, M, W, W\)"):
        inherited_band(samples[0], seeds, power, sigma_max)
    with pytest.raises(MetricError, match="S >= 1"):
        inherited_band(samples[:0], seeds[:0], power, sigma_max)
    with pytest.raises(MetricError, match="seeds"):
        inherited_band(samples, seeds[:2], power, sigma_max)
    with pytest.raises(MetricError, match="reference_power"):
        inherited_band(samples, seeds, power[:8], sigma_max)


def test_inherited_band_rejects_non_finite_input(linear_model) -> None:
    samples = np.array(linear_model["exact"])
    samples[0, 0, 1, 1] = np.nan
    with pytest.raises(MetricError, match="non-finite"):
        inherited_band(
            samples, linear_model["seeds"], linear_model["power"], linear_model["sigma_max"]
        )


# ---------------------------------------------------------------------------------------------
# The result-file contract
# ---------------------------------------------------------------------------------------------


def test_write_json_sorts_keys_and_rounds_floats(tmp_path: Path) -> None:
    path = write_json(
        tmp_path / "metrics" / "ckpt_005000.json",
        {
            "variance_ratio": np.float64(1.234567891),
            "lsd": 0.1234567891,
            "lsd_octaves": {"0.5-1": np.float32(0.25)},
            "n_samples": np.int64(500),
            "curve": np.array([1.0, 2.5]),
            "p_value": 1.2345678e-9,
            "done": True,
            "path": tmp_path,
        },
    )
    text = path.read_text()
    assert text.index('"curve"') < text.index('"lsd"') < text.index('"n_samples"')
    payload = read_json(path)
    assert payload["lsd"] == 0.123457
    assert payload["variance_ratio"] == 1.23457
    assert payload["n_samples"] == 500
    assert payload["curve"] == [1.0, 2.5]
    assert payload["done"] is True
    assert payload["path"] == str(tmp_path)
    # Six significant digits, not six decimals: a small p-value survives (D-T4.1-4).
    assert payload["p_value"] == pytest.approx(1.23457e-9, rel=1e-9)


def test_write_json_refuses_non_finite_values(tmp_path: Path) -> None:
    with pytest.raises(MetricError, match="non-finite"):
        write_json(tmp_path / "bad.json", {"lsd": float("nan")})
    with pytest.raises(MetricError, match="non-finite"):
        write_json(tmp_path / "bad.json", {"lsd": float("inf")})


def test_write_json_refuses_unserialisable_values(tmp_path: Path) -> None:
    with pytest.raises(MetricError, match="cannot serialise"):
        write_json(tmp_path / "bad.json", {"result": object()})


def test_read_json_reports_missing_and_malformed_files(tmp_path: Path) -> None:
    with pytest.raises(MetricError, match="missing result file"):
        read_json(tmp_path / "absent.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    with pytest.raises(MetricError, match="not valid JSON"):
        read_json(broken)


def test_lsd_result_round_trips_through_the_result_contract(
    field_stack: np.ndarray, tmp_path: Path
) -> None:
    """A real ``LsdResult`` serialises under the key names of ``05-metrics.md`` §9."""
    result = lsd(heat_blur(field_stack[:32], 2.0), field_stack[32:])
    path = write_json(
        tmp_path / "ckpt_010000.json",
        {
            "lsd": result.lsd,
            "lsd_octaves": result.octaves,
            "variance_ratio": result.variance_ratio,
            "n_samples": result.n_samples,
        },
    )
    payload = read_json(path)
    assert set(payload) == {"lsd", "lsd_octaves", "variance_ratio", "n_samples"}
    assert set(payload["lsd_octaves"]) == set(OCTAVE_LABELS)
    assert payload["lsd"] == pytest.approx(result.lsd, rel=1e-5)


# ---------------------------------------------------------------------------------------------
# Integration: the real pilot samples and the real reference split
# ---------------------------------------------------------------------------------------------


@pytest.mark.integration
def test_pilot_samples_and_reference_bracket(pilot_sample_dirs) -> None:
    """The endpoints run on real data: pilot LSD above the reference-versus-reference floor."""
    if pilot_sample_dirs is None:
        pytest.skip("the pilot sample folders or the data disk are absent")
    data_root = pilot_sample_dirs["data_root"]
    images = np.load(data_root / "ixi" / "images.npy", mmap_mode="r")
    ref_idx = np.asarray(json.loads((data_root / "ixi" / "splits.json").read_text())["ref"])
    reference = np.asarray(images[ref_idx])

    samples = np.load(pilot_sample_dirs["lsd"] / "samples.npy")
    assert samples.ndim == 4 and samples.dtype == np.uint8
    pilot = lsd(samples.reshape(-1, samples.shape[2], samples.shape[3]), reference)
    floor = lsd(reference[:400], reference[400:])
    assert np.isfinite(pilot.lsd) and np.isfinite(floor.lsd)
    assert floor.lsd < 0.05, floor.lsd
    assert pilot.lsd > floor.lsd

    seed_samples = np.load(pilot_sample_dirs["seed"] / "samples.npy")
    seeds = np.load(pilot_sample_dirs["seed"] / "seeds.npy")
    power_ref = mode_power(reference.astype(np.float32) / np.float32(255.0))
    band = inherited_band(seed_samples, seeds, power_ref, 96.0)
    assert np.isfinite(band.share_measured)
    assert 0.0 < band.share_predicted < 1.0
    assert np.isfinite(band.radial_measured[np.isfinite(band.radial_predicted)]).all()
