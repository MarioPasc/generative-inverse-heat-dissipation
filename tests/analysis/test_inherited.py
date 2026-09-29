"""Tests of ``ihdm.analysis.inherited`` (T6.4): the D23 constants and the corrected readings.

The reference fields are built in the DCT basis with antithetic ``+-sqrt(P)`` fluctuations around a
known mean image, so the population variance of every mode is exactly ``P`` and the population
mean exactly ``mu``: ``sum P``, ``I`` and ``T`` then have closed forms to compare against.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.fft import idctn

from ihdm.analysis.inherited import (
    InheritedError,
    band_constants,
    bias_expected,
    bias_fraction,
    compute_constants,
    experiment_sigmas,
    load_constants,
    main,
    reference_spectrum,
    within_seed_fraction,
    within_seed_share,
)
from ihdm.metrics import inherited_band, log_bin_edges
from ihdm.metrics.diversity import within_seed_diversity
from ihdm.spectral.power import eigenvalues, power_law, radial_profile
from tests.metrics.test_run_eval import _build_dataset

SIZE = 64
N_REF = 16

#: The corrected radial curve of :func:`synthetic_constants`; figure tests look for it.
SYNTHETIC_CORRECTED: tuple[float, ...] = (4.0, 1.2, 1.0)


def synthetic_constants(results_root: Path, out: Path, sum_power: float = 20000.0,
                        n_pix: int = 96, mean_term: float = 0.3,
                        datasets: tuple[str, ...] | None = None) -> Path:
    """A constants file matching the evaluated runs of a synthetic ``results/`` folder.

    Every dataset (or only ``datasets``) gets the ``dataset_sha256`` and ``sigma_max`` its runs
    record, with the given ``sum P_ref``, ``W`` and ``T``; used by the table and figure tests.
    """
    blocks: dict[str, dict] = {}
    with (results_root / "index.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        path = results_root / "runs" / row["run_id"] / "summary.json"
        if not path.is_file() or (datasets is not None and row["dataset"] not in datasets):
            continue
        summary = json.loads(path.read_text())
        sigma = float(summary["final"]["sigma_max"])
        block = blocks.setdefault(row["dataset"], {
            "sha256": summary["dataset_sha256"], "n_pix": n_pix, "sum_power": sum_power,
            "sigmas": {}})
        block["sigmas"][f"{sigma:g}"] = {
            "sigma_max": sigma, "share_predicted": 0.05, "mean_term": mean_term,
            "radial": {"centres": [0.7, 3.0, 20.0], "predicted": [0.5, 0.99, 1.0],
                       "corrected": list(SYNTHETIC_CORRECTED)}}
    out.write_text(json.dumps({"datasets": blocks}))
    return out


def _mean_image(power: np.ndarray) -> np.ndarray:
    """A mean image heavy in the even low modes, with the IXI ratios mu^2 / P (audit §3.2)."""
    mu = np.zeros_like(power)
    for (i, j), ratio in {(0, 2): 25.7, (2, 0): 18.7, (1, 0): 5.7, (3, 1): 0.4}.items():
        mu[i, j] = np.sqrt(ratio * power[i, j])
    return mu


def _reference(power: np.ndarray, mu: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """``N_REF`` images (0-255 scale) whose per-mode mean is ``mu`` and variance exactly ``P``."""
    half = rng.choice([-1.0, 1.0], size=(N_REF // 2, SIZE, SIZE))
    signs = np.concatenate([half, -half], axis=0)
    coefficients = mu[None] + signs * np.sqrt(power)[None]
    coefficients[:, 0, 0] = 0.5 * SIZE  # a DC offset, which every quantity must ignore
    return 255.0 * idctn(coefficients, axes=(-2, -1), norm="ortho")


@pytest.fixture(scope="module")
def spectrum() -> dict[str, np.ndarray]:
    power = power_law(SIZE, 2.0, 1.0)
    power[0, 0] = 0.0
    mu = _mean_image(power)
    return {"power": power, "mu": mu,
            "images": _reference(power, mu, np.random.default_rng(1))}


@pytest.mark.parametrize("sigma_max", [SIZE / 2.0, SIZE / 8.0])
def test_a_field_with_a_known_spectrum_and_a_mean_recovers_t_analytically(spectrum, sigma_max):
    power, mu = spectrum["power"], spectrum["mu"]
    p_ref, mu_ref = reference_spectrum(spectrum["images"])
    np.testing.assert_allclose(p_ref, power, rtol=1e-4, atol=1e-10)
    np.testing.assert_allclose(mu_ref, mu, rtol=1e-6, atol=1e-8)

    band = band_constants(p_ref, mu_ref, sigma_max)
    d = np.exp(-eigenvalues(SIZE) * sigma_max**2 / 2.0)
    # Closed form: only the four modes carrying a mean contribute to T.
    term = sum((1.0 - d[i, j]) ** 2 * mu[i, j] ** 2
               for i, j in ((0, 2), (2, 0), (1, 0), (3, 1))) / power.sum()
    share = float((d**2 * power).sum() / power.sum())
    assert term > 0.05
    np.testing.assert_allclose(band["mean_term"], term, rtol=1e-4)
    np.testing.assert_allclose(band["share_predicted"], share, rtol=1e-4)
    assert band["sigma_max"] == sigma_max and band["t_K"] == sigma_max**2 / 2.0


def test_the_corrected_radial_curve_is_the_mode_formula_on_the_populated_bins(spectrum):
    power, mu = spectrum["power"], spectrum["mu"]
    sigma_max = SIZE / 2.0
    band = band_constants(power, mu, sigma_max)
    d = np.exp(-eigenvalues(SIZE) * sigma_max**2 / 2.0)
    safe = np.where(power > 0, power, 1.0)
    modes = np.where(power > 0, (1.0 - d**2) + (1.0 - d) ** 2 * mu**2 / safe, 0.0)
    expected = radial_profile(modes, log_bin_edges())
    expected_pre = radial_profile(1.0 - d**2, log_bin_edges())
    finite = np.isfinite(expected)
    np.testing.assert_allclose(band["radial"]["corrected"], expected[finite], rtol=1e-12)
    np.testing.assert_allclose(band["radial"]["predicted"], expected_pre[finite], rtol=1e-12)
    assert len(band["radial"]["centres"]) == int(finite.sum())
    # The mean image lifts the low bins above 1 while the pre-registered line cannot exceed 1.
    assert max(band["radial"]["corrected"]) > 5.0 > 1.0 >= max(band["radial"]["predicted"])


def test_the_constants_explain_the_pre_registered_share_of_a_perfect_model(spectrum):
    """With the constants, the stored share of a linear-Gaussian world reads exactly ``I - T``."""
    power, mu = spectrum["power"], spectrum["mu"]
    sigma_max = SIZE / 2.0
    rng = np.random.default_rng(3)
    d = np.exp(-eigenvalues(SIZE) * sigma_max**2 / 2.0)
    n_seeds, m = 3, 6
    x_hat = mu[None] + rng.standard_normal((n_seeds, SIZE, SIZE)) * np.sqrt(power)[None]
    half = rng.choice([-1.0, 1.0], size=(n_seeds, m // 2, SIZE, SIZE))
    e = np.concatenate([half, -half], axis=1) * np.sqrt((1.0 - d**2) * power)[None, None]
    y_hat = mu + d * (x_hat[:, None] - mu) + e
    samples = idctn(y_hat, axes=(-2, -1), norm="ortho")
    seeds = idctn(x_hat, axes=(-2, -1), norm="ortho")
    measured = inherited_band(samples, seeds, power, sigma_max).share_measured
    band = band_constants(power, mu, sigma_max)
    np.testing.assert_allclose(measured, band["share_predicted"] - band["mean_term"],
                               rtol=1e-9, atol=1e-12)
    # Antithetic draws: the 1/M within-seed variance is exactly (1 - d^2) P, so G_w = 1 - I.
    d_pix = within_seed_diversity(samples).D_pix_mean
    g_w = within_seed_fraction(d_pix, SIZE, float(power.sum()))
    np.testing.assert_allclose(g_w, 1.0 - band["share_predicted"], rtol=1e-6)
    g_b = bias_fraction(measured, d_pix, SIZE, float(power.sum()))
    np.testing.assert_allclose(g_b, band["mean_term"], rtol=1e-5)


def test_the_corrected_readings_follow_the_audit_formulas():
    # ixi_A0_s1 of the partial collection (audit §3.3): G_w 0.336, G_b 1.431, I_w 0.657.
    d_pix, measured, total, m = 0.0114333, -0.767507, 1254.13, 50
    g_w = within_seed_fraction(d_pix, 192, total)
    assert g_w == pytest.approx(0.0114333 * (192**2 - 1) / 1254.13, rel=1e-12)
    assert g_w == pytest.approx(0.336, abs=5e-4)
    assert within_seed_share(d_pix, m, 192, total) == pytest.approx(1 - 50 / 49 * g_w, rel=1e-12)
    assert within_seed_share(d_pix, m, 192, total) == pytest.approx(0.657, abs=5e-4)
    assert bias_fraction(measured, d_pix, 192, total) == pytest.approx(1.431, abs=5e-4)
    assert bias_expected(0.00343245, 1.4015, m) == pytest.approx(1.4015 + 0.99656755 / 50)


@pytest.mark.parametrize("args", [
    (math.nan, 50, 192, 1.0), (0.1, 1, 192, 1.0), (0.1, 50, 192, 0.0), (0.1, math.nan, 192, 1.0),
])
def test_missing_inputs_give_nan_not_a_number(args):
    assert math.isnan(within_seed_share(*args))


def test_every_dataset_of_the_design_is_run_at_24_and_96_px():
    assert experiment_sigmas() == {"ixi": (24.0, 96.0), "lsun_church": (24.0, 96.0),
                                   "oasis1": (24.0, 96.0), "lsun_bedroom": (24.0, 96.0)}


def test_the_command_writes_constants_that_load_and_match_by_sha(tmp_path):
    dataset = _build_dataset(tmp_path / "data", dataset_id="ixi")
    out = tmp_path / "constants.json"
    assert main(["--data-root", str(tmp_path / "data"), "--out", str(out),
                 "--datasets", "ixi"]) == 0
    document = json.loads(out.read_text())
    sha = json.loads((dataset / "meta.json").read_text())["sha256_images"]
    block = document["datasets"]["ixi"]
    assert block["sha256"] == sha and block["sha256_verified"] is True
    assert set(block["sigmas"]) == {"24", "96"}
    assert document["arms"]["A0"] == {"schedule": "log_W2", "sigma_max": 96.0, "t_K": 4608.0}

    constants = load_constants(out)
    entry, reason = constants.lookup("ixi", sha, 96.0)
    assert entry is not None and reason == ""
    assert entry.n_pix == block["n_pix"] and entry.sum_power == block["sum_power"]
    assert entry.expected_measured == pytest.approx(entry.share_predicted - entry.mean_term)
    missing, why = constants.lookup("ixi", "0" * 64, 96.0)
    assert missing is None and "differs" in why
    missing, why = constants.lookup("ixi", sha, 48.0)
    assert missing is None and "48" in why
    assert load_constants(None).entries == {}


def test_a_dataset_outside_the_design_or_a_broken_hash_is_refused(tmp_path):
    with pytest.raises(InheritedError, match="not in the design"):
        compute_constants(tmp_path, ["cifar10"])
    dataset = _build_dataset(tmp_path / "data", dataset_id="oasis1")
    meta = json.loads((dataset / "meta.json").read_text())
    meta["sha256_images"] = "f" * 64
    (dataset / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(InheritedError, match="hashes to"):
        compute_constants(tmp_path / "data", ["oasis1"])
    with pytest.raises(InheritedError, match="malformed"):
        load_constants({"datasets": {"ixi": {"sigmas": {"96": {}}}}})


@pytest.mark.integration
def test_the_committed_constants_reproduce_the_audit_on_the_partial_collection():
    """``I_w`` of the 24-run partial collection, from ``results/`` and the committed JSON only."""
    from ihdm.analysis.tables import load_results

    partial = Path("/media/mpascual/MeningD2/spectral_allocation_heat_diffusion_project"
                   "/_results_partial_24")
    if not (partial / "index.csv").is_file():
        pytest.skip("the partial collection is not mounted")
    frame = load_results(partial).frame
    # inherited_band_audit.md §3.3, I_w column.
    for rid, value in {"ixi_A0_s1": 0.657, "ixi_A3_s1": 0.742, "oasis1_A0_s1": 0.559,
                       "lsun_church_A0_s1": 0.943, "lsun_church_A3_s2": 0.944}.items():
        assert frame.at[rid, "inherited_within"] == pytest.approx(value, abs=6e-4), rid
    assert frame.at["ixi_A0_s1", "inherited_expected"] == pytest.approx(-1.398, abs=1e-3)
    for rid in ("ixi_A0_s1", "lsun_church_A3_s1"):
        final = json.loads((partial / "runs" / rid / "final.json").read_text())
        entry, _ = load_constants().lookup(rid.rsplit("_", 2)[0], json.loads(
            (partial / "runs" / rid / "summary.json").read_text())["dataset_sha256"],
            final["sigma_max"])
        np.testing.assert_allclose(entry.radial_predicted, final["radial"]["predicted"],
                                   rtol=1e-4, atol=1e-6)
