"""Tests of T7.2: ``--delta`` trees, ``--final-from-lsd`` and the W rule of the spectral grid.

The default path is pinned against the base commit by ``test_regression_t72.py``; this module
tests what is new: the ``_delta-<repr>`` trees and their records, that a non-default noise level
changes the samples while an explicit default one reproduces them, the reuse of the LSD set as the
final set, the grids at ``W = 128`` and ``W = 192`` and the legacy branch below 128, and an end to
end evaluation of a synthetic 128² run (and ``prepare_eval`` on its dataset).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from ihdm.cli.evaluate_run import build_parser, request_from_args
from ihdm.metrics.errors import MetricError
from ihdm.metrics.run_eval import (
    EvalRequest,
    ensure_seed_lists,
    evaluate_run,
    metrics_dirname,
    samples_dirname,
)
from ihdm.metrics.spectral import (
    LOW_BAND_SIGMA_PX,
    REFERENCE_SIGMA_LP_PX,
    _low_band_mask,
    log_bin_centres,
    log_bin_edges,
    lsd,
    spectral_grid,
)
from ihdm.spectral.power import OCTAVE_EDGES, OCTAVE_LABELS, radial_profile
from tests.metrics.test_run_eval import _build_dataset, _config, _write_run

#: The smoke config's default noise sd: delta_factor 1.25 times sigma 0.01.
DEFAULT_DELTA = 0.0125

#: A small request on the fixture: the final checkpoint only, no Inception.
SMALL: dict = {
    "ckpts": "750", "n_lsd": 8, "n_final": 8, "n_seeds": 2, "n_per_seed": 3,
    "sample_batch": 8, "skip_inception": True, "device": "cpu",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _tree_digest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha(path)
        for path in sorted(Path(root).rglob("*"))
        if path.is_file()
    }


# --------------------------------------------------------------------------------------------
# Tree names
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "amp, delta, samples, metrics",
    [
        ("off", None, "samples", "metrics"),
        ("fp16", None, "samples_amp-fp16", "metrics_amp-fp16"),
        ("off", 0.0125, "samples_delta-0.0125", "metrics_delta-0.0125"),
        ("fp16", 0.02, "samples_amp-fp16_delta-0.02", "metrics_amp-fp16_delta-0.02"),
        ("bf16", 0.03, "samples_amp-bf16_delta-0.03", "metrics_amp-bf16_delta-0.03"),
        ("fp16", 0.0, "samples_amp-fp16_delta-0.0", "metrics_amp-fp16_delta-0.0"),
    ],
)
def test_every_noise_level_has_its_own_trees(amp, delta, samples, metrics):
    assert samples_dirname(amp, delta) == samples
    assert metrics_dirname(amp, delta) == metrics


def test_the_tree_tag_is_the_repr_of_the_float():
    assert samples_dirname("fp16", 2e-2) == samples_dirname("fp16", 0.020)
    assert samples_dirname("off", 1) == "samples_delta-1.0"


@pytest.mark.parametrize("delta", [-0.01, float("nan"), float("inf")])
def test_a_negative_or_non_finite_delta_is_refused(delta):
    with pytest.raises(MetricError):
        samples_dirname("off", delta)
    with pytest.raises(MetricError):
        metrics_dirname("fp16", delta)


# --------------------------------------------------------------------------------------------
# The W rule
# --------------------------------------------------------------------------------------------


def test_the_grid_at_192_is_the_frozen_constants_bit_for_bit():
    grid = spectral_grid(192)
    assert not grid.legacy
    np.testing.assert_array_equal(grid.log_edges, log_bin_edges())
    np.testing.assert_array_equal(grid.log_centres, log_bin_centres())
    assert grid.octave_edges == tuple(float(edge) for edge in OCTAVE_EDGES)
    assert grid.octave_labels == OCTAVE_LABELS
    assert grid.low_band_sigma_px == LOW_BAND_SIGMA_PX == 8.0
    assert grid.sigma_lp_px == REFERENCE_SIGMA_LP_PX == 16.0


def test_the_grid_at_128_scales_with_the_image_side():
    grid = spectral_grid(128)
    assert not grid.legacy
    assert grid.max_cycles == 64.0
    assert grid.log_edges[0] == 0.5 and grid.log_edges[-1] == pytest.approx(64.0, rel=1e-12)
    assert grid.log_edges.size == 49
    assert grid.octave_edges == (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
    assert grid.octave_labels == ("0.5-1", "1-2", "2-4", "4-8", "8-16", "16-32", "32-64")
    assert grid.low_band_sigma_px == pytest.approx(16.0 / 3.0)
    assert grid.sigma_lp_px == pytest.approx(32.0 / 3.0)


@pytest.mark.parametrize("width", [128, 160, 256])
def test_the_low_band_is_fixed_in_cycles_per_image(width):
    """``n <= sqrt(2) W / (pi sigma)`` is 10.80 at every side when sigma scales with W."""
    grid = spectral_grid(width)
    mask = _low_band_mask(width, grid.low_band_sigma_px)
    reference = _low_band_mask(192, 8.0)
    size = min(width, 192)
    np.testing.assert_array_equal(mask[:size, :size], reference[:size, :size])
    radius = np.hypot(*np.meshgrid(np.arange(width), np.arange(width), indexing="ij"))
    assert radius[mask].max() <= math.sqrt(2.0) * 192 / (math.pi * 8.0)


@pytest.mark.parametrize("width", [2, 32, 64, 96, 127])
def test_below_128_the_legacy_constants_are_kept(width):
    """The legacy branch exists for the 96² fixture of test_run_eval (tests/analysis reads it)."""
    grid = spectral_grid(width)
    assert grid.legacy
    np.testing.assert_array_equal(grid.log_edges, log_bin_edges())
    assert grid.octave_labels == OCTAVE_LABELS
    assert grid.low_band_sigma_px == 8.0
    assert grid.sigma_lp_px == 16.0


def test_the_frozen_grid_functions_are_unchanged_without_an_image_side():
    np.testing.assert_array_equal(
        log_bin_edges(48),
        np.logspace(math.log10(0.5), math.log10(96.0), 49),
    )
    np.testing.assert_array_equal(log_bin_edges(48, n_pix=128), spectral_grid(128).log_edges)


def _stack(n: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    from scipy.fft import idctn

    idx = np.arange(width)
    weight = 1.0 / np.maximum(np.hypot(idx[:, None], idx[None, :]), 1.0)
    images = idctn(rng.normal(size=(n, width, width)) * weight * 6.0, axes=(1, 2), norm="ortho")
    return np.clip(np.rint((images + 0.5) * 255.0), 0, 255).astype(np.uint8)


def test_the_lsd_at_128_uses_the_populated_bins_and_seven_octaves():
    samples, reference = _stack(10, 128, 1), _stack(12, 128, 2)
    result = lsd(samples, reference)
    assert tuple(result.octaves) == spectral_grid(128).octave_labels
    assert all(math.isfinite(value) for value in result.octaves.values())
    assert math.isfinite(result.lsd) and result.lsd > 0.0
    assert lsd(reference, reference).lsd == 0.0
    # Of the 48 log bins between 0.5 and 64 c/img, the ones holding no mode of the 128 grid drop.
    populated = np.isfinite(radial_profile(np.ones((128, 128)), spectral_grid(128).log_edges))
    assert 40 <= int(populated.sum()) < 48


# --------------------------------------------------------------------------------------------
# --delta end to end on the 96² fixture
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def delta_runs(tmp_path_factory) -> dict:
    """One run evaluated at the default, at delta = 0.02, and at an explicit default delta."""
    base = tmp_path_factory.mktemp("t72_delta")
    dataset = _build_dataset(base / "data")
    config = _config(dataset)
    workdir = _write_run(base / "runs" / config.run_id, config)
    out = {"workdir": workdir}
    out["default"] = evaluate_run(EvalRequest(run=workdir, **SMALL))
    out["default_tree"] = _tree_digest(workdir / "samples") | {
        f"metrics/{k}": v for k, v in _tree_digest(workdir / "metrics").items()
    }
    out["high"] = evaluate_run(EvalRequest(run=workdir, delta=0.02, **SMALL))
    out["explicit"] = evaluate_run(EvalRequest(run=workdir, delta=DEFAULT_DELTA, **SMALL))
    return out


def test_a_delta_writes_to_its_own_trees_and_leaves_the_default_ones_alone(delta_runs):
    workdir = delta_runs["workdir"]
    for tag in ("0.02", "0.0125"):
        assert (workdir / f"samples_delta-{tag}" / "000750" / "lsd" / "samples.npy").exists()
        assert (workdir / f"samples_delta-{tag}" / "000750" / "heldout" / "samples.npy").exists()
        assert (workdir / f"metrics_delta-{tag}" / "summary.json").exists()
        assert (workdir / f"metrics_delta-{tag}" / "final.json").exists()
        assert (workdir / f"metrics_delta-{tag}" / "ckpt_000750.json").exists()
    after = _tree_digest(workdir / "samples") | {
        f"metrics/{k}": v for k, v in _tree_digest(workdir / "metrics").items()
    }
    assert after == delta_runs["default_tree"]


def test_the_delta_is_recorded_only_when_overridden(delta_runs):
    workdir = delta_runs["workdir"]
    for name in ("ckpt_000750.json", "final.json"):
        assert "delta" not in json.loads((workdir / "metrics" / name).read_text())
        assert json.loads((workdir / "metrics_delta-0.02" / name).read_text())["delta"] == 0.02
    assert "delta" not in delta_runs["default"]["sampling"]
    assert delta_runs["high"]["sampling"]["delta"] == 0.02
    assert delta_runs["high"]["final"]["delta"] == 0.02
    for name in ("lsd", "final", "heldout"):
        default = json.loads((workdir / "samples" / "000750" / name / "request.json").read_text())
        high = json.loads(
            (workdir / "samples_delta-0.02" / "000750" / name / "request.json").read_text()
        )
        assert "delta" not in default
        assert high["delta"] == 0.02
        assert high["signature"]["delta"] == 0.02
        assert default["signature"]["delta"] == DEFAULT_DELTA


def test_a_non_default_delta_changes_the_samples(delta_runs):
    workdir = delta_runs["workdir"]
    for name in ("lsd", "heldout"):
        default = np.load(workdir / "samples" / "000750" / name / "samples.npy")
        high = np.load(workdir / "samples_delta-0.02" / "000750" / name / "samples.npy")
        assert default.shape == high.shape
        assert not np.array_equal(default, high)
    assert delta_runs["high"]["lsd_by_step"][750] != delta_runs["default"]["lsd_by_step"][750]


def test_an_explicit_default_delta_reproduces_the_default_draw_in_its_own_tree(delta_runs):
    """Same seeds, same noise stream, same delta: the tagged tree holds the default samples."""
    workdir = delta_runs["workdir"]
    for name in ("lsd", "final", "heldout"):
        np.testing.assert_array_equal(
            np.load(workdir / "samples_delta-0.0125" / "000750" / name / "samples.npy"),
            np.load(workdir / "samples" / "000750" / name / "samples.npy"),
        )
    assert delta_runs["explicit"]["lsd_by_step"] == delta_runs["default"]["lsd_by_step"]
    assert delta_runs["explicit"]["final"]["M"] == delta_runs["default"]["final"]["M"]


def test_a_delta_tree_holding_another_noise_level_is_refused(delta_runs, tmp_path):
    workdir = delta_runs["workdir"]
    record_path = workdir / "samples_delta-0.02" / "000750" / "lsd" / "request.json"
    original = record_path.read_text()
    record = json.loads(original)
    record["signature"]["delta"] = 0.5
    record_path.write_text(json.dumps(record))
    try:
        with pytest.raises(MetricError, match="noise levels"):
            evaluate_run(EvalRequest(run=workdir, delta=0.02, **SMALL))
    finally:
        record_path.write_text(original)


def test_a_delta_tree_is_reused_on_a_second_call(delta_runs):
    summary = evaluate_run(EvalRequest(run=delta_runs["workdir"], delta=0.02, **SMALL))
    assert all(entry["reused"] for entry in summary["sampling"]["log"])
    # The reused value is read back from ckpt_000750.json, which write_json rounds.
    assert summary["lsd_by_step"][750] == pytest.approx(
        delta_runs["high"]["lsd_by_step"][750], rel=1e-5
    )


# --------------------------------------------------------------------------------------------
# --final-from-lsd
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lsd_final_run(tmp_path_factory) -> tuple[Path, dict]:
    base = tmp_path_factory.mktemp("t72_final_from_lsd")
    dataset = _build_dataset(base / "data")
    config = _config(dataset)
    workdir = _write_run(base / "runs" / config.run_id, config)
    summary = evaluate_run(EvalRequest(run=workdir, delta=0.02, final_from_lsd=True, **SMALL))
    return workdir, summary


def test_final_from_lsd_draws_no_final_set(lsd_final_run):
    workdir, summary = lsd_final_run
    tree = workdir / "samples_delta-0.02" / "000750"
    assert (tree / "lsd").is_dir() and (tree / "heldout").is_dir()
    assert not (tree / "final").exists()
    assert [entry["set"] for entry in summary["sampling"]["log"]] == ["lsd", "heldout"]
    assert summary["sampling"]["final_from_lsd"] is True


def test_final_from_lsd_computes_the_final_metrics_on_the_lsd_set(lsd_final_run):
    workdir, summary = lsd_final_run
    record = json.loads((workdir / "metrics_delta-0.02" / "final.json").read_text())
    lists = ensure_seed_lists(Path(json.loads(
        (workdir / "samples_delta-0.02" / "000750" / "lsd" / "request.json").read_text()
    )["dataset_root"]))
    assert record["final_set"] == "lsd"
    assert record["seed_list_sha256"] == lists.intermediate.sha256
    assert record["lsd"] == record["intermediate_lsd"]
    assert record["lsd"] == pytest.approx(summary["lsd_by_step"][750], rel=1e-5)
    assert record["n_samples"] == SMALL["n_lsd"]
    assert np.load(workdir / "metrics_delta-0.02" / "final_memorisation_per_sample_d.npy").shape \
        == (SMALL["n_lsd"],)


def test_a_final_record_of_the_other_final_set_is_never_reused(lsd_final_run):
    workdir, _ = lsd_final_run
    summary = evaluate_run(EvalRequest(run=workdir, delta=0.02, **SMALL))
    record = json.loads((workdir / "metrics_delta-0.02" / "final.json").read_text())
    assert "final_set" not in record
    assert (workdir / "samples_delta-0.02" / "000750" / "final" / "samples.npy").exists()
    assert summary["final"]["n_samples"] == SMALL["n_final"]
    again = evaluate_run(EvalRequest(run=workdir, delta=0.02, final_from_lsd=True, **SMALL))
    assert again["final"]["final_set"] == "lsd"


# --------------------------------------------------------------------------------------------
# W = 128 end to end
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def run_128(tmp_path_factory) -> tuple[Path, Path, dict]:
    base = tmp_path_factory.mktemp("t72_w128")
    dataset = _build_dataset(base / "data", dataset_id="synthetic128", image_size=128)
    config = _config(dataset, image_size=128)
    workdir = _write_run(base / "runs" / config.run_id, config)
    summary = evaluate_run(EvalRequest(run=workdir, delta=0.02, **SMALL))
    return workdir, dataset, summary


def test_a_128_run_is_evaluated_on_the_128_grid(run_128):
    workdir, _, summary = run_128
    grid = spectral_grid(128)
    ckpt = json.loads((workdir / "metrics_delta-0.02" / "ckpt_000750.json").read_text())
    final = json.loads((workdir / "metrics_delta-0.02" / "final.json").read_text())
    # write_json sorts the keys, so the octaves are compared as sets.
    assert set(ckpt["lsd_octaves"]) == set(grid.octave_labels)
    assert set(final["lsd_octaves"]) == set(grid.octave_labels)
    assert "64-96" not in final["lsd_octaves"]
    assert final["low_band_sigma_px"] == pytest.approx(16.0 / 3.0)
    assert max(final["radial"]["centres"]) < 64.0
    assert final["radial"]["n_bins"] + final["radial"]["n_bins_dropped"] == 48
    samples = np.load(workdir / "samples_delta-0.02" / "000750" / "lsd" / "samples.npy")
    assert samples.shape == (SMALL["n_lsd"], 1, 128, 128)
    for key in ("lsd", "variance_ratio", "M", "M_lp", "diversity_pix", "diversity_lp"):
        assert math.isfinite(summary["final"][key]), key


def test_the_128_low_pass_metrics_use_the_scaled_length_scale(run_128):
    """D_lp of the 128 run is the low-pass at 16 * 128 / 192 px, not at 16 px."""
    from ihdm.metrics.diversity import within_seed_diversity

    workdir, _, summary = run_128
    held = np.load(workdir / "samples_delta-0.02" / "000750" / "heldout" / "samples.npy")
    scaled = within_seed_diversity(held, sigma_lp=32.0 / 3.0).D_lp_mean
    frozen = within_seed_diversity(held, sigma_lp=16.0).D_lp_mean
    assert summary["final"]["diversity_lp"] == scaled
    assert scaled != frozen


def test_an_image_size_mismatch_between_run_and_dataset_is_refused(run_128, tmp_path):
    _, dataset, _ = run_128
    config = _config(dataset, image_size=96)
    workdir = _write_run(tmp_path / "mismatch", config)
    with pytest.raises(MetricError, match="image_size"):
        evaluate_run(EvalRequest(run=workdir, **SMALL))


def test_prepare_eval_works_on_a_128_dataset(run_128, tmp_path, monkeypatch):
    """The one-writer preparation draws, checks and caches on a 128² dataset."""
    import ihdm.metrics.inception as inception
    from slurm.eval import prepare_eval

    _, dataset, _ = run_128
    seen: list[tuple[int, ...]] = []

    def fake_features(images, device="cuda", batch=64):
        stack = np.asarray(images)
        seen.append(tuple(stack.shape))
        rng = np.random.default_rng(int(stack.sum()) % (2**32))
        return rng.normal(size=(stack.shape[0], inception.INCEPTION_FEATURE_DIM))

    monkeypatch.setattr(inception, "inception_features", fake_features)
    monkeypatch.setattr(inception, "inception_weights_path", lambda: tmp_path / "weights.pt")

    # A fresh copy of the dataset inputs, so that the cache files written here are new files.
    data_root = tmp_path / "data"
    target = data_root / dataset.name
    target.mkdir(parents=True)
    for name in prepare_eval.DATASET_INPUTS:
        (target / name).write_bytes((dataset / name).read_bytes())
    lists = ensure_seed_lists(prepare_eval._shadow_dataset(target, tmp_path / "dry" / "x"))
    expected = tmp_path / "expected.csv"
    expected.write_text(
        "dataset,splits_json_sha256,intermediate_sha256,final_sha256\n"
        f"{dataset.name},{prepare_eval._sha256_file(target / 'splits.json')},"
        f"{lists.intermediate.sha256},{lists.final.sha256}\n"
    )

    records = prepare_eval.prepare(
        data_root, [dataset.name], expected, tmp_path / "shadow", "cpu", 16
    )
    assert len(records) == 1
    assert sorted(records[0]["files_written"]) == sorted(prepare_eval.CACHE_FILES)
    assert seen and all(shape[1:] == (128, 128) for shape in seen)
    again = prepare_eval.prepare(
        data_root, [dataset.name], expected, tmp_path / "shadow", "cpu", 16
    )
    assert again[0]["files_written"] == []


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def test_the_cli_defaults_leave_delta_and_final_from_lsd_off():
    request = request_from_args(build_parser().parse_args(["--run", "/x"]))
    assert request.delta is None
    assert request.final_from_lsd is False


def test_the_cli_parses_delta_and_final_from_lsd():
    request = request_from_args(
        build_parser().parse_args(
            ["--run", "/x", "--delta", "0.02", "--final-from-lsd", "--amp", "fp16"]
        )
    )
    assert request.delta == 0.02
    assert request.final_from_lsd is True
    assert metrics_dirname(request.amp, request.delta) == "metrics_amp-fp16_delta-0.02"
