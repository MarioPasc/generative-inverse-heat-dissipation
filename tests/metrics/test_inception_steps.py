"""Tests of T7.3: ``evaluate_run`` with ``inception_steps`` (per-checkpoint Inception and ``M``).

The default path is pinned against the base commit by ``test_regression_t72.py``; this module
tests what is new. The Inception network is replaced by a deterministic per-image feature map
(a hash of the image's bytes seeds a normal vector), so equal images give equal features and the
real ``bootstrap_inception``, ``memorisation_ratio`` and bookkeeping run unchanged on the CPU.
Two runs are exercised: the 96² fixture of ``test_run_eval`` (two checkpoints) and a synthetic
128² run (the W rule of ``spectral_grid``).
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
from ihdm.metrics.run_eval import EvalRequest, evaluate_run, metrics_dirname
from ihdm.metrics.spectral import spectral_grid
from tests.metrics.test_run_eval import _build_dataset, _config, _write_run

#: The per-step request on the fixture: both checkpoints, Inception at both, few resamples.
BASE: dict = {
    "ckpts": "250,750", "n_lsd": 12, "n_final": 8, "n_seeds": 2, "n_per_seed": 3,
    "sample_batch": 8, "device": "cpu", "n_boot_inception": 4,
}

INCEPTION_KEYS = {
    "kid", "kid_ci_low", "kid_ci_high", "fid", "fid_ci_low", "fid_ci_high", "precision",
    "recall", "density", "coverage", "k", "n_samples", "n_reference", "n_boot",
}
MEMORISATION_KEYS = {"M", "M_lp", "seed_nn_fraction", "n_samples", "per_sample_files"}


def _fake_features_factory(calls: list[int]):
    """Return a stand-in for ``inception_features`` that records the stack sizes it sees."""
    from ihdm.metrics.inception import INCEPTION_FEATURE_DIM

    def fake_features(images, device="cuda", batch=64):
        stack = np.asarray(images)
        calls.append(int(stack.shape[0]))
        rows = []
        for image in stack:
            seed = int.from_bytes(hashlib.sha256(image.tobytes()).digest()[:8], "little")
            rows.append(np.random.default_rng(seed).normal(size=INCEPTION_FEATURE_DIM))
        return np.asarray(rows, dtype=np.float64)

    return fake_features


@pytest.fixture(scope="module")
def fake_inception(tmp_path_factory):
    """Patch the Inception extractor and the weights path for the whole module."""
    import ihdm.metrics.inception as inception

    calls: list[int] = []
    weights = tmp_path_factory.mktemp("weights") / "inception-2015-12-05.pt"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(inception, "inception_features", _fake_features_factory(calls))
        patch.setattr(inception, "inception_weights_path", lambda: weights)
        yield calls


def _new_run(base: Path, image_size: int | None = None) -> Path:
    dataset_id = "synthetic96" if image_size is None else f"synthetic{image_size}"
    dataset = _build_dataset(base / "data", dataset_id=dataset_id, image_size=image_size)
    config = _config(dataset, image_size=image_size)
    return _write_run(base / "runs" / config.run_id, config)


@pytest.fixture(scope="module")
def per_step_run(tmp_path_factory, fake_inception) -> tuple[Path, dict]:
    """The 96² fixture evaluated at both checkpoints with Inception at both, final from LSD."""
    workdir = _new_run(tmp_path_factory.mktemp("t73_96"))
    summary = evaluate_run(
        EvalRequest(run=workdir, inception_steps=(250, 750), final_from_lsd=True, **BASE)
    )
    return workdir, summary


def _ckpt(workdir: Path, step: int, amp: str = "off") -> dict:
    return json.loads((workdir / metrics_dirname(amp) / f"ckpt_{step:06d}.json").read_text())


# --------------------------------------------------------------------------------------------
# 96² fixture
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("step", [250, 750])
def test_every_listed_step_holds_an_inception_and_a_memorisation_block(per_step_run, step):
    workdir, _ = per_step_run
    record = _ckpt(workdir, step)
    assert INCEPTION_KEYS <= set(record["inception"])
    assert MEMORISATION_KEYS <= set(record["memorisation"])
    assert record["inception"]["n_samples"] == BASE["n_lsd"]
    assert record["inception"]["n_boot"] == BASE["n_boot_inception"]
    assert record["memorisation"]["n_samples"] == BASE["n_lsd"]
    for key in ("kid", "fid", "precision", "recall", "density", "coverage"):
        assert math.isfinite(record["inception"][key]), key
    for key in ("M", "M_lp", "seed_nn_fraction"):
        assert math.isfinite(record["memorisation"][key]), key
    # The LSD block of the record is untouched by the addition.
    assert {"lsd", "lsd_octaves", "variance_ratio", "seed_list_sha256"} <= set(record)


def test_the_reference_is_the_ref_split(per_step_run):
    workdir, summary = per_step_run
    assert _ckpt(workdir, 250)["inception"]["n_reference"] == summary["env"]["n_reference"]


@pytest.mark.parametrize("step", [250, 750])
def test_the_per_sample_arrays_are_named_by_the_step(per_step_run, step):
    workdir, _ = per_step_run
    metrics = workdir / metrics_dirname()
    record = _ckpt(workdir, step)
    names = [f"ckpt_{step:06d}_memorisation_per_sample_d.npy",
             f"ckpt_{step:06d}_memorisation_per_sample_nn.npy"]
    assert record["memorisation"]["per_sample_files"] == names
    for name in names:
        assert np.load(metrics / name).shape == (BASE["n_lsd"],)
    # The final record keeps its own historical names.
    assert (metrics / "final_memorisation_per_sample_d.npy").exists()


def test_the_summary_carries_the_blocks_by_step(per_step_run):
    workdir, summary = per_step_run
    stored = json.loads((workdir / metrics_dirname() / "summary.json").read_text())
    assert sorted(stored["inception_by_step"]) == ["250", "750"]
    assert sorted(stored["memorisation_by_step"]) == ["250", "750"]
    assert stored["sampling"]["inception_steps"] == [250, 750]
    for step in (250, 750):
        record = _ckpt(workdir, step)
        assert stored["inception_by_step"][str(step)]["kid"] == record["inception"]["kid"]
        assert stored["inception_by_step"][str(step)]["precision"] == \
            record["inception"]["precision"]
        assert stored["memorisation_by_step"][str(step)]["M"] == record["memorisation"]["M"]
    assert summary["inception_by_step"][250]["recall"] == pytest.approx(
        _ckpt(workdir, 250)["inception"]["recall"], rel=1e-5
    )


def test_the_two_steps_are_scored_on_their_own_samples(tmp_path, fake_inception, monkeypatch):
    """Each block is computed from its own step's LSD set (the untrained fixture models draw
    bitwise-equal samples at both steps, so the numbers alone cannot show it)."""
    from ihdm.metrics import run_eval

    seen: list[tuple[str, int]] = []
    original = run_eval._inception_record

    def spy(request, views, sample_set, device):
        seen.append((sample_set.name, int(sample_set.directory.parent.name)))
        return original(request, views, sample_set, device)

    monkeypatch.setattr(run_eval, "_inception_record", spy)
    workdir = _new_run(tmp_path)
    evaluate_run(EvalRequest(run=workdir, inception_steps=(250, 750), **BASE))
    # The two per-step blocks, then the final record (its own 8-seed final set at 750).
    assert seen == [("lsd", 250), ("lsd", 750), ("final", 750)]
    a, b = _ckpt(workdir, 250), _ckpt(workdir, 750)
    assert a["checkpoint_sha256"] != b["checkpoint_sha256"]
    assert a["samples_dir"].endswith("000250/lsd") and b["samples_dir"].endswith("000750/lsd")


def test_at_the_final_step_the_block_equals_the_final_record_from_the_same_set(per_step_run):
    """With final_from_lsd the final record and the 750 block score the same 12 samples."""
    workdir, _ = per_step_run
    final = json.loads((workdir / metrics_dirname() / "final.json").read_text())
    block = _ckpt(workdir, 750)
    assert final["final_set"] == "lsd"
    for key in ("kid", "fid", "precision", "recall", "density", "coverage"):
        assert block["inception"][key] == final["inception"][key], key
    for key in ("M", "M_lp", "seed_nn_fraction"):
        assert block["memorisation"][key] == final[key], key


def test_a_second_call_reuses_the_blocks(per_step_run, fake_inception):
    workdir, _ = per_step_run
    before = {step: _ckpt(workdir, step) for step in (250, 750)}
    n_calls = len(fake_inception)
    evaluate_run(EvalRequest(run=workdir, inception_steps=(250, 750), final_from_lsd=True, **BASE))
    after = {step: _ckpt(workdir, step) for step in (250, 750)}
    assert after == before
    # Only the final record (reused too) could have called the extractor; nothing did.
    assert len(fake_inception) == n_calls


def test_a_default_record_gets_the_blocks_when_asked_later(tmp_path, fake_inception):
    """A ckpt record written without the option is completed from the reused LSD set."""
    workdir = _new_run(tmp_path)
    small = {**BASE, "ckpts": "250"}
    plain = evaluate_run(EvalRequest(run=workdir, skip_inception=True, **small))
    record = _ckpt(workdir, 250)
    assert "inception" not in record and "memorisation" not in record
    assert "inception_by_step" not in plain
    samples = workdir / "samples" / "000250" / "lsd" / "samples.npy"
    digest = hashlib.sha256(samples.read_bytes()).hexdigest()

    summary = evaluate_run(EvalRequest(run=workdir, inception_steps=(250,), **small))
    completed = _ckpt(workdir, 250)
    assert hashlib.sha256(samples.read_bytes()).hexdigest() == digest
    assert summary["sampling"]["log"][0]["reused"] is True
    assert completed["lsd"] == record["lsd"]
    assert INCEPTION_KEYS <= set(completed["inception"])
    assert summary["inception_by_step"][250]["kid"] == pytest.approx(
        completed["inception"]["kid"], rel=1e-5
    )


def test_the_default_path_writes_no_per_step_block(tmp_path):
    workdir = _new_run(tmp_path)
    summary = evaluate_run(EvalRequest(run=workdir, skip_inception=True, **BASE))
    for step in (250, 750):
        assert not {"inception", "memorisation"} & set(_ckpt(workdir, step))
    assert not {"inception_by_step", "memorisation_by_step"} & set(summary)
    assert "inception_steps" not in summary["sampling"]
    assert not list((workdir / metrics_dirname()).glob("ckpt_*_memorisation_*.npy"))


@pytest.mark.parametrize(
    ("inception_steps", "extra", "match"),
    [
        ((500,), {}, "not among the evaluated"),
        ((250, 1000), {}, "not among the evaluated"),
        ((), {}, "names no step"),
        ((250,), {"skip_inception": True}, "skip_inception"),
    ],
)
def test_bad_inception_steps_are_refused(tmp_path, inception_steps, extra, match):
    workdir = _new_run(tmp_path)
    with pytest.raises(MetricError, match=match):
        evaluate_run(EvalRequest(run=workdir, inception_steps=inception_steps,
                                 **{**BASE, **extra}))


def test_the_amp_tree_holds_the_blocks(tmp_path, fake_inception):
    """The blocks follow the request's trees (here bf16), as every other record does."""
    workdir = _new_run(tmp_path)
    evaluate_run(EvalRequest(run=workdir, inception_steps=(750,), amp="bf16",
                             **{**BASE, "ckpts": "750"}))
    assert INCEPTION_KEYS <= set(_ckpt(workdir, 750, amp="bf16")["inception"])
    assert not (workdir / "metrics").exists()


# --------------------------------------------------------------------------------------------
# 128² run
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def run_128(tmp_path_factory, fake_inception) -> tuple[Path, dict]:
    workdir = _new_run(tmp_path_factory.mktemp("t73_128"), image_size=128)
    summary = evaluate_run(
        EvalRequest(run=workdir, inception_steps=(250, 750), final_from_lsd=True, **BASE)
    )
    return workdir, summary


def test_a_128_run_gets_the_blocks_on_the_128_grid(run_128):
    workdir, summary = run_128
    grid = spectral_grid(128)
    for step in (250, 750):
        record = _ckpt(workdir, step)
        assert set(record["lsd_octaves"]) == set(grid.octave_labels)
        assert INCEPTION_KEYS <= set(record["inception"])
        assert record["inception"]["n_samples"] == BASE["n_lsd"]
        assert math.isfinite(record["memorisation"]["M_lp"])
    samples = np.load(workdir / "samples" / "000750" / "lsd" / "samples.npy")
    assert samples.shape == (BASE["n_lsd"], 1, 128, 128)
    assert sorted(summary["inception_by_step"]) == [250, 750]


def test_the_128_memorisation_uses_the_scaled_low_pass(run_128):
    """M_lp of a per-step block is the low-pass at 16 * 128 / 192 px, as in final.json."""
    workdir, _ = run_128
    final = json.loads((workdir / metrics_dirname() / "final.json").read_text())
    block = _ckpt(workdir, 750)["memorisation"]
    assert block["M_lp"] == final["M_lp"]
    assert block["M"] == final["M"]


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def test_the_cli_default_leaves_inception_steps_off():
    assert request_from_args(build_parser().parse_args(["--run", "/x"])).inception_steps is None


def test_the_cli_parses_inception_steps():
    request = request_from_args(
        build_parser().parse_args(
            ["--run", "/x", "--ckpts", "45000,50000,55000,60000",
             "--inception-steps", "45000, 50000,55000,60000"]
        )
    )
    assert request.inception_steps == (45000, 50000, 55000, 60000)


@pytest.mark.parametrize("value", ["", " , ", "45k", "45000;50000"])
def test_the_cli_refuses_a_malformed_list(value):
    with pytest.raises(MetricError):
        request_from_args(build_parser().parse_args(["--run", "/x", "--inception-steps", value]))
