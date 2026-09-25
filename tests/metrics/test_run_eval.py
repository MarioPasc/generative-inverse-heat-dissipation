"""End-to-end tests of ``ihdm.metrics.run_eval`` on a tiny synthetic run.

Two departures from the ticket's fixture, both forced by the frozen metrics:

* **96² images, not 32².** ``ihdm.metrics.spectral`` raises when an octave bin of the ``05`` §1
  grid holds no mode, and that grid runs to 96 cycles per image. At ``W = 32`` the largest radial
  index is ``sqrt(2)·31 = 43.8``, i.e. 21.9 cycles per image, so the two finest octaves are empty
  and every ``lsd()`` call raises. Populating ``[64, 96]`` needs ``sqrt(2)(W-1) >= 128``, i.e.
  ``W >= 92``.
* **14 subjects, not 8.** ``memorisation_ratio``'s denominator is the ``ref`` split with the seed
  subjects removed. With 8 subjects and ``n_seed=2`` the ``ref`` split *is* the two seed
  subjects, so that set is empty.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from ihdm.data.format import DatasetMeta, split_by_subject, write_dataset
from ihdm.metrics.errors import MetricError
from ihdm.metrics.io import read_json
from ihdm.metrics.run_eval import (
    AMP_MODES,
    EVALUATED_STEPS,
    EvalRequest,
    checkpoint_table,
    ensure_seed_lists,
    evaluate_run,
    load_views,
    metrics_dirname,
    samples_dirname,
    select_checkpoints,
)
from ihdm.train.checkpoints import ema_checkpoint_path
from ihdm.train.manifest import config_sha256, write_config_json

N_SUBJECTS = 14
N_SLICES = 8
IMAGE_SIZE = 96
FIXTURE_STEPS = (250, 750)


def _build_dataset(parent: Path, dataset_id: str = "synthetic96") -> Path:
    """Write a 112-image, 96x96 standard-format dataset with a structured spectrum."""
    root = Path(parent) / dataset_id
    n_images = N_SUBJECTS * N_SLICES
    rng = np.random.default_rng(11)

    # A smooth, subject-dependent field: the metrics need a resolvable spectrum in every octave,
    # which white noise gives but a constant image does not.
    grid = np.linspace(-1.0, 1.0, IMAGE_SIZE)
    yy, xx = np.meshgrid(grid, grid, indexing="ij")
    images = np.empty((n_images, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    for row in range(n_images):
        subject = row // N_SLICES
        field = (
            0.5
            + 0.25 * np.sin(2.0 * np.pi * (1.0 + 0.1 * subject) * xx)
            + 0.15 * np.cos(2.0 * np.pi * (2.0 + 0.05 * row) * yy)
            + 0.05 * rng.normal(size=(IMAGE_SIZE, IMAGE_SIZE))
        )
        images[row] = np.clip(field * 255.0, 0, 255).astype(np.uint8)

    subjects = [f"SUBJ{s:03d}" for s in range(N_SUBJECTS) for _ in range(N_SLICES)]
    slices = [sl for _ in range(N_SUBJECTS) for sl in range(N_SLICES)]
    splits = split_by_subject(subjects, rng_seed=2026, train_frac=0.8, n_seed=2)
    train_set, seed_set = set(splits["train"]), set(splits["seed"])
    fine = ["train" if i in train_set else ("seed" if i in seed_set else "ref") for i in
            range(n_images)]

    index = pd.DataFrame(
        {
            "idx": np.arange(n_images),
            "subject": subjects,
            "slice": slices,
            "z_mm": [float(sl) for sl in slices],
            "source": [f"synthetic/{s}/{sl}" for s, sl in zip(subjects, slices, strict=True)],
            "split": fine,
        }
    )
    meta = DatasetMeta(
        dataset_id=dataset_id,
        n_images=n_images,
        image_size=IMAGE_SIZE,
        dtype="uint8",
        pipeline="tests.metrics.test_run_eval",
        pipeline_version="1.0",
        git_sha="test",
        created="2026-09-23T00:00:00",
        raw_root=str(root),
        parameters={"note": "synthetic fixture for T4.3 evaluate_run"},
        counts={
            "subjects_total": N_SUBJECTS,
            "subjects_used": N_SUBJECTS,
            "slices_per_subject": N_SLICES,
        },
    )
    write_dataset(root, images, index, splits, meta)
    return root


def _config(dataset_root: Path):
    """Return the smoke config at 96², pointed at the fixture dataset, on the CPU."""
    from configs.spectral import smoke

    config = smoke.get_config()
    config.data.root = str(dataset_root.parent)
    config.data.dataset = dataset_root.name
    config.dataset_id = dataset_root.name
    config.data.image_size = IMAGE_SIZE
    config.device = torch.device("cpu")
    return config


def _write_run(workdir: Path, config, steps=FIXTURE_STEPS) -> Path:
    """Write a run directory with a config and one randomly initialised EMA checkpoint per step."""
    from model_code.unet import UNetModel

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    write_config_json(workdir, config)
    (workdir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": config.run_id,
                "dataset_id": config.dataset_id,
                "arm": config.arm,
                "seed": int(config.seed),
            }
        )
    )
    for offset, step in enumerate(steps):
        torch.manual_seed(offset)
        model = UNetModel(config)
        path = ema_checkpoint_path(workdir, step)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "step": int(step),
                "ema_state_dict": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
                "schedule": {
                    "name": config.model.blur_schedule_name,
                    "sha256": config.model.blur_schedule_sha256,
                    "values": [
                        float(v) for v in np.asarray(config.model.blur_schedule).ravel()
                    ],
                },
                "run_id": config.run_id,
                "config_sha256": config_sha256(config),
            },
            path,
        )
    return workdir


@pytest.fixture(scope="module")
def tiny_eval_run(tmp_path_factory) -> tuple[Path, Path]:
    """A run directory with two EMA checkpoints and its 96² dataset."""
    base = tmp_path_factory.mktemp("t43")
    dataset = _build_dataset(base / "data")
    config = _config(dataset)
    workdir = _write_run(base / "runs" / config.run_id, config)
    return workdir, dataset


@pytest.fixture(scope="module")
def evaluated(tiny_eval_run) -> tuple[dict, Path, Path]:
    """Run ``evaluate_run`` once for the whole module; the CLI is slow enough to share."""
    workdir, dataset = tiny_eval_run
    summary = evaluate_run(
        EvalRequest(
            run=workdir,
            ckpts="250,750",
            n_lsd=8,
            n_final=8,
            n_seeds=2,
            n_per_seed=3,
            sample_batch=8,
            skip_inception=True,
            device="cpu",
        )
    )
    return summary, workdir, dataset


# --------------------------------------------------------------------------------------------
# The frozen seed lists
# --------------------------------------------------------------------------------------------


def test_ensure_seed_lists_writes_both_lists_with_sidecars(tiny_eval_run):
    """Both lists land in the dataset directory with a sidecar naming the rule and the dataset."""
    _, dataset = tiny_eval_run
    lists = ensure_seed_lists(dataset)

    assert lists.intermediate.path.name == "eval_seeds_500.npy"
    assert lists.final.path.name == "eval_seeds_final_2000.npy"
    assert lists.intermediate.path.exists() and lists.intermediate.sidecar.exists()
    assert lists.final.path.exists() and lists.final.sidecar.exists()

    sidecar = json.loads(lists.intermediate.sidecar.read_text())
    assert sidecar["sha256"] == lists.intermediate.sha256
    assert "default_rng(2026)" in sidecar["rule"]
    assert json.loads(lists.final.sidecar.read_text())["rule"].endswith("default_rng(0)")


def test_the_seed_lists_are_inside_the_training_split(tiny_eval_run):
    """The D17 guard: no index of either list may fall outside ``splits['train']``."""
    _, dataset = tiny_eval_run
    lists = ensure_seed_lists(dataset)
    train = set(json.loads((dataset / "splits.json").read_text())["train"])
    assert set(lists.intermediate.idx.tolist()) <= train
    assert set(lists.final.idx.tolist()) <= train


def test_the_intermediate_list_is_distinct_and_the_final_one_is_not(tiny_eval_run):
    """500 without replacement (capped at the split), 2000 with replacement (``05`` §8a)."""
    _, dataset = tiny_eval_run
    lists = ensure_seed_lists(dataset)
    n_train = len(json.loads((dataset / "splits.json").read_text())["train"])
    assert lists.intermediate.idx.size == min(500, n_train)
    assert np.unique(lists.intermediate.idx).size == lists.intermediate.idx.size
    assert lists.final.idx.size == 2000
    assert np.unique(lists.final.idx).size < lists.final.idx.size


def test_ensure_seed_lists_is_idempotent(tiny_eval_run):
    """A second call reads the files back and returns the same digests."""
    _, dataset = tiny_eval_run
    first = ensure_seed_lists(dataset)
    second = ensure_seed_lists(dataset)
    assert first.intermediate.sha256 == second.intermediate.sha256
    assert first.final.sha256 == second.final.sha256


def test_ensure_seed_lists_rejects_a_tampered_list(tiny_eval_run, tmp_path):
    """A list holding an index outside ``train`` is refused rather than silently used."""
    _, dataset = tiny_eval_run
    copy = tmp_path / "copy"
    copy.mkdir()
    for name in ("images.npy", "index.csv", "splits.json", "meta.json"):
        (copy / name).write_bytes((dataset / name).read_bytes())
    splits = json.loads((copy / "splits.json").read_text())
    outside = sorted(set(range(N_SUBJECTS * N_SLICES)) - set(splits["train"]))[0]
    np.save(copy / "eval_seeds_500.npy", np.array([outside], dtype=np.int64))

    with pytest.raises(MetricError, match="outside"):
        ensure_seed_lists(copy)


# --------------------------------------------------------------------------------------------
# Checkpoint selection
# --------------------------------------------------------------------------------------------


def test_checkpoint_table_finds_both_checkpoints(tiny_eval_run):
    """The table is keyed by step and sorted."""
    workdir, _ = tiny_eval_run
    table = checkpoint_table(workdir)
    assert list(table) == list(FIXTURE_STEPS)
    assert all(path.exists() for path in table.values())


def test_checkpoint_table_raises_on_a_run_without_checkpoints(tmp_path):
    """An empty run directory is a caller error with a path in the message."""
    with pytest.raises(MetricError, match="no ema_iter"):
        checkpoint_table(tmp_path)


@pytest.mark.parametrize(
    ("spec", "expected", "label"),
    [
        ("final", [750], "final"),
        ("250", [250], "explicit"),
        ("250,750", [250, 750], "explicit"),
        ("750,250", [250, 750], "explicit"),
        (" 250 , 750 ", [250, 750], "explicit"),
    ],
)
def test_select_checkpoints_forms(tiny_eval_run, spec, expected, label):
    """``final``, a single step and a comma list, in any order and with spaces."""
    workdir, _ = tiny_eval_run
    steps, selection = select_checkpoints(checkpoint_table(workdir), spec)
    assert steps == expected
    assert selection == label


def test_select_checkpoints_all_falls_back_when_no_evaluated_step_exists(tiny_eval_run):
    """A short pilot has none of 5000..40000; ``all`` then means every checkpoint it holds."""
    workdir, _ = tiny_eval_run
    steps, selection = select_checkpoints(checkpoint_table(workdir), "all")
    assert steps == list(FIXTURE_STEPS)
    assert selection == "all-available"


def test_select_checkpoints_all_picks_the_eight_evaluated_steps():
    """On a real 40k run ``all`` selects exactly the eight steps of D16."""
    table = {step: Path(f"ema_iter_{step:06d}.pt") for step in range(2500, 40001, 2500)}
    steps, selection = select_checkpoints(table, "all")
    assert steps == list(EVALUATED_STEPS)
    assert selection == "all-evaluated"


@pytest.mark.parametrize("spec", ["", "   ", "nonsense", "1234"])
def test_select_checkpoints_rejects_bad_specifications(tiny_eval_run, spec):
    """An empty, unparsable or absent step is refused."""
    workdir, _ = tiny_eval_run
    with pytest.raises(MetricError):
        select_checkpoints(checkpoint_table(workdir), spec)


# --------------------------------------------------------------------------------------------
# The dataset views
# --------------------------------------------------------------------------------------------


def test_views_expose_ref_train_and_the_heldout_denominator(tiny_eval_run):
    """``ref`` is the reference stack; the ``M`` denominator is ``ref`` minus the seed subjects."""
    _, dataset = tiny_eval_run
    views = load_views(dataset)
    splits = json.loads((dataset / "splits.json").read_text())

    assert views.reference_idx.size == len(splits["ref"])
    assert views.heldout_idx.size == len(splits["ref"]) - len(splits["seed"])
    assert set(views.heldout_idx.tolist()).isdisjoint(splits["seed"])
    assert np.all(np.diff(views.train_idx) > 0)
    assert len(views.train_subjects) == views.train_idx.size


def test_train_rows_round_trip_dataset_indices(tiny_eval_run):
    """The mapping ``memorisation_ratio`` needs is asserted, not assumed (T4.2 §6)."""
    _, dataset = tiny_eval_run
    views = load_views(dataset)
    sample = views.train_idx[[0, 3, 7, -1]]
    rows = views.train_rows(sample)
    np.testing.assert_array_equal(views.train_idx[rows], sample)


def test_train_rows_refuses_an_index_outside_the_training_split(tiny_eval_run):
    """A dataset index that is not a training row would silently give a wrong seed fraction."""
    _, dataset = tiny_eval_run
    views = load_views(dataset)
    outside = np.setdiff1d(views.reference_idx, views.train_idx)[:1]
    with pytest.raises(MetricError, match="round-trip"):
        views.train_rows(outside)


# --------------------------------------------------------------------------------------------
# The end-to-end evaluation
# --------------------------------------------------------------------------------------------


def _no_nan(obj, path="") -> None:
    """Assert recursively that no float in a parsed JSON structure is ``nan`` or infinite."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            _no_nan(value, f"{path}.{key}")
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            _no_nan(value, f"{path}[{index}]")
    elif isinstance(obj, float):
        assert np.isfinite(obj), f"non-finite value at {path}: {obj}"


def test_every_result_file_of_section_9_is_written(evaluated):
    """``ckpt_<step>.json`` per checkpoint, ``final.json`` and ``summary.json``."""
    _, workdir, _ = evaluated
    metrics = workdir / "metrics"
    for step in FIXTURE_STEPS:
        assert (metrics / f"ckpt_{step:06d}.json").exists()
    assert (metrics / "final.json").exists()
    assert (metrics / "summary.json").exists()


def test_the_checkpoint_records_hold_every_key(evaluated):
    """``05`` §9's LSD keys plus the provenance the ticket adds."""
    _, workdir, _ = evaluated
    for step in FIXTURE_STEPS:
        record = read_json(workdir / "metrics" / f"ckpt_{step:06d}.json")
        assert set(record) >= {
            "lsd", "lsd_octaves", "variance_ratio", "n_samples", "n_reference",
            "checkpoint_sha256", "seed_list_sha256", "step", "sample_rng_seed", "sample_batch",
        }
        assert len(record["lsd_octaves"]) == 8
        assert record["n_samples"] == 8
        assert record["step"] == step
        _no_nan(record, f"ckpt_{step}")


def test_the_final_record_holds_every_key(evaluated):
    """§9's ``final.json`` keys: LSD, diversity, memorisation, the inherited band and the PCA."""
    _, workdir, _ = evaluated
    record = read_json(workdir / "metrics" / "final.json")
    assert set(record) >= {
        "lsd", "lsd_octaves", "variance_ratio",
        "diversity_pix", "diversity_lp", "diversity_per_seed_pix", "diversity_per_seed_lp",
        "M", "M_lp", "seed_nn_fraction", "d_samples_median", "d_heldout_median",
        "inherited_measured", "inherited_predicted",
        "inherited_measured_low_band", "inherited_predicted_low_band",
        "radial", "pca", "inception", "step", "checkpoint_sha256",
    }
    assert record["inception"] is None
    assert record["inception_note"].startswith("skipped")
    assert len(record["diversity_per_seed_pix"]) == 2
    assert record["n_per_seed"] == 3
    _no_nan(record, "final")


def test_the_inherited_radial_curve_has_its_empty_bins_dropped(evaluated):
    """``write_json`` refuses ``nan``; the bins that hold no mode are removed, not written."""
    _, workdir, _ = evaluated
    radial = read_json(workdir / "metrics" / "final.json")["radial"]
    assert radial["n_bins"] > 2
    assert radial["n_bins_dropped"] >= 0
    assert radial["n_bins"] + radial["n_bins_dropped"] == 48
    assert len(radial["centres"]) == len(radial["measured"]) == len(radial["predicted"])
    assert len(radial["centres"]) == radial["n_bins"]
    assert all(np.isfinite(radial["measured"]))


def test_the_pca_scores_are_lists_and_the_components_are_npy(evaluated):
    """The scores stay in the JSON; the components and the training scores go beside it."""
    _, workdir, _ = evaluated
    metrics = workdir / "metrics"
    pca = read_json(metrics / "final.json")["pca"]
    assert np.asarray(pca["seed_scores"]).shape == (2, 2)
    assert np.asarray(pca["sample_scores"]).shape == (2, 3, 2)
    assert (metrics / "final_pca_components.npy").exists()
    assert (metrics / "final_pca_mean.npy").exists()
    assert (metrics / "final_pca_train_scores.npy").exists()
    assert np.load(metrics / "final_pca_components.npy").shape == (2, IMAGE_SIZE, IMAGE_SIZE)


def test_the_per_sample_memorisation_arrays_are_npy_not_json(evaluated):
    """T4.2 §6: 2 000 numbers per run must not be inlined into the result file."""
    _, workdir, _ = evaluated
    metrics = workdir / "metrics"
    record = read_json(metrics / "final.json")
    assert "per_sample_d" not in record
    assert (metrics / "final_memorisation_per_sample_d.npy").exists()
    assert (metrics / "final_memorisation_per_sample_nn.npy").exists()
    assert np.load(metrics / "final_memorisation_per_sample_d.npy").shape == (8,)


def test_the_summary_merges_the_checkpoints_and_carries_the_run_identity(evaluated):
    """``summary.json`` is what the analysis reads: identity, the LSD curve and the seed lists."""
    summary, workdir, _ = evaluated
    stored = read_json(workdir / "metrics" / "summary.json")
    assert stored["run"]["run_id"] == summary["run"]["run_id"]
    assert set(stored["run"]) >= {"run_id", "dataset", "arm", "seed", "config_sha256"}
    assert sorted(int(step) for step in stored["lsd_by_step"]) == list(FIXTURE_STEPS)
    assert stored["git_sha"]
    assert stored["final_step"] == 750
    assert stored["seed_lists"]["intermediate_sha256"]
    assert stored["seed_lists"]["final_sha256"]
    assert stored["dataset_sha256"]
    _no_nan(stored, "summary")


def test_t_tau_is_null_with_a_note_when_no_threshold_is_given(evaluated):
    """``t_tau`` raises on an empty mapping (T4.1 §6); a missing threshold is not an error."""
    summary, _, _ = evaluated
    assert summary["t_tau"] is None
    assert "not computed" in summary["t_tau_note"]
    assert summary["a0_final_lsd"] is None


def test_t_tau_is_found_when_the_threshold_is_reachable(tiny_eval_run):
    """With a generous A0 threshold, ``T_tau`` is the first evaluated step."""
    workdir, _ = tiny_eval_run
    summary = evaluate_run(
        EvalRequest(
            run=workdir, ckpts="250,750", n_lsd=8, n_final=8, n_seeds=2, n_per_seed=3,
            sample_batch=8, skip_inception=True, device="cpu", a0_final_lsd=99.0,
        )
    )
    assert summary["t_tau"] == FIXTURE_STEPS[0]
    assert summary["a0_final_lsd"] == 99.0

    unreachable = evaluate_run(
        EvalRequest(
            run=workdir, ckpts="250,750", n_lsd=8, n_final=8, n_seeds=2, n_per_seed=3,
            sample_batch=8, skip_inception=True, device="cpu", a0_final_lsd=-1.0,
        )
    )
    assert unreachable["t_tau"] is None
    assert "not reached" in unreachable["t_tau_note"]


def test_the_sample_sets_are_cached_and_reused(evaluated):
    """A rerun hits the cache: the three sets exist on disk and the second run draws nothing."""
    _, workdir, _ = evaluated
    samples = workdir / "samples"
    for step in FIXTURE_STEPS:
        assert (samples / f"{step:06d}" / "lsd" / "samples.npy").exists()
        assert (samples / f"{step:06d}" / "lsd" / "seed_idx.npy").exists()
        assert (samples / f"{step:06d}" / "lsd" / "request.json").exists()
    assert (samples / "000750" / "final" / "samples.npy").exists()
    assert (samples / "000750" / "heldout" / "samples.npy").exists()

    summary = evaluate_run(
        EvalRequest(
            run=workdir, ckpts="250,750", n_lsd=8, n_final=8, n_seeds=2, n_per_seed=3,
            sample_batch=8, skip_inception=True, device="cpu",
        )
    )
    assert all(entry["reused"] for entry in summary["sampling"]["log"])
    assert all(entry["elapsed_s"] == 0.0 for entry in summary["sampling"]["log"])


def test_the_lsd_seeds_are_the_frozen_list_at_every_checkpoint(evaluated):
    """D17's common random numbers: the same seeds and the same RNG at 250 and at 750."""
    _, workdir, dataset = evaluated
    frozen = np.load(dataset / "eval_seeds_500.npy")[:8]
    first = np.load(workdir / "samples" / "000250" / "lsd" / "seed_idx.npy")
    second = np.load(workdir / "samples" / "000750" / "lsd" / "seed_idx.npy")
    np.testing.assert_array_equal(first, frozen)
    np.testing.assert_array_equal(second, frozen)

    requests = [
        json.loads((workdir / "samples" / f"{step:06d}" / "lsd" / "request.json").read_text())
        for step in FIXTURE_STEPS
    ]
    assert requests[0]["signature"]["seed_idx_sha256"] == requests[1]["signature"][
        "seed_idx_sha256"
    ]
    assert requests[0]["signature"]["rng_seed"] == requests[1]["signature"]["rng_seed"] == 2026
    assert requests[0]["signature"]["batch_size"] == requests[1]["signature"]["batch_size"]


def test_the_final_set_uses_the_other_list_and_rng_seed_zero(evaluated):
    """The 2k shared set is drawn with replacement under ``rng_seed`` 0 (``05`` §8a)."""
    _, workdir, dataset = evaluated
    frozen = np.load(dataset / "eval_seeds_final_2000.npy")[:8]
    stored = np.load(workdir / "samples" / "000750" / "final" / "seed_idx.npy")
    np.testing.assert_array_equal(stored, frozen)
    record = json.loads((workdir / "samples" / "000750" / "final" / "request.json").read_text())
    assert record["signature"]["rng_seed"] == 0


def test_the_gate_writes_its_record(tiny_eval_run):
    """``--gate a,b`` writes ``metrics/gate.json`` with the decision and the interval."""
    workdir, _ = tiny_eval_run
    summary = evaluate_run(
        EvalRequest(
            run=workdir, ckpts="750", n_lsd=8, n_final=8, n_seeds=2, n_per_seed=3,
            sample_batch=8, skip_inception=True, device="cpu", gate=(250, 750), n_boot_gate=100,
        )
    )
    record = read_json(workdir / "metrics" / "gate.json")
    assert set(record) >= {
        "lsd_a", "lsd_b", "difference", "extend", "relative_change", "n_seeds", "n_reference",
        "step_a", "step_b", "seed_list_sha256", "rule",
    }
    assert record["step_a"] == 250 and record["step_b"] == 750
    assert isinstance(record["extend"], bool)
    assert record["difference"]["ci_low"] <= record["difference"]["ci_high"]
    assert summary["gate"]["steps"] == [250, 750]
    _no_nan(record, "gate")


def test_the_gate_rejects_a_repeated_or_missing_step(tiny_eval_run):
    """Two different, present checkpoints or nothing."""
    workdir, _ = tiny_eval_run
    common = {
        "run": workdir, "ckpts": "750", "n_lsd": 8, "n_final": 8, "n_seeds": 2,
        "n_per_seed": 3, "sample_batch": 8, "skip_inception": True, "device": "cpu",
    }
    with pytest.raises(MetricError, match="two different steps"):
        evaluate_run(EvalRequest(gate=(750, 750), **common))
    with pytest.raises(MetricError, match="no checkpoint at steps"):
        evaluate_run(EvalRequest(gate=(250, 9999), **common))


# --------------------------------------------------------------------------------------------
# --amp: one cache tree per precision (T5.1)
# --------------------------------------------------------------------------------------------


def _tree_digest(root: Path) -> dict[str, tuple[str, int]]:
    """Return ``{relative path: (sha256, mtime_ns)}`` of every file under ``root``."""
    return {
        str(path.relative_to(root)): (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_mtime_ns,
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.mark.parametrize(
    "amp, samples, metrics",
    [
        ("off", "samples", "metrics"),
        ("fp16", "samples_amp-fp16", "metrics_amp-fp16"),
        ("bf16", "samples_amp-bf16", "metrics_amp-bf16"),
    ],
)
def test_every_precision_has_its_own_trees(amp, samples, metrics):
    """``off`` keeps the historical names, so fp32 caches written before ``--amp`` stay valid."""
    assert samples_dirname(amp) == samples
    assert metrics_dirname(amp) == metrics
    assert amp in AMP_MODES


@pytest.mark.parametrize("amp", ["", "on", "fp32", "FP16", "true"])
def test_an_unknown_precision_is_refused(tiny_eval_run, amp):
    """A mode outside ``AMP_MODES`` fails before anything is loaded or written."""
    workdir, _ = tiny_eval_run
    with pytest.raises(MetricError, match="amp must be one of"):
        samples_dirname(amp)
    with pytest.raises(MetricError, match="amp must be one of"):
        evaluate_run(EvalRequest(run=workdir, ckpts="250", skip_inception=True, amp=amp))


def test_the_default_precision_keeps_the_fp32_signature(evaluated):
    """``--amp off`` records ``amp: false``, the value of every cache written before the flag."""
    summary, workdir, _ = evaluated
    record = json.loads((workdir / "samples" / "000250" / "lsd" / "request.json").read_text())
    assert record["signature"]["amp"] is False
    assert record["amp"] == "off"
    assert summary["sampling"]["amp"] == "off"


def test_a_bf16_evaluation_leaves_the_fp32_cache_untouched(evaluated):
    """The bf16 sets and results land in their own trees; the fp32 trees keep every byte."""
    _, workdir, _ = evaluated
    before_samples = _tree_digest(workdir / "samples")
    before_metrics = _tree_digest(workdir / "metrics")

    summary = evaluate_run(
        EvalRequest(
            run=workdir, ckpts="250,750", n_lsd=8, n_final=8, n_seeds=2, n_per_seed=3,
            sample_batch=8, skip_inception=True, device="cpu", amp="bf16",
        )
    )

    assert _tree_digest(workdir / "samples") == before_samples
    assert _tree_digest(workdir / "metrics") == before_metrics
    bf16 = workdir / "samples_amp-bf16"
    for step in FIXTURE_STEPS:
        record = json.loads((bf16 / f"{step:06d}" / "lsd" / "request.json").read_text())
        assert record["signature"]["amp"] == "bf16"
        assert read_json(workdir / "metrics_amp-bf16" / f"ckpt_{step:06d}.json")["amp"] == "bf16"
    assert (bf16 / "000750" / "final" / "samples.npy").exists()
    assert (bf16 / "000750" / "heldout" / "samples.npy").exists()
    assert summary["sampling"]["amp"] == "bf16"
    assert read_json(workdir / "metrics_amp-bf16" / "summary.json")["sampling"]["amp"] == "bf16"
    assert read_json(workdir / "metrics_amp-bf16" / "final.json")["amp"] == "bf16"
    assert not any(entry["reused"] for entry in summary["sampling"]["log"])


def test_a_cache_of_another_precision_is_refused_not_overwritten(tiny_eval_run, tmp_path):
    """A set found under the wrong tree raises, even with ``force``, and keeps its bytes."""
    source, _ = tiny_eval_run
    workdir = tmp_path / source.name
    shutil.copytree(source, workdir, ignore=shutil.ignore_patterns("samples_amp-*", "metrics*"))
    evaluate_run(
        EvalRequest(
            run=workdir, ckpts="250", n_lsd=8, sample_batch=8, skip_inception=True,
            device="cpu",
        )
    )
    misplaced = workdir / "samples_amp-bf16" / "000250" / "lsd"
    shutil.copytree(workdir / "samples" / "000250" / "lsd", misplaced)
    before = _tree_digest(misplaced)

    with pytest.raises(MetricError, match="refusing to mix or overwrite precisions"):
        evaluate_run(
            EvalRequest(
                run=workdir, ckpts="250", n_lsd=8, sample_batch=8, skip_inception=True,
                device="cpu", amp="bf16", force=True,
            )
        )
    assert _tree_digest(misplaced) == before


def test_the_gate_samples_and_records_under_the_requested_precision(tiny_eval_run, tmp_path):
    """``--gate`` with ``--amp`` draws into the mode's tree and says so in ``gate.json``."""
    source, _ = tiny_eval_run
    workdir = tmp_path / source.name
    shutil.copytree(source, workdir, ignore=shutil.ignore_patterns("samples*", "metrics*"))
    evaluate_run(
        EvalRequest(
            run=workdir, ckpts="250", n_lsd=8, sample_batch=8, skip_inception=True,
            device="cpu", gate=(250, 750), n_boot_gate=50, amp="bf16",
        )
    )
    record = read_json(workdir / "metrics_amp-bf16" / "gate.json")
    assert record["amp"] == "bf16"
    assert (workdir / "samples_amp-bf16" / "000750" / "lsd" / "samples.npy").exists()
    assert not (workdir / "samples").exists()
    assert not (workdir / "metrics").exists()
