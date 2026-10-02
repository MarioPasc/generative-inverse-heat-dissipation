"""Tests of ``slurm/diag_eval/photo_collect.py`` (T7.3): task rows, late window, the reading."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ihdm.metrics.run_eval import metrics_dirname, samples_dirname
from slurm.diag_eval import photo_collect as pc

SEED_SHA = "a" * 64


def _ckpt(step: int, precision: float, kid: float, lsd: float = 1.0, **over) -> dict:
    record = {
        "step": step, "lsd": lsd, "variance_ratio": 0.3, "lsd_octaves": {"0.5-1": 0.1},
        "n_samples": 500, "checkpoint_sha256": f"{step:064d}", "seed_list_sha256": SEED_SHA,
        "sample_rng_seed": 2026, "sample_batch": 32, "amp": "fp16",
        "inception": {"kid": kid, "kid_ci_low": kid / 2, "kid_ci_high": kid * 2, "fid": 100.0,
                      "fid_ci_low": 90.0, "fid_ci_high": 110.0, "precision": precision,
                      "recall": 0.01, "density": 0.002, "coverage": 0.003, "k": 5,
                      "n_samples": 500, "n_reference": 800, "n_boot": 200},
        "memorisation": {"M": 1.5, "M_lp": 1.4, "seed_nn_fraction": 0.0,
                         "d_samples_median": 1.0, "d_heldout_median": 0.7, "n_samples": 500},
    }
    record.update(over)
    return record


def _tree(root: Path, run_id: str, precisions, kids, lsds=None) -> Path:
    metrics = root / run_id / metrics_dirname("fp16")
    metrics.mkdir(parents=True)
    lsds = lsds or [1.0] * 4
    for step, p, k, lsd in zip(pc.LATE_STEPS, precisions, kids, lsds, strict=True):
        (metrics / f"ckpt_{step:06d}.json").write_text(json.dumps(_ckpt(step, p, k, lsd)))
    (metrics / "summary.json").write_text(json.dumps({
        "run": {"run_id": run_id, "dataset": "d", "arm": "A0", "seed": 1},
        "dataset_sha256": "b" * 64, "git_sha": "c" * 40,
        "seed_lists": {"intermediate_sha256": SEED_SHA}, "sampling": {"amp": "fp16"},
        "env": {"n_reference": 800},
    }))
    (metrics / "final.json").write_text(json.dumps({"step": 60000, "final_set": "lsd",
                                                    "diversity_pix": 0.01}))
    return root / run_id


def _task(tmp_path: Path, run_id: str, precisions, kids, lsds=None) -> dict:
    shadow = _tree(tmp_path, run_id, precisions, kids, lsds)
    return pc.collect_task(shadow, run_id, "fp16", list(pc.LATE_STEPS), {"job_id": "1"})


@pytest.mark.parametrize(
    ("precision", "kid", "expected"),
    [
        (0.10, 0.10, True),            # every condition exactly at its threshold
        (0.0999, 0.10, False),         # below the 0.10 floor
        (0.50, 0.1001, False),         # KID above 0.5x the baseline's
        (0.12, 0.05, True),
    ],
)
def test_the_rule_is_applied_literally(precision, kid, expected):
    baseline = {"precision": 0.005, "kid": 0.20}
    result = pc.lifts({"precision": precision, "kid": kid}, baseline)
    assert result["lifts"] is expected
    assert result["kid_threshold"] == pytest.approx(0.10)


def test_the_tenfold_condition_binds_when_the_baseline_is_not_tiny():
    result = pc.lifts({"precision": 0.15, "kid": 0.01}, {"precision": 0.02, "kid": 0.2})
    assert result["conditions"] == {
        "precision_ge_0.10": True, "precision_ge_10x_baseline": False,
        "kid_le_0.5x_baseline": True,
    }
    assert result["lifts"] is False
    assert result["precision_threshold"] == pytest.approx(0.2)


def test_a_zero_baseline_precision_gives_no_ratio():
    result = pc.lifts({"precision": 0.2, "kid": 0.01}, {"precision": 0.0, "kid": 0.2})
    assert result["precision_ratio"] is None
    assert result["lifts"] is True


@pytest.mark.parametrize(
    ("r128", "n32k", "text"),
    [
        (True, False, "resolution and framing"),
        (False, True, "data size"),
        (True, True, "either change suffices"),
        (False, False, "the budget or recipe"),
    ],
)
def test_the_reading_table(r128, n32k, text):
    assert pc.reading(r128, n32k).startswith(text)


def test_the_late_window_is_the_mean_of_the_four_checkpoints(tmp_path):
    task = _task(tmp_path, "x", [0.0, 0.01, 0.02, 0.03], [0.4, 0.3, 0.2, 0.1])
    late = pc.late_window(task)
    assert late["precision"] == pytest.approx(0.015)
    assert late["kid"] == pytest.approx(0.25)
    assert late["n_checkpoints"] == 4
    assert pc.check_protocol(task) == []


def test_a_record_without_the_per_step_blocks_is_refused():
    record = _ckpt(45000, 0.1, 0.1)
    del record["inception"]
    with pytest.raises(pc.CollectError, match="no per-step"):
        pc.step_row(record)
    with pytest.raises(pc.CollectError, match="delta"):
        pc.step_row(_ckpt(45000, 0.1, 0.1, delta=0.02))


def test_off_protocol_rows_are_reported(tmp_path):
    task = _task(tmp_path, "x", [0.1] * 4, [0.1] * 4)
    task["rows"][1]["n_samples"] = 32
    task["rows"][2]["seed_list_sha256"] = "d" * 64
    problems = pc.check_protocol(task)
    assert any("n_samples=32" in item for item in problems)
    assert any("frozen 500-seed list" in item for item in problems)


def _three(tmp_path: Path, r128_precision: float, n32k_precision: float) -> list[Path]:
    paths = []
    specs = {
        pc.ROLES["baseline"]: ([0.003] * 4, [0.25] * 4,
                               [pc.BASELINE_LSD_ANCHORS[s] for s in pc.LATE_STEPS]),
        pc.ROLES["r128"]: ([r128_precision] * 4, [0.05] * 4, None),
        pc.ROLES["n32k"]: ([n32k_precision] * 4, [0.20] * 4, None),
    }
    for run_id, (p, k, lsd) in specs.items():
        task = _task(tmp_path, run_id, p, k, lsd)
        path = tmp_path / f"{run_id}_photo.json"
        path.write_text(json.dumps(task))
        paths.append(path)
    return paths


def test_merge_applies_the_reading_and_checks_the_anchor(tmp_path):
    record = pc.merge(_three(tmp_path, 0.2, 0.2), {"jobs": [1]})
    assert record["lifts"]["r128"]["lifts"] is True
    assert record["lifts"]["n32k"]["lifts"] is False     # KID 0.20 > 0.5 x 0.25
    assert record["reading"]["text"] == "resolution and framing"
    assert record["anchor"]["passed"] is True
    assert record["jobs"] == [1]
    assert set(record["tasks"]) == {"baseline", "r128", "n32k"}


def test_merge_refuses_an_off_protocol_task_unless_allowed(tmp_path):
    paths = _three(tmp_path, 0.01, 0.01)
    task = json.loads(paths[1].read_text())
    task["rows"][0]["amp"] = "off"
    paths[1].write_text(json.dumps(task))
    with pytest.raises(pc.CollectError, match="off-protocol"):
        pc.merge(paths, {})
    record = pc.merge(paths, {}, allow_off_protocol=True)
    assert record["reading"]["text"].startswith("the budget or recipe")


def test_merge_needs_each_role_once(tmp_path):
    paths = _three(tmp_path, 0.2, 0.2)
    with pytest.raises(pc.CollectError, match="lsun_church_n32k_A0_s1"):
        pc.merge(paths[:2], {})


def _lsd_set(shadow: Path, side: int, n: int = 20) -> None:
    directory = shadow / samples_dirname("fp16") / "060000" / "lsd"
    directory.mkdir(parents=True)
    rng = np.random.default_rng(side)
    np.save(directory / "samples.npy", rng.integers(0, 256, (n, 1, side, side), dtype=np.uint8))
    np.save(directory / "seeds.npy", rng.integers(0, 256, (n, side, side), dtype=np.uint8))
    np.save(directory / "seed_idx.npy", np.arange(n, dtype=np.int64) * 3)


def test_the_grid_draws_both_sides_in_one_png(tmp_path):
    _lsd_set(tmp_path / "a", 192)
    _lsd_set(tmp_path / "b", 128)
    out = tmp_path / "grids.png"
    index = tmp_path / "idx.json"
    assert pc.main(["grid", "--panel", f"baseline={tmp_path / 'a'}", "--panel",
                    f"r128={tmp_path / 'b'}", "--n", "16", "--out", str(out),
                    "--index-out", str(index)]) == 0
    assert out.stat().st_size > 0
    assert json.loads(index.read_text())["r128"] == [3 * i for i in range(16)]
    rows = pc.load_grid_rows(tmp_path / "b", "fp16", 60000, 16)
    assert rows["samples"].shape == (16, 128, 128) and rows["seeds"].shape == (16, 128, 128)
