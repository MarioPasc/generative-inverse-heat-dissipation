"""Tests of ``ihdm.cli.check_array`` on synthetic run trees (T3.5).

The trees are written directly (no training): the contract files of ``04-run-artifacts.md`` §3 with
a sparse cadence (train every 5k, eval and grids every 10k, EMA checkpoints every 2.5k), so a 40k
or 60k run costs a few dozen small files.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from ihdm.cli.check_array import CELL_KEYS, check_array, main, read_cells
from ihdm.train.logging import OCTAVE_BIN_NAMES
from ihdm.train.recipe import recipe_of

CELLS = (
    (0, "ixi", "A0", 1),
    (1, "ixi", "A0", 2),
    (2, "ixi", "A3", 1),
    (3, "lsun_church", "A0", 1),
)
IMAGE_SIZE = 8
SCHEDULES = {"A0": ("log_W", 96.0), "A3": ("matched_W8_ixi", 24.0)}


def _schedule(arm: str) -> tuple[str, list[float], str, float]:
    name, sigma_max = SCHEDULES[arm]
    values = [float(v) for v in np.linspace(0.5, sigma_max, 5)]
    sha = hashlib.sha256(json.dumps(values).encode()).hexdigest()
    return name, values, sha, sigma_max


def _config(dataset: str, arm: str, seed: int, n_iters: int) -> dict:
    name, values, sha, sigma_max = _schedule(arm)
    return {
        "run_id": f"{dataset}_{arm}_s{seed}",
        "dataset_id": dataset,
        "arm": arm,
        "seed": seed,
        "device": "cuda:0",
        "data": {"dataset": dataset, "root": "/data", "image_size": IMAGE_SIZE},
        "training": {"n_iters": n_iters, "log_every": 5000, "eval_every": 10000,
                     "ckpt_every": 2500, "grid_every": 10000, "batch_size": 16},
        "optim": {"lr": 1e-4, "warmup": 1000, "automatic_mp": True, "grad_clip": 1.0},
        "model": {"blur_schedule": values, "blur_schedule_name": name,
                  "blur_schedule_sha256": sha, "blur_sigma_max": sigma_max,
                  "blur_schedule_file": f"/schedules/{name}.npy"},
    }


def _recipe_hash(config: dict) -> str:
    payload = json.dumps(recipe_of(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _train(step: int, lr: float) -> dict:
    return {"step": step, "kind": "train", "loss": 0.3 + 1e-6 * step,
            "lr": lr * min(step / 1000, 1.0), "it_per_s": 1.7, "img_per_s": 1.7 * 16,
            "gpu_mem_peak_gb": 28.5, "grad_norm": None if step == 0 else 100.0 + step / 1000,
            "amp_scale": 64.0, "wall_s": step / 1.7,
            "loss_per_octave": {name: 0.3 for name in OCTAVE_BIN_NAMES}}


def _segment(workdir: Path, start: int, stop: int, lr: float, final: bool) -> list[dict]:
    """The records of the steps in [start, stop], ending with the run's done event."""
    records = []
    for step in range(start, stop + 1):
        if step % 5000 == 0:
            records.append(_train(step, lr))
        if step % 10000 == 0:
            records.append({"step": step, "kind": "eval", "loss": 0.3})
        if step > 0 and step % 2500 == 0:
            records.append({"step": step, "kind": "ckpt",
                            "path": str(workdir / "checkpoints" / f"ema_iter_{step:06d}.pt")})
        if step > 0 and (step % 10000 == 0 or step == stop):
            records.append({"step": step, "kind": "grid",
                            "path": str(workdir / "grids" / f"iter_{step:06d}.png")})
    if final:
        records.append({"step": stop, "kind": "done", "n_skipped": 0})
    return records


def make_run(root: Path, dataset: str, arm: str, seed: int, n_iters: int = 40000,
             extended_from: int | None = None, torch_files: bool = False) -> Path:
    """Write one healthy run directory; ``extended_from`` adds a done/resume at that step."""
    workdir = root / f"{dataset}_{arm}_s{seed}"
    for sub in ("checkpoints", "checkpoints-meta", "grids", "tensorboard"):
        (workdir / sub).mkdir(parents=True, exist_ok=True)
    config = _config(dataset, arm, seed, n_iters)
    lr = config["optim"]["lr"]
    name, values, sha, _ = _schedule(arm)
    manifest = {
        "run_id": config["run_id"], "dataset_id": dataset, "arm": arm, "seed": seed,
        "git_sha": "0" * 40, "git_dirty": False, "hostname": "exa02", "gpu": "A100",
        "python": "3.11", "torch": "2.14", "cuda": "13.0", "started": "2026-09-25",
        "slurm_job_id": "1", "data": {"root": "/data", "meta": None, "images_sha256": None},
        "schedule": {"name": name, "file": None, "sha256": sha, "values": values},
        "config_sha256": "c" * 64, "recipe_sha256": _recipe_hash(config), "n_params": 1,
        "batch_size": 16, "n_iters": n_iters,
    }
    if extended_from is None:
        records = _segment(workdir, 0, n_iters, lr, final=True)
    else:
        records = _segment(workdir, 0, extended_from, lr, final=True)
        records.append({"step": extended_from, "kind": "resume", "from": "checkpoint.pth"})
        records += _segment(workdir, extended_from + 1, n_iters, lr, final=True)
    for step in range(2500, n_iters + 1, 2500):
        (workdir / "checkpoints" / f"ema_iter_{step:06d}.pt").write_bytes(b"ema")
    rng = np.random.default_rng(seed)
    np.save(workdir / "grids" / "seeds.npy",
            rng.integers(0, 255, (8, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8))
    for record in records:
        if record["kind"] == "grid":
            pixels = rng.integers(0, 255, (IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
            Image.fromarray(pixels).save(record["path"])
    (workdir / "checkpoints" / "full_final.pt").write_bytes(b"final")
    (workdir / "checkpoints-meta" / "checkpoint.pth").write_bytes(b"meta")
    if torch_files:
        torch.save({"step": n_iters, "ema_state_dict": {"w": torch.zeros(1)}, "schedule": values,
                    "run_id": config["run_id"], "config_sha256": manifest["config_sha256"]},
                   workdir / "checkpoints" / f"ema_iter_{n_iters:06d}.pt")
        torch.save({"optimizer": {}, "model": {}, "step": n_iters + 1, "ema": {}},
                   workdir / "checkpoints" / "full_final.pt")
    (workdir / "config.json").write_text(json.dumps(config, indent=2))
    (workdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (workdir / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (workdir / "DONE").write_bytes(b"")
    return workdir


def write_cells(path: Path, cells=CELLS) -> Path:
    lines = ["index,run_id,dataset_id,arm,seed,tier"]
    lines += [f"{i},{d}_{a}_s{s},{d},{a},{s},1" for i, d, a, s in cells]
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def array(tmp_path) -> tuple[Path, Path]:
    root = tmp_path / "runs"
    for _, dataset, arm, seed in CELLS:
        make_run(root, dataset, arm, seed)
    return root, write_cells(tmp_path / "cells.csv")


def _run(root: Path, cells: Path, n_iters: int = 40000, *extra: str):
    return main(["--run-root", str(root), "--cells", str(cells), "--n-iters", str(n_iters),
                 *extra])


def _report(root: Path, cells: Path, run_id: str, n_iters: int = 40000, **kwargs):
    reports = check_array(read_cells(cells), root, n_iters, **kwargs)
    return next(r for r in reports if r.cell.run_id == run_id), reports


def _rewrite_metrics(workdir: Path, edit) -> None:
    path = workdir / "metrics.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records = edit(records)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def _set_config(workdir: Path, edit) -> None:
    """Edit config.json and keep the manifest's recipe hash consistent with it."""
    config = json.loads((workdir / "config.json").read_text())
    edit(config)
    (workdir / "config.json").write_text(json.dumps(config))
    manifest = json.loads((workdir / "manifest.json").read_text())
    manifest["recipe_sha256"] = _recipe_hash(config)
    (workdir / "manifest.json").write_text(json.dumps(manifest))


def test_a_healthy_array_exits_zero_and_lists_cells_in_table_order(array, capsys):
    root, cells = array
    assert _run(root, cells) == 0
    out = capsys.readouterr().out
    rows = [line.split()[1] for line in out.splitlines() if line[:3].strip().isdigit()]
    assert rows == ["ixi_A0_s1", "ixi_A0_s2", "ixi_A3_s1", "lsun_church_A0_s1"]
    assert "VERDICT: HEALTHY 4/4 cells at 40000" in out
    assert "16/16" in out and "problems:" not in out


def test_the_row_reports_the_last_train_line(array):
    root, cells = array
    report, _ = _report(root, cells, "ixi_A0_s1")
    assert report.healthy, report.problems
    assert report.done_step == 40000 and report.ema_found == report.ema_expected == 16
    assert report.full_final and report.lr == 1e-4 and report.recipe == "ok"
    assert report.last_train == {"step": 40000, "loss": pytest.approx(0.34),
                                 "grad_norm": 140.0, "amp_scale": 64.0}


def test_a_missing_checkpoint_fails_its_cell_only(array, capsys):
    root, cells = array
    (root / "ixi_A0_s2" / "checkpoints" / "ema_iter_020000.pt").unlink()
    assert _run(root, cells) == 1
    report, reports = _report(root, cells, "ixi_A0_s2")
    assert report.status == "FAIL"
    assert (report.ema_found, report.ema_expected) == (15, 16)
    assert "missing ema_iter_020000.pt" in report.problems
    assert [r.healthy for r in reports] == [True, False, True, True]
    assert "UNHEALTHY 3/4" in capsys.readouterr().out


def test_a_missing_full_final_fails(array):
    root, cells = array
    (root / "ixi_A3_s1" / "checkpoints" / "full_final.pt").unlink()
    report, _ = _report(root, cells, "ixi_A3_s1")
    assert not report.full_final
    assert "missing or empty checkpoints/full_final.pt" in report.problems


def test_an_abort_fails_even_with_allow_skips(array):
    root, cells = array
    abort = {"step": 30136, "kind": "abort", "reason": "10 consecutive skips", "loss": None,
             "n_skipped": 29, "consecutive": 10, "abort_state": None}
    _rewrite_metrics(root / "lsun_church_A0_s1", lambda rs: [*rs[:-1], abort, rs[-1]])
    report, _ = _report(root, cells, "lsun_church_A0_s1", allow_skips=True)
    assert report.n_abort == 1
    assert any(p.startswith("abort at step 30136") for p in report.problems)
    assert _run(root, cells, 40000, "--allow-skips") == 1


def test_a_skip_fails_unless_skips_are_allowed(array):
    root, cells = array
    skip = {"step": 27883, "kind": "skip", "loss": None, "n_skipped": 1, "consecutive": 1}
    _rewrite_metrics(root / "ixi_A0_s1", lambda rs: [*rs[:-1], skip, rs[-1]])
    report, _ = _report(root, cells, "ixi_A0_s1")
    assert report.n_skip == 1 and "skip at step 27883" in report.problems
    assert _run(root, cells) == 1
    report, _ = _report(root, cells, "ixi_A0_s1", allow_skips=True)
    assert report.healthy, report.problems
    assert _run(root, cells, 40000, "--allow-skips") == 0


def test_recipe_drift_is_flagged_on_the_drifting_cell(array):
    root, cells = array
    _set_config(root / "ixi_A3_s1", lambda c: c["optim"].update(grad_clip=0.5))
    report, reports = _report(root, cells, "ixi_A3_s1")
    assert report.recipe == "drift"
    assert any("optim.grad_clip: 1.0 -> 0.5" in p for p in report.problems)
    assert [r.recipe for r in reports] == ["ok", "ok", "drift", "ok"]


def test_the_arm_and_cell_keys_are_not_drift(array):
    """A3 differs from A0 only in CELL_KEYS, which the comparison ignores."""
    root, cells = array
    _, reports = _report(root, cells, "ixi_A3_s1")
    assert all(r.healthy for r in reports)
    assert {"arm", "seed", "dataset_id", "data.dataset", "model.blur_sigma_max"} <= CELL_KEYS


def test_arm_keys_that_disagree_within_an_arm_are_flagged(array):
    root, cells = array
    _set_config(root / "ixi_A0_s2", lambda c: c["model"].update(blur_sigma_max=48.0))
    report, _ = _report(root, cells, "ixi_A0_s2")
    assert any("arm keys differ from the other A0 cells" in p for p in report.problems)


def test_a_wrong_lr_is_flagged(array):
    root, cells = array
    workdir = root / "lsun_church_A0_s1"
    _set_config(workdir, lambda c: c["optim"].update(lr=2e-4))
    _rewrite_metrics(workdir, lambda rs: [
        {**r, "lr": 2e-4 * min(r["step"] / 1000, 1.0)} if r["kind"] == "train" else r
        for r in rs])
    report, _ = _report(root, cells, "lsun_church_A0_s1")
    assert "config optim.lr 0.0002 != 0.0001" in report.problems


def test_a_manifest_recipe_hash_that_disagrees_with_the_config_is_flagged(array):
    root, cells = array
    path = root / "ixi_A0_s1" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["recipe_sha256"] = "0" * 64
    path.write_text(json.dumps(manifest))
    report, _ = _report(root, cells, "ixi_A0_s1")
    assert "manifest recipe_sha256 != the hash of config.json's recipe" in report.problems


def test_a_manifest_naming_another_cell_is_flagged(array):
    root, cells = array
    path = root / "ixi_A0_s1" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["seed"] = 2
    path.write_text(json.dumps(manifest))
    report, _ = _report(root, cells, "ixi_A0_s1")
    assert "manifest seed 2 != cells.csv 1" in report.problems


def test_absent_and_running_cells_are_unhealthy(array, capsys):
    root, cells = array
    shutil.rmtree(root / "ixi_A3_s1")
    (root / "ixi_A0_s2" / "DONE").unlink()
    assert _run(root, cells) == 1
    out = capsys.readouterr().out
    assert "UNHEALTHY 2/4 cells healthy at 40000 (1 absent, 1 running)" in out


def test_a_40k_run_checked_at_60k_is_not_done(array):
    root, cells = array
    report, _ = _report(root, cells, "ixi_A0_s1", n_iters=60000)
    assert report.status == "FAIL"
    assert "largest EMA checkpoint 40000 != 60000" in report.problems
    assert "no done event at step 60000" in report.problems
    assert (report.ema_found, report.ema_expected) == (16, 24)


def test_an_extended_run_is_healthy_at_60k_and_not_at_40k(tmp_path):
    root = tmp_path / "runs"
    for _, dataset, arm, seed in CELLS:
        make_run(root, dataset, arm, seed, n_iters=60000, extended_from=40000)
    cells = write_cells(tmp_path / "cells.csv")
    reports = check_array(read_cells(cells), root, 60000)
    assert all(r.healthy for r in reports), [r.problems for r in reports]
    assert reports[0].done_step == 60000 and reports[0].ema_found == 24
    stale = check_array(read_cells(cells), root, 40000)
    assert not any(r.healthy for r in stale)
    assert "largest EMA checkpoint 60000 != 40000" in stale[0].problems


def test_deep_mode_loads_the_checkpoints(tmp_path):
    root = tmp_path / "runs"
    for _, dataset, arm, seed in CELLS:
        make_run(root, dataset, arm, seed, torch_files=True)
    cells = write_cells(tmp_path / "cells.csv")
    assert _run(root, cells, 40000, "--deep") == 0
    # A full_final.pt at the wrong step passes the light check and fails the deep one.
    bad = root / "ixi_A0_s1" / "checkpoints" / "full_final.pt"
    torch.save({"optimizer": {}, "model": {}, "step": 30001, "ema": {}}, bad)
    assert _run(root, cells) == 0
    report, _ = _report(root, cells, "ixi_A0_s1", deep=True)
    assert any(p.startswith("full_final.pt: keys") for p in report.problems)


def test_an_unusable_cell_table_exits_two(tmp_path, capsys):
    assert main(["--run-root", str(tmp_path), "--cells", str(tmp_path / "none.csv"),
                 "--n-iters", "40000"]) == 2
    bad = tmp_path / "bad.csv"
    bad.write_text("index,run_id,dataset_id,arm,seed,tier\n0,ixi_A0_s9,ixi,A0,1,1\n")
    assert main(["--run-root", str(tmp_path), "--cells", str(bad), "--n-iters", "40000"]) == 2
    assert "disagrees with its row" in capsys.readouterr().err


def test_the_repository_cell_table_reads_thirty_cells_in_order():
    cells = read_cells(Path(__file__).resolve().parents[2] / "slurm" / "array" / "cells.csv")
    assert [c.index for c in cells] == list(range(30))
