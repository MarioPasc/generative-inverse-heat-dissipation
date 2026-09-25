"""End-to-end CPU tests of the D19 trainer changes: skip policy, abort paths, recipe check.

The non-finite losses are injected by the test-only driver ``slurm/loginexa/train_inject.py``
(the production entry point cannot trigger it). Two scaler regimes are covered:

* **disabled** — the real ``torch.cuda.amp.GradScaler`` on a process without CUDA disables
  itself, and its ``step`` then calls ``optimizer.step()`` with the non-finite gradients, so the
  step is *not* a no-op (shown directly by the first test) and the trainer must abort at once;
* **enabled** — ``--cpu-grad-scaler`` substitutes ``torch.amp.GradScaler("cpu")``, which has the
  CUDA scaler's unscale/skip/update semantics, so the CPU suite reaches the skip branch. The same
  branch on the real CUDA scaler is item H5 of the loginexa harness.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
AMP = ["--set", "optim.automatic_mp=true"]


def _records(workdir: Path) -> list[dict]:
    with (workdir / "metrics.jsonl").open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _of(records: list[dict], kind: str) -> list[dict]:
    return [r for r in records if r["kind"] == kind]


def _saved_step(workdir: Path) -> int:
    return int(torch.load(workdir / "checkpoints-meta" / "checkpoint.pth",
                          map_location="cpu")["step"])


def _snapshot(workdir: Path) -> dict[str, tuple[int, str]]:
    return {
        str(p.relative_to(workdir)): (
            p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
        for p in sorted(workdir.rglob("*")) if p.is_file()
    }


def test_a_disabled_scaler_applies_non_finite_gradients():
    script = textwrap.dedent(
        """
        import torch
        scaler = torch.cuda.amp.GradScaler()
        assert not scaler.is_enabled(), "CUDA is masked: the CUDA scaler must disable itself"
        layer = torch.nn.Linear(3, 2)
        optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)
        loss = layer(torch.ones(4, 3)).sum() * float("nan")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        print(all(bool(torch.isfinite(p).all()) for p in layer.parameters()))
        """
    )
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"}
    result = subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT, env=env,
                            capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False", "the disabled scaler's step must corrupt the weights"


def test_disabled_scaler_aborts_at_the_first_non_finite_loss(
    tmp_path, smoke_dataset, inject_runner
):
    workdir = tmp_path / "run"
    result = inject_runner(smoke_dataset.parent, workdir,
                           [*AMP, "--inject", "3", "--set", "training.n_iters=6"])
    assert result.returncode == 3, result.stdout + result.stderr
    records = _records(workdir)
    assert _of(records, "skip") == []
    (abort,) = _of(records, "abort")
    assert abort["step"] == 3
    assert "disabled" in abort["reason"]
    assert abort["resume_saved"] is False and abort["n_skipped"] == 0
    # The rolling checkpoint of step 2 (state step 3) is kept, not overwritten by NaN weights.
    state = torch.load(workdir / "checkpoints-meta" / "checkpoint.pth", map_location="cpu")
    assert int(state["step"]) == 3
    tensors = [t for t in state["model"].values() if t.is_floating_point()]
    assert all(bool(torch.isfinite(t).all()) for t in tensors)
    assert not (workdir / "DONE").exists()


def test_enabled_scaler_skips_and_the_run_completes(tmp_path, smoke_dataset, inject_runner):
    workdir = tmp_path / "run"
    result = inject_runner(smoke_dataset.parent, workdir,
                           [*AMP, "--cpu-grad-scaler", "--inject", "2,4",
                            "--set", "training.n_iters=6"])
    assert result.returncode == 0, result.stdout + result.stderr
    records = _records(workdir)
    skips = _of(records, "skip")
    assert [(r["step"], r["n_skipped"], r["consecutive"], r["loss"]) for r in skips] == [
        (2, 1, 1, None), (4, 2, 1, None)]
    assert _of(records, "abort") == []
    train = {r["step"]: r for r in _of(records, "train")}
    assert sorted(train) == [0, 1, 2, 3, 4, 5, 6], "a skipped log step keeps its train line"
    assert train[2]["loss"] is None and train[2]["grad_norm"] is None
    assert train[3]["loss"] is not None and train[3]["grad_norm"] > 0
    # Each skipped step halves the GradScaler scale (65536 -> 32768 -> 16384).
    assert [train[s]["amp_scale"] for s in (0, 1, 2, 3, 4, 5)] == [
        65536.0, 65536.0, 32768.0, 32768.0, 16384.0, 16384.0]
    (done,) = _of(records, "done")
    assert done["step"] == 6 and done["n_skipped"] == 2
    assert (workdir / "DONE").is_file()


def test_ten_consecutive_skips_abort_and_the_count_survives_resume(
    tmp_path, smoke_dataset, inject_runner
):
    workdir = tmp_path / "run"
    args = [*AMP, "--cpu-grad-scaler", "--set", "training.n_iters=20"]
    first = inject_runner(smoke_dataset.parent, workdir, [*args, "--inject", "3-12"])
    assert first.returncode == 3, first.stdout + first.stderr
    records = _records(workdir)
    assert [r["step"] for r in _of(records, "skip")] == list(range(3, 13))
    (abort,) = _of(records, "abort")
    assert abort["step"] == 12 and abort["consecutive"] == 10 and abort["n_skipped"] == 10
    assert abort["resume_saved"] is True
    assert _saved_step(workdir) == 13  # saved after the no-op step 12

    before = len(records)
    second = inject_runner(smoke_dataset.parent, workdir, [*args, "--inject", "15"])
    assert second.returncode == 0, second.stdout + second.stderr
    new = _records(workdir)[before:]
    assert [r["step"] for r in _of(new, "resume")] == [13]
    skips = [(r["step"], r["n_skipped"], r["consecutive"]) for r in _of(new, "skip")]
    assert skips == [(15, 11, 1)], "10 skips carried over from before the resume, plus one"
    (done,) = _of(new, "done")
    assert done["n_skipped"] == 11


def test_a_resume_with_another_recipe_exits_4_and_writes_nothing(
    tmp_path, smoke_run, smoke_dataset, train_runner
):
    workdir = tmp_path / "copy"
    shutil.copytree(smoke_run, workdir)
    before = _snapshot(workdir)
    result = train_runner(smoke_dataset.parent, workdir, n_iters=9,
                          overrides={"optim.lr": 1e-4})
    assert result.returncode == 4, result.stdout + result.stderr
    assert "optim.lr" in result.stderr
    assert _snapshot(workdir) == before, "a refused resume must leave the run directory untouched"


def test_a_resume_with_new_cadences_and_n_iters_is_accepted(
    tmp_path, smoke_run, smoke_dataset, train_runner
):
    workdir = tmp_path / "copy"
    shutil.copytree(smoke_run, workdir)
    saved = _saved_step(workdir)  # 7, or 10 if test_smoke_train's extension ran first
    before = len(_records(workdir))
    result = train_runner(smoke_dataset.parent, workdir, n_iters=saved + 2,
                          overrides={"training.resume_every": 1, "training.eval_every": 2,
                                     "training.log_every": 2})
    assert result.returncode == 0, result.stdout + result.stderr
    new = _records(workdir)[before:]
    assert [r["step"] for r in _of(new, "resume")] == [saved]
    assert [r["step"] for r in _of(new, "train")] == [
        s for s in range(saved, saved + 3) if s % 2 == 0]
    assert (workdir / "checkpoints" / f"ema_iter_{saved + 2:06d}.pt").is_file()
