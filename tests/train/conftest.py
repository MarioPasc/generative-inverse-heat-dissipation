"""Fixtures of the trainer tests: temporary schedule arrays and one shared smoke run.

Two things are shared here because they are expensive:

``schedules_dir``
    the five frozen schedule arrays of ``04-run-artifacts.md`` §1 do not exist in this worktree
    (ticket T1.3 produces them), so the arm-factory tests generate valid arrays into a temporary
    directory and point ``IHDM_SCHEDULES_DIR`` at it;
``smoke_run``
    a full six-iteration CPU training run, reused by ``test_smoke_train.py`` (artefacts, then a
    resume to a larger ``n_iters`` on the same workdir) and ``test_checkpoint_format.py``.

The session-scoped synthetic dataset is built here rather than reused from ``tests/conftest.py``
(whose ``synthetic_dataset`` fixture is function-scoped and owned by ticket T0.1).
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ihdm.data.format import DatasetMeta, split_by_subject, write_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]

#: (name, terminal blur) of every schedule the arm factory can ask for.
SCHEDULE_SPECS: tuple[tuple[str, float], ...] = (
    ("log_W2", 96.0),
    ("log_W8", 24.0),
    ("ixi_W2", 96.0),
    ("ixi_W8", 24.0),
    ("lsun_church_W2", 96.0),
)

K = 200

#: Intra-op threads of the smoke subprocess. The tiny 32x32 network gains nothing from more,
#: and torch's default (one thread per core) collapses under the load of a parallel ticket:
#: measured 4 s at 2 threads against a 600 s timeout at the default on a 24-core host at load 28.
TORCH_THREADS = 2


def log_schedule(sigma_max: float, k: int = K, sigma_min: float = 0.5) -> np.ndarray:
    """Return the released log schedule: ``exp(linspace(log(0.5), log(smax), K))`` with 0 in front.

    Parameters
    ----------
    sigma_max : float
        The terminal blur in pixels.
    k : int
        Number of levels.
    sigma_min : float
        The first blur level in pixels.

    Returns
    -------
    np.ndarray
        ``float64``, shape ``(k + 1,)``.
    """
    levels = np.exp(np.linspace(np.log(sigma_min), np.log(sigma_max), k))
    return np.concatenate([[0.0], levels]).astype(np.float64)


def write_schedules(target: Path) -> Path:
    """Write a valid array for every name of :data:`SCHEDULE_SPECS` into ``target``."""
    target.mkdir(parents=True, exist_ok=True)
    for name, sigma_max in SCHEDULE_SPECS:
        np.save(target / f"{name}.npy", log_schedule(sigma_max))
    return target


@pytest.fixture
def schedules_dir(tmp_path, monkeypatch) -> Path:
    """A temporary ``schedules/`` directory holding valid arrays, exported to the factory."""
    target = write_schedules(tmp_path / "schedules")
    monkeypatch.setenv("IHDM_SCHEDULES_DIR", str(target))
    return target


def build_dataset(parent: Path, dataset_id: str) -> Path:
    """Write a 64-image, 32x32 standard-format dataset (8 subjects x 8 slices) at
    ``parent/dataset_id``.

    Parameters
    ----------
    parent : Path
        The directory that plays the role of ``config.data.root``.
    dataset_id : str
        The dataset folder name, which is also ``config.data.dataset``.

    Returns
    -------
    Path
        The dataset directory.
    """
    root = Path(parent) / dataset_id
    n_subjects, n_slices = 8, 8
    n = n_subjects * n_slices
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, size=(n, 32, 32), dtype=np.uint8)
    subjects = [f"SUBJ{s:03d}" for s in range(n_subjects) for _ in range(n_slices)]
    slices = [sl for _ in range(n_subjects) for sl in range(n_slices)]

    splits = split_by_subject(subjects, rng_seed=2026, train_frac=0.8, n_seed=2)
    train_set, seed_set = set(splits["train"]), set(splits["seed"])
    fine_split = [
        "train" if i in train_set else ("seed" if i in seed_set else "ref") for i in range(n)
    ]
    index = pd.DataFrame(
        {
            "idx": np.arange(n),
            "subject": subjects,
            "slice": slices,
            "z_mm": [float(sl) for sl in slices],
            "source": [f"synthetic/{s}/{sl}" for s, sl in zip(subjects, slices, strict=True)],
            "split": fine_split,
        }
    )
    meta = DatasetMeta(
        dataset_id=dataset_id,
        n_images=n,
        image_size=32,
        dtype="uint8",
        pipeline="tests.train.conftest",
        pipeline_version="1.0",
        git_sha="test",
        created="2026-09-22T00:00:00",
        raw_root=str(root),
        parameters={"note": "synthetic fixture for the T2.1 trainer tests"},
        counts={"subjects_total": n_subjects, "subjects_used": n_subjects,
                "slices_per_subject": n_slices},
    )
    write_dataset(root, images, index, splits, meta)
    return root


@pytest.fixture
def dataset_factory():
    """Expose :func:`build_dataset` so a test can write a dataset under any id."""
    return build_dataset


@pytest.fixture(scope="session")
def smoke_dataset(tmp_path_factory) -> Path:
    """A 64-image, 32x32 standard-format dataset for the smoke run."""
    return build_dataset(tmp_path_factory.mktemp("t21_data"), "synthetic")


def run_training(data_root: Path, workdir: Path, n_iters: int, timeout: int = 300):
    """Run ``train.train`` on the smoke config in a CPU-only subprocess.

    ``model_code/utils.py: create_model`` wraps the model in
    ``torch.nn.DataParallel(model, device_ids=None)``, which scatters the input batch to
    ``cuda:0`` whenever CUDA is visible, regardless of ``config.device``; on a host with a GPU
    a CPU run therefore fails with a device mismatch. Masking CUDA at the process level (rather
    than patching ``torch.cuda.is_available`` in-process, which is unreliable because
    ``torch.cuda.device_count`` is cached for the life of the process) keeps the run strictly on
    the CPU. This is T0.1's pattern, kept per ``04-run-artifacts.md`` §6.

    Parameters
    ----------
    data_root : Path
        The directory *containing* the dataset folder (``config.data.root``).
    workdir : Path
        The run directory.
    n_iters : int
        ``config.training.n_iters`` for this invocation.
    timeout : int
        Subprocess timeout in seconds.

    Returns
    -------
    subprocess.CompletedProcess
        The finished process, so callers can assert on the return code and the output.
    """
    script = textwrap.dedent(
        f"""
        import torch

        torch.set_num_threads({TORCH_THREADS})

        from configs.spectral import smoke
        import train as trainer

        config = smoke.get_config()
        config.data.root = {str(data_root)!r}
        config.training.n_iters = {n_iters}
        trainer.train(config, {str(workdir)!r})
        """
    )
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["OMP_NUM_THREADS"] = str(TORCH_THREADS)
    env["MKL_NUM_THREADS"] = str(TORCH_THREADS)
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


@pytest.fixture(scope="session")
def train_runner():
    """Expose :func:`run_training` to the test modules without importing this conftest."""
    return run_training


@pytest.fixture(scope="session")
def smoke_run(tmp_path_factory, smoke_dataset) -> Path:
    """Run the smoke config to ``n_iters=6`` once per session; return the run directory."""
    workdir = tmp_path_factory.mktemp("t21_run") / "synthetic_smoke_s1"
    result = run_training(smoke_dataset.parent, workdir, n_iters=6)
    assert result.returncode == 0, result.stdout + result.stderr
    return workdir
