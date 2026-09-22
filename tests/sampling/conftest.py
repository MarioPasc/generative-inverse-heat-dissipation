"""Fixtures of the sampler tests: a synthetic dataset, a tiny run directory and its checkpoint.

The tiny run mimics what the trainer writes (``04-run-artifacts.md`` §3): a ``config.json`` from
:func:`ihdm.train.manifest.write_config_json` and a ``checkpoints/ema_iter_000003.pt`` in the
§3.3 format, here filled with the random initialisation of the smoke network rather than with
trained weights. Building it here instead of running the trainer keeps the sampler tests
independent of T2.1's smoke run, which costs several seconds per session.

The dataset is rebuilt here (8 "subjects" x 8 "slices", so ``slice == 5`` exists) rather than
reused from ``tests/conftest.py``, whose fixture is function-scoped and owned by ticket T0.1.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from ihdm.data.format import DatasetMeta, split_by_subject, write_dataset
from ihdm.train.checkpoints import ema_checkpoint_path
from ihdm.train.manifest import config_sha256, write_config_json

N_SUBJECTS = 8
N_SLICES = 8
K_FULL = 200

#: The step of the fixture checkpoint; matches the smoke config's ``ckpt_every``.
FIXTURE_STEP = 3


def build_dataset(parent: Path, dataset_id: str = "synthetic") -> Path:
    """Write a 64-image, 32x32 standard-format dataset at ``parent/dataset_id``.

    Parameters
    ----------
    parent : Path
        The directory that plays the role of ``config.data.root``.
    dataset_id : str
        The dataset folder name.

    Returns
    -------
    Path
        The dataset directory.
    """
    root = Path(parent) / dataset_id
    n = N_SUBJECTS * N_SLICES
    rng = np.random.default_rng(7)
    images = rng.integers(0, 256, size=(n, 32, 32), dtype=np.uint8)
    subjects = [f"SUBJ{s:03d}" for s in range(N_SUBJECTS) for _ in range(N_SLICES)]
    slices = [sl for _ in range(N_SUBJECTS) for sl in range(N_SLICES)]

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
        pipeline="tests.sampling.conftest",
        pipeline_version="1.0",
        git_sha="test",
        created="2026-09-22T00:00:00",
        raw_root=str(root),
        parameters={"note": "synthetic fixture for the T2.2 sampler tests"},
        counts={
            "subjects_total": N_SUBJECTS,
            "subjects_used": N_SUBJECTS,
            "slices_per_subject": N_SLICES,
        },
    )
    write_dataset(root, images, index, splits, meta)
    return root


def log_schedule(sigma_max: float, k: int = K_FULL, sigma_min: float = 0.5) -> np.ndarray:
    """Return the released log schedule with 0 prepended (``04-run-artifacts.md`` §1)."""
    levels = np.exp(np.linspace(np.log(sigma_min), np.log(sigma_max), k))
    return np.concatenate([[0.0], levels]).astype(np.float64)


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory) -> Path:
    """A 64-image, 32x32 standard-format dataset with eight slices per subject."""
    return build_dataset(tmp_path_factory.mktemp("t22_data"))


@pytest.fixture
def tiny_config(tiny_dataset):
    """The smoke config pointed at :func:`tiny_dataset`, on the CPU."""
    from configs.spectral import smoke

    config = smoke.get_config()
    config.data.root = str(tiny_dataset.parent)
    config.device = torch.device("cpu")
    return config


def write_tiny_run(workdir: Path, config, step: int = FIXTURE_STEP) -> Path:
    """Write a run directory holding ``config.json`` and one EMA checkpoint of ``04`` §3.3.

    Parameters
    ----------
    workdir : Path
        The run directory to create.
    config : ml_collections.ConfigDict
        The run's config (the smoke config in these tests).
    step : int
        The step the checkpoint claims.

    Returns
    -------
    Path
        The checkpoint path.
    """
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

    torch.manual_seed(0)
    model = UNetModel(config)
    path = ema_checkpoint_path(workdir, step)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": int(step),
            "ema_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "schedule": {
                "name": config.model.blur_schedule_name,
                "sha256": config.model.blur_schedule_sha256,
                "values": [float(v) for v in np.asarray(config.model.blur_schedule).ravel()],
            },
            "run_id": config.run_id,
            "config_sha256": config_sha256(config),
        },
        path,
    )
    return path


@pytest.fixture
def tiny_run(tmp_path, tiny_config) -> tuple[Path, Path]:
    """A run directory with ``config.json``, ``manifest.json`` and one EMA checkpoint."""
    workdir = tmp_path / "runs" / tiny_config.run_id
    ckpt = write_tiny_run(workdir, tiny_config)
    return workdir, ckpt


@pytest.fixture
def schedules_dir(tmp_path, monkeypatch) -> Path:
    """A temporary ``schedules/`` holding the five frozen names, exported to the arm factory."""
    target = tmp_path / "schedules"
    target.mkdir(parents=True, exist_ok=True)
    for name, sigma_max in (
        ("log_W2", 96.0),
        ("log_W8", 24.0),
        ("ixi_W2", 96.0),
        ("ixi_W8", 24.0),
        ("lsun_church_W2", 96.0),
    ):
        np.save(target / f"{name}.npy", log_schedule(sigma_max))
    monkeypatch.setenv("IHDM_SCHEDULES_DIR", str(target))
    return target
