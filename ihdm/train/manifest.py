"""Run manifest and config serialisation.

Frozen contract: ``docs/SPECIFICATIONS/04-run-artifacts.md`` §3.1. The manifest is written once,
before the first training step, and is the only place that records *which* schedule array, *which*
data hash and *which* git commit produced a run. Every metric downstream (T4.x) is computed
offline from the checkpoints of a run, so a run whose manifest is missing or stale is not
auditable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ihdm.data.format import DatasetMeta

logger = logging.getLogger(__name__)

__all__ = [
    "config_sha256",
    "config_to_json",
    "load_data_meta",
    "run_id_from_config",
    "write_config_json",
    "write_manifest",
]


def _jsonable(value: Any) -> Any:
    """Convert one config value into something :mod:`json` can serialise.

    Parameters
    ----------
    value : Any
        A leaf of the config tree (or a nested mapping/sequence of leaves).

    Returns
    -------
    Any
        A JSON-serialisable equivalent: arrays and tuples become lists, numpy scalars
        become Python scalars, ``torch.device`` and ``Path`` become strings.
    """
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict) or hasattr(value, "items"):  # dict or nested ConfigDict
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (torch.device, Path)):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def config_to_json(config: Any) -> dict[str, Any]:
    """Return the resolved config as a plain, JSON-serialisable dict.

    Parameters
    ----------
    config : ml_collections.ConfigDict
        The config built by :mod:`configs.spectral.arms` (or the smoke config).

    Returns
    -------
    dict[str, Any]
        Nested dict with arrays as lists and ``torch.device`` as a string.
    """
    return {key: _jsonable(value) for key, value in config.items()}


def config_sha256(config: Any) -> str:
    """Return the SHA-256 of the canonical JSON form of ``config``.

    The hash is stored in the manifest and inside every EMA checkpoint, so a checkpoint
    can always be matched to the exact configuration that produced it.

    Parameters
    ----------
    config : ml_collections.ConfigDict
        The resolved config.

    Returns
    -------
    str
        Hex digest of ``json.dumps(config_to_json(config), sort_keys=True)``.
    """
    payload = json.dumps(config_to_json(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_id_from_config(config: Any) -> str:
    """Return ``f"{dataset_id}_{arm}_s{seed}"`` for ``config``.

    Recomputed inside ``train()`` rather than trusted from the factory, because
    ``--config.seed=2`` is applied by ``ml_collections`` *after* the factory has run.

    Parameters
    ----------
    config : ml_collections.ConfigDict
        Must provide ``data.dataset`` and ``seed``; ``arm`` is optional (the smoke
        config has it, the released configs do not).

    Returns
    -------
    str
        The run id of ``04-run-artifacts.md`` §2.
    """
    arm = config.get("arm", "NA")
    return f"{config.data.dataset}_{arm}_s{int(config.seed)}"


def write_config_json(workdir: Path, config: Any) -> Path:
    """Write ``config.json`` (the resolved config, arrays as lists).

    Parameters
    ----------
    workdir : Path
        The run directory.
    config : ml_collections.ConfigDict
        The resolved config.

    Returns
    -------
    Path
        The path written.
    """
    path = Path(workdir) / "config.json"
    path.write_text(json.dumps(config_to_json(config), indent=2, sort_keys=True))
    return path


def load_data_meta(config: Any) -> dict[str, Any] | None:
    """Read ``meta.json`` of the dataset the run trains on.

    Parameters
    ----------
    config : ml_collections.ConfigDict
        Must provide ``data.root`` and ``data.dataset``.

    Returns
    -------
    dict[str, Any] | None
        The parsed ``meta.json``, or ``None`` if the dataset directory has none (the
        released torchvision datasets, which this project does not use).
    """
    path = Path(config.data.root) / config.data.dataset / "meta.json"
    if not path.is_file():
        logger.warning("no meta.json at %s; the manifest will record data.meta = null", path)
        return None
    meta = DatasetMeta.from_json(json.loads(path.read_text()))
    return meta.to_json()


def _git(args: list[str], cwd: Path) -> str:
    """Run a git command and return its stripped stdout, or ``""`` on any failure."""
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=20, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def _git_state(repo_root: Path) -> tuple[str, bool]:
    """Return ``(git_sha, git_dirty)`` for ``repo_root``; ``("unknown", False)`` if not a repo."""
    sha = _git(["rev-parse", "HEAD"], repo_root)
    if not sha:
        return "unknown", False
    return sha, bool(_git(["status", "--porcelain"], repo_root))


def _gpu_name(config: Any) -> str | None:
    """Return the name of the CUDA device the run will use, or ``None`` on CPU."""
    if not torch.cuda.is_available() or str(config.device).startswith("cpu"):
        return None
    try:
        return torch.cuda.get_device_name(0)
    except (RuntimeError, AssertionError):
        return None


def write_manifest(
    workdir: Path,
    config: Any,
    model: torch.nn.Module,
    data_meta: dict[str, Any] | None,
) -> Path:
    """Write ``manifest.json`` per ``04-run-artifacts.md`` §3.1.

    Parameters
    ----------
    workdir : Path
        The run directory; created if missing.
    config : ml_collections.ConfigDict
        The resolved config, with ``run_id`` already updated for the effective seed.
    model : torch.nn.Module
        The constructed model, used only for the parameter count.
    data_meta : dict[str, Any] | None
        The dataset's ``meta.json``, or ``None``.

    Returns
    -------
    Path
        The path of the written manifest.
    """
    from ihdm.paths import repo_root  # local import: avoids a cycle through ihdm.paths' defaults
    from ihdm.train.recipe import recipe_sha256  # local import: recipe.py imports this module

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    git_sha, git_dirty = _git_state(repo_root())
    schedule_values = [float(v) for v in np.asarray(config.model.blur_schedule).ravel()]

    manifest: dict[str, Any] = {
        "run_id": run_id_from_config(config),
        "dataset_id": config.data.dataset,
        "arm": config.get("arm", None),
        "seed": int(config.seed),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "hostname": socket.gethostname(),
        "gpu": _gpu_name(config),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "started": datetime.now(UTC).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "data": {
            "root": str(config.data.root),
            "meta": data_meta,
            "images_sha256": (data_meta or {}).get("sha256_images"),
        },
        "schedule": {
            "name": config.model.get("blur_schedule_name", None),
            "file": config.model.get("blur_schedule_file", None),
            "sha256": config.model.get("blur_schedule_sha256", None),
            "values": schedule_values,
        },
        "config_sha256": config_sha256(config),
        # D19: the reference of the recipe check on resume (ihdm.train.recipe).
        "recipe_sha256": recipe_sha256(config),
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "batch_size": int(config.training.batch_size),
        "n_iters": int(config.training.n_iters),
    }
    path = workdir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path
