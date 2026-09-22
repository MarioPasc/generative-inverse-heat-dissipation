"""Checkpoint writing: EMA snapshots, the final full state and the rolling resume checkpoint.

Frozen contract: ``docs/SPECIFICATIONS/04-run-artifacts.md`` §3.3. Every downstream metric is
computed offline from the EMA checkpoints, so each one carries enough provenance to be
interpreted on its own: the step, the schedule that produced it (name, hash and values) and the
hash of the config. The keys carry no ``module.`` prefix, so a consumer can load them into a bare
``model_code.unet.UNetModel`` without ``torch.nn.DataParallel`` (``04`` §6).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

__all__ = [
    "ema_checkpoint_path",
    "optimizer_state_dict",
    "ema_state_dict",
    "resume_path",
    "save_ema",
    "save_full_final",
    "save_resume",
]


def resume_path(workdir: Path) -> Path:
    """Return the path of the rolling resume checkpoint (released format).

    Parameters
    ----------
    workdir : Path
        The run directory.

    Returns
    -------
    Path
        ``<workdir>/checkpoints-meta/checkpoint.pth``.
    """
    return Path(workdir) / "checkpoints-meta" / "checkpoint.pth"


def ema_checkpoint_path(workdir: Path, step: int) -> Path:
    """Return the path of the EMA checkpoint of ``step``.

    Parameters
    ----------
    workdir : Path
        The run directory.
    step : int
        The training step.

    Returns
    -------
    Path
        ``<workdir>/checkpoints/ema_iter_{step:06d}.pt``.
    """
    return Path(workdir) / "checkpoints" / f"ema_iter_{step:06d}.pt"


def _strip_module_prefix(key: str) -> str:
    """Drop the ``module.`` prefix that ``torch.nn.DataParallel`` adds to every key."""
    return key[len("module.") :] if key.startswith("module.") else key


def ema_state_dict(model: torch.nn.Module, ema: Any) -> dict[str, torch.Tensor]:
    """Return the model's state dict with the EMA parameters, on the CPU, unprefixed.

    The EMA shadow only tracks parameters with ``requires_grad``; the buffers of the network
    are taken from the live model. The live weights are swapped out and restored around the
    snapshot, so the optimisation is not affected.

    Parameters
    ----------
    model : torch.nn.Module
        The (``DataParallel``-wrapped) training model.
    ema : model_code.ema.ExponentialMovingAverage
        The EMA tracker of ``model.parameters()``.

    Returns
    -------
    dict[str, torch.Tensor]
        Detached CPU tensors keyed without the ``module.`` prefix.
    """
    ema.store(model.parameters())
    ema.copy_to(model.parameters())
    try:
        snapshot = {
            _strip_module_prefix(key): value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
    finally:
        ema.restore(model.parameters())
    return snapshot


def _plain(value: Any) -> Any:
    """Recursively replace numpy scalars by Python scalars (tensors are left alone)."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def optimizer_state_dict(optimizer: Any) -> dict[str, Any]:
    """Return the optimiser state with plain Python scalars in ``param_groups``.

    ``scripts/losses.py: optimization_manager`` applies the warm-up as
    ``g["lr"] = lr * np.minimum(step / warmup, 1.0)``, which stores a ``numpy.float64`` in the
    parameter group. Since torch 2.6 ``torch.load`` defaults to ``weights_only=True``, and its
    restricted unpickler rejects ``numpy._core.multiarray.scalar``, so a checkpoint carrying that
    value cannot be read back by the released ``scripts.utils.restore_checkpoint`` and every
    resume fails. Coercing here fixes it at the source, without editing the released files
    outside the hook points of ``04-run-artifacts.md`` §5 and without allow-listing numpy
    internals in the unpickler of every consumer.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The training optimiser.

    Returns
    -------
    dict[str, Any]
        The state dict, with ``param_groups`` free of numpy scalars.
    """
    state = optimizer.state_dict()
    state["param_groups"] = _plain(state.get("param_groups", []))
    return state


def _schedule_record(config: Any) -> dict[str, Any]:
    """Return the ``{"name", "sha256", "values"}`` record of the run's blur schedule."""
    return {
        "name": config.model.get("blur_schedule_name", None),
        "sha256": config.model.get("blur_schedule_sha256", None),
        "values": [float(v) for v in np.asarray(config.model.blur_schedule).ravel()],
    }


def save_ema(
    workdir: Path,
    step: int,
    model: torch.nn.Module,
    ema: Any,
    config: Any,
    run_id: str,
    config_sha256: str,
) -> Path:
    """Write ``checkpoints/ema_iter_{step:06d}.pt`` per ``04-run-artifacts.md`` §3.3.

    Parameters
    ----------
    workdir : Path
        The run directory.
    step : int
        The training step this checkpoint belongs to.
    model : torch.nn.Module
        The training model.
    ema : model_code.ema.ExponentialMovingAverage
        The EMA tracker.
    config : ml_collections.ConfigDict
        The resolved config, for the schedule record.
    run_id : str
        The run id of ``04`` §2.
    config_sha256 : str
        The hash of the resolved config.

    Returns
    -------
    Path
        The path written.
    """
    path = ema_checkpoint_path(workdir, step)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": int(step),
            "ema_state_dict": ema_state_dict(model, ema),
            "schedule": _schedule_record(config),
            "run_id": run_id,
            "config_sha256": config_sha256,
        },
        path,
    )
    return path


def save_resume(workdir: Path, state: dict[str, Any]) -> Path:
    """Write the rolling resume checkpoint in the released four-key format.

    Parameters
    ----------
    workdir : Path
        The run directory.
    state : dict[str, Any]
        The released training state ``{"optimizer", "model", "step", "ema"}``.

    Returns
    -------
    Path
        The path written.
    """
    path = resume_path(workdir)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "optimizer": optimizer_state_dict(state["optimizer"]),
            "model": state["model"].state_dict(),
            "step": state["step"],
            "ema": state["ema"].state_dict(),
        },
        path,
    )
    return path


def save_full_final(workdir: Path, state: dict[str, Any]) -> Path:
    """Write ``checkpoints/full_final.pt`` (model, optimizer, EMA and step at the end).

    Uses the released four-key layout, so ``scripts.utils.restore_checkpoint`` reads it back
    unchanged; written here directly to keep :mod:`ihdm.train` free of the released modules'
    import-time dependencies.

    Parameters
    ----------
    workdir : Path
        The run directory.
    state : dict[str, Any]
        The released training state ``{"optimizer", "model", "step", "ema"}``.

    Returns
    -------
    Path
        The path written.
    """
    path = Path(workdir) / "checkpoints" / "full_final.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "optimizer": optimizer_state_dict(state["optimizer"]),
            "model": state["model"].state_dict(),
            "step": state["step"],
            "ema": state["ema"].state_dict(),
        },
        path,
    )
    return path
