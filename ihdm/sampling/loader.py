"""Load a run's config and an EMA checkpoint into a bare U-Net for offline sampling.

Frozen contracts: ``docs/SPECIFICATIONS/04-run-artifacts.md`` §3.3 (the checkpoint payload written
by :mod:`ihdm.train.checkpoints`) and §4 (the sampler interface).

Two details of ``04`` §6 are load-bearing here:

* the network is instantiated as ``model_code.unet.UNetModel(config).to(device)``, never through
  ``model_code.utils.create_model``, whose ``torch.nn.DataParallel(model, device_ids=None)``
  wrapper scatters every input to ``cuda:0`` as soon as CUDA is visible, regardless of the device
  the caller asked for;
* the EMA state dict is stored without the ``module.`` prefix, so it loads into that bare model
  unchanged.

The schedule hash of the checkpoint is compared against the config's before any weight is loaded
(D12): a checkpoint interpreted under a different blur schedule would silently produce samples
from a process the model was never trained on.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import ml_collections
import numpy as np
import torch

from ihdm.sampling.errors import SamplingError

__all__ = [
    "build_heat_module",
    "checkpoint_sha256",
    "load_checkpoint",
    "load_ema_model",
    "load_run_config",
    "resolve_checkpoint_path",
]

logger = logging.getLogger(__name__)

#: Config keys whose JSON round trip must give back a tuple, because the released ``UNetModel``
#: compares them with ``in`` and iterates them in a fixed order.
_TUPLE_KEYS: tuple[tuple[str, str], ...] = (
    ("model", "channel_mult"),
    ("model", "attention_levels"),
)


def _restore_types(payload: dict[str, Any]) -> dict[str, Any]:
    """Undo the JSON flattening of ``ihdm.train.manifest.config_to_json``.

    ``config.json`` stores ``model.blur_schedule`` as a list, ``device`` as a string and the tuple
    valued model keys as lists. ``DCTBlur`` and ``UNetModel`` are written against the original
    types, so they are restored here rather than at every call site. The restoration happens on
    the plain dict, before the :class:`ml_collections.ConfigDict` is built, because a type-safe
    ``ConfigDict`` refuses to replace a list field with an array afterwards.

    Parameters
    ----------
    payload : dict[str, Any]
        The parsed ``config.json``.

    Returns
    -------
    dict[str, Any]
        The same mapping, with the numpy and torch types restored.
    """
    model = payload.get("model")
    if isinstance(model, dict) and "blur_schedule" in model:
        model["blur_schedule"] = np.asarray(model["blur_schedule"], dtype=np.float64)
    for group, key in _TUPLE_KEYS:
        node = payload.get(group)
        if isinstance(node, dict) and key in node:
            node[key] = tuple(node[key])
    if isinstance(payload.get("device"), str):
        payload["device"] = torch.device(payload["device"])
    return payload


def _config_from_manifest(workdir: Path) -> ml_collections.ConfigDict:
    """Rebuild a config from ``manifest.json``'s ``dataset_id``, ``arm`` and ``seed``."""
    path = workdir / "manifest.json"
    if not path.is_file():
        raise SamplingError(f"neither config.json nor manifest.json found in {workdir}")
    manifest = json.loads(path.read_text())
    missing = [key for key in ("dataset_id", "arm", "seed") if manifest.get(key) is None]
    if missing:
        raise SamplingError(f"{path}: manifest lacks {missing}, cannot rebuild the config")
    from configs.spectral.arms import get_config

    spec = f"{manifest['dataset_id']},{manifest['arm']},seed={int(manifest['seed'])}"
    logger.warning("no config.json in %s; rebuilding it from the manifest (%s)", workdir, spec)
    return get_config(spec)


def load_run_config(workdir: Path) -> ml_collections.ConfigDict:
    """Return the resolved config of a run directory.

    Reads ``<workdir>/config.json`` (written by the trainer, ``04`` §3) and falls back to
    :func:`configs.spectral.arms.get_config` driven by ``manifest.json`` when that file is absent.

    Parameters
    ----------
    workdir : Path
        The run directory.

    Returns
    -------
    ml_collections.ConfigDict
        The config with ``model.blur_schedule`` as a ``float64`` array and ``device`` as a
        ``torch.device``.

    Raises
    ------
    SamplingError
        If neither ``config.json`` nor a usable ``manifest.json`` is present, or ``config.json``
        is not valid JSON.
    """
    workdir = Path(workdir)
    path = workdir / "config.json"
    if not path.is_file():
        return _config_from_manifest(workdir)
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SamplingError(f"{path}: not valid JSON ({exc})") from exc
    return ml_collections.ConfigDict(_restore_types(payload))


def resolve_checkpoint_path(workdir: Path, ckpt: str) -> Path:
    """Return the absolute path of a checkpoint named relative to a run directory.

    Parameters
    ----------
    workdir : Path
        The run directory.
    ckpt : str
        A bare file name (``ema_iter_020000.pt``), a path relative to the run directory, or an
        absolute path.

    Returns
    -------
    Path
        The resolved path.

    Raises
    ------
    SamplingError
        If no file exists at any of the candidate locations.
    """
    workdir = Path(workdir)
    candidates = [Path(ckpt), workdir / ckpt, workdir / "checkpoints" / ckpt]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise SamplingError(f"checkpoint {ckpt!r} not found; tried {[str(c) for c in candidates]}")


def checkpoint_sha256(ckpt_path: Path) -> str:
    """Return the SHA-256 of the checkpoint file's bytes, read in 8 MiB chunks."""
    digest = hashlib.sha256()
    with Path(ckpt_path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(ckpt_path: Path, device: torch.device | str) -> dict[str, Any]:
    """Read an EMA checkpoint payload with the restricted unpickler.

    Parameters
    ----------
    ckpt_path : Path
        The checkpoint file.
    device : torch.device | str
        ``map_location`` for the tensors.

    Returns
    -------
    dict[str, Any]
        The payload of ``04`` §3.3.

    Raises
    ------
    SamplingError
        If the file cannot be read or does not carry the two mandatory keys.
    """
    ckpt_path = Path(ckpt_path)
    try:
        payload = torch.load(ckpt_path, map_location=device, weights_only=True)
    except Exception as exc:  # torch raises a family of unrelated types here
        raise SamplingError(f"{ckpt_path}: cannot be read ({type(exc).__name__}: {exc})") from exc
    if not isinstance(payload, dict) or "ema_state_dict" not in payload:
        raise SamplingError(f"{ckpt_path}: not an EMA checkpoint of 04 §3.3 (no 'ema_state_dict')")
    if "schedule" not in payload:
        raise SamplingError(f"{ckpt_path}: no 'schedule' record, cannot verify the blur schedule")
    return payload


def _check_schedule(payload: dict[str, Any], config: Any, ckpt_path: Path) -> None:
    """Raise unless the checkpoint's schedule hash equals the config's (D12)."""
    ckpt_sha = (payload.get("schedule") or {}).get("sha256")
    config_sha = config.model.get("blur_schedule_sha256", None)
    if not ckpt_sha or not config_sha:
        raise SamplingError(
            f"{ckpt_path}: blur-schedule hash missing "
            f"(checkpoint={ckpt_sha!r}, config={config_sha!r}); refusing to sample"
        )
    if ckpt_sha != config_sha:
        raise SamplingError(
            f"{ckpt_path}: blur-schedule hash mismatch; the checkpoint was trained under "
            f"{(payload['schedule'] or {}).get('name')!r} ({ckpt_sha}) but the config carries "
            f"{config.model.get('blur_schedule_name', None)!r} ({config_sha})"
        )


def load_ema_model(
    ckpt_path: Path,
    config: Any,
    device: torch.device | str = "cpu",
    payload: dict[str, Any] | None = None,
) -> torch.nn.Module:
    """Instantiate the released U-Net and load the checkpoint's EMA weights into it.

    Parameters
    ----------
    ckpt_path : Path
        An ``ema_iter_XXXXXX.pt`` written by :func:`ihdm.train.checkpoints.save_ema`.
    config : ml_collections.ConfigDict
        The run's config; ``model.*`` and ``data.*`` define the architecture and
        ``model.blur_schedule_sha256`` is checked against the checkpoint.
    device : torch.device | str
        Where the model is placed.
    payload : dict[str, Any] | None
        An already-read checkpoint payload, so a caller that needs the metadata first does not
        read the (200 MB+) file twice.

    Returns
    -------
    torch.nn.Module
        A ``model_code.unet.UNetModel`` in eval mode with gradients disabled.

    Raises
    ------
    SamplingError
        If the checkpoint is unreadable, its schedule hash does not match the config's, or its
        state dict does not fit the architecture the config describes.
    """
    from model_code.unet import UNetModel

    ckpt_path = Path(ckpt_path)
    device = torch.device(device)
    if payload is None:
        payload = load_checkpoint(ckpt_path, device)
    elif "ema_state_dict" not in payload:
        raise SamplingError(f"{ckpt_path}: the given payload has no 'ema_state_dict'")
    _check_schedule(payload, config, ckpt_path)

    model = UNetModel(config).to(device)
    try:
        model.load_state_dict(payload["ema_state_dict"], strict=True)
    except RuntimeError as exc:
        raise SamplingError(
            f"{ckpt_path}: the EMA state dict does not fit the architecture of the config ({exc})"
        ) from exc
    model.eval()
    model.requires_grad_(False)
    return model


def build_heat_module(config: Any, device: torch.device | str) -> torch.nn.Module:
    """Return the run's forward blur process (``model_code.utils.DCTBlur``).

    Parameters
    ----------
    config : ml_collections.ConfigDict
        The run's config; ``model.blur_schedule`` and ``data.image_size`` are read.
    device : torch.device | str
        Where the blur kernels are placed.

    Returns
    -------
    torch.nn.Module
        The ``DCTBlur`` module of the run's schedule.
    """
    from model_code.utils import create_forward_process_from_sigmas

    schedule = np.asarray(config.model.blur_schedule, dtype=np.float64)
    return create_forward_process_from_sigmas(config, schedule, torch.device(device))
