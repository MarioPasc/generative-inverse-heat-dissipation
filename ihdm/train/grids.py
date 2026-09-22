"""Sample grids from a fixed set of training seeds.

``04-run-artifacts.md`` §3: ``grids/iter_XXXXXX.png`` holds the eight seed images in the first row
and one sample per seed in the second, drawn with the EMA weights. The seeds are fixed by
``config.seed`` and stored as ``grids/seeds.npy``, so the grids of one run are comparable across
checkpoints and the grids of two runs with the same seed are comparable across arms.

The reverse process itself is the released
``scripts.sampling.get_sampling_fn_inverse_heat``; only the prior draw is built here, because the
released ``get_initial_sample`` draws its own batch from the data loader and cannot be given a
fixed set of images.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ihdm.train.errors import TrainError

__all__ = ["fixed_seed_images", "prepare_seeds", "save_grid"]


def fixed_seed_images(dataset_root: Path, seed: int, n: int = 8) -> np.ndarray:
    """Draw ``n`` images from the train split, deterministically from ``seed``.

    Parameters
    ----------
    dataset_root : Path
        A standard-format dataset directory (``images.npy`` + ``splits.json``).
    seed : int
        The run seed; the same value always selects the same images.
    n : int
        How many seed images to draw.

    Returns
    -------
    np.ndarray
        ``(n, H, W)`` ``uint8``.

    Raises
    ------
    TrainError
        If the train split holds fewer than ``n`` images.
    """
    dataset_root = Path(dataset_root)
    splits = json.loads((dataset_root / "splits.json").read_text())
    train_indices = np.asarray(splits["train"], dtype=np.int64)
    if train_indices.size < n:
        raise TrainError(
            f"train split of {dataset_root} holds {train_indices.size} images, need {n}"
        )
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(train_indices, size=n, replace=False))
    images = np.load(dataset_root / "images.npy", mmap_mode="r")
    return np.asarray(images[chosen], dtype=np.uint8)


def prepare_seeds(workdir: Path, dataset_root: Path, seed: int, n: int = 8) -> np.ndarray:
    """Return the run's seed images, writing ``grids/seeds.npy`` the first time.

    Idempotent across restarts: a resumed run reuses the array already on disk, so the grids
    of a run extended from 20k to 30k iterations (D10) keep the same seeds.

    Parameters
    ----------
    workdir : Path
        The run directory.
    dataset_root : Path
        The dataset directory to draw from.
    seed : int
        The run seed.
    n : int
        How many seed images.

    Returns
    -------
    np.ndarray
        ``(n, H, W)`` ``uint8``.
    """
    path = Path(workdir) / "grids" / "seeds.npy"
    if path.is_file():
        return np.asarray(np.load(path), dtype=np.uint8)
    seeds = fixed_seed_images(dataset_root, seed, n=n)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, seeds)
    return seeds


def _prior_from_seeds(seeds: np.ndarray, config: Any, heat_module: Any, delta: float):
    """Blur the seed images to level ``K`` and add the prior noise when it is enabled."""
    device = config.device
    images = torch.from_numpy(seeds.astype(np.float32) / 255.0)[:, None].to(device)
    levels = torch.full((images.shape[0],), int(config.model.K), dtype=torch.long, device=device)
    prior = heat_module(images, levels).float()
    if config.get("sampling") and config.sampling.get("prior_noise", False):
        prior = prior + delta * torch.randn_like(prior)
    return images, prior


def save_grid(
    workdir: Path,
    step: int,
    model_fn: Any,
    config: Any,
    heat_module: Any,
    seeds: np.ndarray,
) -> Path:
    """Write ``grids/iter_{step:06d}.png``: the seeds on top, one sample each below.

    The caller is responsible for putting the EMA weights into the model before the call
    (``ema.store`` / ``ema.copy_to`` / ``ema.restore``), matching the released pattern.

    Parameters
    ----------
    workdir : Path
        The run directory.
    step : int
        The training step, used in the file name.
    model_fn : Callable
        ``mutils.get_model_fn(model, train=False)``.
    config : ml_collections.ConfigDict
        The resolved config; ``model.K``, ``model.sigma``, ``sampling.*`` and ``device`` are read.
    heat_module : model_code.utils.DCTBlur
        The forward blur process of the run.
    seeds : np.ndarray
        ``(n, H, W)`` ``uint8`` seed images.

    Returns
    -------
    Path
        The path of the written PNG.
    """
    from torchvision.utils import make_grid, save_image

    from scripts.sampling import get_sampling_fn_inverse_heat

    delta = float(config.model.sigma) * float(config.sampling.delta_factor)
    with torch.no_grad():
        images, prior = _prior_from_seeds(seeds, config, heat_module, delta)
        sampler = get_sampling_fn_inverse_heat(
            config,
            initial_sample=prior,
            intermediate_sample_indices=None,
            delta=delta,
            device=config.device,
        )
        sample, _, _ = sampler(model_fn)
        panel = torch.cat([images.cpu(), sample.detach().cpu().clamp(0.0, 1.0)], dim=0)

    path = Path(workdir) / "grids" / f"iter_{step:06d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(make_grid(panel, nrow=seeds.shape[0], padding=2), path)
    return path
