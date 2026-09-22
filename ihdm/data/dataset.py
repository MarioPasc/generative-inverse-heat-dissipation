"""Torch ``Dataset``/``DataLoader`` backend over the standard on-disk format.

Presents a dataset directory (``docs/SPECIFICATIONS/03-data-format.md``) to the
released trainer with the exact tensor contract it already expects:
``(tensor[C, H, W] float32 in [0, 1], anything)`` per item, batched by the released
training loop as ``batch = next(train_iter)[0]``.

Frozen contract: ``docs/SPECIFICATIONS/03-data-format.md`` §7. Hooked into
``scripts.datasets.get_dataset`` via :data:`NPY_DATASETS`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ihdm.data.errors import DataFormatError

# The four real dataset ids plus "synthetic" (the fixture used by the test suite
# and the Picasso import-check job). A frozenset so `config.data.dataset in
# NPY_DATASETS` is a hash lookup; imported by scripts.datasets.get_dataset.
NPY_DATASETS: frozenset[str] = frozenset(
    {"ixi", "oasis1", "lsun_church", "lsun_bedroom", "synthetic"}
)

_VALID_SPLITS: tuple[str, ...] = ("train", "ref", "seed")


class NpyImageDataset(Dataset):
    """A standard-format dataset directory, presented as a torch ``Dataset``.

    Reads split membership directly from ``splits.json[split]``: for ``split
    == "ref"`` this is already the whole reference set (seed subjects
    included), so no merging with ``"seed"`` is needed here.

    Parameters
    ----------
    root : Path
        Directory holding ``images.npy``, ``index.csv``, ``splits.json``,
        ``meta.json``.
    split : str
        One of ``"train"``, ``"ref"``, ``"seed"``.
    random_flip : bool
        Random horizontal flip, applied independently per item. Off by
        default; kept off for MRI (anatomical left/right is meaningful) and,
        by convention, for photographs too (one recipe everywhere).

    Raises
    ------
    DataFormatError
        If ``split`` is not one of the valid names, or a required file is
        missing.
    """

    def __init__(self, root: Path, split: str, random_flip: bool = False) -> None:
        if split not in _VALID_SPLITS:
            raise DataFormatError(f"split must be one of {_VALID_SPLITS}, got {split!r}")
        self.root = Path(root)
        self.split = split
        self.random_flip = random_flip

        splits_path = self.root / "splits.json"
        if not splits_path.is_file():
            raise DataFormatError(f"missing file: {splits_path}")
        splits = json.loads(splits_path.read_text())
        if split not in splits:
            raise DataFormatError(f"splits.json: missing key '{split}'")
        self._indices = np.asarray(splits[split], dtype=np.int64)

        images_path = self.root / "images.npy"
        if not images_path.is_file():
            raise DataFormatError(f"missing file: {images_path}")
        self._images = np.load(images_path, mmap_mode="r")

    def __len__(self) -> int:
        return int(self._indices.shape[0])

    def __getitem__(self, i: int) -> tuple[torch.Tensor, dict[str, Any]]:
        idx = int(self._indices[i])
        img = np.array(self._images[idx])  # copy out of the memmap
        if self.random_flip and np.random.rand() > 0.5:
            img = np.ascontiguousarray(img[:, ::-1])
        tensor = torch.from_numpy(img.astype(np.float32) / 255.0)[None]
        return tensor, {}


def make_loaders(config: Any) -> tuple[DataLoader, DataLoader]:
    """Build the train and eval ``DataLoader``s for the released trainer.

    Parameters
    ----------
    config : ml_collections.ConfigDict
        Must provide ``data.root``, ``data.dataset``, ``data.split_train``,
        ``data.split_eval``, ``data.random_flip``, ``data.num_workers``,
        ``training.batch_size``, ``eval.batch_size`` and ``seed``.

    Returns
    -------
    tuple[DataLoader, DataLoader]
        ``(train_loader, eval_loader)``: the train loader shuffles, drops the
        last incomplete batch and uses a ``torch.Generator`` seeded from
        ``config.seed``; the eval loader does not shuffle.
    """
    root = Path(config.data.root) / config.data.dataset
    train_dataset = NpyImageDataset(root, config.data.split_train, random_flip=config.data.random_flip)
    eval_dataset = NpyImageDataset(root, config.data.split_eval, random_flip=False)

    generator = torch.Generator()
    generator.manual_seed(int(config.seed))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=config.data.num_workers,
        generator=generator,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.eval.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )
    return train_loader, eval_loader
