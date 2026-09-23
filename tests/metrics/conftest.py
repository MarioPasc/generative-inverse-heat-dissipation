"""Fixtures of the metric tests that depend on the machine rather than on the mathematics.

The synthetic fields live in the test module itself; only the discovery of the real sample
folders and of the ``ixi`` dataset is here, so that the ``integration`` tests skip cleanly when
the data disk is not mounted.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def pilot_sample_dirs() -> dict[str, Path] | None:
    """The two pilot sample folders drawn by T4.1, or ``None`` when they are absent.

    Returns
    -------
    dict[str, Path] or None
        ``{"lsd": ..., "seed": ..., "data_root": ...}`` when every required path exists.
    """
    data_root = Path(
        os.environ.get(
            "IHDM_DATA_ROOT", "/media/mpascual/MeningD2/spectral_allocation_heat_diffusion_project"
        )
    )
    run_root = Path(os.environ.get("IHDM_RUN_ROOT", str(data_root / "_runs_local")))
    paths = {
        "lsd": run_root / "t41_lsd_samples",
        "seed": run_root / "t41_seed_samples",
        "data_root": data_root,
    }
    needed = [
        paths["lsd"] / "samples.npy",
        paths["seed"] / "samples.npy",
        paths["seed"] / "seeds.npy",
        data_root / "ixi" / "images.npy",
        data_root / "ixi" / "splits.json",
    ]
    return paths if all(item.is_file() for item in needed) else None


@pytest.fixture(scope="session")
def ixi_dataset(pilot_sample_dirs: dict[str, Path] | None) -> dict[str, Any] | None:
    """The ``ixi`` standard-format dataset the pilot samples were drawn from, or ``None``.

    T4.2's integration tests need the training split, the ``ref``-labelled held-out rows and the
    ``subject`` column, which ``pilot_sample_dirs`` does not carry.

    Parameters
    ----------
    pilot_sample_dirs : dict[str, Path] or None
        T4.1's fixture; ``None`` when the data disk is not mounted.

    Returns
    -------
    dict[str, Any] or None
        ``{"images": memmap (N, H, W) uint8, "index": DataFrame, "splits": dict}``.
    """
    if pilot_sample_dirs is None:
        return None
    root = pilot_sample_dirs["data_root"] / "ixi"
    if not (root / "index.csv").is_file():
        return None
    return {
        "images": np.load(root / "images.npy", mmap_mode="r"),
        "index": pd.read_csv(root / "index.csv"),
        "splits": json.loads((root / "splits.json").read_text()),
    }
