"""Fixtures of the metric tests that depend on the machine rather than on the mathematics.

The synthetic fields live in the test module itself; only the discovery of the real sample
folders is here, so that the ``integration`` test skips cleanly when the data disk is not
mounted.
"""

from __future__ import annotations

import os
from pathlib import Path

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
