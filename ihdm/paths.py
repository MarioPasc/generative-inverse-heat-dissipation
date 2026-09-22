"""Filesystem paths for the spectral-allocation IHDM experiment.

Every path is overridable through an environment variable so the same code runs
unchanged on the local workstations and on Picasso. None of the functions here
create directories; callers are responsible for that.
"""

import os
from pathlib import Path


def repo_root() -> Path:
    """Return the repository root (the directory containing the ``ihdm`` package).

    Returns
    -------
    Path
        Absolute path to the repository root.
    """
    return Path(__file__).resolve().parent.parent


def data_root() -> Path:
    """Return the root directory holding the standard-format datasets.

    Overridable with the ``IHDM_DATA_ROOT`` environment variable.

    Returns
    -------
    Path
        The data root, from the environment or the local default.
    """
    default = "/media/mpascual/MeningD2/spectral_allocation_heat_diffusion_project"
    return Path(os.environ.get("IHDM_DATA_ROOT", default))


def run_root() -> Path:
    """Return the root directory holding local training run outputs.

    Overridable with the ``IHDM_RUN_ROOT`` environment variable.

    Returns
    -------
    Path
        The run root, from the environment or ``<repo_root>/runs``.
    """
    default = str(repo_root() / "runs")
    return Path(os.environ.get("IHDM_RUN_ROOT", default))


def raw_root() -> Path:
    """Return the root directory holding raw (unprocessed) source data.

    Overridable with the ``IHDM_RAW_ROOT`` environment variable.

    Returns
    -------
    Path
        The raw-data root, from the environment or the local default.
    """
    default = "/media/mpascual/MeningD2"
    return Path(os.environ.get("IHDM_RAW_ROOT", default))


def template_dir() -> Path:
    """Return the directory holding the MNI152 registration templates.

    Overridable with the ``IHDM_TEMPLATE_DIR`` environment variable.

    Returns
    -------
    Path
        The template directory, from the environment or the local default.
    """
    default = "/media/mpascual/MeningD2/UNPROCESSED_MRI/templates"
    return Path(os.environ.get("IHDM_TEMPLATE_DIR", default))


def schedules_dir() -> Path:
    """Return the directory holding the frozen blur-schedule arrays.

    Returns
    -------
    Path
        ``<repo_root>/schedules``.
    """
    return repo_root() / "schedules"
