"""Tests of the per-site and sensitivity helpers added by T1.4."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ihdm.data.format import DatasetMeta, write_dataset
from ihdm.spectral.errors import SpectralError
from ihdm.spectral.profile import (
    SENSITIVITY_DIR,
    SENSITIVITY_IDS,
    SITE_ORDER,
    coarse_bin_breakdown,
    sensitivity_root,
    site_profiles,
)

SIDE = 32
SLICES = 2
SITE_OF = {"Guys": "100_Guys/IXI{n:03d}-Guys-0747-T1.nii.gz",
           "HH": "12_HH/IXI{n:03d}-HH-1211-T1.nii.gz",
           "IOP": "581_IOP/IXI{n:03d}-IOP-1145-T1.nii.gz"}


MODE_COEFFICIENT = 0.02 * SIDE


def _mode_image(mode: tuple[int, int], coefficient: float) -> np.ndarray:
    """Return the image of one **orthonormal** DCT-II basis mode, times ``coefficient``.

    Normalising to unit L2 norm matters: ``cos(1 . x) cos(0 . y)`` and
    ``cos(1 . x) cos(1 . y)`` differ by a factor of two in energy, so an un-normalised pair
    would not give the two modes the same variance for the same coefficient.
    """
    grid = (np.arange(SIDE) + 0.5) * np.pi / SIDE
    image = np.cos(mode[0] * grid)[:, None] * np.cos(mode[1] * grid)[None, :]
    return coefficient * image / float(np.linalg.norm(image))


def _write_mri_dataset(root: Path, dataset_id: str, amplitudes: dict[str, float]) -> None:
    """Write a tiny MRI-shaped dataset whose per-site variance sits in a known mode.

    Each site's subjects differ from each other only through the coefficients of the ``(1, 0)``
    mode (an anterior-posterior ramp), scaled by ``amplitudes[site]``, and of ``(1, 1)``, which
    is not scaled. The coefficients are a fixed antisymmetric ladder rather than random draws,
    so the two modes have exactly the same between-subject variance before scaling and the
    ``(1, 0)`` share of the coarse bin is ``a^2 / (a^2 + 1)`` by construction.
    """
    ladder = np.array([-1.5, -0.9, -0.3, 0.3, 0.9, 1.5])
    subjects: list[str] = []
    sources: list[str] = []
    images: list[np.ndarray] = []
    number = 1
    for site, amplitude in amplitudes.items():
        for position in range(len(ladder)):
            subject = f"IXI{number:03d}"
            base = (
                0.5
                + amplitude * ladder[position] * _mode_image((1, 0), MODE_COEFFICIENT)
                + ladder[-1 - position] * _mode_image((1, 1), MODE_COEFFICIENT)
            )
            for _slice in range(SLICES):
                images.append(np.clip(base, 0.0, 1.0))
                subjects.append(subject)
                sources.append(SITE_OF[site].format(n=number))
            number += 1
    stack = np.round(np.stack(images) * 255.0).astype(np.uint8)
    n = stack.shape[0]
    index = pd.DataFrame(
        {
            "idx": np.arange(n),
            "subject": subjects,
            "slice": [i % SLICES for i in range(n)],
            "z_mm": [float(i % SLICES) for i in range(n)],
            "source": sources,
            "split": ["train"] * n,
        }
    )
    unique = sorted(set(subjects))
    splits = {
        "train": list(range(n)), "ref": [], "seed": [],
        "train_subjects": unique, "ref_subjects": [], "seed_subjects": [],
        "rule": "everything in train (test fixture)", "rng_seed": 2026,
    }
    meta = DatasetMeta(
        dataset_id=dataset_id, n_images=n, image_size=SIDE, dtype="uint8",
        pipeline="test", pipeline_version="1.1", git_sha="test",
        created="2026-09-23T00:00:00+00:00", raw_root="test",
        parameters={}, counts={},
    )
    write_dataset(root / dataset_id, stack, index, splits, meta)


def test_coarse_bin_breakdown_reads_the_three_modes() -> None:
    """The breakdown returns the bin's share of the total and ``(1,0)``'s share of the bin."""
    power = np.zeros((64, 64))
    power[0, 1], power[1, 0], power[1, 1] = 1.0, 8.0, 1.0
    power[20, 20] = 10.0
    coarse, mode_10 = coarse_bin_breakdown(power)
    assert mode_10 == pytest.approx(0.8)
    assert coarse == pytest.approx(10.0 / 20.0)


def test_sensitivity_root_is_the_underscore_folder(tmp_path: Path) -> None:
    """The archive lives next to the datasets, under one fixed name."""
    assert sensitivity_root(tmp_path) == tmp_path / SENSITIVITY_DIR
    assert SENSITIVITY_IDS == {"ixi": "ixi_no_n4", "oasis1": "oasis1_no_n4"}


def test_site_profiles_split_a_cohort_by_acquisition_site(tmp_path: Path) -> None:
    """Each site is measured on its own images and ordered by ``SITE_ORDER``."""
    _write_mri_dataset(tmp_path, "ixi", {"Guys": 1.0, "HH": 3.0, "IOP": 0.3})
    profiles = site_profiles(tmp_path, "ixi", "train")

    assert [p.site for p in profiles] == ["Guys", "HH", "IOP"]
    assert all(p.n_subjects == 6 for p in profiles)
    assert all(p.n_images == 6 * SLICES for p in profiles)
    by_site = {p.site: p for p in profiles}
    # The (1,0) share inside the coarse bin must follow the amplitude the fixture imposed.
    for site, amplitude in (("Guys", 1.0), ("HH", 3.0), ("IOP", 0.3)):
        expected = amplitude**2 / (amplitude**2 + 1.0)
        assert by_site[site].mode_10_share == pytest.approx(expected, abs=0.01), site
    assert all(p.coarse_share > 0.98 for p in profiles)


def test_site_profiles_reads_the_archived_copy(tmp_path: Path) -> None:
    """The same code path works on an archived dataset, whose meta has no site map."""
    archive = sensitivity_root(tmp_path)
    _write_mri_dataset(archive, "ixi_no_n4", {"Guys": 1.0, "HH": 3.0})
    meta = json.loads((archive / "ixi_no_n4" / "meta.json").read_text())
    assert "sites" not in meta["parameters"]

    profiles = site_profiles(archive, "ixi_no_n4", "train")
    assert [p.site for p in profiles] == ["Guys", "HH"]
    assert all(p.site in SITE_ORDER for p in profiles)


def test_site_profiles_rejects_a_missing_split(tmp_path: Path) -> None:
    """An empty or unknown split is a hard error, not an empty table."""
    _write_mri_dataset(tmp_path, "ixi", {"Guys": 1.0})
    with pytest.raises(SpectralError):
        site_profiles(tmp_path, "ixi", "ref")
