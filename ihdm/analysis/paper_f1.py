"""F1 of the course paper: the MRI-centric visual abstract (T8.1).

The design is ``docs/SPECIFICATIONS/M8-paper/figure-plan-part1.md`` §1 (version 2) and the data,
selection rules and verification are those of ``T8.1-visual-abstract.md``. Three panels:

* (a) one log frequency axis, in cycles per image (c/img), carrying the heat states of one IXI
  slice at :math:`c = 43.2/\\sigma_B`, the IXI variance share per octave with the LSUN Churches
  outline and the :math:`1/f^2` equal share, the variance kept by the two priors, and the reverse
  levels of the default and matched schedules;
* (b) the mean IXI training slice, and three subjects' noise-free prior states at W/2 and W/8;
* (c) the 2 × 2 key of the four arms and one unseen subject sampled by the default and the
  matched configuration, with the KID and seed-NN changes.

Every annotated number is read from a file at draw time (:class:`Annotations`). The mapping
between a blur level and a frequency is the released ``model_code.utils.DCTBlur``: per DCT mode,
:math:`d = \\exp(-\\lambda t)` with :math:`\\lambda = (\\pi k_x/W)^2 + (\\pi k_y/W)^2` and
:math:`t = \\sigma_B^2/2`. A mode carrying :math:`c` c/img has radial index :math:`n = 2c`, hence
the length-scale :math:`\\sigma_n = \\sqrt{2/\\lambda} = W/(\\sqrt 2\\pi c) \\approx 43.2/c` px
at W = 192, and :math:`d^2 = \\exp(-2\\sigma_B^2/\\sigma_n^2)`. Half the variance survives where
:math:`d^2 = 1/2`, i.e. at
:math:`c_{1/2} = W\\sqrt{\\ln 2}/(2\\pi\\sigma_B) \\approx 25.4/\\sigma_B`.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import re
import tarfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.text import Text
from scipy.fft import dctn, idctn
from scipy.optimize import brentq

from ihdm.analysis.style import INK, MUTED, arm_style, figure_style
from ihdm.analysis.tables import AnalysisError
from ihdm.data.format import read_dataset
from ihdm.spectral.power import eigenvalues, octave_bins, radial_index
from ihdm.spectral.profile import reference_example
from ihdm.spectral.schedules import levels_per_octave

__all__ = [
    "ARM_LABELS",
    "DEFAULT_RULES",
    "PAPER_WIDTH_IN",
    "Annotations",
    "F1Data",
    "F1Inputs",
    "LayoutReport",
    "MappingCheck",
    "MissingInputError",
    "PaperFigureError",
    "SelectionRules",
    "caption_text",
    "cycles_of_sigma",
    "draw_f1",
    "extract_heldout",
    "f1_markdown",
    "half_power_cycles",
    "half_power_cycles_numeric",
    "heat_blur",
    "heat_multiplier",
    "kept_power",
    "kept_variance_per_octave",
    "levels_per_frequency_octave",
    "layout_report",
    "load_annotations",
    "load_f1_data",
    "mode_scale",
    "paper_style",
    "parse_level_table",
    "released_dct_blur",
    "save_f1",
    "sha256_of",
    "verify_annotations",
    "verify_level_counts",
    "verify_mapping",
]

logger = logging.getLogger(__name__)

#: The NIPS 2015 ``\textwidth`` of the course template (decision of 2026-10-06).
PAPER_WIDTH_IN: float = 5.5
#: The ticket's height cap for F1.
PAPER_MAX_HEIGHT_IN: float = 2.9
#: F1's height; the layout below is drawn in inches against it.
F1_HEIGHT_IN: float = 2.85
#: Display names of the arms, by what is matched (figure-plan-part1.md §0, accepted 2026-10-06).
ARM_LABELS: dict[str, str] = {"A0": "default", "A1": "+prior", "A2": "+spacing", "A3": "matched"}
#: Smallest font and thumbnail the ticket allows.
MIN_FONT_PT: float = 7.0
MIN_THUMB_IN: float = 0.35
#: Raster resolution of the PNG; the PDF and SVG embed the 192² thumbnails unresampled.
PNG_DPI: int = 300
PDF_DPI: int = 300
#: Output stem.
F1_NAME: str = "f1_visual_abstract"

#: Heat states of panel (a), terminal blur first (the default prior, then the matched prior).
HEAT_SIGMAS: tuple[float, ...] = (96.0, 24.0, 8.0, 2.0)
#: Blur levels of the frequency-mapping check (ticket verification 1).
VERIFY_SIGMAS: tuple[float, ...] = (2.0, 8.0, 24.0, 96.0)
#: Terminal blurs of the default (W/2) and the matched (W/8) prior at W = 192.
SIGMA_DEFAULT: float = 96.0
SIGMA_MATCHED: float = 24.0
#: Schedules of the two rugs and their expected level counts per sigma_B octave (data_profile §5).
SCHEDULES: dict[str, str] = {"default": "log_W2", "matched": "ixi_W8"}
EXPECTED_LEVEL_COUNTS: dict[str, tuple[int, ...]] = {
    "log_W2": (27, 26, 26, 26, 27, 26, 26, 16),
    "ixi_W8": (24, 37, 46, 43, 32, 18, 0, 0),
}
#: Tolerances of verification 1: relative on the kept share, plus an absolute floor for octaves
#: the prediction puts below double-precision round-off (exp(-hundreds) at sigma 96).
MAPPING_RTOL: float = 1e-6
MAPPING_ATOL: float = 1e-12
HALF_POWER_RTOL: float = 0.01

#: Paper-specific rcParams on top of the house style: every font >= 7 pt, editable SVG text and
#: deterministic SVG ids.
_PAPER_RC: dict[str, object] = {
    "font.size": 7.0,
    "axes.titlesize": 7.0,
    "axes.labelsize": 7.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 7.0,
    "axes.grid": False,
    "svg.fonttype": "none",
    "svg.hashsalt": "ihdm-paper",
    "svg.image_inline": True,
}

#: Neutral inks of the MRI bars and the natural-image reference.
BAR_FILL: str = "#b9b7b1"
BAR_EDGE: str = "#6f6d68"
REFERENCE_INK: str = "#3a3936"


class PaperFigureError(AnalysisError):
    """F1 cannot be drawn: a selection rule, a verification or a source file is inconsistent."""


class MissingInputError(PaperFigureError):
    """A required input file or archive member does not exist."""


@dataclass(frozen=True)
class SelectionRules:
    """The pre-declared selection rules of the ticket, asserted in code.

    Attributes
    ----------
    dataset : str
        Dataset id of every MRI image in the figure.
    example_rng_seed : int
        Seed of ``reference_example`` for the heat-state strip of panel (a).
    n_subjects : int
        Number of ``ref_subjects`` shown in panel (b), from the start of the list.
    subject_slice : int
        Slice of those subjects.
    n_train : int
        Size of the ``train`` split whose mean panel (b) shows.
    runs : tuple[str, str]
        Default and matched runs of the teaser.
    step : str
        Checkpoint directory of the teaser samples.
    teaser_seed_position : int
        Position in ``heldout/seed_idx.npy``.
    teaser_dataset_index : int
        The dataset index that position must hold.
    teaser_sample : int
        Sample of that seed (of ``n_per_seed``).
    request_rng_seed, request_batch_size : int
        Values both runs' ``request.json`` signatures must share.
    """

    dataset: str = "ixi"
    example_rng_seed: int = 2026
    n_subjects: int = 3
    subject_slice: int = 5
    n_train: int = 3200
    runs: tuple[str, str] = ("ixi_A0_s1", "ixi_A3_s1")
    step: str = "060000"
    teaser_seed_position: int = 1
    teaser_dataset_index: int = 35
    teaser_sample: int = 0
    request_rng_seed: int = 2026
    request_batch_size: int = 32


DEFAULT_RULES = SelectionRules()

#: The four members of each evaluation tar that the teaser needs.
HELDOUT_MEMBERS: tuple[str, ...] = ("samples.npy", "seed_idx.npy", "seeds.npy", "request.json")


@dataclass(frozen=True)
class F1Inputs:
    """Every file F1 reads.

    Attributes
    ----------
    data_root : Path
        The IHDM data root holding ``<dataset>/images.npy`` and its index.
    eval_dir : Path
        Folder of the evaluation tars ``<run>_amp-fp16.tar``.
    repo_root : Path
        Repository root holding ``docs/RESULTS`` and ``schedules``.
    work_dir : Path
        Scratch folder for the extracted held-out members.
    """

    data_root: Path
    eval_dir: Path
    repo_root: Path
    work_dir: Path

    def profile(self, dataset: str) -> Path:
        """``docs/RESULTS/data_profile/<dataset>.npz``."""
        return self.repo_root / "docs" / "RESULTS" / "data_profile" / f"{dataset}.npz"

    def schedule(self, name: str) -> Path:
        """``schedules/<name>.npy``."""
        return self.repo_root / "schedules" / f"{name}.npy"

    @property
    def tables_json(self) -> Path:
        """``docs/RESULTS/tables/tables.json``."""
        return self.repo_root / "docs" / "RESULTS" / "tables" / "tables.json"

    @property
    def data_profile_md(self) -> Path:
        """``docs/RESULTS/data_profile.md`` (the level-count table of §5)."""
        return self.repo_root / "docs" / "RESULTS" / "data_profile.md"

    def tar(self, run: str) -> Path:
        """The evaluation tar of ``run``."""
        return self.eval_dir / f"{run}_amp-fp16.tar"

    def required(self, rules: SelectionRules) -> list[Path]:
        """Every input file, for the missing-input check."""
        dataset = self.data_root / rules.dataset
        files = [dataset / name for name in ("images.npy", "index.csv", "splits.json", "meta.json")]
        files += [self.profile(rules.dataset), self.profile("lsun_church")]
        files += [self.schedule(name) for name in SCHEDULES.values()]
        files += [self.tables_json, self.data_profile_md]
        files += [self.tar(run) for run in rules.runs]
        return files


@dataclass(frozen=True)
class Annotations:
    """Every number written on F1, with its source.

    Attributes
    ----------
    octave_labels : tuple[str, ...]
        The eight octave labels in c/img.
    ixi_shares, church_shares : tuple[float, ...]
        Variance share per octave of the IXI and LSUN Churches ``train`` splits.
    inherited_default, inherited_matched : float
        Share of the IXI ``train`` between-image variance kept by the W/2 and W/8 priors.
    kid_relative : float
        IXI ``Δ / A0`` of KID for A3 − A0 (table 3).
    kid_n_seeds : int
        Number of paired seeds behind it.
    seed_nn_default, seed_nn_matched : float
        IXI seed-mean seed-NN fraction of A0 and A3 (table 1b).
    seed_nn_n_seeds : tuple[int, int]
        Runs behind each mean.
    level_counts : dict[str, tuple[int, ...]]
        Levels per sigma_B octave of each rug's schedule.
    sources : dict[str, str]
        Human-readable source of each group of numbers.
    """

    octave_labels: tuple[str, ...]
    ixi_shares: tuple[float, ...]
    church_shares: tuple[float, ...]
    inherited_default: float
    inherited_matched: float
    kid_relative: float
    kid_n_seeds: int
    seed_nn_default: float
    seed_nn_matched: float
    seed_nn_n_seeds: tuple[int, int]
    level_counts: dict[str, tuple[int, ...]]
    sources: dict[str, str] = field(default_factory=dict)

    def share_text(self, which: str, label: str) -> str:
        """``"29.8%"``: one octave share as printed on the figure."""
        shares = self.ixi_shares if which == "ixi" else self.church_shares
        return f"{100.0 * shares[self.octave_labels.index(label)]:.1f}%"

    @property
    def inherited_default_text(self) -> str:
        """``"0.3%"``."""
        return f"{100.0 * self.inherited_default:.1f}%"

    @property
    def inherited_matched_text(self) -> str:
        """``"8.5%"``."""
        return f"{100.0 * self.inherited_matched:.1f}%"

    @property
    def kid_text(self) -> str:
        """``"−63%"`` (typographic minus)."""
        return f"{100.0 * self.kid_relative:+.0f}%".replace("-", "\u2212")

    @property
    def seed_nn_texts(self) -> tuple[str, str]:
        """``("6%", "72%")``."""
        return f"{100.0 * self.seed_nn_default:.0f}%", f"{100.0 * self.seed_nn_matched:.0f}%"

    @property
    def coarsest(self) -> str:
        """Label of the coarsest octave."""
        return self.octave_labels[0]

    @property
    def ixi_peak(self) -> str:
        """Label of IXI's largest octave."""
        return self.octave_labels[int(np.argmax(self.ixi_shares))]


@dataclass
class F1Data:
    """Everything :func:`draw_f1` draws, already selected and verified.

    Attributes
    ----------
    width : int
        Image side in pixels.
    example : numpy.ndarray
        The heat-state example, ``float64`` in [0, 1].
    example_index : int
        Its dataset index.
    heat_states : dict[float, numpy.ndarray]
        Noise-free :math:`u(\\sigma_B)` of the example, per :data:`HEAT_SIGMAS`.
    mean_image : numpy.ndarray
        Mean of the ``train`` split.
    n_train : int
        Number of images averaged.
    subjects : list[tuple[str, int]]
        ``(subject, dataset index)`` of panel (b).
    subject_priors : dict[float, list[numpy.ndarray]]
        Noise-free priors of those subjects at the two terminal blurs.
    teaser_seed : numpy.ndarray
        The unseen subject's seed image.
    teaser_samples : dict[str, numpy.ndarray]
        ``{"default": ..., "matched": ...}`` samples, in [0, 1].
    teaser_index : int
        The seed's dataset index.
    schedules : dict[str, numpy.ndarray]
        ``{"default": log_W2, "matched": ixi_W8}`` levels without level 0.
    annotations : Annotations
        The numbers.
    """

    width: int
    example: np.ndarray
    example_index: int
    heat_states: dict[float, np.ndarray]
    mean_image: np.ndarray
    n_train: int
    subjects: list[tuple[str, int]]
    subject_priors: dict[float, list[np.ndarray]]
    teaser_seed: np.ndarray
    teaser_samples: dict[str, np.ndarray]
    teaser_index: int
    schedules: dict[str, np.ndarray]
    annotations: Annotations


# --------------------------------------------------------------------------------------------
# The frequency mapping
# --------------------------------------------------------------------------------------------


def mode_scale(width: int) -> float:
    """Constant of :math:`c = s/\\sigma`: :math:`s = W/(\\sqrt 2\\pi)` (43.2 at W = 192).

    Parameters
    ----------
    width : int
        Image side in pixels.

    Returns
    -------
    float
        The constant in c/img × px.
    """
    return float(width) / (np.sqrt(2.0) * np.pi)


def cycles_of_sigma(sigma: float | np.ndarray, width: int) -> float | np.ndarray:
    """Frequency, in c/img, of the DCT mode whose length-scale is ``sigma`` px.

    Parameters
    ----------
    sigma : float or numpy.ndarray
        Blur length-scale(s) in pixels, positive.
    width : int
        Image side in pixels.

    Returns
    -------
    float or numpy.ndarray
        :math:`c = W/(\\sqrt 2\\pi\\sigma)`.
    """
    if np.ndim(sigma):
        return mode_scale(width) / np.asarray(sigma, dtype=np.float64)
    return mode_scale(width) / float(sigma)


def kept_power(cycles: np.ndarray, sigma: float, width: int) -> np.ndarray:
    """Variance multiplier :math:`d^2(c) = \\exp(-\\lambda\\sigma^2)` of a mode at ``cycles``.

    Parameters
    ----------
    cycles : numpy.ndarray
        Frequencies in c/img (radial index ``n = 2c``).
    sigma : float
        Blur length-scale in pixels.
    width : int
        Image side in pixels.

    Returns
    -------
    numpy.ndarray
        The kept fraction of each mode's variance.
    """
    lam = (np.pi * 2.0 * np.asarray(cycles, dtype=np.float64) / width) ** 2
    return np.exp(-lam * sigma**2)


def half_power_cycles(sigma: float, width: int) -> float:
    """Analytic frequency at which the blur keeps half a mode's variance.

    Parameters
    ----------
    sigma : float
        Blur length-scale in pixels.
    width : int
        Image side in pixels.

    Returns
    -------
    float
        :math:`c_{1/2} = W\\sqrt{\\ln 2}/(2\\pi\\sigma)` (25.4/σ at W = 192).
    """
    return float(width) * np.sqrt(np.log(2.0)) / (2.0 * np.pi * sigma)


def half_power_cycles_numeric(sigma: float, width: int) -> float:
    """Numeric root of :math:`d^2(c) = 1/2`, independent of :func:`half_power_cycles`.

    Parameters
    ----------
    sigma : float
        Blur length-scale in pixels.
    width : int
        Image side in pixels.

    Returns
    -------
    float
        The root in c/img.
    """
    return float(brentq(lambda c: float(kept_power(np.array([c]), sigma, width)[0]) - 0.5,
                        1e-9, float(width), xtol=1e-14, rtol=1e-14))


def heat_multiplier(width: int, sigma: float) -> np.ndarray:
    """Per-mode heat multiplier :math:`d = \\exp(-\\lambda\\sigma^2/2)` of ``DCTBlur``.

    Parameters
    ----------
    width : int
        Image side in pixels.
    sigma : float
        Blur length-scale in pixels.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(W, W)``.
    """
    return np.exp(-eigenvalues(width) * sigma**2 / 2.0)


def heat_blur(images: np.ndarray, sigma: float) -> np.ndarray:
    """The released ``DCTBlur`` formula in NumPy, float64: ``idct(dct(x) · d)``.

    Parameters
    ----------
    images : numpy.ndarray
        One image ``(W, W)`` or a stack ``(n, W, W)``.
    sigma : float
        Blur length-scale in pixels; 0 returns the input.

    Returns
    -------
    numpy.ndarray
        The heat state :math:`u(\\sigma)`, same shape, ``float64``.
    """
    x = np.asarray(images, dtype=np.float64)
    if sigma == 0.0:
        return x.copy()
    d = heat_multiplier(x.shape[-1], sigma)
    axes = (-2, -1)
    return idctn(dctn(x, axes=axes, norm="ortho") * d, axes=axes, norm="ortho")


def released_dct_blur(images: np.ndarray, sigma: float) -> np.ndarray:
    """Blur a stack with the released ``model_code.utils.DCTBlur`` itself, in float64.

    ``DCTBlur`` builds its frequencies with ``torch.linspace`` and its sigmas with
    ``torch.tensor``, both in the default dtype; it is constructed under a float64 default so
    that the comparison with the float64 prediction is not limited by float32 rounding of
    :math:`\\lambda t` (about 1e-7 relative, i.e. above 1e-6 in :math:`d^2` once
    :math:`\\lambda t > 5`).

    Parameters
    ----------
    images : numpy.ndarray
        Stack ``(n, W, W)``.
    sigma : float
        Blur length-scale in pixels.

    Returns
    -------
    numpy.ndarray
        Blurred stack, ``float64``.
    """
    import torch

    from model_code.utils import DCTBlur

    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        blur = DCTBlur([float(sigma)], int(images.shape[-1]), "cpu")
    finally:
        torch.set_default_dtype(previous)
    x = torch.from_numpy(np.ascontiguousarray(images, dtype=np.float64))
    steps = torch.zeros(x.shape[0], dtype=torch.long)
    with torch.no_grad():
        return blur(x, steps).numpy()


def _octave_masks(width: int) -> list[np.ndarray]:
    """Mask per octave of ``05-metrics.md`` §1 (the last bin closed at 96 c/img; DC excluded)."""
    cycles = radial_index(width) / 2.0
    masks = []
    bins = octave_bins()
    for _, lo, hi in bins:
        inside = (cycles >= lo) & (cycles < hi)
        if hi == bins[-1][2]:
            inside |= cycles == hi
        masks.append(inside)
    return masks


@dataclass(frozen=True)
class MappingCheck:
    """Verification 1 for one blur level.

    Attributes
    ----------
    sigma : float
        Blur level in pixels.
    measured, predicted : tuple[float, ...]
        Per-octave share of the variance kept, by ``DCTBlur`` and by :math:`\\sum d^2P/\\sum P`.
    max_rel_error : float
        Largest relative error over the octaves above the absolute floor.
    half_power : float
        :math:`25.4/\\sigma` as printed (the analytic constant rounded to 0.1).
    half_power_numeric : float
        Numeric root of :math:`d^2 = 1/2`.
    passed : bool
        Both checks within tolerance.
    """

    sigma: float
    measured: tuple[float, ...]
    predicted: tuple[float, ...]
    max_rel_error: float
    half_power: float
    half_power_numeric: float
    passed: bool

    @property
    def half_power_rel_error(self) -> float:
        """Relative gap between the printed constant and the numeric root."""
        return abs(self.half_power - self.half_power_numeric) / self.half_power_numeric


def kept_variance_per_octave(
    images: np.ndarray, sigmas: Sequence[float], chunk: int = 256
) -> tuple[dict[float, np.ndarray], dict[float, np.ndarray]]:
    """Per-octave share of the between-image variance kept by ``DCTBlur``, measured and predicted.

    The stack is centred on its own mean (the between-image variance of ``mode_power``), blurred
    with :func:`released_dct_blur`, and transformed again; the measured share of an octave is the
    blurred stack's variance over the unblurred one's. The prediction is
    :math:`\\sum_{i\\in b} d_i^2 P_i / \\sum_{i\\in b} P_i`.

    Parameters
    ----------
    images : numpy.ndarray
        Stack ``(n, W, W)``; ``uint8`` is scaled to [0, 1].
    sigmas : Sequence[float]
        Blur levels in pixels.
    chunk : int
        Images processed at a time.

    Returns
    -------
    tuple[dict[float, numpy.ndarray], dict[float, numpy.ndarray]]
        ``(measured, predicted)``: eight shares per sigma.

    Raises
    ------
    PaperFigureError
        If the stack is empty or holds no variance in some octave.
    """
    if images.ndim != 3 or images.shape[0] < 2:
        raise PaperFigureError("the mapping check needs a stack of at least two images")
    scale = 255.0 if images.dtype == np.uint8 else 1.0
    n, width = images.shape[0], images.shape[-1]
    mean = np.zeros((width, width))
    for start in range(0, n, chunk):
        mean += np.asarray(images[start:start + chunk], dtype=np.float64).sum(axis=0) / scale
    mean /= n
    power = np.zeros((width, width))
    blurred = {float(s): np.zeros((width, width)) for s in sigmas}
    for start in range(0, n, chunk):
        x = np.asarray(images[start:start + chunk], dtype=np.float64) / scale - mean
        power += np.square(dctn(x, axes=(1, 2), norm="ortho")).sum(axis=0)
        for sigma in blurred:
            xb = released_dct_blur(x, sigma)
            blurred[sigma] += np.square(dctn(xb, axes=(1, 2), norm="ortho")).sum(axis=0)
    masks = _octave_masks(width)
    totals = np.array([power[m].sum() for m in masks])
    if not np.all(totals > 0.0):
        raise PaperFigureError("an octave holds no variance; the mapping check is undefined")
    measured, predicted = {}, {}
    for sigma, pb in blurred.items():
        d2p = heat_multiplier(width, sigma) ** 2 * power
        measured[sigma] = np.array([pb[m].sum() for m in masks]) / totals
        predicted[sigma] = np.array([d2p[m].sum() for m in masks]) / totals
    return measured, predicted


def verify_mapping(images: np.ndarray, sigmas: Sequence[float] = VERIFY_SIGMAS,
                   chunk: int = 256) -> list[MappingCheck]:
    """Verification 1: kept variance per octave and the half-power frequency, per blur level.

    Parameters
    ----------
    images : numpy.ndarray
        The IXI ``train`` stack (or a synthetic stack in the tests).
    sigmas : Sequence[float]
        Blur levels in pixels.
    chunk : int
        Images processed at a time.

    Returns
    -------
    list[MappingCheck]
        One record per blur level.
    """
    width = int(images.shape[-1])
    measured, predicted = kept_variance_per_octave(images, sigmas, chunk)
    # The printed constant: the analytic value rounded to one decimal, as written in the caption.
    printed = round(half_power_cycles(1.0, width), 1)
    checks = []
    for sigma in (float(s) for s in sigmas):
        m, p = measured[sigma], predicted[sigma]
        ok = np.abs(m - p) <= MAPPING_RTOL * np.abs(p) + MAPPING_ATOL
        above = p > MAPPING_ATOL / MAPPING_RTOL
        rel = float(np.max(np.abs(m - p)[above] / p[above])) if above.any() else 0.0
        numeric = half_power_cycles_numeric(sigma, width)
        hp_ok = abs(printed / sigma - numeric) / numeric <= HALF_POWER_RTOL
        checks.append(MappingCheck(sigma, tuple(map(float, m)), tuple(map(float, p)), rel,
                                   printed / sigma, numeric, bool(ok.all() and hp_ok)))
    return checks


# --------------------------------------------------------------------------------------------
# Level counts and annotated numbers
# --------------------------------------------------------------------------------------------


def parse_level_table(path: Path) -> dict[str, tuple[int, ...]]:
    """Read the level-count table of ``data_profile.md`` §5.

    Parameters
    ----------
    path : Path
        ``docs/RESULTS/data_profile.md``.

    Returns
    -------
    dict[str, tuple[int, ...]]
        Eight counts per schedule name (the backticked first cell; annotated rows are skipped).

    Raises
    ------
    PaperFigureError
        If §5 or its table is absent.
    """
    text = Path(path).read_text(encoding="utf-8")
    match = re.search(r"^## 5\..*?$(.*?)^## 6\.", text, flags=re.M | re.S)
    if match is None:
        raise PaperFigureError(f"{path}: no section 5 (levels per sigma_B octave)")
    rows: dict[str, tuple[int, ...]] = {}
    for line in match.group(1).splitlines():
        row = re.match(r"^\|\s*`([A-Za-z0-9_]+)`\s*\|((?:\s*\d+\s*\|){9})\s*$", line)
        if row:
            values = tuple(int(v) for v in row.group(2).split("|") if v.strip())
            rows[row.group(1)] = values[:8]
    if not rows:
        raise PaperFigureError(f"{path}: §5 holds no level-count row")
    return rows


def verify_level_counts(schedules: dict[str, np.ndarray],
                        table: dict[str, tuple[int, ...]]) -> dict[str, tuple[int, ...]]:
    """Verification 2: levels per sigma_B octave equal ``data_profile.md`` §5 and the ticket.

    Parameters
    ----------
    schedules : dict[str, numpy.ndarray]
        Schedule name -> the ``(K + 1,)`` array.
    table : dict[str, tuple[int, ...]]
        :func:`parse_level_table` output.

    Returns
    -------
    dict[str, tuple[int, ...]]
        The counts, per schedule name.

    Raises
    ------
    PaperFigureError
        If a count differs from §5 or from the ticket.
    """
    counts = {}
    for name, schedule in schedules.items():
        got = tuple(levels_per_octave(schedule).values())
        if name not in table:
            raise PaperFigureError(f"data_profile.md §5 has no row for {name}")
        if got != table[name]:
            raise PaperFigureError(f"{name}: levels per octave {got} != data_profile §5 "
                                   f"{table[name]}")
        if name in EXPECTED_LEVEL_COUNTS and got != EXPECTED_LEVEL_COUNTS[name]:
            raise PaperFigureError(f"{name}: levels per octave {got} != ticket "
                                   f"{EXPECTED_LEVEL_COUNTS[name]}")
        counts[name] = got
    return counts


def levels_per_frequency_octave(levels: np.ndarray, width: int) -> tuple[tuple[int, ...], int]:
    """Count reverse levels in the bars' frequency octaves, via :math:`c_k = 43.2/\\sigma_k`.

    The bins are the eight c/img octaves of ``05-metrics.md`` §1 (half-open, the last closed at
    96). Levels whose :math:`c_k` lies below 0.5 c/img (σ_B > 86.4 px at W = 192) are counted in
    the first octave and those above 96 c/img in the last, so the counts sum to K.

    Parameters
    ----------
    levels : numpy.ndarray
        Blur levels in pixels, level 0 excluded (all positive).
    width : int
        Image side in pixels.

    Returns
    -------
    tuple[tuple[int, ...], int]
        The eight counts, and how many levels were folded in from below 0.5 c/img.

    Raises
    ------
    PaperFigureError
        If a level is not positive.
    """
    levels = np.asarray(levels, dtype=np.float64)
    if levels.size == 0 or not np.all(levels > 0.0):
        raise PaperFigureError("frequency-octave counts need positive blur levels")
    bins = octave_bins()
    edges = np.array([lo for _, lo, _ in bins] + [bins[-1][2]])
    cycles = np.asarray(cycles_of_sigma(levels, width), dtype=np.float64)
    counts, _ = np.histogram(np.clip(cycles, edges[0], edges[-1]), bins=edges)
    return tuple(int(n) for n in counts), int((cycles < edges[0]).sum())


def _rows(tables: dict[str, Any], key: str) -> list[dict[str, Any]]:
    try:
        return list(tables["tables"][key]["rows"])
    except (KeyError, TypeError) as exc:
        raise PaperFigureError(f"tables.json has no table {key!r}") from exc


def load_annotations(inputs: F1Inputs, rules: SelectionRules = DEFAULT_RULES,
                     level_counts: dict[str, tuple[int, ...]] | None = None) -> Annotations:
    """Read every number written on F1 from its file.

    Parameters
    ----------
    inputs : F1Inputs
        The input paths.
    rules : SelectionRules
        Gives the dataset id.
    level_counts : dict[str, tuple[int, ...]] or None
        Verified level counts per schedule name (computed from the schedules when absent).

    Returns
    -------
    Annotations
        The numbers and their sources.

    Raises
    ------
    PaperFigureError
        If a source lacks a value.
    """
    ixi = np.load(inputs.profile(rules.dataset), allow_pickle=False)
    church = np.load(inputs.profile("lsun_church"), allow_pickle=False)
    labels = tuple(str(v) for v in ixi["octave_labels"])
    if labels != tuple(str(v) for v in church["octave_labels"]):
        raise PaperFigureError("IXI and Churches profiles disagree on the octave labels")
    inherited = np.asarray(ixi["inherited_train"], dtype=np.float64)
    if inherited.shape != (3,):
        raise PaperFigureError("inherited_train must hold [W/8, W/4, W/2]")

    tables = json.loads(inputs.tables_json.read_text(encoding="utf-8"))
    kid = [r for r in _rows(tables, "t3_interaction") if r.get("endpoint") == "kid"]
    if len(kid) != 1 or kid[0].get("relative_mri") is None:
        raise PaperFigureError("table 3 has no single KID row with a relative IXI change")
    nn = {arm: [float(r["seed_nn_fraction"]) for r in _rows(tables, "t1b_cells_mechanism")
                if r.get("dataset") == rules.dataset and r.get("arm") == arm
                and r.get("seed_nn_fraction") is not None] for arm in ("A0", "A3")}
    if not nn["A0"] or not nn["A3"]:
        raise PaperFigureError("table 1b lacks the IXI A0 or A3 seed-NN fraction")
    if level_counts is None:
        level_counts = {name: tuple(levels_per_octave(np.load(inputs.schedule(name))).values())
                        for name in SCHEDULES.values()}
    sources = {
        "octave shares": f"`{_rel(inputs.profile(rules.dataset), inputs)}` and "
                         f"`{_rel(inputs.profile('lsun_church'), inputs)}`, `octave_shares_train`",
        "inherited shares": f"`{_rel(inputs.profile(rules.dataset), inputs)}`, "
                            "`inherited_train` [W/8, W/4, W/2]: entries 0 and 2",
        "KID change": f"`{_rel(inputs.tables_json, inputs)}`, table 3 (`t3_interaction`), "
                      "row `kid`, `relative_mri` (Δ IXI / A0)",
        "seed-NN fractions": f"`{_rel(inputs.tables_json, inputs)}`, table 1b "
                             "(`t1b_cells_mechanism`), IXI A0 and A3 rows, mean of "
                             "`seed_nn_fraction`",
        "level counts": "`levels_per_octave` of `schedules/log_W2.npy` and "
                        "`schedules/ixi_W8.npy`, checked against `docs/RESULTS/data_profile.md` §5",
    }
    return Annotations(
        octave_labels=labels,
        ixi_shares=tuple(float(v) for v in ixi["octave_shares_train"]),
        church_shares=tuple(float(v) for v in church["octave_shares_train"]),
        inherited_default=float(inherited[2]),
        inherited_matched=float(inherited[0]),
        kid_relative=float(kid[0]["relative_mri"]),
        kid_n_seeds=len(kid[0].get("deltas_mri") or []),
        seed_nn_default=float(np.mean(nn["A0"])),
        seed_nn_matched=float(np.mean(nn["A3"])),
        seed_nn_n_seeds=(len(nn["A0"]), len(nn["A3"])),
        level_counts=dict(level_counts),
        sources=sources,
    )


def _rel(path: Path, inputs: F1Inputs) -> str:
    try:
        return str(Path(path).relative_to(inputs.repo_root))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------------------------
# Data selection
# --------------------------------------------------------------------------------------------


def extract_heldout(tar_path: Path, run: str, step: str, work_dir: Path) -> Path:
    """Extract the four held-out members of one evaluation tar, and nothing else.

    The members are read by exact name and written under ``work_dir/<run>/heldout/``; nothing
    in the archive chooses an output path. A folder already holding the four files is reused.

    Parameters
    ----------
    tar_path : Path
        ``<run>_amp-fp16.tar``.
    run : str
        Run id (the archive's top folder).
    step : str
        Checkpoint folder, e.g. ``"060000"``.
    work_dir : Path
        Scratch folder.

    Returns
    -------
    Path
        The folder holding the four files.

    Raises
    ------
    MissingInputError
        If the archive or one of the members is absent.
    """
    out = Path(work_dir) / run / "heldout"
    if all((out / name).is_file() for name in HELDOUT_MEMBERS):
        logger.info("%s: reusing the extracted held-out members in %s", run, out)
        return out
    if not Path(tar_path).is_file():
        raise MissingInputError(f"missing evaluation archive: {tar_path}")
    prefix = f"{run}/samples_amp-fp16/{step}/heldout/"
    wanted = {prefix + name: name for name in HELDOUT_MEMBERS}
    out.mkdir(parents=True, exist_ok=True)
    found: set[str] = set()
    with tarfile.open(tar_path, mode="r:") as archive:
        for member in archive:
            name = wanted.get(member.name)
            if name is None or not member.isfile():
                continue
            source = archive.extractfile(member)
            if source is None:
                continue
            (out / f".{name}.part").write_bytes(source.read())
            (out / f".{name}.part").replace(out / name)
            found.add(name)
            if len(found) == len(wanted):
                break
    absent = sorted(set(HELDOUT_MEMBERS) - found)
    if absent:
        names = ", ".join(prefix + a for a in absent)
        raise MissingInputError(f"{tar_path}: missing member(s) {names}")
    logger.info("%s: extracted %d held-out members to %s", run, len(found), out)
    return out


def _load_teaser(inputs: F1Inputs, rules: SelectionRules, images: np.ndarray
                 ) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    """Assert the teaser rule and return the seed, the two samples and the dataset index."""
    folders = [extract_heldout(inputs.tar(run), run, rules.step, inputs.work_dir)
               for run in rules.runs]
    seed_idx = [np.load(f / "seed_idx.npy", allow_pickle=False) for f in folders]
    if not np.array_equal(seed_idx[0], seed_idx[1]):
        raise PaperFigureError("the two runs' heldout/seed_idx.npy differ: no common seeds")
    requests = [json.loads((f / "request.json").read_text(encoding="utf-8")) for f in folders]
    for run, request in zip(rules.runs, requests, strict=True):
        signature = request.get("signature", {})
        if signature.get("rng_seed") != rules.request_rng_seed:
            raise PaperFigureError(f"{run}: request rng_seed {signature.get('rng_seed')} != "
                                   f"{rules.request_rng_seed}")
        if signature.get("batch_size") != rules.request_batch_size:
            raise PaperFigureError(f"{run}: request batch_size {signature.get('batch_size')} != "
                                   f"{rules.request_batch_size}")
    position = rules.teaser_seed_position
    if position >= len(seed_idx[0]):
        raise PaperFigureError(f"held-out seed position {position} is out of range")
    index = int(seed_idx[0][position])
    if index != rules.teaser_dataset_index:
        raise PaperFigureError(f"held-out seed position {position} holds dataset index {index}, "
                               f"expected {rules.teaser_dataset_index}")
    seed = np.asarray(images[index], dtype=np.float64) / 255.0
    samples = {}
    for key, folder, request in zip(("default", "matched"), folders, requests, strict=True):
        seeds = np.load(folder / "seeds.npy", mmap_mode="r", allow_pickle=False)
        if not np.array_equal(np.asarray(seeds[position]), np.asarray(images[index])):
            raise PaperFigureError(f"{folder}: seeds.npy[{position}] is not dataset image {index}")
        stack = np.load(folder / "samples.npy", mmap_mode="r", allow_pickle=False)
        shape = tuple(request.get("shape", ()))
        if len(shape) != 4:
            raise PaperFigureError(f"{folder}: request.json shape {shape} is not 4-D")
        stack = stack.reshape(shape)
        if rules.teaser_sample >= shape[1]:
            raise PaperFigureError(f"sample {rules.teaser_sample} out of range {shape[1]}")
        sample = np.asarray(stack[position, rules.teaser_sample])
        scale = 255.0 if sample.dtype == np.uint8 else 1.0
        samples[key] = np.asarray(sample, dtype=np.float64) / scale
    return seed, samples, index


def _split_rows(splits: dict[str, Any], name: str) -> np.ndarray:
    rows = splits.get(name)
    if not isinstance(rows, list) or not rows:
        raise PaperFigureError(f"splits.json has no {name!r} split")
    return np.asarray(sorted(rows), dtype=np.int64)


def _select_subjects(index: Any, splits: dict[str, Any], rules: SelectionRules
                     ) -> list[tuple[str, int]]:
    """The first ``n_subjects`` of ``ref_subjects``, at ``subject_slice`` (asserted unique)."""
    names = splits.get("ref_subjects")
    if not isinstance(names, list) or len(names) < rules.n_subjects:
        raise PaperFigureError("splits.json has fewer ref_subjects than the rule needs")
    picked = []
    for subject in names[:rules.n_subjects]:
        rows = index[(index["subject"] == subject) & (index["slice"] == rules.subject_slice)]
        if len(rows) != 1:
            raise PaperFigureError(f"{subject}: {len(rows)} rows at slice {rules.subject_slice}, "
                                   "expected exactly one")
        picked.append((str(subject), int(rows["idx"].iloc[0])))
    return picked


def _mean_train(images: np.ndarray, splits: dict[str, Any], rules: SelectionRules,
                chunk: int = 256) -> tuple[np.ndarray, int]:
    rows = _split_rows(splits, "train")
    if len(rows) != rules.n_train:
        raise PaperFigureError(f"train split holds {len(rows)} images, the rule says "
                               f"{rules.n_train}")
    total = np.zeros(images.shape[1:], dtype=np.float64)
    for start in range(0, len(rows), chunk):
        total += np.asarray(images[rows[start:start + chunk]], dtype=np.float64).sum(axis=0)
    return total / (255.0 * len(rows)), int(len(rows))


def load_f1_data(inputs: F1Inputs, rules: SelectionRules = DEFAULT_RULES,
                 level_counts: dict[str, tuple[int, ...]] | None = None) -> F1Data:
    """Select, assert and compute everything F1 draws.

    Parameters
    ----------
    inputs : F1Inputs
        The input paths.
    rules : SelectionRules
        The pre-declared selection rules.
    level_counts : dict[str, tuple[int, ...]] or None
        Verified level counts (see :func:`verify_level_counts`).

    Returns
    -------
    F1Data
        The figure's data.

    Raises
    ------
    MissingInputError
        If an input is absent.
    PaperFigureError
        If a selection rule fails.
    """
    missing = [p for p in inputs.required(rules) if not p.exists()]
    if missing:
        raise MissingInputError("missing input(s): " + ", ".join(str(p) for p in missing))
    root = inputs.data_root
    images, index, splits, _ = read_dataset(root / rules.dataset)
    width = int(images.shape[-1])
    example, example_index = reference_example(root, rules.dataset,
                                               np.random.default_rng(rules.example_rng_seed))
    example = np.asarray(example, dtype=np.float64)
    heat_states = {s: heat_blur(example, s) for s in HEAT_SIGMAS}
    mean_image, n_train = _mean_train(images, splits, rules)
    subjects = _select_subjects(index, splits, rules)
    subject_images = np.stack([np.asarray(images[i], dtype=np.float64) / 255.0
                               for _, i in subjects])
    priors = {s: list(heat_blur(subject_images, s)) for s in (SIGMA_DEFAULT, SIGMA_MATCHED)}
    seed, samples, teaser_index = _load_teaser(inputs, rules, images)
    schedules = {key: np.asarray(np.load(inputs.schedule(name)), dtype=np.float64)
                 for key, name in SCHEDULES.items()}
    for key, schedule in schedules.items():
        if schedule.ndim != 1 or schedule[0] != 0.0:
            raise PaperFigureError(f"schedule {SCHEDULES[key]}: level 0 must be 0")
    annotations = load_annotations(inputs, rules, level_counts)
    return F1Data(
        width=width, example=example, example_index=int(example_index),
        heat_states=heat_states, mean_image=mean_image, n_train=n_train, subjects=subjects,
        subject_priors=priors, teaser_seed=seed, teaser_samples=samples,
        teaser_index=teaser_index, schedules={k: v[1:] for k, v in schedules.items()},
        annotations=annotations,
    )


# --------------------------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------------------------


@contextmanager
def paper_style() -> Iterator[None]:
    """House style plus the paper overrides (fonts >= 7 pt, editable deterministic SVG)."""
    with figure_style(), plt.rc_context(_PAPER_RC):
        yield


class _Canvas:
    """Places axes and text in inches on a fixed-size figure."""

    def __init__(self, width: float, height: float) -> None:
        self.w, self.h = width, height
        self.fig = plt.figure(figsize=(width, height))

    def axes(self, x: float, y: float, w: float, h: float, **kwargs: Any) -> Axes:
        return self.fig.add_axes((x / self.w, y / self.h, w / self.w, h / self.h), **kwargs)

    def text(self, x: float, y: float, s: str, **kwargs: Any) -> Text:
        return self.fig.text(x / self.w, y / self.h, s, **kwargs)

    def thumb(self, x: float, y: float, size: float, image: np.ndarray,
              edge: str | None = None, lw: float = 1.2) -> Axes:
        ax = self.axes(x, y, size, size)
        ax.imshow(np.clip(image, 0.0, 1.0), cmap="gray", vmin=0.0, vmax=1.0,
                  interpolation="none", aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(edge is not None)
            if edge is not None:
                spine.set_edgecolor(edge)
                spine.set_linewidth(lw)
        return ax


#: Layout constants, in inches.
_A_LEFT, _A_RIGHT = 0.50, 3.38
_C_MIN, _C_MAX = 0.33, 100.0
_THUMB_A = 0.40
_THUMB_B = 0.37
_RIGHT_X0, _RIGHT_X1 = 3.60, 5.47


def _x_of_c(c: float) -> float:
    """Inch position of frequency ``c`` on panel (a)'s log axis."""
    frac = (np.log10(c) - np.log10(_C_MIN)) / (np.log10(_C_MAX) - np.log10(_C_MIN))
    return _A_LEFT + frac * (_A_RIGHT - _A_LEFT)


def _panel_a(cv: _Canvas, data: F1Data) -> None:
    ann = data.annotations
    width = data.width
    s = mode_scale(width)
    blue, orange = arm_style("A0").color, arm_style("A3").color

    # --- heat-state strip -------------------------------------------------------------------
    y_thumb = 2.10
    bars_top = 1.97
    names = {SIGMA_DEFAULT: "default prior", SIGMA_MATCHED: "matched prior"}
    colours = {SIGMA_DEFAULT: blue, SIGMA_MATCHED: orange}
    for sigma in HEAT_SIGMAS:
        c = s / sigma
        x = _x_of_c(c)
        cv.thumb(x - _THUMB_A / 2, y_thumb, _THUMB_A, data.heat_states[sigma],
                 edge=colours.get(sigma), lw=1.2)
        label = f"u({sigma:g} px)"
        if sigma in names:
            cv.text(x, y_thumb + _THUMB_A + 0.13, names[sigma], ha="center", va="bottom",
                    color=colours[sigma], fontsize=7)
        cv.text(x, y_thumb + _THUMB_A + 0.02, label, ha="center", va="bottom", fontsize=7)
        cv.fig.add_artist(_arrow(cv, (x, y_thumb - 0.01), (x, bars_top + 0.005)))
    x_img = _x_of_c(72.0)
    cv.thumb(x_img - _THUMB_A / 2, y_thumb, _THUMB_A, data.example)
    cv.text(x_img, y_thumb + _THUMB_A + 0.02, "image", ha="center", va="bottom", fontsize=7)
    cv.text(0.04, 2.79, "(a)", fontsize=8, fontweight="bold", va="top")
    cv.text(0.04, y_thumb + _THUMB_A / 2, f"IXI\nheat\nstates\nat c ≈\n{s:.1f}/σ",
            fontsize=7, va="center", ha="left", color=MUTED, linespacing=1.05)

    # --- variance share per octave -------------------------------------------------------------
    ax = cv.axes(_A_LEFT, 1.17, _A_RIGHT - _A_LEFT, bars_top - 1.17)
    lows = np.array([float(lab.split("-")[0]) for lab in ann.octave_labels])
    highs = np.array([float(lab.split("-")[1]) for lab in ann.octave_labels])
    ixi = 100.0 * np.asarray(ann.ixi_shares)
    church = 100.0 * np.asarray(ann.church_shares)
    ymax = 37.0
    grid = np.geomspace(_C_MIN, _C_MAX, 600)
    for sigma, colour in ((SIGMA_DEFAULT, blue), (SIGMA_MATCHED, orange)):
        ax.fill_between(grid, 0.0, ymax * kept_power(grid, sigma, width), color=colour,
                        alpha=0.16, linewidth=0.0, zorder=0)
    ax.bar(lows, ixi, width=highs - lows, align="edge", color=BAR_FILL, edgecolor=BAR_EDGE,
           linewidth=0.5, zorder=2, label="IXI (brain MRI)")
    step_x = np.concatenate([lows, highs[-1:]])
    step_y = np.concatenate([church, church[-1:]])
    ax.step(step_x, step_y, where="post", color=REFERENCE_INK, lw=0.8, ls=(0, (3.0, 1.6)),
            zorder=3, label="LSUN Churches")
    ax.plot([lows[0], highs[-1]], [100.0 / len(lows)] * 2, color=REFERENCE_INK, lw=0.8,
            ls=(0, (1.0, 1.4)), zorder=3, label="1/f²: equal share")
    ax.set_xscale("log")
    ax.set_xlim(_C_MIN, _C_MAX)
    ax.set_ylim(0.0, ymax)
    ax.set_yticks([0, 10, 20, 30])
    ax.set_ylabel("variance per\noctave (%)", labelpad=2, linespacing=1.0)
    ax.set_xticks([])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.spines["bottom"].set_visible(False)
    peak, coarse = ann.ixi_peak, ann.coarsest
    ip, ic = ann.octave_labels.index(peak), ann.octave_labels.index(coarse)
    ax.text(np.sqrt(lows[ip] * highs[ip]), ixi[ip] + 0.6, ann.share_text("ixi", peak),
            ha="center", va="bottom", fontsize=7, zorder=4)
    ax.text(np.sqrt(lows[ic] * highs[ic]), ixi[ic] + 0.6, ann.share_text("ixi", coarse),
            ha="center", va="bottom", fontsize=7, zorder=4)
    ax.text(np.sqrt(lows[ic] * highs[ic]), church[ic] - 0.8, ann.share_text("church", coarse),
            ha="center", va="top", fontsize=7, color=REFERENCE_INK, zorder=4)
    ax.text(0.345, 35.5, f"prior keeps: default {ann.inherited_default_text}",
            color=blue, fontsize=7, va="top", ha="left")
    ax.text(0.345, 30.6, f"matched {ann.inherited_matched_text}", color=orange, fontsize=7,
            va="top", ha="left")
    # The legend sits right of the 8-16 c/img peak and runs into the gutter between panels.
    ax.legend(loc="upper right", bbox_to_anchor=(1.045, 1.05), handlelength=1.4,
              handletextpad=0.4, borderaxespad=0.1, labelspacing=0.2, fontsize=7)

    # --- reverse steps per frequency octave (the bars' bins) --------------------------------
    ar = cv.axes(_A_LEFT, 0.60, _A_RIGHT - _A_LEFT, 0.50)
    ar.set_xscale("log")
    ar.set_xlim(_C_MIN, _C_MAX)
    ar.set_ylim(0.0, 2.0)
    rows = {"default": (1.0, blue, SCHEDULES["default"]),
            "matched": (0.0, orange, SCHEDULES["matched"])}
    for key, (base, colour, _) in rows.items():
        cs = s / data.schedules[key]
        ar.vlines(cs, base + 0.08, base + 0.48, color=colour, lw=0.35, zorder=2)
        counts, _ = levels_per_frequency_octave(data.schedules[key], width)
        for lo, hi, n in zip(lows, highs, counts, strict=True):
            if n == 0:
                continue
            ar.text(np.sqrt(lo * hi), base + 0.55, f"{n}", ha="center", va="bottom",
                    fontsize=7, color=colour)
    c_stop = s / float(data.schedules["matched"].max())
    ar.text(np.sqrt(_C_MIN * c_stop), 0.28, "from the prior", ha="center", va="center",
            fontsize=7, color=orange, zorder=3,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.3})
    for edge in highs[:-1]:
        ar.axvline(edge, color=MUTED, lw=0.3, alpha=0.5, zorder=0)
    ar.set_yticks([0.3, 1.3])
    ar.set_yticklabels(["matched", "default"])
    for tick, colour in zip(ar.get_yticklabels(), (orange, blue), strict=True):
        tick.set_color(colour)
    ar.tick_params(axis="y", length=0, pad=2)
    ar.spines["left"].set_visible(False)
    ticks = [0.5, 1, 2, 4, 8, 16, 32, 64]
    ar.set_xticks(ticks)
    ar.set_xticklabels([f"{t:g}" for t in ticks])
    ar.xaxis.set_minor_locator(plt.NullLocator())
    ar.set_xlabel("spatial frequency c (cycles per image)", labelpad=1.5)

    # --- direction arrows ---------------------------------------------------------------------
    y1, y2 = 0.20, 0.07
    cv.fig.add_artist(_arrow(cv, (_A_RIGHT, y1), (_A_LEFT, y1), colour=MUTED))
    cv.text((_A_LEFT + _A_RIGHT) / 2, y1, " forward (training): the heat equation erases fine "
            "bands ", ha="center", va="center", fontsize=7, color=MUTED,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4})
    cv.fig.add_artist(_arrow(cv, (_A_LEFT, y2), (_A_RIGHT, y2), colour=INK))
    cv.text((_A_LEFT + _A_RIGHT) / 2, y2, " generation: each step restores one band ",
            ha="center", va="center", fontsize=7, color=INK,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4})


def _arrow(cv: _Canvas, start: tuple[float, float], end: tuple[float, float],
           colour: str = MUTED) -> Any:
    return FancyArrowPatch((start[0] / cv.w, start[1] / cv.h), (end[0] / cv.w, end[1] / cv.h),
                           transform=cv.fig.transFigure, arrowstyle="-|>", mutation_scale=6,
                           lw=0.7, color=colour, shrinkA=0, shrinkB=0)


def _panel_b(cv: _Canvas, data: F1Data) -> None:
    blue, orange = arm_style("A0").color, arm_style("A3").color
    x0 = _RIGHT_X0
    cv.text(x0 - 0.06, 2.79, "(b)", fontsize=8, fontweight="bold", va="top")
    cv.text(x0 + 0.20, 2.79, "MRI's anatomical prior", fontsize=7, fontweight="bold", va="top")
    gap = 0.03
    xs = [_RIGHT_X1 - (3 - k) * _THUMB_B - (2 - k) * gap for k in range(3)]
    y_top, y_bot = 2.12, 1.62
    cv.text(xs[0], y_top + _THUMB_B + 0.015, f"default prior u({SIGMA_DEFAULT:g} px)",
            fontsize=7, color=blue, va="bottom")
    cv.text(xs[0], y_bot + _THUMB_B + 0.015, f"matched prior u({SIGMA_MATCHED:g} px)",
            fontsize=7, color=orange, va="bottom")
    for k, x in enumerate(xs):
        cv.thumb(x, y_top, _THUMB_B, data.subject_priors[SIGMA_DEFAULT][k], edge=blue, lw=1.0)
        cv.thumb(x, y_bot, _THUMB_B, data.subject_priors[SIGMA_MATCHED][k], edge=orange, lw=1.0)
    size = 0.42
    xm = x0 + 0.02
    ym = 1.90
    cv.thumb(xm, ym, size, data.mean_image)
    cv.text(xm + size / 2, ym - 0.02, f"mean of\n{data.n_train:,} train\nslices",
            fontsize=7, ha="center", va="top", linespacing=1.0)
    cv.text(xm + size / 2, ym + size + 0.02, "shared\nanatomy", fontsize=7, ha="center",
            va="bottom", linespacing=1.0, color=MUTED)


def _panel_c(cv: _Canvas, data: F1Data) -> None:
    ann = data.annotations
    x0, x1 = _RIGHT_X0, _RIGHT_X1
    cv.text(x0 - 0.06, 1.47, "(c)", fontsize=8, fontweight="bold", va="top")
    cv.text(x0 + 0.20, 1.47, "what we test", fontsize=7, fontweight="bold", va="top")
    head_w = 0.32
    cell_w = (x1 - x0 - head_w) / 2
    cell_h = 0.17
    y_cols = 1.21
    for j, name in enumerate(("log spacing", "IXI-matched")):
        cv.text(x0 + head_w + (j + 0.5) * cell_w, y_cols, name, ha="center", va="bottom",
                fontsize=7, color=MUTED)
    grid = (("W/2", ("A0", "A2")), ("W/8", ("A1", "A3")))
    for i, (prior, arms) in enumerate(grid):
        y = y_cols - 0.02 - (i + 1) * (cell_h + 0.02)
        cv.text(x0, y + cell_h / 2, prior, ha="left", va="center", fontsize=7, color=MUTED)
        for j, arm in enumerate(arms):
            colour = arm_style(arm).color
            xc = x0 + head_w + j * cell_w + 0.02
            patch = FancyBboxPatch((xc / cv.w, y / cv.h), (cell_w - 0.04) / cv.w, cell_h / cv.h,
                                   boxstyle="round,pad=0,rounding_size=0.006",
                                   transform=cv.fig.transFigure, facecolor=colour, alpha=0.22,
                                   edgecolor="none")
            cv.fig.add_artist(patch)
            frame = FancyBboxPatch((xc / cv.w, y / cv.h), (cell_w - 0.04) / cv.w, cell_h / cv.h,
                                   boxstyle="round,pad=0,rounding_size=0.006",
                                   transform=cv.fig.transFigure, facecolor="none",
                                   edgecolor=colour, lw=1.0)
            cv.fig.add_artist(frame)
            cv.text(xc + (cell_w - 0.04) / 2, y + cell_h / 2, f"{ARM_LABELS[arm]} ({arm})",
                    ha="center", va="center", fontsize=7)
    # Teaser: one unseen subject, same seed image and same sampling noise in both arms.
    size = _THUMB_B
    y_t = 0.30
    xs = (x0 + 0.02, x0 + 0.62, x0 + 1.06)
    tiles = ((data.teaser_seed, "unseen seed", MUTED),
             (data.teaser_samples["default"], ARM_LABELS["A0"], arm_style("A0").color),
             (data.teaser_samples["matched"], ARM_LABELS["A3"], arm_style("A3").color))
    for x, (image, label, colour) in zip(xs, tiles, strict=True):
        cv.thumb(x, y_t, size, image, edge=colour, lw=1.0)
        cv.text(x + size / 2, y_t + size + 0.015, label, ha="center", va="bottom", fontsize=7,
                color=colour)
    cv.fig.add_artist(_arrow(cv, (xs[0] + size + 0.02, y_t + size / 2),
                             (xs[1] - 0.03, y_t + size / 2)))
    nn0, nn3 = ann.seed_nn_texts
    cv.text(xs[0], y_t - 0.03, f"KID {ann.kid_text} (IXI, {ann.kid_n_seeds} seeds)",
            fontsize=7, va="top", ha="left")
    cv.text(xs[0], y_t - 0.15, f"seed copying {nn0} → {nn3}", fontsize=7, va="top", ha="left")


def draw_f1(data: F1Data) -> Figure:
    """Draw F1 (5.5 in wide) from verified data.

    Parameters
    ----------
    data : F1Data
        Output of :func:`load_f1_data`.

    Returns
    -------
    Figure
        The figure; save it with :func:`save_f1` inside :func:`paper_style`.
    """
    with paper_style():
        cv = _Canvas(PAPER_WIDTH_IN, F1_HEIGHT_IN)
        _panel_a(cv, data)
        _panel_b(cv, data)
        _panel_c(cv, data)
    return cv.fig


def sha256_of(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_f1(fig: Figure, out_dir: Path, name: str = F1_NAME) -> list[Path]:
    """Write ``<name>.pdf``, ``.svg`` and ``.png`` byte-stably, then close the figure.

    The PDF and SVG embed every thumbnail as its raw 192² array (``interpolation="none"``,
    i.e. about 500 dpi at 0.37–0.42 in); the PNG is rendered at 300 dpi with antialiased
    resampling. No dates are written; SVG ids are salted by ``svg.hashsalt``.

    Parameters
    ----------
    fig : Figure
        Output of :func:`draw_f1`.
    out_dir : Path
        Output folder, created if absent.
    name : str
        File stem.

    Returns
    -------
    list[Path]
        PDF, SVG and PNG paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = [im for ax in fig.axes for im in ax.get_images()]
    paths = [out_dir / f"{name}.{ext}" for ext in ("pdf", "svg", "png")]
    with paper_style():
        for im in images:
            im.set_interpolation("none")
        fig.savefig(paths[0], format="pdf", dpi=PDF_DPI,
                    metadata={"CreationDate": None, "ModDate": None,
                              "Creator": "ihdm.cli.paper_f1", "Producer": "matplotlib"})
        fig.savefig(paths[1], format="svg", metadata={"Date": None, "Creator": "ihdm.cli.paper_f1"})
        for im in images:
            im.set_interpolation("antialiased")
        fig.savefig(paths[2], format="png", dpi=PNG_DPI, metadata={"Software": None})
    plt.close(fig)
    return paths


@dataclass(frozen=True)
class LayoutReport:
    """Automatic legibility checks of a drawn figure (an aid to, not a substitute for, viewing).

    Attributes
    ----------
    size_in : tuple[float, float]
        Figure width and height.
    min_font_pt : float
        Smallest font of any visible, non-empty text.
    min_thumb_in : float
        Smallest image axes side.
    overlaps : list[tuple[str, str]]
        Pairs of visible texts whose boxes intersect.
    outside : list[str]
        Texts that leave the figure.
    """

    size_in: tuple[float, float]
    min_font_pt: float
    min_thumb_in: float
    overlaps: list[tuple[str, str]]
    outside: list[str]


def layout_report(fig: Figure) -> LayoutReport:
    """Measure fonts, thumbnails, text overlaps and clipping of a drawn figure.

    Parameters
    ----------
    fig : Figure
        A figure from :func:`draw_f1` (before :func:`save_f1` closes it).

    Returns
    -------
    LayoutReport
        The measurements.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()
             and t.get_figure() is not None]
    min_font = min(float(t.get_fontsize()) for t in texts)
    boxes = [(t.get_text().replace("\n", " "), t.get_window_extent(renderer)) for t in texts]
    overlaps = [(a, b) for (a, ba), (b, bb) in itertools.combinations(boxes, 2)
                if ba.overlaps(bb) and ba.intersection(ba, bb) is not None
                and ba.intersection(ba, bb).width > 0.5 and ba.intersection(ba, bb).height > 0.5]
    fw, fh = fig.bbox.width, fig.bbox.height
    outside = [s for s, b in boxes if b.x0 < -0.5 or b.y0 < -0.5 or b.x1 > fw + 0.5
               or b.y1 > fh + 0.5]
    thumbs = [ax for ax in fig.axes if ax.get_images()]
    dpi = fig.dpi
    min_thumb = min(min(ax.bbox.width, ax.bbox.height) / dpi for ax in thumbs) if thumbs else 0.0
    w, h = fig.get_size_inches()
    return LayoutReport((float(w), float(h)), min_font, float(min_thumb), overlaps, outside)


# --------------------------------------------------------------------------------------------
# Verification 3 and the F1.md record
# --------------------------------------------------------------------------------------------


def _md_row(text: str, section: str, first_cell: str) -> list[str]:
    """Cells of the first table row of ``section`` whose first cell equals ``first_cell``."""
    match = re.search(rf"^## {re.escape(section)}.*?$(.*?)^## ", text, flags=re.M | re.S)
    if match is None:
        raise PaperFigureError(f"no section {section} in the markdown source")
    for line in match.group(1).splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0] == first_cell:
            return cells[1:]
    raise PaperFigureError(f"section {section} has no row {first_cell!r}")


def verify_annotations(ann: Annotations, inputs: F1Inputs) -> list[tuple[str, str, str, bool]]:
    """Verification 3: every printed number against the human-readable copy of its source.

    The figure's strings are formatted from the ``.npz``/JSON values; this compares them with
    the tables of ``data_profile.md`` §2–§3, ``tables/t3_interaction.md`` and
    ``tables/t1b_cells_mechanism.md``, which the profiling and table tools wrote from the same
    values.

    Parameters
    ----------
    ann : Annotations
        The numbers.
    inputs : F1Inputs
        Input paths.

    Returns
    -------
    list[tuple[str, str, str, bool]]
        ``(item, printed, reference, equal)`` per number.
    """
    profile = inputs.data_profile_md.read_text(encoding="utf-8")
    shares_ixi = _md_row(profile, "2.", "IXI T1")
    shares_church = _md_row(profile, "2.", "LSUN Churches")
    inherited = _md_row(profile, "3.", "IXI T1")
    t3 = (inputs.repo_root / "docs" / "RESULTS" / "tables" / "t3_interaction.md").read_text(
        encoding="utf-8")
    kid_cell = next((line.split("|")[3].strip() for line in t3.splitlines()
                     if line.startswith("| KID")), None)
    if kid_cell is None:
        raise PaperFigureError("t3_interaction.md has no KID row")
    kid_ref = f"{float(kid_cell.rstrip('%')):+.0f}%".replace("-", "\u2212")
    t1b = (inputs.repo_root / "docs" / "RESULTS" / "tables" / "t1b_cells_mechanism.md"
           ).read_text(encoding="utf-8")
    header = next((line for line in t1b.splitlines() if line.startswith("| dataset")), "")
    column = [c.strip() for c in header.strip().strip("|").split("|")].index("seed-NN frac.")
    nn_ref = {}
    for arm in ("A0", "A3"):
        values = [float(cells[column]) for cells in
                  ([c.strip() for c in line.strip().strip("|").split("|")]
                   for line in t1b.splitlines() if line.startswith(f"| IXI | {arm} |"))]
        if not values:
            raise PaperFigureError(f"t1b_cells_mechanism.md has no IXI {arm} row")
        nn_ref[arm] = f"{100.0 * float(np.mean(values)):.0f}%"
    ip = ann.octave_labels.index(ann.ixi_peak)
    ic = ann.octave_labels.index(ann.coarsest)
    nn0, nn3 = ann.seed_nn_texts
    rows = [
        (f"IXI share {ann.coarsest} c/img", ann.share_text("ixi", ann.coarsest), shares_ixi[ic]),
        (f"IXI share {ann.ixi_peak} c/img", ann.share_text("ixi", ann.ixi_peak), shares_ixi[ip]),
        (f"Churches share {ann.coarsest} c/img", ann.share_text("church", ann.coarsest),
         shares_church[ic]),
        ("inherited share, default prior W/2", ann.inherited_default_text, inherited[2]),
        ("inherited share, matched prior W/8", ann.inherited_matched_text, inherited[0]),
        ("KID Δ IXI / A0 (A3 − A0)", ann.kid_text, kid_ref),
        ("seed-NN fraction, IXI A0 (seed mean)", nn0, nn_ref["A0"]),
        ("seed-NN fraction, IXI A3 (seed mean)", nn3, nn_ref["A3"]),
    ]
    return [(item, printed, ref, printed == ref) for item, printed, ref in rows]


def _words(n: int) -> str:
    """Spell counts below ten, as in running text."""
    names = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
    return names[n] if 0 <= n < len(names) else f"{n:,}"


def caption_text(data: F1Data) -> str:
    """The caption draft of F1, with every number formatted from :class:`Annotations`.

    Parameters
    ----------
    data : F1Data
        The figure's data.

    Returns
    -------
    str
        One paragraph of plain text (markdown emphasis on the title sentence).
    """
    ann = data.annotations
    s = mode_scale(data.width)
    nn0, nn3 = ann.seed_nn_texts
    folded = sum(levels_per_frequency_octave(v, data.width)[1] for v in data.schedules.values())
    fold_note = " (levels below 0.5 c/img counted in the first octave)" if folded else ""
    return (
        "**IHDM's defaults assume natural-image statistics (an equal share of variance per "
        "octave, and no layout shared across images); brain MRI departs from both.** "
        "(a) Top: noise-free heat states u(σ_B) of one IXI slice, each above "
        f"c = {s:.1f}/σ_B c/img, the DCT mode of length-scale σ_B. Bars: IXI between-image "
        "variance per octave; "
        "dashed: LSUN Churches; dotted: the equal share of a 1/f² spectrum, which log spacing "
        "assumes. Shading: variance kept by the default prior "
        f"(σ_B,max = {SIGMA_DEFAULT:g} px; {ann.inherited_default_text} handed over) and the "
        f"matched prior ({SIGMA_MATCHED:g} px; {ann.inherited_matched_text}). Rugs: the "
        f"{len(data.schedules['default'])} reverse levels of each schedule, counted per "
        f"frequency octave{fold_note}. (b) The mean of {data.n_train:,} training slices, and "
        f"noise-free priors of {_words(len(data.subjects))} held-out subjects. (c) The four "
        "configurations: default (A0), +prior (A1), +spacing (A2) and matched (A3), and one "
        "unseen subject sampled with the same seed and noise. Over "
        f"{_words(ann.kid_n_seeds)} seeds, matched lowers KID by "
        f"{ann.kid_text.lstrip(chr(0x2212))}; on training seeds, samples whose nearest "
        f"training image is their seed rise from {nn0} to {nn3}."
    )


def f1_markdown(data: F1Data, checks: list[MappingCheck],
                annotation_checks: list[tuple[str, str, str, bool]], report: LayoutReport,
                hashes: dict[str, str], stable: dict[str, bool | None], command: str,
                rules: SelectionRules = DEFAULT_RULES) -> str:
    """Render ``docs/RESULTS/paper/F1.md``.

    Parameters
    ----------
    data : F1Data
        The figure's data.
    checks : list[MappingCheck]
        Verification 1.
    annotation_checks : list[tuple[str, str, str, bool]]
        Verification 3.
    report : LayoutReport
        Automatic layout measurements (verification 4 aid).
    hashes : dict[str, str]
        File name -> sha256 of the written outputs.
    stable : dict[str, bool or None]
        File name -> identical to the previous run's bytes (None: no previous file).
    command : str
        The command that wrote the outputs.
    rules : SelectionRules
        The selection rules applied.

    Returns
    -------
    str
        The markdown text.
    """
    ann = data.annotations
    caption = caption_text(data)
    n_words = len(re.sub(r"[*()]", " ", caption).split())
    lines = [
        "# F1 — the visual abstract",
        "",
        "Written by `ihdm.cli.paper_f1` (T8.1); do not edit by hand. Design: "
        "`docs/SPECIFICATIONS/M8-paper/figure-plan-part1.md` §1; ticket: "
        "`docs/SPECIFICATIONS/M8-paper/T8.1-visual-abstract.md`.",
        "",
        "## Files",
        "",
        "| file | role | sha256 | identical to the previous run |",
        "|---|---|---|---|",
    ]
    roles = {"svg": "editable master (Inkscape: text kept as text, thumbnails embedded)",
             "pdf": "the version for LaTeX (vector, thumbnails embedded unresampled)",
             "png": f"preview, {PNG_DPI} dpi"}
    for name, digest in hashes.items():
        prev = stable.get(name)
        verdict = "no previous file" if prev is None else ("yes" if prev else "**no**")
        lines.append(f"| `{name}` | {roles[name.rsplit('.', 1)[1]]} | `{digest[:16]}…` | "
                     f"{verdict} |")
    lines += [
        "",
        f"Size {report.size_in[0]:g} × {report.size_in[1]:g} in; smallest font "
        f"{report.min_font_pt:g} pt; smallest thumbnail {report.min_thumb_in:.2f} in; "
        f"text boxes overlapping: {len(report.overlaps)}; text outside the figure: "
        f"{len(report.outside)}.",
        "",
        f"## Caption draft ({n_words} words)",
        "",
        caption,
        "",
        "## Selected indices (pre-declared rules, asserted in code)",
        "",
        "| element | rule | selected |",
        "|---|---|---|",
        f"| heat-state strip, (a) | `reference_example(root, \"{rules.dataset}\", "
        f"np.random.default_rng({rules.example_rng_seed}))` | dataset index "
        f"{data.example_index} |",
        f"| three subjects, (b) | first {rules.n_subjects} of `splits.json[\"ref_subjects\"]`, "
        f"slice {rules.subject_slice} | "
        + ", ".join(f"{s} (index {i})" for s, i in data.subjects) + " |",
        f"| mean image, (b) | mean of the `train` split ({rules.n_train:,} images) | "
        f"{data.n_train:,} images averaged |",
        f"| teaser, (c) | held-out seed position {rules.teaser_seed_position}, sample "
        f"{rules.teaser_sample}, runs {', '.join(rules.runs)} at {rules.step}; identical "
        f"`heldout/seed_idx.npy`; `request.json` signatures with `rng_seed` "
        f"{rules.request_rng_seed} and `batch_size` {rules.request_batch_size} | dataset index "
        f"{data.teaser_index} (expected {rules.teaser_dataset_index}) |",
        "",
        "The priors are drawn noise-free (heat states of the released `DCTBlur` formula).",
        "",
        "## Annotated numbers and their sources",
        "",
        "| number | printed | reference copy | equal |",
        "|---|---|---|---|",
    ]
    lines += [f"| {item} | {printed} | {ref} | {'yes' if ok else '**no**'} |"
              for item, printed, ref, ok in annotation_checks]
    counts = " · ".join(f"`{name}` " + "/".join(str(c) for c in ann.level_counts[name])
                        for name in SCHEDULES.values())
    freq = {name: levels_per_frequency_octave(data.schedules[key], data.width)
            for key, name in SCHEDULES.items()}
    freq_text = " · ".join(f"`{name}` " + "/".join(str(c) for c in counts_c)
                           + (f" ({below} below 0.5 c/img folded into the first)" if below else "")
                           for name, (counts_c, below) in freq.items())
    lines += ["", f"Level counts per σ_B octave (0.5–1 … 64–96 px; schedule check): {counts}.",
              "",
              f"Level counts per frequency octave, as printed on the rugs (c_k = "
              f"{mode_scale(data.width):.1f}/σ_k binned on the bars' octaves 0.5–1 … 64–96 "
              f"c/img): {freq_text}.", ""]
    lines += [f"- **{key}:** {value}" for key, value in ann.sources.items()]
    lines += [
        "",
        "## Verification",
        "",
        "**1. Frequency mapping** (released `DCTBlur`, float64, on the IXI `train` split; "
        f"tolerance {MAPPING_RTOL:g} relative + {MAPPING_ATOL:g} absolute; half-power "
        f"constant within {HALF_POWER_RTOL:.0%}).",
        "",
        "| σ_B (px) | kept share per octave, measured (0.5–1 … 64–96 c/img) | max rel. error | "
        "25.4/σ | numeric root | pass |",
        "|---|---|---|---|---|---|",
    ]
    for c in checks:
        kept = " / ".join(f"{v:.4g}" for v in c.measured)
        lines.append(f"| {c.sigma:g} | {kept} | {c.max_rel_error:.1e} | {c.half_power:.4f} | "
                     f"{c.half_power_numeric:.4f} | {'yes' if c.passed else '**no**'} |")
    s = mode_scale(data.width)
    lines += [
        "",
        f"c = {s:.1f}/σ is the length-scale of the DCT mode (σ_n = √(2/λ)); the prior keeps "
        f"half a mode's variance up to c½ = {half_power_cycles(1.0, data.width):.1f}/σ.",
        "",
        "**2. Level counts** per σ_B octave equal `data_profile.md` §5 and the ticket: "
        + counts + ". The rugs print the counts per frequency octave (above), which sum to "
        + " and ".join(str(sum(c)) for c, _ in freq.values()) + ".",
        "",
        "**3. Annotated numbers** equal their sources: "
        + ("all" if all(ok for *_, ok in annotation_checks) else "**not all**")
        + " (table above).",
        "",
        "**4. Visual check:** done on the PNG after each layout iteration; the iterations are "
        "listed in `docs/AGENT-LOGS/M8-paper/T8.1-visual-abstract.md`.",
        "",
        "**5. Byte stability:** see the last column of the file table (a second run of the "
        "command must report \"yes\" for all three files).",
        "",
        "## Command",
        "",
        "```bash",
        command,
        "```",
        "",
    ]
    return "\n".join(lines)
