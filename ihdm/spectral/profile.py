"""Spectral profile of the four datasets: tables, known-results checklist and the data figure.

The profile is computed on the **training** split of every dataset (3200 images), as
``docs/SPECIFICATIONS/M1-data/T1.3-profile-and-schedules.md`` requires; the ``ref`` split is
profiled as a check that the two halves of the same cohort agree.

The figure reproduces the layout, colours and rcParams of
``projects/GenAI/analysis/proposal_data_figure.py`` (IEEE single-column rcParams and the Paul
Tol bright palette, themselves ported there from ``IsalHG/src/isalhg/viz/style.py``), with the
examples taken from the registered ``ref`` splits and the curves from the training splits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ihdm.data.format import read_dataset
from ihdm.preprocess.mri import site_from_source
from ihdm.spectral.errors import SpectralError
from ihdm.spectral.power import (
    fit_alpha,
    inherited_share,
    mode_power,
    octave_shares,
    radial_spectrum,
)

__all__ = [
    "DATASETS",
    "DATASET_LABELS",
    "MRI_DATASETS",
    "PHOTO_DATASETS",
    "SENSITIVITY_DIR",
    "SENSITIVITY_IDS",
    "SITE_ORDER",
    "ARCHIVED",
    "DatasetProfile",
    "ChecklistItem",
    "SiteProfile",
    "coarse_bin_breakdown",
    "split_power",
    "profile_split",
    "reference_example",
    "build_checklist",
    "make_figure",
    "sensitivity_root",
    "site_profiles",
]

logger = logging.getLogger(__name__)

DATASETS: tuple[str, ...] = ("ixi", "oasis1", "lsun_church", "lsun_bedroom")
MRI_DATASETS: tuple[str, ...] = ("ixi", "oasis1")
PHOTO_DATASETS: tuple[str, ...] = ("lsun_church", "lsun_bedroom")

#: Where the uncorrected (pre-N4) MRI datasets are archived, and under which ids.
SENSITIVITY_DIR: str = "_sensitivity"
SENSITIVITY_IDS: dict[str, str] = {"ixi": "ixi_no_n4", "oasis1": "oasis1_no_n4"}

#: Acquisition sites, in the order the per-site table prints them.
SITE_ORDER: tuple[str, ...] = ("Guys", "HH", "IOP", "WashU")

DATASET_LABELS: dict[str, str] = {
    "ixi": "IXI T1",
    "oasis1": "OASIS-1 T1",
    "lsun_church": "LSUN Churches",
    "lsun_bedroom": "LSUN Bedrooms",
    "ixi_no_n4": "IXI T1 (no N4)",
    "oasis1_no_n4": "OASIS-1 T1 (no N4)",
}
IMAGE_SIZE: int = 192

#: The coarsest octave bin is exactly the three DCT modes (0,1), (1,0) and (1,1).
COARSE_BIN: str = "0.5-1"

# The alpha window the ticket's prose names (1 - 48 cycles per image), reported beside the
# ported window of control_profile.fit_alpha; see the log, decision D-T1.3-1.
TICKET_ALPHA_LO_FRAC: float = 2.0 / IMAGE_SIZE
TICKET_ALPHA_HI_FRAC: float = 96.0 / IMAGE_SIZE

# The archived, *unregistered* numbers this ticket is compared against:
# worklog/sessions/2026-09-21_ihdm-knob-proposals/native192_profile_output_unprocessed.md
# (480 head-centred crops per dataset, no registration; last octave closed at 95.5 c/img).
ARCHIVED: dict[str, dict[str, Any]] = {
    "ixi": {
        "alpha": 3.47, "spread_log": 50.8, "spread_ixi": 1.0, "dev_max": 0.00, "dev_median": 0.00,
        "inherited": (0.055, 0.010, 0.001), "coarse_share": 0.008,
        "octaves": (0.008, 0.098, 0.074, 0.148, 0.361, 0.223, 0.080, 0.007),
    },
    "oasis1": {
        "alpha": 3.16, "spread_log": 93.5, "spread_ixi": 3.3, "dev_max": 0.30, "dev_median": 0.08,
        "inherited": (0.028, 0.005, 0.000), "coarse_share": 0.004,
        "octaves": (0.004, 0.051, 0.061, 0.197, 0.335, 0.237, 0.089, 0.025),
    },
    "lsun_church": {
        "alpha": 2.27, "spread_log": 3.9, "spread_ixi": 90.6, "dev_max": 1.87, "dev_median": 0.90,
        "inherited": (0.301, 0.139, 0.019), "coarse_share": 0.251,
        "octaves": (0.251, 0.210, 0.154, 0.119, 0.095, 0.075, 0.062, 0.035),
    },
    "lsun_bedroom": {
        "alpha": 2.62, "spread_log": 8.9, "spread_ixi": 92.8, "dev_max": 2.07, "dev_median": 1.16,
        "inherited": (0.292, 0.124, 0.016), "coarse_share": 0.226,
        "octaves": (0.226, 0.248, 0.202, 0.131, 0.086, 0.056, 0.036, 0.015),
    },
}

# Paul Tol bright palette and the IEEE single-column rcParams of proposal_data_figure.py.
IEEE_RCPARAMS: dict[str, Any] = {
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "font.family": "serif", "font.size": 9.0, "axes.titlesize": 9.0, "axes.labelsize": 8.0,
    "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 7.0, "ytick.labelsize": 7.0, "legend.fontsize": 7.0, "lines.linewidth": 1.0,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}
TOL: dict[str, str] = {
    "blue": "#4477AA", "red": "#EE6677", "green": "#228833", "yellow": "#CCBB44",
    "cyan": "#66CCEE", "purple": "#AA3377", "grey": "#BBBBBB", "orange": "#EE7733",
}
DATASET_COLOURS: dict[str, str] = {
    "ixi": TOL["blue"], "oasis1": TOL["cyan"],
    "lsun_church": TOL["red"], "lsun_bedroom": TOL["orange"],
}
GROUP_COLOURS: dict[str, str] = {"Brain MRI": TOL["blue"], "Photographs": TOL["red"]}
FIGURE_SIZE: tuple[float, float] = (2.7, 4.35)
FIGURE_RNG_SEED: int = 2026
MRI_EXAMPLE_SLICE: int = 5  # the report's Fig. 1 plane (D1', atlas voxel z = 96)


@dataclass(frozen=True)
class DatasetProfile:
    """Every scalar the report quotes for one split of one dataset.

    Parameters
    ----------
    dataset_id, split : str
        Which split of which dataset.
    n_images : int
        Number of images in the split.
    images_sha256 : str
        ``meta.json``'s hash of the whole ``images.npy``, for provenance.
    alpha : float
        Spectral exponent over the analysis scripts' window (10.0 - 67.0 cycles per image).
    alpha_ticket_window : float
        The same fit over the window the ticket's prose names (1 - 48 cycles per image).
    shares : dict[str, float]
        Octave shares of the between-image variance.
    inherited : dict[str, float]
        Inherited share at ``sigma_B,max = W/8``, ``W/4``, ``W/2``.
    total_variance : float
        Sum of the per-mode variance over the non-DC modes.
    """

    dataset_id: str
    split: str
    n_images: int
    images_sha256: str
    alpha: float
    alpha_ticket_window: float
    shares: dict[str, float] = field(repr=False)
    inherited: dict[str, float] = field(repr=False)
    total_variance: float = 0.0


@dataclass(frozen=True)
class ChecklistItem:
    """One item of the known-results checklist of ``docs/HARNESSES/data.md`` §4.

    Parameters
    ----------
    key : str
        ``i`` … ``v``.
    statement : str
        What the item asserts.
    passed : bool
        Whether the measured numbers satisfy it.
    detail : str
        The numbers, always reported, whether the item passes or not.
    """

    key: str
    statement: str
    passed: bool
    detail: str

    @property
    def verdict(self) -> str:
        """``"PASS"`` or ``"FAIL"``."""
        return "PASS" if self.passed else "FAIL"


def _split_rows(index: pd.DataFrame, splits: dict[str, Any], split: str) -> np.ndarray:
    """Return the sorted image indices of a split.

    Parameters
    ----------
    index : pandas.DataFrame
        The dataset's ``index.csv``.
    splits : dict[str, Any]
        The dataset's ``splits.json``.
    split : str
        ``"train"``, ``"ref"`` or ``"seed"``.

    Returns
    -------
    numpy.ndarray
        Sorted ``int`` indices into ``images.npy``.

    Raises
    ------
    SpectralError
        If the split is unknown or empty.
    """
    if split not in splits or not isinstance(splits[split], list) or not splits[split]:
        raise SpectralError(f"split {split!r} is missing or empty")
    rows = np.asarray(sorted(splits[split]), dtype=np.int64)
    if rows.max() >= len(index):
        raise SpectralError(f"split {split!r} indexes past the end of the index")
    return rows


def split_power(root: Path, dataset_id: str, split: str) -> tuple[np.ndarray, int, str]:
    """Per-mode variance of one split of one dataset.

    Parameters
    ----------
    root : Path
        The data root holding ``<dataset_id>/``.
    dataset_id : str
        One of :data:`DATASETS`.
    split : str
        ``"train"`` or ``"ref"``.

    Returns
    -------
    tuple[numpy.ndarray, int, str]
        ``(power, n_images, images_sha256)``.
    """
    images, index, splits, meta = read_dataset(Path(root) / dataset_id)
    rows = _split_rows(index, splits, split)
    stack = np.asarray(images[rows], dtype=np.float32) / 255.0
    logger.info("%s/%s: %d images of %d^2", dataset_id, split, len(rows), stack.shape[1])
    return mode_power(stack), int(len(rows)), meta.sha256_images


def profile_split(
    power: np.ndarray, dataset_id: str, split: str, n_images: int, images_sha256: str
) -> DatasetProfile:
    """Assemble the scalar profile of one split from its per-mode variance.

    Parameters
    ----------
    power : numpy.ndarray
        Per-mode variance of shape ``(W, W)``.
    dataset_id, split : str
        Provenance of the spectrum.
    n_images : int
        Number of images it was computed from.
    images_sha256 : str
        The dataset's image hash.

    Returns
    -------
    DatasetProfile
        The profile.
    """
    width = power.shape[0]
    return DatasetProfile(
        dataset_id=dataset_id,
        split=split,
        n_images=n_images,
        images_sha256=images_sha256,
        alpha=fit_alpha(power),
        alpha_ticket_window=fit_alpha(power, TICKET_ALPHA_LO_FRAC, TICKET_ALPHA_HI_FRAC),
        shares=octave_shares(power),
        inherited={
            f"W/{d}": inherited_share(power, width / d) for d in (8, 4, 2)
        },
        total_variance=float(power.sum()),
    )


def reference_example(
    root: Path, dataset_id: str, rng: np.random.Generator
) -> tuple[np.ndarray, int]:
    """One example image from the ``ref`` split, for the figure.

    For MRI the candidates are restricted to the report's Fig. 1 plane (``slice == 5``); for
    photographs every reference image is a candidate.

    Parameters
    ----------
    root : Path
        The data root.
    dataset_id : str
        One of :data:`DATASETS`.
    rng : numpy.random.Generator
        Drawn from once.

    Returns
    -------
    tuple[numpy.ndarray, int]
        The image as ``float32`` in ``[0, 1]``, and its row index.
    """
    images, index, splits, _ = read_dataset(Path(root) / dataset_id)
    rows = _split_rows(index, splits, "ref")
    on_plane = index.loc[rows, "slice"].to_numpy() == MRI_EXAMPLE_SLICE
    if on_plane.any():
        rows = rows[on_plane]
    pick = int(rng.choice(rows))
    return np.asarray(images[pick], dtype=np.float32) / 255.0, pick


@dataclass(frozen=True)
class SiteProfile:
    """The coarse-bin numbers of one acquisition site of one dataset.

    Parameters
    ----------
    dataset_id, site : str
        Which site of which dataset.
    n_subjects, n_images : int
        Size of the site inside the split.
    coarse_share : float
        Share of the split's between-image variance in the 0.5-1 cycles-per-image bin,
        measured on that site's images alone.
    mode_10_share : float
        Share of that bin carried by the anterior-posterior ramp mode ``(1, 0)``.
    """

    dataset_id: str
    site: str
    n_subjects: int
    n_images: int
    coarse_share: float
    mode_10_share: float


def coarse_bin_breakdown(power: np.ndarray) -> tuple[float, float]:
    """Return the coarse-bin share and the ``(1, 0)`` mode's share inside that bin.

    The 0.5-1 cycles-per-image bin is exactly the DCT modes ``(0, 1)`` (a left-right ramp),
    ``(1, 0)`` (an anterior-posterior ramp on the MRI sets) and ``(1, 1)`` (a diagonal one),
    so "how much of the coarse variance is one gradient" is a two-number statement.

    Parameters
    ----------
    power : numpy.ndarray
        Per-mode variance of shape ``(W, W)``.

    Returns
    -------
    tuple[float, float]
        ``(coarse share of the total, (1,0) share of the coarse bin)``.
    """
    values = np.array([power[0, 1], power[1, 0], power[1, 1]], dtype=float)
    return float(octave_shares(power)[COARSE_BIN]), float(values[1] / values.sum())


def sensitivity_root(root: Path) -> Path:
    """Return the directory holding the archived uncorrected datasets.

    Parameters
    ----------
    root : Path
        The data root.

    Returns
    -------
    Path
        ``<root>/_sensitivity``.
    """
    return Path(root) / SENSITIVITY_DIR


def _cohort_of(dataset_id: str) -> str:
    """Return the raw cohort a dataset id belongs to (``ixi_no_n4`` -> ``ixi``)."""
    return dataset_id.removesuffix("_no_n4")


def site_profiles(root: Path, dataset_id: str, split: str = "train") -> list[SiteProfile]:
    """Measure the coarse-bin numbers separately for each acquisition site.

    The site is read from ``index.csv``'s ``source`` column rather than from
    ``meta.json.parameters.sites``, so the same code path serves the corrected datasets and
    the archived uncorrected ones, which were written by pipeline 1.0 and carry no site map.

    Parameters
    ----------
    root : Path
        The directory holding ``<dataset_id>/``.
    dataset_id : str
        An MRI dataset id, corrected (``ixi``) or archived (``ixi_no_n4``).
    split : str
        Which split to measure; the profile uses ``"train"``.

    Returns
    -------
    list[SiteProfile]
        One entry per site present, ordered by :data:`SITE_ORDER`.

    Raises
    ------
    SpectralError
        If the split is missing or empty.
    """
    images, index, splits, _ = read_dataset(Path(root) / dataset_id)
    rows = _split_rows(index, splits, split)
    cohort = _cohort_of(dataset_id)
    sites = index.loc[rows, "source"].map(lambda s: site_from_source(cohort, str(s)))
    out: list[SiteProfile] = []
    for site in SITE_ORDER:
        selected = rows[(sites == site).to_numpy()]
        if selected.size == 0:
            continue
        stack = np.asarray(images[selected], dtype=np.float32) / 255.0
        coarse, mode_10 = coarse_bin_breakdown(mode_power(stack))
        out.append(
            SiteProfile(
                dataset_id=dataset_id,
                site=site,
                n_subjects=int(index.loc[selected, "subject"].nunique()),
                n_images=int(selected.size),
                coarse_share=coarse,
                mode_10_share=mode_10,
            )
        )
    logger.info("%s/%s: %d site(s)", dataset_id, split, len(out))
    return out


def _mri_vs_photo(values: dict[str, float]) -> tuple[float, float, float, float]:
    """Return ``(min_mri, max_mri, min_photo, max_photo)`` of a per-dataset quantity."""
    mri = [values[d] for d in MRI_DATASETS]
    photo = [values[d] for d in PHOTO_DATASETS]
    return min(mri), max(mri), min(photo), max(photo)


def build_checklist(
    alphas: dict[str, float],
    coarse_shares: dict[str, float],
    spread_log: dict[str, float],
    spread_ixi: dict[str, float],
    deviation_max: dict[str, float],
) -> list[ChecklistItem]:
    """The five known-results items of the ticket (§4 i–v) and ``docs/HARNESSES/data.md`` §4.

    Parameters
    ----------
    alphas : dict[str, float]
        Spectral exponent per dataset, training split.
    coarse_shares : dict[str, float]
        Share of the between-image variance in the 0.5–1 cycles-per-image octave.
    spread_log : dict[str, float]
        Per-level target spread under ``log_W2``.
    spread_ixi : dict[str, float]
        Per-level target spread under ``ixi_W2``.
    deviation_max : dict[str, float]
        ``max_k |s_own(k) / s_ixi(k) - 1|`` of each dataset's own ``W/2`` matched schedule.

    Returns
    -------
    list[ChecklistItem]
        Five items, each with its verdict and its numbers.
    """
    lo_mri_a, _, _, hi_photo_a = _mri_vs_photo(alphas)
    _, hi_mri_c, lo_photo_c, _ = _mri_vs_photo(coarse_shares)
    lo_mri_s, _, _, hi_photo_s = _mri_vs_photo(spread_log)
    gap = lo_mri_a - hi_photo_a
    return [
        ChecklistItem(
            "i", "MRI alpha exceeds photograph alpha by at least 0.5",
            gap >= 0.5,
            f"min MRI alpha {lo_mri_a:.2f} - max photograph alpha {hi_photo_a:.2f} "
            f"= {gap:+.2f}; per dataset "
            + ", ".join(f"{DATASET_LABELS[d]} {alphas[d]:.2f}" for d in DATASETS),
        ),
        ChecklistItem(
            "ii", "the 0.5-1 c/img share is below 3% on both MRI sets and above 15% on both "
            "photograph sets",
            hi_mri_c < 0.03 and lo_photo_c > 0.15,
            ", ".join(f"{DATASET_LABELS[d]} {coarse_shares[d]:.2%}" for d in DATASETS),
        ),
        ChecklistItem(
            "iii", "the log-schedule spread is larger on MRI than on photographs",
            lo_mri_s > hi_photo_s,
            f"min MRI {lo_mri_s:.1f}x vs max photograph {hi_photo_s:.1f}x; per dataset "
            + ", ".join(f"{DATASET_LABELS[d]} {spread_log[d]:.1f}x" for d in DATASETS),
        ),
        ChecklistItem(
            "iv", "the IXI schedule reduces OASIS-1's spread and increases Churches'",
            spread_ixi["oasis1"] < spread_log["oasis1"]
            and spread_ixi["lsun_church"] > spread_log["lsun_church"],
            "OASIS-1 {:.1f}x -> {:.1f}x, Churches {:.1f}x -> {:.1f}x (log -> IXI-matched)".format(
                spread_log["oasis1"], spread_ixi["oasis1"],
                spread_log["lsun_church"], spread_ixi["lsun_church"],
            ),
        ),
        ChecklistItem(
            "v", "the two brain schedules agree within 35% (max) and the photograph schedule "
            "differs from IXI's by more than 50%",
            deviation_max["oasis1"] <= 0.35 and deviation_max["lsun_church"] > 0.50,
            "max abs(s_own / s_ixi - 1): "
            + ", ".join(
                f"{DATASET_LABELS[d]} {deviation_max[d]:.0%}" for d in DATASETS if d != "ixi"
            ),
        ),
    ]


def _example_axes(fig: Any, grid: Any, examples: dict[str, np.ndarray]) -> dict[str, list[Any]]:
    """Draw the four example panels and return the axes of each group row."""
    rows = [("Brain MRI", MRI_DATASETS), ("Photographs", PHOTO_DATASETS)]
    row_axes: dict[str, list[Any]] = {}
    for r, (group, names) in enumerate(rows):
        row_axes[group] = []
        for c, name in enumerate(names):
            ax = fig.add_subplot(grid[r, c])
            ax.imshow(examples[name], cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("0.25")
                spine.set_linewidth(0.5)
            ax.set_title(DATASET_LABELS[name], fontsize=7, pad=4, color="0.15")
            row_axes[group].append(ax)
    return row_axes


def _group_box(fig: Any, axes: list[Any], colour: str, label: str) -> None:
    """Rounded box around a row of example axes, with the rotated group label.

    Ported from ``proposal_data_figure._group_box``.
    """
    from matplotlib.patches import FancyBboxPatch

    fig.canvas.draw()
    boxes = [
        ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        for ax in axes
    ]
    x0, y0 = min(b.x0 for b in boxes), min(b.y0 for b in boxes)
    x1, y1 = max(b.x1 for b in boxes), max(b.y1 for b in boxes)
    mx, my = 0.035, 0.012
    x0 -= mx + 0.045  # room for the rotated group label
    patch = FancyBboxPatch(
        (x0, y0 - my), x1 - x0 + mx, y1 - y0 + 2 * my, transform=fig.transFigure,
        boxstyle="round,pad=0,rounding_size=0.025", facecolor=colour, alpha=0.10,
        edgecolor=colour, linewidth=1.0, zorder=0, clip_on=False,
    )
    fig.patches.append(patch)
    fig.text(
        x0 + 0.03, (y0 + y1) / 2, label, rotation=90, ha="center", va="center",
        fontsize=7.5, fontweight="bold", color=colour,
    )


def _spectrum_axis(
    fig: Any, cell: Any, powers: dict[str, np.ndarray], alphas: dict[str, float]
) -> None:
    """Draw the log-log radial share-of-variance panel with the four curves."""
    ax = fig.add_subplot(cell)
    for name in DATASETS:
        freq, var = radial_spectrum(powers[name])
        ax.loglog(
            freq, var / var.sum(), color=DATASET_COLOURS[name], lw=1.1,
            label=f"{DATASET_LABELS[name]} ($\\alpha$={alphas[name]:.1f})",
        )
    f_ref = np.array([1.0, 60.0])
    ax.loglog(f_ref, 3e-2 * f_ref**-2.0, color="0.4", lw=0.8, ls=":", label="$1/f^{2}$")
    ax.set_xlabel("radial frequency [cycles / image]", fontsize=7)
    ax.set_ylabel("share of variance", fontsize=7)
    ax.set_xlim(0.45, 110)
    ax.set_ylim(1e-6, 1.0)
    ax.set_yticks([1e0, 1e-2, 1e-4, 1e-6])
    ax.tick_params(labelsize=6)
    ax.grid(True, which="major", lw=0.3, alpha=0.5)
    ax.legend(
        frameon=False, loc="lower left", handlelength=1.5, labelcolor="0.15",
        fontsize=5.8, borderaxespad=0.2,
    )
    ax.text(
        0.98, 0.95, "MRI", color=GROUP_COLOURS["Brain MRI"], fontsize=7, fontweight="bold",
        ha="right", va="top", transform=ax.transAxes,
    )
    ax.text(
        0.98, 0.84, "photographs", color=GROUP_COLOURS["Photographs"], fontsize=7,
        fontweight="bold", ha="right", va="top", transform=ax.transAxes,
    )


def make_figure(
    examples: dict[str, np.ndarray],
    powers: dict[str, np.ndarray],
    alphas: dict[str, float],
    out_stem: Path,
) -> tuple[Path, Path]:
    """Render the report's data figure to ``<out_stem>.pdf`` and ``<out_stem>.png``.

    Top: a 2x2 grid of examples (row 1 brain MRI, row 2 photographs) framed in its group
    colour. Bottom: the radially averaged between-image variance per mode of the four training
    splits, as a share of the total, log-log, with the ``1/f^2`` scale-invariant line and the
    fitted exponent in the legend. Layout, colours and rcParams from
    ``proposal_data_figure.make_figure``; the figure is printed at about 2.1 in wide in a
    ``wrapfigure``, hence the small fonts.

    Parameters
    ----------
    examples : dict[str, numpy.ndarray]
        One ``(W, W)`` image per dataset, from the ``ref`` split.
    powers : dict[str, numpy.ndarray]
        Per-mode variance per dataset, from the training split.
    alphas : dict[str, float]
        Fitted exponent per dataset, for the legend.
    out_stem : Path
        Output path without extension.

    Returns
    -------
    tuple[Path, Path]
        The PDF and PNG paths.

    Raises
    ------
    SpectralError
        If a dataset is missing from ``examples`` or ``powers``.
    """
    missing = [d for d in DATASETS if d not in examples or d not in powers]
    if missing:
        raise SpectralError(f"the figure needs every dataset; missing {missing}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(IEEE_RCPARAMS)
    fig = plt.figure(figsize=FIGURE_SIZE)
    gs = fig.add_gridspec(
        2, 1, height_ratios=[1.0, 0.78], hspace=0.10, left=0.19, right=0.98, top=0.97, bottom=0.10
    )
    top = gs[0].subgridspec(2, 2, wspace=0.18, hspace=0.42)
    row_axes = _example_axes(fig, top, examples)
    for group, axes in row_axes.items():
        _group_box(fig, axes, GROUP_COLOURS[group], group)
    _spectrum_axis(fig, gs[1], powers, alphas)

    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf, png = out_stem.with_suffix(".pdf"), out_stem.with_suffix(".png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    logger.info("wrote %s and %s", pdf, png)
    return pdf, png
