"""Plotting helpers for the Figure 2 panels re-drawn in ``ihdm_mri_tutorial.ipynb`` (T9.1).

The helpers only draw what ``assets/results/f2_results.json`` holds (built by
``ihdm.cli.build_tutorial_assets``); they compute nothing beyond seed means. The computations the
notebook explains (heat blur, per-mode variance, octave shares, inherited share, per-level
target) stay in the notebook's own cells.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.axes import Axes

# Configurations in the order of the report, with Paul Tol's "bright" colours.
CONFIGURATION_ORDER: tuple[str, ...] = ("default", "+spacing", "+prior", "matched")
CONFIGURATION_COLORS: dict[str, str] = {
    "default": "#4477AA",
    "+spacing": "#CCBB44",
    "+prior": "#66CCEE",
    "matched": "#EE6677",
}
SET_OFFSET: dict[str, float] = {"F": -0.12, "H": 0.12}


def select_runs(
    results: dict[str, Any], dataset: str, configuration: str
) -> list[dict[str, Any]]:
    """The run entries of one dataset and configuration, in run-id order.

    Parameters
    ----------
    results : dict[str, Any]
        Content of ``f2_results.json``.
    dataset : str
        ``"ixi"`` or ``"oasis1"``.
    configuration : str
        One of :data:`CONFIGURATION_ORDER`.

    Returns
    -------
    list[dict[str, Any]]
        Possibly empty (OASIS-1 has no +prior or +spacing runs).
    """
    return [
        results["runs"][run]
        for run in sorted(results["runs"])
        if results["runs"][run]["dataset"] == dataset
        and results["runs"][run]["configuration"] == configuration
    ]


def _faces(set_name: str, color: str) -> str:
    """Filled markers for set F, hollow for set H."""
    return color if set_name == "F" else "none"


def plot_precision_recall(ax: Axes, results: dict[str, Any], *, dataset: str = "ixi") -> None:
    """Precision against recall (against R⁻) per run and per configuration mean.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    results : dict[str, Any]
        Content of ``f2_results.json``.
    dataset : str, default "ixi"
        Which cohort to draw.
    """
    for configuration in CONFIGURATION_ORDER:
        runs = select_runs(results, dataset, configuration)
        if not runs:
            continue
        color = CONFIGURATION_COLORS[configuration]
        for set_name in ("F", "H"):
            precision = [r["precision_recall"][set_name]["precision"] for r in runs]
            recall = [r["precision_recall"][set_name]["recall"] for r in runs]
            face = _faces(set_name, color)
            ax.scatter(recall, precision, s=12, facecolors=face, edgecolors=color, alpha=0.6)
            ax.scatter(
                np.mean(recall), np.mean(precision), s=70, facecolors=face, edgecolors=color,
                linewidths=1.5, label=f"{configuration}, {set_name}",
            )
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))


def plot_distance(
    ax: Axes, results: dict[str, Any], quantity: str, *, dataset: str = "ixi"
) -> None:
    """Median distance (``"seed"`` or ``"train"``) per run, seed means and pooled IQRs.

    The large markers are the seed means that ``F2.md`` prints (``distance_seed_mean``), which
    F2 computed from unrounded per-run medians.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    results : dict[str, Any]
        Content of ``f2_results.json``.
    quantity : {"seed", "train"}
        ``q_seed`` or ``q_train`` of the report.
    dataset : str, default "ixi"
        Which cohort to draw.
    """
    ax.axhspan(*results["distance_band_rminus"][dataset], color="0.88", zorder=0)
    pooled = results["distance_pooled_iqr"][dataset]
    labels = [c for c in CONFIGURATION_ORDER if select_runs(results, dataset, c)]
    for x, configuration in enumerate(labels):
        runs = select_runs(results, dataset, configuration)
        color = CONFIGURATION_COLORS[configuration]
        for set_name, offset in SET_OFFSET.items():
            key = f"{set_name}_{quantity}"
            values = [r["distance_median"][key] for r in runs]
            face = _faces(set_name, color)
            ax.vlines(x + offset, *pooled[configuration][key], color=color, linewidth=1)
            ax.scatter([x + offset] * len(values), values, s=10, facecolors=face,
                       edgecolors=color, alpha=0.6)
            mean = results["distance_seed_mean"][dataset][configuration][key]
            ax.scatter(x + offset, mean, s=60, facecolors=face, edgecolors=color, linewidths=1.5)
    ax.axhline(1.0, color="0.5", linewidth=0.6, linestyle=":")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel(rf"median $q_{{\mathrm{{{quantity}}}}}$")


def plot_octave_errors(ax: Axes, results: dict[str, Any]) -> None:
    """Octave error ``e_o`` per configuration: seed mean (line) and runs (dots), IXI only.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    results : dict[str, Any]
        Content of ``f2_results.json``.
    """
    octaves = results["octaves"]
    x = np.arange(len(octaves))
    for configuration in CONFIGURATION_ORDER:
        runs = select_runs(results, "ixi", configuration)
        errors = np.array([[r["lsd_octaves"][o] for o in octaves] for r in runs])  # (runs, 8)
        color = CONFIGURATION_COLORS[configuration]
        for row in errors:
            ax.scatter(x, row, s=8, color=color, alpha=0.5)
        ax.plot(x, errors.mean(axis=0), marker="o", color=color, label=configuration)
    ax.axhline(0.0, color="0.3", linewidth=0.8)
    ax.set_xticks(x, octaves, rotation=45)
    ax.set_xlabel("octave (c/img)")
    ax.set_ylabel(r"$e_o$ ($\log_{10}$ power ratio)")
    ax.legend(fontsize=7, frameon=False)
