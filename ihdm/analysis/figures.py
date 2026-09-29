"""The report's seven figures, drawn from a T5.2 ``results/`` folder and nothing else (T6.2).

Each ``fig_*`` function reads the runs it needs from :class:`Results`, draws what exists, and
returns a :class:`FigureRecord` that says how many runs it used out of how many the design has
(the "n/N runs" of the caption draft) and which runs were absent. Nothing is dropped silently: an
absent run is listed in the record and in the README; a panel with no run left says so.

Units follow ``docs/SPECIFICATIONS/05-metrics.md``: LSD and its per-octave profile on cycles per
image; the per-octave training loss on the trainer's σ_B bins in pixels
(``ihdm.train.logging.OCTAVE_BIN_EDGES``); the inherited band on cycles per image.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from PIL import Image

from ihdm.analysis.style import (
    ARM_ORDER,
    DATASET_LABELS,
    DATASET_ORDER,
    EXTENSION_SHADE,
    FULL_WIDTH_IN,
    INK,
    MUTED,
    PNG_DPI,
    SINGLE_WIDTH_IN,
    arm_style,
    figure_style,
    save_figure,
)

__all__ = [
    "EXTENSION_START",
    "FIGURES",
    "FigureError",
    "FigureRecord",
    "Results",
    "Run",
    "draw_all",
    "fig_diversity_memorisation",
    "fig_grids",
    "fig_inherited_band",
    "fig_lsd_octaves",
    "fig_lsd_vs_iteration",
    "fig_pca_seed",
    "fig_training_sanity",
    "load_results",
    "write_readme",
]

logger = logging.getLogger(__name__)

Json: TypeAlias = dict[str, Any]

#: First step of the 40k -> 60k extension (D22): every run resumed at 40,001.
EXTENSION_START: int = 40_001
#: Octave bins of the LSD profile, cycles per image (05-metrics.md §1), in order.
LSD_OCTAVES: tuple[str, ...] = ("0.5-1", "1-2", "2-4", "4-8", "8-16", "16-32", "32-64", "64-96")
#: Held-out seeds shown in the PCA figure (the first two of the 40 seed subjects).
PCA_SEEDS: tuple[int, ...] = (0, 1)
#: Datasets and arms of the PCA figure (ticket: A0 vs A3, IXI vs Churches), run seed 1.
PCA_DATASETS: tuple[str, ...] = ("ixi", "lsun_church")
#: The two terminal blurs of the inherited-band figure: A0 (96 px) and A3 (24 px).
BAND_ARMS: tuple[str, ...] = ("A0", "A3")
#: Arms and run seed of the sample-grid figure, plus the run the caption must discuss (D21 d).
GRID_ARMS: tuple[str, ...] = ("A0", "A3")
GRID_SEED: int = 1
STRIPE_RUN: str = "lsun_church_A3_s1"
#: Smoothing window of the training loss, in train records (one record per 50 steps).
LOSS_WINDOW: int = 20
#: The per-octave loss profile averages the train records of the last this-many steps.
OCTAVE_LOSS_TAIL: int = 5_000

_LSD_COLUMN = re.compile(r"^lsd_(\d{6})$")


class FigureError(Exception):
    """The results folder cannot be read (no ``index.csv``, or an unreadable row)."""


@dataclass(frozen=True)
class Run:
    """One cell of ``index.csv``: identity plus the row itself."""

    index: int
    run_id: str
    dataset: str
    arm: str
    seed: int
    n_iters: int
    row: dict[str, str]


@dataclass
class FigureRecord:
    """What one figure drew: files, the n/N behind it, absent runs, and the caption draft."""

    number: int
    name: str
    title: str
    width_in: float
    n_used: int
    n_expected: int
    missing: list[str]
    caption: str
    files: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> str:
        return f"{self.n_used}/{self.n_expected} runs"


class Results:
    """Read-only view of a ``results/`` folder written by ``ihdm.cli.collect_results``."""

    def __init__(self, root: Path, runs: list[Run], collection: Json | None) -> None:
        self.root = root
        self.runs = runs
        self.collection = collection
        self._json: dict[Path, Json | None] = {}
        self._history: dict[str, list[Json] | None] = {}

    def run_dir(self, run: Run) -> Path:
        return self.root / "runs" / run.run_id

    def select(self, dataset: str | None = None, arms: Iterable[str] | None = None,
               seed: int | None = None) -> list[Run]:
        """Runs of the design (present or not) matching the filters, in index order."""
        wanted = set(arms) if arms is not None else None
        return [r for r in self.runs
                if (dataset is None or r.dataset == dataset)
                and (wanted is None or r.arm in wanted)
                and (seed is None or r.seed == seed)]

    def _read_json(self, path: Path) -> Json | None:
        if path not in self._json:
            self._json[path] = json.loads(path.read_text()) if path.is_file() else None
        return self._json[path]

    def final(self, run: Run) -> Json | None:
        """``final.json`` of the run, or None when the run was not evaluated."""
        return self._read_json(self.run_dir(run) / "final.json")

    def lsd_curve(self, run: Run) -> tuple[np.ndarray, np.ndarray] | None:
        """(steps, LSD) of the 500-seed checkpoint set from ``index.csv``; None when absent."""
        points = sorted((int(m.group(1)), float(v)) for k, v in run.row.items()
                        if (m := _LSD_COLUMN.match(k)) and v not in ("", None))
        if not points:
            return None
        steps, values = zip(*points, strict=True)
        return np.asarray(steps), np.asarray(values)

    def history(self, run: Run) -> list[Json] | None:
        """Records of ``metrics.canonical.jsonl``; None when the file is absent."""
        if run.run_id not in self._history:
            path = self.run_dir(run) / "metrics.canonical.jsonl"
            self._history[run.run_id] = (
                [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                if path.is_file() else None)
        return self._history[run.run_id]

    def grid(self, run: Run) -> Path | None:
        path = self.run_dir(run) / "grid_final.png"
        return path if path.is_file() else None

    def pca_train_scores(self, run: Run) -> np.ndarray | None:
        path = self.run_dir(run) / "final_pca_train_scores.npy"
        return np.load(path) if path.is_file() else None

    @cached_property
    def datasets(self) -> list[str]:
        present = {r.dataset for r in self.runs}
        return [d for d in DATASET_ORDER if d in present] + sorted(present - set(DATASET_ORDER))

    def arms_of(self, dataset: str) -> list[str]:
        present = {r.arm for r in self.runs if r.dataset == dataset}
        return [a for a in ARM_ORDER if a in present]


def load_results(root: Path) -> Results:
    """Read ``index.csv`` (and ``collection.json`` when present) of a results folder.

    Parameters
    ----------
    root : Path
        The ``results/`` folder.

    Returns
    -------
    Results
        The cells of the design, in ``index.csv`` order.

    Raises
    ------
    FigureError
        If ``index.csv`` is absent or a row lacks its identity columns.
    """
    index = root / "index.csv"
    if not index.is_file():
        raise FigureError(f"{index} not found: is {root} a results folder of collect_results?")
    runs = []
    with index.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                runs.append(Run(index=int(row["index"]), run_id=row["run_id"],
                                dataset=row["dataset"], arm=row["arm"], seed=int(row["seed"]),
                                n_iters=int(row["n_iters"]), row=row))
            except (KeyError, ValueError) as exc:
                raise FigureError(f"{index}: unreadable row {row!r}") from exc
    if not runs:
        raise FigureError(f"{index} holds no run")
    collection_path = root / "collection.json"
    collection = json.loads(collection_path.read_text()) if collection_path.is_file() else None
    return Results(root, runs, collection)


# ---------------------------------------------------------------------------------------------
# helpers


def _n_iters(results: Results) -> int:
    return max(r.n_iters for r in results.runs)


def _panel_grid(n_panels: int, n_cols: int, width: float, panel_h: float,
                sharex: bool = False) -> tuple[plt.Figure, np.ndarray]:
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width, panel_h * n_rows + 0.55),
                             squeeze=False, sharex=sharex)
    for ax in axes.flat[n_panels:]:
        ax.set_visible(False)
    return fig, axes


def _empty_panel(ax: Axes, text: str) -> None:
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color=MUTED,
            fontsize=7)


def _arm_legend(fig: plt.Figure, arms: list[str], extra: list[Line2D] | None = None,
                ncol: int | None = None) -> None:
    handles = [Line2D([], [], color=arm_style(a).color, marker=arm_style(a).marker,
                      label=arm_style(a).label) for a in arms]
    handles += extra or []
    fig.legend(handles=handles, loc="upper center", ncol=ncol or min(len(handles), 3),
               bbox_to_anchor=(0.5, 1.0), handlelength=1.8, columnspacing=1.2)


def _mark_extension(ax: Axes, n_iters: int, label: bool) -> None:
    """Shade the 40k -> 60k extension (D22) on an axis in thousands of iterations."""
    if n_iters < EXTENSION_START:
        return
    ax.axvspan(EXTENSION_START / 1000, n_iters / 1000, color=EXTENSION_SHADE, zorder=0, lw=0)
    ax.axvline(EXTENSION_START / 1000, color=MUTED, lw=0.6, ls=":", zorder=1)
    if label:
        ax.text(EXTENSION_START / 1000 + 0.6, 0.97, "extension\n(resume at 40,001)",
                transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=5.5,
                color=MUTED)


def _missing(runs: list[Run], present: Callable[[Run], bool]) -> tuple[list[Run], list[str]]:
    used = [r for r in runs if present(r)]
    return used, [r.run_id for r in runs if not present(r)]


def _missing_clause(missing: list[str]) -> str:
    if not missing:
        return ""
    return f" Absent (not evaluated or not collected): {', '.join(f'`{m}`' for m in missing)}."


def _seed_dodge(seed: int, n: int = 3, width: float = 0.18) -> float:
    return (seed - (n + 1) / 2) * width / max(n - 1, 1)


# ---------------------------------------------------------------------------------------------
# Figure 1: LSD versus iteration


def fig_lsd_vs_iteration(results: Results, out_dir: Path, pdf: bool = True,
                         png_dpi: int = PNG_DPI) -> FigureRecord:
    """LSD of the 500-seed checkpoint set against iteration, one panel per dataset.

    Arms are thick lines (seed mean over the seeds present), seeds thin lines; the dashed line is
    the A0 seed-mean LSD at the last checkpoint (the $T_\\tau$ reference); the shaded span is the
    40k -> 60k extension (D22).
    """
    n_iters = _n_iters(results)
    used, missing = _missing(results.runs, lambda r: results.lsd_curve(r) is not None)
    with figure_style():
        fig, axes = _panel_grid(len(results.datasets), 2, FULL_WIDTH_IN, 1.95)
        for ax, dataset in zip(axes.flat, results.datasets, strict=False):
            _lsd_panel(ax, results, dataset, n_iters)
        for ax in axes[-1]:
            ax.set_xlabel("training iteration (thousands)")
        for ax in axes[:, 0]:
            ax.set_ylabel("LSD (500 seeds)")
        ref = Line2D([], [], color=MUTED, ls="--", lw=0.9,
                     label=r"A0 final LSD, seed mean ($T_\tau$ reference)")
        seed = Line2D([], [], color=MUTED, lw=0.6, alpha=0.6, label="single seed")
        _arm_legend(fig, [a for a in ARM_ORDER if any(r.arm == a for r in results.runs)],
                    [ref, seed], ncol=4)
        fig.tight_layout(rect=(0, 0, 1, 0.9), h_pad=0.8, w_pad=1.2)
        files = _save(fig, out_dir, "fig1_lsd_vs_iteration", pdf, png_dpi)
    caption = (
        "Log-spectral distance (LSD, 43 populated log-spaced bins, 0.5-96 cycles per image) "
        "between 500 training-seeded samples and the 800-image reference split, at every "
        f"checkpoint from 5k to {n_iters // 1000}k iterations (12 checkpoints; common random "
        "numbers across checkpoints and arms, D17). One panel per dataset (top: development "
        "pair, IXI and LSUN Churches; bottom: transfer pair, OASIS-1 and LSUN Bedrooms). Thick "
        "lines with markers: seed mean per arm; thin lines: individual seeds; no smoothing, "
        "linear y axis. Dashed line: the mean over the A0 seeds of their LSD at the last "
        "checkpoint on this 500-seed set. The pre-registered $T_\\tau$ compares each seed's "
        "curve with its own A0 seed's value, so the line shows the level of the reference, not "
        "the per-seed thresholds, and no $T_\\tau$ is marked on the curves. The curves are not "
        "monotone in iteration: the LSD at 5k is often at or below its value at the last "
        "checkpoint and rises before it falls, so a first crossing of the reference can occur "
        "at the first checkpoint. Shaded: the extension from 40k to 60k by resume at step "
        "40,001 (D22); the pre-registered gate triggered it on IXI. The y axes are not shared: "
        "LSD is never compared across datasets. "
        f"n = {len(used)}/{len(results.runs)} runs.{_missing_clause(missing)}")
    return FigureRecord(1, "fig1_lsd_vs_iteration", "LSD versus iteration", FULL_WIDTH_IN,
                        len(used), len(results.runs), missing, caption, files)


def _lsd_panel(ax: Axes, results: Results, dataset: str, n_iters: int) -> None:
    ax.set_title(DATASET_LABELS.get(dataset, dataset))
    _mark_extension(ax, n_iters, label=dataset == results.datasets[0])
    drawn = False
    for arm in results.arms_of(dataset):
        curves = [c for r in results.select(dataset, [arm]) if (c := results.lsd_curve(r))]
        if not curves:
            continue
        drawn = True
        style = arm_style(arm)
        for steps, values in curves:
            ax.plot(steps / 1000, values, color=style.color, lw=0.6, alpha=0.45, zorder=2)
        steps, mean = _mean_curve(curves)
        ax.plot(steps / 1000, mean, color=style.color, marker=style.marker, lw=1.4,
                markersize=3, zorder=3)
    a0 = [c[1][-1] for r in results.select(dataset, ["A0"]) if (c := results.lsd_curve(r))]
    if a0:
        ax.axhline(float(np.mean(a0)), color=MUTED, ls="--", lw=0.9, zorder=1)
    elif drawn:
        ax.text(0.02, 0.04, "no A0 run: no $T_\\tau$ reference", transform=ax.transAxes,
                fontsize=5.5, color=MUTED)
    if not drawn:
        _empty_panel(ax, "no evaluated run")
    ax.set_xlim(0, n_iters / 1000 + 1)


def _mean_curve(curves: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Seed mean over the steps every curve has."""
    common = sorted(set.intersection(*(set(s.tolist()) for s, _ in curves)))
    stacked = np.array([[v[list(s).index(step)] for step in common] for s, v in curves])
    return np.asarray(common), stacked.mean(axis=0)


# ---------------------------------------------------------------------------------------------
# Figure 2: per-octave LSD profile at the final checkpoint


def fig_lsd_octaves(results: Results, out_dir: Path, pdf: bool = True,
                    png_dpi: int = PNG_DPI) -> FigureRecord:
    """Signed per-octave log-variance ratio of the final 2k set, per dataset and arm."""
    used, missing = _missing(results.runs, lambda r: _octaves(results, r) is not None)
    x = np.arange(len(LSD_OCTAVES))
    with figure_style():
        fig, axes = _panel_grid(len(results.datasets), 2, FULL_WIDTH_IN, 1.85)
        for ax, dataset in zip(axes.flat, results.datasets, strict=False):
            ax.set_title(DATASET_LABELS.get(dataset, dataset))
            ax.axhline(0.0, color=MUTED, lw=0.7, zorder=1)
            arms = results.arms_of(dataset)
            drawn = False
            for k, arm in enumerate(arms):
                profiles = [p for r in results.select(dataset, [arm])
                            if (p := _octaves(results, r)) is not None]
                if not profiles:
                    continue
                drawn = True
                style = arm_style(arm)
                shift = (k - (len(arms) - 1) / 2) * 0.1
                for p in profiles:
                    ax.plot(x + shift, p, ls="none", marker=style.marker, color=style.color,
                            markersize=2.2, alpha=0.55, zorder=2)
                ax.plot(x + shift, np.mean(profiles, axis=0), color=style.color, lw=1.2,
                        marker=style.marker, markersize=3.2, zorder=3)
            if not drawn:
                _empty_panel(ax, "no evaluated run")
            ax.set_xticks(x, [b.replace("-", "–") for b in LSD_OCTAVES])
            ax.set_xlim(-0.5, len(LSD_OCTAVES) - 0.5)
        for ax in axes[-1]:
            ax.set_xlabel("octave band (cycles per image)")
        for ax in axes[:, 0]:
            ax.set_ylabel(r"$\log_{10}\bar P_S - \log_{10}\bar P_R$")
        seed = Line2D([], [], color=MUTED, ls="none", marker="o", markersize=2.2, alpha=0.6,
                      label="single seed")
        _arm_legend(fig, [a for a in ARM_ORDER if any(r.arm == a for r in results.runs)],
                    [seed], ncol=3)
        fig.tight_layout(rect=(0, 0, 1, 0.9), h_pad=0.8, w_pad=1.2)
        files = _save(fig, out_dir, "fig2_lsd_octaves", pdf, png_dpi)
    caption = (
        "Per-octave spectral error of the final sample set (2,000 training-seeded samples at "
        f"{_n_iters(results) // 1000}k iterations) against the reference split: "
        "$\\log_{10}\\bar P_S(b) - \\log_{10}\\bar P_R(b)$ on the eight octave bands of the "
        "project, in cycles per image (05-metrics §2; sign kept: positive means the samples "
        "carry too much variance in that band; the DC mode and modes above 96 cycles per image "
        "are excluded). Lines with markers: seed mean per arm; small markers: individual seeds; "
        "arms are offset horizontally for legibility. One panel per dataset; y axes not shared. "
        f"n = {len(used)}/{len(results.runs)} runs.{_missing_clause(missing)}")
    return FigureRecord(2, "fig2_lsd_octaves", "Per-octave LSD profile at the final checkpoint",
                        FULL_WIDTH_IN, len(used), len(results.runs), missing, caption, files)


def _octaves(results: Results, run: Run) -> np.ndarray | None:
    final = results.final(run)
    if final is None or "lsd_octaves" not in final:
        return None
    return np.array([final["lsd_octaves"][b] for b in LSD_OCTAVES], dtype=float)


# ---------------------------------------------------------------------------------------------
# Figure 3: diversity and memorisation

_DIVERSITY_METRICS: tuple[tuple[str, str, float | None], ...] = (
    ("diversity_pix", "$D_{\\mathrm{pix}}$\n(non-DC var. / pixel)", None),
    ("diversity_lp", "$D_{\\mathrm{lp}}$\n($\\sigma_B$ = 16 px low-pass)", None),
    ("M", "$M$\n(NN-distance ratio)", 1.0),
    ("M_lp", "$M_{\\mathrm{lp}}$\n(NN ratio, low-pass)", 1.0),
)


def fig_diversity_memorisation(results: Results, out_dir: Path, pdf: bool = True,
                               png_dpi: int = PNG_DPI) -> FigureRecord:
    """Within-seed diversity and memorisation ratio per dataset and arm, one point per seed."""
    used, missing = _missing(results.runs, lambda r: results.final(r) is not None)
    datasets = results.datasets
    with figure_style():
        fig, axes = plt.subplots(len(_DIVERSITY_METRICS), len(datasets),
                                 figsize=(FULL_WIDTH_IN, 5.9), squeeze=False)
        for col, dataset in enumerate(datasets):
            for row, (key, label, ref) in enumerate(_DIVERSITY_METRICS):
                ax = axes[row, col]
                _strip_panel(ax, results, dataset, key, ref)
                if row == 0:
                    ax.set_title(DATASET_LABELS.get(dataset, dataset))
                if col == 0:
                    ax.set_ylabel(label)
        seed = Line2D([], [], color=MUTED, ls="none", marker="o", markersize=3,
                      label="one seed (run)")
        mean = Line2D([], [], color=MUTED, lw=1.6, label="seed mean")
        _arm_legend(fig, [a for a in ARM_ORDER if any(r.arm == a for r in results.runs)],
                    [seed, mean], ncol=4)
        fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=0.6, w_pad=0.8)
        files = _save(fig, out_dir, "fig3_diversity_memorisation", pdf, png_dpi)
    caption = (
        "Within-seed diversity and memorisation at the final checkpoint "
        f"({_n_iters(results) // 1000}k), per dataset (columns) and arm (x position); each "
        "marker is one training run (seed), the horizontal bar is the seed mean. Rows 1-2: "
        "$D_{\\mathrm{pix}}$ and $D_{\\mathrm{lp}}$, the non-DC per-pixel variance across 50 "
        "samples drawn from the same prior state, averaged over the 40 held-out seed subjects "
        "(05-metrics §3; $D_{\\mathrm{lp}}$ after the $\\sigma_B = 16$ px heat-kernel low-pass). "
        "Rows 3-4: $M$ and $M_{\\mathrm{lp}}$, the median nearest-training-image distance of 2,000 "
        "training-seeded samples divided by that of held-out real images (05-metrics §4); the "
        "dotted line $M = 1$ is a new real image, $M < 1$ indicates copying. Axes are per panel. "
        f"n = {len(used)}/{len(results.runs)} runs.{_missing_clause(missing)}")
    return FigureRecord(3, "fig3_diversity_memorisation", "Diversity and memorisation",
                        FULL_WIDTH_IN, len(used), len(results.runs), missing, caption, files)


def _strip_panel(ax: Axes, results: Results, dataset: str, key: str, ref: float | None) -> None:
    arms = results.arms_of(dataset)
    ax.grid(axis="x", visible=False)
    if ref is not None:
        ax.axhline(ref, color=MUTED, lw=0.7, ls=":", zorder=1)
    for k, arm in enumerate(arms):
        style = arm_style(arm)
        runs = results.select(dataset, [arm])
        n_seeds = max(r.seed for r in runs)
        values = []
        for r in runs:
            final = results.final(r)
            if final is None or final.get(key) is None:
                continue
            values.append(float(final[key]))
            ax.plot(k + _seed_dodge(r.seed, n_seeds), final[key], ls="none", marker=style.marker,
                    color=style.color, markersize=3.4, zorder=3)
        if values:
            ax.plot([k - 0.25, k + 0.25], [np.mean(values)] * 2, color=style.color, lw=1.6,
                    zorder=2, solid_capstyle="round")
        else:
            ax.text(k, 0.5, "n/a", transform=ax.get_xaxis_transform(), ha="center",
                    va="center", fontsize=6, color=MUTED)
    ax.set_xticks(range(len(arms)), [a.replace("A2p", "A2′") for a in arms])
    ax.set_xlim(-0.6, len(arms) - 0.4)
    # Plain tick labels: an axis offset text (x10^-3) would push the column title up.
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.3g}"))


# ---------------------------------------------------------------------------------------------
# Figure 4: PCA around the seed


def fig_pca_seed(results: Results, out_dir: Path, pdf: bool = True,
                 png_dpi: int = PNG_DPI) -> FigureRecord:
    """Two held-out seeds and their 50 samples in the dataset's training-PCA plane."""
    expected = [r for d in PCA_DATASETS for a in BAND_ARMS
                for r in results.select(d, [a], seed=1)]
    used, missing = _missing(expected, lambda r: (results.final(r) or {}).get("pca") is not None)
    with figure_style():
        fig, axes = plt.subplots(len(PCA_DATASETS), len(BAND_ARMS),
                                 figsize=(SINGLE_WIDTH_IN, 3.55), squeeze=False)
        for row, dataset in enumerate(PCA_DATASETS):
            drawn: list[tuple[Axes, Json]] = []
            for col, arm in enumerate(BAND_ARMS):
                ax = axes[row, col]
                ax.set_title(f"{DATASET_LABELS[dataset]}, {arm}", fontsize=7)
                runs = results.select(dataset, [arm], seed=1)
                pca = _pca_panel(ax, results, runs[0] if runs else None)
                if pca is not None:
                    drawn.append((ax, pca))
            _zoom_row(axes[row], [_pca_points(pca) for _, pca in drawn])
            for ax, pca in drawn:
                _pca_arrows(ax, pca)
        handles = [
            Line2D([], [], ls="none", marker=".", color="#b4b2ab", label="training split"),
            Line2D([], [], ls="none", marker="*", markerfacecolor="none", markeredgecolor=INK,
                   markersize=6, label="seed"),
            Line2D([], [], ls="none", marker="o", color=MUTED, markersize=2.5,
                   label=f"samples, seed {PCA_SEEDS[0] + 1}"),
            Line2D([], [], ls="none", marker="^", color=MUTED, markersize=2.5,
                   label=f"samples, seed {PCA_SEEDS[1] + 1}"),
        ]
        fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0),
                   handletextpad=0.3, columnspacing=0.8)
        for ax in axes[-1]:
            ax.set_xlabel("PC 1", labelpad=1)
        for ax in axes[:, 0]:
            ax.set_ylabel("PC 2", labelpad=1)
        fig.tight_layout(rect=(0, 0, 1, 0.9), h_pad=0.5, w_pad=0.4)
        files = _save(fig, out_dir, "fig4_pca_seed", pdf, png_dpi)
    caption = (
        "Samples around their seed in the plane of the first two principal components of the "
        "training split (3,200 images, DC removed; 05-metrics §6; the basis is the same for "
        "every run of a dataset). Each row is zoomed to its seeds and samples; grey points are "
        "the training images that fall in that window. Hollow star: a held-out seed image "
        f"(seed subjects {PCA_SEEDS[0] + 1} and {PCA_SEEDS[1] + 1} of 40, slice 5); circles and "
        "triangles: its 50 samples drawn from the same prior state (seed blurred to the "
        "terminal level, plus prior noise); arrow: seed to sample centroid, drawn when the "
        "shift exceeds 4% of the window. Rows: IXI and LSUN Churches; columns: A0 "
        "($\\sigma_{B,\\max}$ 96, log spacing) and A3 ($\\sigma_{B,\\max}$ 24, IXI-matched "
        "spacing), training seed 1 of each. Axes are shared within a row. "
        f"n = {len(used)}/{len(expected)} runs.{_missing_clause(missing)}")
    return FigureRecord(4, "fig4_pca_seed", "PCA around the seed", SINGLE_WIDTH_IN, len(used),
                        len(expected), missing, caption, files)


def _pca_selection(pca: Json) -> tuple[np.ndarray, np.ndarray]:
    """(seed scores, sample scores) of the :data:`PCA_SEEDS` present in the run."""
    seeds = np.asarray(pca["seed_scores"], dtype=float)
    samples = np.asarray(pca["sample_scores"], dtype=float)
    keep = [s for s in PCA_SEEDS if s < len(seeds)]
    return seeds[keep], samples[keep]


def _pca_points(pca: Json) -> np.ndarray:
    seeds, samples = _pca_selection(pca)
    return np.concatenate([seeds, samples.reshape(-1, 2)])


def _pca_panel(ax: Axes, results: Results, run: Run | None) -> Json | None:
    """Draw the training cloud, the seeds and their samples; return the run's PCA block."""
    ax.tick_params(labelsize=5.5)
    ax.grid(False)
    final = results.final(run) if run else None
    if run is None or final is None or "pca" not in final:
        _empty_panel(ax, "not evaluated")
        return None
    train = results.pca_train_scores(run)
    if train is not None:
        ax.scatter(train[:, 0], train[:, 1], s=1.2, color="#b4b2ab", lw=0, rasterized=True,
                   zorder=1)
    seeds, samples = _pca_selection(final["pca"])
    style = arm_style(run.arm)
    for marker, seed, cloud in zip(("o", "^"), seeds, samples, strict=False):
        ax.scatter(cloud[:, 0], cloud[:, 1], s=5, marker=marker, color=style.color, lw=0,
                   alpha=0.85, zorder=3)
        ax.scatter(*seed, s=42, marker="*", facecolor="none", edgecolor=INK, lw=0.7, zorder=5)
    return final["pca"]


def _pca_arrows(ax: Axes, pca: Json) -> None:
    """Arrow from each seed to its sample centroid, when the shift is visible at this zoom."""
    span = max(np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()))
    seeds, samples = _pca_selection(pca)
    for seed, cloud in zip(seeds, samples, strict=False):
        centroid = cloud.mean(axis=0)
        if np.hypot(*(centroid - seed)) < 0.04 * span:
            continue
        ax.annotate("", xy=centroid, xytext=seed,
                    arrowprops={"arrowstyle": "->", "color": INK, "lw": 0.7,
                                "shrinkA": 3, "shrinkB": 0}, zorder=4)


def _zoom_row(axes_row: Iterable[Axes], points: list[np.ndarray]) -> None:
    """Shared limits for a row: the seeds and samples of every panel, padded by a third."""
    if not points:
        return
    stacked = np.concatenate(points)
    lo, hi = stacked.min(axis=0), stacked.max(axis=0)
    pad = np.maximum((hi - lo) / 3, 1e-3 + 0.05 * np.abs(hi - lo).max())
    for ax in axes_row:
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])


# ---------------------------------------------------------------------------------------------
# Figure 5: inherited band


def fig_inherited_band(results: Results, out_dir: Path, pdf: bool = True,
                       png_dpi: int = PNG_DPI) -> FigureRecord:
    """Measured versus linear-Gaussian predicted regenerated variance, A0 and A3 per dataset."""
    expected = [r for r in results.runs if r.arm in BAND_ARMS]
    used, missing = _missing(expected,
                             lambda r: (results.final(r) or {}).get("radial") is not None)
    with figure_style():
        fig, axes = _panel_grid(len(results.datasets), 4, FULL_WIDTH_IN, 1.75)
        drawn = [ax for ax, dataset in zip(axes.flat, results.datasets, strict=False)
                 if _band_panel(ax, results, dataset)]
        for ax in axes.flat[:len(results.datasets)]:
            ax.set_xlabel("cycles per image")
        # One y range for every drawn panel: the ratio is dimensionless and comparable.
        if drawn:
            lows, highs = zip(*(ax.get_ylim() for ax in drawn), strict=True)
            for ax in drawn:
                ax.set_ylim(min(lows), max(highs))
        axes[0, 0].set_ylabel(r"$\langle V\rangle / P_{\mathrm{ref}}$")
        measured = Line2D([], [], color=MUTED, lw=1.3, label="measured (seed mean)")
        seed = Line2D([], [], color=MUTED, lw=0.6, alpha=0.6, label="single seed")
        predicted = Line2D([], [], color=MUTED, lw=1.0, ls="--",
                           label=r"predicted $1-d_K^2$")
        _arm_legend(fig, [a for a in BAND_ARMS if any(r.arm == a for r in expected)],
                    [measured, seed, predicted], ncol=5)
        fig.tight_layout(rect=(0, 0, 1, 0.84), w_pad=0.6)
        files = _save(fig, out_dir, "fig5_inherited_band", pdf, png_dpi)
    caption = (
        "The inherited band (05-metrics §5): per-mode variance of 50 samples about their prior "
        "state $d_K \\hat x_s$, averaged over the 40 held-out seeds and divided by the "
        "population variance $P_{\\mathrm{ref}}$, as a radial profile on the 43 populated "
        "log-spaced bins (solid; thin: single seeds), against the linear-Gaussian prediction "
        "$1 - d_K^2(n) = 1 - e^{-2\\lambda_n t_K}$ with $t_K = \\sigma_{B,\\max}^2/2$ (dashed). "
        "The two terminal blurs: A0 ($\\sigma_{B,\\max}$ = 96 px, prediction $\\approx 1$ above "
        "one cycle per image) and A3 ($\\sigma_{B,\\max}$ = 24 px, the prior keeps the "
        "low band). A measured curve on its dashed line means the model regenerates exactly the "
        "variance the prior removed; below it, the samples inherit more of the seed. Log axes, "
        "one y range for all panels. The bins below 3 cycles per image hold 1-6 DCT modes each "
        "on the $192^2$ grid, so the measured ratio there is noisy. "
        f"n = {len(used)}/{len(expected)} runs.{_missing_clause(missing)}")
    return FigureRecord(5, "fig5_inherited_band", "Inherited band", FULL_WIDTH_IN, len(used),
                        len(expected), missing, caption, files)


def _band_panel(ax: Axes, results: Results, dataset: str) -> bool:
    """Draw one dataset's measured and predicted curves; False when no A0/A3 run exists."""
    ax.set_title(DATASET_LABELS.get(dataset, dataset))
    drawn = False
    for arm in BAND_ARMS:
        radials = [f["radial"] for r in results.select(dataset, [arm])
                   if (f := results.final(r)) and f.get("radial")]
        if not radials:
            continue
        if not drawn:
            ax.set_xscale("log")
            ax.set_yscale("log")
        drawn = True
        style = arm_style(arm)
        centres = np.asarray(radials[0]["centres"], dtype=float)
        measured = [np.asarray(rad["measured"], dtype=float) for rad in radials]
        for m in measured:
            ax.plot(centres, _positive(m), color=style.color, lw=0.6, alpha=0.45, zorder=2)
        ax.plot(centres, _positive(np.mean(measured, axis=0)), color=style.color, lw=1.3,
                zorder=3)
        ax.plot(centres, _positive(np.asarray(radials[0]["predicted"], dtype=float)),
                color=style.color, lw=1.0, ls="--", zorder=4)
    if not drawn:
        _empty_panel(ax, "no evaluated A0/A3 run")
        ax.set_xticks([])
        ax.set_yticks([])
    return drawn


def _positive(values: np.ndarray) -> np.ndarray:
    """Mask non-positive values on a log axis instead of letting matplotlib drop them silently."""
    return np.where(values > 0, values, np.nan)


# ---------------------------------------------------------------------------------------------
# Figure 6: sample grids


def fig_grids(results: Results, out_dir: Path, pdf: bool = True,
              png_dpi: int = PNG_DPI) -> FigureRecord:
    """The 60k training grids (seed row above, samples below) of A0 and A3 for each dataset."""
    expected = [r for d in results.datasets for a in GRID_ARMS
                for r in results.select(d, [a], seed=GRID_SEED)]
    used, missing = _missing(expected, lambda r: results.grid(r) is not None)
    datasets = results.datasets
    with figure_style():
        fig, axes = plt.subplots(len(datasets), len(GRID_ARMS),
                                 figsize=(FULL_WIDTH_IN, 0.95 * len(datasets) + 0.25),
                                 squeeze=False)
        for row, dataset in enumerate(datasets):
            for col, arm in enumerate(GRID_ARMS):
                ax = axes[row, col]
                ax.set_axis_off()
                runs = results.select(dataset, [arm], seed=GRID_SEED)
                path = results.grid(runs[0]) if runs else None
                if path is None:
                    _empty_panel(ax, "grid absent")
                else:
                    with Image.open(path) as image:
                        ax.imshow(np.asarray(image.convert("L")), cmap="gray", vmin=0, vmax=255,
                                  interpolation="none")
                name = runs[0].run_id if runs else f"{dataset}_{arm}_s{GRID_SEED}"
                ax.set_title(f"{DATASET_LABELS.get(dataset, dataset)}, {arm}  (`{name}`)"
                             .replace("`", ""), fontsize=6.5, pad=2)
        fig.subplots_adjust(left=0.005, right=0.995, top=0.96, bottom=0.005, wspace=0.03,
                            hspace=0.22)
        files = _save(fig, out_dir, "fig6_grids", pdf, png_dpi, pdf_dpi=300)
    stripe = next((r for r in expected if r.run_id == STRIPE_RUN), None)
    stripe_note = (
        f" `{STRIPE_RUN}` (Churches, A3, right column) was flagged at 40k for a saturated "
        "vertical stripe at the right edge of every sample (D21 d; seeds 2 and 3 do not show "
        "it). The stripe is not visible in the 60k grid shown here."
        if stripe is not None else "")
    notes = ([f"`{STRIPE_RUN}` at 60k, checked on 2026-09-29 (T6.2): in each of the 8 samples "
              "the mean of the 3 right-most pixel columns differs from that of the 15 columns "
              "inside them by at most 14 grey levels (`lsun_church_A3_s2`: 17; "
              "`lsun_church_A0_s1`: 6), and no right-edge column averages above 206 of 255. "
              "Recheck if the grid file changes."]
             if stripe is not None else [])
    caption = (
        f"Training sample grids at the last iteration ({_n_iters(results) // 1000}k, EMA weights) "
        f"of training seed {GRID_SEED}, for A0 (left) and A3 (right) on each dataset. In each "
        "grid the top row holds 8 training images used as seeds and the bottom row the sample "
        "each seed produces after blurring to the terminal level. These are the trainer's "
        "monitoring grids, not the evaluation sets." + stripe_note +
        f" n = {len(used)}/{len(expected)} runs.{_missing_clause(missing)}")
    return FigureRecord(6, "fig6_grids", "Sample grids", FULL_WIDTH_IN, len(used), len(expected),
                        missing, caption, files, notes)


# ---------------------------------------------------------------------------------------------
# Figure 7: training sanity panel


def fig_training_sanity(results: Results, out_dir: Path, pdf: bool = True,
                        png_dpi: int = PNG_DPI) -> FigureRecord:
    """Training loss, throughput and the late per-octave loss from the canonical histories."""
    used, missing = _missing(results.runs, lambda r: results.history(r) is not None)
    n_iters = _n_iters(results)
    datasets = results.datasets
    with figure_style():
        fig, axes = plt.subplots(3, len(datasets), figsize=(FULL_WIDTH_IN, 5.6), squeeze=False)
        for col, dataset in enumerate(datasets):
            _training_column(axes[:, col], results, dataset, n_iters, label=col == 0)
            axes[0, col].set_title(DATASET_LABELS.get(dataset, dataset))
        axes[0, 0].set_ylabel(f"train loss ({LOSS_WINDOW * 50}-it. mean)")
        axes[1, 0].set_ylabel("iterations / s")
        axes[2, 0].set_ylabel("loss per band\n(last 5k it.)")
        seed = Line2D([], [], color=MUTED, lw=0.6, alpha=0.6, label="one seed (run)")
        _arm_legend(fig, [a for a in ARM_ORDER if any(r.arm == a for r in results.runs)],
                    [seed], ncol=3)
        fig.tight_layout(rect=(0, 0, 1, 0.91), h_pad=1.0, w_pad=0.6)
        files = _save(fig, out_dir, "fig7_training_sanity", pdf, png_dpi, pdf_dpi=300)
    skips = {r.run_id: n for r in used
             if (n := sum(1 for rec in results.history(r) or [] if rec.get("kind") == "skip"))}
    skip_text = (", ".join(f"`{k}` {v} skipped steps" for k, v in skips.items())
                 if skips else "none")
    caption = (
        "Training sanity from each run's canonical history (`metrics.canonical.jsonl`: the "
        "segment abandoned by run 11's resume at 30,001 is already removed, D21). Top: training "
        f"loss, running mean over {LOSS_WINDOW} logged steps ({LOSS_WINDOW * 50} iterations), "
        "log scale; middle: throughput in iterations per second as logged (A100, batch 16). "
        f"Both y ranges span the records from step {SETTLED_STEP:,} on; the warm-up and the "
        "first logged step lie above or below them. Bottom: the per-octave training loss "
        f"averaged over the last {OCTAVE_LOSS_TAIL:,} iterations, on the trainer's σ_B bands in "
        "pixels (labelled by their lower edge; bands an arm's blur range never visits are not "
        "logged and not drawn), log scale. Thin lines: single runs; thick lines in the bottom "
        "row: seed mean. The loss is the regression loss of each arm's own blur schedule, so "
        "its level is not comparable across arms; the panel checks convergence and stability "
        "only. Shaded: the 40k → 60k extension, resumed at "
        f"step 40,001 (D22). Skipped fp16 steps kept in the canonical histories: {skip_text}. "
        f"n = {len(used)}/{len(results.runs)} runs.{_missing_clause(missing)}")
    return FigureRecord(7, "fig7_training_sanity", "Training sanity panel", FULL_WIDTH_IN,
                        len(used), len(results.runs), missing, caption, files)


def _train_series(history: list[Json], key: str) -> tuple[np.ndarray, np.ndarray]:
    points = [(rec["step"], rec[key]) for rec in history
              if rec.get("kind") == "train" and rec.get(key) is not None]
    if not points:
        return np.empty(0), np.empty(0)
    steps, values = zip(*points, strict=True)
    return np.asarray(steps, dtype=float), np.asarray(values, dtype=float)


def _running_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return values
    cumsum = np.cumsum(np.insert(values, 0, 0.0))
    head = cumsum[1:window] / np.arange(1, window)
    return np.concatenate([head, (cumsum[window:] - cumsum[:-window]) / window])


def _nanmean_columns(table: np.ndarray) -> np.ndarray:
    """Column means ignoring NaN; NaN (without a warning) for an all-NaN column."""
    counts = np.sum(~np.isnan(table), axis=0)
    sums = np.where(np.isnan(table), 0.0, table).sum(axis=0)
    return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def _octave_loss(history: list[Json], n_iters: int, bands: tuple[str, ...]) -> np.ndarray:
    rows = [[(rec["loss_per_octave"] or {}).get(b) for b in bands] for rec in history
            if rec.get("kind") == "train" and rec["step"] > n_iters - OCTAVE_LOSS_TAIL
            and rec.get("loss_per_octave")]
    table = np.array([[np.nan if v is None else v for v in row] for row in rows], dtype=float)
    if table.size == 0:
        return np.full(len(bands), np.nan)
    return _nanmean_columns(table)


#: Records before this step are off the loss and throughput y ranges (warm-up and start-up).
SETTLED_STEP: int = 2_500


def _settled_limits(values: list[tuple[np.ndarray, np.ndarray]], log: bool,
                    ) -> tuple[float, float] | None:
    """y limits spanning the settled part (step >= :data:`SETTLED_STEP`) of every series."""
    settled = np.concatenate([v[s >= SETTLED_STEP] for s, v in values]) if values else []
    settled = np.asarray(settled, dtype=float)
    settled = settled[np.isfinite(settled) & ((settled > 0) if log else True)]
    if settled.size == 0:
        return None
    lo, hi = float(settled.min()), float(settled.max())
    if log:
        return lo / 1.12, hi * 1.12
    pad = max(0.08 * (hi - lo), 0.01 * abs(hi))
    return lo - pad, hi + pad


def _training_column(axes: np.ndarray, results: Results, dataset: str, n_iters: int,
                     label: bool) -> None:
    from ihdm.train.logging import OCTAVE_BIN_EDGES, OCTAVE_BIN_NAMES

    ax_loss, ax_speed, ax_oct = axes
    for ax in (ax_loss, ax_speed):
        _mark_extension(ax, n_iters, label=label and ax is ax_loss)
        ax.set_xlim(0, n_iters / 1000 + 1)
    ax_speed.set_xlabel("iteration (thousands)")
    arms = results.arms_of(dataset)
    x = np.arange(len(OCTAVE_BIN_NAMES))
    losses, speeds = [], []
    for k, arm in enumerate(arms):
        style = arm_style(arm)
        profiles = []
        for run in results.select(dataset, [arm]):
            history = results.history(run)
            if history is None:
                continue
            steps, loss = _train_series(history, "loss")
            losses.append((steps, _running_mean(loss, LOSS_WINDOW)))
            ax_loss.plot(steps / 1000, losses[-1][1], color=style.color, lw=0.6, alpha=0.7)
            speeds.append(_train_series(history, "it_per_s"))
            ax_speed.plot(speeds[-1][0] / 1000, speeds[-1][1], color=style.color, lw=0.5,
                          alpha=0.6)
            profiles.append(_octave_loss(history, n_iters, OCTAVE_BIN_NAMES))
        shift = (k - (len(arms) - 1) / 2) * 0.1
        for p in profiles:
            ax_oct.plot(x + shift, p, color=style.color, lw=0.5, alpha=0.5, marker=style.marker,
                        markersize=1.8)
        if profiles:
            ax_oct.plot(x + shift, _nanmean_columns(np.array(profiles)), color=style.color,
                        lw=1.2, marker=style.marker, markersize=3)
    if not losses:
        for ax in axes:
            _empty_panel(ax, "no history")
    else:
        ax_loss.set_yscale("log")
        ax_oct.set_yscale("log")
        # Majors at 1, 2, (3,) 5 x 10^k so an axis narrower than a decade keeps its labels.
        for axis, subs in ((ax_loss.yaxis, (1.0, 2.0, 3.0, 5.0)), (ax_oct.yaxis, (1.0, 3.0))):
            axis.set_major_locator(LogLocator(base=10, subs=subs))
            axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            axis.set_minor_formatter(NullFormatter())
        if (limits := _settled_limits(losses, log=True)) is not None:
            ax_loss.set_ylim(*limits)
        if (limits := _settled_limits(speeds, log=False)) is not None:
            ax_speed.set_ylim(*limits)
    ax_oct.set_xticks(x, [f"{e:g}" for e in OCTAVE_BIN_EDGES[:-1]], fontsize=5.5)
    ax_oct.set_xlim(-0.5, len(x) - 0.5)
    ax_oct.set_xlabel(r"$\sigma_B$ band, lower edge (px)")


# ---------------------------------------------------------------------------------------------
# driver


def _save(fig: plt.Figure, out_dir: Path, name: str, pdf: bool, png_dpi: int,
          pdf_dpi: int = 300) -> list[Path]:
    return save_figure(fig, out_dir, name, pdf=pdf, png_dpi=png_dpi, pdf_dpi=pdf_dpi)


FigureFn: TypeAlias = Callable[..., FigureRecord]

#: The report figures in caption order.
FIGURES: tuple[tuple[str, FigureFn], ...] = (
    ("fig1_lsd_vs_iteration", fig_lsd_vs_iteration),
    ("fig2_lsd_octaves", fig_lsd_octaves),
    ("fig3_diversity_memorisation", fig_diversity_memorisation),
    ("fig4_pca_seed", fig_pca_seed),
    ("fig5_inherited_band", fig_inherited_band),
    ("fig6_grids", fig_grids),
    ("fig7_training_sanity", fig_training_sanity),
)


def draw_all(results: Results, out_dir: Path, pdf: bool = True, png_dpi: int = PNG_DPI,
             only: Iterable[str] | None = None) -> list[FigureRecord]:
    """Draw every figure (or the ``only`` subset) and return their records in figure order.

    Parameters
    ----------
    results : Results
        The loaded results folder.
    out_dir : Path
        Where the PDF and PNG files go.
    pdf : bool
        Write the vector PDFs as well as the PNG previews.
    png_dpi : int
        Resolution of the PNG previews.
    only : Iterable[str] | None
        Figure names to draw; all when None.

    Returns
    -------
    list[FigureRecord]
        One record per figure drawn.
    """
    wanted = set(only) if only is not None else None
    records = []
    for name, fn in FIGURES:
        if wanted is not None and name not in wanted:
            continue
        records.append(fn(results, out_dir, pdf=pdf, png_dpi=png_dpi))
    return records


def write_readme(records: list[FigureRecord], results: Results, path: Path,
                 command: str) -> Path:
    """Write the caption drafts, the n/N behind each figure and the absent runs to ``path``.

    Parameters
    ----------
    records : list[FigureRecord]
        The records returned by :func:`draw_all`.
    results : Results
        The results folder the figures were drawn from.
    path : Path
        The README to write; image links are relative to its folder.
    command : str
        The command line that produced the figures, quoted in the README.

    Returns
    -------
    Path
        ``path``.
    """
    collection = results.collection or {}
    complete = collection.get("complete")
    lines = [
        "# Report figures (T6.2)",
        "",
        "Drawn by `python -m ihdm.cli.figures` from a `results/` folder of "
        "`python -m ihdm.cli.collect_results` (`docs/RESULTS/collection.md`) and nothing else. "
        "Code: `ihdm/analysis/figures.py`, `ihdm/analysis/style.py`, `ihdm/cli/figures.py`.",
        "",
        "| field | value |",
        "|---|---|",
        f"| command | `{command}` |",
        f"| results folder | `{results.root}` |",
        f"| collection complete | {complete} (verdict `{collection.get('verdict')}`, created "
        f"{collection.get('created')}, git `{collection.get('git_sha')}`) |",
        f"| runs in `index.csv` | {len(results.runs)} |",
        "",
    ]
    if complete is not True:
        lines += ["> **Partial collection.** Captions give the n/N runs each figure uses and name "
                  "the absent runs; rerun the command on the complete collection before quoting "
                  "any figure.", ""]
    lines += ["Style: one colour and one marker per arm in every figure (A0 blue circle, A3 "
              "orange square, A1 aqua triangle, A2 yellow diamond, A2′ pink inverted triangle; "
              "the `dataviz` reference palette, colour-blind checked); widths 3.3 in (single "
              "column) and 6.75 in (full width); PDFs are vector with no creation date.", ""]
    for record in records:
        files = ", ".join(f"[`{f.name}`]({_relative(f, path.parent)})" for f in record.files)
        lines += [
            f"## Figure {record.number}. {record.title}",
            "",
            f"Files: {files}. Width {record.width_in} in. Coverage: **{record.coverage}**.",
            "",
            f"**Caption draft.** {record.caption}",
            "",
        ]
        lines += [f"- {note}" for note in record.notes]
        if record.notes:
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")
    return path


def _relative(target: Path, base: Path) -> str:
    return Path(os.path.relpath(target.resolve(), base.resolve())).as_posix()
