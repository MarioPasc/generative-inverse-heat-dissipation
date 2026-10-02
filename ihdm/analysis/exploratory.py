"""Exploratory readings of the collected results, defined after seeing the data (2026-10-01).

Nothing in this module is pre-registered. Every number is either descriptive or computed with
the statistics of :mod:`ihdm.analysis.tables` (seed-paired percentile bootstrap, exact
permutation p with its floor), so "CI excludes 0" means "all seeds agree in sign", exactly as in
tables 2-6 (``docs/RESULTS/tables/README.md``).

Each reading answers a question the pre-registered tables raised:

* **LSD decomposition.** The per-octave errors :math:`e_b = \\log_{10}\\bar P_S(b) -
  \\log_{10}\\bar P_R(b)` of the final 2k set split exactly into a level and a shape,
  :math:`\\mathrm{RMS}(e)^2 = \\bar e^2 + \\mathrm{sd}(e)^2`. The level is a broadband variance
  deficit (samples too alike at every scale); the shape is a mis-allocation of variance across
  scales. The octave RMS is also split at 2 and 4 cycles per image into three bands: below
  2 c/img a :math:`\\sigma_{B,\\max} = 24` prior carries most of the seed's variance
  (:math:`d_K^2` falls from 0.86 at 0.5 c/img to 0.085 at 2); the 2-4 c/img band is the hand-off,
  where :math:`d_K^2` drops further, to 0.008 at 2.8 c/img and :math:`5\\times10^{-5}` at 4;
  above 4 c/img no arm's prior carries anything (:math:`d_K^2 < 10^{-4}`).
* **Late-window LSD.** The mean of the 500-seed LSD curve over its last four checkpoints
  (45k-60k), beside the single-checkpoint final LSD, whose photograph curves oscillate.
* **Inception precision and density**, stored by every run but not tabulated by T6.1.
* **Regeneration ratio** :math:`\\rho = (1 - I_w) / (1 - I)`: the within-seed variance the chain
  adds over the variance the terminal blur removed (the D23 quantities of table 1c). 1 = the
  chain regenerates exactly what the blur removed; below 1 = under-dispersion.

It also summarises the checkpoint curves (LSD, variance ratio and octave level at 5k, at the LSD
peak and at the last step) and the within-run correlation of the LSD with the octave level,
which is what the non-monotone LSD of table 7 needs.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ihdm.analysis.style import (
    ARM_ORDER,
    DATASET_LABELS,
    DATASET_ORDER,
    FULL_WIDTH_IN,
    INK,
    MUTED,
    arm_style,
    figure_style,
    save_figure,
)
from ihdm.analysis.tables import (
    AnalysisError,
    ContrastResult,
    Results,
    ResultsNotFound,
    arm_label,
    compute_contrast,
    compute_interaction,
    fmt_ci,
    fmt_num,
    fmt_p,
    load_results,
)

__all__ = [
    "CONTRASTS",
    "ENDPOINTS",
    "OCTAVES",
    "ExploratoryEndpoint",
    "add_endpoints",
    "checkpoint_frame",
    "curve_summary",
    "main",
    "octave_summary",
    "run",
]

logger = logging.getLogger(__name__)

#: The eight octave bins of ``05-metrics.md`` §1, in cycles per image.
OCTAVES: tuple[str, ...] = ("0.5-1", "1-2", "2-4", "4-8", "8-16", "16-32", "32-64", "64-96")
#: 0.5-2 c/img: a sigma_B,max = 24 prior carries most of the seed's variance here
#: (d_K^2 falls from 0.86 at 0.5 c/img to 0.085 at 2).
PRIOR_OCTAVES: tuple[str, ...] = OCTAVES[:2]
#: 2-4 c/img: the hand-off band, where d_K^2 drops from 0.085 at 2 c/img to 5e-5 at 4
#: (0.008 at 2.8).
MID_OCTAVES: tuple[str, ...] = OCTAVES[2:3]
#: Above 4 c/img, where no sigma_B,max = 24 prior carries anything (d_K^2 < 1e-4).
HIGH_OCTAVES: tuple[str, ...] = OCTAVES[3:]
#: Checkpoints averaged by the late-window LSD.
LATE_STEPS: tuple[int, ...] = (45000, 50000, 55000, 60000)
DEFINED: str = "2026-10-01"


@dataclass(frozen=True)
class ExploratoryEndpoint:
    """A per-run scalar added to the analysis frame.

    Parameters
    ----------
    key : str
        Column of ``results.frame``.
    label : str
        Markdown label.
    reading : str
        Which direction is better, in words.
    """

    key: str
    label: str
    reading: str


ENDPOINTS: tuple[ExploratoryEndpoint, ...] = (
    ExploratoryEndpoint("lsd_final", "LSD (final, 2k; pre-registered, for reference)",
                        "lower is better"),
    ExploratoryEndpoint("lsd_late", "LSD, mean of 45k-60k (500 seeds)", "lower is better"),
    ExploratoryEndpoint("oct_rms", "octave RMS error (final)", "≈ LSD on 8 bins; lower is better"),
    ExploratoryEndpoint("oct_level", "octave level ē (final)",
                        "0 = right total variance; negative = samples too alike"),
    ExploratoryEndpoint("oct_shape", "octave shape sd(e) (final)",
                        "0 = right allocation across scales"),
    ExploratoryEndpoint("oct_rms_prior", "octave RMS, 0.5-2 c/img",
                        "the band a σ_B,max = 24 prior mostly carries (d_K² 0.86 at 0.5 c/img, "
                        "0.085 at 2)"),
    ExploratoryEndpoint("oct_rms_mid", "octave RMS, 2-4 c/img",
                        "the hand-off band (d_K² 0.085 at 2 c/img, 0.008 at 2.8, 5e-5 at 4)"),
    ExploratoryEndpoint("oct_rms_high", "octave RMS, 4-96 c/img", "bands no prior carries"),
    ExploratoryEndpoint("log_vr", "log10 variance ratio (final)", "0 = right total variance"),
    ExploratoryEndpoint("precision", "precision (Inception, k = 5)", "higher is better"),
    ExploratoryEndpoint("density", "density (Inception, k = 5)", "higher is better"),
    ExploratoryEndpoint("regen_ratio", "regeneration ratio ρ = (1 − I_w)/(1 − I)",
                        "1 = the chain regenerates what the blur removed"),
)

#: (dataset, arm, reference) of the contrast table; the transfer rows carry no inference (§8).
CONTRASTS: tuple[tuple[str, str, str], ...] = (
    ("ixi", "A3", "A0"), ("ixi", "A1", "A0"), ("ixi", "A2", "A0"),
    ("lsun_church", "A3", "A0"), ("lsun_church", "A1", "A0"), ("lsun_church", "A2", "A0"),
    ("lsun_church", "A2p", "A0"), ("lsun_church", "A2p", "A2"),
)
TRANSFER: tuple[str, ...] = ("oasis1", "lsun_bedroom")


# --------------------------------------------------------------------------------------------
# Per-run quantities
# --------------------------------------------------------------------------------------------


def octave_errors(record: dict[str, Any]) -> np.ndarray:
    """Return the eight octave errors of one ``final.json`` or ``ckpt_<step>.json`` record."""
    return np.array([float(record["lsd_octaves"][o]) for o in OCTAVES])


def octave_summary(errors: Sequence[float]) -> dict[str, float]:
    """Split the octave errors into RMS, level and shape, and the RMS of the three bands.

    Parameters
    ----------
    errors : Sequence[float]
        The eight per-octave errors, in :data:`OCTAVES` order.

    Returns
    -------
    dict[str, float]
        ``oct_rms``, ``oct_level`` (mean), ``oct_shape`` (population sd; ``oct_rms**2 ==
        oct_level**2 + oct_shape**2``), ``oct_rms_prior`` (:data:`PRIOR_OCTAVES`, 0.5-2 c/img),
        ``oct_rms_mid`` (:data:`MID_OCTAVES`, the 2-4 c/img hand-off) and ``oct_rms_high``
        (:data:`HIGH_OCTAVES`, 4-96 c/img).

    Raises
    ------
    ValueError
        If there are not eight finite errors.
    """
    e = np.asarray(errors, dtype=float)
    if e.shape != (len(OCTAVES),) or not np.all(np.isfinite(e)):
        raise ValueError(f"expected {len(OCTAVES)} finite octave errors, got {e!r}")
    n_prior, n_mid = len(PRIOR_OCTAVES), len(MID_OCTAVES)
    return {
        "oct_rms": float(np.sqrt(np.mean(e ** 2))),
        "oct_level": float(np.mean(e)),
        "oct_shape": float(np.std(e)),
        "oct_rms_prior": float(np.sqrt(np.mean(e[:n_prior] ** 2))),
        "oct_rms_mid": float(np.sqrt(np.mean(e[n_prior:n_prior + n_mid] ** 2))),
        "oct_rms_high": float(np.sqrt(np.mean(e[n_prior + n_mid:] ** 2))),
    }


def regeneration_ratio(within_share: float, predicted_share: float) -> float:
    """Return ``(1 - I_w) / (1 - I)``: within-seed variance added over the variance removed."""
    if not (math.isfinite(within_share) and math.isfinite(predicted_share)):
        return math.nan
    if predicted_share >= 1.0:
        raise ValueError(f"the predicted inherited share must be below 1, got {predicted_share}")
    return (1.0 - within_share) / (1.0 - predicted_share)


def _read(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise AnalysisError(f"{path}: {error}") from error


def _octave_summary_or_raise(rid: str, errors: Sequence[float]) -> dict[str, float]:
    """:func:`octave_summary`, naming the run id instead of letting a bare ValueError escape."""
    try:
        return octave_summary(errors)
    except ValueError as error:
        raise AnalysisError(f"{rid}: {error}") from error


def _log_vr_or_raise(rid: str, variance_ratio: float) -> float:
    """``log10(variance_ratio)``, naming the run id when the ratio is not strictly positive."""
    try:
        return math.log10(variance_ratio)
    except ValueError as error:
        raise AnalysisError(
            f"{rid}: variance_ratio must be positive to take log10, got {variance_ratio}"
        ) from error


def add_endpoints(results: Results) -> None:
    """Add the exploratory columns of :data:`ENDPOINTS` to ``results.frame`` in place.

    Parameters
    ----------
    results : Results
        The loaded folder; absent runs get ``NaN``.

    Raises
    ------
    AnalysisError
        If a present run lacks its ``final.json`` or a late checkpoint.
    """
    frame = results.frame
    late = [f"lsd_{s:06d}" for s in LATE_STEPS if s in results.steps]
    if len(late) != len(LATE_STEPS):
        raise AnalysisError(f"the late window needs checkpoints {LATE_STEPS}, "
                            f"the collection has {results.steps}")
    columns: dict[str, list[float]] = {k: [] for k in
                                       ("oct_rms", "oct_level", "oct_shape", "oct_rms_prior",
                                        "oct_rms_mid", "oct_rms_high", "log_vr", "lsd_late",
                                        "regen_ratio")}
    for rid in frame.index:
        if not frame.at[rid, "present"]:
            for values in columns.values():
                values.append(math.nan)
            continue
        final = _read(results.root / "runs" / rid / "final.json")
        for key, value in _octave_summary_or_raise(rid, octave_errors(final)).items():
            columns[key].append(value)
        columns["log_vr"].append(_log_vr_or_raise(rid, float(final["variance_ratio"])))
        columns["lsd_late"].append(float(np.mean([float(frame.at[rid, c]) for c in late])))
        columns["regen_ratio"].append(regeneration_ratio(
            float(frame.at[rid, "inherited_within"]), float(frame.at[rid, "inherited_predicted"])))
    for key, values in columns.items():
        frame[key] = values


def checkpoint_frame(results: Results) -> pd.DataFrame:
    """One row per present run and evaluated checkpoint: LSD, variance ratio, octave summary."""
    rows = []
    for rid in sorted(results.present):
        identity = results.frame.loc[rid, ["dataset", "arm", "seed"]].to_dict()
        for step in results.steps:
            record = _read(results.root / "runs" / rid / f"ckpt_{step:06d}.json")
            rows.append({"run_id": rid, **identity, "step": int(step),
                         "lsd": float(record["lsd"]),
                         "variance_ratio": float(record["variance_ratio"]),
                         **_octave_summary_or_raise(rid, octave_errors(record))})
    return pd.DataFrame(rows)


def _within_run_corr(ckpt: pd.DataFrame, x: str, y: str) -> float:
    """Pearson correlation of ``x`` and ``y`` after removing each run's mean."""
    dx = ckpt[x] - ckpt.groupby("run_id")[x].transform("mean")
    dy = ckpt[y] - ckpt.groupby("run_id")[y].transform("mean")
    return float(np.corrcoef(dx, dy)[0, 1])


def curve_summary(ckpt: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Seed-mean LSD, variance ratio and octave level at 5k, at the LSD peak and at the end.

    Parameters
    ----------
    ckpt : pandas.DataFrame
        :func:`checkpoint_frame`.

    Returns
    -------
    tuple[pandas.DataFrame, dict[str, float]]
        One row per dataset and arm; and the pooled correlations (``lsd~|level|`` within runs,
        ``lsd~|log10 vr|`` within runs, ``lsd~oct_rms`` across all records, and the mean share
        of ``oct_rms**2`` that the level carries).
    """
    ckpt = ckpt.assign(abs_level=ckpt["oct_level"].abs(),
                       abs_log_vr=np.log10(ckpt["variance_ratio"]).abs())
    first, last = int(ckpt["step"].min()), int(ckpt["step"].max())
    rows = []
    for (dataset, arm), group in ckpt.groupby(["dataset", "arm"], sort=False):
        mean = group.groupby("step")[["lsd", "variance_ratio", "oct_level"]].mean()
        peak = int(mean["lsd"].idxmax())
        rows.append({"dataset": dataset, "arm": arm, "n_seeds": group["run_id"].nunique(),
                     "peak_step": peak,
                     **{f"{q}_{tag}": float(mean.at[step, q])
                        for tag, step in (("first", first), ("peak", peak), ("last", last))
                        for q in ("lsd", "variance_ratio", "oct_level")}})
    order = {d: i for i, d in enumerate(DATASET_ORDER)}
    arms = {a: i for i, a in enumerate(("A0", "A3", "A1", "A2", "A2p"))}
    summary = pd.DataFrame(rows).sort_values(
        by=["dataset", "arm"], key=lambda s: s.map(order) if s.name == "dataset" else s.map(arms))
    stats = {
        "within_run_corr_lsd_abs_level": _within_run_corr(ckpt, "lsd", "abs_level"),
        "within_run_corr_lsd_abs_log_vr": _within_run_corr(ckpt, "lsd", "abs_log_vr"),
        "corr_lsd_oct_rms": float(np.corrcoef(ckpt["lsd"], ckpt["oct_rms"])[0, 1]),
        "mean_level_share_of_oct_rms2": float(np.mean(ckpt["oct_level"] ** 2
                                                      / ckpt["oct_rms"] ** 2)),
        "median_level_share_of_oct_rms2": float(np.median(ckpt["oct_level"] ** 2
                                                          / ckpt["oct_rms"] ** 2)),
        "n_records": int(len(ckpt)),
        "first_step": first,
        "last_step": last,
    }
    return summary.reset_index(drop=True), stats


# --------------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------------


def _label(dataset: str) -> str:
    return DATASET_LABELS[dataset]


def _cell_table(results: Results) -> list[str]:
    frame = results.frame[results.frame["present"]]
    head = "| dataset | arm | n | " + " | ".join(e.label for e in ENDPOINTS) + " |"
    lines = [head, "|---|---|---:|" + "---:|" * len(ENDPOINTS)]
    for dataset in DATASET_ORDER:
        for arm in ("A0", "A3", "A1", "A2", "A2p"):
            cell = frame[(frame["dataset"] == dataset) & (frame["arm"] == arm)]
            if cell.empty:
                continue
            values = " | ".join(fmt_num(cell[e.key].mean(), signed=e.key.startswith(("oct_level",
                                                                                     "log_vr")))
                                for e in ENDPOINTS)
            lines.append(f"| {_label(dataset)} | {arm_label(arm)} | {len(cell)} | {values} |")
    return lines


def _contrast_line(c: ContrastResult, label: str) -> str:
    if not c.ok:
        return f"| {label} | {c.status} | | | | | |"
    per_seed = " / ".join(fmt_num(d, signed=True) for d in c.deltas)
    if math.isnan(c.ci_low):
        return (f"| {label} | {len(c.seeds)} | {fmt_num(c.mean, signed=True)} | — | {per_seed} "
                f"| — | sign only (§8) |")
    return (f"| {label} | {len(c.seeds)} | {fmt_num(c.mean, signed=True)} | "
            f"{fmt_ci(c.ci_low, c.ci_high)} | {per_seed} | {fmt_p(c.p_value)} (min "
            f"{fmt_p(c.p_min)}) | {c.verdict} |")


def _contrast_tables(results: Results) -> tuple[list[str], list[dict[str, Any]]]:
    lines: list[str] = []
    records: list[dict[str, Any]] = []
    for endpoint in ENDPOINTS[1:]:
        lines += ["", f"**{endpoint.label}** ({endpoint.reading})", "",
                  "| contrast | seeds | mean Δ | 95% CI | per-seed Δ | perm. p | reading |",
                  "|---|---:|---:|---:|---|---:|---|"]
        for dataset, arm, reference in CONTRASTS:
            c = compute_contrast(results, dataset, arm, endpoint.key, reference=reference)
            lines.append(_contrast_line(c, f"{_label(dataset)} {arm_label(arm)}−"
                                           f"{arm_label(reference)}"))
            records.append(c.to_json())
        for dataset in TRANSFER:
            c = compute_contrast(results, dataset, "A3", endpoint.key, inference=False)
            lines.append(_contrast_line(c, f"{_label(dataset)} A3−A0 (transfer)"))
            records.append(c.to_json())
        i = compute_interaction(results, endpoint.key)
        if i.status == "ok":
            lines.append(f"| **interaction** Δ_IXI − Δ_Churches (A3) | 3 + 3 | "
                         f"{fmt_num(i.point, signed=True)} | {fmt_ci(i.ci_low, i.ci_high)} | — | "
                         f"{fmt_p(i.p_value)} (min {fmt_p(i.p_min)}) | {i.verdict} |")
        else:
            lines.append(f"| **interaction** | {i.status} | | | | | |")
        records.append({"interaction": True, **i.to_json()})
    return lines, records


def _curve_table(summary: pd.DataFrame, stats: dict[str, float]) -> list[str]:
    first, last = stats["first_step"] // 1000, stats["last_step"] // 1000
    lines = [f"| dataset | arm | n | LSD {first}k | LSD peak (step) | LSD {last}k | "
             f"var. ratio {first}k | var. ratio at peak | var. ratio {last}k | "
             f"level ē {first}k | ē at peak | ē {last}k |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples():
        lines.append(
            f"| {_label(row.dataset)} | {arm_label(row.arm)} | {row.n_seeds} | "
            f"{row.lsd_first:.3f} | {row.lsd_peak:.3f} ({row.peak_step // 1000}k) | "
            f"{row.lsd_last:.3f} | {row.variance_ratio_first:.2f} | "
            f"{row.variance_ratio_peak:.2f} | {row.variance_ratio_last:.2f} | "
            f"{row.oct_level_first:+.3f} | {row.oct_level_peak:+.3f} | {row.oct_level_last:+.3f} |")
    lines += ["",
              f"- {stats['n_records']} checkpoint records (every run, every evaluated step). "
              f"Octave RMS against LSD: r = {stats['corr_lsd_oct_rms']:.4f}. The level ē² "
              f"carries a mean {stats['mean_level_share_of_oct_rms2']:.0%} (median "
              f"{stats['median_level_share_of_oct_rms2']:.0%}) of the octave RMS².",
              f"- Within runs (each run's mean removed), LSD correlates with |ē| at r = "
              f"{stats['within_run_corr_lsd_abs_level']:.3f} and with |log10 variance ratio| at "
              f"r = {stats['within_run_corr_lsd_abs_log_vr']:.3f}."]
    return lines


def build_markdown(results: Results, ckpt: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    """Return the exploratory report and its JSON record."""
    summary, stats = curve_summary(ckpt)
    contrast_lines, contrast_records = _contrast_tables(results)
    banner = (f"**Exploratory: defined after seeing the data ({DEFINED}), not pre-registered.** "
              "Same statistics as tables 2-6: with 3 seeds a 95% CI is [min, max] of the per-seed "
              "Δ, so 'CI excludes 0' means 'all seeds agree in sign'; p_min is 0.1 at 3 vs 3 "
              "seeds and 1/3 at 2 vs 2. Transfer rows give signs only.")
    lines = [
        "# Exploratory readings of the 30-run results",
        "",
        banner,
        "",
        "Command: `PYTHONPATH=$PWD python -m ihdm.analysis.exploratory --results <results_dir> "
        "--out docs/RESULTS/exploratory/`. Source: " + f"`{results.root}` (collection "
        f"{results.collection.get('verdict')}).",
        "",
        "Definitions. e_b = log10 P̄_S(b) − log10 P̄_R(b) on the eight octaves of the final 2k "
        "set (`final.json` `lsd_octaves`); octave RMS = sqrt(mean e_b²) (tracks LSD, which "
        "uses 43 finer bins); level ē = mean e_b; shape = sd(e_b), so RMS² = ē² + sd². 'Prior' "
        "= 0.5-2 c/img (a σ_B,max = 24 prior keeps d_K² ≈ 0.86 at 0.5 c/img, falling to 0.085 "
        "at 2), 'mid' = 2-4 c/img (the hand-off band: d_K² ≈ 0.008 at 2.8 c/img, 5e-5 at 4), "
        "'high' = 4-96 c/img (no prior carries anything there). Variance ratio = ΣP_S/ΣP_R "
        "(non-DC). ρ = (1 − I_w)/(1 − I) from table 1c. Precision and density: Naeem et al. "
        "(2020), k = 5, Inception pool features, as stored by every run.",
        "",
        "## X1. Seed means per cell",
        "",
        *_cell_table(results),
        "",
        "## X2. Contrasts, transfer signs and the A3 interaction",
        *contrast_lines,
        "",
        "## X3. The checkpoint curves (500-seed set, seed means)",
        "",
        *_curve_table(summary, stats),
        "",
        "Figure: `fx1_dispersion.pdf` / `.png` (top: variance ratio against iteration; bottom: "
        "LSD against the octave level |ē| over every checkpoint, with the line LSD = |ē| that a "
        "pure broadband deficit would follow).",
        "",
    ]
    record = {"defined": DEFINED, "results": str(results.root),
              "collection_verdict": results.collection.get("verdict"),
              "endpoints": [e.__dict__ for e in ENDPOINTS],
              "per_run": results.frame.loc[results.frame["present"],
                                           ["dataset", "arm", "seed",
                                            *[e.key for e in ENDPOINTS]]]
              .reset_index().to_dict(orient="records"),
              "contrasts": contrast_records, "curves": summary.to_dict(orient="records"),
              "curve_stats": stats}
    return "\n".join(lines), record


# --------------------------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------------------------


def draw_dispersion(ckpt: pd.DataFrame, out: Path, png_dpi: int = 200) -> list[Path]:
    """Variance ratio against iteration, and LSD against the octave level, per dataset."""
    datasets = [d for d in DATASET_ORDER if d in set(ckpt["dataset"])]
    with figure_style():
        fig, axes = plt.subplots(2, len(datasets), figsize=(FULL_WIDTH_IN, 3.6), squeeze=False,
                                 constrained_layout=True)
        for col, dataset in enumerate(datasets):
            sub = ckpt[ckpt["dataset"] == dataset]
            top, bottom = axes[0, col], axes[1, col]
            for arm in [a for a in ARM_ORDER if a in set(sub["arm"])]:
                style = arm_style(arm)
                cell = sub[sub["arm"] == arm]
                for _, run in cell.groupby("run_id"):
                    top.plot(run["step"] / 1000, run["variance_ratio"], color=style.color,
                             lw=0.5, alpha=0.45)
                mean = cell.groupby("step")["variance_ratio"].mean()
                top.plot(mean.index / 1000, mean.values, color=style.color, marker=style.marker,
                         label=style.label)
                bottom.scatter(cell["oct_level"].abs(), cell["lsd"], s=6, color=style.color,
                               marker=style.marker, linewidths=0)
            top.axhline(1.0, color=MUTED, lw=0.6, ls=":")
            top.set_title(_label(dataset))
            top.set_xlabel("iteration (thousands)")
            hi = float(max(sub["lsd"].max(), sub["oct_level"].abs().max())) * 1.05
            bottom.plot([0, hi], [0, hi], color=INK, lw=0.6, ls="--")
            bottom.set_xlim(0, hi)
            bottom.set_ylim(0, hi)
            bottom.set_xlabel("|octave level ē|")
        axes[0, 0].set_ylabel("variance ratio ΣP_S / ΣP_R")
        axes[1, 0].set_ylabel("LSD (500 seeds)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        for ax in axes[0, 1:]:
            for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
                if lab not in labels:
                    handles.append(h)
                    labels.append(lab)
        fig.legend(handles, labels, loc="outside upper center", ncol=3)
        return save_figure(fig, out, "fx1_dispersion", png_dpi=png_dpi)


# --------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------


def run(results_dir: Path, out: Path, png_dpi: int = 200,
        inherited_constants: Path | None = None) -> Path:
    """Compute every exploratory reading and write ``exploratory.md``, ``.json`` and the figure.

    Parameters
    ----------
    results_dir : Path
        A folder written by ``python -m ihdm.cli.collect_results``.
    out : Path
        Output directory; created if absent.
    png_dpi : int
        Resolution of the PNG preview.
    inherited_constants : Path or None
        D23 constants file; default the committed one (see :func:`ihdm.analysis.tables.load_results`).

    Returns
    -------
    Path
        The Markdown report.
    """
    results = (load_results(results_dir) if inherited_constants is None
               else load_results(results_dir, inherited_constants))
    add_endpoints(results)
    ckpt = checkpoint_frame(results)
    text, record = build_markdown(results, ckpt)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "exploratory.md"
    path.write_text(text + "\n")
    (out / "exploratory.json").write_text(json.dumps(record, indent=1, default=float) + "\n")
    draw_dispersion(ckpt, out, png_dpi=png_dpi)
    return path


def main(argv: list[str] | None = None) -> int:
    """Command line: ``python -m ihdm.analysis.exploratory --results <dir> --out <dir>``.

    Returns
    -------
    int
        0 written; 1 the folder cannot be analysed; 2 not a results folder.
    """
    parser = argparse.ArgumentParser(prog="python -m ihdm.analysis.exploratory",
                                     description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, required=True, help="collected results folder")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--png-dpi", type=int, default=200)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        path = run(args.results, args.out, png_dpi=args.png_dpi)
    except ResultsNotFound as error:
        logger.error("%s", error)
        return 2
    except AnalysisError as error:
        logger.error("%s", error)
        return 1
    logger.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
