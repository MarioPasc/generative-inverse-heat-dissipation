"""The report's tables and statistics (T6.1), computed from one collected ``results/`` folder.

Input: the folder ``python -m ihdm.cli.collect_results`` publishes (``docs/RESULTS/collection.md``)
and nothing else. Output: the tables of ``docs/SPECIFICATIONS/M6-analysis/T6.1``, each as
Markdown and as a LaTeX ``booktabs`` float, plus one ``tables.json``.

Every statistic is the pre-registered one of ``05-metrics.md`` §8, computed with
:mod:`ihdm.stats`:

* a contrast ``Delta = m(arm) - m(reference)`` is paired by seed over the seeds both cells hold;
  its interval is the percentile bootstrap over seeds (:func:`ihdm.stats.paired_delta`, 10,000
  draws, 95%) and its p-value the exact permutation of arm labels over the pooled seeds of the
  two cells (:func:`ihdm.stats.permutation_test`), always printed with the floor ``p_min``;
* the interaction ``Delta_MRI - Delta_photo`` (IXI against Churches, A3 against A0) resamples
  the two sets of per-seed differences independently (:func:`ihdm.stats.interaction`); its
  permutation test permutes the data-type label over the pooled per-seed differences;
* a CI that contains zero is reported as "not detectable at this budget".

``T_tau`` (``05-metrics.md`` §2) is computed here from each run's LSD curve (``lsd_<step>`` of
``index.csv``). The primary threshold is the same-seed A0 run's LSD at its last checkpoint on the
curve's own estimator (500 frozen seeds, common random numbers, D17), so the curve and its
threshold come from one sample set; the A0 run's 2,000-sample final LSD is kept as a labelled
sensitivity threshold (``main``, 2026-09-29).

A table that needs a run the collection lacks is written anyway and marked
``incomplete: n/N runs``; its rows that need the missing run carry no statistic.

Table 9, the settling step of the LSD curve, is exploratory: defined after seeing the data
(2026-09-29), not pre-registered, and reported descriptively without CI or p-value.

The inherited band is read as D23 prescribes (``docs/RESULTS/inherited_band_audit.md``): the
pre-registered ``inherited_measured`` stays, labelled biased under a non-zero mean image, and two
endpoints are added from the stored scalars and the reference constants of
``docs/RESULTS/inherited_band_constants.json`` (:mod:`ihdm.analysis.inherited`): the within-seed
share ``I_w`` (read against ``inherited_predicted``) and the seed-mean bias fraction ``G_b`` (read
against ``T + (1 - I) / M``). Table 1c sets them side by side per run; tables 2-6 test them like
every other endpoint.
"""

from __future__ import annotations

import logging
import math
import re
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from configs.spectral.arms import EXPERIMENT_CELLS
from ihdm.analysis.collect import strict_json
from ihdm.analysis.inherited import (
    DEFAULT_CONSTANTS_PATH,
    InheritedConstants,
    InheritedError,
    bias_expected,
    bias_fraction,
    load_constants,
    within_seed_share,
)
from ihdm.metrics.io import write_json
from ihdm.metrics.spectral import t_tau
from ihdm.paths import repo_root
from ihdm.stats import cell_table, interaction, paired_delta, permutation_test

__all__ = [
    "D23_COLUMNS",
    "ENDPOINTS",
    "PREREGISTERED_LABEL",
    "AnalysisError",
    "Column",
    "ContrastResult",
    "Endpoint",
    "InteractionResult",
    "Results",
    "ResultsNotFound",
    "Table",
    "analyse",
    "build_tables",
    "compute_contrast",
    "compute_interaction",
    "load_results",
    "t_tau_frame",
    "write_tables",
]

logger = logging.getLogger(__name__)

#: Resamples, level and RNG seed of every bootstrap interval (``05-metrics.md`` §8).
N_BOOT: int = 10_000
ALPHA: float = 0.05
RNG_SEED: int = 0

REFERENCE_ARM: str = "A0"
#: The development pair of ``00-overview.md`` §1: the headline interaction is IXI vs Churches.
MRI_DATASET: str = "ixi"
PHOTO_DATASET: str = "lsun_church"
#: The arms contrasted against A0 on each development dataset (T6.1 table 2).
DEVELOPMENT_ARMS: dict[str, tuple[str, ...]] = {
    MRI_DATASET: ("A3", "A1", "A2"),
    PHOTO_DATASET: ("A3", "A1", "A2", "A2p"),
}
#: Development dataset -> its transfer dataset (A0 and A3 only, 2 seeds).
TRANSFER_PAIRS: tuple[tuple[str, str], ...] = ((MRI_DATASET, "oasis1"),
                                               (PHOTO_DATASET, "lsun_bedroom"))
DATASET_ORDER: tuple[str, ...] = ("ixi", "lsun_church", "oasis1", "lsun_bedroom")
ARM_ORDER: tuple[str, ...] = ("A0", "A3", "A1", "A2", "A2p")
DATASET_LABEL: dict[str, str] = {"ixi": "IXI", "oasis1": "OASIS-1", "lsun_church": "Churches",
                                 "lsun_bedroom": "Bedrooms"}

STATUS_OK: str = "ok"
NOT_DETECTABLE: str = "not detectable at this budget"
DETECTABLE: str = "CI excludes 0"
MISSING: str = "—"
NOT_REACHED: str = "not reached"

#: ``index.csv`` column -> the dotted key :func:`ihdm.stats.cell_table` gives the same number.
_SUMMARY_KEYS: dict[str, str] = {
    "lsd_final": "final.lsd",
    "lsd_final_ckpt": "final.intermediate_lsd",
    "kid": "final.inception.kid",
    "kid_ci_low": "final.inception.kid_ci_low",
    "kid_ci_high": "final.inception.kid_ci_high",
    "fid": "final.inception.fid",
    "fid_ci_low": "final.inception.fid_ci_low",
    "fid_ci_high": "final.inception.fid_ci_high",
    "fid_n_reference": "final.inception.n_reference",
    "recall": "final.inception.recall",
    "coverage": "final.inception.coverage",
    "M": "final.M",
    "M_lp": "final.M_lp",
    "seed_nn_fraction": "final.seed_nn_fraction",
    "D_pix": "final.diversity_pix",
    "D_lp": "final.diversity_lp",
    "inherited_measured": "final.inherited_measured",
    "inherited_predicted": "final.inherited_predicted",
}
#: Columns only the summaries hold (the low-band inherited shares of ``05-metrics.md`` §5, and
#: the terminal blur and samples per seed that the D23 reading needs).
_SUMMARY_ONLY: dict[str, str] = {
    "inherited_measured_low": "final.inherited_measured_low_band",
    "inherited_predicted_low": "final.inherited_predicted_low_band",
    "sigma_max": "final.sigma_max",
    "n_per_seed": "final.n_per_seed",
}
#: Columns of the D23 reading (``inherited_band_audit.md`` §3.3, §5), added by the loader.
D23_COLUMNS: tuple[str, ...] = ("inherited_within", "inherited_bias", "inherited_bias_expected",
                                "inherited_mean_term", "inherited_expected")
_INDEX_REQUIRED: tuple[str, ...] = ("index", "run_id", "dataset", "arm", "seed", "n_skipped",
                                    *_SUMMARY_KEYS)
_GATE_FILE = re.compile(r"^(?P<run_id>.+)_gate_(?P<early>\d{6})_(?P<late>\d{6})\.json$")


class AnalysisError(Exception):
    """Raised when a results folder cannot be analysed (integrity or consistency problem)."""


class ResultsNotFound(AnalysisError):
    """Raised when the path given is not a results folder at all."""


# --------------------------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Endpoint:
    """A scalar endpoint that the contrast tables test.

    Parameters
    ----------
    key : str
        Column of the analysis frame.
    label : str
        Markdown label.
    tex : str
        LaTeX label.
    reading : str
        Which direction is better, in words, for the README and the notes.
    kind : {"float", "step"}
        ``"step"`` values are checkpoint steps (``T_tau``), formatted in thousands.
    ratio : bool
        Whether the endpoint is a positive quantity on a ratio scale, so that the descriptive
        ``Delta / A0`` of tables 3 and 6 means something. ``False`` for proportions that sit
        near zero (recall, coverage, seed-NN fraction) and for the signed inherited share,
        where dividing by the A0 value inflates or flips the reading.
    """

    key: str
    label: str
    tex: str
    reading: str
    kind: str = "float"
    ratio: bool = True


#: How the pre-registered inherited share is labelled since D23.
PREREGISTERED_LABEL: str = ("as pre-registered (biased under a non-zero mean image; see "
                            "inherited_band_audit.md)")

#: The endpoints of tables 2-6 (``00-overview.md`` §1, ``05-metrics.md`` §2-§7), in report order;
#: the two D23 inherited-band endpoints follow the pre-registered share.
ENDPOINTS: tuple[Endpoint, ...] = (
    Endpoint("lsd_final", "LSD (final, 2k set)", "LSD (final)", "lower is better"),
    Endpoint("t_tau", "T_τ (steps)", r"$T_\tau$ (steps)", "lower is earlier", "step"),
    Endpoint("kid", "KID (headline)", "KID", "lower is better"),
    Endpoint("fid", "FID", "FID", "lower is better; biased upward at N_ref = 800"),
    Endpoint("recall", "recall", "recall", "higher is better", ratio=False),
    Endpoint("coverage", "coverage", "coverage", "higher is better", ratio=False),
    Endpoint("M", "M", "$M$", "1 = as far as held-out data; below 1 = copying"),
    Endpoint("M_lp", "M_lp", r"$M_{\mathrm{lp}}$", "as M, after the sigma_B = 16 low-pass"),
    Endpoint("seed_nn_fraction", "seed-NN fraction", "seed-NN frac.",
             "share of samples whose nearest training image is their own seed", ratio=False),
    Endpoint("D_pix", "D_pix", r"$D_{\mathrm{pix}}$", "higher = more within-seed diversity"),
    Endpoint("D_lp", "D_lp", r"$D_{\mathrm{lp}}$", "as D_pix, after the low-pass"),
    Endpoint("inherited_measured",
             f"inherited share {PREREGISTERED_LABEL}", "inherited (pre-reg., biased)",
             "model expectation I − T, not I (table 1c; D23)", ratio=False),
    Endpoint("inherited_within", "within-seed share I_w (D23)", r"$I_w$ (D23)",
             "read against the predicted share I (table 1c)", ratio=False),
    Endpoint("inherited_bias", "seed-mean bias fraction G_b (D23)", r"$G_b$ (D23)",
             "read against T + (1 − I)/M (table 1c)", ratio=False),
    Endpoint("t_tau_2k", "T_τ, 2k-set threshold (sensitivity)",
             r"$T_\tau$, 2k threshold (sens.)", "lower is earlier", "step"),
)
ENDPOINT_BY_KEY: dict[str, Endpoint] = {e.key: e for e in ENDPOINTS}


# --------------------------------------------------------------------------------------------
# Design helpers
# --------------------------------------------------------------------------------------------


def run_id(dataset: str, arm: str, seed: int) -> str:
    """Return the run id of a cell seed (``slurm/array/cells.csv`` convention)."""
    return f"{dataset}_{arm}_s{int(seed)}"


def design_seeds(dataset: str, arm: str) -> tuple[int, ...]:
    """Return the seeds the design gives ``(dataset, arm)``; empty if the cell is not run."""
    for cell_dataset, cell_arm, seeds in EXPERIMENT_CELLS:
        if cell_dataset == dataset and cell_arm == arm:
            return tuple(int(s) for s in seeds)
    return ()


def design_runs() -> list[tuple[str, str, str, int]]:
    """Return ``(run_id, dataset, arm, seed)`` for the 30 runs of the design, in report order."""
    runs = [(run_id(d, a, s), d, a, int(s)) for d, a, seeds in EXPERIMENT_CELLS for s in seeds]
    return sorted(runs, key=lambda r: (DATASET_ORDER.index(r[1]), ARM_ORDER.index(r[2]), r[3]))


def arm_label(arm: str) -> str:
    """Markdown label of an arm (``A2p`` is the paper's A2′)."""
    return "A2′" if arm == "A2p" else arm


# --------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Results:
    """One collected ``results/`` folder, as the analysis sees it.

    Parameters
    ----------
    root : Path
        The folder.
    frame : pandas.DataFrame
        One row per design run (30), indexed by ``run_id``: identity, ``present``, every metric
        column of ``index.csv``, the low-band inherited shares and the ``T_tau`` columns of
        :func:`t_tau_frame`. Metrics of an absent run are ``NaN``.
    collection : dict
        ``collection.json``.
    gates : tuple[dict, ...]
        The gate records of ``gates/``, each with ``run_id``, ``early`` and ``late`` added.
    steps : tuple[int, ...]
        The evaluated checkpoint steps.
    constants_label : str
        The D23 constants file the inherited-band columns were read against (repository-relative
        when it lives in the repository).
    inherited_notes : tuple[str, ...]
        Why a present run has no D23 value (no constants for its dataset, sigma or sha256).
    """

    root: Path
    frame: pd.DataFrame
    collection: dict[str, Any]
    gates: tuple[dict[str, Any], ...]
    steps: tuple[int, ...]
    constants_label: str = "none given"
    inherited_notes: tuple[str, ...] = ()

    @property
    def present(self) -> frozenset[str]:
        """Run ids that the collection holds evaluated."""
        return frozenset(self.frame.index[self.frame["present"]])

    @property
    def missing(self) -> tuple[str, ...]:
        """Design run ids that the collection lacks, in report order."""
        return tuple(r for r in self.frame.index if not self.frame.at[r, "present"])

    @property
    def last_step_column(self) -> str:
        """The ``lsd_<step>`` column of the last checkpoint."""
        return f"lsd_{max(self.steps):06d}"

    def value(self, run: str, key: str) -> float:
        """Return one metric of one run as a float (``NaN`` when absent)."""
        value = self.frame.at[run, key]
        return float(value) if value is not None else math.nan


def _read_json(path: Path) -> Any:
    try:
        return strict_json(path.read_bytes())
    except (OSError, ValueError) as error:
        raise AnalysisError(f"{path}: {error}") from error


def _read_index(path: Path) -> pd.DataFrame:
    index = pd.read_csv(path, dtype={"run_id": str, "dataset": str, "arm": str})
    absent = [c for c in _INDEX_REQUIRED if c not in index.columns]
    if absent:
        raise AnalysisError(f"{path}: missing columns {absent}")
    if index["run_id"].duplicated().any():
        raise AnalysisError(f"{path}: a run_id appears twice")
    return index.set_index("run_id", drop=False)


def _check_identity(index: pd.DataFrame, path: Path) -> None:
    design = {r[0]: r for r in design_runs()}
    foreign = sorted(set(index.index) - set(design))
    if foreign:
        raise AnalysisError(f"{path}: runs outside the design {foreign}")
    for rid, row in index.iterrows():
        _, dataset, arm, seed = design[str(rid)]
        if (row["dataset"], row["arm"], int(row["seed"])) != (dataset, arm, seed):
            raise AnalysisError(f"{path}: row {rid} holds identity "
                                f"{row['dataset']}/{row['arm']}/{row['seed']}")


def _same(a: Any, b: Any) -> bool:
    fa, fb = float(a), float(b)
    if math.isnan(fa) or math.isnan(fb):
        return math.isnan(fa) and math.isnan(fb)
    return math.isclose(fa, fb, rel_tol=1e-9, abs_tol=1e-12)


def _summary_frame(root: Path, present: Sequence[str], steps: Sequence[int]) -> pd.DataFrame:
    """The summaries of the present runs through :func:`ihdm.stats.cell_table`."""
    wanted = [*_SUMMARY_KEYS.values(), *_SUMMARY_ONLY.values(),
              *(f"lsd_by_step.{s}" for s in steps)]
    if not present:
        return pd.DataFrame(columns=wanted, dtype=float)
    summaries = {rid: _read_json(root / "runs" / rid / "summary.json") for rid in present}
    frame = cell_table(summaries).set_index("run_id")
    for key in wanted:
        if key not in frame.columns:
            frame[key] = math.nan
    # cell_table keeps numbers only; the D23 constants are matched on this string.
    frame["dataset_sha256"] = [str(summaries[rid].get("dataset_sha256") or "")
                               for rid in frame.index]
    return frame


def _cross_check(index: pd.DataFrame, summaries: pd.DataFrame, steps: Sequence[int]) -> None:
    """Every number of ``index.csv`` must equal the run's own ``summary.json``."""
    pairs = dict(_SUMMARY_KEYS)
    pairs.update({f"lsd_{s:06d}": f"lsd_by_step.{s}" for s in steps})
    problems = []
    for rid in summaries.index:
        for column, key in pairs.items():
            if not _same(index.at[rid, column], summaries.at[rid, key]):
                problems.append(f"{rid}.{column}: index {index.at[rid, column]} "
                                f"!= summary {summaries.at[rid, key]}")
    if problems:
        raise AnalysisError("index.csv disagrees with the runs' summary.json: "
                            + "; ".join(problems[:6]))


def _read_gates(root: Path) -> tuple[dict[str, Any], ...]:
    gates = []
    folder = root / "gates"
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        match = _GATE_FILE.match(path.name)
        if match is None:
            raise AnalysisError(f"{path}: not a <run_id>_gate_<early>_<late>.json name")
        record = dict(_read_json(path))
        record.update({"run_id": match["run_id"], "early": int(match["early"]),
                       "late": int(match["late"]), "file": path.name})
        gates.append(record)
    return tuple(gates)


def _d23_columns(frame: pd.DataFrame, summaries: pd.DataFrame,
                 constants: InheritedConstants) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Add the D23 inherited-band columns (audit §3.3) to the analysis frame.

    ``inherited_within`` is ``I_w = 1 - M/(M-1) D_pix (W^2-1) / sum P_ref``;
    ``inherited_bias`` is ``G_b = 1 - inherited_measured - D_pix (W^2-1) / sum P_ref``;
    ``inherited_bias_expected`` is ``T + (1 - I)/M``; ``inherited_expected`` is ``I - T``, the
    model expectation of the pre-registered share. ``I`` is the run's own
    ``inherited_predicted``. A run whose dataset, ``sigma_max`` or ``dataset_sha256`` has no
    matching constants gets ``NaN`` and a note.
    """
    values: dict[str, list[float]] = {c: [] for c in D23_COLUMNS}
    notes: list[str] = []
    for rid in frame.index:
        row = frame.loc[rid]
        record = dict.fromkeys(D23_COLUMNS, math.nan)
        if bool(row["present"]):
            sha = str(summaries.at[rid, "dataset_sha256"])
            entry, reason = constants.lookup(str(row["dataset"]), sha, float(row["sigma_max"]))
            if entry is None:
                notes.append(reason)
            else:
                share, m = float(row["inherited_predicted"]), float(row["n_per_seed"])
                d_pix, measured = float(row["D_pix"]), float(row["inherited_measured"])
                record = {
                    "inherited_within": within_seed_share(d_pix, m, entry.n_pix, entry.sum_power),
                    "inherited_bias": bias_fraction(measured, d_pix, entry.n_pix,
                                                    entry.sum_power),
                    "inherited_bias_expected": bias_expected(share, entry.mean_term, m),
                    "inherited_mean_term": entry.mean_term,
                    "inherited_expected": share - entry.mean_term,
                }
        for column in D23_COLUMNS:
            values[column].append(record[column])
    for column in D23_COLUMNS:
        frame[column] = values[column]
    unique = tuple(dict.fromkeys(notes))
    for note in unique:
        logger.warning("inherited band (D23): %s; its columns are left blank", note)
    return frame, unique


def load_results(root: str | Path,
                 inherited_constants: str | Path | None = DEFAULT_CONSTANTS_PATH) -> Results:
    """Load and check one ``results/`` folder.

    Parameters
    ----------
    root : str or Path
        The folder written by ``python -m ihdm.cli.collect_results``.
    inherited_constants : str, Path or None
        The D23 constants file (:mod:`ihdm.analysis.inherited`); default the committed
        ``docs/RESULTS/inherited_band_constants.json``. ``None`` leaves the D23 columns blank.

    Returns
    -------
    Results
        The analysis frame, the provenance and the gates.

    Raises
    ------
    ResultsNotFound
        If ``root`` holds no ``collection.json`` or ``index.csv``.
    AnalysisError
        If the collection's verdict is ``FAIL``, a row is outside the design or carries the
        wrong identity, a run is evaluated in ``index.csv`` but has no ``summary.json`` (or the
        reverse), ``index.csv`` disagrees with a run's ``summary.json``, or the constants file
        cannot be read.
    """
    root = Path(root)
    try:
        constants = load_constants(inherited_constants)
    except InheritedError as error:
        raise AnalysisError(str(error)) from error
    for name in ("collection.json", "index.csv"):
        if not (root / name).is_file():
            raise ResultsNotFound(f"{root} is not a results folder: no {name}")
    collection = _read_json(root / "collection.json")
    if collection.get("verdict") == "FAIL":
        raise AnalysisError(f"{root}: the collection's verdict is FAIL; its numbers are unchecked")
    steps = tuple(int(s) for s in collection.get("evaluated_steps") or ())
    if not steps:
        raise AnalysisError(f"{root}/collection.json lists no evaluated_steps")

    index = _read_index(root / "index.csv")
    _check_identity(index, root / "index.csv")
    for step in steps:
        if f"lsd_{step:06d}" not in index.columns:
            raise AnalysisError(f"{root}/index.csv has no lsd_{step:06d} column")

    evaluated = {rid for rid in index.index if not pd.isna(index.at[rid, "lsd_final"])}
    with_summary = {rid for rid in index.index if (root / "runs" / rid / "summary.json").is_file()}
    if evaluated != with_summary:
        raise AnalysisError(f"{root}: index.csv and runs/ disagree on which runs are evaluated: "
                            f"{sorted(evaluated ^ with_summary)}")

    summaries = _summary_frame(root, sorted(evaluated), steps)
    _cross_check(index, summaries, steps)

    runs = design_runs()
    frame = pd.DataFrame(runs, columns=["run_id", "dataset", "arm", "seed"]).set_index(
        "run_id", drop=False)
    frame["present"] = [rid in evaluated for rid in frame.index]
    metric_columns = [c for c in index.columns if c not in {"run_id", "dataset", "arm", "seed"}]
    frame = frame.join(index[metric_columns], how="left")
    for column, key in _SUMMARY_ONLY.items():
        frame[column] = [float(summaries.at[r, key]) if r in summaries.index else math.nan
                         for r in frame.index]
    frame, notes = _d23_columns(frame, summaries, constants)
    frame = pd.concat([frame, t_tau_frame(frame, steps)], axis=1)
    return Results(root=root, frame=frame, collection=collection, gates=_read_gates(root),
                   steps=steps, constants_label=constants.label, inherited_notes=notes)


# --------------------------------------------------------------------------------------------
# T_tau
# --------------------------------------------------------------------------------------------


def t_settle(lsd_by_step: Mapping[int, float], threshold: float) -> int | None:
    """First checkpoint after which the LSD stays at or below ``threshold`` to the last step.

    **Exploratory, defined after seeing the data (2026-09-29), not pre-registered.** The
    pre-registered ``T_tau`` (first crossing) fires at the first checkpoint on these curves,
    because the LSD is low at 5k, rises over 10k-30k and falls again; the settling step is the
    descriptive alternative ``main`` asked for, reported without CI or p-value.

    Parameters
    ----------
    lsd_by_step : Mapping[int, float]
        LSD keyed by checkpoint step.
    threshold : float
        The target LSD.

    Returns
    -------
    int or None
        The step, or ``None`` when the last checkpoint is above the threshold.

    Raises
    ------
    AnalysisError
        If the curve is empty or holds a non-finite value.
    """
    items = sorted((int(s), float(v)) for s, v in lsd_by_step.items())
    if not items or not all(math.isfinite(v) for _, v in items):
        raise AnalysisError("t_settle needs a non-empty finite curve")
    settle = None
    for step, value in reversed(items):
        if value > float(threshold):
            break
        settle = step
    return settle


def t_tau_frame(frame: pd.DataFrame, steps: Sequence[int]) -> pd.DataFrame:
    """``T_tau`` of every run (``05-metrics.md`` §2) under both thresholds.

    For run ``(dataset, arm, seed)`` the curve is ``lsd_<step>`` over ``steps`` and the threshold
    is taken from the A0 run of the same dataset and seed: its LSD at the last checkpoint of the
    same curve estimator (primary; 500 frozen seeds, common random numbers) or its 2,000-sample
    final LSD (sensitivity). ``T_tau`` is the first step whose LSD is at or below the threshold,
    ``inf`` when none is ("not reached"), and ``NaN`` when the run or its A0 run is absent. The
    exploratory settling step (:func:`t_settle`) uses the primary threshold.

    Parameters
    ----------
    frame : pandas.DataFrame
        Indexed by run id, with ``dataset``, ``seed``, ``present``, ``lsd_final`` and the
        ``lsd_<step>`` columns.
    steps : Sequence[int]
        The evaluated steps.

    Returns
    -------
    pandas.DataFrame
        Same index; columns ``t_tau``, ``t_tau_threshold``, ``t_tau_2k``,
        ``t_tau_threshold_2k``, ``t_settle`` (exploratory) and ``t_tau_reference`` (the A0
        run id).
    """
    last = f"lsd_{max(steps):06d}"
    records = []
    for _, row in frame.iterrows():
        reference = run_id(row["dataset"], REFERENCE_ARM, int(row["seed"]))
        record: dict[str, Any] = {"t_tau": math.nan, "t_tau_threshold": math.nan,
                                  "t_tau_2k": math.nan, "t_tau_threshold_2k": math.nan,
                                  "t_settle": math.nan, "t_tau_reference": reference}
        usable = (reference in frame.index and bool(row["present"])
                  and bool(frame.at[reference, "present"]))
        if usable:
            curve = {int(s): float(row[f"lsd_{s:06d}"]) for s in steps}
            for suffix, source in (("", last), ("_2k", "lsd_final")):
                threshold = float(frame.at[reference, source])
                step = t_tau(curve, threshold)
                record[f"t_tau{suffix}"] = math.inf if step is None else float(step)
                record[f"t_tau_threshold{suffix}"] = threshold
            settle = t_settle(curve, record["t_tau_threshold"])
            record["t_settle"] = math.inf if settle is None else float(settle)
        records.append(record)
    return pd.DataFrame(records, index=frame.index)


# --------------------------------------------------------------------------------------------
# Contrasts and the interaction
# --------------------------------------------------------------------------------------------


def _incomplete(n_present: int, n_required: int) -> str:
    return f"incomplete: {n_present}/{n_required} runs"


@dataclass(frozen=True)
class ContrastResult:
    """A seed-paired contrast ``m(arm) - m(reference)`` on one dataset and endpoint.

    Parameters
    ----------
    dataset, arm, reference, endpoint : str
        What is contrasted.
    required : tuple[str, ...]
        The runs the contrast needs (both cells, over the paired seeds).
    missing : tuple[str, ...]
        Those of them the collection lacks.
    seeds : tuple[int, ...]
        The paired seeds.
    deltas : tuple[float, ...]
        Per-seed ``arm - reference``; empty unless the status is ``"ok"`` (or only the
        permutation-free sign summary was asked for).
    reference_mean : float
        Mean of the reference cell over the paired seeds (``NaN`` if unavailable).
    mean, ci_low, ci_high : float
        The mean paired difference and its percentile bootstrap interval (``NaN`` if not
        computed).
    p_value, p_min : float
        Two-sided permutation p-value and the floor the design allows (``NaN`` if not
        computed).
    n_assignments : int
        Assignments the permutation test enumerated (0 if not computed).
    status : str
        ``"ok"``, ``"incomplete: n/N runs"`` or ``"not computable: <reason>"``.
    """

    dataset: str
    arm: str
    reference: str
    endpoint: str
    required: tuple[str, ...]
    missing: tuple[str, ...]
    seeds: tuple[int, ...]
    deltas: tuple[float, ...] = ()
    reference_mean: float = math.nan
    mean: float = math.nan
    ci_low: float = math.nan
    ci_high: float = math.nan
    p_value: float = math.nan
    p_min: float = math.nan
    n_assignments: int = 0
    status: str = STATUS_OK

    @property
    def ok(self) -> bool:
        """Whether the contrast was computed."""
        return self.status == STATUS_OK

    @property
    def excludes_zero(self) -> bool | None:
        """Whether the interval lies strictly on one side of zero (``None`` if not computed)."""
        if math.isnan(self.ci_low):
            return None
        return self.ci_low > 0.0 or self.ci_high < 0.0

    @property
    def verdict(self) -> str:
        """``05-metrics.md`` §8 wording of the interval, or the status when not computed."""
        if self.excludes_zero is None:
            return self.status
        return DETECTABLE if self.excludes_zero else NOT_DETECTABLE

    @property
    def label(self) -> str:
        """``A3−A0``-style label."""
        return f"{arm_label(self.arm)}−{arm_label(self.reference)}"

    def to_json(self) -> dict[str, Any]:
        """Return the record written into ``tables.json``."""
        return {
            "dataset": self.dataset, "arm": self.arm, "reference": self.reference,
            "endpoint": self.endpoint, "seeds": list(self.seeds), "deltas": list(self.deltas),
            "reference_mean": self.reference_mean, "mean": self.mean,
            "ci_low": self.ci_low, "ci_high": self.ci_high, "excludes_zero": self.excludes_zero,
            "p_value": self.p_value, "p_min": self.p_min, "n_assignments": self.n_assignments,
            "status": self.status, "verdict": self.verdict, "required": list(self.required),
            "missing": list(self.missing),
        }


def compute_contrast(
    results: Results,
    dataset: str,
    arm: str,
    endpoint: str,
    reference: str = REFERENCE_ARM,
    seeds: Iterable[int] | None = None,
    inference: bool = True,
) -> ContrastResult:
    """Seed-paired contrast of one endpoint between two arms of one dataset (``05`` §8).

    Parameters
    ----------
    results : Results
        The loaded folder.
    dataset, arm, endpoint : str
        What is contrasted; ``endpoint`` is a column of ``results.frame``.
    reference : str
        The arm subtracted (A0 unless the table says otherwise).
    seeds : Iterable[int] or None
        Restrict the paired seeds (e.g. to the two seeds of A1/A2); default all seeds the design
        gives both cells.
    inference : bool
        ``False`` computes the per-seed differences and means only (the transfer table reports
        signs and magnitudes, no p-values).

    Returns
    -------
    ContrastResult
        The contrast; its ``status`` says why nothing was computed when a run is missing or a
        value is not finite (``T_tau`` "not reached").

    Raises
    ------
    AnalysisError
        If the design pairs no seed of the two cells.
    """
    paired = sorted(set(design_seeds(dataset, arm)) & set(design_seeds(dataset, reference)))
    if seeds is not None:
        paired = sorted(set(paired) & {int(s) for s in seeds})
    if not paired:
        raise AnalysisError(f"no paired seeds for {dataset} {arm} vs {reference}")
    required = tuple(run_id(dataset, a, s) for a in (arm, reference) for s in paired)
    missing = tuple(r for r in required if r not in results.present)
    base = dict(dataset=dataset, arm=arm, reference=reference, endpoint=endpoint,
                required=required, missing=missing, seeds=tuple(paired))
    if missing:
        return ContrastResult(**base, status=_incomplete(len(required) - len(missing),
                                                         len(required)))
    a = {s: results.value(run_id(dataset, arm, s), endpoint) for s in paired}
    b = {s: results.value(run_id(dataset, reference, s), endpoint) for s in paired}
    # short "A2 s1" names keep the reason inside a table cell; the run ids are in ``required``
    not_reached = [f"{arm_label(arm_)} s{s}" for arm_, values in ((arm, a), (reference, b))
                   for s, v in values.items() if math.isinf(v)]
    if not_reached:
        return ContrastResult(**base, status=f"not computable: T_τ not reached "
                                             f"({', '.join(not_reached)})")
    absent = [f"{arm_label(arm_)} s{s}" for arm_, values in ((arm, a), (reference, b))
              for s, v in values.items() if math.isnan(v)]
    if absent:
        return ContrastResult(**base, status=f"not computable: no value ({', '.join(absent)})")

    deltas = tuple(a[s] - b[s] for s in paired)
    reference_mean = float(np.mean([b[s] for s in paired]))
    if not inference:
        return ContrastResult(**base, deltas=deltas, reference_mean=reference_mean,
                              mean=float(np.mean(deltas)))
    delta = paired_delta(a, b, n_boot=N_BOOT, alpha=ALPHA, rng_seed=RNG_SEED)
    test = permutation_test([a[s] for s in paired], [b[s] for s in paired])
    return ContrastResult(
        **base, deltas=deltas, reference_mean=reference_mean, mean=delta.mean,
        ci_low=delta.interval.low, ci_high=delta.interval.high, p_value=test.p_value,
        p_min=test.p_min, n_assignments=test.n_assignments,
    )


@dataclass(frozen=True)
class InteractionResult:
    """``Delta_MRI - Delta_photo`` of one endpoint (``05-metrics.md`` §8).

    Parameters
    ----------
    endpoint : str
        The endpoint.
    mri, photo : ContrastResult
        The two paired contrasts it is built from.
    point, ci_low, ci_high : float
        The interaction and its bootstrap interval (``NaN`` if not computed).
    p_value, p_min : float
        Permutation p-value of the data-type label over the pooled per-seed differences, and
        its floor.
    n_assignments : int
        Assignments enumerated.
    status : str
        ``"ok"`` or why it was not computed.
    """

    endpoint: str
    mri: ContrastResult
    photo: ContrastResult
    point: float = math.nan
    ci_low: float = math.nan
    ci_high: float = math.nan
    p_value: float = math.nan
    p_min: float = math.nan
    n_assignments: int = 0
    status: str = STATUS_OK

    @property
    def excludes_zero(self) -> bool | None:
        """Whether the interval lies strictly on one side of zero (``None`` if not computed)."""
        if math.isnan(self.ci_low):
            return None
        return self.ci_low > 0.0 or self.ci_high < 0.0

    @property
    def verdict(self) -> str:
        """``05-metrics.md`` §8 wording of the interval, or the status when not computed."""
        if self.excludes_zero is None:
            return self.status
        return DETECTABLE if self.excludes_zero else NOT_DETECTABLE

    def to_json(self) -> dict[str, Any]:
        """Return the record written into ``tables.json``."""
        return {
            "endpoint": self.endpoint, "point": self.point, "ci_low": self.ci_low,
            "ci_high": self.ci_high, "excludes_zero": self.excludes_zero,
            "p_value": self.p_value, "p_min": self.p_min, "n_assignments": self.n_assignments,
            "status": self.status, "verdict": self.verdict, "mri": self.mri.to_json(),
            "photo": self.photo.to_json(),
        }


def compute_interaction(
    results: Results,
    endpoint: str,
    arm: str = "A3",
    mri: str = MRI_DATASET,
    photo: str = PHOTO_DATASET,
) -> InteractionResult:
    """The interaction ``Delta_MRI - Delta_photo`` of ``arm`` against A0 for one endpoint.

    Parameters
    ----------
    results : Results
        The loaded folder.
    endpoint : str
        A column of ``results.frame``.
    arm : str
        The arm contrasted against A0 on both datasets.
    mri, photo : str
        The two datasets (the development pair by default).

    Returns
    -------
    InteractionResult
        The estimate, its interval (:func:`ihdm.stats.interaction`) and the permutation test of
        the data-type label over the pooled per-seed differences.
    """
    c_mri = compute_contrast(results, mri, arm, endpoint)
    c_photo = compute_contrast(results, photo, arm, endpoint)
    if not (c_mri.ok and c_photo.ok):
        required = len(c_mri.required) + len(c_photo.required)
        missing = len(c_mri.missing) + len(c_photo.missing)
        status = (_incomplete(required - missing, required) if missing
                  else f"not computable: {c_mri.status if not c_mri.ok else c_photo.status}")
        return InteractionResult(endpoint=endpoint, mri=c_mri, photo=c_photo, status=status)
    estimate = interaction(c_mri.deltas, c_photo.deltas, n_boot=N_BOOT, alpha=ALPHA,
                           rng_seed=RNG_SEED)
    test = permutation_test(c_mri.deltas, c_photo.deltas)
    return InteractionResult(
        endpoint=endpoint, mri=c_mri, photo=c_photo, point=estimate.point,
        ci_low=estimate.interval.low, ci_high=estimate.interval.high, p_value=test.p_value,
        p_min=test.p_min, n_assignments=test.n_assignments,
    )


# --------------------------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------------------------


def _missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def fmt_num(value: Any, sig: int = 4, signed: bool = False) -> str:
    """Four significant digits, fixed notation between 1e-3 and 1e5, scientific outside."""
    if _missing(value):
        return MISSING
    x = float(value)
    if math.isinf(x):
        return NOT_REACHED
    sign = "+" if signed and x > 0 else ""
    if x == 0.0:
        return "0"
    magnitude = math.floor(math.log10(abs(x)))
    if -3 <= magnitude < 5:
        return f"{sign}{x:.{max(0, sig - 1 - magnitude)}f}"
    return f"{sign}{x:.{sig - 1}e}"


def fmt_step(value: Any, signed: bool = False) -> str:
    """A checkpoint step in thousands (``5k``); ``inf`` is "not reached"."""
    if _missing(value):
        return MISSING
    x = float(value)
    if math.isinf(x):
        return NOT_REACHED
    sign = "+" if signed and x > 0 else ""
    if x == 0.0:
        return "0"
    if float(x).is_integer() and int(x) % 1000 == 0:
        return f"{sign}{int(x) // 1000}k"
    return f"{sign}{x / 1000:.2f}k"


def fmt_value(value: Any, kind: str, signed: bool = False) -> str:
    """Format a value of an endpoint of ``kind`` (``"float"`` or ``"step"``)."""
    return fmt_step(value, signed) if kind == "step" else fmt_num(value, signed=signed)


def fmt_p(value: Any) -> str:
    """A p-value with three decimals."""
    return MISSING if _missing(value) else f"{float(value):.3f}"


def fmt_ci(low: Any, high: Any, kind: str = "float") -> str:
    """``[low, high]``."""
    if _missing(low) or _missing(high):
        return MISSING
    return f"[{fmt_value(low, kind, True)}, {fmt_value(high, kind, True)}]"


def fmt_deltas(deltas: Sequence[float], kind: str) -> str:
    """Per-seed differences joined by `` / `` in seed order."""
    return " / ".join(fmt_value(d, kind, True) for d in deltas) if deltas else MISSING


_TEX_ESCAPES = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_",
                "{": r"\{", "}": r"\}", "$": r"\$", "~": r"\textasciitilde{}",
                "^": r"\textasciicircum{}"}
_TEX_UNICODE = {"−": r"\ensuremath{-}", "′": r"\ensuremath{'}", "Δ": r"\ensuremath{\Delta}",
                "τ": r"\ensuremath{\tau}", "—": "--", "→": r"\ensuremath{\to}",
                "≤": r"\ensuremath{\le}", "σ": r"\ensuremath{\sigma}"}
_SCIENTIFIC = re.compile(r"([+-]?\d\.\d+)e([+-]\d+)")
_NEGATIVE = re.compile(r"(?<![\w.{}])-(?=\d)")


def tex_escape(text: str) -> str:
    """Escape plain text for LaTeX, typesetting minus signs and scientific notation."""
    out = "".join(_TEX_ESCAPES.get(ch, ch) for ch in text)
    out = _SCIENTIFIC.sub(lambda m: rf"\ensuremath{{{m[1]}\times10^{{{int(m[2])}}}}}", out)
    out = _NEGATIVE.sub(r"\\ensuremath{-}", out)
    return "".join(_TEX_UNICODE.get(ch, ch) for ch in out)


# --------------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------------

Formatter = Callable[[Any, Mapping[str, Any]], str]


@dataclass(frozen=True)
class Column:
    """One column of a rendered table.

    Parameters
    ----------
    key : str
        Key of the row record the value comes from.
    header : str
        Markdown header.
    tex : str
        LaTeX header (raw LaTeX).
    fmt : Callable[[Any, Mapping[str, Any]], str] or None
        Formats ``(value, row)``; default ``str`` of the value (``—`` when absent).
    align : {"l", "r"}
        Alignment.
    tex_fmt : Callable[[Any, Mapping[str, Any]], str] or None
        Raw-LaTeX formatter; default the escaped Markdown text.
    """

    key: str
    header: str
    tex: str
    fmt: Formatter | None = None
    align: str = "r"
    tex_fmt: Formatter | None = None

    def text(self, row: Mapping[str, Any]) -> str:
        """The cell text of ``row``."""
        value = row.get(self.key)
        if self.fmt is not None:
            return self.fmt(value, row)
        return MISSING if _missing(value) else str(value)

    def latex(self, row: Mapping[str, Any]) -> str:
        """The LaTeX cell of ``row``."""
        if self.tex_fmt is not None:
            return self.tex_fmt(row.get(self.key), row)
        return tex_escape(self.text(row))


@dataclass
class Table:
    """A report table: raw row records plus how to render them.

    Parameters
    ----------
    name : str
        File stem (``t3_interaction``) and LaTeX label suffix.
    number : str
        The ticket's table number (``"3"``, ``"1a"``).
    title : str
        One-line title.
    columns : tuple[Column, ...]
        The rendered columns.
    rows : list[dict]
        Raw records; ``_group`` starts a new LaTeX block when it changes.
    required : tuple[str, ...]
        The runs (or gates) the table needs.
    available : frozenset[str]
        Those present.
    notes : tuple[str, ...]
        Plain-language notes printed under the table.
    unit : str
        What ``required`` counts (``"runs"`` or ``"gates"``).
    tex_title : str
        Raw-LaTeX title for the caption; default the escaped ``title``.
    extra : dict
        Further records written into ``tables.json`` beside the rows.
    """

    name: str
    number: str
    title: str
    columns: tuple[Column, ...]
    rows: list[dict[str, Any]]
    required: tuple[str, ...]
    available: frozenset[str]
    notes: tuple[str, ...] = ()
    unit: str = "runs"
    tex_title: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def missing(self) -> tuple[str, ...]:
        """Required items the collection lacks."""
        return tuple(r for r in self.required if r not in self.available)

    @property
    def complete(self) -> bool:
        """Whether every required item is present."""
        return not self.missing

    @property
    def status(self) -> str:
        """``complete (N/N runs)`` or the ticket's ``incomplete: n/N runs``."""
        n_required = len(self.required)
        n_present = n_required - len(self.missing)
        if self.complete:
            return f"complete ({n_present}/{n_required} {self.unit})"
        return f"incomplete: {n_present}/{n_required} {self.unit}"

    def to_json(self) -> dict[str, Any]:
        """Return the record written into ``tables.json`` (raw numbers, not text)."""
        return {
            "number": self.number, "title": self.title, "status": self.status,
            "complete": self.complete, "required": list(self.required),
            "missing": list(self.missing), "columns": [c.key for c in self.columns],
            "rows": [_json_safe({k: v for k, v in row.items() if not k.startswith("_")})
                     for row in self.rows],
            "notes": list(self.notes), **_json_safe(self.extra),
        }

    def to_markdown(self, banner: str = "", source: str = "") -> str:
        """Render the table as Markdown with its status line and notes."""
        lines = [f"## Table {self.number} — {self.title}", ""]
        if banner:
            lines += [banner, ""]
        status = f"**Status: {self.status}**"
        if self.missing:
            status += f" (missing: {', '.join(self.missing)})"
        lines += [status, ""]
        lines.append("| " + " | ".join(c.header for c in self.columns) + " |")
        lines.append("|" + "|".join("---:" if c.align == "r" else "---"
                                    for c in self.columns) + "|")
        for row in self.rows:
            cells = [c.text(row).replace("|", "\\|") for c in self.columns]
            lines.append("| " + " | ".join(cells) + " |")
        if self.notes:
            lines += ["", *(f"- {note}" for note in self.notes)]
        if source:
            lines += ["", f"<sub>{source}</sub>"]
        return "\n".join(lines) + "\n"

    def to_latex(self, banner: str = "") -> str:
        """Render the table as a LaTeX ``booktabs`` float (``\\label{tab:<name>}``)."""
        caption = (self.tex_title or tex_escape(self.title)) + "."
        if not self.complete:
            caption += rf" \textbf{{{tex_escape(self.status)}.}}"
        if banner:
            caption += rf" \textbf{{{tex_escape(banner.strip('*'))}}}"
        lines = [
            f"% Table {self.number}: generated by python -m ihdm.cli.analyse (T6.1); do not edit.",
            f"% status: {self.status}",
            *(f"% note: {note}" for note in self.notes),
            r"\begin{table}[htbp]",
            r"\centering",
            r"\small",
            rf"\caption{{{caption}}}",
            rf"\label{{tab:{self.name.replace('_', '-')}}}",
            r"\begin{tabular}{" + "".join(c.align for c in self.columns) + "}",
            r"\toprule",
            " & ".join(c.tex for c in self.columns) + r" \\",
            r"\midrule",
        ]
        group = None
        for position, row in enumerate(self.rows):
            if position and row.get("_group") != group:
                lines.append(r"\midrule")
            group = row.get("_group")
            lines.append(" & ".join(c.latex(row) for c in self.columns) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        return "\n".join(lines) + "\n"


def _json_safe(value: Any) -> Any:
    """NaN becomes ``None`` and ``inf`` (a ``T_tau`` never reached) the string "not reached"."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return NOT_REACHED
    return value


# ---- column formatters -----------------------------------------------------------------------


def _col_num(key: str, header: str, tex: str, signed: bool = False) -> Column:
    return Column(key, header, tex, lambda v, r: fmt_num(v, signed=signed))


def _col_step(key: str, header: str, tex: str) -> Column:
    return Column(key, header, tex, lambda v, r: fmt_step(v))


def _endpoint_value(key: str, signed: bool = True) -> Formatter:
    """Format ``row[key]`` in the unit of the row's endpoint."""
    return lambda v, r: fmt_value(v, ENDPOINT_BY_KEY[r["endpoint"]].kind, signed)


def _endpoint_ci(low: str, high: str) -> Formatter:
    return lambda v, r: fmt_ci(r.get(low), r.get(high), ENDPOINT_BY_KEY[r["endpoint"]].kind)


def _endpoint_deltas(key: str) -> Formatter:
    return lambda v, r: fmt_deltas(r.get(key) or (), ENDPOINT_BY_KEY[r["endpoint"]].kind)


def _estimate(prefix: str) -> Formatter:
    """``mean [low, high]`` of a contrast stored under ``<prefix>_mean`` etc."""
    def render(value: Any, row: Mapping[str, Any]) -> str:
        kind = ENDPOINT_BY_KEY[row["endpoint"]].kind
        if _missing(row.get(f"{prefix}_mean")):
            return row.get(f"{prefix}_status") or MISSING
        mean = fmt_value(row[f"{prefix}_mean"], kind, True)
        return f"{mean} {fmt_ci(row.get(f'{prefix}_ci_low'), row.get(f'{prefix}_ci_high'), kind)}"
    return render


def _fmt_share(value: Any, row: Mapping[str, Any]) -> str:
    return MISSING if _missing(value) else f"{float(value):.2f}"


def _fmt_relative(value: Any, row: Mapping[str, Any]) -> str:
    return MISSING if _missing(value) else f"{100 * float(value):+.1f}%"


ENDPOINT_COLUMN = Column("endpoint", "endpoint", "endpoint",
                         lambda v, r: ENDPOINT_BY_KEY[v].label, "l",
                         lambda v, r: ENDPOINT_BY_KEY[v].tex)
DATASET_COLUMN = Column("dataset", "dataset", "dataset", lambda v, r: DATASET_LABEL.get(v, v), "l")
ARM_COLUMN = Column("arm", "arm", "arm", lambda v, r: arm_label(v), "l",
                    lambda v, r: tex_escape(arm_label(v)))
SEED_COLUMN = Column("seed", "seed", "seed", lambda v, r: str(int(v)))
STATUS_COLUMN = Column("status", "status", "status", None, "l")


# ---- table 1: the cell table -----------------------------------------------------------------


def _run_rows(results: Results, columns: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for rid, row in results.frame.iterrows():
        record: dict[str, Any] = {"run_id": rid, "dataset": row["dataset"], "arm": row["arm"],
                                  "seed": int(row["seed"]), "_group": row["dataset"]}
        present = bool(row["present"])
        for column in columns:
            value = row.get(column)
            record[column] = (float(value) if present and not _missing(value) else math.nan)
        record["status"] = STATUS_OK if present else "missing: not evaluated"
        rows.append(record)
    return rows


def build_cell_tables(results: Results) -> list[Table]:
    """Table 1 in two parts: fidelity endpoints (1a) and mechanism endpoints (1b)."""
    required = tuple(results.frame.index)
    fidelity = ("lsd_final", results.last_step_column, "t_tau", "kid", "kid_ci_low",
                "kid_ci_high", "fid", "fid_ci_low", "fid_ci_high", "fid_n_reference", "recall",
                "coverage")
    mechanism = ("M", "M_lp", "seed_nn_fraction", "D_pix", "D_lp", "inherited_measured",
                 "inherited_predicted", "inherited_measured_low", "inherited_predicted_low")
    rows_a = _run_rows(results, fidelity)
    for row in rows_a:
        row["lsd_last_ckpt"] = row.pop(results.last_step_column)
    last_k = f"{max(results.steps) // 1000}k"
    table_a = Table(
        name="t1a_cells_fidelity", number="1a",
        title="Cell table, one row per run: spectral and Inception endpoints",
        columns=(DATASET_COLUMN, ARM_COLUMN, SEED_COLUMN,
                 _col_num("lsd_final", "LSD final (2k)", "LSD (2k)"),
                 _col_num("lsd_last_ckpt", f"LSD {last_k} (500)", f"LSD {last_k} (500)"),
                 _col_step("t_tau", "T_τ", r"$T_\tau$"),
                 _col_num("kid", "KID", "KID"),
                 Column("kid_ci_low", "KID 95% CI", r"KID 95\% CI",
                        lambda v, r: fmt_ci(r.get("kid_ci_low"), r.get("kid_ci_high"))),
                 _col_num("fid", "FID", "FID"),
                 Column("fid_ci_low", "FID 95% CI", r"FID 95\% CI",
                        lambda v, r: fmt_ci(r.get("fid_ci_low"), r.get("fid_ci_high"))),
                 Column("fid_n_reference", "N_ref", r"$N_{\mathrm{ref}}$",
                        lambda v, r: MISSING if _missing(v) else str(int(v))),
                 _col_num("recall", "recall", "recall"),
                 _col_num("coverage", "coverage", "coverage"),
                 STATUS_COLUMN),
        rows=rows_a, required=required, available=results.present,
        notes=(
            "LSD final is the 2,000-sample final set (rng_seed 0, with replacement); LSD "
            f"{last_k} (500) is the 500 frozen seeds of the curve at the last checkpoint.",
            "T_τ uses the primary threshold (same-seed A0 LSD at the last checkpoint, 500 "
            "seeds); table 7 gives both thresholds.",
            "KID is the headline Inception metric. FID against N_ref = 800 is biased upward "
            "and its bootstrap interval over resampled samples is shifted upward (T4.3), so "
            "it can lie above the point estimate.",
        ),
    )
    rows_b = _run_rows(results, (*mechanism, "n_skipped"))
    for row, (_, source) in zip(rows_b, results.frame.iterrows(), strict=True):
        row["n_skipped"] = int(source["n_skipped"]) if not _missing(source["n_skipped"]) else None
    table_b = Table(
        name="t1b_cells_mechanism", number="1b",
        title="Cell table, one row per run: memorisation, diversity and inherited band",
        columns=(DATASET_COLUMN, ARM_COLUMN, SEED_COLUMN,
                 _col_num("M", "M", "$M$"), _col_num("M_lp", "M_lp", r"$M_{\mathrm{lp}}$"),
                 _col_num("seed_nn_fraction", "seed-NN frac.", "seed-NN"),
                 _col_num("D_pix", "D_pix", r"$D_{\mathrm{pix}}$"),
                 _col_num("D_lp", "D_lp", r"$D_{\mathrm{lp}}$"),
                 _col_num("inherited_measured", "inherited meas. (pre-reg., biased)",
                          "inh. meas. (pre-reg.)"),
                 _col_num("inherited_predicted", "inherited pred.", "inh. pred."),
                 _col_num("inherited_measured_low", "low-band meas. (pre-reg., biased)",
                          "low meas. (pre-reg.)"),
                 _col_num("inherited_predicted_low", "low-band pred.", "low pred."),
                 Column("n_skipped", "n_skipped", r"$n_{\mathrm{skip}}$"),
                 STATUS_COLUMN),
        rows=rows_b, required=required, available=results.present,
        notes=(
            "M and M_lp: median nearest-training-image distance of the 2,000 training-seeded "
            "samples over that of held-out real images (05 §4); 1 means as far as new data.",
            "D_pix / D_lp: within-seed diversity over 40 held-out seeds × 50 samples (05 §3).",
            "Inherited share over all non-DC modes, measured 1 − ΣV/ΣP_ref "
            f"{PREREGISTERED_LABEL}, beside the linear-Gaussian prediction I; the low-band "
            "columns mask both to σ_n ≥ 8 px (05 §5). The corrected reading is table 1c.",
            "n_skipped counts the no-op fp16 overflow steps kept in the canonical history "
            "(run 11, lsun_church_A3_s3: 6, all before its resume at 30,001).",
        ),
    )
    return [table_a, table_b, _build_inherited_cell_table(results, required)]


#: The D23 note shared by every table that shows an inherited-band endpoint.
_D23_NOTE = ("Inherited band (D23, inherited_band_audit.md): the pre-registered share "
             "1 − ΣV/ΣP_ref is biased under a non-zero mean image, its model expectation is "
             "I − T with T = Σ(1−d)²μ²/ΣP_ref. I_w = 1 − M/(M−1)·D_pix(W²−1)/ΣP_ref is the "
             "within-seed share (expectation I whatever the mean image) and G_b = 1 − "
             "pre-registered share − D_pix(W²−1)/ΣP_ref the seed-mean bias fraction "
             "(expectation T + (1−I)/M). ΣP_ref and T come from the ref split "
             "(inherited_band_constants.json).")


def _build_inherited_cell_table(results: Results, required: tuple[str, ...]) -> Table:
    """Table 1c: the D23 reading of the inherited band, one row per run."""
    rows = _run_rows(results, ("inherited_measured", "inherited_expected", "inherited_within",
                               "inherited_predicted", "inherited_bias",
                               "inherited_bias_expected", "inherited_mean_term", "n_per_seed"))
    notes = [
        _D23_NOTE,
        "Read each measured column against the expectation on its right: pre-registered share "
        "against I − T, I_w against I, G_b against T + (1−I)/M. The model is the "
        "linear-Gaussian one of 05 §5 written with a mean image; an I_w above I means the "
        "samples of one seed vary less than the variance the blur removed (under-dispersion), "
        "not that more of the seed is inherited.",
        f"M = n_per_seed samples per seed (final.json); W² − 1 non-DC modes; constants file: "
        f"{results.constants_label}.",
    ]
    notes += [f"No D23 value: {note}." for note in results.inherited_notes]
    return Table(
        name="t1c_cells_inherited", number="1c",
        title="Cell table, one row per run: the inherited band read as D23 prescribes",
        columns=(DATASET_COLUMN, ARM_COLUMN, SEED_COLUMN,
                 _col_num("inherited_measured", "pre-reg. share (biased)",
                          "pre-reg. (biased)", signed=True),
                 _col_num("inherited_expected", "expected I − T", r"exp.\ $I-T$", signed=True),
                 _col_num("inherited_within", "I_w", "$I_w$"),
                 _col_num("inherited_predicted", "expected I", r"exp.\ $I$"),
                 _col_num("inherited_bias", "G_b", "$G_b$"),
                 _col_num("inherited_bias_expected", "expected T + (1−I)/M",
                          r"exp.\ $T+(1-I)/M$"),
                 _col_num("inherited_mean_term", "T", "$T$"),
                 Column("n_per_seed", "M", "$M$",
                        lambda v, r: MISSING if _missing(v) else str(int(v))),
                 STATUS_COLUMN),
        rows=rows, required=required, available=results.present, notes=tuple(notes),
    )


# ---- table 2: paired contrasts against A0 -----------------------------------------------------


def _contrast_row(contrast: ContrastResult, group: str) -> dict[str, Any]:
    return {"_group": group, "contrast": contrast.label, "endpoint": contrast.endpoint,
            "n_seeds": len(contrast.seeds), "mean": contrast.mean, "ci_low": contrast.ci_low,
            "ci_high": contrast.ci_high, "deltas": list(contrast.deltas),
            "p_value": contrast.p_value, "p_min": contrast.p_min,
            "n_assignments": contrast.n_assignments, "verdict": contrast.verdict,
            "status": contrast.status, "reference_mean": contrast.reference_mean,
            "seeds": list(contrast.seeds)}


_CONTRAST_COLUMNS: tuple[Column, ...] = (
    Column("contrast", "contrast", "contrast", None, "l"),
    ENDPOINT_COLUMN,
    Column("n_seeds", "seeds", "$s$"),
    Column("mean", "mean Δ", r"mean $\Delta$", _endpoint_value("mean")),
    Column("ci_low", "95% CI", r"95\% CI", _endpoint_ci("ci_low", "ci_high")),
    Column("deltas", "per-seed Δ (s1 / s2 / s3)", r"per-seed $\Delta$", _endpoint_deltas("deltas"),
           "l"),
    Column("p_value", "perm. p", "perm.\\ $p$", lambda v, r: fmt_p(v)),
    Column("p_min", "p_min", r"$p_{\min}$", lambda v, r: fmt_p(v)),
    Column("verdict", "reading", "reading", None, "l"),
)


def build_contrast_tables(results: Results) -> list[Table]:
    """Table 2: paired contrasts against A0 on each development dataset."""
    tables = []
    for letter, (dataset, arms) in zip("ab", DEVELOPMENT_ARMS.items(), strict=True):
        rows = [
            _contrast_row(compute_contrast(results, dataset, arm, e.key), arm)
            for arm in arms for e in ENDPOINTS
        ]
        required = tuple(run_id(dataset, arm, s) for arm in ("A0", *arms)
                         for s in design_seeds(dataset, arm))
        tables.append(Table(
            name=f"t2{letter}_contrasts_{dataset}", number=f"2{letter}",
            title=f"Paired contrasts against A0, {DATASET_LABEL[dataset]}",
            columns=_CONTRAST_COLUMNS, rows=rows, required=required, available=results.present,
            notes=_CONTRAST_NOTES + ((_CHURCHES_NOTE,) if dataset == PHOTO_DATASET else ()),
        ))
    return tables


_CONTRAST_NOTES: tuple[str, ...] = (
    "Δ = arm − A0, paired by seed. CI: percentile bootstrap over seeds, 10,000 draws. With 3 "
    "seeds the 95% interval is exactly [min, max] of the three per-seed Δ, so 'CI excludes 0' "
    "means 'all three seeds agree in sign' (coverage 75% under a symmetric null); with 2 seeds "
    "it is [min, max] of two values (coverage 50%).",
    "p: exact permutation of arm labels over the pooled seeds of the two cells; its floor p_min "
    "is 0.1 with 3 vs 3 seeds (20 assignments) and 1/3 with 2 vs 2 (6 assignments).",
    "LSD CIs resample seeds only: the samples themselves are not in results/, so the "
    "sample-level resampling of 05 §8 is not applied.",
    "T_τ rows are 'not computable' when a seed never reaches the threshold; nothing is imputed.",
    _D23_NOTE,
)
_CHURCHES_NOTE = ("Churches is undertrained (final LSD ≈ 1.0–1.45, oscillating across "
                  "checkpoints, blurry samples); its contrasts are reported with that caveat.")


# ---- table 3: the interaction -----------------------------------------------------------------


def build_interaction_table(results: Results) -> Table:
    """Table 3, the headline: ``Delta_IXI - Delta_Churches`` for A3 vs A0, per endpoint."""
    rows = []
    for endpoint in ENDPOINTS:
        result = compute_interaction(results, endpoint.key)
        rows.append({
            "endpoint": endpoint.key, "delta_mri": result.mri.mean,
            "deltas_mri": list(result.mri.deltas), "delta_photo": result.photo.mean,
            "deltas_photo": list(result.photo.deltas), "point": result.point,
            "ci_low": result.ci_low, "ci_high": result.ci_high, "p_value": result.p_value,
            "p_min": result.p_min, "n_assignments": result.n_assignments,
            "verdict": result.verdict, "status": result.status,
            "relative_mri": _relative(endpoint, result.mri),
            "relative_photo": _relative(endpoint, result.photo),
        })
    required = tuple(run_id(d, a, s) for d in (MRI_DATASET, PHOTO_DATASET) for a in ("A0", "A3")
                     for s in design_seeds(d, a))
    return Table(
        name="t3_interaction", number="3",
        title="Interaction: Δ_IXI − Δ_Churches for A3 vs A0 (the headline)",
        tex_title=(r"Interaction $\Delta_{\mathrm{IXI}}-\Delta_{\mathrm{Churches}}$ for A3 vs "
                   r"A0 (the headline)"),
        columns=(
            ENDPOINT_COLUMN,
            Column("delta_mri", "Δ IXI (A3−A0)", r"$\Delta_{\mathrm{IXI}}$",
                   _endpoint_value("delta_mri")),
            Column("relative_mri", "Δ IXI / A0", "rel.", _fmt_relative),
            Column("delta_photo", "Δ Churches (A3−A0)", r"$\Delta_{\mathrm{Churches}}$",
                   _endpoint_value("delta_photo")),
            Column("relative_photo", "Δ Churches / A0", "rel.", _fmt_relative),
            Column("point", "Δ_IXI − Δ_Churches", r"$\Delta_{\mathrm{IXI}}-\Delta_{\mathrm{Ch}}$",
                   _endpoint_value("point")),
            Column("ci_low", "95% CI", r"95\% CI", _endpoint_ci("ci_low", "ci_high")),
            Column("p_value", "perm. p", "perm.\\ $p$", lambda v, r: fmt_p(v)),
            Column("p_min", "p_min", r"$p_{\min}$", lambda v, r: fmt_p(v)),
            Column("verdict", "reading", "reading", None, "l"),
        ),
        rows=rows, required=required, available=results.present,
        notes=(
            "Δ = A3 − A0 paired by seed (3 seeds on each dataset). The interaction CI resamples "
            "the three IXI and the three Churches per-seed Δ independently (10,000 draws, "
            "percentile). A negative value on a lower-is-better endpoint (LSD, T_τ, KID, FID) "
            "means A3 helps IXI more than Churches, the direction the claim predicts.",
            "The interaction is on each metric's raw scale, as pre-registered; where the two "
            "datasets' A0 levels differ (Churches' LSD is about five times IXI's) the raw "
            "difference is dominated by the larger scale. 'Δ / A0' (mean Δ over the mean A0 "
            "value, same seeds) is a descriptive, scale-free reading and carries no test; it is "
            "left blank for recall, coverage, the seed-NN fraction and the inherited share, "
            "which are proportions near 0 or signed.",
            "p: exact permutation of the dataset label over the six pooled per-seed Δ "
            "(C(6,3) = 20 assignments), so p cannot fall below p_min = 0.1.",
            "Only the development pair enters: OASIS-1 and Bedrooms (2 seeds, A0/A3 only) are "
            "in the transfer table 6, without p-values.",
            _D23_NOTE + " On the pre-registered share the interaction mixes ΔI with ΔT, which "
            "moves with σ_B,max on MRI (a property of the data, audit §3.5).",
            _CHURCHES_NOTE,
        ),
    )


# ---- table 4: decomposition -------------------------------------------------------------------


def _share(numerator: float, denominator: float) -> float:
    if _missing(numerator) or _missing(denominator) or denominator == 0.0:
        return math.nan
    return float(numerator) / float(denominator)


def _relative(endpoint: Endpoint, contrast: ContrastResult) -> float:
    """Descriptive ``mean Delta / mean A0`` on ratio-scale endpoints with a positive A0 mean."""
    if not endpoint.ratio or _missing(contrast.reference_mean) or contrast.reference_mean <= 0:
        return math.nan
    return _share(contrast.mean, contrast.reference_mean)


def build_decomposition_table(results: Results) -> Table:
    """Table 4: the share of the A3 effect carried by A1 (terminal blur) and A2 (spacing)."""
    rows = []
    required: list[str] = []
    for dataset in DEVELOPMENT_ARMS:
        seeds = sorted(set(design_seeds(dataset, "A1")) & set(design_seeds(dataset, "A2")))
        a3_full = {e.key: compute_contrast(results, dataset, "A3", e.key) for e in ENDPOINTS}
        for endpoint in ENDPOINTS:
            c3 = compute_contrast(results, dataset, "A3", endpoint.key, seeds=seeds,
                                  inference=False)
            c1 = compute_contrast(results, dataset, "A1", endpoint.key, seeds=seeds,
                                  inference=False)
            c2 = compute_contrast(results, dataset, "A2", endpoint.key, seeds=seeds,
                                  inference=False)
            share_1, share_2 = _share(c1.mean, c3.mean), _share(c2.mean, c3.mean)
            full = a3_full[endpoint.key]
            flags = [c.status for c in (c3, c1, c2, full) if not c.ok]
            if not flags and full.excludes_zero is False:
                flags.append("A3−A0 not detectable (table 2): shares not interpretable")
            rows.append({
                "_group": dataset, "dataset": dataset, "endpoint": endpoint.key,
                "seeds": list(seeds), "delta_a3": c3.mean, "delta_a1": c1.mean,
                "delta_a2": c2.mean, "share_a1": share_1, "share_a2": share_2,
                "share_sum": share_1 + share_2, "a3_detectable": full.excludes_zero,
                "status": "; ".join(dict.fromkeys(flags)) if flags else STATUS_OK,
            })
        required += [run_id(dataset, a, s) for a in ("A0", "A3", "A1", "A2")
                     for s in design_seeds(dataset, a)]
    return Table(
        name="t4_decomposition", number="4",
        title="Decomposition of the A3 effect into terminal blur (A1) and spacing (A2)",
        columns=(
            DATASET_COLUMN, ENDPOINT_COLUMN,
            Column("delta_a3", "Δ A3 (s1,s2)", r"$\Delta_{A3}$", _endpoint_value("delta_a3")),
            Column("delta_a1", "Δ A1", r"$\Delta_{A1}$", _endpoint_value("delta_a1")),
            Column("delta_a2", "Δ A2", r"$\Delta_{A2}$", _endpoint_value("delta_a2")),
            Column("share_a1", "share A1", "share A1", _fmt_share),
            Column("share_a2", "share A2", "share A2", _fmt_share),
            Column("share_sum", "A1 + A2", "sum", _fmt_share),
            Column("status", "reading", "reading", None, "l"),
        ),
        rows=rows, required=tuple(required), available=results.present,
        notes=(
            "All Δ are against A0 and use only seeds 1 and 2, the seeds A1 and A2 have, so the "
            "three Δ share their A0 values. share A1 = Δ_A1 / Δ_A3 and share A2 = Δ_A2 / Δ_A3; "
            "a sum near 1 means the two knobs add up to the A3 effect, a sum far from 1 means "
            "they interact. No CI: A1 and A2 have 2 seeds.",
            "A share is flagged 'not interpretable' when the 3-seed A3−A0 interval of table 2 "
            "contains 0: a ratio over an undetectable denominator carries no information.",
        ),
    )


# ---- table 5: the A2' control -----------------------------------------------------------------


def build_a2p_table(results: Results) -> Table:
    """Table 5: A2′ (Churches-matched spacing) against A0 and against A2 on Churches."""
    rows = []
    pairs = (("a2p_a0", "A2p", "A0"), ("a2_a0", "A2", "A0"), ("a2p_a2", "A2p", "A2"))
    for endpoint in ENDPOINTS:
        row: dict[str, Any] = {"endpoint": endpoint.key}
        for prefix, arm, reference in pairs:
            c = compute_contrast(results, PHOTO_DATASET, arm, endpoint.key, reference=reference)
            row.update({f"{prefix}_mean": c.mean, f"{prefix}_ci_low": c.ci_low,
                        f"{prefix}_ci_high": c.ci_high, f"{prefix}_p": c.p_value,
                        f"{prefix}_p_min": c.p_min, f"{prefix}_deltas": list(c.deltas),
                        f"{prefix}_verdict": c.verdict, f"{prefix}_status": c.status})
        rows.append(row)
    required = tuple(run_id(PHOTO_DATASET, a, s) for a in ("A0", "A2", "A2p")
                     for s in design_seeds(PHOTO_DATASET, "A2p"))
    columns = [ENDPOINT_COLUMN]
    for prefix, arm, reference in pairs:
        label = f"{arm_label(arm)}−{arm_label(reference)}"
        columns += [
            Column(f"{prefix}_mean", f"{label}: mean Δ [95% CI]",
                   tex_escape(label) + r" [95\% CI]",
                   _estimate(prefix)),
            Column(f"{prefix}_p", "p", "$p$", lambda v, r: fmt_p(v)),
        ]
    return Table(
        name="t5_a2p_control", number="5",
        title="The A2′ control on Churches: Churches-matched against IXI-matched spacing",
        columns=tuple(columns), rows=rows, required=required, available=results.present,
        notes=(
            "A2′ uses the spacing variance-matched to Churches, A2 the one matched to IXI, both "
            "at σ_B,max = 96. A2′−A2 isolates whether matching the spacing to the data's own "
            "spectrum matters on photographs.",
            "Every contrast is paired over seeds 1 and 2 (the seeds A2 and A2′ have): the CI is "
            "[min, max] of two per-seed Δ and the permutation p has floor p_min = 1/3 "
            "(2 vs 2 seeds, 6 assignments). Nothing in this table can reach p < 0.33.",
            _CHURCHES_NOTE,
        ),
    )


# ---- table 6: transfer signs ------------------------------------------------------------------


def _sign(value: float) -> str:
    if _missing(value):
        return MISSING
    return "+" if value > 0 else ("−" if value < 0 else "0")


def build_transfer_table(results: Results) -> Table:
    """Table 6: the sign of A3−A0 on the development and the transfer dataset of each type."""
    rows = []
    required: list[str] = []
    for development, transfer in TRANSFER_PAIRS:
        for endpoint in ENDPOINTS:
            dev = compute_contrast(results, development, "A3", endpoint.key, inference=False)
            tra = compute_contrast(results, transfer, "A3", endpoint.key, inference=False)
            agree = (None if not (dev.ok and tra.ok)
                     else bool(np.sign(dev.mean) == np.sign(tra.mean)))
            flags = [c.status for c in (dev, tra) if not c.ok]
            rows.append({
                "_group": development, "pair": f"{DATASET_LABEL[development]} → "
                                              f"{DATASET_LABEL[transfer]}",
                "endpoint": endpoint.key, "dev_mean": dev.mean, "dev_deltas": list(dev.deltas),
                "dev_relative": _relative(endpoint, dev),
                "transfer_mean": tra.mean, "transfer_deltas": list(tra.deltas),
                "transfer_relative": _relative(endpoint, tra),
                "dev_sign": _sign(dev.mean), "transfer_sign": _sign(tra.mean),
                "agree": agree, "status": "; ".join(flags) if flags else STATUS_OK,
            })
        required += [run_id(d, a, s) for d in (development, transfer) for a in ("A0", "A3")
                     for s in design_seeds(d, a)]

    return Table(
        name="t6_transfer", number="6",
        title="Transfer: the sign of A3−A0 on the development and the transfer dataset",
        columns=(
            Column("pair", "development → transfer", "pair", None, "l"),
            ENDPOINT_COLUMN,
            Column("dev_mean", "Δ dev (3 seeds)", r"$\Delta$ dev", _endpoint_value("dev_mean")),
            Column("dev_relative", "Δ dev / A0", "rel.", _fmt_relative),
            Column("transfer_mean", "Δ transfer (2 seeds)", r"$\Delta$ transfer",
                   _endpoint_value("transfer_mean")),
            Column("transfer_deltas", "transfer per seed (s1 / s2)", "per seed",
                   _endpoint_deltas("transfer_deltas"), "l"),
            Column("transfer_relative", "Δ transfer / A0", "rel.", _fmt_relative),
            Column("agree", "same sign", "same sign",
                   lambda v, r: r["status"] if v is None else ("yes" if v else "no"), "l"),
        ),
        rows=rows, required=tuple(required), available=results.present,
        notes=(
            "Signs and magnitudes only, no p-values (05 §8): a transfer cell has 2 seeds. "
            "'Δ / A0' divides the mean Δ by the mean A0 value over the same seeds, so the "
            "magnitude can be read across datasets whose metric scales differ; it is left blank "
            "for proportions near 0 and the signed inherited share.",
            "The development Δ uses the 3 seeds of IXI or Churches; the transfer Δ the 2 seeds "
            "of OASIS-1 or Bedrooms. The A3 schedule is the IXI-matched one on every dataset, "
            "transferred frozen (D12).",
            _D23_NOTE,
        ),
    )


# ---- table 7: T_tau ---------------------------------------------------------------------------


def build_t_tau_table(results: Results) -> Table:
    """Table 7: ``T_tau`` of every run under the primary and the sensitivity threshold."""
    frame = results.frame
    first = f"lsd_{min(results.steps):06d}"
    rows = []
    required: list[str] = []
    for rid, row in frame.iterrows():
        reference = row["t_tau_reference"]
        needed = [rid] if rid == reference else [rid, reference]
        required += [r for r in needed if r not in required]
        missing = [r for r in needed if r not in results.present]
        rows.append({
            "_group": row["dataset"], "run_id": rid, "dataset": row["dataset"],
            "arm": row["arm"], "seed": int(row["seed"]), "reference": reference,
            "lsd_first": float(row[first]) if row["present"] else math.nan,
            "lsd_last": float(row[results.last_step_column]) if row["present"] else math.nan,
            "threshold": row["t_tau_threshold"], "t_tau": row["t_tau"],
            "threshold_2k": row["t_tau_threshold_2k"], "t_tau_2k": row["t_tau_2k"],
            "t_settle": row["t_settle"],
            "curve": ({str(s): float(row[f"lsd_{s:06d}"]) for s in results.steps}
                      if row["present"] else {}),
            "status": (STATUS_OK if not missing
                       else f"missing: {', '.join(missing)}"),
        })
    first_k, last_k = f"{min(results.steps) // 1000}k", f"{max(results.steps) // 1000}k"
    return Table(
        name="t7_t_tau", number="7",
        title="T_τ: first checkpoint at which a run's LSD reaches its A0 run's final LSD",
        tex_title=(r"$T_\tau$: first checkpoint at which a run's LSD reaches its A0 run's "
                   r"final LSD"),
        columns=(
            DATASET_COLUMN, ARM_COLUMN, SEED_COLUMN,
            _col_num("lsd_first", f"LSD {first_k}", f"LSD {first_k}"),
            _col_num("lsd_last", f"LSD {last_k}", f"LSD {last_k}"),
            _col_num("threshold", "threshold (A0, 500)", "thr. (500)"),
            _col_step("t_tau", "T_τ", r"$T_\tau$"),
            _col_num("threshold_2k", "threshold (A0, 2k)", "thr. (2k)"),
            _col_step("t_tau_2k", "T_τ, 2k threshold (sensitivity)", r"$T_\tau$ (2k)"),
            _col_step("t_settle", "settling step (exploratory)", "settle (expl.)"),
            STATUS_COLUMN,
        ),
        rows=rows, required=tuple(required), available=results.present,
        notes=(
            "05 §2: 'T_τ(arm) is the smallest checkpoint step s at which LSD_arm(s) ≤ "
            "LSD^A0_final', LSD^A0_final being the A0 run with the same seed at its last "
            "checkpoint; +∞ ('not reached') if never.",
            f"Primary threshold: the A0 run's LSD at {last_k} on the curve's own estimator (the "
            "500 frozen seeds and noise stream of every curve point, D17), so curve and threshold "
            "share one sample set. For A0 itself T_τ ≤ the last step by construction, and it "
            "is earlier when the A0 curve dipped below its final value before the end.",
            "Sensitivity: the A0 run's 2,000-sample final LSD. It differs from the primary "
            "threshold by the set difference (0.003 on IXI A0 s1, as large as the gate's "
            "resolution), which can move T_τ by itself; read the primary column.",
            "Resolution 5k steps (the evaluated checkpoints, D16/D22).",
            f"The settling step is {EXPLORATORY_LABEL}: the first checkpoint after which the "
            "curve stays at or below the primary threshold to the end (table 9).",
        ),
    )


# ---- table 9: the exploratory settling step --------------------------------------------------

EXPLORATORY_LABEL: str = ("exploratory, defined after seeing the data (2026-09-29), "
                          "not pre-registered")


def _per_seed(results: Results, dataset: str, arm: str, seeds: Sequence[int],
              key: str) -> list[float]:
    return [results.value(run_id(dataset, arm, s), key)
            if run_id(dataset, arm, s) in results.present else math.nan for s in seeds]


def _finite_mean(values: Sequence[float]) -> float:
    if not values or not all(math.isfinite(v) for v in values):
        return math.nan
    return float(np.mean(values))


def build_settle_table(results: Results) -> Table:
    """Table 9 (exploratory): the settling step of every arm against A0, descriptively."""
    rows = []
    required: list[str] = []
    for dataset in DATASET_ORDER:
        arms = [a for a in ARM_ORDER if a != REFERENCE_ARM and design_seeds(dataset, a)]
        for arm in arms:
            seeds = sorted(set(design_seeds(dataset, arm)) & set(design_seeds(dataset, "A0")))
            needed = [run_id(dataset, a, s) for a in (arm, REFERENCE_ARM) for s in seeds]
            required += [r for r in needed if r not in required]
            arm_values = _per_seed(results, dataset, arm, seeds, "t_settle")
            a0_values = _per_seed(results, dataset, REFERENCE_ARM, seeds, "t_settle")
            deltas = [a - b for a, b in zip(arm_values, a0_values, strict=True)]
            missing = [r for r in needed if r not in results.present]
            if missing:
                status = _incomplete(len(needed) - len(missing), len(needed))
            elif not all(math.isfinite(d) for d in deltas):
                status = "Δ undefined: a curve ends above its threshold"
            else:
                status = STATUS_OK
            rows.append({
                "_group": dataset, "dataset": dataset, "contrast": f"{arm_label(arm)}−A0",
                "arm": arm, "seeds": seeds, "arm_values": arm_values, "a0_values": a0_values,
                "arm_mean": _finite_mean(arm_values), "a0_mean": _finite_mean(a0_values),
                "mean_delta": _finite_mean(deltas) if status == STATUS_OK else math.nan,
                "status": status,
            })
    by_arm = {(r["dataset"], r["arm"]): r for r in rows}
    ixi, church = by_arm[(MRI_DATASET, "A3")], by_arm[(PHOTO_DATASET, "A3")]
    rows.append({
        "_group": "interaction", "dataset": "interaction",
        "contrast": "(A3−A0) IXI − (A3−A0) Churches", "arm": "A3", "seeds": [],
        "arm_values": [], "a0_values": [], "arm_mean": math.nan, "a0_mean": math.nan,
        "mean_delta": ixi["mean_delta"] - church["mean_delta"],
        "status": STATUS_OK if ixi["status"] == church["status"] == STATUS_OK
        else "; ".join(s for s in (ixi["status"], church["status"]) if s != STATUS_OK),
    })

    def steps(key: str) -> Formatter:
        return lambda v, r: " / ".join(fmt_step(x) for x in r.get(key) or []) or MISSING

    return Table(
        name="t9_settle_exploratory", number="9",
        title=f"Settling step of the LSD curve ({EXPLORATORY_LABEL})",
        columns=(
            Column("dataset", "dataset", "dataset",
                   lambda v, r: DATASET_LABEL.get(v, v), "l"),
            Column("contrast", "contrast", "contrast", None, "l"),
            Column("arm_values", "arm per seed", "arm per seed", steps("arm_values"), "l"),
            Column("a0_values", "A0 per seed", "A0 per seed", steps("a0_values"), "l"),
            _col_step("arm_mean", "arm mean", "arm mean"),
            _col_step("a0_mean", "A0 mean", "A0 mean"),
            Column("mean_delta", "mean Δ", r"mean $\Delta$",
                   lambda v, r: fmt_step(v, signed=True)),
            STATUS_COLUMN,
        ),
        rows=rows, required=tuple(required), available=results.present,
        notes=(
            f"{EXPLORATORY_LABEL.capitalize()}. Settling step: the first checkpoint after which "
            "the run's LSD curve (500 frozen seeds) stays at or below the primary T_τ threshold "
            "(the same-seed A0 run's LSD at the last checkpoint) through the last step.",
            "Descriptive only: per-seed values, means and differences, no CI and no p-value, "
            "so nothing here is confirmatory.",
            "Why it exists: the pre-registered T_τ (first crossing, tables 2, 3 and 7) fires at "
            "the first checkpoint in most runs because the LSD curve is low at 5k, higher over "
            "10k-30k and lower again by the end.",
        ),
    )


# ---- table 8: the gates -----------------------------------------------------------------------


def build_gate_table(results: Results) -> Table:
    """Table 8: the plateau gates of D10/D17 (35k vs 40k) and D22 (55k vs 60k)."""
    rows = []
    for gate in sorted(results.gates, key=lambda g: (g["run_id"], g["early"])):
        difference = gate.get("difference") or {}
        rows.append({
            "run_id": gate["run_id"], "pair": f"{gate['early'] // 1000}k vs "
                                              f"{gate['late'] // 1000}k",
            "early": gate["early"], "late": gate["late"], "lsd_a": gate.get("lsd_a"),
            "lsd_b": gate.get("lsd_b"), "difference": difference.get("point"),
            "ci_low": difference.get("ci_low"), "ci_high": difference.get("ci_high"),
            "relative_change": gate.get("relative_change"), "extend": gate.get("extend"),
            "n_seeds": gate.get("n_seeds"), "amp": gate.get("amp"), "file": gate["file"],
        })
    required, available = _required_gates(results)
    return Table(
        name="t8_gates", number="8", title="Plateau gates (D10/D17, D22)",
        columns=(
            Column("run_id", "run", "run", None, "l"),
            Column("pair", "checkpoints", "checkpoints", None, "l"),
            _col_num("lsd_a", "LSD early", "LSD early"),
            _col_num("lsd_b", "LSD late", "LSD late"),
            _col_num("difference", "early − late", "early $-$ late", signed=True),
            Column("ci_low", "95% CI", r"95\% CI",
                   lambda v, r: fmt_ci(r.get("ci_low"), r.get("ci_high"))),
            Column("relative_change", "relative", "rel.",
                   lambda v, r: MISSING if _missing(v) else f"{100 * float(v):+.2f}%"),
            Column("extend", "extend", "extend", lambda v, r: "yes" if v else "no", "l"),
        ),
        rows=rows, required=required, available=available, unit="gates",
        notes=(
            "Paired bootstrap over the 500 frozen evaluation seeds (1,000 resamples, fp16); a "
            "run is extended only when the CI of LSD(early) − LSD(late) lies above zero.",
            "35k vs 40k (job 2432703) said extend on IXI and not on Churches; all 30 runs were "
            "extended to 60k (D22). The informational 55k vs 60k gate (job 2486891) found no "
            "further gain on either run, so 60k is the final length.",
        ),
    )


def _required_gates(results: Results) -> tuple[tuple[str, ...], frozenset[str]]:
    by_index = {int(results.frame.at[r, "index"]): r for r in results.frame.index
                if not _missing(results.frame.at[r, "index"])}
    pairs = [tuple(p) for p in results.collection.get("required_gate_pairs") or ()]
    cells = [int(c) for c in results.collection.get("gate_cells") or ()]
    required = tuple(f"{by_index[c]}_gate_{a:06d}_{b:06d}" for c in cells if c in by_index
                     for a, b in pairs)
    available = frozenset(g["file"].removesuffix(".json") for g in results.gates)
    return required, available


# --------------------------------------------------------------------------------------------
# All tables, writing, provenance
# --------------------------------------------------------------------------------------------


def build_tables(results: Results) -> list[Table]:
    """Every table in report order (1a, 1b, 1c, 2a, 2b, 3-8), then exploratory 9."""
    return [
        *build_cell_tables(results),
        *build_contrast_tables(results),
        build_interaction_table(results),
        build_decomposition_table(results),
        build_a2p_table(results),
        build_transfer_table(results),
        build_t_tau_table(results),
        build_gate_table(results),
        build_settle_table(results),
    ]


def _git_sha() -> str:
    try:
        out = subprocess.run(["git", "-C", str(repo_root()), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def provenance(results: Results) -> dict[str, Any]:
    """What produced the tables: the results folder, its collection, this code, the method."""
    collection = results.collection
    return {
        "tool": "python -m ihdm.cli.analyse (T6.1)",
        "created": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "results": str(results.root),
        "collection": {k: collection.get(k) for k in ("created", "git_sha", "verdict",
                                                      "complete", "n_iters", "amp")},
        "n_runs_present": len(results.present),
        "n_runs_design": len(results.frame),
        "missing_runs": list(results.missing),
        "inherited_constants": {
            "path": results.constants_label,
            "decision": "D23 (docs/RESULTS/inherited_band_audit.md)",
            "runs_without_constants": list(results.inherited_notes),
        },
        "statistics": {
            "bootstrap": f"percentile, {N_BOOT} draws over seeds, alpha {ALPHA}, "
                         f"rng_seed {RNG_SEED} (ihdm.stats.paired_delta / interaction)",
            "permutation": "exact over all C(n_a + n_b, n_a) label assignments when the pooled "
                           "size is at most 6 (ihdm.stats.permutation_test), two-sided",
            "t_tau_threshold": "same-seed A0 LSD at the last checkpoint on the 500-seed curve "
                               "(primary); same-seed A0 2k final LSD (sensitivity)",
        },
    }


def _banner(results: Results) -> str:
    collection = results.collection
    if collection.get("complete") and not results.missing:
        return ""
    return (f"**PARTIAL COLLECTION (verdict {collection.get('verdict')}, "
            f"{len(results.present)}/{len(results.frame)} runs evaluated): not for quoting.**")


def _source(results: Results, meta: Mapping[str, Any]) -> str:
    collection = results.collection
    return (f"source: {results.root} (collection {collection.get('verdict')}, created "
            f"{collection.get('created')}, git {str(collection.get('git_sha'))[:10]}); "
            f"tables: git {str(meta['git_sha'])[:10]}, {meta['created']}")


def write_tables(results: Results, tables: Sequence[Table], out: str | Path) -> list[Path]:
    """Write ``<name>.md`` and ``<name>.tex`` per table and one ``tables.json`` into ``out``.

    Files of other names in ``out`` (the hand-written README, older samples) are left alone.

    Parameters
    ----------
    results : Results
        The loaded folder (for the provenance and the partial-collection banner).
    tables : Sequence[Table]
        The tables of :func:`build_tables`.
    out : str or Path
        Destination folder; created if needed.

    Returns
    -------
    list[Path]
        The files written.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta = provenance(results)
    banner, source = _banner(results), _source(results, meta)
    written = []
    for table in tables:
        md = out / f"{table.name}.md"
        md.write_text(table.to_markdown(banner, source))
        tex = out / f"{table.name}.tex"
        tex.write_text(table.to_latex(banner))
        written += [md, tex]
    document = {
        **meta,
        "complete": all(t.complete for t in tables) and not results.missing,
        "tables": {t.name: t.to_json() for t in tables},
    }
    written.append(write_json(out / "tables.json", _json_safe(document)))
    return written


@dataclass(frozen=True)
class AnalysisReport:
    """What :func:`analyse` did, for the command line."""

    results: Results
    tables: tuple[Table, ...]
    written: tuple[Path, ...]

    @property
    def complete(self) -> bool:
        """Whether every table is complete."""
        return all(t.complete for t in self.tables) and not self.results.missing

    def summary(self) -> str:
        """One line per table and a verdict line."""
        lines = [f"results: {self.results.root} (collection "
                 f"{self.results.collection.get('verdict')}, {len(self.results.present)}/"
                 f"{len(self.results.frame)} runs)"]
        lines += [f"  Table {t.number:<3} {t.name:<26} {t.status}" for t in self.tables]
        lines.append(f"wrote {len(self.written)} files")
        lines.append("VERDICT: " + ("COMPLETE" if self.complete else "INCOMPLETE"))
        return "\n".join(lines)


def analyse(results_dir: str | Path, out: str | Path,
            inherited_constants: str | Path | None = DEFAULT_CONSTANTS_PATH) -> AnalysisReport:
    """Load a results folder, build every table and write it.

    Parameters
    ----------
    results_dir : str or Path
        The collected ``results/`` folder.
    out : str or Path
        Where the tables go.
    inherited_constants : str, Path or None
        The D23 constants file; default the committed one.

    Returns
    -------
    AnalysisReport
        The results, the tables and the files written.
    """
    results = load_results(results_dir, inherited_constants)
    tables = build_tables(results)
    written = write_tables(results, tables, out)
    return AnalysisReport(results=results, tables=tuple(tables), written=tuple(written))
