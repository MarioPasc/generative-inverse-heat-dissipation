"""F3 of the course paper (T8.5): fidelity against diversity.

Two panels. (a) Per-sample realism (Inception precision, k = 5) against population diversity
(recall), both against R⁻, for the 2,000 training-seeded samples (F, filled markers) and the 40
unseen seeds × 50 (H, hollow markers) of every IXI configuration and of OASIS-1 default/matched.
(b) Within-seed diversity: the ideal within-seed share :math:`1-I` of the linear-Gaussian model
(outline bar), the measured :math:`1-I_w` (filled bar, run dots), the regenerated share
:math:`\\rho = (1-I_w)/(1-I)` printed above, and the total variance ratio of the F samples
(diamond), all as shares of the reference's non-DC between-image variance.

Every number is read from a result file at build time. Per-run :math:`I_w` and
:math:`\\rho` are recomputed from ``index.csv`` and the constants file and compared with the
committed tables, the seed means are compared with the ticket's expectations, and any mismatch
raises :class:`PaperF3Error` (exit code 1 of the CLI).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from ihdm.analysis.exploratory import regeneration_ratio
from ihdm.analysis.inherited import (
    InheritedConstants,
    InheritedError,
    load_constants,
    within_seed_share,
)
from ihdm.analysis.paper_f2 import (
    ARM_LABELS,
    DATASET_NAMES,
    IXI_ARMS,
    OASIS_ARMS,
    PAPER_WIDTH_IN,
    IndexRun,
    PaperF2Error,
    audit_layout,
    file_sha256,
    merge_heldout,
    paper_style,
    read_index,
    runs_of,
    save_paper_figure,
)
from ihdm.analysis.paper_f2 import MissingInputError as F2MissingInputError
from ihdm.analysis.tables import AnalysisError

__all__ = [
    "FIGURE_NAME",
    "F3Data",
    "F3Expectations",
    "F3ExpectationError",
    "F3MissingInputError",
    "F3Paths",
    "F3Report",
    "PaperF3Error",
    "PRCell",
    "WithinCell",
    "assert_pr_means",
    "assert_within_means",
    "build",
    "denominator_check",
    "draw_f3",
    "load_heldout",
    "load_pr",
    "load_within",
    "render_f3_notes",
    "text_overlaps",
]

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------------

FIGURE_NAME: str = "f3_fidelity_diversity"
FIG_HEIGHT_IN: float = 2.3
MIN_HEIGHT_IN: float = 2.0
MAX_HEIGHT_IN: float = 2.4
MIN_FONT_PT: float = 7.0
SVG_HASHSALT: str = "ihdm-t8.5"

#: The six cells of the figure, in display order.
CELLS: tuple[tuple[str, str], ...] = (
    *(("ixi", arm) for arm in IXI_ARMS), *(("oasis1", arm) for arm in OASIS_ARMS),
)
#: Dataset = marker shape (never colour); sample set = fill.
DATASET_MARKERS: dict[str, str] = {"ixi": "o", "oasis1": "s"}
TOTAL_MARKER: str = "D"
#: Datasets whose single runs are drawn in panel (a); the others show seed means only.
RUNS_IN_A: tuple[str, ...] = ("ixi",)
SET_LABELS: dict[str, str] = {"F": "2,000 training seeds", "H": "40 unseen seeds × 50"}
OASIS_GREY: str = "#8a8780"


# --------------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------------


class PaperF3Error(AnalysisError):
    """F3 cannot be built as the ticket requires (exit code 1)."""


class F3MissingInputError(PaperF3Error):
    """A file the build needs does not exist (exit code 2)."""


class F3ExpectationError(PaperF3Error):
    """A seed mean differs from the value the ticket expects."""


class F3ConsistencyError(PaperF3Error):
    """Two files disagree on a number they should share (duplicated runs, I_w, ρ, I)."""


_T = TypeVar("_T")


def _call(func: Callable[..., _T], *args: Any) -> _T:
    """Call a function of another module and turn its errors into this module's."""
    try:
        return func(*args)
    except F2MissingInputError as error:
        raise F3MissingInputError(str(error)) from error
    except PaperF2Error as error:
        raise F3ConsistencyError(str(error)) from error
    except InheritedError as error:
        raise PaperF3Error(str(error)) from error


# --------------------------------------------------------------------------------------------
# Expectations (the ticket's numbers; synthetic tests pass their own)
# --------------------------------------------------------------------------------------------

Cell = tuple[str, str]

#: Panel (a) seed means (precision F, recall F, precision H, recall H), from the ticket. Values
#: whose 3-decimal ticket number sits exactly on the ±0.0005 bound are given to 4 decimals, as
#: computed by hand from the per-run JSON values (main, 2026-10-07). OASIS-1 matched F is
#: corrected from the ticket's 0.835/0.526, which averaged per-run values already rounded.
_PR_EXPECTED: dict[Cell, tuple[float, float, float, float]] = {
    ("ixi", "A0"): (0.655, 0.343, 0.661, 0.2375),
    ("ixi", "A2"): (0.668, 0.395, 0.683, 0.2375),
    ("ixi", "A1"): (0.700, 0.4375, 0.715, 0.094),
    ("ixi", "A3"): (0.757, 0.4625, 0.7585, 0.0825),
    ("oasis1", "A0"): (0.705, 0.456, 0.708, 0.361),
    ("oasis1", "A3"): (0.834, 0.525, 0.842, 0.141),
}
#: Panel (b) seed means (1 − I, 1 − I_w, ρ, total variance ratio), from the ticket.
_WITHIN_EXPECTED: dict[Cell, tuple[float, float, float, float]] = {
    ("ixi", "A0"): (0.9966, 0.342, 0.343, 0.665),
    ("ixi", "A2"): (0.9966, 0.347, 0.348, 0.671),
    ("ixi", "A1"): (0.9205, 0.262, 0.285, 0.848),
    ("ixi", "A3"): (0.9205, 0.262, 0.285, 0.885),
    ("oasis1", "A0"): (0.9983, 0.448, 0.449, 0.663),
    ("oasis1", "A3"): (0.9574, 0.323, 0.337, 0.818),
}
#: The ticket's own numbers where they were replaced, for the notes.
TICKET_CORRECTIONS: tuple[tuple[str, float, float, str], ...] = (
    ("OASIS-1 matched, F precision", 0.835, 0.834,
     "the ticket averaged per-run values rounded to 3 decimals (0.842, 0.827)"),
    ("OASIS-1 matched, F recall", 0.526, 0.525,
     "the ticket averaged per-run values rounded to 3 decimals (0.458, 0.593)"),
)


@dataclass(frozen=True)
class F3Expectations:
    """The values the build asserts.

    Parameters
    ----------
    pr : dict
        ``{(dataset, arm): (precision F, recall F, precision H, recall H)}`` seed means.
    within : dict
        ``{(dataset, arm): (1 − I, 1 − I_w, ρ, total variance ratio)}`` seed means.
    pr_tol, within_tol, ideal_tol : float
        Absolute tolerances (inclusive) of panel (a), of panel (b) and of :math:`1 - I`.
    recompute_tol : float
        Absolute tolerance of the recomputed per-run :math:`I_w` and ρ against the tables.
    """

    pr: dict[Cell, tuple[float, float, float, float]] = field(
        default_factory=lambda: dict(_PR_EXPECTED))
    within: dict[Cell, tuple[float, float, float, float]] = field(
        default_factory=lambda: dict(_WITHIN_EXPECTED))
    pr_tol: float = 0.0005
    within_tol: float = 0.005
    ideal_tol: float = 0.0005
    recompute_tol: float = 1e-6


def _label(cell: Cell) -> str:
    return f"{DATASET_NAMES[cell[0]]} {ARM_LABELS[cell[1]]}"


def _check(name: str, value: float, expected: float, tol: float, failures: list[str]) -> str:
    # Inclusive bound plus float slack; no expectation sits on the bound (4-decimal values).
    ok = abs(value - expected) <= tol + 1e-12
    if not ok:
        failures.append(f"{name} = {value:.5f}, expected {expected} ± {tol}")
    return f"{name}: {value:.4f} (expected {expected:g} ± {tol:g}) {'ok' if ok else 'FAIL'}"


# --------------------------------------------------------------------------------------------
# Panel (a): precision and recall against R⁻
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PRCell:
    """Panel (a) values of one cell, one entry per run in seed order."""

    dataset: str
    arm: str
    run_ids: tuple[str, ...]
    precision_f: np.ndarray
    recall_f: np.ndarray
    precision_h: np.ndarray
    recall_h: np.ndarray
    recall_h_r5: np.ndarray

    @property
    def n(self) -> int:
        return len(self.run_ids)

    def mean(self) -> tuple[float, float, float, float]:
        """Seed means (precision F, recall F, precision H, recall H)."""
        return (float(np.mean(self.precision_f)), float(np.mean(self.recall_f)),
                float(np.mean(self.precision_h)), float(np.mean(self.recall_h)))


def load_heldout(paths: Sequence[Path]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """:func:`paper_f2.merge_heldout` with this module's errors.

    Raises
    ------
    F3MissingInputError
        If a file is absent.
    F3ConsistencyError
        If a run present in several files carries different metrics, or the references differ.
    """
    return _call(merge_heldout, list(paths))


def _metric(record: Mapping[str, Any], run_id: str, block: str, key: str) -> float:
    try:
        return float(record[block][key])
    except (KeyError, TypeError, ValueError) as error:
        raise PaperF3Error(f"{run_id}: held-out metrics lack {block}.{key}") from error


def load_pr(index: Sequence[IndexRun], heldout: Mapping[str, Mapping[str, Any]]
            ) -> dict[Cell, PRCell]:
    """Collect precision and recall (F and H against R⁻; H against R5) of every cell.

    Raises
    ------
    F3MissingInputError
        If a cell has no run in ``index.csv`` or a run is in no held-out JSON.
    """
    out: dict[Cell, PRCell] = {}
    for dataset, arm in CELLS:
        cell = _call(runs_of, index, dataset, arm)
        missing = [r.run_id for r in cell if r.run_id not in heldout]
        if missing:
            raise F3MissingInputError(f"not in any --heldout JSON: {', '.join(missing)}")
        ids = tuple(r.run_id for r in cell)

        def col(block: str, key: str, ids: tuple[str, ...] = ids) -> np.ndarray:
            return np.array([_metric(heldout[rid], rid, block, key) for rid in ids])

        out[(dataset, arm)] = PRCell(dataset, arm, ids, col("F_rminus", "precision"),
                                     col("F_rminus", "recall"), col("H_rminus", "precision"),
                                     col("H_rminus", "recall"), col("H_r5", "recall"))
    return out


def assert_pr_means(pr: Mapping[Cell, PRCell], expect: F3Expectations) -> list[str]:
    """Assert the panel (a) seed means; return the check lines.

    Raises
    ------
    F3ExpectationError
        If any mean is off by more than ``expect.pr_tol``.
    """
    failures: list[str] = []
    lines: list[str] = []
    names = ("precision F", "recall F", "precision H", "recall H")
    for cell, expected in expect.pr.items():
        if cell not in pr:
            raise F3ExpectationError(f"panel (a): no data for {_label(cell)}")
        for name, value, exp in zip(names, pr[cell].mean(), expected, strict=True):
            lines.append(_check(f"{_label(cell)} {name}", value, exp, expect.pr_tol, failures))
    if failures:
        raise F3ExpectationError("panel (a) seed means: " + "; ".join(failures))
    return lines


# --------------------------------------------------------------------------------------------
# Panel (b): within-seed diversity
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class WithinCell:
    """Panel (b) values of one cell, one entry per run in seed order.

    ``within`` is :math:`1 - I_w` (table 1c), ``rho`` the regeneration ratio
    (``exploratory.json``), ``total`` the variance ratio ``10**log_vr`` of the F samples, and
    ``ideal`` :math:`1 - I` (constants file).
    """

    dataset: str
    arm: str
    run_ids: tuple[str, ...]
    sigma_max: float
    ideal: float
    within: np.ndarray
    rho: np.ndarray
    total: np.ndarray

    @property
    def n(self) -> int:
        return len(self.run_ids)

    def mean(self) -> tuple[float, float, float, float]:
        """(1 − I, mean 1 − I_w, mean ρ, mean total variance ratio)."""
        return (self.ideal, float(np.mean(self.within)), float(np.mean(self.rho)),
                float(np.mean(self.total)))


def _read_json(path: Path) -> Any:
    path = Path(path)
    if not path.is_file():
        raise F3MissingInputError(f"{path} not found")
    try:
        return json.loads(path.read_text())
    except ValueError as error:
        raise PaperF3Error(f"{path}: {error}") from error


def read_table_1c(tables_json: Path) -> dict[str, dict[str, Any]]:
    """Return the rows of table 1c (``t1c_cells_inherited``) of ``tables.json`` by run id."""
    try:
        rows = _read_json(tables_json)["tables"]["t1c_cells_inherited"]["rows"]
        return {str(r["run_id"]): r for r in rows}
    except (KeyError, TypeError) as error:
        raise PaperF3Error(f"{tables_json}: no table 1c rows ({error!r})") from error


def read_exploratory(exploratory_json: Path) -> dict[str, dict[str, Any]]:
    """Return the ``per_run`` records of ``exploratory.json`` by run id."""
    try:
        return {str(r["run_id"]): r for r in _read_json(exploratory_json)["per_run"]}
    except (KeyError, TypeError) as error:
        raise PaperF3Error(f"{exploratory_json}: no per_run records ({error!r})") from error


@dataclass(frozen=True)
class RunRecord:
    """What panel (b) reads of one run's own result files."""

    dataset_sha256: str
    sigma_max: float
    n_per_seed: float
    variance_ratio: float
    n_reference: int
    inherited_predicted: float


def read_run_record(results: Path, run_id: str) -> RunRecord:
    """Read ``summary.json`` (dataset hash) and ``final.json`` of one run."""
    folder = Path(results) / "runs" / run_id
    summary, final = _read_json(folder / "summary.json"), _read_json(folder / "final.json")
    try:
        return RunRecord(str(summary["dataset_sha256"]), float(final["sigma_max"]),
                         float(final["n_per_seed"]), float(final["variance_ratio"]),
                         int(final["n_reference"]), float(final["inherited_predicted"]))
    except (KeyError, TypeError, ValueError) as error:
        raise PaperF3Error(f"{folder}: unreadable run record ({error!r})") from error


def _field(rows: Mapping[str, Mapping[str, Any]], run_id: str, key: str, source: str) -> float:
    if run_id not in rows:
        raise F3MissingInputError(f"{source} has no row for {run_id}")
    try:
        return float(rows[run_id][key])
    except (KeyError, TypeError, ValueError) as error:
        raise PaperF3Error(f"{source}: {run_id} lacks {key}") from error


def load_within(index: Sequence[IndexRun], results: Path, t1c: Mapping[str, Mapping[str, Any]],
                exploratory: Mapping[str, Mapping[str, Any]], constants: InheritedConstants,
                tol: float = 1e-6) -> tuple[dict[Cell, WithinCell], list[str]]:
    """Collect panel (b) per run and cross-check every number against its recomputation.

    Per run: :math:`I_w` recomputed from ``index.csv`` ``D_pix`` with
    :func:`inherited.within_seed_share` must equal table 1c; ρ recomputed with
    :func:`exploratory.regeneration_ratio` must equal ``regen_ratio``; :math:`I` of the constants
    must equal table 1c's, ``index.csv``'s and ``final.json``'s; ``10**log_vr`` must equal
    ``final.json`` ``variance_ratio``.

    Returns
    -------
    tuple[dict, list[str]]
        The cells and the check lines.

    Raises
    ------
    F3ConsistencyError
        If a recomputed value differs by more than ``tol``.
    """
    cells: dict[Cell, WithinCell] = {}
    problems: list[str] = []
    worst = {"I_w": 0.0, "rho": 0.0, "I": 0.0, "variance_ratio": 0.0}
    for dataset, arm in CELLS:
        runs = _call(runs_of, index, dataset, arm)
        within, rho, total, sigmas, ideals = [], [], [], set(), set()
        for run in runs:
            rid = run.run_id
            record = read_run_record(results, rid)
            entry, reason = constants.lookup(dataset, record.dataset_sha256, record.sigma_max)
            if entry is None:
                raise PaperF3Error(f"{rid}: {reason}")
            i_w = _field(t1c, rid, "inherited_within", "table 1c")
            i_w_new = within_seed_share(_call(run.value, "D_pix"), record.n_per_seed,
                                        entry.n_pix, entry.sum_power)
            try:
                rho_new = regeneration_ratio(i_w_new, entry.share_predicted)
            except ValueError as error:
                raise PaperF3Error(f"{rid}: {error}") from error
            rho_file = _field(exploratory, rid, "regen_ratio", "exploratory.json")
            vr = 10.0 ** _field(exploratory, rid, "log_vr", "exploratory.json")
            diffs = {
                "I_w": abs(i_w_new - i_w),
                "rho": abs(rho_new - rho_file),
                "I": max(abs(entry.share_predicted - _field(t1c, rid, "inherited_predicted",
                                                             "table 1c")),
                         abs(entry.share_predicted - _call(run.value, "inherited_predicted")),
                         abs(entry.share_predicted - record.inherited_predicted)),
                "variance_ratio": abs(vr - record.variance_ratio),
            }
            for key, diff in diffs.items():
                worst[key] = max(worst[key], diff)
                if not diff <= tol:
                    problems.append(f"{rid}: {key} differs by {diff:.3g}")
            within.append(1.0 - i_w)
            rho.append(rho_file)
            total.append(vr)
            sigmas.add(entry.sigma_max)
            ideals.add(1.0 - entry.share_predicted)
        if len(sigmas) != 1:
            raise PaperF3Error(f"{_label((dataset, arm))}: runs at several terminal blurs {sigmas}")
        cells[(dataset, arm)] = WithinCell(dataset, arm, tuple(r.run_id for r in runs),
                                           sigmas.pop(), ideals.pop(), np.array(within),
                                           np.array(rho), np.array(total))
    if problems:
        raise F3ConsistencyError("panel (b) cross-checks: " + "; ".join(problems))
    lines = [
        f"I_w recomputed from index.csv D_pix with inherited.within_seed_share equals table 1c "
        f"on every run (max |Δ| {worst['I_w']:.1e}, tolerance {tol:g}).",
        f"ρ recomputed with exploratory.regeneration_ratio equals exploratory.json regen_ratio "
        f"(max |Δ| {worst['rho']:.1e}).",
        f"I of the constants equals table 1c, index.csv and final.json inherited_predicted "
        f"(max |Δ| {worst['I']:.1e}).",
        f"10**log_vr equals final.json variance_ratio (max |Δ| {worst['variance_ratio']:.1e}).",
    ]
    return cells, lines


def assert_within_means(within: Mapping[Cell, WithinCell], expect: F3Expectations) -> list[str]:
    """Assert the panel (b) seed means; return the check lines.

    Raises
    ------
    F3ExpectationError
        If a mean is off by more than its tolerance.
    """
    failures: list[str] = []
    lines: list[str] = []
    names = ("1 − I", "1 − I_w", "ρ", "total variance ratio")
    for cell, expected in expect.within.items():
        if cell not in within:
            raise F3ExpectationError(f"panel (b): no data for {_label(cell)}")
        tols = (expect.ideal_tol, expect.within_tol, expect.within_tol, expect.within_tol)
        for name, value, exp, tol in zip(names, within[cell].mean(), expected, tols, strict=True):
            lines.append(_check(f"{_label(cell)} {name}", value, exp, tol, failures))
    if failures:
        raise F3ExpectationError("panel (b) seed means: " + "; ".join(failures))
    return lines


def denominator_check(results: Path, constants_path: Path,
                      within: Mapping[Cell, WithinCell]) -> dict[str, Any]:
    """Compare the reference of the variance ratio with the :math:`\\sum P_{ref}` of :math:`I_w`.

    Both are computed in code from the ``ref`` split: ``run_eval`` passes ``views.reference`` to
    ``metrics.spectral.lsd`` (``variance_ratio = sum P_S / sum P_ref`` over the non-DC modes,
    per-image DC removed, uint8/255), and ``inherited.reference_spectrum`` computes ``P_ref`` with
    the same ``mode_power`` of ``ref/255`` with the DC mode zeroed. Removing an image's DC changes
    only the DC coefficient, so the two non-DC sums are the same quantity. This function checks
    the data side: every run's ``final.json`` ``n_reference`` equals the constants' ``n_ref``.

    Returns
    -------
    dict
        ``same`` (bool), the per-dataset reference sizes and the runs checked.
    """
    document = _read_json(constants_path)
    sizes: dict[str, set[int]] = {}
    n_ref: dict[str, int] = {}
    for (dataset, _), cell in within.items():
        try:
            n_ref[dataset] = int(document["datasets"][dataset]["n_ref"])
        except (KeyError, TypeError, ValueError) as error:
            raise PaperF3Error(f"{constants_path}: no n_ref for {dataset}") from error
        for rid in cell.run_ids:
            sizes.setdefault(dataset, set()).add(read_run_record(results, rid).n_reference)
    same = all(sizes[d] == {n_ref[d]} for d in n_ref)
    return {"same": same, "n_ref": n_ref, "n_reference": {d: sorted(s) for d, s in sizes.items()},
            "n_runs": sum(c.n for c in within.values())}


# --------------------------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class F3Data:
    """Everything F3 draws; built from files by :func:`build` or synthetically by tests."""

    pr: dict[Cell, PRCell]
    within: dict[Cell, WithinCell]


@dataclass(frozen=True)
class F3Audit:
    """Measured legibility of the drawn figure."""

    width_in: float
    height_in: float
    min_font_pt: float
    n_texts: int
    overlaps: tuple[str, ...]

    def problems(self) -> list[str]:
        out = []
        if abs(self.width_in - PAPER_WIDTH_IN) > 1e-6:
            out.append(f"width {self.width_in} in != {PAPER_WIDTH_IN}")
        if not MIN_HEIGHT_IN - 1e-6 <= self.height_in <= MAX_HEIGHT_IN + 1e-6:
            out.append(f"height {self.height_in} in outside [{MIN_HEIGHT_IN}, {MAX_HEIGHT_IN}]")
        if self.min_font_pt < MIN_FONT_PT:
            out.append(f"smallest font {self.min_font_pt} pt < {MIN_FONT_PT}")
        if self.overlaps:
            out.append("overlapping texts: " + "; ".join(self.overlaps))
        return out


# Layout in inches (origin bottom-left).
A_X, A_W = 0.44, 2.22
B_X, B_W = 3.18, 1.62
AX_Y = 0.66
AX_TOP = FIG_HEIGHT_IN - 0.30
TITLE_Y = FIG_HEIGHT_IN - 0.12
#: Panel (b) bar positions: IXI's four, a gap, OASIS-1's two.
BAR_X: dict[Cell, float] = {("ixi", "A0"): 0.0, ("ixi", "A2"): 1.0, ("ixi", "A1"): 2.0,
                            ("ixi", "A3"): 3.0, ("oasis1", "A0"): 4.6, ("oasis1", "A3"): 5.6}
BAR_W: float = 0.66
RHO_Y: float = 1.06
B_YMAX: float = 1.22


def _axes(fig: Any, x: float, y: float, w: float, h: float, gid: str) -> Any:
    ax = fig.add_axes((x / PAPER_WIDTH_IN, y / FIG_HEIGHT_IN, w / PAPER_WIDTH_IN,
                       h / FIG_HEIGHT_IN))
    ax.set_gid(gid)
    return ax


def _fig_text(fig: Any, x: float, y: float, text: str, gid: str, **kw: Any) -> Any:
    artist = fig.text(x / PAPER_WIDTH_IN, y / FIG_HEIGHT_IN, text, **kw)
    artist.set_gid(gid)
    return artist


def _gid(cell: Cell) -> str:
    return f"{cell[0]}_{cell[1]}"


def _arrow(ax: Any, start: tuple[float, float], end: tuple[float, float], gid: str,
           color: str, lw: float = 0.9) -> None:
    arrow = ax.annotate("", xy=end, xytext=start,
                        arrowprops={"arrowstyle": "-|>", "lw": lw, "mutation_scale": 7,
                                    "color": color, "shrinkA": 4.5, "shrinkB": 4.5}, zorder=3)
    arrow.arrow_patch.set_gid(gid)


def _draw_panel_a(fig: Any, data: F3Data) -> Any:
    from ihdm.analysis.style import INK, arm_style

    ax = _axes(fig, A_X, AX_Y, A_W, AX_TOP - AX_Y, "a_plane")
    _fig_text(fig, 0.02, TITLE_Y, "(a)", "a_label", fontsize=8, fontweight="bold",
              va="center", ha="left")
    _fig_text(fig, 0.27, TITLE_Y, "per-sample realism vs population diversity",
              "a_title", fontsize=7.5, va="center", ha="left", color=INK)
    for cell, pts in data.pr.items():
        color, marker, gid = arm_style(cell[1]).color, DATASET_MARKERS[cell[0]], _gid(cell)
        if cell[0] in RUNS_IN_A:
            ax.plot(pts.recall_f, pts.precision_f, ls="none", marker=marker, color=color,
                    ms=2.6, alpha=0.6, mew=0, gid=f"a_runs_F_{gid}", zorder=2)
            ax.plot(pts.recall_h, pts.precision_h, ls="none", marker=marker, mfc="white",
                    mec=color, ms=2.8, mew=0.6, alpha=0.8, gid=f"a_runs_H_{gid}", zorder=2)
        pf, rf, ph, rh = pts.mean()
        ax.plot([rf], [pf], ls="none", marker=marker, color=color, ms=5.5, mec="white",
                mew=0.5, gid=f"a_mean_F_{gid}", zorder=4)
        ax.plot([rh], [ph], ls="none", marker=marker, mfc="white", mec=color, ms=5.2,
                mew=1.2, gid=f"a_mean_H_{gid}", zorder=4)
    for dataset, color in (("ixi", INK), ("oasis1", OASIS_GREY)):
        m0, m3 = data.pr[(dataset, "A0")].mean(), data.pr[(dataset, "A3")].mean()
        _arrow(ax, (m0[1], m0[0]), (m3[1], m3[0]), f"a_arrow_F_{dataset}", color)
        _arrow(ax, (m0[3], m0[2]), (m3[3], m3[2]), f"a_arrow_H_{dataset}", color)

    # Direct labels on the IXI arrows: where the set and the configuration change.
    i0, i3 = data.pr[("ixi", "A0")].mean(), data.pr[("ixi", "A3")].mean()
    ax.annotate("default", xy=(i0[1], i0[0]), xytext=(4, -6), textcoords="offset points",
                ha="left", va="top", fontsize=7, color=INK, gid="a_label_default_F")
    ax.annotate("matched", xy=(i3[1], i3[0]), xytext=(0, 5), textcoords="offset points",
                ha="center", va="bottom", fontsize=7, color=INK, gid="a_label_matched_F")
    o3 = data.pr[("oasis1", "A3")].mean()
    ax.annotate("OASIS-1", xy=(o3[3], o3[2]), xytext=(6, 0), textcoords="offset points",
                ha="left", va="center", fontsize=7, color=OASIS_GREY, gid="a_label_oasis")
    mid_h = ((i0[3] + i3[3]) / 2, (i0[2] + i3[2]) / 2)
    ax.annotate("IXI", xy=mid_h, xytext=(-3, -3), textcoords="offset points",
                ha="right", va="top", fontsize=7, color=INK, gid="a_label_ixi")

    # Limits from what is drawn: every run of RUNS_IN_A, the seed means of the rest.
    drawn = [(np.concatenate([p.recall_f, p.recall_h]), np.concatenate([p.precision_f,
                                                                         p.precision_h]))
             if c[0] in RUNS_IN_A else (np.array(p.mean()[1::2]), np.array(p.mean()[0::2]))
             for c, p in data.pr.items()]
    values_x = np.concatenate([x for x, _ in drawn])
    values_y = np.concatenate([y for _, y in drawn])
    ax.set_xlim(max(0.0, float(values_x.min()) - 0.04), float(values_x.max()) + 0.04)
    ax.set_ylim(float(values_y.min()) - 0.025, float(values_y.max()) + 0.03)
    ax.set_xlabel("recall vs R⁻ (population diversity)", labelpad=2)
    ax.set_ylabel("precision vs R⁻ (realism)", labelpad=2)
    ax.tick_params(axis="both", pad=1.5)
    return ax


def _key_marker(ax: Any, x: float, y: float, gid: str, **kw: Any) -> None:
    ax.plot([x], [y], ls="none", gid=gid, clip_on=False, **kw)


def _draw_key_a(fig: Any) -> None:
    """Fill (sample set) and shape (dataset) key under panel (a)."""
    from ihdm.analysis.style import INK, MUTED

    ax = _axes(fig, 0.0, 0.0, A_X + A_W, 0.30, "a_key")
    ax.set_axis_off()
    ax.set_xlim(0.0, A_X + A_W)
    ax.set_ylim(0.0, 0.30)
    y0, y1 = 0.20, 0.07
    _key_marker(ax, 0.10, y0, "a_key_F", marker="o", color=MUTED, ms=5.0, mec="white", mew=0.5)
    ax.text(0.17, y0, f"F: {SET_LABELS['F']}", fontsize=7, va="center", ha="left", color=INK,
            gid="a_key_F_text")
    _key_marker(ax, 1.48, y0, "a_key_H", marker="o", mfc="white", mec=MUTED, ms=4.8, mew=1.1)
    ax.text(1.55, y0, f"H: {SET_LABELS['H']}", fontsize=7, va="center", ha="left", color=INK,
            gid="a_key_H_text")
    _key_marker(ax, 0.10, y1, "a_key_ixi", marker="o", color=MUTED, ms=4.5)
    ax.text(0.17, y1, "IXI", fontsize=7, va="center", ha="left", color=INK, gid="a_key_ixi_text")
    _key_marker(ax, 0.48, y1, "a_key_oasis", marker="s", color=MUTED, ms=4.2)
    ax.text(0.55, y1, "OASIS-1", fontsize=7, va="center", ha="left", color=INK,
            gid="a_key_oasis_text")
    _key_marker(ax, 1.48, y1, "a_key_run", marker="o", color=MUTED, ms=2.6, alpha=0.6, mew=0)
    _key_marker(ax, 1.57, y1, "a_key_mean", marker="o", color=MUTED, ms=5.0)
    ax.text(1.66, y1, "run · seed mean", fontsize=7, va="center", ha="left", color=INK,
            gid="a_key_sizes_text")


def _draw_panel_b(fig: Any, data: F3Data) -> Any:
    from matplotlib.patches import Rectangle

    from ihdm.analysis.style import INK, MUTED, arm_style

    ax = _axes(fig, B_X, AX_Y, B_W, AX_TOP - AX_Y, "b_bars")
    _fig_text(fig, B_X - 0.42, TITLE_Y, "(b)", "b_label", fontsize=8, fontweight="bold",
              va="center", ha="left")
    _fig_text(fig, B_X - 0.17, TITLE_Y, "within-seed diversity", "b_title", fontsize=7.5,
              va="center", ha="left", color=INK)
    ax.axhline(1.0, color=MUTED, lw=0.8, ls=(0, (1, 1.5)), zorder=1, gid="b_real_line")
    for cell, pts in data.within.items():
        x, color, gid = BAR_X[cell], arm_style(cell[1]).color, _gid(cell)
        ideal, within, rho, total = pts.mean()
        outline = Rectangle((x - BAR_W / 2, 0.0), BAR_W, ideal, fill=False, edgecolor=color,
                            lw=0.9, zorder=2)
        outline.set_gid(f"b_ideal_{gid}")
        ax.add_patch(outline)
        filled = Rectangle((x - BAR_W / 2, 0.0), BAR_W, within, facecolor=color, lw=0,
                           zorder=2)
        filled.set_gid(f"b_measured_{gid}")
        ax.add_patch(filled)
        offsets = (np.arange(pts.n) - (pts.n - 1) / 2) * 0.17
        ax.plot(x + offsets, pts.within, ls="none", marker=DATASET_MARKERS[cell[0]], ms=2.4,
                mfc=INK, mec="white", mew=0.4, zorder=3, gid=f"b_runs_{gid}")
        # Runs of the total variance ratio: their range, behind the seed-mean diamond.
        ax.plot([x, x], [float(pts.total.min()), float(pts.total.max())], color=INK, lw=0.8,
                solid_capstyle="butt", zorder=3, gid=f"b_total_range_{gid}")
        ax.plot([x], [total], ls="none", marker=TOTAL_MARKER, ms=4.8, mfc=color, mec=INK,
                mew=0.6, zorder=4, gid=f"b_total_{gid}")
        ax.text(x, RHO_Y, f"{rho:.2f}", fontsize=7, ha="center", va="bottom", color=INK,
                gid=f"b_rho_{gid}")
    ax.text(-0.62, RHO_Y, r"$\rho$", fontsize=7.5, ha="right", va="bottom", color=INK,
            gid="b_rho_label")
    last = data.within[CELLS[-1]].mean()
    x_end = BAR_X[CELLS[-1]] + BAR_W / 2
    for gid, y, dy, text in (("b_key_ideal", last[0], -3.5, r"ideal $1-I$"),
                             ("b_key_total", last[3], -1.0, "total (F)"),
                             ("b_key_measured", last[1], 0.0, "measured\n" r"$1-I_w$ (H)")):
        ax.annotate(text, xy=(x_end, y), xytext=(7, dy), textcoords="offset points",
                    linespacing=1.0,
                    ha="left", va="center", fontsize=7, color=INK, gid=gid,
                    annotation_clip=False,
                    arrowprops={"arrowstyle": "-", "lw": 0.5, "color": MUTED, "shrinkA": 1,
                                "shrinkB": 0})
    ax.text(x_end + 0.1, 1.0, "real", fontsize=7, ha="left", va="bottom", color=MUTED,
            gid="b_real_label", clip_on=False)
    ax.set_xlim(-0.55, x_end + 0.12)
    ax.set_ylim(0.0, B_YMAX)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(_plain_formatter())
    ax.set_xticks([BAR_X[c] for c in CELLS])
    ax.set_xticklabels([ARM_LABELS[a] for _, a in CELLS], rotation=35, ha="right",
                       rotation_mode="anchor")
    ax.tick_params(axis="x", length=0, pad=2)
    ax.tick_params(axis="y", pad=1.5)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("share of real variance", labelpad=2)
    for dataset, (lo, hi) in (("ixi", (0.0, 3.0)), ("oasis1", (4.6, 5.6))):
        ax.text((lo + hi) / 2, -0.47, DATASET_NAMES[dataset], fontsize=7, ha="center",
                va="top", color=MUTED, gid=f"b_group_{dataset}", clip_on=False,
                fontstyle="italic")
    return ax


def _plain_formatter() -> Any:
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(lambda v, _: f"{v:g}")


def draw_f3(data: F3Data) -> Any:
    """Draw F3 at 5.5 × :data:`FIG_HEIGHT_IN` in and return the figure (caller saves it)."""
    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(PAPER_WIDTH_IN, FIG_HEIGHT_IN))
    _draw_panel_a(fig, data)
    _draw_key_a(fig)
    _draw_panel_b(fig, data)
    return fig


def text_overlaps(fig: Any, pad_px: float = 0.0) -> list[str]:
    """Return the pairs of visible, non-empty texts whose rendered boxes intersect."""
    from matplotlib.text import Text

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    hidden = _undrawn_tick_labels(fig)
    boxes = []
    for t in fig.findobj(Text):
        if not (t.get_visible() and t.get_text().strip()) or id(t) in hidden:
            continue
        box = t.get_window_extent(renderer)
        if box.width > 1.0 and box.height > 1.0:
            boxes.append((t.get_gid() or t.get_text(), box, bool(t.get_rotation() % 360)))
    out = []
    for i, (name_a, a, rot_a) in enumerate(boxes):
        for name_b, b, rot_b in boxes[i + 1:]:
            # A rotated text's axis-aligned box is much larger than the text (the panel (b)
            # tick labels); those are inspected on the rendered figure instead.
            if rot_a or rot_b:
                continue
            if (a.x0 < b.x1 - pad_px and b.x0 < a.x1 - pad_px and a.y0 < b.y1 - pad_px
                    and b.y0 < a.y1 - pad_px):
                out.append(f"{name_a!s} / {name_b!s}")
    return out


def _undrawn_tick_labels(fig: Any) -> set[int]:
    """Ids of tick labels that exist but are not drawn (axes off, tick outside the view)."""
    hidden: set[int] = set()
    for ax in fig.axes:
        for axis, (lo, hi) in ((ax.xaxis, sorted(ax.get_xlim())),
                               (ax.yaxis, sorted(ax.get_ylim()))):
            for tick in [*axis.get_major_ticks(), *axis.get_minor_ticks()]:
                if not ax.axison or not lo - 1e-9 <= tick.get_loc() <= hi + 1e-9:
                    hidden.update((id(tick.label1), id(tick.label2)))
    return hidden


def audit_f3(fig: Any) -> F3Audit:
    """Measure size, smallest font (:func:`paper_f2.audit_layout`) and text overlaps."""
    base = audit_layout(fig)
    return F3Audit(base.width_in, base.height_in, base.min_font_pt, base.n_texts,
                   tuple(text_overlaps(fig)))


# --------------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class F3Paths:
    """Inputs and outputs of one build (the CLI arguments)."""

    results: Path
    heldout: tuple[Path, ...]
    tables_json: Path
    exploratory_json: Path
    constants: Path
    out: Path


@dataclass
class F3Report:
    """What one build read, checked and wrote; ``F3.md`` is rendered from it."""

    paths: F3Paths
    data: F3Data
    heldout_record: dict[str, Any]
    pr_checks: list[str]
    within_checks: list[str]
    consistency: list[str]
    denominator: dict[str, Any]
    audit: F3Audit
    files: dict[str, Path] = field(default_factory=dict)

    def sha256(self) -> dict[str, str]:
        return {name: file_sha256(path) for name, path in sorted(self.files.items())}


def check_inputs(paths: F3Paths) -> None:
    """Raise :class:`F3MissingInputError` naming every absent top-level input."""
    wanted = (Path(paths.results) / "index.csv", Path(paths.tables_json),
              Path(paths.exploratory_json), Path(paths.constants), *map(Path, paths.heldout))
    missing = [str(p) for p in wanted if not p.is_file()]
    if missing:
        raise F3MissingInputError("missing input: " + ", ".join(missing))


def load_data(paths: F3Paths, expect: F3Expectations
              ) -> tuple[F3Data, dict[str, Any], list[str], list[str], list[str]]:
    """Read, cross-check and assert everything F3 draws."""
    check_inputs(paths)
    index = _call(read_index, paths.results)
    heldout, record = load_heldout(paths.heldout)
    pr = load_pr(index, heldout)
    pr_checks = assert_pr_means(pr, expect)
    constants = _call(load_constants, paths.constants)
    within, consistency = load_within(index, paths.results, read_table_1c(paths.tables_json),
                                      read_exploratory(paths.exploratory_json), constants,
                                      expect.recompute_tol)
    within_checks = assert_within_means(within, expect)
    return F3Data(pr, within), record, pr_checks, within_checks, consistency


def build(paths: F3Paths, expect: F3Expectations | None = None) -> F3Report:
    """Build F3 and ``F3.md`` from the files named in ``paths``.

    Raises
    ------
    F3MissingInputError
        If an input file is absent.
    PaperF3Error
        If a cross-check, an expectation or the layout audit fails.
    """
    from matplotlib import pyplot as plt

    expect = expect or F3Expectations()
    data, record, pr_checks, within_checks, consistency = load_data(paths, expect)
    denominator = denominator_check(paths.results, paths.constants, data.within)
    out = Path(paths.out)
    with paper_style(), plt.rc_context({"svg.hashsalt": SVG_HASHSALT}):
        fig = draw_f3(data)
        audit = audit_f3(fig)
        problems = audit.problems()
        if problems:
            plt.close(fig)
            raise PaperF3Error("F3 layout: " + "; ".join(problems))
        files = {f"{FIGURE_NAME}.{k}": v
                 for k, v in save_paper_figure(fig, out, FIGURE_NAME).items()}
    report = F3Report(paths, data, record, pr_checks, within_checks, consistency, denominator,
                      audit, files)
    (out / "F3.md").write_text(render_f3_notes(report))
    return report


# --------------------------------------------------------------------------------------------
# Notes: F3.md (generated; every number comes from the report)
# --------------------------------------------------------------------------------------------


def _n_line(cells: Mapping[Cell, Any], dataset: str) -> str:
    return "/".join(str(c.n) for (d, _), c in cells.items() if d == dataset)


def _r(value: float, places: int = 3) -> str:
    """Round half up at ``places`` decimals (0.2375 -> 0.238), after removing float noise."""
    quantum = Decimal(1).scaleb(-places)
    return str(Decimal(f"{value:.10f}").quantize(quantum, rounding=ROUND_HALF_UP))


def caption(report: F3Report) -> str:
    """Return the caption draft (blockquote), every number from the report."""
    pr, w = report.data.pr, report.data.within
    a0, a3 = pr[("ixi", "A0")].mean(), pr[("ixi", "A3")].mean()
    w0, w3 = w[("ixi", "A0")].mean(), w[("ixi", "A3")].mean()
    rhos = [c.mean()[2] for c in w.values()]
    n_ref = report.denominator["n_ref"]["ixi"]
    r_minus = report.heldout_record["references"]["ixi"]["n_r_minus"]
    return "\n".join([
        "> **Seeding from many training images raises realism and diversity together; within a "
        "seed, every model under-disperses.**",
        ">",
        "> (a) Inception precision and recall (k = 5) against R⁻, the reference split without the "
        f"40 seed subjects ({r_minus} images). Filled: 2,000 training-seeded samples (F); hollow: "
        f"40 unseen seeds × 50 (H). Circles: IXI (n = {_n_line(pr, 'ixi')} for "
        f"default/+spacing/+prior/matched; small: runs), squares OASIS-1 seed means "
        f"(n = {_n_line(pr, 'oasis1')}). Default → matched: F moves up and right "
        f"(precision {_r(a0[0])} → {_r(a3[0])}, recall {_r(a0[1])} → {_r(a3[1])}); H gains the "
        f"same precision ({_r(a0[2])} → {_r(a3[2])}) but loses recall ({_r(a0[3])} → "
        f"{_r(a3[3])}). H has 40 prior states, all at one slice level, so its recall measures "
        "how much of R⁻ 40 seeds reach: W/8's population diversity comes from the seed set.",
        ">",
        "> (b) Shares of the real between-image variance (non-DC, full reference split, "
        f"{n_ref} images): outline, ideal 1 − I; fill, measured 1 − I_w on H "
        "(dots: runs); ρ = (1 − I_w)/(1 − I); diamond, total variance ratio of F. Models "
        f"regenerate {_r(100 * min(rhos), 0)}–{_r(100 * max(rhos), 0)}% of the removed "
        f"variance. W/8 lowers 1 − I_w by {_r(100 * (1 - w3[1] / w0[1]), 0)}% ({_r(w0[1])} → "
        f"{_r(w3[1])}) while the total rises ({_r(w0[3], 2)} → {_r(w3[3], 2)}).",
        ">",
        "> Precision, recall and ρ are post hoc or exploratory; no significance tests.",
    ])


def _r5_sentence(report: F3Report) -> str:
    pr = report.data.pr

    def m(dataset: str, arm: str, attr: str) -> float:
        return float(np.mean(getattr(pr[(dataset, arm)], attr)))

    parts, verdicts = [], []
    for dataset in ("ixi", "oasis1"):
        r5 = (m(dataset, "A0", "recall_h_r5"), m(dataset, "A3", "recall_h_r5"))
        rm = (m(dataset, "A0", "recall_h"), m(dataset, "A3", "recall_h"))
        parts.append(f"{DATASET_NAMES[dataset]} {_r(r5[0])} → {_r(r5[1])} (R⁻: {_r(rm[0])} → "
                     f"{_r(rm[1])})")
        verdicts.append((r5[1] < r5[0], r5[0] - r5[1] < rm[0] - rm[1]))
    n_r5 = report.heldout_record["references"]["ixi"]["n_r5"]
    falls = all(v[0] for v in verdicts)
    smaller = all(v[1] for v in verdicts)
    trend = ("still falls" if falls else "does not fall on every dataset") + (
        ", but less" if smaller else "")
    reading = ("Part of the drop against R⁻ is the slice-level restriction of the 40 seeds; its "
               "direction does not depend on the reference." if falls and smaller else
               "The reading of panel (a) depends on the reference; see the table.")
    return (f"Against R5 (the slice-5 image of each training subject, {n_r5} images: the "
            "population at the held-out seeds' slice level), H recall from default to matched "
            f"{trend}: " + "; ".join(parts) + ". " + reading)


def _pr_table(report: F3Report) -> list[str]:
    src = report.heldout_record["sources"]
    lines = [
        "| dataset | configuration | run | precision F | recall F | precision H | recall H "
        "| recall H vs R5 | source |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for cell, c in report.data.pr.items():
        ds, label = DATASET_NAMES[cell[0]], ARM_LABELS[cell[1]]
        for i, rid in enumerate(c.run_ids):
            lines.append(f"| {ds} | {label} | `{rid}` | {c.precision_f[i]:.4f} | "
                         f"{c.recall_f[i]:.4f} | {c.precision_h[i]:.4f} | {c.recall_h[i]:.4f} | "
                         f"{c.recall_h_r5[i]:.4f} | `{Path(src[rid]).parent.name}/"
                         f"{Path(src[rid]).name}` |")
        m = c.mean()
        lines.append(f"| {ds} | **{label} mean** (n = {c.n}) | | **{m[0]:.4f}** | "
                     f"**{m[1]:.4f}** | **{m[2]:.4f}** | **{m[3]:.4f}** | "
                     f"**{float(np.mean(c.recall_h_r5)):.4f}** | |")
    return lines


def _within_table(report: F3Report) -> list[str]:
    lines = [
        "| dataset | configuration | run | σ_max (px) | 1 − I | 1 − I_w | ρ | total variance "
        "ratio |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for cell, c in report.data.within.items():
        ds, label = DATASET_NAMES[cell[0]], ARM_LABELS[cell[1]]
        for i, rid in enumerate(c.run_ids):
            lines.append(f"| {ds} | {label} | `{rid}` | {c.sigma_max:g} | {c.ideal:.4f} | "
                         f"{c.within[i]:.4f} | {c.rho[i]:.4f} | {c.total[i]:.4f} |")
        m = c.mean()
        lines.append(f"| {ds} | **{label} mean** (n = {c.n}) | | {c.sigma_max:g} | "
                     f"**{m[0]:.4f}** | **{m[1]:.4f}** | **{m[2]:.4f}** | **{m[3]:.4f}** |")
    return lines


def _denominator_lines(report: F3Report) -> list[str]:
    d = report.denominator
    r_minus = report.heldout_record["references"]["ixi"]["n_r_minus"]
    verdict = "the same" if d["same"] else "**different**"
    sizes = ", ".join(f"{DATASET_NAMES[k]} {v}" for k, v in sorted(d["n_ref"].items()))
    return [
        f"**Verdict: {verdict}.** The total variance ratio (`10**log_vr` = `final.json` "
        "`variance_ratio`) and I_w divide by the same non-DC reference power ΣP_ref.",
        "",
        "- *Code.* `run_eval` passes `views.reference` (the full `ref` split) to "
        "`metrics.spectral.lsd`, whose `variance_ratio` is ΣP_S/ΣP_ref over every non-DC mode "
        "(`mode_power` of uint8/255 with each image's DC removed). `inherited.reference_spectrum` "
        "computes P_ref of the same `ref` split with the same `mode_power` of ref/255 and the "
        "DC mode zeroed. In the orthonormal DCT, removing an image's mean changes only the DC "
        "coefficient, so the two non-DC sums are the same number.",
        f"- *Data.* Every one of the {d['n_runs']} runs' `final.json` `n_reference` equals the "
        f"constants' `n_ref` ({sizes}); `inherited_predicted` of every `final.json` equals the "
        "constants' I (checked above).",
        f"- *Not the same reference as panel (a).* Precision and recall use R⁻ ({r_minus} images "
        "on IXI and OASIS-1: the `ref` split without the 40 seed subjects); the variance ratio "
        f"and I_w use the full `ref` split ({sizes}). The caption says so.",
    ]


def render_f3_notes(report: F3Report) -> str:
    """Return ``F3.md``: caption draft, numbers with sources, R5 sentence, checks, command."""
    p, a = report.paths, report.audit
    held = report.heldout_record
    lines = [
        "# F3 — fidelity against diversity (T8.5)",
        "",
        "Generated by `python -m ihdm.cli.paper_f3`; do not edit by hand, rerun the command.",
        f"Inkscape master: `{FIGURE_NAME}.svg` (text kept as text; every marker, bar, arrow and "
        f"label a separate element with its own `id`); LaTeX input: `{FIGURE_NAME}.pdf`; "
        f"preview: `{FIGURE_NAME}.png` (300 dpi).",
        "",
        "## Caption draft",
        "",
        caption(report),
        "",
        "## Numbers and their sources",
        "",
        "### Panel (a): precision and recall (Inception, k = 5)",
        "",
        "`runs.<id>.metrics.{F_rminus,H_rminus}.{precision,recall}` and "
        "`runs.<id>.metrics.H_r5.recall` of "
        + ", ".join(f"`{h}`" for h in p.heldout)
        + ". F = 2,000 training-seeded samples (one per seed image); H = 40 unseen subjects "
        "(slice 5) × 50 samples. R⁻ = `ref` minus the 40 seed subjects.",
        "",
        *_pr_table(report),
        "",
        "### Panel (b): within-seed diversity",
        "",
        f"- 1 − I: `share_predicted` of `{p.constants}` at the run's terminal blur "
        "(linear-Gaussian prediction, `results_discussion.md` §2a.6).",
        f"- 1 − I_w: `inherited_within` of table 1c, `{p.tables_json}`, measured on H "
        "(I_w = 1 − (50/49) D_pix (W² − 1)/ΣP_ref, D23); recomputed per run from `D_pix` of "
        f"`{Path(p.results) / 'index.csv'}`.",
        f"- ρ = (1 − I_w)/(1 − I): `regen_ratio` of `{p.exploratory_json}` `per_run` "
        "(exploratory).",
        "- total variance ratio ΣP_S/ΣP_ref of the F samples: `10**log_vr` of the same file "
        "(= `final.json` `variance_ratio`).",
        "",
        *_within_table(report),
        "",
        "## R5 sensitivity (not plotted)",
        "",
        _r5_sentence(report),
        "",
        "## Denominator of the total variance ratio",
        "",
        *_denominator_lines(report),
        "",
        "## Corrections to the ticket's expectation table",
        "",
        "| quantity | ticket | asserted | reason |",
        "|---|---:|---:|---|",
        *(f"| {q} | {t:.3f} | {n:.3f} | {why} |" for q, t, n, why in TICKET_CORRECTIONS),
        "",
        "Means whose 3-decimal ticket value lies exactly on the ±0.0005 bound are asserted at "
        "their 4-decimal value (computed by hand from the per-run JSON values): IXI default H "
        "recall 0.2375, +spacing H recall 0.2375, +prior F recall 0.4375, matched F recall "
        "0.4625, matched H precision 0.7585, matched H recall 0.0825.",
        "",
        "## Checks (asserted by the build)",
        "",
        "- **Held-out JSONs:** runs present in more than one file carry identical metrics: "
        + (", ".join(f"`{r}`" for r in held["identical_in_several_files"]) or "none")
        + "; the reference sets (sha256 of the ref, R⁻ and R5 indices) agree across files.",
        *(f"- {line}" for line in report.consistency),
        *(f"- {line}" for line in report.pr_checks),
        *(f"- {line}" for line in report.within_checks),
        f"- **Layout (measured on the drawn figure):** {a.width_in:.2f} × {a.height_in:.2f} in, "
        f"smallest font {a.min_font_pt:g} pt over {a.n_texts} text elements, no overlapping "
        f"text boxes (limits: {PAPER_WIDTH_IN} in wide, {MIN_HEIGHT_IN}–{MAX_HEIGHT_IN} in tall, "
        f"≥ {MIN_FONT_PT:g} pt).",
        "",
        "## Files",
        "",
        "| file | sha256 |",
        "|---|---|",
        *(f"| `{name}` | `{digest}` |" for name, digest in report.sha256().items()),
        "",
        "Byte stability: a second run of the command reproduces these digests. The PDF and SVG "
        "metadata name `ihdm.cli.paper_f2` as creator, because `paper_f2.save_paper_figure` is "
        "reused unchanged.",
        "",
        "## Command",
        "",
        "```bash",
        f"python -m ihdm.cli.paper_f3 --results {p.results} --heldout "
        f"{' '.join(str(h) for h in p.heldout)} --tables {p.tables_json} --exploratory "
        f"{p.exploratory_json} --constants {p.constants} --out {p.out}",
        "```",
        "",
        "Environment: `OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=\"\" nice -n 19`, "
        "`PYTHONPATH=<repository>`, conda env `ihdm`.",
        "",
    ]
    return "\n".join(lines)
