"""Health check of the whole training array: one row per cell of ``cells.csv`` and a verdict.

``python -m ihdm.cli.check_array --run-root <root> --cells slurm/array/cells.csv --n-iters N
[--allow-skips] [--data-root <root>] [--deep]``

The per-run checks are those of :mod:`ihdm.train.validate_run` (``04-run-artifacts.md`` §3 with
D19): strict ``metrics.jsonl`` schema, values and cadence, the ``done`` event at ``N``, the
manifest and schedule hashes, the grids. By default the run is checked **light**: checkpoint files
are counted, never loaded, so the check reads a few MB per run and is safe on the login node.
``--deep`` calls :func:`ihdm.train.validate_run.check_run_dir`, which also loads the ``N``-step EMA
checkpoint and ``full_final.pt`` (≈ 1.2 GB per run).

On top of those, per cell: ``DONE``; the largest EMA checkpoint is ``N`` (no later one);
``config.json``'s ``training.n_iters`` is ``N``; ``optim.lr`` is 1e-4 (D19); the manifest's
``recipe_sha256`` is the hash of the recipe in ``config.json``; the manifest names the cell of its
row. Across cells: the recipe is identical apart from :data:`CELL_KEYS`, and the arm keys agree
within every arm. Nothing is written. Exit 0 only if every cell is healthy (T3.5, D22).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ihdm.paths import repo_root
from ihdm.train.recipe import recipe_diff, recipe_of
from ihdm.train.validate_run import (
    TOP_LEVEL,
    _check_grids,
    _check_manifest,
    _expected_steps,
    check_metrics,
    check_run_dir,
    expectation_from_config,
    read_metrics_strict,
)

__all__ = ["ARM_KEYS", "CELL_KEYS", "EXPECTED_LR", "CellReport", "Cell", "check_array", "main"]

#: The post-warm-up lr of recipe v2 (D19).
EXPECTED_LR: float = 1e-4

#: Recipe keys that belong to the arm: equal within an arm, free across arms.
ARM_KEYS: tuple[str, ...] = (
    "model.blur_schedule",
    "model.blur_schedule_name",
    "model.blur_schedule_sha256",
    "model.blur_sigma_max",
)

#: Recipe keys that may differ between cells (on top of those ``recipe_of`` already drops:
#: ``run_id``, the path and host keys, the iteration count and the cadences).
CELL_KEYS: frozenset[str] = frozenset({"arm", "dataset_id", "data.dataset", "seed", *ARM_KEYS})

#: Exit codes.
EXIT_HEALTHY, EXIT_UNHEALTHY, EXIT_USAGE = 0, 1, 2


class CheckArrayError(Exception):
    """Raised when the cell table cannot be used."""


@dataclass(frozen=True)
class Cell:
    """One row of ``cells.csv``."""

    index: int
    run_id: str
    dataset_id: str
    arm: str
    seed: int


@dataclass
class CellReport:
    """What the check found for one cell.

    Parameters
    ----------
    cell : Cell
        The row.
    present : bool
        The run directory exists.
    done_marker : bool
        ``DONE`` exists.
    done_step : int or None
        The step of the last ``done`` event.
    ema_found, ema_expected : int
        EMA checkpoints present among those expected every ``ckpt_every`` up to ``N``.
    last_ema : int
        The largest EMA checkpoint step (0 when none).
    full_final : bool
        ``checkpoints/full_final.pt`` exists and is non-empty.
    n_abort, n_skip : int
        ``abort`` and ``skip`` events.
    lr : float or None
        ``config.json`` ``optim.lr``.
    recipe : str
        ``ok``, ``drift`` or ``-`` (no config).
    last_train : dict
        The last ``train`` line (step, loss, grad_norm, amp_scale), empty when none.
    problems : list[str]
        Every problem found; the cell is healthy when this is empty.
    """

    cell: Cell
    present: bool = False
    done_marker: bool = False
    done_step: int | None = None
    ema_found: int = 0
    ema_expected: int = 0
    last_ema: int = 0
    full_final: bool = False
    n_abort: int = 0
    n_skip: int = 0
    lr: float | None = None
    recipe: str = "-"
    last_train: dict[str, Any] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    shared_recipe: dict[str, Any] | None = None
    arm_values: str | None = None

    @property
    def healthy(self) -> bool:
        """No problem was found."""
        return not self.problems

    @property
    def status(self) -> str:
        """``ok``, ``absent``, ``running`` (no ``DONE``) or ``FAIL``."""
        if self.healthy:
            return "ok"
        if not self.present:
            return "absent"
        if not self.done_marker:
            return "running"
        return "FAIL"


def read_cells(path: Path) -> list[Cell]:
    """Read ``cells.csv`` in file order.

    Parameters
    ----------
    path : Path
        The cell table (``index,run_id,dataset_id,arm,seed,tier``).

    Returns
    -------
    list[Cell]
        One per data row, in file order.

    Raises
    ------
    CheckArrayError
        If the file is missing, has no rows, or a row disagrees with its own run id.
    """
    if not Path(path).is_file():
        raise CheckArrayError(f"no cell table at {path}")
    cells = []
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                cell = Cell(int(row["index"]), row["run_id"], row["dataset_id"], row["arm"],
                            int(row["seed"]))
            except (KeyError, TypeError, ValueError) as error:
                raise CheckArrayError(f"{path}: unusable row {row}") from error
            if cell.run_id != f"{cell.dataset_id}_{cell.arm}_s{cell.seed}":
                raise CheckArrayError(f"{path}: run_id {cell.run_id} disagrees with its row")
            cells.append(cell)
    if not cells:
        raise CheckArrayError(f"{path} has no data rows")
    return cells


def _recipe_hash(config_json: dict[str, Any]) -> str:
    """The ``recipe_sha256`` of a JSON-form config (as :func:`ihdm.train.recipe.recipe_sha256`)."""
    payload = json.dumps(recipe_of(config_json), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _ema_steps(workdir: Path) -> list[int]:
    steps = []
    for path in (workdir / "checkpoints").glob("ema_iter_*.pt"):
        digits = path.stem.removeprefix("ema_iter_")
        if digits.isdigit():
            steps.append(int(digits))
    return sorted(steps)


def _light_files(workdir: Path, exp: Any) -> list[str]:
    """File presence without loading: the torch-free half of ``_check_checkpoints``."""
    problems = []
    for step in sorted(_expected_steps(exp)["ckpt"]):
        if not (workdir / "checkpoints" / f"ema_iter_{step:06d}.pt").is_file():
            problems.append(f"missing ema_iter_{step:06d}.pt")
    final = workdir / "checkpoints" / "full_final.pt"
    if not (final.is_file() and final.stat().st_size > 0):
        problems.append("missing or empty checkpoints/full_final.pt")
    if not (workdir / "checkpoints-meta" / "checkpoint.pth").is_file():
        problems.append("missing checkpoints-meta/checkpoint.pth")
    if (workdir / "DONE").stat().st_size != 0:
        problems.append("DONE is not empty")
    return problems


def _facts_from_metrics(report: CellReport, records: list[dict[str, Any]]) -> None:
    done = [r for r in records if r.get("kind") == "done"]
    report.done_step = done[-1].get("step") if done else None
    report.n_abort = sum(r.get("kind") == "abort" for r in records)
    report.n_skip = sum(r.get("kind") == "skip" for r in records)
    train = [r for r in records if r.get("kind") == "train"]
    if train:
        last = train[-1]
        report.last_train = {k: last.get(k) for k in ("step", "loss", "grad_norm", "amp_scale")}


def _check_identity(report: CellReport, config: dict[str, Any], manifest: dict[str, Any]) -> None:
    cell = report.cell
    want = {"run_id": cell.run_id, "dataset_id": cell.dataset_id, "arm": cell.arm,
            "seed": cell.seed}
    for key, value in want.items():
        if manifest.get(key) != value:
            report.problems.append(f"manifest {key} {manifest.get(key)!r} != cells.csv {value!r}")
    if config.get("data", {}).get("dataset") != cell.dataset_id:
        report.problems.append(f"config data.dataset {config.get('data', {}).get('dataset')!r} "
                               f"!= {cell.dataset_id!r}")


def check_cell(cell: Cell, run_root: Path, n_iters: int, *, allow_skips: bool = False,
               data_root: Path | None = None, deep: bool = False) -> CellReport:
    """Check one cell's run directory.

    Parameters
    ----------
    cell : Cell
        The row of ``cells.csv``.
    run_root : Path
        The directory that holds one folder per ``run_id``.
    n_iters : int
        The step every run must have finished at.
    allow_skips : bool
        Tolerate ``skip`` events (``abort`` events are never tolerated).
    data_root : Path or None
        When given, the manifest's data hash is compared with the dataset's ``meta.json``.
    deep : bool
        Load the last EMA checkpoint and ``full_final.pt`` (``validate_run.check_run_dir``).

    Returns
    -------
    CellReport
        The facts and problems of the cell.
    """
    report = CellReport(cell=cell)
    workdir = Path(run_root) / cell.run_id
    report.present = workdir.is_dir()
    if not report.present:
        report.problems.append(f"no run directory {workdir}")
        return report
    report.done_marker = (workdir / "DONE").is_file()
    ema = _ema_steps(workdir)
    report.last_ema = ema[-1] if ema else 0
    final = workdir / "checkpoints" / "full_final.pt"
    report.full_final = final.is_file() and final.stat().st_size > 0

    present = {p.name for p in workdir.iterdir()}
    missing = sorted((TOP_LEVEL | {"DONE"}) - present)
    config_path, metrics_path = workdir / "config.json", workdir / "metrics.jsonl"
    config = json.loads(config_path.read_text()) if config_path.is_file() else None
    records: list[dict[str, Any]] = []
    parse_problems: list[str] = []
    if metrics_path.is_file():
        records, parse_problems = read_metrics_strict(metrics_path)
        _facts_from_metrics(report, records)
    if missing:
        report.problems.append(f"missing top-level entries {missing}")
    if config is None:
        return report

    report.lr = float(config["optim"]["lr"])
    exp = expectation_from_config(config, n_iters=n_iters, allow_skips=allow_skips)
    expected_ema = sorted(_expected_steps(exp)["ckpt"])
    report.ema_expected = len(expected_ema)
    report.ema_found = sum(step in set(ema) for step in expected_ema)
    report.shared_recipe = {k: v for k, v in recipe_of(config).items() if k not in CELL_KEYS}
    flat = recipe_of(config)
    report.arm_values = json.dumps({k: flat.get(k) for k in ARM_KEYS}, sort_keys=True)

    if int(config["training"]["n_iters"]) != n_iters:
        report.problems.append(f"config training.n_iters {config['training']['n_iters']} != "
                               f"{n_iters}")
    if not math.isclose(report.lr, EXPECTED_LR, rel_tol=1e-12):
        report.problems.append(f"config optim.lr {report.lr} != {EXPECTED_LR}")
    if report.last_ema != n_iters:
        report.problems.append(f"largest EMA checkpoint {report.last_ema} != {n_iters}")
    if missing:
        return report

    manifest = json.loads((workdir / "manifest.json").read_text())
    _check_identity(report, config, manifest)
    if manifest.get("recipe_sha256") != _recipe_hash(config):
        report.problems.append("manifest recipe_sha256 != the hash of config.json's recipe")
    if deep:
        report.problems += check_run_dir(workdir, exp, data_root)
    else:
        report.problems += parse_problems + check_metrics(records, exp)
        report.problems += _check_manifest(workdir, config, exp, data_root)
        report.problems += _light_files(workdir, exp)
        report.problems += _check_grids(workdir, exp, int(config["data"]["image_size"]))
    return report


def _cross_cell(reports: list[CellReport]) -> None:
    """Flag recipe drift across cells and arm keys that disagree within an arm."""
    with_recipe = [r for r in reports if r.shared_recipe is not None]
    if not with_recipe:
        return
    canon = {id(r): json.dumps(r.shared_recipe, sort_keys=True) for r in with_recipe}
    reference_text, _ = Counter(canon.values()).most_common(1)[0]
    reference = next(r for r in with_recipe if canon[id(r)] == reference_text)
    for report in with_recipe:
        if canon[id(report)] == reference_text:
            report.recipe = "ok"
            continue
        report.recipe = "drift"
        diff = recipe_diff(reference.shared_recipe or {}, report.shared_recipe or {})
        report.problems.append(f"recipe differs from {reference.cell.run_id}'s: "
                               + "; ".join(diff[:5]) + (" ..." if len(diff) > 5 else ""))
    by_arm: dict[str, list[CellReport]] = {}
    for report in with_recipe:
        by_arm.setdefault(report.cell.arm, []).append(report)
    for arm, members in by_arm.items():
        values = Counter(r.arm_values for r in members)
        if len(values) > 1:
            majority, _ = values.most_common(1)[0]
            for report in members:
                if report.arm_values != majority:
                    report.problems.append(f"arm keys differ from the other {arm} cells: "
                                           f"{report.arm_values}")


def check_array(cells: list[Cell], run_root: Path, n_iters: int, *, allow_skips: bool = False,
                data_root: Path | None = None, deep: bool = False) -> list[CellReport]:
    """Check every cell, then the cross-cell recipe.

    Parameters
    ----------
    cells : list[Cell]
        The rows of ``cells.csv``, in file order.
    run_root, n_iters, allow_skips, data_root, deep
        As in :func:`check_cell`.

    Returns
    -------
    list[CellReport]
        One per cell, in the order of ``cells``.
    """
    reports = [check_cell(cell, run_root, n_iters, allow_skips=allow_skips, data_root=data_root,
                          deep=deep) for cell in cells]
    _cross_cell(reports)
    return reports


def _fmt(value: Any, spec: str) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return format(value, spec)
    return str(value)


def format_table(reports: list[CellReport]) -> str:
    """Render one row per cell, then every problem under its run id."""
    header = (f"{'idx':>3} {'run_id':<20} {'status':<7} {'done@':>6} {'ema':>6} {'final':<5} "
              f"{'abort':>5} {'skip':>4} {'lr':>7} {'recipe':<6} {'last':>6} {'loss':>7} "
              f"{'grad_norm':>9} {'amp_scale':>9}")
    lines = [header, "-" * len(header)]
    for r in reports:
        t = r.last_train
        lines.append(
            f"{r.cell.index:>3} {r.cell.run_id:<20} {r.status:<7} {_fmt(r.done_step, 'd'):>6} "
            f"{f'{r.ema_found}/{r.ema_expected}':>6} {('yes' if r.full_final else 'no'):<5} "
            f"{r.n_abort:>5} {r.n_skip:>4} {_fmt(r.lr, '.0e'):>7} {r.recipe:<6} "
            f"{_fmt(t.get('step'), 'd'):>6} {_fmt(t.get('loss'), '.4f'):>7} "
            f"{_fmt(t.get('grad_norm'), '.1f'):>9} {_fmt(t.get('amp_scale'), 'g'):>9}"
        )
    flagged = [r for r in reports if not r.healthy]
    if flagged:
        lines.append("")
        lines.append("problems:")
        for r in flagged:
            shown = r.problems[:8]
            more = f" (+{len(r.problems) - 8} more)" if len(r.problems) > 8 else ""
            lines.append(f"  {r.cell.run_id}: " + " | ".join(shown) + more)
    return "\n".join(lines)


def verdict(reports: list[CellReport], n_iters: int) -> str:
    """One line: HEALTHY, or UNHEALTHY with the counts by status."""
    healthy = sum(r.healthy for r in reports)
    if healthy == len(reports):
        return f"VERDICT: HEALTHY {healthy}/{len(reports)} cells at {n_iters}"
    counts = Counter(r.status for r in reports if not r.healthy)
    parts = ", ".join(f"{n} {status}" for status, n in sorted(counts.items()))
    return (f"VERDICT: UNHEALTHY {healthy}/{len(reports)} cells healthy at {n_iters} "
            f"({parts})")


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser of the command."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-root", required=True, type=Path,
                        help="directory holding one folder per run_id")
    parser.add_argument("--cells", type=Path,
                        default=repo_root() / "slurm" / "array" / "cells.csv",
                        help="the cell table (default: slurm/array/cells.csv)")
    parser.add_argument("--n-iters", required=True, type=int,
                        help="the step every run must be DONE at (40000, 60000)")
    parser.add_argument("--allow-skips", action="store_true",
                        help="tolerate skip events (abort events are always refused)")
    parser.add_argument("--data-root", type=Path,
                        default=Path(os.environ["IHDM_DATA_ROOT"])
                        if os.environ.get("IHDM_DATA_ROOT") else None,
                        help="compare data.images_sha256 with <data-root>/<dataset>/meta.json "
                        "(default: $IHDM_DATA_ROOT when set)")
    parser.add_argument("--deep", action="store_true",
                        help="also load the last EMA checkpoint and full_final.pt "
                        "(validate_run.check_run_dir; ~1.2 GB read per run)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Check the array, print the table and the verdict.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        0 if every cell is healthy, 1 otherwise, 2 if the cell table is unusable.
    """
    args = build_parser().parse_args(argv)
    try:
        cells = read_cells(args.cells)
    except CheckArrayError as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return EXIT_USAGE
    mode = "deep" if args.deep else "light"
    print(f"check_array: {len(cells)} cells of {args.cells}, run root {args.run_root}, "
          f"n_iters {args.n_iters}, {mode}, allow_skips={args.allow_skips}, "
          f"data_root={args.data_root}")
    reports = check_array(cells, args.run_root, args.n_iters, allow_skips=args.allow_skips,
                          data_root=args.data_root, deep=args.deep)
    print(format_table(reports))
    print(verdict(reports, args.n_iters))
    return EXIT_HEALTHY if all(r.healthy for r in reports) else EXIT_UNHEALTHY


if __name__ == "__main__":
    sys.exit(main())
