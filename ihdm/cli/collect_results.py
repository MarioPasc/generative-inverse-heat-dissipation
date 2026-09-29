"""Collect the evaluated runs into one checked ``results/`` folder (T5.2).

``python -m ihdm.cli.collect_results --eval-dir <dir> --run-root <root> --out <results>
[--gate-dir <dir>] [--cells slurm/array/cells.csv] [--n-iters 60000] [--amp fp16]
[--allow-missing]``

Thin shell of :mod:`ihdm.analysis.collect`. Exit codes: 0 every check passed and the folder was
published; 1 a check failed or an input is absent (nothing is published, except under
``--allow-missing``); 2 unusable arguments or tables; 3 only absent inputs, and
``--allow-missing`` published a partial folder.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ihdm.analysis.collect import (
    DEFAULT_GATE_CELLS,
    CollectConfig,
    CollectError,
    collect,
    format_report,
)
from ihdm.metrics.run_eval import AMP_MODES
from ihdm.paths import repo_root

__all__ = ["EXIT_FAIL", "EXIT_OK", "EXIT_PARTIAL", "EXIT_USAGE", "build_parser", "main"]

EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_PARTIAL = 0, 1, 2, 3


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def _gate_cells(text: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in text.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"--gate-cells wants integers, got {text!r}") from error


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser of the command.

    Returns
    -------
    argparse.ArgumentParser
        The parser.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    eval_dir, run_root = _env_path("IHDM_EVAL_HOME"), _env_path("IHDM_RUN_ROOT")
    parser.add_argument("--eval-dir", type=Path, default=eval_dir, required=eval_dir is None,
                        help="the per-run summaries and tars (default: $IHDM_EVAL_HOME)")
    parser.add_argument("--gate-dir", type=Path, default=None,
                        help="the gate JSONs (default: <eval-dir>/gate)")
    parser.add_argument("--run-root", type=Path, default=run_root, required=run_root is None,
                        help="one training run directory per run_id (default: $IHDM_RUN_ROOT)")
    parser.add_argument("--cells", type=Path,
                        default=repo_root() / "slurm" / "array" / "cells.csv",
                        help="the cell table (default: slurm/array/cells.csv)")
    parser.add_argument("--expected-seed-lists", type=Path,
                        default=repo_root() / "slurm" / "eval" / "expected_seed_lists.csv",
                        help="seed-list digests (default: slurm/eval/expected_seed_lists.csv)")
    parser.add_argument("--out", type=Path, required=True,
                        help="the results folder to publish; must not exist")
    parser.add_argument("--n-iters", type=int, default=60000,
                        help="run length; the evaluated steps are every 5000 up to it "
                        "(default 60000, D22)")
    parser.add_argument("--amp", choices=AMP_MODES, default="fp16",
                        help="the sampling precision of every run (default fp16, D20)")
    parser.add_argument("--gate-cells", type=_gate_cells, default=DEFAULT_GATE_CELLS,
                        help="cells whose gates are required (default 0,3)")
    parser.add_argument("--allow-missing", action="store_true",
                        help="dry runs only: tolerate absent inputs and publish a partial "
                        "folder marked complete=false (exit 3)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the collection, print the report, return the exit code.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        0 complete, 1 a check failed, 2 unusable input, 3 partial (``--allow-missing``).
    """
    args = build_parser().parse_args(argv)
    config = CollectConfig(
        eval_dir=args.eval_dir,
        gate_dir=args.gate_dir if args.gate_dir is not None else args.eval_dir / "gate",
        run_root=args.run_root,
        cells=args.cells,
        out=args.out,
        n_iters=args.n_iters,
        amp=args.amp,
        allow_missing=args.allow_missing,
        expected_seed_lists=args.expected_seed_lists,
        gate_cells=args.gate_cells,
    )
    try:
        report = collect(config)
    except CollectError as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return EXIT_USAGE
    print(format_report(report))
    if report.verdict == "COMPLETE":
        return EXIT_OK
    # An absent input is an integrity failure unless the dry-run flag published a partial folder.
    return EXIT_PARTIAL if report.verdict == "INCOMPLETE" and args.allow_missing else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
