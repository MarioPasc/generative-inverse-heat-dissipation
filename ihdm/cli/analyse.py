"""Write the report's tables and statistics from one collected ``results/`` folder (T6.1).

``python -m ihdm.cli.analyse --results <results_dir> --out docs/RESULTS/tables/``

Thin shell of :mod:`ihdm.analysis.tables`. It writes ``<table>.md`` and ``<table>.tex`` per table
and one ``tables.json`` into ``--out``, overwriting its own earlier output and leaving every other
file there (the hand-written README) alone. Exit codes: 0 every table complete; 1 the folder
cannot be analysed (its collection verdict is FAIL, or ``index.csv`` disagrees with the runs);
2 ``--results`` is not a results folder; 3 the tables were written but at least one is
``incomplete: n/N runs`` (a partial collection).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ihdm.analysis.tables import AnalysisError, ResultsNotFound, analyse

__all__ = ["EXIT_FAIL", "EXIT_OK", "EXIT_PARTIAL", "EXIT_USAGE", "build_parser", "main"]

EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_PARTIAL = 0, 1, 2, 3


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser of the command.

    Returns
    -------
    argparse.ArgumentParser
        The parser.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, required=True,
                        help="the folder written by python -m ihdm.cli.collect_results")
    parser.add_argument("--out", type=Path, required=True,
                        help="where the tables go (e.g. docs/RESULTS/tables/)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build and write the tables, print one status line per table, return the exit code.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        0 complete, 1 unusable results, 2 not a results folder, 3 written but incomplete.
    """
    args = build_parser().parse_args(argv)
    try:
        report = analyse(args.results, args.out)
    except ResultsNotFound as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return EXIT_USAGE
    except AnalysisError as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return EXIT_FAIL
    print(report.summary())
    return EXIT_OK if report.complete else EXIT_PARTIAL


if __name__ == "__main__":
    sys.exit(main())
