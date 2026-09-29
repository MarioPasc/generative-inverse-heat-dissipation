"""Draw the report's figures from a collected ``results/`` folder (T6.2).

``python -m ihdm.cli.figures --results <results_dir> --out docs/RESULTS/figures/ [--readme PATH]
[--no-pdf] [--png-dpi 200] [--only fig1_lsd_vs_iteration ...]``

Thin shell of :mod:`ihdm.analysis.figures`. Writes ``<out>/<figure>.pdf`` and ``.png`` for the
seven figures and the caption drafts in ``<out>/README.md`` (or ``--readme``). Exit codes: 0 every
figure drawn from every run of the design; 3 every figure drawn, but some runs were absent (the
README names them); 1 the results folder is unreadable; 2 unusable arguments.
"""

from __future__ import annotations

import argparse
import logging
import shlex
import sys
from pathlib import Path

from ihdm.analysis.figures import FIGURES, FigureError, draw_all, load_results, write_readme
from ihdm.analysis.style import PNG_DPI

__all__ = ["EXIT_FAIL", "EXIT_OK", "EXIT_PARTIAL", "EXIT_USAGE", "build_parser", "main"]

EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_PARTIAL = 0, 1, 2, 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m ihdm.cli.figures", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True,
                        help="results/ folder written by ihdm.cli.collect_results")
    parser.add_argument("--out", type=Path, required=True, help="output folder of the figures")
    parser.add_argument("--readme", type=Path, default=None,
                        help="caption README to write (default: <out>/README.md)")
    parser.add_argument("--no-pdf", action="store_true",
                        help="write the PNG previews only (partial collections)")
    parser.add_argument("--png-dpi", type=int, default=PNG_DPI,
                        help=f"resolution of the PNG previews (default {PNG_DPI})")
    parser.add_argument("--only", nargs="+", choices=[name for name, _ in FIGURES],
                        help="draw only these figures")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return EXIT_OK if exc.code == 0 else EXIT_USAGE
    if args.png_dpi < 30:
        print("--png-dpi must be at least 30", file=sys.stderr)
        return EXIT_USAGE
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    # fontTools warns about the epoch timestamps of the subset fonts in every PDF; not actionable.
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    try:
        results = load_results(args.results)
    except (FigureError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_FAIL
    records = draw_all(results, args.out, pdf=not args.no_pdf, png_dpi=args.png_dpi,
                       only=args.only)
    command = "python -m ihdm.cli.figures " + shlex.join(
        sys.argv[1:] if argv is None else argv)
    readme = write_readme(records, results, args.readme or args.out / "README.md", command)
    for record in records:
        absent = f"; absent: {', '.join(record.missing)}" if record.missing else ""
        print(f"figure {record.number} {record.name}: {record.coverage}{absent}")
    print(f"README: {readme}")
    partial = any(record.missing for record in records)
    print("VERDICT:", "PARTIAL (some runs absent)" if partial else "COMPLETE")
    return EXIT_PARTIAL if partial else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
