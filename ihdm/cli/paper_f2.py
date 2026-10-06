"""Build F2 (the four arms on brain MRI), T1 (the main table) and their notes (T8.2).

``python -m ihdm.cli.paper_f2 --data-root <root> --results <_results> --eval-dir <tars>
--heldout <json> [<json> ...] --out <dir> [--work <scratch>] [--tables <tables.json>]``

``python -m ihdm.cli.paper_f2 a2-heldout --eval-dir <tars> --data-root <root> --work <T7.5 cache>
--out <dir> --runs <ids> [--reference <T7.5 json>]`` runs the unchanged T7.5 CLI with +spacing
(A2) admitted, leaving the shared cache's own files unchanged.

Exit codes: 0 written; 1 analysis error (including a failed CRN, selection, expectation or
layout assertion); 2 missing input.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

from ihdm.analysis.paper_f2 import (
    F2Paths,
    MissingInputError,
    PaperF2Error,
    build,
    run_a2_heldout,
)
from ihdm.analysis.tables import AnalysisError
from ihdm.paths import repo_root

__all__ = ["EXIT_FAIL", "EXIT_MISSING", "EXIT_OK", "build_parser", "main"]

EXIT_OK, EXIT_FAIL, EXIT_MISSING = 0, 1, 2
A2_COMMAND: str = "a2-heldout"


def _default_tables() -> Path:
    return repo_root() / "docs" / "RESULTS" / "tables" / "tables.json"


def build_parser() -> argparse.ArgumentParser:
    """Return the parser of the figure-and-table command."""
    parser = argparse.ArgumentParser(prog="python -m ihdm.cli.paper_f2", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="dataset root (holds ixi/images.npy)")
    parser.add_argument("--results", type=Path, required=True,
                        help="the _results folder of collect_results (index.csv, runs/)")
    parser.add_argument("--eval-dir", type=Path, required=True,
                        help="folder of the evaluation tars <run>_amp-fp16.tar")
    parser.add_argument("--heldout", type=Path, nargs="+", required=True,
                        help="T7.5 heldout_fidelity.json files (T7.5 first, then the A2 extension)")
    parser.add_argument("--out", type=Path, required=True, help="output folder")
    parser.add_argument("--work", type=Path, default=None,
                        help="scratch folder for the extracted sample sets "
                             "(default: <tmp>/ihdm_paper_f2)")
    parser.add_argument("--tables", type=Path, default=None,
                        help="tables.json for the T1 cross-check "
                             "(default: docs/RESULTS/tables/tables.json)")
    return parser


def build_a2_parser() -> argparse.ArgumentParser:
    """Return the parser of the ``a2-heldout`` subcommand."""
    parser = argparse.ArgumentParser(prog=f"python -m ihdm.cli.paper_f2 {A2_COMMAND}")
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True, help="the shared T7.5 cache")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", required=True, help="comma-separated run ids")
    parser.add_argument("--reference", type=Path, default=None,
                        help="T7.5 heldout_fidelity.json (default: docs/RESULTS/heldout_fidelity/)")
    return parser


def _run_a2(argv: list[str]) -> int:
    args = build_a2_parser().parse_args(argv)
    reference = args.reference or (repo_root() / "docs" / "RESULTS" / "heldout_fidelity"
                                   / "heldout_fidelity.json")
    if not reference.is_file():
        print(f"FATAL: {reference} not found", file=sys.stderr)
        return EXIT_MISSING
    code = run_a2_heldout(args.eval_dir, args.data_root, args.work, args.out, args.runs,
                          reference)
    if code == EXIT_OK:
        print(f"A2 extension written to {args.out}; A0 equals {reference}; cache unchanged")
    return code


def _run(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    paths = F2Paths(
        data_root=args.data_root, results=args.results, eval_dir=args.eval_dir,
        heldout=tuple(args.heldout), out=args.out,
        work=args.work or Path(tempfile.gettempdir()) / "ihdm_paper_f2",
        tables_json=args.tables or _default_tables(),
    )
    report = build(paths)
    a = report.audit
    print(f"F2: {a.width_in:.2f} x {a.height_in:.2f} in, smallest font {a.min_font_pt:g} pt, "
          f"smallest thumbnail {a.min_thumb_in:.2f} in; rows "
          f"{[r.image_index for r in report.rows]}; {report.selection}")
    for name, digest in report.sha256().items():
        print(f"  {name}  {digest}")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    """Run the command and return its exit code.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        0 written, 1 analysis error, 2 missing input.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if argv and argv[0] == A2_COMMAND:
            return _run_a2(argv[1:])
        return _run(argv)
    except MissingInputError as error:
        print(f"FATAL (missing input): {error}", file=sys.stderr)
        return EXIT_MISSING
    except (PaperF2Error, AnalysisError) as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
