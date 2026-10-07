"""Build F3 (fidelity against diversity) and its notes ``F3.md`` (T8.5).

``python -m ihdm.cli.paper_f3 --results <_results> --heldout <json> [<json> ...]
--tables <tables.json> --exploratory <exploratory.json>
--constants <inherited_band_constants.json> --out <dir>``

Every option but ``--results`` defaults to the committed file under ``docs/RESULTS/``.

Exit codes: 0 written; 1 analysis error (including a failed cross-check, expectation or layout
assertion); 2 missing input.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ihdm.analysis.paper_f3 import F3MissingInputError, F3Paths, PaperF3Error, build
from ihdm.analysis.tables import AnalysisError
from ihdm.paths import repo_root

__all__ = ["EXIT_FAIL", "EXIT_MISSING", "EXIT_OK", "build_parser", "main"]

EXIT_OK, EXIT_FAIL, EXIT_MISSING = 0, 1, 2


def _results_dir() -> Path:
    return repo_root() / "docs" / "RESULTS"


def build_parser() -> argparse.ArgumentParser:
    """Return the parser of the command."""
    res = _results_dir()
    parser = argparse.ArgumentParser(prog="python -m ihdm.cli.paper_f3", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True,
                        help="the _results folder of collect_results (index.csv, runs/)")
    parser.add_argument("--heldout", type=Path, nargs="+",
                        default=[res / "heldout_fidelity" / "heldout_fidelity.json",
                                 res / "paper" / "f2_a2_heldout" / "heldout_fidelity.json"],
                        help="T7.5 heldout_fidelity.json files (T7.5 first, then the A2 extension)")
    parser.add_argument("--tables", type=Path, default=res / "tables" / "tables.json",
                        help="tables.json (table 1c: I_w per run)")
    parser.add_argument("--exploratory", type=Path,
                        default=res / "exploratory" / "exploratory.json",
                        help="exploratory.json (per_run regen_ratio and log_vr)")
    parser.add_argument("--constants", type=Path, default=res / "inherited_band_constants.json",
                        help="inherited_band_constants.json (sum P_ref, I)")
    parser.add_argument("--out", type=Path, default=res / "paper", help="output folder")
    return parser


def _run(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    paths = F3Paths(results=args.results, heldout=tuple(args.heldout), tables_json=args.tables,
                    exploratory_json=args.exploratory, constants=args.constants, out=args.out)
    report = build(paths)
    a = report.audit
    print(f"F3: {a.width_in:.2f} x {a.height_in:.2f} in, smallest font {a.min_font_pt:g} pt, "
          f"{a.n_texts} texts, no overlaps; denominator "
          f"{'same' if report.denominator['same'] else 'DIFFERENT'}")
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
        return _run(argv)
    except F3MissingInputError as error:
        print(f"FATAL (missing input): {error}", file=sys.stderr)
        return EXIT_MISSING
    except (PaperF3Error, AnalysisError) as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
