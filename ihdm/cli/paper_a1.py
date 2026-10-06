"""Build appendix A1 of the paper: Figure A1, Table A1 and their notes (T8.3).

``python -m ihdm.cli.paper_a1 --results <_results> --eval-dir <tars> --out <dir>
[--work <scratch>] [--photo-diagnostic <json>]``

Exit codes: 0 written; 1 analysis error (including a failed CRN, setup or layout assertion);
2 missing input.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

from ihdm.analysis.paper_a1 import A1Paths, MissingInputError, PaperA1Error, build
from ihdm.analysis.tables import AnalysisError
from ihdm.paths import repo_root

__all__ = ["EXIT_FAIL", "EXIT_MISSING", "EXIT_OK", "build_parser", "main"]

EXIT_OK, EXIT_FAIL, EXIT_MISSING = 0, 1, 2


def _default_photo_diagnostic() -> Path:
    return repo_root() / "docs" / "RESULTS" / "photo_diagnostic" / "photo_diagnostic.json"


def build_parser() -> argparse.ArgumentParser:
    """Return the parser of the appendix command."""
    parser = argparse.ArgumentParser(prog="python -m ihdm.cli.paper_a1", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True,
                        help="the _results folder of collect_results (index.csv, runs/)")
    parser.add_argument("--eval-dir", type=Path, required=True,
                        help="folder of the evaluation tars <run>_amp-fp16.tar")
    parser.add_argument("--out", type=Path, required=True, help="output folder")
    parser.add_argument("--work", type=Path, default=None,
                        help="scratch folder for the extracted LSD sets "
                             "(default: <tmp>/ihdm_paper_a1)")
    parser.add_argument("--photo-diagnostic", type=Path, default=None,
                        help="photo_diagnostic.json of M7 "
                             "(default: docs/RESULTS/photo_diagnostic/photo_diagnostic.json)")
    return parser


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
    # Font subsetting in the PDF/SVG writers logs every pruned table at INFO.
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    paths = A1Paths(
        results=args.results, eval_dir=args.eval_dir, out=args.out,
        work=args.work or Path(tempfile.gettempdir()) / "ihdm_paper_a1",
        photo_diagnostic=args.photo_diagnostic or _default_photo_diagnostic(),
    )
    try:
        report = build(paths)
    except MissingInputError as error:
        print(f"FATAL (missing input): {error}", file=sys.stderr)
        return EXIT_MISSING
    except (PaperA1Error, AnalysisError) as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return EXIT_FAIL
    a, n = report.audit, report.numbers
    print(f"A1: {a.width_in:.2f} x {a.height_in:.2f} in, smallest font {a.min_font_pt:g} pt, "
          f"smallest thumbnail {a.min_thumb_in:.2f} in; images "
          f"{[(r.run_id, r.image_index) for r in report.rows]}; plateau {n.plateau_step} it "
          f"({n.plateau_epoch:g} epochs); precision photo max {n.photo_max:.4f}, MRI "
          f"{n.mri_min:.4f}-{n.mri_max:.4f}")
    for name, digest in report.sha256().items():
        print(f"  {name}  {digest}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
