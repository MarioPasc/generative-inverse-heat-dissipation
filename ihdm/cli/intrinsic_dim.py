"""Measure the intrinsic dimension of the four training splits (T8.0).

``python -m ihdm.cli.intrinsic_dim --data-root <root> --out <dir>`` writes
``intrinsic_dimension.json`` and ``id_vs_n.{svg,pdf,png}`` into ``<dir>``. CPU only; one dataset
at a time (about 1 GB and one minute of Gram matrix per dataset at two BLAS threads).

Exit codes: 0 written; 1 analysis error; 2 missing input.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ihdm.analysis.intrinsic_dim import IDConfig, MissingInputError, run
from ihdm.analysis.tables import AnalysisError

__all__ = ["EXIT_FAIL", "EXIT_MISSING", "EXIT_OK", "build_parser", "main"]

EXIT_OK, EXIT_FAIL, EXIT_MISSING = 0, 1, 2


def build_parser() -> argparse.ArgumentParser:
    """Return the command's parser."""
    parser = argparse.ArgumentParser(prog="python -m ihdm.cli.intrinsic_dim", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="dataset root holding ixi/, oasis1/, lsun_church/, lsun_bedroom/")
    parser.add_argument("--out", type=Path, required=True, help="output folder")
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
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    try:
        doc = run(args.data_root, args.out, IDConfig())
    except MissingInputError as error:
        print(f"FATAL (missing input): {error}", file=sys.stderr)
        return EXIT_MISSING
    except AnalysisError as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return EXIT_FAIL
    for name, entry in doc["datasets"].items():
        full = entry["mle"]["k=10"]["N=3200"]["mean"]
        lin = entry["linear"]
        print(f"{name:13s} MLE k=10 N=3200 {full:6.2f}  PR {lin['participation_ratio']:6.2f}  "
              f"90% {lin['components_for_fraction']['0.9']:5d}")
    for statement, value in doc["reading"].items():
        if isinstance(value, dict):
            print(f"reading: {statement}: {'holds' if value['holds'] else 'DOES NOT HOLD'}")
    for name, digest in doc["_files"].items():
        print(f"  {name}  {digest}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
