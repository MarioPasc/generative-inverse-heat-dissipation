"""Draw FM of the course paper, the method figure, and write its record ``FM.md`` (T8.4).

``python -m ihdm.cli.paper_fm --data-root <IHDM data root> --out <dir> [--eval-dir <tars>]
[--work <scratch>]``

Thin shell of :mod:`ihdm.analysis.paper_fm`. It selects F1's IXI slice, verifies the inputs (the
octave shares against the profile's power array, the level counts against
``data_profile.md`` §5 and F1's ticket), checks the macro-step identities with the released
``DCTBlur``, and writes ``<out>/fm_method.{pdf,svg,png}`` and ``<out>/FM.md``. FM draws no model
sample, so ``--eval-dir`` and ``--work`` are accepted (the frozen interface of the paper figures)
and not read.

Exit codes: 0 written; 1 analysis error (an inconsistent input or a failed verification); 2
missing input or unusable arguments.
"""

from __future__ import annotations

import argparse
import logging
import shlex
import sys
from pathlib import Path

from ihdm.analysis import paper_f1 as f1
from ihdm.analysis import paper_fm as fm

__all__ = ["EXIT_FAIL", "EXIT_MISSING", "EXIT_OK", "REPO_ROOT", "build_parser", "main"]

EXIT_OK, EXIT_FAIL, EXIT_MISSING = 0, 1, 2

#: Repository root holding ``docs/RESULTS`` and ``schedules`` (this file is ihdm/cli/paper_fm.py).
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

logger = logging.getLogger("ihdm.cli.paper_fm")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m ihdm.cli.paper_fm", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="IHDM data root holding ixi/images.npy, index.csv, splits.json")
    parser.add_argument("--out", type=Path, required=True, help="output folder")
    parser.add_argument("--eval-dir", type=Path, default=None,
                        help="folder of the evaluation tars (accepted; FM draws no sample)")
    parser.add_argument("--work", type=Path, default=None,
                        help="scratch folder (accepted; FM extracts nothing)")
    return parser


def _run(args: argparse.Namespace, command: str) -> int:
    inputs = fm.FMInputs(args.data_root, REPO_ROOT)
    data = fm.load_fm_data(inputs)
    checks = fm.verify_macro_steps(data.width)
    failed = [c.label for c in checks if not c.passed]
    if failed:
        raise fm.MethodFigureError("macro-step check failed at octave(s) " + ", ".join(failed))

    fig = fm.draw_fm(data)
    report = fm.fm_layout_report(fig)
    names = [f"{fm.FM_NAME}.{ext}" for ext in ("pdf", "svg", "png")]
    previous = {n: f1.sha256_of(args.out / n) for n in names if (args.out / n).is_file()}
    paths = fm.save_fm(fig, args.out)
    hashes = {p.name: f1.sha256_of(p) for p in paths}
    stable = {n: (previous[n] == h if n in previous else None) for n, h in hashes.items()}
    eval_note = ("`--eval-dir` given and not read (no model sample is drawn)"
                 if args.eval_dir is not None else "none read (no model sample is drawn)")
    markdown = fm.fm_markdown(data, checks, report, hashes, stable, command, eval_note)
    missing = fm.check_markdown_numbers(markdown, data)
    if missing:
        raise fm.MethodFigureError("FM.md lacks the numbers " + ", ".join(missing))
    (args.out / "FM.md").write_text(markdown, encoding="utf-8")

    for c in checks:
        print(f"macro-step {c.label}: sigma {c.sigma:.4g} px, d(c_b) {c.d_edge:.6f} "
              f"(DCTBlur {c.d_edge_released:.6f}), bin mean {c.ring_mean:.4f}, "
              f"max d beyond {c.beyond_max:.6f} -> {'PASS' if c.passed else 'FAIL'}")
    for key in ("default", "matched"):
        print(f"levels {f1.SCHEDULES[key]} per frequency octave: "
              + "/".join(map(str, data.level_counts[key])))
    print("modes per octave: " + "/".join(map(str, data.mode_counts)))
    base = report.base
    print(f"layout: {base.size_in[0]:g} x {base.size_in[1]:g} in, min font {base.min_font_pt:g} "
          f"pt, min mode thumbnail {report.min_mode_in:.2f} in, min state thumbnail "
          f"{report.min_state_in:.2f} in, overlaps {len(base.overlaps)}, outside "
          f"{len(base.outside)}")
    for a, b in base.overlaps:
        print(f"  overlap: {a!r} / {b!r}")
    print(f"selected: example {data.example_index}")
    for name, digest in hashes.items():
        verdict = {None: "new", True: "identical", False: "CHANGED"}[stable[name]]
        print(f"wrote {args.out / name} sha256 {digest[:16]} ({verdict})")
    print(f"wrote {args.out / 'FM.md'}")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return EXIT_OK if exc.code == 0 else EXIT_MISSING
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    # fontTools warns about the epoch timestamps of the subset fonts in every PDF; not actionable.
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    command = "python -m ihdm.cli.paper_fm " + shlex.join(sys.argv[1:] if argv is None else argv)
    try:
        return _run(args, command)
    except fm.MissingMethodInputError as exc:
        print(f"MISSING: {exc}", file=sys.stderr)
        return EXIT_MISSING
    except (f1.PaperFigureError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
