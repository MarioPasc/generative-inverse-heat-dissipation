"""Draw F1 of the course paper, the visual abstract, and write its record ``F1.md`` (T8.1).

``python -m ihdm.cli.paper_f1 --data-root <IHDM data root> --eval-dir <dir of evaluation tars>
--out <dir> [--work <scratch>]``

Thin shell of :mod:`ihdm.analysis.paper_f1`. It extracts the four held-out members of the two
teaser runs' evaluation tars into ``--work`` (a temporary folder when absent; an existing
extraction is reused), asserts the pre-declared selection rules, runs the verifications of the
ticket (the frequency mapping on the IXI ``train`` split, the level counts, the annotated
numbers), and writes ``<out>/f1_visual_abstract.{pdf,svg,png}`` and ``<out>/F1.md``.

Exit codes: 0 written; 1 analysis error (a failed selection rule or verification); 2 missing
input or unusable arguments.
"""

from __future__ import annotations

import argparse
import logging
import shlex
import sys
import tempfile
from pathlib import Path

import numpy as np

from ihdm.analysis import paper_f1 as f1
from ihdm.data.format import read_dataset

__all__ = ["EXIT_FAIL", "EXIT_MISSING", "EXIT_OK", "REPO_ROOT", "build_parser", "main"]

EXIT_OK, EXIT_FAIL, EXIT_MISSING = 0, 1, 2

#: Repository root holding ``docs/RESULTS`` and ``schedules`` (this file is ihdm/cli/paper_f1.py).
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
#: The selection rules applied by the command (the tests swap in rules for a tiny dataset).
RULES: f1.SelectionRules = f1.DEFAULT_RULES

logger = logging.getLogger("ihdm.cli.paper_f1")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m ihdm.cli.paper_f1", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="IHDM data root holding ixi/images.npy, index.csv, splits.json")
    parser.add_argument("--eval-dir", type=Path, required=True,
                        help="folder of the evaluation tars <run>_amp-fp16.tar")
    parser.add_argument("--out", type=Path, required=True, help="output folder")
    parser.add_argument("--work", type=Path, default=None,
                        help="scratch folder for the extracted held-out members (default: a "
                             "temporary folder, removed afterwards)")
    return parser


def _run(args: argparse.Namespace, work: Path, command: str) -> int:
    inputs = f1.F1Inputs(args.data_root, args.eval_dir, REPO_ROOT, work)
    missing = [p for p in inputs.required(RULES) if not p.exists()]
    if missing:
        raise f1.MissingInputError("missing input(s): " + ", ".join(map(str, missing)))

    names = list(f1.SCHEDULES.values())
    table = f1.parse_level_table(inputs.data_profile_md)
    counts = f1.verify_level_counts({n: np.load(inputs.schedule(n)) for n in names}, table)
    data = f1.load_f1_data(inputs, RULES, counts)

    images, _, splits, _ = read_dataset(args.data_root / RULES.dataset)
    train = images[np.asarray(sorted(splits["train"]), dtype=np.int64)]
    checks = f1.verify_mapping(train, f1.VERIFY_SIGMAS)
    failed = [c for c in checks if not c.passed]
    if failed:
        raise f1.PaperFigureError("frequency mapping failed at sigma "
                                  + ", ".join(f"{c.sigma:g}" for c in failed))
    annotation_checks = f1.verify_annotations(data.annotations, inputs)
    wrong = [item for item, _, _, ok in annotation_checks if not ok]
    if wrong:
        raise f1.PaperFigureError("annotated numbers differ from their sources: "
                                  + ", ".join(wrong))

    fig = f1.draw_f1(data)
    report = f1.layout_report(fig)
    names_out = [f"{f1.F1_NAME}.{ext}" for ext in ("pdf", "svg", "png")]
    previous = {n: f1.sha256_of(args.out / n) for n in names_out if (args.out / n).is_file()}
    paths = f1.save_f1(fig, args.out)
    hashes = {p.name: f1.sha256_of(p) for p in paths}
    stable = {n: (previous[n] == h if n in previous else None) for n, h in hashes.items()}
    markdown = f1.f1_markdown(data, checks, annotation_checks, report, hashes, stable, command,
                              RULES)
    (args.out / "F1.md").write_text(markdown, encoding="utf-8")

    for c in checks:
        print(f"mapping sigma={c.sigma:g}: max rel err {c.max_rel_error:.1e}, "
              f"c1/2 {c.half_power:.4f} vs {c.half_power_numeric:.4f} -> "
              f"{'PASS' if c.passed else 'FAIL'}")
    for name in names:
        print(f"levels {name}: {'/'.join(map(str, counts[name]))}")
    for item, printed, ref, ok in annotation_checks:
        print(f"number {item}: {printed} vs {ref} -> {'PASS' if ok else 'FAIL'}")
    print(f"layout: {report.size_in[0]:g} x {report.size_in[1]:g} in, min font "
          f"{report.min_font_pt:g} pt, min thumbnail {report.min_thumb_in:.2f} in, "
          f"overlaps {len(report.overlaps)}, outside {len(report.outside)}")
    print(f"selected: example {data.example_index}; subjects "
          + ", ".join(f"{s}={i}" for s, i in data.subjects) + f"; teaser {data.teaser_index}")
    for name, digest in hashes.items():
        verdict = {None: "new", True: "identical", False: "CHANGED"}[stable[name]]
        print(f"wrote {args.out / name} sha256 {digest[:16]} ({verdict})")
    print(f"wrote {args.out / 'F1.md'}")
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
    command = "python -m ihdm.cli.paper_f1 " + shlex.join(sys.argv[1:] if argv is None else argv)
    try:
        if args.work is not None:
            args.work.mkdir(parents=True, exist_ok=True)
            return _run(args, args.work, command)
        with tempfile.TemporaryDirectory(prefix="paper_f1_") as tmp:
            return _run(args, Path(tmp), command)
    except f1.MissingInputError as exc:
        print(f"MISSING: {exc}", file=sys.stderr)
        return EXIT_MISSING
    except (f1.PaperFigureError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
