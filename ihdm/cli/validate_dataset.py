"""Check one or more standard-format datasets against the format contract.

Thin imperative shell around :func:`ihdm.data.format.validate_dataset`: it exists so the
cluster import check (``docs/SPECIFICATIONS/M3-picasso/T3.1-picasso-setup.md`` §4) can assert
that the data copied to Picasso is the data that was validated locally, without writing a
throwaway ``python -c`` one-liner into a batch script.

Run as ``python -m ihdm.cli.validate_dataset <root> [<root> ...]``. A ``<root>`` that is not
itself a directory is looked up under :func:`ihdm.paths.data_root`, so with ``IHDM_DATA_ROOT``
exported the bare dataset ids ``ixi`` and ``lsun_church`` are valid arguments. A valid dataset
prints ``OK <id>: <N> images, <train>/<ref>/<seed>``; an invalid one prints ``FAIL <id>`` and one
line per violation. The exit code is 0 only when every root is valid. Note that the three split
sizes do not sum to ``N``: ``train`` and ``ref`` partition the dataset and ``seed`` is a subset of
``ref`` (:func:`ihdm.data.format.split_by_subject`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ihdm.data.format import SPLIT_NAMES, validate_dataset
from ihdm.paths import data_root

__all__ = ["build_parser", "describe", "main", "resolve_root"]


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser of the command.

    Returns
    -------
    argparse.ArgumentParser
        Parser accepting one or more dataset roots.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "roots",
        nargs="+",
        help="dataset directories, or dataset ids resolved under IHDM_DATA_ROOT",
    )
    return parser


def resolve_root(spec: str) -> Path:
    """Resolve a command-line argument to a dataset directory.

    Parameters
    ----------
    spec : str
        A path to a dataset directory, or a dataset id to look up under the data root.

    Returns
    -------
    Path
        ``spec`` itself when it names an existing directory, otherwise
        ``data_root() / spec`` when that exists, otherwise ``spec`` unchanged so the
        validator reports the missing files against the path the caller typed.
    """
    direct = Path(spec)
    if direct.is_dir():
        return direct
    candidate = data_root() / spec
    return candidate if candidate.is_dir() else direct


def describe(root: Path) -> str:
    """Return the one-line summary of a dataset that passed validation.

    Parameters
    ----------
    root : Path
        Directory of a dataset already known to satisfy the format contract.

    Returns
    -------
    str
        ``"<id>: <N> images, <train>/<ref>/<seed>"``. Counts that cannot be read fall back
        to ``"?"`` rather than raising: the validator, not this function, decides validity.
    """
    try:
        meta = json.loads((root / "meta.json").read_text())
        dataset_id = str(meta.get("dataset_id", root.name))
        n_images = meta.get("n_images", "?")
    except (OSError, ValueError):
        dataset_id, n_images = root.name, "?"

    try:
        splits = json.loads((root / "splits.json").read_text())
        sizes = "/".join(str(len(splits.get(name, []))) for name in SPLIT_NAMES)
    except (OSError, TypeError, ValueError):
        sizes = "/".join("?" for _ in SPLIT_NAMES)

    return f"{dataset_id}: {n_images} images, {sizes}"


def main(argv: list[str] | None = None) -> int:
    """Validate every root given on the command line.

    Parameters
    ----------
    argv : list[str] | None
        Argument vector; ``None`` reads ``sys.argv[1:]``.

    Returns
    -------
    int
        0 if every dataset is valid, 1 otherwise.
    """
    args = build_parser().parse_args(argv)
    failed = False

    for spec in args.roots:
        root = resolve_root(spec)
        violations = validate_dataset(root)
        if violations:
            failed = True
            print(f"FAIL {root}: {len(violations)} violation(s)")
            for violation in violations:
                print(f"  - {violation}")
        else:
            print(f"OK {describe(root)}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
